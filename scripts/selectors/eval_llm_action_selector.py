#!/usr/bin/env python3
"""Selection-only eval for an LLM sequential action evidence selector."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.selectors.llm_action import (
    SCORE_MODE_ACTION_TOKEN,
    SCORE_MODE_CONTINUATION,
)
from fact_checking.selectors.llm_action_eval import (
    evaluate_llm_action_selection,
    selection_history_record,
    write_selection_eval_outputs,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
    write_json,
)


@dataclass(frozen=True)
class ModelDirResolution:
    model_dir_input: Path
    resolved_model_dir: Path
    run_dir: Path | None
    metadata: dict[str, Any]
    layout: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an LLM action selector against Stage2 oracle order.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--model-name", default=None, help="Base model path override for LoRA adapter checkpoints.")
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--score-mode", default=None, choices=[SCORE_MODE_ACTION_TOKEN, SCORE_MODE_CONTINUATION])
    p.add_argument("--choice-batch-size", type=int, default=64)
    p.add_argument("--max-candidate-chars", type=int, default=180)
    p.add_argument("--no-retrieval-scores", action="store_true")
    p.add_argument("--reference-metrics", nargs="*", default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--log-file", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _init_eval_logger(Path(args.log_file) if args.log_file else None)

    resolution = _resolve_model_dir(Path(args.model_dir))
    metadata = resolution.metadata
    max_length = int(args.max_length or metadata.get("max_length") or 1024)
    score_mode = str(args.score_mode or metadata.get("score_mode") or SCORE_MODE_ACTION_TOKEN)
    if logger is not None:
        logger.info(
            "Starting selection eval model_dir_input=%s resolved_model_dir=%s output_dir=%s",
            resolution.model_dir_input,
            resolution.resolved_model_dir,
            out_dir,
        )
    model, tokenizer = _load_model_and_tokenizer(
        model_dir=resolution.resolved_model_dir,
        model_name=args.model_name or metadata.get("base_model_name_or_path"),
        device=str(args.device),
    )

    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=str(args.expected_chunk_mmr_fingerprint),
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No evaluation examples after Stage2 audit/filtering.")

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    result = evaluate_llm_action_selection(
        model,
        tokenizer,
        examples,
        device=device,
        split=str(args.split),
        top_k=int(args.top_k),
        max_length=max_length,
        score_mode=score_mode,
        choice_batch_size=int(args.choice_batch_size),
        max_candidate_chars=int(args.max_candidate_chars),
        include_retrieval_scores=not bool(args.no_retrieval_scores),
        disable_progress=bool(args.no_progress),
    )
    selector_metrics = result["metrics"]["selector"]
    controls = result["metrics"]["controls"]
    metrics = {
        **result["metrics"],
        "model_dir_input": str(resolution.model_dir_input),
        "resolved_model_dir": str(resolution.resolved_model_dir),
        "run_dir": str(resolution.run_dir) if resolution.run_dir is not None else None,
        "model_dir": str(resolution.resolved_model_dir),
        "model_dir_layout": str(resolution.layout),
        "eval_output_dir": str(out_dir),
        "oracle_results": str(args.oracle_results),
        "filter_policy": str(args.filter_policy),
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "reference_metrics": _reference_metrics(args.reference_metrics or []),
        "selector_metadata": metadata,
        "total_elapsed_seconds": round(time.time() - started_at, 3),
    }
    result["metrics"] = metrics
    write_selection_eval_outputs(out_dir, result)
    _write_run_eval_summary(resolution.run_dir, metrics, out_dir)

    print(f"Wrote selection metrics: {out_dir / 'selection_metrics.json'}")
    print(
        "LLM-action Recall@5={rec:.4f}, Jaccard@5={jac:.4f}, NDCG@5={ndcg:.4f}; "
        "Hybrid Jaccard@5={hjac:.4f}".format(
            rec=float(selector_metrics.get("recall@5", float("nan"))),
            jac=float(selector_metrics.get("jaccard@5", float("nan"))),
            ndcg=float(selector_metrics.get("oracle_rank_ndcg@5", float("nan"))),
            hjac=float(controls["hybrid_score_top5"].get("jaccard@5", float("nan"))),
        )
    )
    if logger is not None:
        logger.info(
            "Finished selection eval n_claims=%d top_k=%d score_mode=%s elapsed_seconds=%.3f "
            "claims_per_second=%.6f estimated_forward_steps=%d output_dir=%s",
            int(metrics.get("n_claims", 0)),
            int(metrics.get("top_k", 0)),
            str(metrics.get("score_mode")),
            float(metrics.get("elapsed_seconds", 0.0)),
            float(metrics.get("claims_per_second", 0.0)),
            int(metrics.get("estimated_forward_steps", 0)),
            out_dir,
        )


def _load_model_and_tokenizer(
    *,
    model_dir: Path,
    model_name: str | None,
    device: str,
) -> tuple[torch.nn.Module, Any]:
    if (model_dir / "adapter_config.json").exists():
        if not model_name:
            raise ValueError("--model-name is required when selector_metadata.json has no base model path.")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Evaluating a LoRA action selector requires the `peft` package.") from exc
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and str(device) != "cpu" else torch.float32,
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
    else:
        load_path = str(model_dir if (model_dir / "config.json").exists() else model_name)
        if not load_path:
            raise ValueError("Could not determine model path.")
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and str(device) != "cpu" else torch.float32,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.to(target_device)
    return model, tokenizer


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "selector_metadata.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_model_dir(model_dir: Path) -> ModelDirResolution:
    model_dir = Path(model_dir)
    input_metadata = _load_metadata(model_dir)
    if not model_dir.exists():
        raise ValueError(f"--model-dir does not exist: {model_dir}")

    best_rel = input_metadata.get("best_checkpoint_dir")
    if best_rel and not _is_default_best_dir(model_dir):
        best_dir = model_dir / str(best_rel)
        if _is_checkpoint_dir(best_dir):
            metadata = _merge_metadata(input_metadata, _load_metadata(best_dir))
            return ModelDirResolution(
                model_dir_input=model_dir,
                resolved_model_dir=best_dir,
                run_dir=model_dir,
                metadata=metadata,
                layout="run_dir_best_checkpoint",
            )
        raise ValueError(
            f"selector_metadata.json points to best_checkpoint_dir={best_rel!r}, "
            f"but no checkpoint marker was found under {best_dir}."
        )

    if _is_checkpoint_dir(model_dir):
        run_dir = _infer_run_dir(model_dir, input_metadata)
        run_metadata = _load_metadata(run_dir) if run_dir is not None and run_dir != model_dir else {}
        metadata = _merge_metadata(run_metadata, input_metadata)
        return ModelDirResolution(
            model_dir_input=model_dir,
            resolved_model_dir=model_dir,
            run_dir=run_dir,
            metadata=metadata,
            layout="direct_checkpoint",
        )

    fallback_best = model_dir / "checkpoints" / "best"
    if _is_checkpoint_dir(fallback_best):
        metadata = _merge_metadata(input_metadata, _load_metadata(fallback_best))
        return ModelDirResolution(
            model_dir_input=model_dir,
            resolved_model_dir=fallback_best,
            run_dir=model_dir,
            metadata=metadata,
            layout="run_dir_default_best_checkpoint",
        )

    if input_metadata:
        raise ValueError(
            f"{model_dir} looks like an LLM action selector run directory, but no best checkpoint was found. "
            "Expected adapter_config.json/config.json at the run root for legacy runs or under checkpoints/best."
        )
    raise ValueError(
        f"--model-dir must be a checkpoint directory or an LLM action selector run directory: {model_dir}"
    )


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or (path / "config.json").exists()


def _is_default_best_dir(path: Path) -> bool:
    return path.name == "best" and path.parent.name == "checkpoints"


def _infer_run_dir(model_dir: Path, metadata: dict[str, Any]) -> Path | None:
    raw_run_dir = metadata.get("run_output_dir")
    if raw_run_dir:
        return Path(str(raw_run_dir))
    if _is_default_best_dir(model_dir):
        return model_dir.parent.parent
    if metadata:
        return model_dir
    return None


def _merge_metadata(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        merged.update(dict(item or {}))
    return merged


def _write_run_eval_summary(run_dir: Path | None, metrics: dict[str, Any], out_dir: Path) -> None:
    if run_dir is None:
        return
    record = selection_history_record(
        metrics,
        output_dir=str(out_dir),
        reason="post_train_eval",
    )
    record["eval_output_dir"] = str(out_dir)
    record["resolved_model_dir"] = str(metrics.get("resolved_model_dir"))
    record["model_dir_input"] = str(metrics.get("model_dir_input"))
    metrics_dir = Path(run_dir) / "metrics"
    write_json(metrics_dir / "latest_selection_eval.json", record)
    _append_jsonl(Path(run_dir) / "eval_history.jsonl", record)


def _init_eval_logger(path: Path | None) -> logging.Logger | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"llm_action_selector_eval.{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reference_metrics(paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    wanted = {
        "ridge_all_step0_static",
        "single_margin_step0_static",
        "hybrid_score_top5",
        "deberta_sequential_deep",
    }
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        source = str(path)
        for key, value in (payload.get("selection_metrics") or {}).items():
            if key in wanted:
                out[key] = {"source": source, "metrics": value}
        for key, value in (payload.get("controls") or {}).items():
            if key in wanted:
                out[key] = {"source": source, "metrics": value}
        if "selector" in payload and "deberta_sequential" in source:
            out["deberta_sequential_deep"] = {"source": source, "metrics": payload["selector"]}
    return out


if __name__ == "__main__":
    main()
