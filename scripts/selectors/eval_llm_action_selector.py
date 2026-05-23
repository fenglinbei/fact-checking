#!/usr/bin/env python3
"""Selection-only eval for an LLM sequential action evidence selector."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.selectors.llm_action import (
    SCORE_MODE_ACTION_TOKEN,
    SCORE_MODE_CONTINUATION,
    action_token,
    build_action_prompt,
    score_action_choices,
)
from fact_checking.selectors.metrics import (
    build_order_control_trace,
    build_selection_trace,
    ranked_indices_from_candidate_pool,
    ranked_indices_from_hybrid,
    random_order_controls,
    reorder_predicted_set,
    summarize_ordered_selection,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    load_stage2_oracle_examples,
    write_json,
    write_jsonl,
)


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(Path(args.model_dir))
    max_length = int(args.max_length or metadata.get("max_length") or 1024)
    score_mode = str(args.score_mode or metadata.get("score_mode") or SCORE_MODE_ACTION_TOKEN)
    model, tokenizer = _load_model_and_tokenizer(
        model_dir=Path(args.model_dir),
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

    traces: list[dict[str, Any]] = []
    hybrid_traces: list[dict[str, Any]] = []
    candidate_order_traces: list[dict[str, Any]] = []
    same_set_hybrid_traces: list[dict[str, Any]] = []
    same_set_candidate_traces: list[dict[str, Any]] = []
    random_traces: list[dict[str, Any]] = []

    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    for example in tqdm(
        examples,
        desc=f"llm action eval [{args.split}]",
        disable=bool(args.no_progress),
        dynamic_ncols=True,
    ):
        prediction = _rollout_example(
            model,
            tokenizer,
            example,
            device=device,
            top_k=int(args.top_k),
            max_length=max_length,
            score_mode=score_mode,
            choice_batch_size=int(args.choice_batch_size),
            max_candidate_chars=int(args.max_candidate_chars),
            include_retrieval_scores=not bool(args.no_retrieval_scores),
        )
        trace = build_selection_trace(
            example,
            _rank_vector_from_order(len(example.candidates), prediction["ordered_indices"]),
            selector_name="llm_action_selector",
            top_k=int(args.top_k),
        )
        trace["selector_ordered_indices"] = [int(idx) for idx in prediction["ordered_indices"]]
        trace["per_step_action_scores"] = prediction["per_step_action_scores"]
        traces.append(trace)

        hybrid_traces.append(
            build_order_control_trace(
                trace,
                ranked_indices_from_hybrid(example, top_k=int(args.top_k)),
                selector_name="hybrid_score_top5",
                top_k=int(args.top_k),
            )
        )
        candidate_order_traces.append(
            build_order_control_trace(
                trace,
                ranked_indices_from_candidate_pool(example, top_k=int(args.top_k)),
                selector_name="candidate_pool_order_top5",
                top_k=int(args.top_k),
            )
        )
        predicted = [int(idx) for idx in trace["selector_ordered_indices"]]
        same_set_hybrid_traces.append(
            build_order_control_trace(
                trace,
                reorder_predicted_set(predicted, example=example, mode="hybrid_order"),
                selector_name="same_set_hybrid_order",
                top_k=int(args.top_k),
            )
        )
        same_set_candidate_traces.append(
            build_order_control_trace(
                trace,
                reorder_predicted_set(predicted, example=example, mode="candidate_pool_order"),
                selector_name="same_set_candidate_pool_order",
                top_k=int(args.top_k),
            )
        )
        random_traces.extend(
            random_order_controls(
                predicted,
                example=example,
                seeds=[0, 1, 2, 3, 4],
                top_k=int(args.top_k),
            )
        )

    selector_metrics = summarize_ordered_selection(traces)
    controls = {
        "hybrid_score_top5": summarize_ordered_selection(hybrid_traces),
        "candidate_pool_order_top5": summarize_ordered_selection(candidate_order_traces),
        "same_set_hybrid_order": summarize_ordered_selection(same_set_hybrid_traces),
        "same_set_candidate_pool_order": summarize_ordered_selection(same_set_candidate_traces),
        "same_set_random_order_mean": summarize_ordered_selection(random_traces),
    }
    metrics = {
        "model_dir": str(args.model_dir),
        "oracle_results": str(args.oracle_results),
        "split": str(args.split),
        "filter_policy": str(args.filter_policy),
        "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        "n_claims": len(examples),
        "max_length": max_length,
        "score_mode": score_mode,
        "selector": selector_metrics,
        "controls": controls,
        "reference_metrics": _reference_metrics(args.reference_metrics or []),
        "selector_metadata": metadata,
    }
    write_json(out_dir / "selection_metrics.json", metrics)
    write_jsonl(out_dir / "selection_trace.jsonl", traces)
    write_jsonl(out_dir / "control_hybrid_trace.jsonl", hybrid_traces)
    write_jsonl(out_dir / "control_candidate_pool_trace.jsonl", candidate_order_traces)
    _write_markdown(out_dir / "analysis.md", metrics)

    print(f"Wrote selection metrics: {out_dir / 'selection_metrics.json'}")
    print(
        "LLM-action Recall@5={rec:.4f}, Jaccard@5={jac:.4f}, NDCG@5={ndcg:.4f}; "
        "Hybrid Jaccard@5={hjac:.4f}".format(
            rec=float(selector_metrics.get("recall@5", np.nan)),
            jac=float(selector_metrics.get("jaccard@5", np.nan)),
            ndcg=float(selector_metrics.get("oracle_rank_ndcg@5", np.nan)),
            hjac=float(controls["hybrid_score_top5"].get("jaccard@5", np.nan)),
        )
    )


def _rollout_example(
    model: torch.nn.Module,
    tokenizer: Any,
    example: Stage2OracleExample,
    *,
    device: torch.device,
    top_k: int,
    max_length: int,
    score_mode: str,
    choice_batch_size: int,
    max_candidate_chars: int,
    include_retrieval_scores: bool,
) -> dict[str, Any]:
    selected: list[int] = []
    per_step: list[dict[str, Any]] = []
    for step in range(int(top_k)):
        remaining = [idx for idx in range(len(example.candidates)) if idx not in selected]
        if not remaining:
            break
        prompt = build_action_prompt(
            example,
            prefix_indices=selected,
            remaining_indices=remaining,
            max_candidate_chars=int(max_candidate_chars),
            include_retrieval_scores=bool(include_retrieval_scores),
        )
        sample = {
            "prompt": prompt,
            "choices": [
                {"candidate_idx": int(idx), "action": action_token(idx)}
                for idx in remaining
            ],
        }
        with torch.inference_mode():
            scored = score_action_choices(
                model,
                tokenizer,
                [sample],
                device=device,
                max_length=int(max_length),
                score_mode=str(score_mode),
                choice_batch_size=int(choice_batch_size),
            )
        scores = scored.scores[0]
        best_pos = int(torch.argmax(scores).detach().cpu().item())
        best_idx = int(scored.candidate_indices[0][best_pos])
        selected.append(best_idx)
        per_step.append(
            {
                "step": int(step),
                "selected_idx": int(best_idx),
                "selected_action": action_token(best_idx),
                "selected_score": float(scores[best_pos].detach().cpu().item()),
                "oracle_idx": int(example.selected_indices[step]) if step < len(example.selected_indices) else None,
                "choice_scores": [
                    {
                        "candidate_idx": int(idx),
                        "action": action_token(idx),
                        "score": float(score.detach().cpu().item()),
                    }
                    for idx, score in zip(scored.candidate_indices[0], scores)
                ],
            }
        )
    return {"ordered_indices": selected[: int(top_k)], "per_step_action_scores": per_step}


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


def _rank_vector_from_order(n_candidates: int, ordered: list[int]) -> np.ndarray:
    scores = np.full((int(n_candidates),), -1.0e9, dtype=np.float32)
    for rank, idx in enumerate(ordered):
        if 0 <= int(idx) < scores.size:
            scores[int(idx)] = float(len(ordered) - rank)
    return scores


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


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    selector = metrics.get("selector", {})
    controls = metrics.get("controls", {})
    refs = metrics.get("reference_metrics", {})
    lines = [
        "# LLM Action Selector Eval",
        "",
        f"- n_claims: {metrics.get('n_claims')}",
        f"- selector Jaccard@5: {float(selector.get('jaccard@5', math.nan)):.4f}",
        f"- selector NDCG@5: {float(selector.get('oracle_rank_ndcg@5', math.nan)):.4f}",
        f"- selector pairwise_order_acc@5: {float(selector.get('pairwise_order_acc@5', math.nan)):.4f}",
        f"- hybrid Jaccard@5: {float(controls.get('hybrid_score_top5', {}).get('jaccard@5', math.nan)):.4f}",
    ]
    single = refs.get("single_margin_step0_static", {}).get("metrics", {})
    if single:
        delta = float(selector.get("jaccard@5", math.nan)) - float(single.get("jaccard@5", math.nan))
        lines.append(f"- single_margin_step0_static Jaccard@5: {float(single.get('jaccard@5', math.nan)):.4f}")
        lines.append(f"- delta_vs_single_margin_step0_static: {delta:.4f}")
        lines.append(f"- decision: {'go' if delta > 0 else 'no_go'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
