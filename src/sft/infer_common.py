from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from fact_checking.data.io import load_jsonl
from fact_checking.config import load_yaml
from sft.data.types import PreparedSample
from sft.runtime.adapters import checkpoint_has_hf_artifacts, checkpoint_has_peft_adapter


@dataclass
class InferenceContext:
    run_dir: Path
    checkpoint_name: str
    checkpoint_dir: Path
    is_peft_adapter: bool
    split: str
    cfg: dict[str, Any]
    baseline_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    model_name_or_path: str
    tokenizer: AutoTokenizer
    max_length: int
    samples: list[PreparedSample]
    eval_output_dir: Path


def load_inference_config(run_dir: Path, config_path: str | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path) if config_path else run_dir / "config.resolved.yaml"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Cannot find resolved config at {resolved_path}. "
            "Pass --config explicitly or use a run directory produced by the updated trainer."
        )
    return load_yaml(resolved_path)


def _load_prebuilt_samples(rows: list[dict]) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for row in rows:
        gold_label = str(row.get("gold_label", ""))
        if not gold_label:
            continue
        samples.append(PreparedSample(
            prompt=str(row["prompt"]),
            target=str(row["target"]),
            prompt_add_special_tokens=bool(row.get("prompt_add_special_tokens", False)),
            preserve_prompt_prefix=bool(row.get("preserve_prompt_prefix", True)),
            gold_id=int(row.get("gold_id", -1)),
            gold_label=gold_label,
            gold_explain=str(row.get("gold_explain", "")),
            prompt_token_count=int(row.get("prompt_token_count", 0)),
            target_token_count=int(row.get("target_token_count", 0)),
            evidence_count=int(row.get("evidence_count", 0)),
            was_truncated=bool(row.get("was_truncated", False)),
            claim=str(row.get("claim", "")),
            no_evidence=int(row.get("evidence_count", 0)) == 0,
        ))
    return samples


def resolve_checkpoint_dir(run_dir: Path, checkpoint: str) -> tuple[str, Path]:
    checkpoint_path = Path(checkpoint)
    resolved = checkpoint_path if checkpoint_path.is_absolute() else run_dir / checkpoint
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {resolved}")

    checkpoint_name = checkpoint_path.name if checkpoint_path.name else resolved.name
    return checkpoint_name, resolved


def ensure_inferable_checkpoint(checkpoint_dir: Path) -> None:
    if checkpoint_has_hf_artifacts(checkpoint_dir):
        return

    ds_checkpoint_dir = checkpoint_dir / "ds_checkpoint"
    if ds_checkpoint_dir.exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} only contains a DeepSpeed checkpoint at {ds_checkpoint_dir} and has no Hugging Face "
            "weights. This run likely predates automatic HF export; re-save or export this checkpoint once."
        )

    raise FileNotFoundError(
        f"{checkpoint_dir} is missing Hugging Face model artifacts (for example config.json and model weights)."
    )


def build_inference_context(
    *,
    run_dir: str | Path,
    checkpoint: str,
    split: str,
    config_path: str | None = None,
) -> InferenceContext:
    resolved_run_dir = Path(run_dir)
    checkpoint_name, checkpoint_dir = resolve_checkpoint_dir(resolved_run_dir, checkpoint)
    ensure_inferable_checkpoint(checkpoint_dir)

    cfg = load_inference_config(resolved_run_dir, config_path=config_path)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]
    train_cfg = cfg["sft_train"]
    model_name_or_path = str(
        cfg.get("model_name_or_path")
        or baseline_cfg.get("model_name_or_path", "")
    )
    is_peft_adapter = checkpoint_has_peft_adapter(checkpoint_dir)

    split_map = {
        "train": str(data_cfg["train_candidates"]),
        "val": str(data_cfg["val_candidates"]),
        "test": str(data_cfg["test_candidates"]),
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split={split}. Use one of {sorted(split_map)}.")

    tokenizer_dir = checkpoint_dir
    if is_peft_adapter and not (checkpoint_dir / "tokenizer_config.json").exists():
        tokenizer_dir = Path(str(baseline_cfg.get("model_name_or_path") or model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(train_cfg.get("max_length", 2048))
    rows = load_jsonl(split_map[split])
    samples = _load_prebuilt_samples(rows)

    return InferenceContext(
        run_dir=resolved_run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_dir=checkpoint_dir,
        is_peft_adapter=is_peft_adapter,
        split=split,
        cfg=cfg,
        baseline_cfg=baseline_cfg,
        train_cfg=train_cfg,
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        max_length=max_length,
        samples=samples,
        eval_output_dir=resolved_run_dir / "eval" / split / checkpoint_name,
    )


def build_serializable_metrics(eval_metrics: dict[str, object]) -> dict[str, object]:
    prediction_records = eval_metrics.get("prediction_records", [])
    return {
        "num_samples": len(prediction_records) if isinstance(prediction_records, list) else 0,
        "accuracy": float(eval_metrics["accuracy"]),
        "macro_precision": float(eval_metrics["macro_precision"]),
        "macro_recall": float(eval_metrics["macro_recall"]),
        "macro_f1": float(eval_metrics["macro_f1"]),
        "parse_error_rate": float(eval_metrics["parse_error_rate"]),
        "per_class": eval_metrics.get("per_class", {}),
    }
