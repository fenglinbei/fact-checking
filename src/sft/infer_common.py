from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from fact_checking.data.io import load_jsonl
from fact_checking.config import load_yaml
from fact_checking.data.constants import labels_for_schema
from sft.data.io import load_prebuilt_samples
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
    label_schema: str
    labels: list[str]


def build_label_decoding_prompt(sample: PreparedSample, label_prefix: str) -> str:
    """Match LabelTokenDataset's prompt context before scoring/generating a label."""
    prompt_text = sample.prompt.rstrip()
    if sample.prompt_add_special_tokens:
        prompt_text += " "
    return prompt_text + label_prefix


def label_choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def build_label_scoring_prompt(sample: PreparedSample, label_prefix: str, letter: str) -> str:
    return build_label_decoding_prompt(sample, label_prefix) + label_choice_text(label_prefix, letter)


def load_inference_config(run_dir: Path, config_path: str | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path) if config_path else run_dir / "config.resolved.yaml"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Cannot find resolved config at {resolved_path}. "
            "Pass --config explicitly or use a run directory produced by the updated trainer."
        )
    return load_yaml(resolved_path)


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
    label_schema = str(
        train_cfg.get("label_schema")
        or cfg.get("label_schema")
        or baseline_cfg.get("label_schema")
        or "liar6"
    )
    labels = labels_for_schema(label_schema)
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
    samples = load_prebuilt_samples(rows)

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
        label_schema=label_schema,
        labels=labels,
    )


def label_name_from_id(label_id: int, label_schema: str | None = None) -> str:
    """Map a label index to its string name; returns 'parse_error' for out-of-range ids."""
    labels = labels_for_schema(label_schema)
    if 0 <= int(label_id) < len(labels):
        return labels[int(label_id)]
    return "parse_error"


def build_vllm_prediction_record(
    sample_idx: int,
    sample: PreparedSample,
    raw_completion: str,
    *,
    use_label_decoding: bool,
    label_prefix: str = "Label:",
    label_schema: str | None = None,
) -> dict[str, object]:
    """Build a single prediction record dict for vLLM inference results."""
    from sft.parser import _parse_label_id

    raw_output = f"{label_prefix}{raw_completion}" if use_label_decoding else raw_completion
    pred_id = _parse_label_id(raw_output, label_schema=label_schema)
    return {
        "sample_idx": sample_idx,
        "prompt": sample.prompt,
        "target": sample.target,
        "raw_output": raw_output,
        "raw_completion": raw_completion,
        "pred_id": int(pred_id),
        "pred_label": label_name_from_id(int(pred_id), label_schema),
        "gold_id": int(sample.gold_id),
        "gold_label": sample.gold_label,
        "gold_explain": sample.gold_explain,
    }


def create_vllm_logit_processors(logit_adjust_cfg: dict | None) -> list:
    """Return logit processors for vLLM, or empty list if label decoding is disabled."""
    if not (logit_adjust_cfg and logit_adjust_cfg.get("enabled")):
        return []
    from sft.logit_adjust import create_label_choice_processor

    return [create_label_choice_processor(logit_adjust_cfg)]


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
