#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.data.constants import LABELS, LETTER_ORDER
from sft.infer_common import build_inference_context


logger = logging.getLogger("torch-label-eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a label-token CE verifier checkpoint with the Hugging Face/PEFT "
            "torch forward path. This bypasses vLLM and generation parsing."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Verifier train run directory.")
    parser.add_argument("--checkpoint", type=str, default="best", help="Checkpoint name or absolute path.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Resolved train config to use for candidate prompts. Defaults to run-dir/config.resolved.yaml.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for eval artifacts.")
    parser.add_argument(
        "--base-model-name-or-path",
        type=str,
        default=None,
        help="Override the base model path used when loading a PEFT adapter.",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default=None,
        help="Override label_token_ce.label_prefix. Defaults to the training config value.",
    )
    parser.add_argument(
        "--strict-label-token-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if run-dir/label_token_ce_meta.json disagrees with the active tokenizer/prefix.",
    )
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Model dtype. auto mirrors training config bf16 behavior on CUDA.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager", "default"],
        help="Attention implementation for AutoModelForCausalLM.from_pretrained.",
    )
    parser.add_argument(
        "--merge-lora-for-forward",
        action="store_true",
        help="Optionally merge a PEFT adapter before evaluation. Default keeps PEFT active.",
    )
    parser.add_argument(
        "--dedupe-sample-idx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate gathered prediction records by sample_idx before reporting main metrics.",
    )
    parser.add_argument("--eval-log-predictions", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _metric_payload(metrics: dict[str, Any], *, num_samples: int) -> dict[str, Any]:
    keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "parse_error_rate",
        "eval_loss",
        "true_side_macro_f1",
        "selection_score",
        "per_class",
    ]
    payload: dict[str, Any] = {"num_samples": int(num_samples)}
    for key in keys:
        if key in metrics:
            payload[key] = _to_jsonable(metrics[key])
    return payload


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")


def _save_confusion_png(path: Path, confusion_matrix: np.ndarray, confusion_labels: list[str], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is unavailable; skip confusion matrix png: %s", path)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(confusion_labels)))
    ax.set_xticklabels(confusion_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, str(int(confusion_matrix[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _dedupe_records_by_sample_idx(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: int(row["sample_idx"])):
        sample_idx = int(record["sample_idx"])
        if sample_idx in seen:
            continue
        seen.add(sample_idx)
        deduped.append(record)
    return deduped


def _summarize_records(records: list[dict[str, Any]], *, eval_logger: logging.Logger) -> dict[str, Any]:
    from sft.eval import summarize_prediction_records

    return summarize_prediction_records(
        records,
        eval_logger=eval_logger,
        log_predictions_limit=0,
    )


def _true_side_macro_f1(metrics: dict[str, Any]) -> float:
    per_class = metrics.get("per_class", {}) or {}
    values: list[float] = []
    for label in ("mostly-true", "true"):
        label_metrics = per_class.get(label, {}) if isinstance(per_class, dict) else {}
        if isinstance(label_metrics, dict):
            values.append(float(label_metrics.get("f1", 0.0)))
    return float(np.mean(values)) if values else 0.0


def _selection_score(metrics: dict[str, Any], train_cfg: dict[str, Any]) -> float:
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    metric = str(label_cfg.get("early_stopping_metric", "macro_f1_plus_true_side")).strip().lower()
    macro_f1 = float(metrics["macro_f1"])
    true_side = _true_side_macro_f1(metrics)
    if metric in {"macro_f1", "f1"}:
        return macro_f1
    if metric in {"true_side_macro_f1", "true_side"}:
        return true_side
    if metric in {"accuracy", "acc"}:
        return float(metrics["accuracy"])
    if metric in {"macro_f1_plus_true_side", "macro_f1+true_side"}:
        return macro_f1 + float(label_cfg.get("true_side_metric_weight", 0.5)) * true_side
    raise ValueError(
        "Unsupported sft_train.label_token_ce.early_stopping_metric="
        f"{metric!r}. Use macro_f1, true_side_macro_f1, accuracy, or macro_f1_plus_true_side."
    )


def _augment_metrics(metrics: dict[str, Any], train_cfg: dict[str, Any], *, eval_loss: float | None = None) -> None:
    if eval_loss is not None:
        metrics["eval_loss"] = float(eval_loss)
    metrics["true_side_macro_f1"] = _true_side_macro_f1(metrics)
    metrics["selection_score"] = _selection_score(metrics, train_cfg)


def _resolve_torch_dtype(name: str, train_cfg: dict[str, Any]):
    import torch

    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() and bool(train_cfg.get("bf16", True)) else torch.float32


def _resolve_model_kwargs(args: argparse.Namespace, train_cfg: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _resolve_torch_dtype(str(args.torch_dtype), train_cfg),
    }

    attn_impl = str(args.attn_implementation)
    if attn_impl == "auto":
        if bool(train_cfg.get("use_flash_attention_2", True)):
            from sft.runtime.deps import flash_attn2_available

            if flash_attn2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                logger.warning(
                    "sft_train.use_flash_attention_2=true, but flash-attn is unavailable; "
                    "falling back to the default attention implementation."
                )
    elif attn_impl != "default":
        model_kwargs["attn_implementation"] = attn_impl

    return model_kwargs


def _load_model(args: argparse.Namespace, context):
    from transformers import AutoModelForCausalLM

    model_kwargs = _resolve_model_kwargs(args, context.train_cfg)
    if context.is_peft_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "This checkpoint is a PEFT adapter, but `peft` is not installed. "
                "Install project dependencies or `pip install peft` on the target machine."
            ) from exc

        base_model_path = str(args.base_model_name_or_path or context.model_name_or_path)
        if not base_model_path:
            raise ValueError("Cannot resolve base model path for PEFT checkpoint; pass --base-model-name-or-path.")
        logger.info("Loading base model for PEFT adapter: %s", base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(context.checkpoint_dir), is_trainable=False)
        if args.merge_lora_for_forward:
            logger.info("Merging PEFT adapter into base model before eval.")
            model = model.merge_and_unload()
    else:
        logger.info("Loading full HF checkpoint: %s", context.checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(str(context.checkpoint_dir), **model_kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def _read_label_token_meta(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "label_token_ce_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_label_token_meta(
    *,
    meta: dict[str, Any] | None,
    token_meta: dict[str, Any],
    strict: bool,
) -> None:
    if meta is None:
        logger.warning("label_token_ce_meta.json is missing; cannot compare saved label-token metadata.")
        return

    expected = {
        "label_prefix": meta.get("label_prefix"),
        "prefix_token_ids": meta.get("prefix_token_ids"),
        "label_token_ids": meta.get("label_token_ids"),
    }
    actual = {
        "label_prefix": token_meta.get("label_prefix"),
        "prefix_token_ids": token_meta.get("prefix_token_ids"),
        "label_token_ids": token_meta.get("label_token_ids"),
    }
    if expected == actual:
        logger.info("Label-token metadata matches run-dir/label_token_ce_meta.json.")
        return

    message = (
        "Active tokenizer/label_prefix disagree with run-dir/label_token_ce_meta.json.\n"
        f"expected={expected}\n"
        f"actual={actual}"
    )
    if strict:
        raise RuntimeError(message)
    logger.warning(message)


def _save_artifacts(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    raw_metrics: dict[str, Any],
    raw_prediction_records: list[dict[str, Any]],
    report: dict[str, Any],
    split: str,
    checkpoint_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = _metric_payload(metrics, num_samples=len(prediction_records))
    _save_json(output_dir / "metrics.json", metrics_payload)
    _save_jsonl(output_dir / f"{split}_predictions.jsonl", prediction_records)

    confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    confusion_labels = [str(x) for x in metrics["confusion_labels"]]
    _save_json(
        output_dir / "confusion_matrix.json",
        {
            "gold_labels": LABELS,
            "pred_labels": confusion_labels,
            "matrix": confusion_matrix.tolist(),
        },
    )
    _save_confusion_png(
        output_dir / "confusion_matrix.png",
        confusion_matrix,
        confusion_labels,
        title=f"Torch Forward Label-Token Eval: {split}/{checkpoint_name}",
    )

    if raw_prediction_records is not prediction_records:
        _save_jsonl(output_dir / f"{split}_predictions.raw_gathered.jsonl", raw_prediction_records)
        _save_json(output_dir / "metrics.raw_gathered.json", _metric_payload(raw_metrics, num_samples=len(raw_prediction_records)))

    _save_json(output_dir / "eval_report.json", report)
    return {
        "metrics_path": str(output_dir / "metrics.json"),
        "predictions_path": str(output_dir / f"{split}_predictions.jsonl"),
        "report_path": str(output_dir / "eval_report.json"),
    }


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    from accelerate import Accelerator
    import torch

    from sft.dataset.loaders import build_dataloader
    from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
    from sft.label_token_trainer import _build_label_token_ids, _class_weight_tensor, _evaluate_label_token

    context = build_inference_context(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        config_path=args.config,
    )
    label_cfg = context.train_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(args.label_prefix if args.label_prefix is not None else label_cfg.get("label_prefix", "Label:"))
    label_token_id_list, token_meta = _build_label_token_ids(context.tokenizer, label_prefix=label_prefix)
    _validate_label_token_meta(
        meta=_read_label_token_meta(context.run_dir),
        token_meta=token_meta,
        strict=bool(args.strict_label_token_meta),
    )

    mixed_precision = "bf16" if torch.cuda.is_available() and str(args.torch_dtype) in {"auto", "bf16"} else "no"
    if str(args.torch_dtype) == "fp16":
        mixed_precision = "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    if accelerator.is_main_process:
        logger.info("run_dir=%s", context.run_dir)
        logger.info("checkpoint=%s", context.checkpoint_dir)
        logger.info("split=%s samples=%d max_length=%d", context.split, len(context.samples), context.max_length)
        logger.info("is_peft_adapter=%s", context.is_peft_adapter)
        logger.info("label_token_ids=%s", token_meta["label_token_ids"])

    model = _load_model(args, context)

    per_device_eval_batch_size = int(
        args.per_device_eval_batch_size
        if args.per_device_eval_batch_size is not None
        else context.train_cfg.get("per_device_eval_batch_size", 1)
    )
    num_workers = int(
        args.dataloader_num_workers
        if args.dataloader_num_workers is not None
        else context.train_cfg.get("dataloader_num_workers", 0)
    )
    dataset = LabelTokenDataset(
        context.samples,
        context.tokenizer,
        max_length=context.max_length,
        label_prefix=label_prefix,
    )
    dataloader = build_dataloader(
        dataset,
        collator=LabelTokenCollator(tokenizer=context.tokenizer, pad_to_multiple_of=8),
        batch_size=per_device_eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        use_length_bucket=False,
    )

    label_token_ids = torch.tensor(label_token_id_list, dtype=torch.long)
    class_weights = _class_weight_tensor(context.train_cfg)
    model, dataloader = accelerator.prepare(model, dataloader)

    eval_metrics = _evaluate_label_token(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        label_token_ids=label_token_ids,
        class_weights=class_weights,
        label_prefix=label_prefix,
        eval_logger=logger,
        log_predictions_limit=int(args.eval_log_predictions),
    )
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    raw_records = list(eval_metrics.get("prediction_records", []))
    raw_metrics = dict(eval_metrics)
    _augment_metrics(raw_metrics, context.train_cfg)

    if bool(args.dedupe_sample_idx):
        records = _dedupe_records_by_sample_idx(raw_records)
        metrics = _summarize_records(records, eval_logger=logger)
        _augment_metrics(metrics, context.train_cfg, eval_loss=float(eval_metrics.get("eval_loss", float("nan"))))
    else:
        records = raw_records
        metrics = raw_metrics

    output_dir = Path(args.output_dir)
    report = {
        "script": "scripts/verifier/eval_label_token_torch_forward.py",
        "run_dir": str(context.run_dir),
        "checkpoint_name": context.checkpoint_name,
        "checkpoint_dir": str(context.checkpoint_dir),
        "is_peft_adapter": bool(context.is_peft_adapter),
        "split": context.split,
        "config_path": str(args.config) if args.config else str(context.run_dir / "config.resolved.yaml"),
        "output_dir": str(output_dir),
        "model_name_or_path": str(context.model_name_or_path),
        "base_model_name_or_path_override": args.base_model_name_or_path,
        "max_length": int(context.max_length),
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "dataloader_num_workers": num_workers,
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "merge_lora_for_forward": bool(args.merge_lora_for_forward),
        "dedupe_sample_idx": bool(args.dedupe_sample_idx),
        "raw_num_prediction_records": len(raw_records),
        "main_num_prediction_records": len(records),
        "duplicate_prediction_records": len(raw_records) - len(records),
        "label_token_meta": token_meta,
        "metrics": _metric_payload(metrics, num_samples=len(records)),
        "raw_metrics": _metric_payload(raw_metrics, num_samples=len(raw_records)),
    }
    artifacts = _save_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        prediction_records=records,
        raw_metrics=raw_metrics,
        raw_prediction_records=raw_records,
        report=report,
        split=context.split,
        checkpoint_name=context.checkpoint_name,
    )
    logger.info(
        "torch-forward label-token eval done: accuracy=%.6f macro_f1=%.6f n=%d artifacts=%s",
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        len(records),
        artifacts,
    )


if __name__ == "__main__":
    main()
