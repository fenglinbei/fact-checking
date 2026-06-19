from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator

from fact_checking.data.constants import COVERAGE_LETTER_ORDER, labels_for_schema, letter_order_for_schema
from fact_checking.data.io import load_jsonl
from fact_checking.utils.logging import init_logger
from sft.data.io import load_prebuilt_samples, save_eval_artifacts
from sft.dataset.loaders import build_dataloader
from sft.eval import log_eval_summary
from sft.infer_common import build_inference_context, build_serializable_metrics
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.label_token_infer import _config_with_logit_adjust_override, _resolve_logit_adjust_cfg
from sft.label_token_trainer import (
    _build_label_token_ids,
    _checkpoint_selection_score,
    _class_weight_tensor,
    _coverage_class_weight_tensor,
    _coverage_label_token_cfg,
    _coverage_label_token_enabled,
    _evaluate_label_token,
    _true_side_macro_f1,
)
from sft.runtime.adapters import checkpoint_has_peft_adapter
from sft.runtime.model_loading import is_mistral_common_tokenizer, load_causal_lm_compatible_model
from sft.runtime.deps import flash_attn2_available


logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple label-token splits/tau settings for one checkpoint with one model load."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--plan-json", type=str, default=None)
    parser.add_argument("--plan-file", type=str, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--log-predictions", type=int, default=0)
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = _load_plan(args)
    if not plan:
        raise ValueError("multi-infer plan is empty.")

    first_split = _first_plan_split(plan)
    context = build_inference_context(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        split=first_split,
        config_path=args.config,
    )
    train_cfg = context.train_cfg
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(label_cfg.get("label_prefix", "Label:"))
    labels = labels_for_schema(context.label_schema)
    letter_order = letter_order_for_schema(context.label_schema)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model = _load_model(context=context, train_cfg=train_cfg, mixed_precision=mixed_precision)
    model = accelerator.prepare(model)

    label_token_id_list, _ = _build_label_token_ids(
        context.tokenizer,
        label_prefix=label_prefix,
        letter_order=letter_order,
    )
    label_token_ids = torch.tensor(label_token_id_list, dtype=torch.long)
    class_weights = _class_weight_tensor(train_cfg, labels=labels)
    coverage_enabled = _coverage_label_token_enabled(train_cfg)
    coverage_cfg = _coverage_label_token_cfg(train_cfg)
    coverage_label_token_ids = None
    coverage_class_weights = None
    if coverage_enabled:
        coverage_label_token_id_list, _ = _build_label_token_ids(
            context.tokenizer,
            label_prefix=str(coverage_cfg.get("label_prefix", "Coverage:")),
            letter_order=COVERAGE_LETTER_ORDER,
        )
        coverage_label_token_ids = torch.tensor(coverage_label_token_id_list, dtype=torch.long)
        coverage_class_weights = _coverage_class_weight_tensor(train_cfg)

    split_samples: dict[str, list[Any]] = {}
    for item in plan:
        item_type = str(item.get("type", "fixed"))
        if item_type == "fixed":
            _run_fixed_item(
                item,
                context=context,
                model=model,
                accelerator=accelerator,
                labels=labels,
                letter_order=letter_order,
                label_prefix=label_prefix,
                label_token_ids=label_token_ids,
                class_weights=class_weights,
                coverage_label_token_ids=coverage_label_token_ids,
                coverage_class_weights=coverage_class_weights,
                coverage_enabled=coverage_enabled,
                coverage_cfg=coverage_cfg,
                split_samples=split_samples,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                dataloader_num_workers=args.dataloader_num_workers,
                log_predictions=int(args.log_predictions),
                force_eval=bool(args.force_eval),
            )
            continue
        if item_type == "val_selected_tau":
            _run_val_selected_tau_item(
                item,
                context=context,
                model=model,
                accelerator=accelerator,
                labels=labels,
                letter_order=letter_order,
                label_prefix=label_prefix,
                label_token_ids=label_token_ids,
                class_weights=class_weights,
                coverage_label_token_ids=coverage_label_token_ids,
                coverage_class_weights=coverage_class_weights,
                coverage_enabled=coverage_enabled,
                coverage_cfg=coverage_cfg,
                split_samples=split_samples,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                dataloader_num_workers=args.dataloader_num_workers,
                log_predictions=int(args.log_predictions),
                force_eval=bool(args.force_eval),
            )
            continue
        raise ValueError(f"Unsupported multi-infer plan item type={item_type!r}")


def _load_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.plan_json) == bool(args.plan_file):
        raise ValueError("Pass exactly one of --plan-json or --plan-file.")
    payload = Path(args.plan_file).read_text(encoding="utf-8") if args.plan_file else str(args.plan_json)
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise TypeError("multi-infer plan must be a JSON list.")
    return [dict(item) for item in parsed]


def _first_plan_split(plan: list[Mapping[str, Any]]) -> str:
    for item in plan:
        if str(item.get("type", "fixed")) == "fixed":
            return str(item.get("split", "val"))
    return "val"


def _load_model(*, context, train_cfg: Mapping[str, Any], mixed_precision: str):
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and mixed_precision == "bf16" else torch.float32,
    }
    if bool(train_cfg.get("use_flash_attention_2", True)) and flash_attn2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if checkpoint_has_peft_adapter(context.checkpoint_dir):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("LoRA label-token inference requires the `peft` package.") from exc

        model = load_causal_lm_compatible_model(context.model_name_or_path, **model_kwargs)
        return PeftModel.from_pretrained(model, str(context.checkpoint_dir))
    return load_causal_lm_compatible_model(str(context.checkpoint_dir), **model_kwargs)


def _run_val_selected_tau_item(
    item: Mapping[str, Any],
    *,
    context,
    model,
    accelerator: Accelerator,
    labels: list[str],
    letter_order: list[str],
    label_prefix: str,
    label_token_ids: torch.Tensor,
    class_weights: torch.Tensor,
    coverage_label_token_ids: torch.Tensor | None,
    coverage_class_weights: torch.Tensor | None,
    coverage_enabled: bool,
    coverage_cfg: Mapping[str, Any],
    split_samples: dict[str, list[Any]],
    per_device_eval_batch_size: int | None,
    dataloader_num_workers: int | None,
    log_predictions: int,
    force_eval: bool,
) -> None:
    experiment = str(item["experiment"])
    metric = str(item.get("metric", "macro_f1"))
    rows: list[dict[str, Any]] = []
    for tau in item.get("taus", []):
        tau_value = float(tau)
        val_item = {
            "type": "fixed",
            "experiment": experiment,
            "split": "val",
            "logit_adjust": str(item.get("logit_adjust", "on")),
            "logit_adjust_tau": tau_value,
            "output_dir": _render_tau_template(str(item["val_output_dir_template"]), tau_value),
        }
        metrics = _run_fixed_item(
            val_item,
            context=context,
            model=model,
            accelerator=accelerator,
            labels=labels,
            letter_order=letter_order,
            label_prefix=label_prefix,
            label_token_ids=label_token_ids,
            class_weights=class_weights,
            coverage_label_token_ids=coverage_label_token_ids,
            coverage_class_weights=coverage_class_weights,
            coverage_enabled=coverage_enabled,
            coverage_cfg=coverage_cfg,
            split_samples=split_samples,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            log_predictions=log_predictions,
            force_eval=force_eval,
        )
        rows.append({"tau": tau_value, "metric": float(metrics[metric])})

    if not rows:
        raise ValueError(f"val-selected tau item has no taus: experiment={experiment}")
    selected = sorted(rows, key=lambda row: (-float(row["metric"]), float(row["tau"])))[0]
    selected_tau = float(selected["tau"])
    if accelerator.is_main_process:
        logger.info(
            "[multi-infer] selected tau=%.6g for experiment=%s by val %s=%.6f",
            selected_tau,
            experiment,
            metric,
            float(selected["metric"]),
        )
    test_item = {
        "type": "fixed",
        "experiment": experiment,
        "split": "test",
        "logit_adjust": str(item.get("logit_adjust", "on")),
        "logit_adjust_tau": selected_tau,
        "output_dir": _render_tau_template(str(item["test_output_dir_template"]), selected_tau),
    }
    _run_fixed_item(
        test_item,
        context=context,
        model=model,
        accelerator=accelerator,
        labels=labels,
        letter_order=letter_order,
        label_prefix=label_prefix,
        label_token_ids=label_token_ids,
        class_weights=class_weights,
        coverage_label_token_ids=coverage_label_token_ids,
        coverage_class_weights=coverage_class_weights,
        coverage_enabled=coverage_enabled,
        coverage_cfg=coverage_cfg,
        split_samples=split_samples,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        log_predictions=log_predictions,
        force_eval=force_eval,
    )


def _run_fixed_item(
    item: Mapping[str, Any],
    *,
    context,
    model,
    accelerator: Accelerator,
    labels: list[str],
    letter_order: list[str],
    label_prefix: str,
    label_token_ids: torch.Tensor,
    class_weights: torch.Tensor,
    coverage_label_token_ids: torch.Tensor | None,
    coverage_class_weights: torch.Tensor | None,
    coverage_enabled: bool,
    coverage_cfg: Mapping[str, Any],
    split_samples: dict[str, list[Any]],
    per_device_eval_batch_size: int | None,
    dataloader_num_workers: int | None,
    log_predictions: int,
    force_eval: bool,
) -> dict[str, Any]:
    split = str(item["split"])
    output_dir = Path(str(item["output_dir"]))
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not force_eval:
        if accelerator.is_main_process:
            logger.info("[multi-infer] reuse existing eval metrics: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    effective_cfg = _config_with_logit_adjust_override(
        context.cfg,
        mode=str(item.get("logit_adjust", "config")),
        tau=float(item["logit_adjust_tau"]) if item.get("logit_adjust_tau") is not None else None,
    )
    train_cfg = effective_cfg["sft_train"]
    logit_adjust_cfg = _resolve_logit_adjust_cfg(
        context=context,
        effective_cfg=effective_cfg,
        mode=str(item.get("logit_adjust", "config")),
        tau=float(item["logit_adjust_tau"]) if item.get("logit_adjust_tau") is not None else None,
    )
    samples = split_samples.setdefault(split, _load_samples_for_split(context=context, split=split))
    dataset = LabelTokenDataset(
        samples,
        context.tokenizer,
        max_length=context.max_length,
        label_prefix=label_prefix,
        label_schema=context.label_schema,
        coverage_enabled=coverage_enabled,
        coverage_label_prefix=str(coverage_cfg.get("label_prefix", "Coverage:")),
    )
    collator = LabelTokenCollator(tokenizer=context.tokenizer, pad_to_multiple_of=8)
    dataloader = build_dataloader(
        dataset,
        collator=collator,
        batch_size=int(
            per_device_eval_batch_size
            if per_device_eval_batch_size is not None
            else int(train_cfg.get("per_device_eval_batch_size", 1))
        ),
        num_workers=int(
            dataloader_num_workers
            if dataloader_num_workers is not None
            else int(train_cfg.get("dataloader_num_workers", 0))
        ),
        shuffle=False,
        use_length_bucket=False,
    )
    dataloader = accelerator.prepare(dataloader)
    eval_logger = logger if accelerator.is_main_process else None
    eval_metrics = _evaluate_label_token(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        label_token_ids=label_token_ids,
        class_weights=class_weights,
        coverage_label_token_ids=coverage_label_token_ids,
        coverage_class_weights=coverage_class_weights,
        train_cfg=train_cfg,
        label_prefix=label_prefix,
        labels=labels,
        letter_order=letter_order,
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions,
        logit_adjust_cfg=logit_adjust_cfg,
    )
    true_side = _true_side_macro_f1(eval_metrics)
    checkpoint_selection_score = _checkpoint_selection_score(eval_metrics, train_cfg)
    metrics = build_serializable_metrics(eval_metrics)
    metrics.update(
        {
            "label_schema": context.label_schema,
            "eval_backend": "label_token_logits",
            "eval_experiment": str(item.get("experiment", "")),
            "checkpoint": context.checkpoint_name,
            "split": split,
            "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
            "eval_ce_loss": float(eval_metrics.get("eval_ce_loss", float("nan"))),
            "eval_ordinal_loss": float(eval_metrics.get("eval_ordinal_loss", float("nan"))),
            "true_side_macro_f1": true_side,
            "checkpoint_selection_score": checkpoint_selection_score,
            "logit_adjust": logit_adjust_cfg or {"enabled": False},
        }
    )
    if accelerator.is_main_process:
        log_eval_summary(
            eval_metrics,
            eval_logger=logger,
            split=split,
            checkpoint=context.checkpoint_name,
            extra_metrics={
                "true_side_macro_f1": true_side,
                "checkpoint_selection_score": checkpoint_selection_score,
            },
        )
        if logit_adjust_cfg:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "logit_adjust.json").write_text(
                json.dumps(logit_adjust_cfg, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        artifacts = save_eval_artifacts(
            eval_dir=output_dir,
            metrics=metrics,
            confusion_matrix=eval_metrics["confusion_matrix"],
            confusion_labels=eval_metrics["confusion_labels"],
            prediction_records=eval_metrics.get("prediction_records", []),
            predictions_filename=f"{split}_predictions.jsonl",
            title=f"Label-Token Confusion Matrix ({split}/{context.checkpoint_name})",
            labels=labels,
        )
        logger.info(
            "[multi-infer] %s %s saved to %s (metrics=%s, predictions=%s)",
            split,
            context.checkpoint_name,
            output_dir,
            artifacts["metrics_path"],
            artifacts["predictions_path"],
        )

    accelerator.wait_for_everyone()
    return metrics


def _load_samples_for_split(*, context, split: str) -> list[Any]:
    data_cfg = context.cfg["data"]
    split_map = {
        "train": str(data_cfg["train_candidates"]),
        "val": str(data_cfg["val_candidates"]),
        "test": str(data_cfg["test_candidates"]),
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split={split}. Use one of {sorted(split_map)}.")
    samples = load_prebuilt_samples(load_jsonl(split_map[split]))
    if is_mistral_common_tokenizer(context.tokenizer) and any(sample.prompt_input_ids is None for sample in samples):
        raise ValueError(
            f"{split} split build rows are missing prompt_input_ids for a MistralCommon tokenizer. "
            "Rebuild the run with FORCE_BUILD=true so prompts are generated with "
            "apply_chat_template(tokenize=True)."
        )
    return samples


def _render_tau_template(template: str, tau: float) -> str:
    tag = _tau_tag(tau)
    return template.replace("{tag}", tag).replace("{tau}", str(tau))


def _tau_tag(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m").replace("+", "").removesuffix("p0")


if __name__ == "__main__":
    main()
