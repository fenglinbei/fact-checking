from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator

from fact_checking.data.constants import labels_for_schema, letter_order_for_schema
from fact_checking.utils.logging import init_logger
from sft.data.io import save_eval_artifacts
from sft.dataset.loaders import build_dataloader
from sft.eval import log_eval_summary
from sft.infer_common import build_inference_context, build_serializable_metrics
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.label_token_trainer import (
    _coverage_class_weight_tensor,
    _coverage_label_token_cfg,
    _coverage_label_token_enabled,
    _build_label_token_ids,
    _class_weight_tensor,
    _evaluate_label_token,
    _checkpoint_selection_score,
    _true_side_macro_f1,
)
from sft.logit_adjust import build_logit_adjust_cfg_from_train_config, load_logit_adjust_cfg
from sft.runtime.adapters import checkpoint_has_peft_adapter
from sft.runtime.deps import flash_attn2_available
from sft.runtime.model_loading import load_causal_lm_compatible_model

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SFT checkpoint with label-token logits.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--log-predictions", type=int, default=0)
    parser.add_argument(
        "--logit-adjust",
        choices=["config", "on", "off"],
        default="config",
        help="Use config/default logit adjustment, force it on, or force it off for eval-only runs.",
    )
    parser.add_argument("--logit-adjust-tau", type=float, default=None, help="Override sft_train.logit_adjust.tau.")
    return parser.parse_args()


def _config_with_logit_adjust_override(
    cfg: dict,
    *,
    mode: str,
    tau: float | None,
) -> dict:
    if mode == "config" and tau is None:
        return cfg

    updated = dict(cfg)
    sft_train = dict(updated.get("sft_train") or {})
    logit_adjust = dict(sft_train.get("logit_adjust") or {})
    if mode == "on":
        logit_adjust["enabled"] = True
    elif mode == "off":
        logit_adjust["enabled"] = False
    if tau is not None:
        logit_adjust["tau"] = float(tau)
    sft_train["logit_adjust"] = logit_adjust
    updated["sft_train"] = sft_train
    return updated


def _resolve_logit_adjust_cfg(
    *,
    context,
    effective_cfg: dict,
    mode: str,
    tau: float | None,
) -> dict | None:
    if mode == "off":
        return None

    logit_adjust_cfg = load_logit_adjust_cfg(context.run_dir)
    if logit_adjust_cfg is None or mode == "on":
        logit_adjust_cfg = build_logit_adjust_cfg_from_train_config(effective_cfg, context.tokenizer)
    if logit_adjust_cfg is None:
        if mode == "on":
            raise RuntimeError("logit_adjust was forced on, but no valid label-token logit_adjust config could be built.")
        return None
    if tau is not None:
        logit_adjust_cfg = dict(logit_adjust_cfg)
        logit_adjust_cfg["tau"] = float(tau)
    return logit_adjust_cfg


def main() -> None:
    args = parse_args()
    context = build_inference_context(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        config_path=args.config,
    )
    effective_cfg = _config_with_logit_adjust_override(
        context.cfg,
        mode=str(args.logit_adjust),
        tau=args.logit_adjust_tau,
    )
    train_cfg = effective_cfg["sft_train"]
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(label_cfg.get("label_prefix", "Label:"))
    labels = labels_for_schema(context.label_schema)
    letter_order = letter_order_for_schema(context.label_schema)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)

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
        model = PeftModel.from_pretrained(model, str(context.checkpoint_dir))
    else:
        model = load_causal_lm_compatible_model(str(context.checkpoint_dir), **model_kwargs)

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
        from fact_checking.data.constants import COVERAGE_LETTER_ORDER

        coverage_label_token_id_list, _ = _build_label_token_ids(
            context.tokenizer,
            label_prefix=str(coverage_cfg.get("label_prefix", "Coverage:")),
            letter_order=COVERAGE_LETTER_ORDER,
        )
        coverage_label_token_ids = torch.tensor(coverage_label_token_id_list, dtype=torch.long)
        coverage_class_weights = _coverage_class_weight_tensor(train_cfg)
    logit_adjust_cfg = _resolve_logit_adjust_cfg(
        context=context,
        effective_cfg=effective_cfg,
        mode=str(args.logit_adjust),
        tau=args.logit_adjust_tau,
    )

    dataset = LabelTokenDataset(
        context.samples,
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
            args.per_device_eval_batch_size
            if args.per_device_eval_batch_size is not None
            else int(train_cfg.get("per_device_eval_batch_size", 1))
        ),
        num_workers=int(
            args.dataloader_num_workers
            if args.dataloader_num_workers is not None
            else int(train_cfg.get("dataloader_num_workers", 0))
        ),
        shuffle=False,
        use_length_bucket=False,
    )
    model, dataloader = accelerator.prepare(model, dataloader)
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
        log_predictions_limit=int(args.log_predictions),
        logit_adjust_cfg=logit_adjust_cfg,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        true_side = _true_side_macro_f1(eval_metrics)
        checkpoint_selection_score = _checkpoint_selection_score(eval_metrics, train_cfg)
        log_eval_summary(
            eval_metrics,
            eval_logger=logger,
            split=context.split,
            checkpoint=context.checkpoint_name,
            extra_metrics={
                "true_side_macro_f1": true_side,
                "checkpoint_selection_score": checkpoint_selection_score,
            },
        )
        metrics = build_serializable_metrics(eval_metrics)
        metrics.update(
            {
                "label_schema": context.label_schema,
                "eval_backend": "label_token_logits",
                "checkpoint": context.checkpoint_name,
                "split": context.split,
                "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "eval_ce_loss": float(eval_metrics.get("eval_ce_loss", float("nan"))),
                "eval_ordinal_loss": float(eval_metrics.get("eval_ordinal_loss", float("nan"))),
                "true_side_macro_f1": true_side,
                "checkpoint_selection_score": checkpoint_selection_score,
                "logit_adjust": logit_adjust_cfg or {"enabled": False},
            }
        )
        eval_dir = Path(args.output_dir) if args.output_dir else context.eval_output_dir / "label_token"
        if logit_adjust_cfg:
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "logit_adjust.json").write_text(
                json.dumps(logit_adjust_cfg, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        artifacts = save_eval_artifacts(
            eval_dir=eval_dir,
            metrics=metrics,
            confusion_matrix=eval_metrics["confusion_matrix"],
            confusion_labels=eval_metrics["confusion_labels"],
            prediction_records=eval_metrics.get("prediction_records", []),
            predictions_filename=f"{context.split}_predictions.jsonl",
            title=f"Label-Token Confusion Matrix ({context.split}/{context.checkpoint_name})",
            labels=labels,
        )
        logger.info(
            "[INFO] %s label-token eval for %s saved to %s (metrics=%s, predictions=%s)",
            context.split,
            context.checkpoint_name,
            eval_dir,
            artifacts["metrics_path"],
            artifacts["predictions_path"],
        )


if __name__ == "__main__":
    main()
