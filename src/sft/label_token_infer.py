from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

from fact_checking.data.constants import labels_for_schema, letter_order_for_schema
from fact_checking.utils.logging import init_logger
from sft.data.io import save_eval_artifacts
from sft.dataset.loaders import build_dataloader
from sft.infer_common import build_inference_context, build_serializable_metrics
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.label_token_trainer import (
    _build_label_token_ids,
    _class_weight_tensor,
    _evaluate_label_token,
)
from sft.runtime.adapters import checkpoint_has_peft_adapter
from sft.runtime.deps import flash_attn2_available

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
    parser.add_argument("--log-predictions", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = build_inference_context(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        split=args.split,
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

        model = AutoModelForCausalLM.from_pretrained(context.model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, str(context.checkpoint_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(context.checkpoint_dir), **model_kwargs)

    label_token_id_list, _ = _build_label_token_ids(
        context.tokenizer,
        label_prefix=label_prefix,
        letter_order=letter_order,
    )
    label_token_ids = torch.tensor(label_token_id_list, dtype=torch.long)
    class_weights = _class_weight_tensor(train_cfg, labels=labels)

    dataset = LabelTokenDataset(
        context.samples,
        context.tokenizer,
        max_length=context.max_length,
        label_prefix=label_prefix,
        label_schema=context.label_schema,
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
        label_prefix=label_prefix,
        labels=labels,
        letter_order=letter_order,
        eval_logger=eval_logger,
        log_predictions_limit=int(args.log_predictions),
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrics = build_serializable_metrics(eval_metrics)
        metrics.update(
            {
                "label_schema": context.label_schema,
                "eval_backend": "label_token_logits",
                "checkpoint": context.checkpoint_name,
                "split": context.split,
                "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
            }
        )
        eval_dir = Path(args.output_dir) if args.output_dir else context.eval_output_dir / "label_token"
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
