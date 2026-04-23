from __future__ import annotations

import argparse

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

from fact_checking.utils.logging import init_logger
from sft.data.io import save_eval_artifacts
from sft.dataset.datasets import EvalPromptDataset
from sft.dataset.loaders import build_eval_dataloader
from sft.eval import evaluate
from sft.infer_common import build_inference_context, build_serializable_metrics
from sft.runtime.deps import flash_attn2_available

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SFT checkpoint on train/val/test.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
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

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mixed_precision = "bf16" if bool(context.train_cfg.get("bf16", True)) else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and mixed_precision == "bf16" else torch.float32,
    }
    if bool(context.train_cfg.get("use_flash_attention_2", True)) and flash_attn2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(str(context.checkpoint_dir), **model_kwargs)
    eval_dataset = EvalPromptDataset(context.samples)
    eval_dataloader = build_eval_dataloader(
        eval_dataset,
        tokenizer=context.tokenizer,
        batch_size=int(
            args.per_device_eval_batch_size
            if args.per_device_eval_batch_size is not None
            else int(context.train_cfg.get("per_device_eval_batch_size", 1))
        ),
        num_workers=int(
            args.dataloader_num_workers
            if args.dataloader_num_workers is not None
            else int(context.train_cfg.get("dataloader_num_workers", 0))
        ),
        max_length=context.max_length,
        padding=str(context.train_cfg.get("padding", "max_length")),
    )

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    eval_logger = logger if accelerator.is_main_process else None
    eval_metrics = evaluate(
        model,
        eval_dataloader,
        context.tokenizer,
        accelerator,
        max_length=context.max_length,
        max_new_tokens=int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else int(context.baseline_cfg.get("max_new_tokens", 24))
        ),
        eval_logger=eval_logger,
        log_predictions_limit=int(args.log_predictions),
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        artifacts = save_eval_artifacts(
            eval_dir=context.eval_output_dir,
            metrics=build_serializable_metrics(eval_metrics),
            confusion_matrix=eval_metrics["confusion_matrix"],
            confusion_labels=eval_metrics["confusion_labels"],
            prediction_records=eval_metrics.get("prediction_records", []),
            predictions_filename=f"{context.split}_predictions.jsonl",
            title=f"Confusion Matrix ({context.split}/{context.checkpoint_name})",
        )
        logger.info(
            "[INFO] %s eval for %s saved to %s (metrics=%s, predictions=%s)",
            context.split,
            context.checkpoint_name,
            context.eval_output_dir,
            artifacts["metrics_path"],
            artifacts["predictions_path"],
        )


if __name__ == "__main__":
    main()
