from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from fact_checking.data.io import load_jsonl
from fact_checking.config import load_yaml, save_yaml
from fact_checking.utils.logging import init_logger
from sft.data.io import _save_eval_artifacts, save_model
from sft.data.labels import normalize_gold_label
from sft.data.sampling import select_mini_val_rows
from sft.data.types import PreparedSample
from sft.dataset.collators import CausalLMSFTCollator
from sft.dataset.datasets import EvalPromptDataset, SFTDatasetBuilder
from sft.dataset.loaders import build_dataloader, build_eval_dataloader
from sft.eval import evaluate
from sft.prompting.stats import (
    build_prompt_snapshots,
    log_prompt_summary,
    save_prompt_statistics,
    summarize_prebuilt_prompts,
)
from sft.runtime.adapters import apply_lora_if_enabled, lora_enabled
from sft.runtime.config import apply_runtime_output_layout
from sft.runtime.deps import flash_attn2_available, fla_fast_path_available
from sft.runtime.device import enable_tf32_if_available, maybe_empty_cache
from sft.runtime.tracking import build_tracking_setup, log_metrics
from sft.vllm_online_eval import OnlineVLLMEvaluator, online_vllm_eval_enabled

logger = init_logger(__name__)


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
            long_claim=len(str(row.get("claim", "")).split()) > 64,
        ))
    return samples


def _broadcast_object_from_main(obj: object, accelerator: Accelerator) -> object:
    if accelerator.num_processes <= 1:
        return obj
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return obj
    payload = [obj if accelerator.is_main_process else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT train for fact-checking experiments (Accelerate).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mini-val-size",
        type=int,
        default=None,
        help="If > 0, randomly sample this many examples from val set for faster eval (e.g., 32/64/128).",
    )
    parser.add_argument(
        "--mini-val-seed",
        type=int,
        default=None,
        help="Random seed for mini val sampling. Falls back to sft_train.seed or 42.",
    )
    parser.add_argument(
        "--prompt-length-stats-only",
        action="store_true",
        help="Only build prompts and report prompt token length statistics, then exit.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_runtime_output_layout(cfg)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]
    train_cfg = cfg["sft_train"]

    enable_tf32_if_available()
    tracking_setup = build_tracking_setup(cfg)

    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        mixed_precision=mixed_precision,
        log_with=tracking_setup.log_with,
    )

    if tracking_setup.enabled:
        accelerator.init_trackers(
            project_name=tracking_setup.project_name,
            config=cfg,
            init_kwargs=tracking_setup.init_kwargs,
        )

    train_rows = load_jsonl(data_cfg["train_candidates"])
    val_rows = load_jsonl(data_cfg["val_candidates"])

    mini_val_size = int(
        args.mini_val_size if args.mini_val_size is not None else int(train_cfg.get("mini_val_size", 0))
    )
    mini_val_seed = int(
        args.mini_val_seed
        if args.mini_val_seed is not None
        else int(train_cfg.get("mini_val_seed", train_cfg.get("seed", 42)))
    )
    val_rows = select_mini_val_rows(
        rows=val_rows,
        mini_val_size=mini_val_size,
        mini_val_seed=mini_val_seed,
        accelerator=accelerator,
    )

    train_samples = _load_prebuilt_samples(train_rows)
    val_samples = _load_prebuilt_samples(val_rows)

    output_dir = Path(cfg.get("output_dir", "outputs/runs/train"))
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger = init_logger(__name__, log_dir=output_dir / "logs", log_filename="train_loop.log")
        save_yaml(cfg, output_dir / "config.resolved.yaml")
        logger.info("[INFO] saved resolved config to %s", output_dir / "config.resolved.yaml")
    else:
        logger = init_logger(__name__)

    model_name_or_path = str(
        cfg.get("model_name_or_path")
        or baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(train_cfg.get("max_length", 2048))

    train_instances = [
        {
            "prompt": sample.prompt,
            "target": sample.target,
            "prompt_add_special_tokens": sample.prompt_add_special_tokens,
            "preserve_prompt_prefix": sample.preserve_prompt_prefix,
        }
        for sample in train_samples
    ]
    val_instances = [
        {
            "prompt": sample.prompt,
            "target": sample.target,
            "prompt_add_special_tokens": sample.prompt_add_special_tokens,
            "preserve_prompt_prefix": sample.preserve_prompt_prefix,
        }
        for sample in val_samples
    ]
    val_eval_ds = EvalPromptDataset(val_samples)

    train_prompt_summary = summarize_prebuilt_prompts(
        train_samples,
        max_length=max_length,
        split="train",
    )
    val_prompt_summary = summarize_prebuilt_prompts(
        val_samples,
        max_length=max_length,
        split="val",
    )
    if accelerator.is_main_process:
        log_prompt_summary(train_prompt_summary, logger=logger)
        log_prompt_summary(val_prompt_summary, logger=logger)
        train_prompt_snapshots = build_prompt_snapshots(
            train_samples,
            split="train",
        )
        val_prompt_snapshots = build_prompt_snapshots(
            val_samples,
            split="val",
        )
        prompt_stats_path = save_prompt_statistics(
            output_dir=output_dir,
            train_summary=train_prompt_summary,
            val_summary=val_prompt_summary,
            train_snapshots=train_prompt_snapshots,
            val_snapshots=val_prompt_snapshots,
        )
        logger.info("[INFO] prompt statistics saved to %s", prompt_stats_path)

    accelerator.wait_for_everyone()

    if args.prompt_length_stats_only:
        if accelerator.is_main_process:
            logger.info("[INFO] prompt length stats finished. Exiting due to --prompt-length-stats-only.")
        return

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and mixed_precision == "bf16" else torch.float32,
    }

    if bool(train_cfg.get("use_flash_attention_2", True)):
        if flash_attn2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif accelerator.is_main_process:
            logger.warning(
                "[WARN] sft_train.use_flash_attention_2=true, but flash-attn is not installed. "
                "Falling back to the default attention implementation."
            )
    if accelerator.is_main_process and not fla_fast_path_available():
        logger.info(
            "[INFO] FLA fast path is unavailable (requires both `fla` and `causal_conv1d`). "
            "This is separate from flash-attn and does not block training."
        )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    model = apply_lora_if_enabled(
        model,
        train_cfg,
        gradient_checkpointing=gradient_checkpointing,
        logger=logger if accelerator.is_main_process else None,
    )

    cache_dir_cfg = train_cfg.get("tokenized_cache_dir")
    cache_dir = Path(str(cache_dir_cfg)) if cache_dir_cfg else output_dir / "tokenized_cache"
    padding_strategy = str(train_cfg.get("padding", "max_length"))
    if padding_strategy not in {"max_length", "longest"}:
        raise ValueError(f"Unsupported sft_train.padding={padding_strategy}. Use 'max_length' or 'longest'.")
    use_length_bucket = bool(train_cfg.get("use_length_bucket", True))

    builder = SFTDatasetBuilder(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=padding_strategy,
        cache_dir=cache_dir,
    )
    train_ds = builder.build(train_instances, split="train", accelerator=accelerator)
    val_ds = builder.build(val_instances, split="val", accelerator=accelerator)

    data_collator = CausalLMSFTCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    train_dl = build_dataloader(
        train_ds,
        collator=data_collator,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        num_workers=num_workers,
        shuffle=True,
        use_length_bucket=use_length_bucket,
    )
    val_dl = build_dataloader(
        val_ds,
        collator=data_collator,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        use_length_bucket=False,
    )

    val_eval_dl = build_eval_dataloader(
        val_eval_ds,
        tokenizer=tokenizer,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        max_length=max_length,
        padding=padding_strategy,
    )

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found. Check full fine-tuning or LoRA configuration.")
    if accelerator.is_main_process and lora_enabled(train_cfg):
        logger.info("[INFO] Optimizer will update LoRA/trainable parameters only.")

    optimizer = AdamW(
        trainable_parameters,
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )

    num_epochs = int(math.ceil(float(train_cfg.get("num_train_epochs", 2.0))))
    effective_grad_accum_steps = int(accelerator.gradient_accumulation_steps)
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is not None:
        ds_cfg = getattr(ds_plugin, "deepspeed_config", {}) or {}
        ds_grad_accum_steps = ds_cfg.get("gradient_accumulation_steps")
        if ds_grad_accum_steps is not None:
            effective_grad_accum_steps = int(ds_grad_accum_steps)
    cfg_grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", effective_grad_accum_steps))
    if accelerator.is_main_process and effective_grad_accum_steps != cfg_grad_accum_steps:
        logger.warning(
            "[WARN] gradient_accumulation_steps mismatch detected: sft_train=%d, effective=%d. "
            "Using effective value for max_train_steps/progress bar.",
            cfg_grad_accum_steps,
            effective_grad_accum_steps,
        )
    pre_prepare_train_dl_len = len(train_dl)
    model, optimizer, train_dl, val_dl, val_eval_dl = accelerator.prepare(
        model, optimizer, train_dl, val_dl, val_eval_dl
    )
    post_prepare_train_dl_len = len(train_dl)
    if accelerator.is_main_process:
        logger.info(
            "[INFO] dataloader length around accelerator.prepare: before=%d, after=%d",
            pre_prepare_train_dl_len,
            post_prepare_train_dl_len,
        )
    if accelerator.is_main_process and post_prepare_train_dl_len != pre_prepare_train_dl_len:
        logger.warning(
            "[WARN] len(train_dl) changed after accelerator.prepare: before=%d, after=%d. "
            "This may indicate duplicated sharding/re-partitioning across distributed samplers.",
            pre_prepare_train_dl_len,
            post_prepare_train_dl_len,
        )
    update_steps_per_epoch = max(1, math.ceil(post_prepare_train_dl_len / effective_grad_accum_steps))
    max_train_steps = num_epochs * update_steps_per_epoch
    warmup_steps = int(max_train_steps * float(train_cfg.get("warmup_ratio", 0.03)))
    scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    scheduler = accelerator.prepare(scheduler)
    if accelerator.is_main_process:
        logger.info(
            "[INFO] train progress setup (post-prepare): num_epochs=%d, len(train_dl)=%d, "
            "effective_grad_accum_steps=%d, update_steps_per_epoch=%d, max_train_steps=%d",
            num_epochs,
            post_prepare_train_dl_len,
            effective_grad_accum_steps,
            update_steps_per_epoch,
            max_train_steps,
        )

    online_vllm_evaluator: OnlineVLLMEvaluator | None = None
    use_online_vllm_eval = online_vllm_eval_enabled(train_cfg)
    if use_online_vllm_eval:
        if accelerator.num_processes < 1:
            raise RuntimeError("online_vllm_eval requires at least one training process.")
        init_error = None
        if accelerator.is_main_process:
            try:
                online_vllm_evaluator = OnlineVLLMEvaluator(
                    model_name_or_path=model_name_or_path,
                    tokenizer_name_or_path=model_name_or_path,
                    samples=val_samples,
                    max_length=max_length,
                    temperature=float(train_cfg.get("temperature", baseline_cfg.get("temperature", 0.0))),
                    baseline_cfg=baseline_cfg,
                    train_cfg=train_cfg,
                    logger=logger,
                )
            except Exception:
                init_error = traceback.format_exc()
        init_error = _broadcast_object_from_main(init_error, accelerator)
        if init_error:
            raise RuntimeError(f"Failed to initialize online vLLM evaluator on rank 0:\n{init_error}")
        accelerator.wait_for_everyone()

    logging_steps = int(train_cfg.get("logging_steps", 20))
    eval_steps = int(train_cfg.get("eval_steps", 500))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    empty_cache_steps = int(train_cfg.get("empty_cache_steps", 0))
    empty_cache_on_eval = bool(train_cfg.get("empty_cache_on_eval", False))
    empty_cache_on_save = bool(train_cfg.get("empty_cache_on_save", False))

    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)
    global_step = 0
    best_val_loss = float("-inf")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % logging_steps == 0:
                    train_loss = accelerator.gather_for_metrics(loss.detach().float().unsqueeze(0)).mean().item()
                    lr = scheduler.get_last_lr()[0]
                    log_metrics(
                        accelerator,
                        {"train/loss": train_loss, "train/lr": lr, "train/epoch": epoch},
                        step=global_step,
                        backend=tracking_setup.backend,
                    )

                if global_step % eval_steps == 0:
                    if use_online_vllm_eval:
                        accelerator.wait_for_everyone()
                        eval_error = None
                        if accelerator.is_main_process:
                            try:
                                if online_vllm_evaluator is None:
                                    raise RuntimeError(
                                        "online_vllm_eval is enabled but the rank-0 evaluator is missing."
                                    )
                                eval_metrics = online_vllm_evaluator.evaluate(
                                    model=accelerator.unwrap_model(model),
                                    max_new_tokens=int(train_cfg.get("max_new_tokens", baseline_cfg.get("max_new_tokens", 24))),
                                    log_predictions_limit=int(train_cfg.get("eval_log_predictions", 5)),
                                )
                            except Exception:
                                eval_metrics = None
                                eval_error = traceback.format_exc()
                        else:
                            eval_metrics = None
                        eval_error = _broadcast_object_from_main(eval_error, accelerator)
                        if eval_error:
                            raise RuntimeError(f"Online vLLM eval failed on rank 0:\n{eval_error}")
                        eval_metrics = _broadcast_object_from_main(eval_metrics, accelerator)
                        accelerator.wait_for_everyone()
                    else:
                        eval_metrics = evaluate(
                            model,
                            val_eval_dl,
                            tokenizer,
                            accelerator,
                            max_length=max_length,
                            max_new_tokens=int(train_cfg.get("max_new_tokens", baseline_cfg.get("max_new_tokens", 24))),
                            eval_logger=logger,
                            log_predictions_limit=int(train_cfg.get("eval_log_predictions", 5)),
                        )
                    macro_f1 = float(eval_metrics["macro_f1"])
                    if accelerator.is_main_process:
                        per_class_summary = eval_metrics.get("per_class", {}) or {}
                        per_class_lines = []
                        if isinstance(per_class_summary, dict):
                            for label in sorted(per_class_summary.keys()):
                                label_metrics = per_class_summary[label]
                                if isinstance(label_metrics, dict):
                                    per_class_lines.append(
                                        f"  - {label}: P={float(label_metrics.get('precision', 0.0)):.4f} "
                                        f"R={float(label_metrics.get('recall', 0.0)):.4f} "
                                        f"F1={float(label_metrics.get('f1', 0.0)):.4f}"
                                    )
                        summary_lines = [
                            f"[eval] step={global_step} "
                            f"accuracy={float(eval_metrics['accuracy']):.4f} "
                            f"macro_precision={float(eval_metrics['macro_precision']):.4f} "
                            f"macro_recall={float(eval_metrics['macro_recall']):.4f} "
                            f"macro_f1={macro_f1:.4f} "
                            f"parse_error_rate={float(eval_metrics['parse_error_rate']):.4f}",
                        ]
                        if per_class_lines:
                            summary_lines.append("[eval] per_class:")
                            summary_lines.extend(per_class_lines)
                        for line in summary_lines:
                            logger.info(line)
                    log_metrics(
                        accelerator,
                        {
                            "eval/accuracy": float(eval_metrics["accuracy"]),
                            "eval/macro_precision": float(eval_metrics["macro_precision"]),
                            "eval/macro_recall": float(eval_metrics["macro_recall"]),
                            "eval/macro_f1": macro_f1,
                            "eval/parse_error_rate": float(eval_metrics["parse_error_rate"]),
                        },
                        step=global_step,
                        backend=tracking_setup.backend,
                    )
                    per_class = eval_metrics.get("per_class", {})
                    if isinstance(per_class, dict):
                        for label, label_metrics in per_class.items():
                            if isinstance(label_metrics, dict):
                                log_metrics(
                                    accelerator,
                                    {
                                        f"eval/{label}/precision": float(label_metrics.get("precision", 0.0)),
                                        f"eval/{label}/recall": float(label_metrics.get("recall", 0.0)),
                                        f"eval/{label}/f1": float(label_metrics.get("f1", 0.0)),
                                    },
                                    step=global_step,
                                    backend=tracking_setup.backend,
                                )

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        artifacts = _save_eval_artifacts(
                            output_dir=output_dir,
                            global_step=global_step,
                            metrics={
                                "step": global_step,
                                "accuracy": float(eval_metrics["accuracy"]),
                                "macro_precision": float(eval_metrics["macro_precision"]),
                                "macro_recall": float(eval_metrics["macro_recall"]),
                                "macro_f1": macro_f1,
                                "parse_error_rate": float(eval_metrics["parse_error_rate"]),
                                "per_class": per_class,
                            },
                            confusion_matrix=eval_metrics["confusion_matrix"],
                            confusion_labels=eval_metrics["confusion_labels"],
                            prediction_records=eval_metrics.get("prediction_records", []),
                        )
                        log_metrics(
                            accelerator,
                            {
                                "eval/metrics_path": artifacts["metrics_path"],
                                "eval/confusion_data_path": artifacts["confusion_data_path"],
                                "eval/confusion_png_path": artifacts["confusion_png_path"],
                                "eval/predictions_path": artifacts["predictions_path"],
                            },
                            step=global_step,
                            backend=tracking_setup.backend,
                        )

                    if macro_f1 > best_val_loss:
                        best_val_loss = macro_f1
                        save_model(accelerator, model, tokenizer, output_dir / "best")

                    if empty_cache_on_eval:
                        maybe_empty_cache(accelerator)

                if global_step % save_steps == 0:
                    save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_step}")

                    if empty_cache_on_save:
                        maybe_empty_cache(accelerator)

                if empty_cache_steps > 0 and global_step % empty_cache_steps == 0:
                    maybe_empty_cache(accelerator)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    save_model(accelerator, model, tokenizer, output_dir / "final")
    if online_vllm_evaluator is not None:
        online_vllm_evaluator.shutdown()
    accelerator.end_training()


if __name__ == "__main__":
    main()
