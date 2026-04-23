from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from fact_checking.baselines.llm_baseline import load_jsonl
from fact_checking.config import load_yaml, save_yaml
from fact_checking.utils.logging import init_logger
from sft.data.io import _save_eval_artifacts, save_model
from sft.data.sampling import select_mini_val_rows
from sft.dataset.collators import CausalLMSFTCollator
from sft.dataset.datasets import EvalPromptDataset, SFTDatasetBuilder
from sft.dataset.loaders import build_dataloader, build_eval_dataloader
from sft.eval import evaluate
from sft.prompting.output import build_output_strategy
from sft.prompting.preparation import build_prepared_samples
from sft.prompting.stats import (
    build_prompt_snapshots,
    log_prompt_summary,
    save_prompt_statistics,
    summarize_prompt_preparation,
)
from sft.prompting.truncation import build_prompt_truncation_strategy
from sft.runtime.config import apply_runtime_output_layout, normalize_prompt_truncation_config
from sft.runtime.deps import flash_attn2_available, fla_fast_path_available
from sft.runtime.device import enable_tf32_if_available, maybe_empty_cache

logger = init_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT train for LLM baselines (Accelerate).")
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
    cfg = normalize_prompt_truncation_config(cfg)
    cfg = apply_runtime_output_layout(cfg)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]
    train_cfg = cfg["sft_train"]
    wandb_cfg = cfg.get("wandb", {})

    enable_tf32_if_available()

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "fact-checking-stage-ab")))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        os.environ.setdefault("WANDB_LOG_MODEL", str(wandb_cfg.get("log_model", "false")))
        os.environ.setdefault("WANDB_WATCH", str(wandb_cfg.get("watch", "false")))

    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    report_to = "wandb" if wandb_enabled else None
    run_name = str(wandb_cfg.get("run_name", "llm_baseline_sft"))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        mixed_precision=mixed_precision,
        log_with=report_to,
    )

    if wandb_enabled:
        accelerator.init_trackers(
            project_name=os.environ["WANDB_PROJECT"],
            config=cfg,
            init_kwargs={"wandb": {"name": run_name}},
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

    use_context = bool(baseline_cfg.get("use_context", False))
    top_k = int(baseline_cfg.get("top_k", 8))
    context_k = int(baseline_cfg.get("context_k", 1))
    output_strategy = build_output_strategy(baseline_cfg)
    truncation_strategy = build_prompt_truncation_strategy(baseline_cfg)

    output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline"))
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger = init_logger(__name__, log_dir=output_dir / "logs", log_filename="train_loop.log")
        save_yaml(cfg, output_dir / "config.resolved.yaml")
        logger.info("[INFO] saved resolved config to %s", output_dir / "config.resolved.yaml")
    else:
        logger = init_logger(__name__)

    model_name_or_path = str(baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(train_cfg.get("max_length", 2048))
    train_samples, train_prompt_records = build_prepared_samples(
        train_rows,
        top_k=top_k,
        use_context=use_context,
        context_k=context_k,
        tokenizer=tokenizer,
        max_length=max_length,
        output_strategy=output_strategy,
        truncation_strategy=truncation_strategy,
    )
    val_samples, val_prompt_records = build_prepared_samples(
        val_rows,
        top_k=top_k,
        use_context=use_context,
        context_k=context_k,
        tokenizer=tokenizer,
        max_length=max_length,
        output_strategy=output_strategy,
        truncation_strategy=truncation_strategy,
    )

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

    train_prompt_summary = summarize_prompt_preparation(
        train_prompt_records,
        max_length=max_length,
        split="train",
        truncation_strategy_name=truncation_strategy.name,
        output_mode=output_strategy.name,
        prompt_version=output_strategy.prompt_version,
    )
    val_prompt_summary = summarize_prompt_preparation(
        val_prompt_records,
        max_length=max_length,
        split="val",
        truncation_strategy_name=truncation_strategy.name,
        output_mode=output_strategy.name,
        prompt_version=output_strategy.prompt_version,
    )
    if accelerator.is_main_process:
        log_prompt_summary(train_prompt_summary, logger=logger)
        log_prompt_summary(val_prompt_summary, logger=logger)
        train_prompt_snapshots = build_prompt_snapshots(
            train_samples,
            train_prompt_records,
            split="train",
        )
        val_prompt_snapshots = build_prompt_snapshots(
            val_samples,
            val_prompt_records,
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

    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

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

    optimizer = AdamW(
        model.parameters(),
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
                    accelerator.log({"train/loss": train_loss, "train/lr": lr, "train/epoch": epoch}, step=global_step)

                if global_step % eval_steps == 0:
                    logger.info("[rank %d] before evaluate step=%d", accelerator.process_index, global_step)
                    eval_metrics = evaluate(
                        model,
                        val_eval_dl,
                        tokenizer,
                        accelerator,
                        max_length=max_length,
                        max_new_tokens=int(baseline_cfg.get("max_new_tokens", 24)),
                        eval_logger=logger,
                        log_predictions_limit=int(train_cfg.get("eval_log_predictions", 5)),
                    )
                    logger.info("[rank %d] after evaluate step=%d", accelerator.process_index, global_step)
                    macro_f1 = float(eval_metrics["macro_f1"])
                    accelerator.log(
                        {
                            "eval/accuracy": float(eval_metrics["accuracy"]),
                            "eval/macro_precision": float(eval_metrics["macro_precision"]),
                            "eval/macro_recall": float(eval_metrics["macro_recall"]),
                            "eval/macro_f1": macro_f1,
                            "eval/parse_error_rate": float(eval_metrics["parse_error_rate"]),
                        },
                        step=global_step,
                    )
                    per_class = eval_metrics.get("per_class", {})
                    if isinstance(per_class, dict):
                        for label, label_metrics in per_class.items():
                            if isinstance(label_metrics, dict):
                                accelerator.log(
                                    {
                                        f"eval/{label}/precision": float(label_metrics.get("precision", 0.0)),
                                        f"eval/{label}/recall": float(label_metrics.get("recall", 0.0)),
                                        f"eval/{label}/f1": float(label_metrics.get("f1", 0.0)),
                                    },
                                    step=global_step,
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
                        accelerator.log(
                            {
                                "eval/metrics_path": artifacts["metrics_path"],
                                "eval/confusion_data_path": artifacts["confusion_data_path"],
                                "eval/confusion_png_path": artifacts["confusion_png_path"],
                                "eval/predictions_path": artifacts["predictions_path"],
                            },
                            step=global_step,
                        )

                    if macro_f1 > best_val_loss:
                        best_val_loss = macro_f1
                        logger.info("[rank %d] before save best step=%d", accelerator.process_index, global_step)
                        save_model(accelerator, model, tokenizer, output_dir / "best")
                        logger.info("[rank %d] after save best step=%d", accelerator.process_index, global_step)

                    if empty_cache_on_eval:
                        maybe_empty_cache(accelerator)

                if global_step % save_steps == 0:
                    logger.info("[rank %d] before save checkpoint step=%d", accelerator.process_index, global_step)
                    save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_step}")
                    logger.info("[rank %d] after save checkpoint step=%d", accelerator.process_index, global_step)

                    if empty_cache_on_save:
                        maybe_empty_cache(accelerator)

                if empty_cache_steps > 0 and global_step % empty_cache_steps == 0:
                    maybe_empty_cache(accelerator)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    save_model(accelerator, model, tokenizer, output_dir / "final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
