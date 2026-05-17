from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from fact_checking.config import load_yaml, save_yaml
from fact_checking.data.constants import LABEL2ID, LABELS, LETTER_ORDER
from fact_checking.data.io import load_jsonl
from fact_checking.utils.logging import init_logger
from sft.data.io import _save_eval_artifacts, save_model
from sft.data.sampling import select_mini_val_rows
from sft.data.types import PreparedSample
from sft.dataset.loaders import build_dataloader
from sft.eval import build_eval_metrics
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.prompting.stats import (
    build_prompt_snapshots,
    flatten_prompt_statistics,
    log_prompt_summary,
    save_prompt_statistics,
    summarize_prebuilt_prompts,
)
from sft.runtime.adapters import apply_lora_if_enabled, lora_enabled
from sft.runtime.config import apply_runtime_output_layout
from sft.runtime.deps import flash_attn2_available, fla_fast_path_available
from sft.runtime.device import enable_tf32_if_available, maybe_empty_cache
from sft.runtime.tracking import build_tracking_setup, log_metrics

logger = init_logger(__name__)


def _load_prebuilt_samples(rows: list[dict]) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for row in rows:
        gold_label = str(row.get("gold_label", ""))
        if not gold_label:
            continue
        samples.append(
            PreparedSample(
                prompt=str(row["prompt"]),
                target=str(row["target"]),
                prompt_add_special_tokens=bool(row.get("prompt_add_special_tokens", False)),
                preserve_prompt_prefix=bool(row.get("preserve_prompt_prefix", True)),
                gold_id=int(row.get("gold_id", LABEL2ID.get(gold_label, -1))),
                gold_label=gold_label,
                gold_explain=str(row.get("gold_explain", "")),
                prompt_token_count=int(row.get("prompt_token_count", 0)),
                target_token_count=int(row.get("target_token_count", 0)),
                evidence_count=int(row.get("evidence_count", 0)),
                was_truncated=bool(row.get("was_truncated", False)),
                claim=str(row.get("claim", "")),
                no_evidence=int(row.get("evidence_count", 0)) == 0,
                long_claim=len(str(row.get("claim", "")).split()) > 64,
            )
        )
    return samples


def _choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def _build_label_token_ids(tokenizer: AutoTokenizer, *, label_prefix: str) -> tuple[list[int], dict[str, Any]]:
    prefix_ids = tokenizer(label_prefix, add_special_tokens=False, truncation=False)["input_ids"]
    if not prefix_ids:
        raise ValueError(f"label_prefix={label_prefix!r} produced no tokens.")

    token_ids: list[int] = []
    token_texts: dict[str, str] = {}
    for letter in LETTER_ORDER:
        token_text = _choice_text(label_prefix, letter)
        ids = tokenizer(token_text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(
                f"label-token CE requires {token_text!r} to be one tokenizer token for letter={letter!r}; got {ids}."
            )
        token_ids.append(int(ids[0]))
        token_texts[letter] = token_text

    return token_ids, {
        "label_prefix": label_prefix,
        "prefix_token_ids": [int(x) for x in prefix_ids],
        "label_token_ids": {letter: int(token_id) for letter, token_id in zip(LETTER_ORDER, token_ids)},
        "label_token_texts": token_texts,
    }


def _class_weight_tensor(train_cfg: dict[str, Any]) -> torch.Tensor:
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    configured = label_cfg.get("class_weights", {}) or {}
    weights = [float(configured.get(label, 1.0)) for label in LABELS]
    return torch.tensor(weights, dtype=torch.float32)


def _forward_label_logits(
    model: AutoModelForCausalLM,
    batch: dict[str, Any],
    label_token_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    attention_mask = batch["attention_mask"]
    last_positions = attention_mask.long().sum(dim=1) - 1
    batch_indices = torch.arange(batch["input_ids"].shape[0], device=batch["input_ids"].device)
    next_token_logits = outputs.logits[batch_indices, last_positions]
    return next_token_logits.index_select(1, label_token_ids.to(next_token_logits.device))


def _label_name(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


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


def _evaluate_label_token(
    *,
    model: AutoModelForCausalLM,
    dataloader,
    accelerator: Accelerator,
    label_token_ids: torch.Tensor,
    class_weights: torch.Tensor,
    label_prefix: str,
    eval_logger,
    log_predictions_limit: int,
) -> dict[str, Any]:
    model.eval()

    all_pred_ids: list[torch.Tensor] = []
    all_gold_ids: list[torch.Tensor] = []
    all_sample_indices: list[torch.Tensor] = []
    all_losses: list[torch.Tensor] = []
    pad_id = -100
    sample_pad_id = -1
    dataset_samples = getattr(getattr(dataloader, "dataset", None), "samples", None)
    progress = tqdm(
        total=len(dataloader),
        desc="label-token-eval",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    with torch.no_grad():
        for batch in dataloader:
            gold_ids = batch["gold_ids"]
            label_logits = _forward_label_logits(model, batch, label_token_ids)
            loss = F.cross_entropy(
                label_logits.float(),
                gold_ids,
                weight=class_weights.to(label_logits.device, dtype=torch.float32),
            )
            pred_ids = label_logits.argmax(dim=-1).to(torch.long)

            pred_ids = accelerator.pad_across_processes(pred_ids, dim=0, pad_index=pad_id)
            gold_ids = accelerator.pad_across_processes(gold_ids, dim=0, pad_index=pad_id)
            sample_indices = accelerator.pad_across_processes(
                batch["sample_indices"],
                dim=0,
                pad_index=sample_pad_id,
            )
            gathered_pred = accelerator.gather(pred_ids)
            gathered_gold = accelerator.gather(gold_ids)
            gathered_sample_indices = accelerator.gather(sample_indices)
            valid_mask = (gathered_gold != pad_id) & (gathered_sample_indices != sample_pad_id)
            if valid_mask.any():
                all_pred_ids.append(gathered_pred[valid_mask].cpu())
                all_gold_ids.append(gathered_gold[valid_mask].cpu())
                all_sample_indices.append(gathered_sample_indices[valid_mask].cpu())

            all_losses.append(accelerator.gather_for_metrics(loss.detach().float().unsqueeze(0)).cpu())
            progress.update(1)

    progress.close()

    if not all_gold_ids:
        model.train()
        metrics = build_eval_metrics(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            prediction_records=[],
            eval_logger=eval_logger,
            log_predictions_limit=log_predictions_limit,
            log_prediction_examples=accelerator.is_main_process,
        )
        metrics["eval_loss"] = float("nan")
        return metrics

    pred_np = torch.cat(all_pred_ids).numpy()
    gold_np = torch.cat(all_gold_ids).numpy()
    sample_indices_np = torch.cat(all_sample_indices).numpy()

    prediction_records: list[dict[str, object]] = []
    if accelerator.is_main_process and dataset_samples is not None:
        for sample_idx, pred_id, gold_id in zip(sample_indices_np.tolist(), pred_np.tolist(), gold_np.tolist()):
            sample = dataset_samples[int(sample_idx)]
            letter = LETTER_ORDER[int(pred_id)]
            prediction_records.append(
                {
                    "sample_idx": int(sample_idx),
                    "prompt": str(sample.prompt),
                    "target": str(sample.target),
                    "raw_output": f"{label_prefix}{_choice_text(label_prefix, letter)}",
                    "pred_id": int(pred_id),
                    "pred_label": _label_name(int(pred_id)),
                    "gold_id": int(gold_id),
                    "gold_label": str(sample.gold_label),
                    "gold_explain": str(sample.gold_explain),
                }
            )

    metrics = build_eval_metrics(
        pred_np,
        gold_np,
        prediction_records=prediction_records if accelerator.is_main_process else [],
        eval_logger=eval_logger,
        log_predictions_limit=log_predictions_limit,
        log_prediction_examples=accelerator.is_main_process,
    )
    if all_losses:
        metrics["eval_loss"] = float(torch.cat(all_losses).mean().item())
    else:
        metrics["eval_loss"] = float("nan")
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-token weighted CE verifier training.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mini-val-size", type=int, default=None)
    parser.add_argument("--mini-val-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_runtime_output_layout(cfg)
    data_cfg = cfg["data"]
    baseline_cfg = cfg.get("baseline", {})
    train_cfg = cfg["sft_train"]
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(label_cfg.get("label_prefix", "Label:"))

    set_seed(int(train_cfg.get("seed", 42)))
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
        active_logger = init_logger(__name__, log_dir=output_dir / "logs", log_filename="train_loop.log")
        save_yaml(cfg, output_dir / "config.resolved.yaml")
        active_logger.info("[INFO] saved resolved config to %s", output_dir / "config.resolved.yaml")
    else:
        active_logger = init_logger(__name__)

    model_name_or_path = str(
        cfg.get("model_name_or_path")
        or baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(train_cfg.get("max_length", 2048))
    label_token_id_list, token_meta = _build_label_token_ids(tokenizer, label_prefix=label_prefix)
    label_token_ids = torch.tensor(label_token_id_list, dtype=torch.long)
    class_weights = _class_weight_tensor(train_cfg)

    train_prompt_summary = summarize_prebuilt_prompts(train_samples, max_length=max_length, split="train")
    val_prompt_summary = summarize_prebuilt_prompts(val_samples, max_length=max_length, split="val")
    if accelerator.is_main_process:
        log_prompt_summary(train_prompt_summary, logger=active_logger)
        log_prompt_summary(val_prompt_summary, logger=active_logger)
        prompt_stats_path = save_prompt_statistics(
            output_dir=output_dir,
            train_summary=train_prompt_summary,
            val_summary=val_prompt_summary,
            train_snapshots=build_prompt_snapshots(train_samples, split="train"),
            val_snapshots=build_prompt_snapshots(val_samples, split="val"),
        )
        active_logger.info("[INFO] prompt statistics saved to %s", prompt_stats_path)
        meta_path = output_dir / "label_token_ce_meta.json"
        meta = {
            **token_meta,
            "class_weights": {label: float(weight) for label, weight in zip(LABELS, class_weights.tolist())},
            "early_stopping_metric": str(label_cfg.get("early_stopping_metric", "macro_f1_plus_true_side")),
            "true_side_metric_weight": float(label_cfg.get("true_side_metric_weight", 0.5)),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        active_logger.info("[INFO] label-token CE metadata saved to %s", meta_path)

    log_metrics(
        accelerator,
        flatten_prompt_statistics({"train": train_prompt_summary, "val": val_prompt_summary}),
        step=0,
        backend=tracking_setup.backend,
    )
    accelerator.wait_for_everyone()

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and mixed_precision == "bf16" else torch.float32,
    }
    if bool(train_cfg.get("use_flash_attention_2", True)):
        if flash_attn2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif accelerator.is_main_process:
            active_logger.warning(
                "[WARN] sft_train.use_flash_attention_2=true, but flash-attn is not installed. "
                "Falling back to the default attention implementation."
            )
    if accelerator.is_main_process and not fla_fast_path_available():
        active_logger.info(
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
        logger=active_logger if accelerator.is_main_process else None,
    )

    padding_strategy = str(train_cfg.get("padding", "max_length"))
    if padding_strategy not in {"max_length", "longest"}:
        raise ValueError(f"Unsupported sft_train.padding={padding_strategy}. Use 'max_length' or 'longest'.")
    use_length_bucket = bool(train_cfg.get("use_length_bucket", True))
    train_ds = LabelTokenDataset(train_samples, tokenizer, max_length=max_length, label_prefix=label_prefix)
    val_ds = LabelTokenDataset(val_samples, tokenizer, max_length=max_length, label_prefix=label_prefix)
    collator = LabelTokenCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    train_dl = build_dataloader(
        train_ds,
        collator=collator,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        num_workers=num_workers,
        shuffle=True,
        use_length_bucket=use_length_bucket,
    )
    val_dl = build_dataloader(
        val_ds,
        collator=collator,
        batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        use_length_bucket=False,
    )

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found. Check full fine-tuning or LoRA configuration.")
    if accelerator.is_main_process and lora_enabled(train_cfg):
        active_logger.info("[INFO] Optimizer will update LoRA/trainable parameters only.")

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
    pre_prepare_train_dl_len = len(train_dl)
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    post_prepare_train_dl_len = len(train_dl)
    if accelerator.is_main_process:
        active_logger.info(
            "[INFO] dataloader length around accelerator.prepare: before=%d, after=%d",
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

    logging_steps = int(train_cfg.get("logging_steps", 20))
    eval_steps = int(train_cfg.get("eval_steps", 500))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    empty_cache_steps = int(train_cfg.get("empty_cache_steps", 0))
    empty_cache_on_eval = bool(train_cfg.get("empty_cache_on_eval", False))
    empty_cache_on_save = bool(train_cfg.get("empty_cache_on_save", False))
    patience = int(train_cfg.get("early_stopping_patience", 0))

    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)
    global_step = 0
    best_score = float("-inf")
    no_improve_count = 0
    should_stop = False

    def run_eval_and_maybe_save_best(step: int) -> None:
        nonlocal best_score, no_improve_count, should_stop
        eval_metrics = _evaluate_label_token(
            model=model,
            dataloader=val_dl,
            accelerator=accelerator,
            label_token_ids=label_token_ids,
            class_weights=class_weights,
            label_prefix=label_prefix,
            eval_logger=active_logger,
            log_predictions_limit=int(train_cfg.get("eval_log_predictions", 5)),
        )
        score = _selection_score(eval_metrics, train_cfg)
        macro_f1 = float(eval_metrics["macro_f1"])
        true_side = _true_side_macro_f1(eval_metrics)
        per_class = eval_metrics.get("per_class", {})

        if accelerator.is_main_process:
            active_logger.info(
                "[eval] step=%d loss=%.4f accuracy=%.4f macro_f1=%.4f true_side_macro_f1=%.4f selection_score=%.4f",
                step,
                float(eval_metrics.get("eval_loss", float("nan"))),
                float(eval_metrics["accuracy"]),
                macro_f1,
                true_side,
                score,
            )
            if isinstance(per_class, dict) and per_class:
                active_logger.info("[eval] per_class:")
                for label in sorted(per_class.keys()):
                    label_metrics = per_class[label]
                    if isinstance(label_metrics, dict):
                        active_logger.info(
                            "  - %s: P=%.4f R=%.4f F1=%.4f",
                            label,
                            float(label_metrics.get("precision", 0.0)),
                            float(label_metrics.get("recall", 0.0)),
                            float(label_metrics.get("f1", 0.0)),
                        )

        log_metrics(
            accelerator,
            {
                "eval/loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "eval/accuracy": float(eval_metrics["accuracy"]),
                "eval/macro_precision": float(eval_metrics["macro_precision"]),
                "eval/macro_recall": float(eval_metrics["macro_recall"]),
                "eval/macro_f1": macro_f1,
                "eval/true_side_macro_f1": true_side,
                "eval/selection_score": score,
                "eval/parse_error_rate": float(eval_metrics["parse_error_rate"]),
            },
            step=step,
            backend=tracking_setup.backend,
        )
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
                        step=step,
                        backend=tracking_setup.backend,
                    )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            artifacts = _save_eval_artifacts(
                output_dir=output_dir,
                global_step=step,
                metrics={
                    "step": step,
                    "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                    "accuracy": float(eval_metrics["accuracy"]),
                    "macro_precision": float(eval_metrics["macro_precision"]),
                    "macro_recall": float(eval_metrics["macro_recall"]),
                    "macro_f1": macro_f1,
                    "true_side_macro_f1": true_side,
                    "selection_score": score,
                    "parse_error_rate": float(eval_metrics["parse_error_rate"]),
                    "per_class": per_class,
                },
                confusion_matrix=eval_metrics["confusion_matrix"],
                confusion_labels=eval_metrics["confusion_labels"],
                prediction_records=eval_metrics.get("prediction_records", []),
            )
            active_logger.info(
                "[eval] artifacts saved: metrics=%s confusion=%s",
                artifacts["metrics_path"],
                artifacts["confusion_png_path"],
            )

        if score > best_score:
            best_score = score
            save_model(accelerator, model, tokenizer, output_dir / "best")
            no_improve_count = 0
        elif patience > 0:
            no_improve_count += 1
            if no_improve_count >= patience:
                if accelerator.is_main_process:
                    active_logger.info(
                        "[early-stop] no val selection-score improvement for %d evals, stopping at step=%d",
                        patience,
                        step,
                    )
                should_stop = True

        if empty_cache_on_eval:
            maybe_empty_cache(accelerator)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            with accelerator.accumulate(model):
                label_logits = _forward_label_logits(model, batch, label_token_ids)
                loss = F.cross_entropy(
                    label_logits.float(),
                    batch["gold_ids"],
                    weight=class_weights.to(label_logits.device, dtype=torch.float32),
                )
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
                    run_eval_and_maybe_save_best(global_step)

                if global_step % save_steps == 0:
                    save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_step}")
                    if empty_cache_on_save:
                        maybe_empty_cache(accelerator)

                if empty_cache_steps > 0 and global_step % empty_cache_steps == 0:
                    maybe_empty_cache(accelerator)

                if global_step >= max_train_steps or should_stop:
                    break

        if should_stop or global_step >= max_train_steps:
            break

    if best_score == float("-inf"):
        run_eval_and_maybe_save_best(global_step)

    save_model(accelerator, model, tokenizer, output_dir / "final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
