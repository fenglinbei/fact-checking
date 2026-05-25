#!/usr/bin/env python3
"""Train a Qwen/LoRA sequential action selector with VIG soft supervision."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.selectors.llm_action import (
    ACTION_LABEL_MODE_GLOBAL_INDEX,
    ACTION_LABEL_MODES,
    CANDIDATE_ORDER_CANDIDATE_POOL,
    CANDIDATE_ORDER_RANDOM,
    CANDIDATE_ORDER_MODES,
    SCORE_MODE_ACTION_TOKEN,
    SCORE_MODE_CONTINUATION,
    rebuild_action_sample_with_order,
    score_action_choices,
    softmax_deltas,
)
from fact_checking.selectors.llm_action_eval import (
    evaluate_llm_action_selection,
    selection_history_record,
    write_selection_eval_outputs,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    load_stage2_oracle_examples,
    read_jsonl,
    write_json,
)
from sft.runtime.adapters import DEFAULT_LORA_TARGET_MODULES, apply_lora_if_enabled
from sft.runtime.deps import flash_attn2_available


TRAIN_ORDER_AUGMENTATION_NONE = "none"
TRAIN_ORDER_AUGMENTATION_DYNAMIC_RANDOM = "dynamic_random"
TRAIN_ORDER_AUGMENTATION_MODES = {TRAIN_ORDER_AUGMENTATION_NONE, TRAIN_ORDER_AUGMENTATION_DYNAMIC_RANDOM}


class ActionSampleDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        order_augmentation: str = TRAIN_ORDER_AUGMENTATION_NONE,
        action_label_mode: str = ACTION_LABEL_MODE_GLOBAL_INDEX,
        candidate_order_seed: int = 20260524,
    ) -> None:
        self.rows = rows
        self.order_augmentation = str(order_augmentation)
        self.action_label_mode = str(action_label_mode)
        self.candidate_order_seed = int(candidate_order_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        if self.order_augmentation == TRAIN_ORDER_AUGMENTATION_NONE:
            return row
        if self.order_augmentation != TRAIN_ORDER_AUGMENTATION_DYNAMIC_RANDOM:
            raise ValueError(f"Unsupported train order augmentation: {self.order_augmentation!r}")
        return rebuild_action_sample_with_order(
            row,
            action_label_mode=self.action_label_mode,
            candidate_order_mode=CANDIDATE_ORDER_RANDOM,
            candidate_order_seed=int(self.candidate_order_seed),
            epoch=int(self.epoch),
            row_index=int(idx),
        )


def _sample_eval_rows(rows: list[dict[str, Any]], *, limit: int, mode: str, seed: int) -> list[dict[str, Any]]:
    if int(limit) <= 0 or int(limit) >= len(rows):
        return rows
    if mode == "head":
        return rows[: int(limit)]
    if mode != "random":
        raise ValueError(f"Unsupported eval sample mode: {mode!r}")
    rng = random.Random(int(seed))
    indices = sorted(rng.sample(range(len(rows)), int(limit)))
    return [rows[idx] for idx in indices]


def _mark_sample_rows(rows: list[dict[str, Any]], *, sample_type: str) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("sample_type", str(sample_type))
        if str(sample_type) == "bad_prefix":
            item.setdefault("has_hard_target", False)
        else:
            item.setdefault("has_hard_target", True)
        marked.append(item)
    return marked


@dataclass(frozen=True)
class SwanLabRun:
    enabled: bool
    module: Any | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an LLM action evidence selector.")
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="/home/fenglin/project/hateSpeechDetection/models/base/Qwen2.5-7B-Instruct")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--choice-batch-size", type=int, default=64)
    p.add_argument("--score-mode", default=SCORE_MODE_ACTION_TOKEN, choices=[SCORE_MODE_ACTION_TOKEN, SCORE_MODE_CONTINUATION])
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=1.0e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--soft-loss-weight", type=float, default=0.3)
    p.add_argument("--soft-tau", type=float, default=0.2)
    p.add_argument("--set-loss-weight", type=float, default=0.0)
    p.add_argument("--set-loss-type", default="multi_positive_ce", choices=["multi_positive_ce", "bce"])
    p.add_argument("--hard-loss-weight", type=float, default=1.0)
    p.add_argument("--pairwise-loss-weight", type=float, default=0.05)
    p.add_argument("--bad-prefix-hard-loss-weight", type=float, default=0.0)
    p.add_argument("--bad-prefix-train-data", default=None)
    p.add_argument("--bad-prefix-val-data", default=None)
    p.add_argument(
        "--train-order-augmentation",
        default=TRAIN_ORDER_AUGMENTATION_DYNAMIC_RANDOM,
        choices=sorted(TRAIN_ORDER_AUGMENTATION_MODES),
    )
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-sample-limit", type=int, default=None)
    p.add_argument("--eval-sample-mode", default="random", choices=["head", "random"])
    p.add_argument("--eval-sample-seed", type=int, default=20260524)
    p.add_argument("--selection-eval-oracle-results", default=None)
    p.add_argument("--selection-eval-mode", default="none", choices=["none", "best", "every_eval", "final"])
    p.add_argument("--selection-eval-sample-limit", type=int, default=128)
    p.add_argument("--selection-eval-top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--selection-eval-max-candidate-chars", type=int, default=180)
    p.add_argument("--selection-eval-output-dir", default=None)
    p.add_argument("--best-selection-metric", default="jaccard@5", choices=["recall@5", "jaccard@5", "oracle_rank_ndcg@5", "top1_match"])
    p.add_argument("--primary-checkpoint", default="selection", choices=["action", "selection"])
    p.add_argument("--initial-eval", action="store_true")
    p.add_argument("--action-label-mode", default=None, choices=sorted(ACTION_LABEL_MODES))
    p.add_argument("--candidate-order-mode", default=CANDIDATE_ORDER_CANDIDATE_POOL, choices=sorted(CANDIDATE_ORDER_MODES))
    p.add_argument("--candidate-order-seed", type=int, default=20260524)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260523)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--use-flash-attention-2", action="store_true", default=True)
    p.add_argument("--no-flash-attention-2", dest="use_flash_attention_2", action="store_false")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default=",".join(DEFAULT_LORA_TARGET_MODULES))
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-swanlab", action="store_true")
    p.add_argument("--swanlab-project", default="fact-checking-llm-action-selector")
    p.add_argument("--swanlab-experiment-name", default=None)
    p.add_argument("--swanlab-workspace", default=None)
    p.add_argument("--swanlab-mode", default=None)
    p.add_argument("--swanlab-logdir", default=None)
    p.add_argument("--swanlab-tags", default="selector,llm_action,vig_soft")
    p.add_argument("--swanlab-description", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    started_at = time.time()
    _enable_tf32_if_available()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        mixed_precision="bf16" if bool(args.bf16) else "no",
    )
    _set_seed(int(args.seed) + accelerator.process_index)

    out_dir = Path(args.output_dir)
    metrics_dir = _metrics_dir(out_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    logger = _init_run_logger(out_dir, enabled=accelerator.is_main_process)

    train_rows = _mark_sample_rows(read_jsonl(args.train_data), sample_type="oracle_prefix")
    val_rows = _mark_sample_rows(read_jsonl(args.val_data), sample_type="oracle_prefix")
    bad_prefix_train_rows = (
        _mark_sample_rows(read_jsonl(args.bad_prefix_train_data), sample_type="bad_prefix")
        if args.bad_prefix_train_data
        else []
    )
    bad_prefix_val_rows = (
        _mark_sample_rows(read_jsonl(args.bad_prefix_val_data), sample_type="bad_prefix")
        if args.bad_prefix_val_data
        else []
    )
    if args.eval_sample_limit is not None:
        val_rows = _sample_eval_rows(
            val_rows,
            limit=int(args.eval_sample_limit),
            mode=str(args.eval_sample_mode),
            seed=int(args.eval_sample_seed),
        )
        if bad_prefix_val_rows:
            bad_prefix_val_rows = _sample_eval_rows(
                bad_prefix_val_rows,
                limit=int(args.eval_sample_limit),
                mode=str(args.eval_sample_mode),
                seed=int(args.eval_sample_seed) + 1,
            )
    if bad_prefix_train_rows:
        train_rows = train_rows + bad_prefix_train_rows
    if not train_rows:
        raise ValueError("No train action samples.")
    if not val_rows:
        raise ValueError("No val action samples.")
    sample_label_mode = _sample_settings(train_rows).get("action_label_mode") or ACTION_LABEL_MODE_GLOBAL_INDEX
    args.action_label_mode = str(args.action_label_mode or sample_label_mode)
    if logger is not None:
        logger.info(
            "Loaded action samples: train=%d val=%d bad_train=%d bad_val=%d",
            len(train_rows),
            len(val_rows),
            len(bad_prefix_train_rows),
            len(bad_prefix_val_rows),
        )
        logger.info(
            "score_mode=%s max_length=%d choice_batch_size=%d hard_loss_weight=%.4f "
            "soft_loss_weight=%.4f set_loss_weight=%.4f pairwise_loss_weight=%.4f set_loss_type=%s",
            args.score_mode,
            args.max_length,
            args.choice_batch_size,
            float(args.hard_loss_weight),
            float(args.soft_loss_weight),
            float(args.set_loss_weight),
            float(args.pairwise_loss_weight),
            str(args.set_loss_type),
        )
        if args.eval_sample_limit is not None:
            logger.info(
                "eval sample: limit=%d mode=%s seed=%d",
                int(args.eval_sample_limit),
                str(args.eval_sample_mode),
                int(args.eval_sample_seed),
            )
        logger.info(
            "action_label_mode=%s selection_candidate_order_mode=%s candidate_order_seed=%d",
            args.action_label_mode,
            args.candidate_order_mode,
            int(args.candidate_order_seed),
        )
        logger.info("train_order_augmentation=%s", args.train_order_augmentation)

    selection_examples: list[Stage2OracleExample] | None = None
    if str(args.selection_eval_mode) != "none":
        if not args.selection_eval_oracle_results:
            raise ValueError("--selection-eval-oracle-results is required when --selection-eval-mode is not none.")
        if accelerator.is_main_process:
            selection_examples = load_stage2_oracle_examples(
                args.selection_eval_oracle_results,
                expected_fingerprint=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
                top_k=int(args.selection_eval_top_k),
                sample_limit=int(args.selection_eval_sample_limit),
            )
            if not selection_examples:
                raise ValueError("No examples available for training-time selection eval.")
            if logger is not None:
                logger.info(
                    "Loaded selection eval examples: n=%d mode=%s sample_limit=%s",
                    len(selection_examples),
                    args.selection_eval_mode,
                    args.selection_eval_sample_limit,
                )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and bool(args.bf16) else torch.float32,
    }
    if bool(args.use_flash_attention_2) and flash_attn2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    lora_modules = [item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()]
    model = apply_lora_if_enabled(
        model,
        {
            "lora": {
                "enabled": not bool(args.no_lora),
                "r": int(args.lora_r),
                "alpha": int(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "target_modules": lora_modules,
                "modules_to_save": None,
            }
        },
        gradient_checkpointing=bool(args.gradient_checkpointing),
        logger=None,
    )

    train_dataset = ActionSampleDataset(
        train_rows,
        order_augmentation=str(args.train_order_augmentation),
        action_label_mode=str(args.action_label_mode),
        candidate_order_seed=int(args.candidate_order_seed),
    )
    val_dataset = ActionSampleDataset(val_rows)
    bad_prefix_val_dataset = ActionSampleDataset(bad_prefix_val_rows) if bad_prefix_val_rows else None
    train_dl = DataLoader(
        train_dataset,
        batch_size=int(args.per_device_train_batch_size),
        shuffle=True,
        collate_fn=lambda items: list(items),
        num_workers=int(args.dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=int(args.per_device_eval_batch_size),
        shuffle=False,
        collate_fn=lambda items: list(items),
        num_workers=int(args.dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    bad_prefix_val_dl = (
        DataLoader(
            bad_prefix_val_dataset,
            batch_size=int(args.per_device_eval_batch_size),
            shuffle=False,
            collate_fn=lambda items: list(items),
            num_workers=int(args.dataloader_num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        if bad_prefix_val_dataset is not None
        else None
    )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Check LoRA configuration.")
    optimizer = AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    epochs = int(math.ceil(float(args.num_train_epochs)))
    scheduler_steps_per_epoch = max(math.ceil(len(train_dl) / max(int(args.gradient_accumulation_steps), 1)), 1)
    scheduler_num_training_steps = max(scheduler_steps_per_epoch * epochs, 1)
    scheduler_warmup_steps = int(scheduler_num_training_steps * float(args.warmup_ratio))
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=scheduler_num_training_steps,
    )

    if bad_prefix_val_dl is not None:
        model, optimizer, train_dl, val_dl, bad_prefix_val_dl, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dl,
            val_dl,
            bad_prefix_val_dl,
            scheduler,
        )
    else:
        model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dl,
            val_dl,
            scheduler,
        )

    actual_steps_per_epoch = max(math.ceil(len(train_dl) / max(int(args.gradient_accumulation_steps), 1)), 1)
    runtime_max_train_steps = max(actual_steps_per_epoch * epochs, 1)
    runtime_info = {
        "actual_steps_per_epoch": int(actual_steps_per_epoch),
        "runtime_max_train_steps": int(runtime_max_train_steps),
        "scheduler_steps_per_epoch": int(scheduler_steps_per_epoch),
        "scheduler_num_training_steps": int(scheduler_num_training_steps),
        "scheduler_warmup_steps": int(scheduler_warmup_steps),
        "world_size": int(accelerator.num_processes),
        "effective_global_batch_size": int(
            int(args.per_device_train_batch_size)
            * max(int(accelerator.num_processes), 1)
            * max(int(args.gradient_accumulation_steps), 1)
        ),
    }
    if logger is not None:
        logger.info(
            "runtime steps: actual_steps_per_epoch=%d runtime_max_train_steps=%d "
            "scheduler_steps_per_epoch=%d scheduler_num_training_steps=%d scheduler_warmup_steps=%d world_size=%d "
            "effective_global_batch_size=%d",
            int(runtime_info["actual_steps_per_epoch"]),
            int(runtime_info["runtime_max_train_steps"]),
            int(runtime_info["scheduler_steps_per_epoch"]),
            int(runtime_info["scheduler_num_training_steps"]),
            int(runtime_info["scheduler_warmup_steps"]),
            int(runtime_info["world_size"]),
            int(runtime_info["effective_global_batch_size"]),
        )
        logger.info(
            "optimizer/scheduler: learning_rate=%.8g weight_decay=%.8g warmup_ratio=%.6g "
            "gradient_accumulation_steps=%d max_grad_norm=%.6g",
            float(args.learning_rate),
            float(args.weight_decay),
            float(args.warmup_ratio),
            int(args.gradient_accumulation_steps),
            float(args.max_grad_norm),
        )

    metadata = _metadata(
        args,
        n_train=len(train_rows),
        n_val=len(val_rows),
        runtime_info=runtime_info,
        train_rows=train_rows,
        val_rows=val_rows,
        bad_prefix_train_rows=bad_prefix_train_rows,
        bad_prefix_val_rows=bad_prefix_val_rows,
    )
    lr_runtime = _init_lr_runtime(args, initial_lr=_current_lr(optimizer), runtime_info=runtime_info)
    _update_metadata_lr_runtime(metadata, lr_runtime)
    if logger is not None:
        logger.info("initial optimizer lr after scheduler prepare=%.8g", float(lr_runtime["initial_lr"]))
    swanlab_run = _init_swanlab(args, out_dir, is_main=accelerator.is_main_process, config=metadata)
    if accelerator.is_main_process:
        write_json(out_dir / "selector_metadata.json", metadata)
        train_history_path = out_dir / "train_history.jsonl"
        val_history_path = out_dir / "val_history.jsonl"
        selection_history_path = out_dir / "selection_history.jsonl"
        _reset_jsonl(train_history_path)
        _reset_jsonl(val_history_path)
        _reset_jsonl(selection_history_path)
        _log_swanlab(swanlab_run, _metadata_metrics(metadata), step=0)
    else:
        train_history_path = out_dir / "train_history.jsonl"
        val_history_path = out_dir / "val_history.jsonl"
        selection_history_path = out_dir / "selection_history.jsonl"

    history: list[dict[str, Any]] = []
    best_action_accuracy = float("-inf")
    best_selection_metric = float("-inf")
    global_step = 0
    last_eval_step: int | None = None
    log_every = max(int(args.logging_steps), 1)
    progress = tqdm(total=runtime_max_train_steps, disable=not accelerator.is_local_main_process or bool(args.no_progress))

    if bool(args.initial_eval):
        result = _evaluate(
            model,
            tokenizer,
            val_dl,
            bad_prefix_val_dl=bad_prefix_val_dl,
            accelerator=accelerator,
            args=args,
            global_step=0,
            epoch=0,
        )
        selection_record = _run_training_selection_eval(
            model,
            tokenizer,
            selection_examples,
            accelerator=accelerator,
            args=args,
            output_dir=out_dir,
            selection_history_path=selection_history_path,
            global_step=0,
            epoch=0,
            reason="initial",
            logger=logger,
        )
        if accelerator.is_main_process:
            if selection_record is not None:
                result["selection"] = selection_record
            history.append(result)
            _append_jsonl(val_history_path, result)
            write_json(metrics_dir / "initial_val.json", result)
            write_json(metrics_dir / "latest_val.json", result)
            _log_swanlab(swanlab_run, _val_swanlab_payload(result), step=0)
            if logger is not None:
                logger.info(
                    "initial eval loss=%.4f set=%.4f pair=%.4f acc=%.4f bad_hit1=%.4f",
                    float(result["val_loss"]),
                    float(result.get("val_set_loss") or 0.0),
                    float(result.get("val_pairwise_loss") or 0.0),
                    float(result["val_action_accuracy"]),
                    float(result.get("bad_prefix_remaining_oracle_hit@1") or 0.0),
                )
        last_eval_step = 0

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        for batch in train_dl:
            with accelerator.accumulate(model):
                loss, parts = _batch_loss(
                    model,
                    tokenizer,
                    batch,
                    device=accelerator.device,
                    max_length=int(args.max_length),
                    choice_batch_size=int(args.choice_batch_size),
                    score_mode=str(args.score_mode),
                    soft_tau=float(args.soft_tau),
                    hard_loss_weight=float(args.hard_loss_weight),
                    soft_loss_weight=float(args.soft_loss_weight),
                    set_loss_weight=float(args.set_loss_weight),
                    set_loss_type=str(args.set_loss_type),
                    pairwise_loss_weight=float(args.pairwise_loss_weight),
                    bad_prefix_hard_loss_weight=float(args.bad_prefix_hard_loss_weight),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                current_lr = _current_lr(optimizer)
                _update_lr_runtime(lr_runtime, current_lr, global_step=global_step)
                _update_metadata_lr_runtime(metadata, lr_runtime)
                progress.update(1)
                if global_step % log_every == 0:
                    train_record = _aggregate_train_parts(
                        parts,
                        accelerator=accelerator,
                        global_step=global_step,
                        epoch=epoch,
                        lr=current_lr,
                    )
                    if accelerator.is_main_process:
                        _append_jsonl(train_history_path, train_record)
                        write_json(metrics_dir / "latest_train.json", train_record)
                        _log_swanlab(swanlab_run, _train_swanlab_payload(train_record), step=global_step)
                        message = (
                            "step={step} loss={loss:.4f} hard={hard:.4f} soft={soft:.4f} "
                            "set={set_loss:.4f} pair={pairwise:.4f} acc={acc:.4f} hit1={hit1:.4f}".format(
                                step=global_step,
                                loss=float(train_record["loss"]),
                                hard=float(train_record["hard_loss"]),
                                soft=float(train_record["soft_loss"]),
                                set_loss=float(train_record["set_loss"]),
                                pairwise=float(train_record["pairwise_loss"]),
                                acc=float(train_record["accuracy"]),
                                hit1=float(train_record["remaining_oracle_hit@1"]),
                            )
                        )
                        print(message)
                        if logger is not None:
                            logger.info(message)
                if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
                    result = _evaluate(
                        model,
                        tokenizer,
                        val_dl,
                        bad_prefix_val_dl=bad_prefix_val_dl,
                        accelerator=accelerator,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                    )
                    history.append(result)
                    best_action_accuracy, best_selection_metric = _process_action_eval_result(
                        result,
                        best_action_accuracy=best_action_accuracy,
                        best_selection_metric=best_selection_metric,
                        accelerator=accelerator,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=out_dir,
                        metadata=metadata,
                        args=args,
                        val_history_path=val_history_path,
                        selection_history_path=selection_history_path,
                        selection_examples=selection_examples,
                        swanlab_run=swanlab_run,
                        logger=logger,
                    )
                    last_eval_step = int(global_step)

        if last_eval_step != int(global_step):
            result = _evaluate(
                model,
                tokenizer,
                val_dl,
                bad_prefix_val_dl=bad_prefix_val_dl,
                accelerator=accelerator,
                args=args,
                global_step=global_step,
                epoch=epoch,
            )
            history.append(result)
            best_action_accuracy, best_selection_metric = _process_action_eval_result(
                result,
                best_action_accuracy=best_action_accuracy,
                best_selection_metric=best_selection_metric,
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=out_dir,
                metadata=metadata,
                args=args,
                val_history_path=val_history_path,
                selection_history_path=selection_history_path,
                selection_examples=selection_examples,
                swanlab_run=swanlab_run,
                logger=logger,
            )
            last_eval_step = int(global_step)
        elif accelerator.is_main_process and logger is not None:
            logger.info("skip duplicate epoch-end eval step=%d", int(global_step))

    progress.close()
    _update_metadata_lr_runtime(metadata, lr_runtime)
    if str(args.selection_eval_mode) == "final":
        final_selection = _run_training_selection_eval(
            model,
            tokenizer,
            selection_examples,
            accelerator=accelerator,
            args=args,
            output_dir=out_dir,
            selection_history_path=selection_history_path,
            global_step=global_step,
            epoch=epochs - 1,
            reason="final",
            logger=logger,
        )
        final_selection_metric = _broadcast_selection_metric(
            final_selection,
            metric_name=str(args.best_selection_metric),
            accelerator=accelerator,
        )
        final_selection_updated = final_selection_metric > best_selection_metric
        final_selection_checkpoint_dir = _checkpoint_dir(out_dir, "best_selection")
        if final_selection_updated:
            from sft.data.io import save_model

            save_model(accelerator, model, tokenizer, final_selection_checkpoint_dir)
            best_selection_metric = final_selection_metric
        if accelerator.is_main_process and final_selection is not None:
            _log_swanlab(swanlab_run, _selection_swanlab_payload(final_selection), step=global_step)
            if final_selection_updated:
                final_result = {
                    "global_step": int(global_step),
                    "epoch": int(epochs - 1),
                    "selection": final_selection,
                }
                _write_checkpoint_metadata(
                    output_dir=out_dir,
                    checkpoint_dir=final_selection_checkpoint_dir,
                    metadata=metadata,
                    result=final_result,
                    role="best_selection",
                    metric_name=str(args.best_selection_metric),
                    metric_value=best_selection_metric,
                    make_primary=str(args.primary_checkpoint) == "selection",
                )
                _log_swanlab(
                    swanlab_run,
                    {f"best/selection/{str(args.best_selection_metric)}": best_selection_metric},
                    step=global_step,
                )
    if accelerator.is_main_process:
        payload = {
            "metadata": metadata,
            "history": history,
            "best_action_accuracy": best_action_accuracy,
            "best_selection_metric": best_selection_metric,
            "best_selection_metric_name": str(args.best_selection_metric),
            "lr_runtime": lr_runtime,
            "runtime": runtime_info,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        write_json(out_dir / "selector_metadata.json", metadata)
        _sync_checkpoint_metadata(output_dir=out_dir, metadata=metadata)
        write_json(out_dir / "training_metrics.json", payload)
        _log_swanlab(
            swanlab_run,
            {
                "best/action_accuracy": best_action_accuracy,
                f"best/selection/{str(args.best_selection_metric)}": best_selection_metric,
                "optimizer/learning_rate": metadata.get("learning_rate"),
                "optimizer/lr_peak": lr_runtime.get("peak_lr"),
                "optimizer/lr_peak_step": lr_runtime.get("peak_step"),
                "optimizer/lr_final": lr_runtime.get("final_lr"),
                **_cuda_memory_metrics(),
            },
            step=global_step,
        )
        print(f"Wrote training metrics: {out_dir / 'training_metrics.json'}")
        if logger is not None:
            logger.info(
                "lr runtime: requested=%.8g initial=%.8g peak=%.8g peak_step=%s final=%.8g "
                "scheduler_warmup_steps=%s scheduler_num_training_steps=%s",
                float(lr_runtime.get("requested_lr") or 0.0),
                float(lr_runtime.get("initial_lr") or 0.0),
                float(lr_runtime.get("peak_lr") or 0.0),
                str(lr_runtime.get("peak_step")),
                float(lr_runtime.get("final_lr") or 0.0),
                str(lr_runtime.get("scheduler_warmup_steps")),
                str(lr_runtime.get("scheduler_num_training_steps")),
            )
            logger.info("Wrote training metrics: %s", out_dir / "training_metrics.json")
        _finish_swanlab(swanlab_run)


def _batch_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    choice_batch_size: int,
    score_mode: str,
    soft_tau: float,
    hard_loss_weight: float,
    soft_loss_weight: float,
    set_loss_weight: float,
    set_loss_type: str,
    pairwise_loss_weight: float,
    bad_prefix_hard_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    scored = score_action_choices(
        model,
        tokenizer,
        batch,
        device=device,
        max_length=int(max_length),
        choice_batch_size=int(choice_batch_size),
        score_mode=str(score_mode),
    )
    losses: list[torch.Tensor] = []
    hard_losses: list[torch.Tensor] = []
    soft_losses: list[torch.Tensor] = []
    set_losses: list[torch.Tensor] = []
    pairwise_losses: list[torch.Tensor] = []
    positive_probs: list[torch.Tensor] = []
    correct = 0
    positive_hit = 0
    count = 0
    hard_count = 0
    soft_count = 0
    positive_count = 0
    bad_prefix_count = 0
    for sample, scores, indices in zip(batch, scored.scores, scored.candidate_indices):
        if scores.numel() == 0:
            continue
        is_bad_prefix = _is_bad_prefix_sample(sample)
        bad_prefix_count += int(is_bad_prefix)
        positive_indices = _remaining_oracle_indices(sample, indices)
        positive_positions = [pos for pos, idx in enumerate(indices) if int(idx) in positive_indices]
        log_probs = torch.log_softmax(scores, dim=0)
        hard_loss = scores.new_zeros(())
        hard_target_weight = float(bad_prefix_hard_loss_weight) if is_bad_prefix else float(hard_loss_weight)
        target_idx = _target_idx_for_loss(sample, positive_indices)
        has_hard_target = bool(sample.get("has_hard_target", True)) or hard_target_weight > 0.0
        if has_hard_target and target_idx is not None:
            try:
                target_pos = indices.index(int(target_idx))
            except ValueError as exc:
                raise ValueError(
                    f"target_idx={target_idx} not found in remaining indices for {sample.get('event_id')}"
                ) from exc
            hard_loss = -log_probs[target_pos]
            hard_losses.append(hard_loss.detach())
            correct += int(torch.argmax(scores.detach()).item() == target_pos)
            hard_count += 1
        soft_loss = scores.new_zeros(())
        soft_target_weight = 0.0 if is_bad_prefix else float(soft_loss_weight)
        if not is_bad_prefix:
            deltas = [float(choice.get("delta_margin", 0.0)) for choice in sample.get("choices") or []]
            soft_probs = torch.tensor(
                softmax_deltas(deltas, tau=float(soft_tau)),
                dtype=scores.dtype,
                device=scores.device,
            )
            soft_loss = -(soft_probs * log_probs).sum()
            soft_losses.append(soft_loss.detach())
        set_loss = _set_aware_loss(
            sample,
            scores,
            log_probs,
            indices=indices,
            set_loss_type=str(set_loss_type),
        )
        pairwise_loss = _pairwise_positive_loss(scores, positive_positions=positive_positions)
        losses.append(
            float(hard_target_weight) * hard_loss
            + soft_target_weight * soft_loss
            + float(set_loss_weight) * set_loss
            + float(pairwise_loss_weight) * pairwise_loss
        )
        set_losses.append(set_loss.detach())
        pairwise_losses.append(pairwise_loss.detach())
        if positive_positions:
            positions = torch.tensor(positive_positions, dtype=torch.long, device=scores.device)
            probs = torch.softmax(scores, dim=0).index_select(0, positions)
            positive_probs.append(probs.sum().detach())
            positive_hit += int(torch.argmax(scores.detach()).item() in positive_positions)
            positive_count += 1
        if not is_bad_prefix:
            soft_count += 1
        count += 1
    if not losses:
        raise ValueError("Batch produced no valid action choices.")
    loss = torch.stack(losses).mean()
    parts = {
        "loss": float(loss.detach().float().item()),
        "hard_loss": _mean_detached(hard_losses),
        "soft_loss": _mean_detached(soft_losses),
        "set_loss": float(torch.stack(set_losses).mean().float().item()),
        "pairwise_loss": float(torch.stack(pairwise_losses).mean().float().item()),
        "accuracy": float(correct / max(hard_count, 1)),
        "remaining_oracle_hit@1": float(positive_hit / max(positive_count, 1)),
        "positive_prob": _mean_detached(positive_probs),
        "n_samples": float(count),
        "n_hard_samples": float(hard_count),
        "n_soft_samples": float(soft_count),
        "n_positive_samples": float(positive_count),
        "n_bad_prefix_samples": float(bad_prefix_count),
    }
    return loss, parts


def _set_aware_loss(
    sample: dict[str, Any],
    scores: torch.Tensor,
    log_probs: torch.Tensor,
    *,
    indices: list[int],
    set_loss_type: str,
) -> torch.Tensor:
    positive_indices = _remaining_oracle_indices(sample, indices)
    positive_positions = [pos for pos, idx in enumerate(indices) if int(idx) in positive_indices]
    if not positive_positions:
        return scores.new_zeros(())
    if set_loss_type == "multi_positive_ce":
        target = torch.zeros_like(scores)
        target[positive_positions] = 1.0 / float(len(positive_positions))
        return -(target * log_probs).sum()
    if set_loss_type == "bce":
        target = torch.zeros_like(scores)
        target[positive_positions] = 1.0
        centered_scores = scores - scores.mean()
        return F.binary_cross_entropy_with_logits(centered_scores, target)
    raise ValueError(f"Unsupported set_loss_type={set_loss_type!r}")


def _pairwise_positive_loss(
    scores: torch.Tensor,
    *,
    positive_positions: list[int],
) -> torch.Tensor:
    positive = [int(pos) for pos in positive_positions]
    if not positive:
        return scores.new_zeros(())
    positive_set = set(positive)
    negative = [pos for pos in range(scores.numel()) if pos not in positive_set]
    if not negative:
        return scores.new_zeros(())
    pos_scores = scores.index_select(0, torch.tensor(positive, dtype=torch.long, device=scores.device))
    neg_scores = scores.index_select(0, torch.tensor(negative, dtype=torch.long, device=scores.device))
    return F.softplus(-(pos_scores[:, None] - neg_scores[None, :])).mean()


def _mean_detached(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).mean().float().item())


def _is_bad_prefix_sample(sample: dict[str, Any]) -> bool:
    if str(sample.get("sample_type") or "") == "bad_prefix":
        return True
    if sample.get("has_hard_target") is False:
        return True
    return str(sample.get("prefix_source") or "oracle") not in {"", "oracle"}


def _target_idx_for_loss(sample: dict[str, Any], positive_indices: set[int]) -> int | None:
    raw = sample.get("target_idx")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    if positive_indices:
        return int(sorted(positive_indices)[0])
    return None


def _remaining_oracle_indices(sample: dict[str, Any], indices: list[int]) -> set[int]:
    valid = {int(idx) for idx in indices}
    if sample.get("remaining_oracle_indices") is not None:
        return {int(idx) for idx in sample.get("remaining_oracle_indices") or [] if int(idx) in valid}
    oracle_indices = sample.get("oracle_selected_indices")
    if oracle_indices is not None:
        prefix = {int(idx) for idx in sample.get("prefix_indices") or []}
        return {int(idx) for idx in oracle_indices if int(idx) not in prefix and int(idx) in valid}
    target = sample.get("target_idx")
    if target is None:
        return set()
    return {int(target)}


def _evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    val_dl: DataLoader,
    *,
    bad_prefix_val_dl: DataLoader | None = None,
    accelerator: Accelerator,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    main = _evaluate_loader(
        model,
        tokenizer,
        val_dl,
        accelerator=accelerator,
        args=args,
        desc="action val",
    )
    bad_prefix = (
        _evaluate_loader(
            model,
            tokenizer,
            bad_prefix_val_dl,
            accelerator=accelerator,
            args=args,
            desc="bad-prefix val",
        )
        if bad_prefix_val_dl is not None
        else None
    )
    result = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "val_loss": float(main["loss"]),
        "val_hard_loss": float(main["hard_loss"]),
        "val_soft_loss": float(main["soft_loss"]),
        "val_set_loss": float(main["set_loss"]),
        "val_pairwise_loss": float(main["pairwise_loss"]),
        "val_action_accuracy": float(main["accuracy"]),
        "val_remaining_oracle_hit@1": float(main["remaining_oracle_hit@1"]),
        "val_positive_prob": float(main["positive_prob"]),
        "n_val_samples": int(main["n_samples"]),
        "n_val_hard_samples": int(main["n_hard_samples"]),
    }
    if bad_prefix is not None:
        result.update(
            {
                "bad_prefix_val_loss": float(bad_prefix["loss"]),
                "bad_prefix_val_hard_loss": float(bad_prefix["hard_loss"]),
                "bad_prefix_val_set_loss": float(bad_prefix["set_loss"]),
                "bad_prefix_val_pairwise_loss": float(bad_prefix["pairwise_loss"]),
                "bad_prefix_remaining_oracle_hit@1": float(bad_prefix["remaining_oracle_hit@1"]),
                "bad_prefix_positive_prob": float(bad_prefix["positive_prob"]),
                "n_bad_prefix_val_samples": int(bad_prefix["n_samples"]),
            }
        )
    if accelerator.is_main_process:
        print(
            "eval step={global_step} loss={val_loss:.4f} set={val_set_loss:.4f} "
            "pair={val_pairwise_loss:.4f} acc={val_action_accuracy:.4f}".format(**result)
        )
    model.train()
    return result


def _evaluate_loader(
    model: torch.nn.Module,
    tokenizer: Any,
    data_loader: DataLoader,
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, float]:
    sums = torch.zeros((13,), dtype=torch.float64, device=accelerator.device)
    for batch in tqdm(
        data_loader,
        desc=desc,
        disable=not accelerator.is_local_main_process or bool(args.no_progress),
        leave=False,
    ):
        with torch.no_grad():
            _loss, parts = _batch_loss(
                model,
                tokenizer,
                batch,
                device=accelerator.device,
                max_length=int(args.max_length),
                choice_batch_size=int(args.choice_batch_size),
                score_mode=str(args.score_mode),
                soft_tau=float(args.soft_tau),
                hard_loss_weight=float(args.hard_loss_weight),
                soft_loss_weight=float(args.soft_loss_weight),
                set_loss_weight=float(args.set_loss_weight),
                set_loss_type=str(args.set_loss_type),
                pairwise_loss_weight=float(args.pairwise_loss_weight),
                bad_prefix_hard_loss_weight=float(args.bad_prefix_hard_loss_weight),
            )
        sums += _parts_to_sums(parts, device=accelerator.device)
    gathered = accelerator.gather_for_metrics(sums.unsqueeze(0)).sum(dim=0)
    return _summarize_sums(gathered)


def _maybe_save_action_best(
    result: dict[str, Any],
    *,
    best_action_accuracy: float,
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
) -> tuple[float, bool, Path]:
    current = float(result["val_action_accuracy"])
    checkpoint_dir = _checkpoint_dir(output_dir, "best_action")
    if current <= best_action_accuracy:
        return best_action_accuracy, False, checkpoint_dir
    from sft.data.io import save_model

    save_model(accelerator, model, tokenizer, checkpoint_dir)
    if accelerator.is_main_process:
        print(f"Saved best action-accuracy checkpoint: {checkpoint_dir}")
    return current, True, checkpoint_dir


def _process_action_eval_result(
    result: dict[str, Any],
    *,
    best_action_accuracy: float,
    best_selection_metric: float,
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    val_history_path: Path,
    selection_history_path: Path,
    selection_examples: list[Stage2OracleExample] | None,
    swanlab_run: SwanLabRun,
    logger: logging.Logger | None,
) -> tuple[float, float]:
    best_action_accuracy, action_updated, action_checkpoint_dir = _maybe_save_action_best(
        result,
        best_action_accuracy=best_action_accuracy,
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    selection_record: dict[str, Any] | None = None
    if _should_run_selection_eval(str(args.selection_eval_mode), best_updated=action_updated):
        selection_record = _run_training_selection_eval(
            model,
            tokenizer,
            selection_examples,
            accelerator=accelerator,
            args=args,
            output_dir=output_dir,
            selection_history_path=selection_history_path,
            global_step=int(result["global_step"]),
            epoch=int(result["epoch"]),
            reason="best_action" if action_updated else "every_eval",
            logger=logger,
        )
    selection_metric_value = _broadcast_selection_metric(
        selection_record,
        metric_name=str(args.best_selection_metric),
        accelerator=accelerator,
    )
    selection_updated = selection_metric_value > best_selection_metric
    selection_checkpoint_dir = _checkpoint_dir(output_dir, "best_selection")
    if selection_updated:
        from sft.data.io import save_model

        save_model(accelerator, model, tokenizer, selection_checkpoint_dir)
        best_selection_metric = selection_metric_value
        if accelerator.is_main_process:
            print(
                "Saved best selection checkpoint: "
                f"{selection_checkpoint_dir} ({args.best_selection_metric}={selection_metric_value:.4f})"
            )

    if accelerator.is_main_process:
        if selection_record is not None:
            result["selection"] = selection_record
        _append_jsonl(val_history_path, result)
        write_json(_metrics_dir(output_dir) / "latest_val.json", result)
        _log_swanlab(swanlab_run, _val_swanlab_payload(result), step=int(result["global_step"]))
        if logger is not None:
            logger.info(
                "eval step=%d loss=%.4f set=%.4f pair=%.4f acc=%.4f bad_hit1=%.4f",
                int(result["global_step"]),
                float(result["val_loss"]),
                float(result.get("val_set_loss") or 0.0),
                float(result.get("val_pairwise_loss") or 0.0),
                float(result["val_action_accuracy"]),
                float(result.get("bad_prefix_remaining_oracle_hit@1") or 0.0),
            )
        if action_updated:
            _write_checkpoint_metadata(
                output_dir=output_dir,
                checkpoint_dir=action_checkpoint_dir,
                metadata=metadata,
                result=result,
                role="best_action",
                metric_name="val_action_accuracy",
                metric_value=best_action_accuracy,
                make_primary=str(args.primary_checkpoint) == "action" or metadata.get("best_checkpoint_dir") is None,
            )
            _log_swanlab(swanlab_run, {"best/action_accuracy": best_action_accuracy}, step=int(result["global_step"]))
        if selection_updated and selection_record is not None:
            _write_checkpoint_metadata(
                output_dir=output_dir,
                checkpoint_dir=selection_checkpoint_dir,
                metadata=metadata,
                result=result,
                role="best_selection",
                metric_name=str(args.best_selection_metric),
                metric_value=best_selection_metric,
                make_primary=str(args.primary_checkpoint) == "selection",
            )
            _log_swanlab(
                swanlab_run,
                {f"best/selection/{str(args.best_selection_metric)}": best_selection_metric},
                step=int(result["global_step"]),
            )
    return best_action_accuracy, best_selection_metric


def _run_training_selection_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Stage2OracleExample] | None,
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
    output_dir: Path,
    selection_history_path: Path,
    global_step: int,
    epoch: int,
    reason: str,
    logger: logging.Logger | None,
) -> dict[str, Any] | None:
    if str(args.selection_eval_mode) == "none":
        return None
    accelerator.wait_for_everyone()
    record: dict[str, Any] | None = None
    if accelerator.is_main_process:
        if examples is None:
            raise ValueError("Selection eval examples were not loaded on the main process.")
        step_dir = Path(args.selection_eval_output_dir or (output_dir / "evals" / "during_train")) / f"step_{int(global_step):06d}"
        eval_model = accelerator.unwrap_model(model)
        was_training = bool(eval_model.training)
        eval_model.eval()
        if logger is not None:
            logger.info(
                "selection eval start reason=%s step=%d n_claims=%d",
                reason,
                int(global_step),
                len(examples),
            )
        result = evaluate_llm_action_selection(
            eval_model,
            tokenizer,
            examples,
            device=accelerator.device,
            split="val",
            top_k=int(args.selection_eval_top_k),
            max_length=int(args.max_length),
            score_mode=str(args.score_mode),
            choice_batch_size=int(args.choice_batch_size),
            max_candidate_chars=int(args.selection_eval_max_candidate_chars),
            include_retrieval_scores=True,
            action_label_mode=str(args.action_label_mode),
            candidate_order_mode=str(args.candidate_order_mode),
            candidate_order_seed=int(args.candidate_order_seed),
            disable_progress=bool(args.no_progress),
        )
        metrics = {
            **result["metrics"],
            "global_step": int(global_step),
            "epoch": int(epoch),
            "reason": str(reason),
            "oracle_results": str(args.selection_eval_oracle_results),
            "eval_output_dir": str(step_dir),
        }
        result["metrics"] = metrics
        write_selection_eval_outputs(step_dir, result)
        record = selection_history_record(
            metrics,
            global_step=int(global_step),
            epoch=int(epoch),
            output_dir=str(step_dir),
            reason=str(reason),
        )
        record["selection_metrics_path"] = str(step_dir / "selection_metrics.json")
        _append_jsonl(selection_history_path, record)
        write_json(_metrics_dir(output_dir) / "latest_selection_val.json", record)
        if logger is not None:
            logger.info(
                "selection eval done reason=%s step=%d recall@5=%.4f jaccard@5=%.4f ndcg@5=%.4f elapsed=%.3fs",
                reason,
                int(global_step),
                float(record.get("recall@5") or 0.0),
                float(record.get("jaccard@5") or 0.0),
                float(record.get("oracle_rank_ndcg@5") or 0.0),
                float(record.get("elapsed_seconds") or 0.0),
            )
        if was_training:
            eval_model.train()
    accelerator.wait_for_everyone()
    model.train()
    return record


def _should_run_selection_eval(mode: str, *, best_updated: bool) -> bool:
    if mode == "every_eval":
        return True
    if mode == "best" and bool(best_updated):
        return True
    return False


def _metrics_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "metrics"


def _checkpoint_dir(output_dir: Path, role: str) -> Path:
    if role not in {"best_action", "best_selection"}:
        raise ValueError(f"Unsupported checkpoint role: {role!r}")
    return Path(output_dir) / "checkpoints" / role


def _broadcast_selection_metric(
    record: dict[str, Any] | None,
    *,
    metric_name: str,
    accelerator: Accelerator,
) -> float:
    value = float("-inf")
    if accelerator.is_main_process and record is not None:
        raw_value = record.get(metric_name)
        if raw_value is not None:
            value = float(raw_value)
    tensor = torch.tensor([value], dtype=torch.float64, device=accelerator.device)
    if (
        int(accelerator.num_processes) > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.broadcast(tensor, src=0)
    return float(tensor.item())


def _write_checkpoint_metadata(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    result: dict[str, Any],
    role: str,
    metric_name: str,
    metric_value: float,
    make_primary: bool,
) -> None:
    rel = checkpoint_dir.relative_to(output_dir).as_posix()
    best_key = "best_action" if role == "best_action" else "best_selection"
    metadata[f"{role}_checkpoint_dir"] = rel
    metadata[best_key] = {
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "checkpoint_dir": rel,
        "result": result,
    }
    if make_primary:
        metadata["best_checkpoint_dir"] = rel
        metadata["best_checkpoint_role"] = str(role)
        metadata["best"] = result
    write_json(output_dir / "selector_metadata.json", metadata)
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update(
        {
            "checkpoint_role": str(role),
            "checkpoint_metric_name": str(metric_name),
            "checkpoint_metric_value": float(metric_value),
            "run_output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
        }
    )
    write_json(checkpoint_dir / "selector_metadata.json", checkpoint_metadata)
    metrics_dir = _metrics_dir(output_dir)
    write_json(metrics_dir / f"{role}_val.json", result)
    if role == "best_action":
        write_json(metrics_dir / "best_val.json", result)
    if make_primary:
        write_json(metrics_dir / "best_primary_val.json", result)


def _sync_checkpoint_metadata(*, output_dir: Path, metadata: dict[str, Any]) -> None:
    for role, best_key in (("best_action", "best_action"), ("best_selection", "best_selection")):
        rel = metadata.get(f"{role}_checkpoint_dir")
        if not rel:
            continue
        checkpoint_dir = output_dir / str(rel)
        metadata_path = checkpoint_dir / "selector_metadata.json"
        if not metadata_path.exists():
            continue
        best_record = metadata.get(best_key) or {}
        checkpoint_metadata = dict(metadata)
        checkpoint_metadata.update(
            {
                "checkpoint_role": role,
                "checkpoint_metric_name": best_record.get("metric_name"),
                "checkpoint_metric_value": best_record.get("metric_value"),
                "run_output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
        write_json(metadata_path, checkpoint_metadata)


def _metadata(
    args: argparse.Namespace,
    *,
    n_train: int,
    n_val: int,
    runtime_info: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    bad_prefix_train_rows: list[dict[str, Any]],
    bad_prefix_val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_sample_settings = _sample_settings(train_rows)
    val_sample_settings = _sample_settings(val_rows)
    return {
        "selector_type": "llm_action_selector",
        "base_model_name_or_path": str(args.model_name),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "bad_prefix_train_data": str(args.bad_prefix_train_data) if args.bad_prefix_train_data else None,
        "bad_prefix_val_data": str(args.bad_prefix_val_data) if args.bad_prefix_val_data else None,
        "max_length": int(args.max_length),
        "choice_batch_size": int(args.choice_batch_size),
        "score_mode": str(args.score_mode),
        "num_train_epochs": float(args.num_train_epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "scheduler_type": "cosine",
        "scheduler_warmup_steps": int(runtime_info["scheduler_warmup_steps"]),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_grad_norm": float(args.max_grad_norm),
        "action_label_mode": str(args.action_label_mode),
        "candidate_order_mode": str(args.candidate_order_mode),
        "candidate_order_seed": int(args.candidate_order_seed),
        "train_order_augmentation": str(args.train_order_augmentation),
        "checkpoint_layout_version": 2,
        "best_checkpoint_dir": None,
        "best_checkpoint_role": None,
        "best_action_checkpoint_dir": "checkpoints/best_action",
        "best_selection_checkpoint_dir": "checkpoints/best_selection",
        "metrics_dir": "metrics",
        "initial_eval": bool(args.initial_eval),
        "eval_sample_limit": int(args.eval_sample_limit) if args.eval_sample_limit is not None else None,
        "eval_sample_mode": str(args.eval_sample_mode),
        "eval_sample_seed": int(args.eval_sample_seed),
        "selection_eval_mode": str(args.selection_eval_mode),
        "selection_eval_oracle_results": str(args.selection_eval_oracle_results) if args.selection_eval_oracle_results else None,
        "selection_eval_sample_limit": int(args.selection_eval_sample_limit),
        "selection_eval_top_k": int(args.selection_eval_top_k),
        "selection_eval_max_candidate_chars": int(args.selection_eval_max_candidate_chars),
        "selection_eval_output_dir": str(args.selection_eval_output_dir) if args.selection_eval_output_dir else None,
        "best_selection_metric": str(args.best_selection_metric),
        "primary_checkpoint": str(args.primary_checkpoint),
        "n_train_samples": int(n_train),
        "n_val_samples": int(n_val),
        "n_oracle_train_samples": int(n_train - len(bad_prefix_train_rows)),
        "n_bad_prefix_train_samples": int(len(bad_prefix_train_rows)),
        "n_bad_prefix_val_samples": int(len(bad_prefix_val_rows)),
        "max_train_steps": int(runtime_info["runtime_max_train_steps"]),
        "runtime_max_train_steps": int(runtime_info["runtime_max_train_steps"]),
        "actual_steps_per_epoch": int(runtime_info["actual_steps_per_epoch"]),
        "scheduler_num_training_steps": int(runtime_info["scheduler_num_training_steps"]),
        "scheduler_steps_per_epoch": int(runtime_info["scheduler_steps_per_epoch"]),
        "world_size": int(runtime_info["world_size"]),
        "effective_global_batch_size": int(runtime_info["effective_global_batch_size"]),
        "lr_runtime": {},
        "train_action_label_mode": train_sample_settings.get("action_label_mode"),
        "val_action_label_mode": val_sample_settings.get("action_label_mode"),
        "train_candidate_order_mode": train_sample_settings.get("candidate_order_mode"),
        "val_candidate_order_mode": val_sample_settings.get("candidate_order_mode"),
        "train_candidate_order_seed": train_sample_settings.get("candidate_order_seed"),
        "val_candidate_order_seed": val_sample_settings.get("candidate_order_seed"),
        "soft_loss_weight": float(args.soft_loss_weight),
        "soft_tau": float(args.soft_tau),
        "hard_loss_weight": float(args.hard_loss_weight),
        "set_loss_weight": float(args.set_loss_weight),
        "set_loss_type": str(args.set_loss_type),
        "pairwise_loss_weight": float(args.pairwise_loss_weight),
        "bad_prefix_hard_loss_weight": float(args.bad_prefix_hard_loss_weight),
        "action_format": "A..O",
        "lora": {
            "enabled": not bool(args.no_lora),
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": [item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()],
        },
    }


def _sample_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["action_label_mode", "candidate_order_mode", "candidate_order_seed"]
    settings: dict[str, Any] = {}
    for key in keys:
        values = []
        seen = set()
        for row in rows:
            if key not in row:
                continue
            value = row.get(key)
            marker = json.dumps(_json_safe_value(value), sort_keys=True, ensure_ascii=False)
            if marker not in seen:
                seen.add(marker)
                values.append(value)
            if len(values) > 1:
                break
        if not values:
            settings[key] = None
        elif len(values) == 1:
            settings[key] = values[0]
        else:
            settings[key] = "mixed"
    return settings


def _init_run_logger(out_dir: Path, *, enabled: bool) -> logging.Logger | None:
    if not enabled:
        return None
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"llm_action_selector.{Path(out_dir).resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(log_dir / "train_llm_action_selector.log", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _init_swanlab(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    is_main: bool,
    config: dict[str, Any],
) -> SwanLabRun:
    if not is_main or bool(args.no_swanlab):
        return SwanLabRun(enabled=False)
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab logging is enabled but `swanlab` is not installed. Install it or pass --no-swanlab."
        ) from exc

    experiment_name = str(args.swanlab_experiment_name or Path(args.output_dir).name)
    tags = [item.strip() for item in str(args.swanlab_tags or "").split(",") if item.strip()]
    init_config = {
        key: _json_safe_value(value)
        for key, value in vars(args).items()
        if key not in {"no_progress"}
    }
    init_config.update(_json_safe_value(config))
    init_config["output_dir"] = str(out_dir)

    init_kwargs: dict[str, Any] = {
        "project": str(args.swanlab_project),
        "experiment_name": experiment_name,
        "config": init_config,
    }
    if args.swanlab_workspace:
        init_kwargs["workspace"] = str(args.swanlab_workspace)
    if args.swanlab_mode:
        init_kwargs["mode"] = str(args.swanlab_mode)
    if args.swanlab_logdir:
        init_kwargs["logdir"] = str(args.swanlab_logdir)
    if tags:
        init_kwargs["tags"] = tags
    if args.swanlab_description:
        init_kwargs["description"] = str(args.swanlab_description)

    try:
        swanlab.init(**init_kwargs)
    except TypeError:
        minimal_kwargs = {
            "project": str(args.swanlab_project),
            "experiment_name": experiment_name,
            "config": init_config,
        }
        try:
            swanlab.init(**minimal_kwargs)
        except TypeError:
            swanlab.init(project=str(args.swanlab_project), experiment_name=experiment_name)
    return SwanLabRun(enabled=True, module=swanlab)


def _finish_swanlab(run: SwanLabRun) -> None:
    if not run.enabled or run.module is None:
        return
    finish = getattr(run.module, "finish", None)
    if callable(finish):
        finish()


def _log_swanlab(run: SwanLabRun, values: dict[str, Any], *, step: int) -> None:
    if not run.enabled or run.module is None:
        return
    payload = {
        key: scalar
        for key, value in values.items()
        for scalar in [_as_loggable_scalar(value)]
        if scalar is not None
    }
    if payload:
        run.module.log(payload, step=int(step))


def _aggregate_train_parts(
    parts: dict[str, float],
    *,
    accelerator: Any,
    global_step: int,
    epoch: int,
    lr: float,
) -> dict[str, Any]:
    sums = _parts_to_sums(parts, device=accelerator.device)
    gathered = accelerator.gather_for_metrics(sums.unsqueeze(0)).sum(dim=0)
    summary = _summarize_sums(gathered)
    return {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "loss": float(summary["loss"]),
        "hard_loss": float(summary["hard_loss"]),
        "soft_loss": float(summary["soft_loss"]),
        "set_loss": float(summary["set_loss"]),
        "pairwise_loss": float(summary["pairwise_loss"]),
        "accuracy": float(summary["accuracy"]),
        "remaining_oracle_hit@1": float(summary["remaining_oracle_hit@1"]),
        "positive_prob": float(summary["positive_prob"]),
        "n_samples": int(summary["n_samples"]),
        "n_hard_samples": int(summary["n_hard_samples"]),
        "n_bad_prefix_samples": int(summary["n_bad_prefix_samples"]),
        "lr": float(lr),
        **_cuda_memory_metrics(),
    }


def _parts_to_sums(parts: dict[str, float], *, device: torch.device) -> torch.Tensor:
    n = float(parts.get("n_samples") or 0.0)
    n_hard = float(parts.get("n_hard_samples") or 0.0)
    n_soft = float(parts.get("n_soft_samples") or 0.0)
    n_positive = float(parts.get("n_positive_samples") or 0.0)
    n_bad_prefix = float(parts.get("n_bad_prefix_samples") or 0.0)
    return torch.tensor(
        [
            float(parts.get("loss") or 0.0) * n,
            float(parts.get("hard_loss") or 0.0) * n_hard,
            float(parts.get("soft_loss") or 0.0) * n_soft,
            float(parts.get("set_loss") or 0.0) * n,
            float(parts.get("pairwise_loss") or 0.0) * n,
            float(parts.get("accuracy") or 0.0) * n_hard,
            float(parts.get("positive_prob") or 0.0) * n_positive,
            float(parts.get("remaining_oracle_hit@1") or 0.0) * n_positive,
            n,
            n_hard,
            n_soft,
            n_positive,
            n_bad_prefix,
        ],
        dtype=torch.float64,
        device=device,
    )


def _summarize_sums(sums: torch.Tensor) -> dict[str, float]:
    n = max(float(sums[8].item()), 1.0)
    n_hard = max(float(sums[9].item()), 1.0)
    n_soft = max(float(sums[10].item()), 1.0)
    n_positive = max(float(sums[11].item()), 1.0)
    return {
        "loss": float(sums[0].item() / n),
        "hard_loss": float(sums[1].item() / n_hard),
        "soft_loss": float(sums[2].item() / n_soft),
        "set_loss": float(sums[3].item() / n),
        "pairwise_loss": float(sums[4].item() / n),
        "accuracy": float(sums[5].item() / n_hard),
        "positive_prob": float(sums[6].item() / n_positive),
        "remaining_oracle_hit@1": float(sums[7].item() / n_positive),
        "n_samples": float(sums[8].item()),
        "n_hard_samples": float(sums[9].item()),
        "n_soft_samples": float(sums[10].item()),
        "n_positive_samples": float(sums[11].item()),
        "n_bad_prefix_samples": float(sums[12].item()),
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _init_lr_runtime(
    args: argparse.Namespace,
    *,
    initial_lr: float,
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requested_lr": float(args.learning_rate),
        "initial_lr": float(initial_lr),
        "first_step_lr": None,
        "first_step": None,
        "peak_lr": float(initial_lr),
        "peak_step": 0,
        "final_lr": float(initial_lr),
        "final_step": 0,
        "warmup_ratio": float(args.warmup_ratio),
        "scheduler_type": "cosine",
        "scheduler_warmup_steps": int(runtime_info["scheduler_warmup_steps"]),
        "scheduler_num_training_steps": int(runtime_info["scheduler_num_training_steps"]),
        "scheduler_steps_per_epoch": int(runtime_info["scheduler_steps_per_epoch"]),
        "runtime_max_train_steps": int(runtime_info["runtime_max_train_steps"]),
    }


def _update_lr_runtime(stats: dict[str, Any], lr: float, *, global_step: int) -> None:
    current = float(lr)
    if stats.get("first_step_lr") is None:
        stats["first_step_lr"] = current
        stats["first_step"] = int(global_step)
    if current > float(stats.get("peak_lr") or 0.0):
        stats["peak_lr"] = current
        stats["peak_step"] = int(global_step)
    stats["final_lr"] = current
    stats["final_step"] = int(global_step)


def _update_metadata_lr_runtime(metadata: dict[str, Any], stats: dict[str, Any]) -> None:
    metadata["lr_runtime"] = dict(stats)
    metadata["observed_initial_lr"] = stats.get("initial_lr")
    metadata["observed_first_step_lr"] = stats.get("first_step_lr")
    metadata["observed_peak_lr"] = stats.get("peak_lr")
    metadata["observed_peak_lr_step"] = stats.get("peak_step")
    metadata["observed_final_lr"] = stats.get("final_lr")
    metadata["observed_final_lr_step"] = stats.get("final_step")


def _reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_safe_value(row), ensure_ascii=False) + "\n")


def _train_swanlab_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "train/loss": record.get("loss"),
        "train/hard_loss": record.get("hard_loss"),
        "train/soft_loss": record.get("soft_loss"),
        "train/set_loss": record.get("set_loss"),
        "train/pairwise_loss": record.get("pairwise_loss"),
        "train/action_accuracy": record.get("accuracy"),
        "train/remaining_oracle_hit@1": record.get("remaining_oracle_hit@1"),
        "train/positive_prob": record.get("positive_prob"),
        "train/n_bad_prefix_samples": record.get("n_bad_prefix_samples"),
        "train/lr": record.get("lr"),
        "train/epoch": record.get("epoch"),
        "train/global_step": record.get("global_step"),
        "system/cuda_max_memory_allocated_gb": record.get("cuda_max_memory_allocated_gb"),
        "system/cuda_memory_reserved_gb": record.get("cuda_memory_reserved_gb"),
    }


def _val_swanlab_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "val/loss": result.get("val_loss"),
        "val/hard_loss": result.get("val_hard_loss"),
        "val/soft_loss": result.get("val_soft_loss"),
        "val/set_loss": result.get("val_set_loss"),
        "val/pairwise_loss": result.get("val_pairwise_loss"),
        "val/action_accuracy": result.get("val_action_accuracy"),
        "val/remaining_oracle_hit@1": result.get("val_remaining_oracle_hit@1"),
        "val/positive_prob": result.get("val_positive_prob"),
        "val/n_samples": result.get("n_val_samples"),
        "val/bad_prefix/loss": result.get("bad_prefix_val_loss"),
        "val/bad_prefix/set_loss": result.get("bad_prefix_val_set_loss"),
        "val/bad_prefix/pairwise_loss": result.get("bad_prefix_val_pairwise_loss"),
        "val/bad_prefix/remaining_oracle_hit@1": result.get("bad_prefix_remaining_oracle_hit@1"),
        "val/bad_prefix/positive_prob": result.get("bad_prefix_positive_prob"),
        "val/bad_prefix/n_samples": result.get("n_bad_prefix_val_samples"),
    }
    payload.update(_selection_swanlab_payload(result.get("selection") or {}))
    return payload


def _selection_swanlab_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "val/selection/recall@5": record.get("recall@5"),
        "val/selection/jaccard@5": record.get("jaccard@5"),
        "val/selection/ndcg@5": record.get("oracle_rank_ndcg@5"),
        "val/selection/top1_match": record.get("top1_match"),
        "val/selection/n_claims": record.get("n_claims"),
    }


def _metadata_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "config/max_length": metadata.get("max_length"),
        "config/choice_batch_size": metadata.get("choice_batch_size"),
        "config/learning_rate": metadata.get("learning_rate"),
        "config/weight_decay": metadata.get("weight_decay"),
        "config/warmup_ratio": metadata.get("warmup_ratio"),
        "config/scheduler_type": metadata.get("scheduler_type"),
        "config/scheduler_warmup_steps": metadata.get("scheduler_warmup_steps"),
        "config/gradient_accumulation_steps": metadata.get("gradient_accumulation_steps"),
        "config/per_device_train_batch_size": metadata.get("per_device_train_batch_size"),
        "config/train_order_augmentation": metadata.get("train_order_augmentation"),
        "config/hard_loss_weight": metadata.get("hard_loss_weight"),
        "config/set_loss_weight": metadata.get("set_loss_weight"),
        "config/set_loss_type": metadata.get("set_loss_type"),
        "config/pairwise_loss_weight": metadata.get("pairwise_loss_weight"),
        "config/bad_prefix_hard_loss_weight": metadata.get("bad_prefix_hard_loss_weight"),
        "config/n_bad_prefix_train_samples": metadata.get("n_bad_prefix_train_samples"),
        "config/n_bad_prefix_val_samples": metadata.get("n_bad_prefix_val_samples"),
        "config/eval_sample_mode": metadata.get("eval_sample_mode"),
        "config/selection_eval_sample_limit": metadata.get("selection_eval_sample_limit"),
        "config/selection_eval_top_k": metadata.get("selection_eval_top_k"),
        "config/best_selection_metric": metadata.get("best_selection_metric"),
        "config/n_train_samples": metadata.get("n_train_samples"),
        "config/n_val_samples": metadata.get("n_val_samples"),
        "config/runtime_max_train_steps": metadata.get("runtime_max_train_steps"),
        "config/scheduler_num_training_steps": metadata.get("scheduler_num_training_steps"),
        "config/world_size": metadata.get("world_size"),
        "config/effective_global_batch_size": metadata.get("effective_global_batch_size"),
    }


def _cuda_memory_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_max_memory_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "cuda_memory_reserved_gb": float(torch.cuda.memory_reserved() / (1024**3)),
    }


def _as_loggable_scalar(value: Any) -> float | int | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Number) and not isinstance(value, complex):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return None


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _enable_tf32_if_available() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    main()
