#!/usr/bin/env python3
"""Train a Qwen/LoRA sequential action selector with VIG soft supervision."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.selectors.llm_action import (
    SCORE_MODE_ACTION_TOKEN,
    SCORE_MODE_CONTINUATION,
    score_action_choices,
    softmax_deltas,
)
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json
from sft.runtime.adapters import DEFAULT_LORA_TARGET_MODULES, apply_lora_if_enabled
from sft.runtime.deps import flash_attn2_available


class ActionSampleDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


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
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-sample-limit", type=int, default=None)
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
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    logger = _init_run_logger(out_dir, enabled=accelerator.is_main_process)

    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    if args.eval_sample_limit is not None:
        val_rows = val_rows[: int(args.eval_sample_limit)]
    if not train_rows:
        raise ValueError("No train action samples.")
    if not val_rows:
        raise ValueError("No val action samples.")
    if logger is not None:
        logger.info("Loaded action samples: train=%d val=%d", len(train_rows), len(val_rows))
        logger.info("score_mode=%s max_length=%d choice_batch_size=%d", args.score_mode, args.max_length, args.choice_batch_size)

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

    train_dl = DataLoader(
        ActionSampleDataset(train_rows),
        batch_size=int(args.per_device_train_batch_size),
        shuffle=True,
        collate_fn=lambda items: list(items),
        num_workers=int(args.dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_dl = DataLoader(
        ActionSampleDataset(val_rows),
        batch_size=int(args.per_device_eval_batch_size),
        shuffle=False,
        collate_fn=lambda items: list(items),
        num_workers=int(args.dataloader_num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Check LoRA configuration.")
    optimizer = AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    epochs = int(math.ceil(float(args.num_train_epochs)))
    steps_per_epoch = max(math.ceil(len(train_dl) / max(int(args.gradient_accumulation_steps), 1)), 1)
    max_train_steps = max(steps_per_epoch * epochs, 1)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * float(args.warmup_ratio)),
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dl,
        val_dl,
        scheduler,
    )

    metadata = _metadata(args, n_train=len(train_rows), n_val=len(val_rows), max_train_steps=max_train_steps)
    swanlab_run = _init_swanlab(args, out_dir, is_main=accelerator.is_main_process, config=metadata)
    if accelerator.is_main_process:
        write_json(out_dir / "selector_metadata.json", metadata)
        train_history_path = out_dir / "train_history.jsonl"
        val_history_path = out_dir / "val_history.jsonl"
        _reset_jsonl(train_history_path)
        _reset_jsonl(val_history_path)
        _log_swanlab(swanlab_run, _metadata_metrics(metadata), step=0)
    else:
        train_history_path = out_dir / "train_history.jsonl"
        val_history_path = out_dir / "val_history.jsonl"

    best_accuracy = float("-inf")
    history: list[dict[str, Any]] = []
    global_step = 0
    log_every = max(int(args.logging_steps), 1)
    progress = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process or bool(args.no_progress))

    for epoch in range(epochs):
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
                    soft_loss_weight=float(args.soft_loss_weight),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % log_every == 0:
                    train_record = _aggregate_train_parts(
                        parts,
                        accelerator=accelerator,
                        global_step=global_step,
                        epoch=epoch,
                        lr=_current_lr(optimizer),
                    )
                    if accelerator.is_main_process:
                        _append_jsonl(train_history_path, train_record)
                        _log_swanlab(swanlab_run, _train_swanlab_payload(train_record), step=global_step)
                        message = (
                            "step={step} loss={loss:.4f} hard={hard:.4f} soft={soft:.4f} acc={acc:.4f}".format(
                                step=global_step,
                                loss=float(train_record["loss"]),
                                hard=float(train_record["hard_loss"]),
                                soft=float(train_record["soft_loss"]),
                                acc=float(train_record["accuracy"]),
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
                        accelerator=accelerator,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                    )
                    history.append(result)
                    if accelerator.is_main_process:
                        _append_jsonl(val_history_path, result)
                        _log_swanlab(swanlab_run, _val_swanlab_payload(result), step=global_step)
                        if logger is not None:
                            logger.info(
                                "eval step=%d loss=%.4f acc=%.4f",
                                global_step,
                                float(result["val_loss"]),
                                float(result["val_action_accuracy"]),
                            )
                    previous_best = best_accuracy
                    best_accuracy = _maybe_save_best(
                        result,
                        best_accuracy=best_accuracy,
                        accelerator=accelerator,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=out_dir,
                        metadata=metadata,
                    )
                    if accelerator.is_main_process and best_accuracy > previous_best:
                        _log_swanlab(swanlab_run, {"best/action_accuracy": best_accuracy}, step=global_step)

        result = _evaluate(
            model,
            tokenizer,
            val_dl,
            accelerator=accelerator,
            args=args,
            global_step=global_step,
            epoch=epoch,
        )
        history.append(result)
        if accelerator.is_main_process:
            _append_jsonl(val_history_path, result)
            _log_swanlab(swanlab_run, _val_swanlab_payload(result), step=global_step)
            if logger is not None:
                logger.info(
                    "eval step=%d loss=%.4f acc=%.4f",
                    global_step,
                    float(result["val_loss"]),
                    float(result["val_action_accuracy"]),
                )
        previous_best = best_accuracy
        best_accuracy = _maybe_save_best(
            result,
            best_accuracy=best_accuracy,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            output_dir=out_dir,
            metadata=metadata,
        )
        if accelerator.is_main_process and best_accuracy > previous_best:
            _log_swanlab(swanlab_run, {"best/action_accuracy": best_accuracy}, step=global_step)

    progress.close()
    if accelerator.is_main_process:
        payload = {
            "metadata": metadata,
            "history": history,
            "best_action_accuracy": best_accuracy,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        write_json(out_dir / "training_metrics.json", payload)
        _log_swanlab(
            swanlab_run,
            {
                "best/action_accuracy": best_accuracy,
                **_cuda_memory_metrics(),
            },
            step=global_step,
        )
        print(f"Wrote training metrics: {out_dir / 'training_metrics.json'}")
        if logger is not None:
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
    soft_loss_weight: float,
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
    correct = 0
    count = 0
    for sample, scores, indices in zip(batch, scored.scores, scored.candidate_indices):
        if scores.numel() == 0:
            continue
        target_idx = int(sample["target_idx"])
        try:
            target_pos = indices.index(target_idx)
        except ValueError as exc:
            raise ValueError(f"target_idx={target_idx} not found in remaining indices for {sample['event_id']}") from exc
        log_probs = torch.log_softmax(scores, dim=0)
        hard_loss = -log_probs[target_pos]
        deltas = [float(choice.get("delta_margin", 0.0)) for choice in sample.get("choices") or []]
        soft_probs = torch.tensor(
            softmax_deltas(deltas, tau=float(soft_tau)),
            dtype=scores.dtype,
            device=scores.device,
        )
        soft_loss = -(soft_probs * log_probs).sum()
        losses.append(hard_loss + float(soft_loss_weight) * soft_loss)
        hard_losses.append(hard_loss.detach())
        soft_losses.append(soft_loss.detach())
        correct += int(torch.argmax(scores.detach()).item() == target_pos)
        count += 1
    if not losses:
        raise ValueError("Batch produced no valid action choices.")
    loss = torch.stack(losses).mean()
    parts = {
        "loss": float(loss.detach().float().item()),
        "hard_loss": float(torch.stack(hard_losses).mean().float().item()),
        "soft_loss": float(torch.stack(soft_losses).mean().float().item()),
        "accuracy": float(correct / max(count, 1)),
        "n_samples": float(count),
    }
    return loss, parts


def _evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    val_dl: DataLoader,
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    sums = torch.zeros((5,), dtype=torch.float64, device=accelerator.device)
    for batch in tqdm(
        val_dl,
        desc="action val",
        disable=not accelerator.is_local_main_process or bool(args.no_progress),
        leave=False,
    ):
        with torch.no_grad():
            loss, parts = _batch_loss(
                model,
                tokenizer,
                batch,
                device=accelerator.device,
                max_length=int(args.max_length),
                choice_batch_size=int(args.choice_batch_size),
                score_mode=str(args.score_mode),
                soft_tau=float(args.soft_tau),
                soft_loss_weight=float(args.soft_loss_weight),
            )
        n = float(parts["n_samples"])
        sums += torch.tensor(
            [
                float(loss.detach().item()) * n,
                float(parts["hard_loss"]) * n,
                float(parts["soft_loss"]) * n,
                float(parts["accuracy"]) * n,
                n,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
    gathered = accelerator.gather_for_metrics(sums.unsqueeze(0)).sum(dim=0)
    n_total = max(float(gathered[4].item()), 1.0)
    result = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "val_loss": float(gathered[0].item() / n_total),
        "val_hard_loss": float(gathered[1].item() / n_total),
        "val_soft_loss": float(gathered[2].item() / n_total),
        "val_action_accuracy": float(gathered[3].item() / n_total),
        "n_val_samples": int(n_total),
    }
    if accelerator.is_main_process:
        print(
            "eval step={global_step} loss={val_loss:.4f} acc={val_action_accuracy:.4f}".format(
                **result
            )
        )
    model.train()
    return result


def _maybe_save_best(
    result: dict[str, Any],
    *,
    best_accuracy: float,
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    metadata: dict[str, Any],
) -> float:
    current = float(result["val_action_accuracy"])
    if current <= best_accuracy:
        return best_accuracy
    from sft.data.io import save_model

    save_model(accelerator, model, tokenizer, output_dir)
    if accelerator.is_main_process:
        metadata = dict(metadata)
        metadata["best"] = result
        write_json(output_dir / "selector_metadata.json", metadata)
        print(f"Saved best action selector checkpoint: {output_dir}")
    return current


def _metadata(
    args: argparse.Namespace,
    *,
    n_train: int,
    n_val: int,
    max_train_steps: int,
) -> dict[str, Any]:
    return {
        "selector_type": "llm_action_selector",
        "base_model_name_or_path": str(args.model_name),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "max_length": int(args.max_length),
        "choice_batch_size": int(args.choice_batch_size),
        "score_mode": str(args.score_mode),
        "n_train_samples": int(n_train),
        "n_val_samples": int(n_val),
        "max_train_steps": int(max_train_steps),
        "soft_loss_weight": float(args.soft_loss_weight),
        "soft_tau": float(args.soft_tau),
        "action_format": "E00..E14",
        "lora": {
            "enabled": not bool(args.no_lora),
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": [item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()],
        },
    }


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
    n = float(parts["n_samples"])
    sums = torch.tensor(
        [
            float(parts["loss"]) * n,
            float(parts["hard_loss"]) * n,
            float(parts["soft_loss"]) * n,
            float(parts["accuracy"]) * n,
            n,
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    gathered = accelerator.gather_for_metrics(sums.unsqueeze(0)).sum(dim=0)
    n_total = max(float(gathered[4].item()), 1.0)
    return {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "loss": float(gathered[0].item() / n_total),
        "hard_loss": float(gathered[1].item() / n_total),
        "soft_loss": float(gathered[2].item() / n_total),
        "accuracy": float(gathered[3].item() / n_total),
        "n_samples": int(n_total),
        "lr": float(lr),
        **_cuda_memory_metrics(),
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


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
        "train/action_accuracy": record.get("accuracy"),
        "train/lr": record.get("lr"),
        "train/epoch": record.get("epoch"),
        "train/global_step": record.get("global_step"),
        "system/cuda_max_memory_allocated_gb": record.get("cuda_max_memory_allocated_gb"),
        "system/cuda_memory_reserved_gb": record.get("cuda_memory_reserved_gb"),
    }


def _val_swanlab_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "val/loss": result.get("val_loss"),
        "val/hard_loss": result.get("val_hard_loss"),
        "val/soft_loss": result.get("val_soft_loss"),
        "val/action_accuracy": result.get("val_action_accuracy"),
        "val/n_samples": result.get("n_val_samples"),
    }


def _metadata_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "config/max_length": metadata.get("max_length"),
        "config/choice_batch_size": metadata.get("choice_batch_size"),
        "config/n_train_samples": metadata.get("n_train_samples"),
        "config/n_val_samples": metadata.get("n_val_samples"),
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
