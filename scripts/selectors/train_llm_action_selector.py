#!/usr/bin/env python3
"""Train a Qwen/LoRA sequential action selector with VIG soft supervision."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.selectors.llm_action import score_action_choices, softmax_deltas
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an LLM action evidence selector.")
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="/home/fenglin/project/hateSpeechDetection/models/base/Qwen2.5-7B-Instruct")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--choice-batch-size", type=int, default=64)
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

    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    if args.eval_sample_limit is not None:
        val_rows = val_rows[: int(args.eval_sample_limit)]
    if not train_rows:
        raise ValueError("No train action samples.")
    if not val_rows:
        raise ValueError("No val action samples.")

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
    if accelerator.is_main_process:
        write_json(out_dir / "selector_metadata.json", metadata)

    best_accuracy = float("-inf")
    history: list[dict[str, Any]] = []
    global_step = 0
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
                if global_step % int(args.logging_steps) == 0 and accelerator.is_main_process:
                    print(
                        "step={step} loss={loss:.4f} hard={hard:.4f} soft={soft:.4f} acc={acc:.4f}".format(
                            step=global_step,
                            loss=float(parts["loss"]),
                            hard=float(parts["hard_loss"]),
                            soft=float(parts["soft_loss"]),
                            acc=float(parts["accuracy"]),
                        )
                    )
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
                    best_accuracy = _maybe_save_best(
                        result,
                        best_accuracy=best_accuracy,
                        accelerator=accelerator,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=out_dir,
                        metadata=metadata,
                    )

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
        best_accuracy = _maybe_save_best(
            result,
            best_accuracy=best_accuracy,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            output_dir=out_dir,
            metadata=metadata,
        )

    progress.close()
    if accelerator.is_main_process:
        payload = {
            "metadata": metadata,
            "history": history,
            "best_action_accuracy": best_accuracy,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        write_json(out_dir / "training_metrics.json", payload)
        print(f"Wrote training metrics: {out_dir / 'training_metrics.json'}")


def _batch_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    choice_batch_size: int,
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
