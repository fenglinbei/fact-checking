from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from stage_d.inference import build_graph_prediction_records
from utils.io import ensure_dir, save_json


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def macro_prf(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if len(y_true) == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    train_loader,
    num_epochs: int,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
    warmup_ratio: float,
):
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _get_group_lrs(optimizer) -> dict[str, float]:
    lrs = {}
    if len(optimizer.param_groups) >= 1:
        lrs["encoder"] = float(optimizer.param_groups[0]["lr"])
    if len(optimizer.param_groups) >= 2:
        lrs["task"] = float(optimizer.param_groups[1]["lr"])
    for i, group in enumerate(optimizer.param_groups[2:], start=2):
        lrs[f"group_{i}"] = float(group["lr"])
    return lrs


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    max_grad_norm: float,
    epoch: int | None = None,
    global_step: int = 0,
    log_interval_steps: int = 10,
    log_fn: Callable[[dict, int | None], None] | None = None,
    progress_desc: str = "Train",
    show_progress: bool = True,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_ordinal = 0.0
    total_correct = 0
    total_examples = 0
    total_steps = 0

    progress = tqdm(
        loader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        batch_size = int(labels.size(0))

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            node_mask=batch["node_mask"],
            node_type_ids=batch["node_type_ids"],
            stance_ids=batch["stance_ids"],
            scalar_feats=batch["scalar_feats"],
            adj=batch["adj"],
        )

        loss_dict = criterion(outputs, labels)
        loss = loss_dict["loss"]
        ce = loss_dict["ce"]
        ordinal = loss_dict.get("ordinal", 0.0)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        preds = outputs["class_logits"].argmax(dim=-1)
        correct = int((preds == labels).sum().item())

        loss_value = _to_float(loss)
        ce_value = _to_float(ce)
        ordinal_value = _to_float(ordinal)
        grad_norm_value = _to_float(grad_norm)

        total_loss += loss_value * batch_size
        total_ce += ce_value * batch_size
        total_ordinal += ordinal_value * batch_size
        total_correct += correct
        total_examples += batch_size
        total_steps += 1
        global_step += 1

        running_acc = total_correct / max(total_examples, 1)
        lrs = _get_group_lrs(optimizer)
        display_lr = lrs.get("task", lrs.get("encoder", 0.0))

        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            ce=f"{ce_value:.4f}",
            ord=f"{ordinal_value:.4f}",
            acc=f"{running_acc:.4f}",
            lr=f"{display_lr:.2e}",
        )

        if log_fn is not None and (global_step == 1 or global_step % max(log_interval_steps, 1) == 0):
            payload = {
                "train_step/loss": loss_value,
                "train_step/ce": ce_value,
                "train_step/ordinal": ordinal_value,
                "train_step/accuracy": correct / max(batch_size, 1),
                "train_step/grad_norm": grad_norm_value,
                "train_step/global_step": global_step,
            }
            if epoch is not None:
                payload["epoch"] = epoch
            for name, lr_value in lrs.items():
                payload[f"train_step/lr_{name}"] = lr_value
            log_fn(payload, step=global_step)

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "ce": total_ce / max(total_examples, 1),
        "ordinal": total_ordinal / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "steps": total_steps,
    }
    for name, lr_value in _get_group_lrs(optimizer).items():
        metrics[f"lr_{name}"] = lr_value

    return metrics, global_step


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    export_predictions: bool = False,
    progress_desc: str = "Eval",
    show_progress: bool = True,
):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_ordinal = 0.0
    total_correct = 0
    total_examples = 0
    total_steps = 0

    y_true = []
    y_pred = []
    predictions = []

    progress = tqdm(
        loader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        batch_size = int(labels.size(0))

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            node_mask=batch["node_mask"],
            node_type_ids=batch["node_type_ids"],
            stance_ids=batch["stance_ids"],
            scalar_feats=batch["scalar_feats"],
            adj=batch["adj"],
        )

        loss_dict = criterion(outputs, labels)
        preds = outputs["class_logits"].argmax(dim=-1)

        loss_value = _to_float(loss_dict["loss"])
        ce_value = _to_float(loss_dict["ce"])
        ordinal_value = _to_float(loss_dict.get("ordinal", 0.0))

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

        total_loss += loss_value * batch_size
        total_ce += ce_value * batch_size
        total_ordinal += ordinal_value * batch_size
        total_correct += int((preds == labels).sum().item())
        total_examples += batch_size
        total_steps += 1

        progress.set_postfix(
            loss=f"{(total_loss / max(total_examples, 1)):.4f}",
            acc=f"{(total_correct / max(total_examples, 1)):.4f}",
        )

        if export_predictions:
            predictions.extend(build_graph_prediction_records(batch, outputs))

    metrics = macro_prf(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["ce"] = total_ce / max(total_examples, 1)
    metrics["ordinal"] = total_ordinal / max(total_examples, 1)
    metrics["accuracy"] = total_correct / max(total_examples, 1)
    metrics["steps"] = total_steps

    return metrics, predictions


def save_checkpoint(model, tokenizer_name: str, config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_name": tokenizer_name,
            "config": config,
        },
        path,
    )


def save_history(history: list[dict[str, Any]], path: str | Path) -> None:
    save_json(history, path)