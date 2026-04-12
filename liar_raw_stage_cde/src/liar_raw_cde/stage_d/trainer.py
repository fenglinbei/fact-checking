from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from liar_raw_cde.stage_d.inference import build_graph_prediction_records
from liar_raw_cde.utils.io import ensure_dir, save_json


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def macro_prf(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}


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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    max_grad_norm: float,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
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
        loss_dict = criterion(outputs, batch["labels"])
        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        total_steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / max(total_steps, 1)}


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    export_predictions: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    y_true = []
    y_pred = []
    predictions = []

    progress = tqdm(loader, desc="Eval", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
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
        loss_dict = criterion(outputs, batch["labels"])
        preds = outputs["class_logits"].argmax(dim=-1)
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        total_loss += float(loss_dict["loss"].item())
        total_steps += 1

        if export_predictions:
            predictions.extend(build_graph_prediction_records(batch, outputs))

    metrics = macro_prf(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_steps, 1)
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
