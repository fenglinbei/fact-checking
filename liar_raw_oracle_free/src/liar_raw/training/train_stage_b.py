from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from liar_raw import ID2LABEL
from liar_raw.config import load_yaml
from liar_raw.models.latent_evidence import LatentEvidenceOrdinalModel
from liar_raw.models.ordinal import coral_loss
from liar_raw.training.metrics import compute_macro_metrics
from liar_raw.training.stage_b_data import StageBCollator, StageBDataset, build_tokenizer


MARGIN_TARGETS = torch.tensor([-1.0, -0.7, -0.2, 0.2, 0.7, 1.0], dtype=torch.float)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def prepare_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")



def compute_class_weights(dataset: StageBDataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(6, dtype=np.float64)
    for row in dataset.rows:
        label_id = int(row["label_id"]) if "label_id" in row else None
        if label_id is None:
            # For rows loaded directly from jsonl.
            from liar_raw import LABEL2ID

            label_id = LABEL2ID[row["label"]]
        counts[label_id] += 1.0
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=device)



def forward_loss(
    model: LatentEvidenceOrdinalModel,
    batch: dict[str, Any],
    class_weights: torch.Tensor,
    lambda_ordinal: float,
    lambda_margin: float,
    lambda_overlap: float,
    lambda_entropy: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    labels = batch["labels"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    candidate_mask = batch["candidate_mask"].to(device)
    hybrid_scores = batch["hybrid_scores"].to(device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_mask=candidate_mask,
        hybrid_scores=hybrid_scores,
    )

    ce = F.cross_entropy(output.class_logits, labels, weight=class_weights)
    ord_loss = coral_loss(output.ordinal_logits, labels, num_classes=6)

    margin_targets = MARGIN_TARGETS.to(device)[labels]
    margin_loss = F.mse_loss(torch.tanh(output.margin * 2.0), margin_targets)

    valid = candidate_mask.float()
    overlap = (output.support_prob * output.refute_prob * valid).sum() / valid.sum().clamp(min=1.0)

    attn = output.attention_weights.clamp(min=1e-8)
    entropy = -torch.sum(attn * torch.log(attn), dim=-1).mean()

    loss = ce + lambda_ordinal * ord_loss + lambda_margin * margin_loss + lambda_overlap * overlap + lambda_entropy * entropy
    logs = {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "ordinal": float(ord_loss.detach().cpu()),
        "margin": float(margin_loss.detach().cpu()),
        "overlap": float(overlap.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }
    return loss, logs, output.class_logits.detach()


@torch.inference_mode()
def evaluate(
    model: LatentEvidenceOrdinalModel,
    loader: DataLoader,
    class_weights: torch.Tensor,
    lambda_ordinal: float,
    lambda_margin: float,
    lambda_overlap: float,
    lambda_entropy: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    losses: list[float] = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        loss, _, logits = forward_loss(
            model=model,
            batch=batch,
            class_weights=class_weights,
            lambda_ordinal=lambda_ordinal,
            lambda_margin=lambda_margin,
            lambda_overlap=lambda_overlap,
            lambda_entropy=lambda_entropy,
            device=device,
        )
        preds = logits.argmax(dim=-1).cpu().tolist()
        labels = batch["labels"].cpu().tolist()
        all_true.extend(labels)
        all_pred.extend(preds)
        losses.append(float(loss.cpu()))
    metrics = compute_macro_metrics(all_true, all_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics



def save_checkpoint(
    save_dir: Path,
    model: LatentEvidenceOrdinalModel,
    tokenizer_name: str,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_name": tokenizer_name,
            "config": cfg,
            "metrics": metrics,
        },
        ckpt_path,
    )
    with (save_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train oracle-free Stage B latent evidence model.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    set_seed(int(train_cfg.get("seed", 13)))
    device = prepare_device(str(train_cfg.get("device", "cuda")))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StageBDataset(data_cfg["train_candidates"], top_k=int(data_cfg.get("top_k", 24)))
    val_ds = StageBDataset(data_cfg["val_candidates"], top_k=int(data_cfg.get("top_k", 24)))
    tokenizer = build_tokenizer(model_cfg["backbone_model"])
    collator = StageBCollator(
        tokenizer=tokenizer,
        max_length=int(model_cfg.get("max_length", 256)),
        top_k=int(data_cfg.get("top_k", 24)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 4))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )

    model = LatentEvidenceOrdinalModel(
        model_name=model_cfg["backbone_model"],
        num_classes=6,
        dropout=float(model_cfg.get("dropout", 0.1)),
        unfreeze_last_n_layers=int(model_cfg.get("unfreeze_last_n_layers", 2)),
    ).to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params, lr=float(train_cfg.get("lr", 2e-5)))
    total_steps = len(train_loader) * int(train_cfg.get("epochs", 5))
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    class_weights = compute_class_weights(train_ds, device=device)

    lambda_ordinal = float(train_cfg.get("lambda_ordinal", 0.5))
    lambda_margin = float(train_cfg.get("lambda_margin", 0.2))
    lambda_overlap = float(train_cfg.get("lambda_overlap", 0.05))
    lambda_entropy = float(train_cfg.get("lambda_entropy", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("patience", 2))

    best_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Train epoch {epoch}")
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, logs, _ = forward_loss(
                model=model,
                batch=batch,
                class_weights=class_weights,
                lambda_ordinal=lambda_ordinal,
                lambda_margin=lambda_margin,
                lambda_overlap=lambda_overlap,
                lambda_entropy=lambda_entropy,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += logs["loss"]
            if step % int(train_cfg.get("log_every", 20)) == 0 or step == 1:
                progress.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            class_weights=class_weights,
            lambda_ordinal=lambda_ordinal,
            lambda_margin=lambda_margin,
            lambda_overlap=lambda_overlap,
            lambda_entropy=lambda_entropy,
            device=device,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / max(1, len(train_loader)),
                    **val_metrics,
                },
                indent=2,
            )
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir, model, model_cfg["backbone_model"], cfg, val_metrics)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    print("Best validation metrics:")
    print(json.dumps({"best_epoch": best_epoch, **best_metrics}, indent=2))


if __name__ == "__main__":
    main()
