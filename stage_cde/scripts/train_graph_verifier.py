from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from liar_raw_cde.stage_d.collator import GraphBatchCollator
from liar_raw_cde.stage_d.dataset import GraphDataset
from liar_raw_cde.stage_d.losses import GraphVerifierCriterion, build_class_balanced_weights
from liar_raw_cde.stage_d.model import GraphVerifier
from liar_raw_cde.stage_d.trainer import (
    build_optimizer_and_scheduler,
    evaluate,
    save_checkpoint,
    save_history,
    train_one_epoch,
)
from liar_raw_cde.utils.io import ensure_dir, load_yaml
from liar_raw_cde.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = GraphDataset(cfg["data"]["train_path"])
    val_ds = GraphDataset(cfg["data"]["val_path"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"], use_fast=True)
    collator = GraphBatchCollator(tokenizer=tokenizer, max_length=128)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collator, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collator, num_workers=cfg["train"]["num_workers"])

    model = GraphVerifier(
        encoder_name=cfg["model"]["encoder_name"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        num_labels=cfg["model"]["num_labels"],
        use_ordinal_head=cfg["model"]["use_ordinal_head"],
    ).to(device)

    class_weights = build_class_balanced_weights(train_ds.label_ids(), num_classes=cfg["model"]["num_labels"], beta=cfg["train"]["beta_for_cb"]).to(device)
    criterion = GraphVerifierCriterion(class_weights=class_weights, lambda_ordinal=cfg["train"]["lambda_ordinal"]).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=cfg["train"]["num_epochs"],
        lr=cfg["train"]["lr"],
        encoder_lr=cfg["train"]["encoder_lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    output_dir = ensure_dir(cfg["output"]["dir"])
    best_f1 = -1.0
    history = []

    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            max_grad_norm=cfg["train"]["max_grad_norm"],
        )
        val_metrics, _ = evaluate(model, val_loader, criterion, device, export_predictions=False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, cfg["model"]["encoder_name"], cfg, Path(output_dir) / "best_model.pt")
            print(f"Saved best checkpoint. val_macro_f1={best_f1:.4f}")

    save_history(history, Path(output_dir) / "train_history.json")


if __name__ == "__main__":
    main()
