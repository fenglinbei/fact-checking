from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

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


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def build_wandb_run(cfg: dict, output_dir: Path):
    wb_cfg = cfg.get("wandb", {})
    enabled = bool(wb_cfg.get("enable", False))
    if not enabled:
        return None

    if wandb is None:
        raise ImportError(
            "wandb is enabled in config, but the package is not installed. "
            "Please run: pip install wandb"
        )

    run = wandb.init(
        project=wb_cfg.get("project", "graph-verifier"),
        name=wb_cfg.get("run_name"),
        entity=wb_cfg.get("entity"),
        tags=wb_cfg.get("tags"),
        mode=wb_cfg.get("mode"),  # e.g. online / offline / disabled
        dir=str(output_dir),
        config=cfg,
    )
    return run


def prefix_numeric_metrics(metrics: dict, prefix: str) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[f"{prefix}/{k}"] = float(v)
    return out


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = GraphDataset(cfg["data"]["train_path"])
    val_ds = GraphDataset(cfg["data"]["val_path"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"], use_fast=True)
    collator = GraphBatchCollator(
        tokenizer=tokenizer,
        max_length=int(cfg["model"].get("max_length", 128)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg["train"]["num_workers"],
    )

    model = GraphVerifier(
        encoder_name=cfg["model"]["encoder_name"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        num_labels=cfg["model"]["num_labels"],
        use_ordinal_head=cfg["model"]["use_ordinal_head"],
    ).to(device)

    class_weights = build_class_balanced_weights(
        train_ds.label_ids(),
        num_classes=cfg["model"]["num_labels"],
        beta=cfg["train"]["beta_for_cb"],
    ).to(device)

    criterion = GraphVerifierCriterion(
        class_weights=class_weights,
        lambda_ordinal=cfg["train"]["lambda_ordinal"],
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=cfg["train"]["num_epochs"],
        lr=cfg["train"]["lr"],
        encoder_lr=cfg["train"]["encoder_lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    output_dir = Path(ensure_dir(cfg["output"]["dir"]))

    run = build_wandb_run(cfg, output_dir) if is_main_process() else None
    wandb_cfg = cfg.get("wandb", {})
    log_interval_steps = int(wandb_cfg.get("log_interval_steps", 10))
    watch_model = bool(wandb_cfg.get("watch_model", False))

    if run is not None and watch_model:
        run.watch(model, log="all", log_freq=log_interval_steps)

    def log_fn(data: dict, step: int | None = None):
        if run is None:
            return
        if step is None:
            run.log(data)
        else:
            run.log(data, step=step)

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    early_stopping_patience = int(cfg["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0))
    history = []
    global_step = 0

    epoch_bar = tqdm(
        range(1, cfg["train"]["num_epochs"] + 1),
        desc="Epochs",
        dynamic_ncols=True,
        disable=not is_main_process(),
    )

    for epoch in epoch_bar:
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            max_grad_norm=cfg["train"]["max_grad_norm"],
            epoch=epoch,
            global_step=global_step,
            log_interval_steps=log_interval_steps,
            log_fn=log_fn if run is not None else None,
            progress_desc=f"Train {epoch}/{cfg['train']['num_epochs']}",
            show_progress=is_main_process(),
        )

        val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            export_predictions=False,
            progress_desc=f"Eval {epoch}/{cfg['train']['num_epochs']}",
            show_progress=is_main_process(),
        )

        improved = val_metrics["macro_f1"] > (best_f1 + early_stopping_min_delta)
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                cfg["model"]["encoder_name"],
                cfg,
                output_dir / "best_model.pt",
            )
            print(f"Saved best checkpoint. val_macro_f1={best_f1:.4f}")
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_f1=f"{val_metrics['macro_f1']:.4f}",
        )

        if run is not None:
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "best/val_macro_f1": float(best_f1),
                "best/epoch": int(best_epoch),
            }
            payload.update(prefix_numeric_metrics(train_metrics, "train_epoch"))
            payload.update(prefix_numeric_metrics(val_metrics, "val"))
            log_fn(payload, step=global_step)
            run.summary["best_val_macro_f1"] = float(best_f1)
            run.summary["best_epoch"] = int(best_epoch)

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping triggered at epoch "
                f"{epoch}. best_epoch={best_epoch}, best_val_macro_f1={best_f1:.4f}"
            )
            break

    save_history(history, output_dir / "train_history.json")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
