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
from liar_raw_cde.stage_d.trainer import evaluate
from liar_raw_cde.utils.io import ensure_dir, load_yaml, save_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    graph_path = cfg["data"][f"{args.split}_path"]
    ds = GraphDataset(graph_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"], use_fast=True)
    collator = GraphBatchCollator(tokenizer=tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collator, num_workers=cfg["train"]["num_workers"])

    model = GraphVerifier(
        encoder_name=cfg["model"]["encoder_name"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        num_labels=cfg["model"]["num_labels"],
        use_ordinal_head=cfg["model"]["use_ordinal_head"],
    ).to(device)

    ckpt_path = args.checkpoint or str(Path(cfg["output"]["dir"]) / "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    class_weights = build_class_balanced_weights(ds.label_ids(), num_classes=cfg["model"]["num_labels"], beta=cfg["train"]["beta_for_cb"]).to(device)
    criterion = GraphVerifierCriterion(class_weights=class_weights, lambda_ordinal=cfg["train"]["lambda_ordinal"]).to(device)

    metrics, predictions = evaluate(model, loader, criterion, device, export_predictions=True)
    out_dir = ensure_dir(cfg["output"]["dir"])
    write_jsonl(predictions, Path(out_dir) / f"{args.split}.graph_predictions.jsonl")
    save_json(metrics, Path(out_dir) / f"{args.split}.graph_metrics.json")
    print(f"{args.split} macro-F1 = {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
