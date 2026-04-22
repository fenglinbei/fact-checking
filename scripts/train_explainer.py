from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from fact_checking.stage_e.dataset import ExplainerTrainDataset
from fact_checking.stage_e.trainer import Seq2SeqCollator, build_model_and_tokenizer, save_model_bundle, train_seq2seq
from fact_checking.utils.io import ensure_dir, load_yaml
from fact_checking.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if cfg["model"]["mode"] != "seq2seq":
        raise ValueError("train_explainer.py 仅适用于 model.mode=seq2seq")

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = ExplainerTrainDataset(cfg["data"]["raw_train_path"], cfg["data"]["train_graph_predictions"])
    val_ds = ExplainerTrainDataset(cfg["data"]["raw_val_path"], cfg["data"]["val_graph_predictions"])

    model, tokenizer = build_model_and_tokenizer(cfg["model"]["model_name"])
    collator = Seq2SeqCollator(tokenizer, cfg["model"]["max_input_length"], cfg["model"]["max_output_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collator, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collator, num_workers=cfg["train"]["num_workers"])

    history, best_state = train_seq2seq(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=cfg["train"]["num_epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = ensure_dir(Path(cfg["output"]["dir"]) / "seq2seq_explainer")
    save_model_bundle(model, tokenizer, out_dir, history)
    print(f"Saved seq2seq explainer to {out_dir}")


if __name__ == "__main__":
    main()
