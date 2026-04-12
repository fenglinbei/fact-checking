from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from liar_raw import ID2LABEL
from liar_raw.config import load_yaml
from liar_raw.models.latent_evidence import LatentEvidenceOrdinalModel
from liar_raw.training.stage_b_data import StageBCollator, StageBDataset, build_tokenizer



def prepare_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage B inference and export latent evidence.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    device = prepare_device(str(train_cfg.get("device", "cuda")))

    split_to_path = {
        "train": data_cfg["train_candidates"],
        "val": data_cfg["val_candidates"],
        "test": data_cfg["test_candidates"],
    }
    ds = StageBDataset(split_to_path[args.split], top_k=int(data_cfg.get("top_k", 24)))
    tokenizer = build_tokenizer(model_cfg["backbone_model"])
    collator = StageBCollator(
        tokenizer=tokenizer,
        max_length=int(model_cfg.get("max_length", 256)),
        top_k=int(data_cfg.get("top_k", 24)),
    )
    loader = DataLoader(
        ds,
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
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stage_b_predictions_{args.split}.jsonl"

    with output_path.open("w", encoding="utf-8") as writer:
        for batch in tqdm(loader, desc=f"Predict [{args.split}]"):
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
            probs = torch.softmax(output.class_logits, dim=-1).cpu()
            preds = probs.argmax(dim=-1).tolist()
            evidence = model.extract_evidence(output, metadata=batch["metadata"], top_n=3)

            for event_id, claim, gold, pred, prob, ev in zip(
                batch["event_ids"],
                batch["claims"],
                batch["labels"].tolist(),
                preds,
                probs.tolist(),
                evidence,
            ):
                writer.write(
                    json.dumps(
                        {
                            "event_id": event_id,
                            "claim": claim,
                            "gold_label": ID2LABEL[int(gold)],
                            "pred_label": ID2LABEL[int(pred)],
                            "class_probs": prob,
                            "support_evidence": ev["support"],
                            "refute_evidence": ev["refute"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
