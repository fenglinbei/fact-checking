from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from fact_checking.stage_e.faithfulness import FaithfulnessFilter
from fact_checking.stage_e.inference import generate_seq2seq_records, generate_template_records
from fact_checking.utils.io import ensure_dir, load_yaml, read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    pred_path = cfg["data"][f"{args.split}_graph_predictions"]
    pred_items = read_jsonl(pred_path)

    mode = cfg["model"]["mode"]
    out_dir = ensure_dir(cfg["output"]["dir"])

    if mode == "template":
        outputs = generate_template_records(pred_items)
    else:
        faithfulness_filter = None
        if cfg["faithfulness"]["enable"]:
            faithfulness_filter = FaithfulnessFilter(
                embedder_model=cfg["faithfulness"]["embedder_model"],
                semantic_threshold=cfg["faithfulness"]["semantic_threshold"],
                lexical_jaccard_threshold=cfg["faithfulness"]["lexical_jaccard_threshold"],
                device=args.device,
            )
        model_dir = Path(cfg["output"]["dir"]) / "seq2seq_explainer"
        outputs = generate_seq2seq_records(
            pred_items=pred_items,
            model_dir=model_dir,
            max_input_length=cfg["model"]["max_input_length"],
            max_output_length=cfg["model"]["max_output_length"],
            faithfulness_filter=faithfulness_filter,
            device=args.device,
        )

    out_path = Path(out_dir) / f"{args.split}.explanations.jsonl"
    write_jsonl(outputs, out_path)
    print(f"Saved explanations to {out_path}")


if __name__ == "__main__":
    main()
