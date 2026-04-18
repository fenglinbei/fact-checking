from __future__ import annotations

import argparse
from pathlib import Path

from liar_raw.baselines.llm_baseline import BaselineConfig, run_inference
from liar_raw.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0/B1 LLM baselines.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    baseline_cfg = cfg["baseline"]

    baseline = BaselineConfig(
        model_name_or_path=str(baseline_cfg.get("model_name_or_path", "/data/models/Qwen3.5-9B")),
        top_k=int(baseline_cfg.get("top_k", 8)),
        use_context=bool(baseline_cfg.get("use_context", False)),
        context_k=int(baseline_cfg.get("context_k", 1)),
        prompt_mode=str(baseline_cfg.get("prompt_mode", "few_shot")),
        few_shot_k=int(baseline_cfg.get("few_shot_k", 10)),
        few_shot_mmr_lambda=float(baseline_cfg.get("few_shot_mmr_lambda", 0.7)),
        retrieval_model=str(baseline_cfg.get("retrieval_model", "/home/fenglin/project/models/bge-base-en-v1.5/")),
        retrieval_batch_size=int(baseline_cfg.get("retrieval_batch_size", 64)),
        retrieval_max_length=int(baseline_cfg.get("retrieval_max_length", 256)),
        max_new_tokens=int(baseline_cfg.get("max_new_tokens", 24)),
        temperature=float(baseline_cfg.get("temperature", 0.0)),
        do_sample=bool(baseline_cfg.get("do_sample", False)),
    )

    split_map = {
        "train": str(data_cfg["train_candidates"]),
        "val": str(data_cfg["val_candidates"]),
        "test": str(data_cfg["test_candidates"]),
    }
    input_path = split_map[args.split]

    output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "b1" if baseline.use_context else "b0"
    out_path = output_dir / f"{tag}_{args.split}.predictions.jsonl"

    run_inference(
        cfg=baseline,
        input_path=input_path,
        output_path=out_path,
        train_path_for_few_shot=split_map["train"],
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
