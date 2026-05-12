"""Step 1: Generate build JSONL for each λ value from PreMMR cache.

Reuses the existing _mmr_phase_from_premmr to produce prompts identical to
those the main pipeline would generate, one file per λ.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/generate_oracle_prompts.py \
        --premmr-cache outputs/cache/pre_mmr/53a3588e485d/train.pkl \
        --output-dir outputs/learned_lambda/prompts/ \
        --model-name-or-path /data/models/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fact_checking.build.candidates import (
    SentenceChunking,
    _load_pickle,
    _load_prompt_tokenizer,
    _mmr_phase_from_premmr,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate oracle-λ prompts for each λ value.")
    p.add_argument("--premmr-cache", type=str, required=True, help="Path to PreMMR cache pickle (e.g. train.pkl)")
    p.add_argument("--output-dir", type=str, required=True, help="Directory for per-λ JSONL outputs")
    p.add_argument("--model-name-or-path", type=str, required=True, help="Tokenizer model path for prompt construction")
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--alpha-dense", type=float, default=0.70)
    p.add_argument("--alpha-lexical", type=float, default=0.20)
    p.add_argument("--alpha-bm25", type=float, default=0.10)
    p.add_argument("--lambda-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--prompt-max-length", type=int, default=1024)
    p.add_argument("--prompt-output-mode", type=str, default="label_only")
    p.add_argument("--prompt-label-format", type=str, default="letter")
    p.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PreMMR cache: {args.premmr_cache}", flush=True)
    pre_samples = _load_pickle(Path(args.premmr_cache))
    print(f"Loaded {len(pre_samples)} samples", flush=True)

    tokenizer = _load_prompt_tokenizer(args.model_name_or_path)
    prompt_cfg = {
        "auto_length": True,
        "max_length": args.prompt_max_length,
        "output_mode": args.prompt_output_mode,
        "label_format": args.prompt_label_format,
        "system_prompt": None,
    }
    strategy = SentenceChunking()

    for lam in lambda_grid:
        output_path = output_dir / f"lambda_{lam:.1f}_{args.split_name}.jsonl"
        print(f"Generating prompts for λ={lam:.1f} → {output_path}", flush=True)
        _mmr_phase_from_premmr(
            pre_samples=pre_samples,
            mmr_lambda=lam,
            top_k=args.top_k,
            alpha_dense=args.alpha_dense,
            alpha_lexical=args.alpha_lexical,
            alpha_bm25=args.alpha_bm25,
            strategy=strategy,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            output_path=output_path,
        )
        print(f"  Done: {output_path}", flush=True)

    print(f"All {len(lambda_grid)} λ values processed. Output dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
