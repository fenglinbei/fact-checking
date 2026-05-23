#!/usr/bin/env python3
"""Build step-wise LLM action-selector samples from Stage2 oracle + VIG rows."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fact_checking.selectors.llm_action import build_action_samples
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
    read_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Stage2 oracle examples and saved-score VIG rows into LLM action SFT samples."
    )
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--vig-cache", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--max-candidate-chars", type=int, default=180)
    p.add_argument("--no-retrieval-scores", action="store_true")
    p.add_argument("--allow-missing-vig", action="store_true")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path for prompt length statistics.")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=str(args.expected_chunk_mmr_fingerprint),
        max_candidates=int(args.max_candidates),
        top_k=int(args.top_k),
        filter_policy=str(args.filter_policy),
        min_margin=float(args.min_margin),
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No Stage2 oracle examples after audit/filtering.")

    vig_rows = read_jsonl(args.vig_cache)
    if not vig_rows:
        raise ValueError(f"No VIG rows found: {args.vig_cache}")

    samples, manifest = build_action_samples(
        examples,
        vig_rows=vig_rows,
        split=str(args.split),
        top_k=int(args.top_k),
        max_candidate_chars=int(args.max_candidate_chars),
        include_retrieval_scores=not bool(args.no_retrieval_scores),
        strict=not bool(args.allow_missing_vig),
        show_progress=not bool(args.no_progress),
    )
    if not samples:
        raise ValueError("No action selector samples were generated.")

    manifest.update(
        {
            "oracle_results": str(args.oracle_results),
            "vig_cache": str(args.vig_cache),
            "output_jsonl": str(args.output_jsonl),
            "filter_policy": str(args.filter_policy),
            "chunk_mmr_fingerprint": str(args.expected_chunk_mmr_fingerprint),
        }
    )
    if args.tokenizer:
        manifest["prompt_length_stats"] = _prompt_length_stats(
            samples,
            tokenizer_name=str(args.tokenizer),
            max_length=int(args.max_length),
            show_progress=not bool(args.no_progress),
        )

    write_jsonl(args.output_jsonl, samples)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output_jsonl).with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(f"Wrote LLM action selector samples: {args.output_jsonl}")
    print(f"Wrote manifest: {manifest_path}")
    print(
        "samples={n_samples} examples={n_examples} missing_steps={missing_vig_steps} "
        "missing_candidates={missing_vig_candidates}".format(**manifest)
    )


def _prompt_length_stats(
    samples: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    max_length: int,
    show_progress: bool,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    lengths: list[int] = []
    over_budget = 0
    iterator = tqdm(
        samples,
        desc="prompt length stats",
        unit="sample",
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )
    for sample in iterator:
        text = str(sample["prompt"]) + str(sample["target_action"])
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        length = len(ids)
        lengths.append(length)
        over_budget += int(length > int(max_length))
    arr = sorted(lengths)
    return {
        "tokenizer": tokenizer_name,
        "max_length": int(max_length),
        "n": int(len(arr)),
        "min": int(arr[0]) if arr else 0,
        "mean": float(sum(arr) / max(len(arr), 1)),
        "p50": _percentile(arr, 0.50),
        "p95": _percentile(arr, 0.95),
        "p99": _percentile(arr, 0.99),
        "max": int(arr[-1]) if arr else 0,
        "over_budget": int(over_budget),
    }


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    pos = min(max(float(q), 0.0), 1.0) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return int(sorted_values[lo])
    frac = pos - lo
    return int(round(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac))


if __name__ == "__main__":
    main()
