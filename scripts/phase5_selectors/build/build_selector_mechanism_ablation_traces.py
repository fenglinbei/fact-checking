#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.selector_mechanism_ablation import (  # noqa: E402
    SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
    SELECTOR_MECHANISM_GRAPH_VERSION,
    SELECTOR_MECHANISM_NAMES,
    SelectorMechanismParams,
    build_claim_candidate_pool_row,
    build_selector_mechanism_trace_row,
    summarize_selector_mechanism_traces,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl  # noqa: E402


DEFAULT_CHUNK_CACHE = "outputs/cache/chunk_mmr/d4cbf7c18126/val.pkl"
DEFAULT_UNION_JSONL = (
    "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/03_atom_union/"
    "atom_union_candidate_pool_val.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/selectors/selector_mechanism_ablation/liar_raw_selector_mech_s2_claim_pool_hybrid_top5_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S0-S4 selector mechanism ablation traces.")
    parser.add_argument("--chunk-cache-path", default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--atom-union-jsonl", default=DEFAULT_UNION_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--selector-name", required=True, choices=SELECTOR_MECHANISM_NAMES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--claim-pool-top-n", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--merge-mmr-lambda", type=float, default=0.70)
    parser.add_argument("--chunk-mmr-fingerprint", default="d4cbf7c18126")
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    chunk_cache_path = Path(args.chunk_cache_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_chunk_cache(chunk_cache_path)
    if int(args.sample_limit) > 0:
        samples = samples[: int(args.sample_limit)]
    if not samples:
        raise SystemExit(f"No samples read from {chunk_cache_path}")

    params = SelectorMechanismParams(
        top_k=int(args.top_k),
        claim_pool_top_n=int(args.claim_pool_top_n),
        random_seed=int(args.random_seed),
        merge_mmr_lambda=float(args.merge_mmr_lambda),
    )
    claim_rows = {
        str(row.get("event_id") or ""): row
        for row in (build_claim_candidate_pool_row(sample, params=params) for sample in samples)
    }
    union_rows = _load_union_rows(args.atom_union_jsonl) if args.selector_name == SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5 else {}

    traces: list[dict[str, Any]] = []
    for event_id, claim_row in claim_rows.items():
        union_row = union_rows.get(event_id)
        if args.selector_name == SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5 and union_row is None:
            raise SystemExit(f"Missing atom union row for event_id={event_id}")
        traces.append(
            build_selector_mechanism_trace_row(
                claim_pool_row=claim_row,
                union_row=union_row,
                selector_name=str(args.selector_name),
                params=params,
                chunk_mmr_fingerprint=str(args.chunk_mmr_fingerprint or ""),
            )
        )

    diagnostics = summarize_selector_mechanism_traces(traces)
    manifest = _manifest(args=args, chunk_cache_path=chunk_cache_path, n_input_rows=len(samples), n_trace_rows=len(traces))

    write_jsonl(traces, output_dir / f"selection_trace_{args.split}.jsonl")
    save_json(diagnostics, output_dir / "selector_diagnostics.json")
    save_json(manifest, output_dir / "manifest.json")

    print(f"Wrote {len(traces)} selector mechanism ablation traces to {output_dir}")
    print(f"Selector: {args.selector_name}")
    print(f"Diagnostics: {output_dir / 'selector_diagnostics.json'}")
    return 0


def _load_chunk_cache(path: Path) -> list[Any]:
    if not path.exists():
        raise SystemExit(f"Missing chunk cache: {path}")
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, list):
        raise SystemExit(f"Expected list chunk cache at {path}, got {type(loaded).__name__}")
    return loaded


def _load_union_rows(path_value: str | None) -> dict[str, dict[str, Any]]:
    if not path_value:
        raise SystemExit("--atom-union-jsonl is required for S4")
    path = Path(path_value)
    if not path.exists():
        raise SystemExit(f"Missing atom union jsonl: {path}")
    return {str(row.get("event_id") or ""): row for row in read_jsonl(path)}


def _manifest(
    *,
    args: argparse.Namespace,
    chunk_cache_path: Path,
    n_input_rows: int,
    n_trace_rows: int,
) -> dict[str, Any]:
    return {
        "graph_version": SELECTOR_MECHANISM_GRAPH_VERSION,
        "selector_name": str(args.selector_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_cache_path": str(chunk_cache_path),
        "atom_union_jsonl": str(args.atom_union_jsonl or ""),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "params": {
            "top_k": int(args.top_k),
            "claim_pool_top_n": int(args.claim_pool_top_n),
            "random_seed": int(args.random_seed),
            "merge_mmr_lambda": float(args.merge_mmr_lambda),
            "chunk_mmr_fingerprint": str(args.chunk_mmr_fingerprint or ""),
        },
        "n_input_rows": int(n_input_rows),
        "n_trace_rows": int(n_trace_rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
