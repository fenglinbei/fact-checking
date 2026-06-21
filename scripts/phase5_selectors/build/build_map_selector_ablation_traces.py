#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.map_selector_ablation import (  # noqa: E402
    DEFAULT_MAP_SELECTOR_FINGERPRINT,
    MAP_SELECTOR_GRAPH_VERSION,
    MAP_SELECTOR_NAMES,
    MapSelectorAblationParams,
    build_map_selector_ablation_trace,
    summarize_map_selector_ablation_traces,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl  # noqa: E402


DEFAULT_INPUT = (
    "outputs/selectors/evidence_map_selector/liar_raw_v0_7_atom_facts_abc_val/"
    "candidate_evidence_map_features_val.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_chain_graph/liar_raw_map_selector_s2_map_quality_top5_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed top-k map-selector ablation traces.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="candidate_evidence_map_features_*.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--selector-name", required=True, choices=MAP_SELECTOR_NAMES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--chunk-mmr-fingerprint", default=DEFAULT_MAP_SELECTOR_FINGERPRINT)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if int(args.sample_limit) > 0:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise SystemExit(f"No rows read from {input_path}")

    params = MapSelectorAblationParams(
        selector_name=str(args.selector_name),
        top_k=int(args.top_k),
        candidate_top_n=int(args.candidate_top_n),
        chunk_mmr_fingerprint=str(args.chunk_mmr_fingerprint or ""),
    )
    traces = [build_map_selector_ablation_trace(row, params=params) for row in rows]
    diagnostics = summarize_map_selector_ablation_traces(traces)
    manifest = _manifest(args=args, input_path=input_path, n_input_rows=len(rows), n_trace_rows=len(traces))

    write_jsonl(traces, output_dir / f"selection_trace_{args.split}.jsonl")
    save_json(diagnostics, output_dir / "selector_diagnostics.json")
    save_json(manifest, output_dir / "manifest.json")

    print(f"Wrote {len(traces)} map selector ablation traces to {output_dir}")
    print(f"Selector: {args.selector_name}")
    print(f"Diagnostics: {output_dir / 'selector_diagnostics.json'}")
    return 0


def _manifest(
    *,
    args: argparse.Namespace,
    input_path: Path,
    n_input_rows: int,
    n_trace_rows: int,
) -> dict[str, Any]:
    return {
        "graph_version": MAP_SELECTOR_GRAPH_VERSION,
        "selector_name": str(args.selector_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "params": {
            "top_k": int(args.top_k),
            "candidate_top_n": int(args.candidate_top_n),
            "chunk_mmr_fingerprint": str(args.chunk_mmr_fingerprint or ""),
        },
        "n_input_rows": int(n_input_rows),
        "n_trace_rows": int(n_trace_rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
