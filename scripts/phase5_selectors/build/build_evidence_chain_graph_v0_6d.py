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

from fact_checking.selectors.evidence_chain_graph import (
    DEFAULT_CHUNK_MMR_FINGERPRINT,
    DEFAULT_IMPORTANT_ATOM_THRESHOLD,
    DEFAULT_SUFFICIENCY_WEIGHTED_COVERAGE_THRESHOLD,
    SUFFICIENCY_CONTRADICTION_GRAPH_VERSION,
    SUFFICIENCY_CONTRADICTION_SELECTOR,
    SufficiencyContradictionEvidenceChainParams,
    build_sufficiency_contradiction_evidence_chain_graph_row,
    render_sufficiency_contradiction_case_studies,
    summarize_sufficiency_contradiction_chain_graph_rows,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_INPUT = "outputs/selectors/evidence_map_selector/v0_6b_val/candidate_evidence_map_features_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_chain_graph/v0_6d_sufficiency_contradiction_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0.6d sufficiency/contradiction evidence-chain graph artifacts.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="candidate_evidence_map_features_*.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--min-top-k", type=int, default=5)
    parser.add_argument("--max-top-k", type=int, default=10)
    parser.add_argument("--chunk-mmr-fingerprint", default=DEFAULT_CHUNK_MMR_FINGERPRINT)
    parser.add_argument("--important-atom-threshold", type=float, default=DEFAULT_IMPORTANT_ATOM_THRESHOLD)
    parser.add_argument(
        "--sufficiency-weighted-coverage-threshold",
        type=float,
        default=DEFAULT_SUFFICIENCY_WEIGHTED_COVERAGE_THRESHOLD,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    if int(args.sample_limit) > 0:
        rows = rows[: int(args.sample_limit)]
    params = SufficiencyContradictionEvidenceChainParams(
        candidate_top_n=int(args.candidate_top_n),
        min_top_k=int(args.min_top_k),
        max_top_k=int(args.max_top_k),
        chunk_mmr_fingerprint=str(args.chunk_mmr_fingerprint or ""),
        important_atom_threshold=float(args.important_atom_threshold),
        sufficiency_weighted_coverage_threshold=float(args.sufficiency_weighted_coverage_threshold),
    )
    graph_rows = [build_sufficiency_contradiction_evidence_chain_graph_row(row, params=params) for row in rows]
    traces = [row.get("selection_trace") or {} for row in graph_rows]
    diagnostics = summarize_sufficiency_contradiction_chain_graph_rows(graph_rows)
    manifest = _manifest(args=args, input_path=input_path, rows=rows, graph_rows=graph_rows)

    write_jsonl(graph_rows, output_dir / f"chain_graph_{args.split}.jsonl")
    write_jsonl(traces, output_dir / f"selection_trace_{args.split}.jsonl")
    save_json(diagnostics, output_dir / "graph_diagnostics.json")
    save_json(manifest, output_dir / "manifest.json")
    (output_dir / "case_studies.md").write_text(render_sufficiency_contradiction_case_studies(graph_rows), encoding="utf-8")

    print(f"Wrote {len(graph_rows)} graph rows to {output_dir}")
    print(f"Selector: {SUFFICIENCY_CONTRADICTION_SELECTOR}")
    print(f"Diagnostics: {output_dir / 'graph_diagnostics.json'}")


def _manifest(*, args: argparse.Namespace, input_path: Path, rows: list[dict[str, Any]], graph_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "graph_version": SUFFICIENCY_CONTRADICTION_GRAPH_VERSION,
        "selector_name": SUFFICIENCY_CONTRADICTION_SELECTOR,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "params": {
            "candidate_top_n": int(args.candidate_top_n),
            "min_top_k": int(args.min_top_k),
            "max_top_k": int(args.max_top_k),
            "chunk_mmr_fingerprint": str(args.chunk_mmr_fingerprint or ""),
            "important_atom_threshold": float(args.important_atom_threshold),
            "sufficiency_weighted_coverage_threshold": float(args.sufficiency_weighted_coverage_threshold),
        },
        "n_input_rows": len(rows),
        "n_graph_rows": len(graph_rows),
    }


if __name__ == "__main__":
    main()
