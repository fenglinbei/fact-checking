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
    BUDGETED_MARGINAL_GRAPH_VERSION,
    DEFAULT_BUDGETED_INSUFFICIENT_GAIN_THRESHOLD,
    DEFAULT_BUDGETED_STOP_GAIN_THRESHOLD,
    DEFAULT_BUDGETED_TARGET_COVERAGE,
    DEFAULT_CHUNK_MMR_FINGERPRINT,
    BudgetedMarginalChainParams,
    BudgetedMarginalObjectiveWeights,
    build_budgeted_marginal_chain_graph_row,
    budgeted_marginal_chain_selector_name,
    render_budgeted_marginal_case_studies,
    summarize_budgeted_marginal_chain_graph_rows,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_INPUT = "outputs/selectors/evidence_map_selector/v0_6b_val/candidate_evidence_map_features_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive3_10_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0.7 budgeted marginal evidence-chain graph artifacts.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="candidate_evidence_map_features_*.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--min-top-k", type=int, default=3)
    parser.add_argument("--max-top-k", type=int, default=10)
    parser.add_argument("--chunk-mmr-fingerprint", default=DEFAULT_CHUNK_MMR_FINGERPRINT)
    parser.add_argument("--target-coverage", type=float, default=DEFAULT_BUDGETED_TARGET_COVERAGE)
    parser.add_argument("--stop-gain-threshold", type=float, default=DEFAULT_BUDGETED_STOP_GAIN_THRESHOLD)
    parser.add_argument("--insufficient-gain-threshold", type=float, default=DEFAULT_BUDGETED_INSUFFICIENT_GAIN_THRESHOLD)
    parser.add_argument("--objective-coverage", type=float, default=BudgetedMarginalObjectiveWeights.coverage)
    parser.add_argument("--objective-map-quality", type=float, default=BudgetedMarginalObjectiveWeights.map_quality)
    parser.add_argument("--objective-base-score", type=float, default=BudgetedMarginalObjectiveWeights.base_score)
    parser.add_argument("--objective-key-span", type=float, default=BudgetedMarginalObjectiveWeights.key_span)
    parser.add_argument("--objective-complements", type=float, default=BudgetedMarginalObjectiveWeights.complements)
    parser.add_argument("--objective-corroborates", type=float, default=BudgetedMarginalObjectiveWeights.corroborates)
    parser.add_argument("--objective-conditional-tension", type=float, default=BudgetedMarginalObjectiveWeights.conditional_tension)
    parser.add_argument("--objective-bridge-context", type=float, default=BudgetedMarginalObjectiveWeights.bridge_context)
    parser.add_argument("--objective-duplicate-repeat", type=float, default=BudgetedMarginalObjectiveWeights.duplicate_repeat)
    parser.add_argument("--objective-background-or-irrelevant", type=float, default=BudgetedMarginalObjectiveWeights.background_or_irrelevant)
    parser.add_argument("--objective-same-source-excess-after-two", type=float, default=BudgetedMarginalObjectiveWeights.same_source_excess_after_two)
    parser.add_argument("--objective-length", type=float, default=BudgetedMarginalObjectiveWeights.length)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    if int(args.sample_limit) > 0:
        rows = rows[: int(args.sample_limit)]
    params = _budgeted_params_from_args(args)
    graph_rows = [build_budgeted_marginal_chain_graph_row(row, params=params) for row in rows]
    traces = [row.get("selection_trace") or {} for row in graph_rows]
    diagnostics = summarize_budgeted_marginal_chain_graph_rows(graph_rows)
    manifest = _manifest(args=args, input_path=input_path, rows=rows, graph_rows=graph_rows, params=params)

    write_jsonl(graph_rows, output_dir / f"chain_graph_{args.split}.jsonl")
    write_jsonl(traces, output_dir / f"selection_trace_{args.split}.jsonl")
    save_json(diagnostics, output_dir / "graph_diagnostics.json")
    save_json(manifest, output_dir / "manifest.json")
    (output_dir / "case_studies.md").write_text(render_budgeted_marginal_case_studies(graph_rows), encoding="utf-8")

    print(f"Wrote {len(graph_rows)} graph rows to {output_dir}")
    print(f"Selector: {budgeted_marginal_chain_selector_name(int(args.min_top_k), int(args.max_top_k))}")
    print(f"Diagnostics: {output_dir / 'graph_diagnostics.json'}")


def _manifest(
    *,
    args: argparse.Namespace,
    input_path: Path,
    rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    params: BudgetedMarginalChainParams,
) -> dict[str, Any]:
    return {
        "graph_version": BUDGETED_MARGINAL_GRAPH_VERSION,
        "selector_name": budgeted_marginal_chain_selector_name(int(args.min_top_k), int(args.max_top_k)),
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
            "target_coverage": float(args.target_coverage),
            "stop_gain_threshold": float(args.stop_gain_threshold),
            "insufficient_gain_threshold": float(args.insufficient_gain_threshold),
            "objective_weights": params.objective_weights.__dict__,
        },
        "n_input_rows": len(rows),
        "n_graph_rows": len(graph_rows),
    }


def _budgeted_params_from_args(args: argparse.Namespace) -> BudgetedMarginalChainParams:
    objective_weights = BudgetedMarginalObjectiveWeights(
        coverage=float(args.objective_coverage),
        map_quality=float(args.objective_map_quality),
        base_score=float(args.objective_base_score),
        key_span=float(args.objective_key_span),
        complements=float(args.objective_complements),
        corroborates=float(args.objective_corroborates),
        conditional_tension=float(args.objective_conditional_tension),
        bridge_context=float(args.objective_bridge_context),
        duplicate_repeat=float(args.objective_duplicate_repeat),
        background_or_irrelevant=float(args.objective_background_or_irrelevant),
        same_source_excess_after_two=float(args.objective_same_source_excess_after_two),
        length=float(args.objective_length),
    )
    return BudgetedMarginalChainParams(
        candidate_top_n=int(args.candidate_top_n),
        min_top_k=int(args.min_top_k),
        max_top_k=int(args.max_top_k),
        chunk_mmr_fingerprint=str(args.chunk_mmr_fingerprint or ""),
        target_coverage=float(args.target_coverage),
        stop_gain_threshold=float(args.stop_gain_threshold),
        insufficient_gain_threshold=float(args.insufficient_gain_threshold),
        objective_weights=objective_weights,
    )


if __name__ == "__main__":
    main()
