#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.atom_anchored_qec import (
    ADAPTIVE_POLICY,
    GRAPH_VERSION,
    AtomAnchoredQECParams,
    atom_anchored_qec_selector_name,
    build_atom_anchored_qec_trace_row,
    summarize_atom_anchored_qec_traces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AA-QEC Stage 1 trace artifacts.")
    parser.add_argument("--input", required=True, help="Input v0.7 selection_trace_*.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--min-chain-steps", type=int, default=5)
    parser.add_argument("--max-chain-steps", type=int, default=10)
    parser.add_argument("--cue-policy", default="qd_prefer")
    parser.add_argument("--candidate-scope", default="selected")
    parser.add_argument("--selection-policy", default="keep_all_reorder")
    parser.add_argument("--source-selector-name", default="v0_7_budgeted_marginal_chain_adaptive5_10")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = AtomAnchoredQECParams(
        candidate_top_n=int(args.candidate_top_n),
        min_chain_steps=int(args.min_chain_steps),
        max_chain_steps=int(args.max_chain_steps),
        cue_policy=str(args.cue_policy),
        candidate_scope=str(args.candidate_scope),
        selection_policy=str(args.selection_policy),
        source_selector_name=str(args.source_selector_name),
        random_seed=int(args.random_seed),
    )
    rows = _read_jsonl(input_path, sample_limit=int(args.sample_limit))
    traces = [build_atom_anchored_qec_trace_row(row, params=params) for row in rows]
    diagnostics = summarize_atom_anchored_qec_traces(traces)
    manifest = _manifest(args=args, input_path=input_path, params=params, n_input_rows=len(rows), n_trace_rows=len(traces))

    _write_jsonl(output_dir / f"chain_graph_{args.split}.jsonl", traces)
    _write_jsonl(output_dir / f"selection_trace_{args.split}.jsonl", traces)
    _write_json(output_dir / "graph_diagnostics.json", diagnostics)
    _write_json(output_dir / "manifest.json", manifest)

    print(f"Wrote {len(traces)} AA-QEC traces to {output_dir}")
    print(f"Selector: {atom_anchored_qec_selector_name(params)}")
    print(f"Diagnostics: {output_dir / 'graph_diagnostics.json'}")


def _read_jsonl(path: Path, *, sample_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if sample_limit > 0 and len(rows) >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows read from {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(
    *,
    args: argparse.Namespace,
    input_path: Path,
    params: AtomAnchoredQECParams,
    n_input_rows: int,
    n_trace_rows: int,
) -> dict[str, Any]:
    return {
        "graph_version": GRAPH_VERSION,
        "adaptive_policy": ADAPTIVE_POLICY,
        "selector_name": atom_anchored_qec_selector_name(params),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "params": {
            "candidate_top_n": int(params.candidate_top_n),
            "min_chain_steps": int(params.min_chain_steps),
            "max_chain_steps": int(params.max_chain_steps),
            "cue_policy": str(params.cue_policy),
            "candidate_scope": str(params.candidate_scope),
            "selection_policy": str(params.selection_policy),
            "source_selector_name": str(params.source_selector_name),
            "random_seed": int(params.random_seed),
        },
        "n_input_rows": int(n_input_rows),
        "n_trace_rows": int(n_trace_rows),
    }


if __name__ == "__main__":
    main()
