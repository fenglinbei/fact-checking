#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fact_checking.selectors.evidence_map_selector import (
    build_evidence_map_trace,
    summarize_evidence_map_traces,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_CANDIDATE_FEATURES = (
    "outputs/selectors/evidence_map_selector/v0_5a_val/"
    "candidate_evidence_map_features_val.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/selectors/evidence_map_selector/"
    "v0_5c_val_prompt_evidence_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a virtual oracle_top5 selector trace from v0.5a evidence-map candidate features."
    )
    parser.add_argument("--candidate-features", default=DEFAULT_CANDIDATE_FEATURES)
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_DIR}/traces")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--selector-name", default="oracle_top5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--base-selection-trace",
        default=None,
        help="Optional existing trace to concatenate with oracle_top5 into merged_selection_trace_<split>.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.candidate_features)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise ValueError(f"No candidate feature rows loaded from {args.candidate_features}")

    traces: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    annotation_status: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for row in rows:
        event_id = str(row.get("event_id") or "")
        selected = _oracle_selected_candidates(row, selector_name=str(args.selector_name), top_k=int(args.top_k))
        if not selected:
            skipped["no_oracle_selected_candidates"] += 1
            continue
        trace = build_evidence_map_trace(
            row,
            selected,
            selector_name=str(args.selector_name),
            top_k=int(args.top_k),
            slot_trace=_slot_trace(selected),
        )
        status = _map_annotation_status(row, selected)
        trace["map_annotation_status"] = status
        trace["evidence_map_parse_status"] = str(row.get("evidence_map_parse_status") or "")
        trace["oracle_selected_count"] = len(selected)
        trace["source_candidate_features"] = str(args.candidate_features)
        traces.append(trace)
        annotation_status[status] += 1
        selected_counts[str(len(selected))] += 1
        if len(selected) < int(args.top_k):
            skipped["oracle_selected_lt_top_k"] += 1
            if event_id:
                skipped[f"lt_top_k:{event_id}"] += 1

    if not traces:
        raise ValueError("No oracle_top5 traces produced.")

    trace_path = out_dir / f"oracle_top5_trace_{args.split}.jsonl"
    write_jsonl(traces, trace_path)

    merged_path: Path | None = None
    if args.base_selection_trace:
        base_rows = read_jsonl(args.base_selection_trace)
        merged_path = out_dir / f"merged_selection_trace_{args.split}.jsonl"
        write_jsonl([*traces, *base_rows], merged_path)

    selector_metrics = summarize_evidence_map_traces(traces)
    save_json(selector_metrics, out_dir / "oracle_top5_selector_metrics.json")
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "candidate_features": str(args.candidate_features),
        "base_selection_trace": str(args.base_selection_trace or ""),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "selector_name": str(args.selector_name),
        "top_k": int(args.top_k),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "n_input_rows": len(rows),
        "n_traces": len(traces),
        "selected_count_distribution": dict(selected_counts),
        "map_annotation_status": dict(annotation_status),
        "skipped": dict(skipped),
        "outputs": {
            "oracle_trace": str(trace_path),
            "merged_selection_trace": str(merged_path) if merged_path else None,
            "selector_metrics": str(out_dir / "oracle_top5_selector_metrics.json"),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "oracle_top5_trace_manifest.json")

    print(f"Wrote oracle_top5 trace: {trace_path}")
    if merged_path:
        print(f"Wrote merged trace: {merged_path}")


def _oracle_selected_candidates(row: dict[str, Any], *, selector_name: str, top_k: int) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in row.get("candidates") or [] if bool(candidate.get("oracle_selected"))]
    if not candidates:
        oracle_keys = [str(key) for key in row.get("oracle_ordered_keys") or []]
        by_key = {str(candidate.get("candidate_key") or ""): dict(candidate) for candidate in row.get("candidates") or []}
        candidates = [by_key[key] for key in oracle_keys if key in by_key]
    candidates.sort(key=_oracle_order_key)
    selected = candidates[: int(top_k)]
    for rank, candidate in enumerate(selected, start=1):
        candidate["selector_name"] = selector_name
        candidate["selection_rank"] = rank
        candidate["slot_score"] = float(1.0 / rank)
    return selected


def _oracle_order_key(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        _int_or_default(candidate.get("oracle_step"), 10**9),
        _int_or_default(candidate.get("oracle_selected_rank"), 10**9),
        _int_or_default(candidate.get("original_pool_position"), 10**9),
        str(candidate.get("candidate_key") or ""),
    )


def _slot_trace(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected, start=1):
        out.append(
            {
                "slot": rank,
                "candidate_uid": str(candidate.get("candidate_uid") or ""),
                "candidate_key": str(candidate.get("candidate_key") or ""),
                "oracle_step": _int_or_default(candidate.get("oracle_step"), rank - 1),
                "map_relation": str(candidate.get("map_relation") or ""),
                "map_directness": str(candidate.get("map_directness") or ""),
                "covered_atom_ids": list(candidate.get("covered_atom_ids") or []),
                "map_annotation_status": "ok" if _candidate_has_map(candidate) else "missing",
            }
        )
    return out


def _map_annotation_status(row: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    parse_status = str(row.get("evidence_map_parse_status") or "").strip()
    if selected and all(_candidate_has_map(candidate) for candidate in selected):
        if parse_status == "ok":
            return "ok"
        if parse_status.startswith("fallback"):
            return "fallback"
        return "ok"
    if parse_status.startswith("fallback"):
        return "fallback"
    return "missing"


def _candidate_has_map(candidate: dict[str, Any]) -> bool:
    return all(key in candidate for key in ("map_relation", "map_directness", "covered_atom_ids", "key_spans"))


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


if __name__ == "__main__":
    main()
