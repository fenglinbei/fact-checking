#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.minimal_resolving_chain import (  # noqa: E402
    MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
    MREC_SELECTION_POLICY_LEARNED_MARGINAL_REWARD,
    MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY,
    MREC_SELECTION_POLICY_TRANSITION_V0_1,
    MREC_SELECTOR_NAME,
    MRECSelectorParams,
    build_mrec_trace_row,
)
from fact_checking.selectors.mrec_learned_marginal import (  # noqa: E402
    learned_marginal_weight_fingerprint,
)
from fact_checking.selectors.mrec_schema import (  # noqa: E402
    MREC_TRACE_VERSION,
    summarize_mrec_trace,
    validate_mrec_trace,
)


MREC_ADAPTIVE_POLICY = "minimal_resolving_chain_v0_1"
MREC_LEARNED_MARGINAL_ADAPTIVE_POLICY = "learned_marginal_proxy_v0_2"
MREC_LEARNED_MARGINAL_REWARD_ADAPTIVE_POLICY = "learned_marginal_reward_v0_2"
MREC_MAP_QUALITY_GREEDY_ADAPTIVE_POLICY = "map_quality_greedy_v0_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MREC selector trace artifacts.")
    parser.add_argument("--input", required=True, help="Input selection_trace_*.jsonl with candidate_pool.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--target-resolved-rate", type=float, default=0.80)
    parser.add_argument("--continue-after-target-for-contrast", action="store_true")
    parser.add_argument("--post-target-fill-policy", default="contrast_only")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--cue-policy", default="atom_proposition", choices=["atom_proposition", "atom_query", "legacy_route_prefer"])
    parser.add_argument("--selector-name", default=MREC_SELECTOR_NAME)
    parser.add_argument(
        "--selection-policy",
        default=MREC_SELECTION_POLICY_TRANSITION_V0_1,
        choices=[
            MREC_SELECTION_POLICY_TRANSITION_V0_1,
            MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
            MREC_SELECTION_POLICY_LEARNED_MARGINAL_REWARD,
            MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY,
        ],
    )
    parser.add_argument("--weight-file", default="")
    parser.add_argument("--stop-threshold", type=float, default=0.0)
    parser.add_argument("--source-selector-name", default="v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.selection_policy) in {
        MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
        MREC_SELECTION_POLICY_LEARNED_MARGINAL_REWARD,
    } and not str(args.weight_file or ""):
        raise SystemExit("--weight-file is required when --selection-policy learned_marginal_proxy/reward is used.")

    params = MRECSelectorParams(
        candidate_top_n=int(args.candidate_top_n),
        max_steps=int(args.max_steps),
        min_steps=int(args.min_steps),
        token_budget=int(args.token_budget) if int(args.token_budget) > 0 else None,
        target_resolved_rate=float(args.target_resolved_rate),
        continue_after_target_for_contrast=bool(args.continue_after_target_for_contrast),
        post_target_fill_policy=str(args.post_target_fill_policy),
        allow_fallback=not bool(args.disable_fallback),
        cue_policy=str(args.cue_policy),
        selector_name=str(args.selector_name),
        selection_policy=str(args.selection_policy),
        weight_file=str(args.weight_file or ""),
        stop_threshold=float(args.stop_threshold),
    )
    rows = _read_jsonl(input_path, sample_limit=int(args.sample_limit))
    traces = [_build_trace(row, params=params, source_selector_name=str(args.source_selector_name)) for row in rows]
    manifest = _manifest(
        args=args,
        input_path=input_path,
        params=params,
        n_input_rows=len(rows),
        n_trace_rows=len(traces),
    )
    diagnostics = _summarize_traces(traces)

    _write_jsonl(output_dir / f"mrec_trace_{args.split}.jsonl", traces)
    _write_jsonl(output_dir / f"selection_trace_{args.split}.jsonl", traces)
    _write_json(output_dir / "mrec_diagnostics.json", diagnostics)
    _write_json(output_dir / f"mrec_diagnostics_{args.split}.json", diagnostics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / f"manifest_{args.split}.json", manifest)

    print(f"Wrote {len(traces)} MREC traces to {output_dir}")
    print(f"Selector: {params.selector_name}")
    print(f"Diagnostics: {output_dir / 'mrec_diagnostics.json'}")
    return 0


def _build_trace(
    row: Mapping[str, Any],
    *,
    params: MRECSelectorParams,
    source_selector_name: str,
) -> dict[str, Any]:
    trace = build_mrec_trace_row(row, params=params)
    source_selector = str(row.get("selector_name") or source_selector_name)
    fingerprint = _row_fingerprint(row)
    metadata = dict(row.get("candidate_pool_metadata") or {})
    adaptive_policy = _adaptive_policy_for_params(params)
    metadata.update(
        {
            "selector_name": str(params.selector_name),
            "graph_version": MREC_TRACE_VERSION,
            "adaptive_policy": adaptive_policy,
            "source_selector_name": source_selector,
        }
    )
    if fingerprint:
        metadata["chunk_mmr_fingerprint"] = fingerprint
        trace["fingerprint"] = fingerprint

    trace["candidate_pool_metadata"] = metadata
    trace["adaptive_policy"] = adaptive_policy
    trace["source_selector_name"] = source_selector
    for key in ("candidate_scores", "oracle_ordered_indices", "evidence_map"):
        if key in row and key not in trace:
            trace[key] = row[key]

    errors = validate_mrec_trace(trace)
    if errors:
        event_id = str(trace.get("event_id") or row.get("event_id") or "")
        raise ValueError(f"MREC validation failed for event_id={event_id}: {errors}")
    return trace


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
    params: MRECSelectorParams,
    n_input_rows: int,
    n_trace_rows: int,
) -> dict[str, Any]:
    weight_fingerprint = _weight_fingerprint_for_params(params)
    return {
        "graph_version": MREC_TRACE_VERSION,
        "adaptive_policy": _adaptive_policy_for_params(params),
        "selector_name": str(params.selector_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "params": {
            "candidate_top_n": int(params.candidate_top_n),
            "max_steps": int(params.max_steps),
            "min_steps": int(params.min_steps),
            "token_budget": params.token_budget,
            "target_resolved_rate": float(params.target_resolved_rate),
            "continue_after_target_for_contrast": bool(params.continue_after_target_for_contrast),
            "post_target_fill_policy": str(params.post_target_fill_policy),
            "allow_fallback": bool(params.allow_fallback),
            "cue_policy": str(params.cue_policy),
            "selection_policy": str(params.selection_policy),
            "weight_file": str(params.weight_file or ""),
            "stop_threshold": float(params.stop_threshold),
        },
        "weight_fingerprint": weight_fingerprint,
        "source_selector_name": str(args.source_selector_name),
        "n_input_rows": int(n_input_rows),
        "n_trace_rows": int(n_trace_rows),
    }


def _summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [summarize_mrec_trace(row) for row in rows]
    step_counts = [int(summary.get("step_count", 0)) for summary in summaries]
    resolved_rates = [
        float((row.get("mrec_diagnostics") or {}).get("resolved_atom_rate", 0.0))
        for row in rows
    ]
    return {
        "n_rows": len(rows),
        "mrec_trace_version": MREC_TRACE_VERSION,
        "selector_names": _counts(str(row.get("selector_name") or "") for row in rows),
        "selection_policies": _counts(str((row.get("mrec_diagnostics") or {}).get("selection_policy") or "") for row in rows),
        "step_count": _numeric_summary(step_counts),
        "resolved_atom_rate": _numeric_summary(resolved_rates),
        "operation_counts": _sum_counter(summary.get("operation_counts") for summary in summaries),
        "state_after_counts": _sum_counter(summary.get("state_after_counts") for summary in summaries),
        "cue_source_counts": _sum_counter(summary.get("cue_source_counts") for summary in summaries),
    }


def _adaptive_policy_for_params(params: MRECSelectorParams) -> str:
    if str(params.selection_policy) == MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY:
        return MREC_MAP_QUALITY_GREEDY_ADAPTIVE_POLICY
    if str(params.selection_policy) == MREC_SELECTION_POLICY_LEARNED_MARGINAL_REWARD:
        return MREC_LEARNED_MARGINAL_REWARD_ADAPTIVE_POLICY
    if str(params.selection_policy) == MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY:
        return MREC_LEARNED_MARGINAL_ADAPTIVE_POLICY
    return MREC_ADAPTIVE_POLICY


def _weight_fingerprint_for_params(params: MRECSelectorParams) -> str:
    if str(params.selection_policy) not in {
        MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
        MREC_SELECTION_POLICY_LEARNED_MARGINAL_REWARD,
    }:
        return ""
    return learned_marginal_weight_fingerprint(params.weight_file or None)


def _row_fingerprint(row: Mapping[str, Any]) -> str:
    metadata = row.get("candidate_pool_metadata") or {}
    values = [
        row.get("chunk_mmr_fingerprint"),
        row.get("fingerprint"),
        metadata.get("chunk_mmr_fingerprint") if isinstance(metadata, Mapping) else None,
        metadata.get("fingerprint") if isinstance(metadata, Mapping) else None,
    ]
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def _sum_counter(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key, value in item.items():
            if not key:
                continue
            out[str(key)] = out.get(str(key), 0) + int(value)
    return out


def _numeric_summary(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    return {
        "count": float(count),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
    }


if __name__ == "__main__":
    raise SystemExit(main())
