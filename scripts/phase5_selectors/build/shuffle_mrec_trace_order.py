#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.mrec_schema import (  # noqa: E402
    MREC_TRACE_VERSION,
    mrec_steps_to_compat_chain_steps,
    validate_mrec_trace,
)


ABLATION_GRAPH_VERSION = "selector_mechanism_ablation_v0"
TRACE_SHUFFLE_SELECTION_POLICY = "learned_marginal_proxy_trace_shuffle"
DEFAULT_SELECTOR_NAME = "selector_mech_s6_learned_marginal_proxy_trace_shuffle"
DEFAULT_ADAPTIVE_POLICY = "learned_marginal_proxy_trace_shuffle_v0_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle selected MREC trace order while preserving evidence identity.")
    parser.add_argument("--input", required=True, help="Input learned-marginal selection_trace_*.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--selector-name", default=DEFAULT_SELECTOR_NAME)
    parser.add_argument("--adaptive-policy", default=DEFAULT_ADAPTIVE_POLICY)
    parser.add_argument("--source-selector-name", default="mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(input_path, sample_limit=int(args.sample_limit))
    traces = [
        _shuffle_trace(
            row,
            row_index=idx,
            selector_name=str(args.selector_name),
            adaptive_policy=str(args.adaptive_policy),
            source_selector_name=str(args.source_selector_name),
            seed=int(args.seed),
        )
        for idx, row in enumerate(rows)
    ]
    diagnostics = _summarize_traces(traces)
    manifest = _manifest(
        args=args,
        input_path=input_path,
        n_input_rows=len(rows),
        n_trace_rows=len(traces),
        diagnostics=diagnostics,
    )

    _write_jsonl(output_dir / f"mrec_trace_{args.split}.jsonl", traces)
    _write_jsonl(output_dir / f"selection_trace_{args.split}.jsonl", traces)
    _write_json(output_dir / "mrec_diagnostics.json", diagnostics)
    _write_json(output_dir / f"mrec_diagnostics_{args.split}.json", diagnostics)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / f"manifest_{args.split}.json", manifest)

    print(f"Wrote {len(traces)} shuffled MREC traces to {output_dir}")
    print(f"Selector: {args.selector_name}")
    print(f"Shuffle seed: {int(args.seed)}")
    return 0


def _shuffle_trace(
    row: Mapping[str, Any],
    *,
    row_index: int,
    selector_name: str,
    adaptive_policy: str,
    source_selector_name: str,
    seed: int,
) -> dict[str, Any]:
    out = dict(row)
    event_id = str(row.get("event_id") or f"row-{row_index}")
    source_selector = str(row.get("selector_name") or row.get("mrec_selector_name") or source_selector_name)
    candidate_pool = [dict(candidate) for candidate in row.get("candidate_pool") or [] if isinstance(candidate, Mapping)]
    source_items = _selected_items(row, candidate_pool=candidate_pool)
    permutation = _stable_permutation(len(source_items), seed=seed, event_id=event_id)
    shuffled_items = [source_items[idx] for idx in permutation]

    selected_indices = [int(item["candidate_idx"]) for item in shuffled_items]
    selected_candidates = [
        _candidate_for_index(candidate_pool, idx, fallback=item.get("selected_candidate"))
        for idx, item in zip(selected_indices, shuffled_items)
    ]
    mrec_steps = _renumber_steps([item["mrec_step"] for item in shuffled_items if isinstance(item.get("mrec_step"), Mapping)])
    compat_steps = mrec_steps_to_compat_chain_steps(mrec_steps)
    selected_evidence_ids = [
        str(step.get("evidence_id") or candidate.get("evidence_id") or "")
        for step, candidate in zip(mrec_steps, selected_candidates)
    ]
    selected_keys = [str(candidate.get("candidate_key") or "") for candidate in selected_candidates]
    original_ids = _source_selected_ids(row)
    order_changed = selected_evidence_ids != original_ids
    selected_set_preserved = sorted(selected_evidence_ids) == sorted(original_ids)

    metadata = dict(row.get("candidate_pool_metadata") or {})
    metadata.update(
        {
            "selector_name": selector_name,
            "graph_version": ABLATION_GRAPH_VERSION,
            "adaptive_policy": adaptive_policy,
            "shuffle_seed": int(seed),
            "shuffle_source_selector_name": source_selector,
            "shuffle_preserves_selected_set": bool(selected_set_preserved),
            "shuffle_preserves_candidate_pool": True,
        }
    )
    diagnostics = dict(row.get("mrec_diagnostics") or {})
    diagnostics.update(
        {
            "selection_policy": TRACE_SHUFFLE_SELECTION_POLICY,
            "shuffle_seed": int(seed),
            "shuffle_source_selector_name": source_selector,
            "shuffle_order_changed": bool(order_changed),
            "shuffle_preserves_selected_set": bool(selected_set_preserved),
            "step_count": len(mrec_steps),
        }
    )

    out.update(
        {
            "mrec_trace_version": MREC_TRACE_VERSION,
            "mrec_selector_name": selector_name,
            "selector_name": selector_name,
            "graph_version": ABLATION_GRAPH_VERSION,
            "adaptive_policy": adaptive_policy,
            "source_selector_name": source_selector,
            "shuffle_source_selector_name": source_selector,
            "shuffle_seed": int(seed),
            "candidate_pool": candidate_pool,
            "candidate_pool_metadata": metadata,
            "selected_indices": selected_indices,
            "selector_ordered_indices": selected_indices,
            "selected_candidates": selected_candidates,
            "selected_evidence_ids": selected_evidence_ids,
            "selected_keys": selected_keys,
            "mrec_steps": mrec_steps,
            "compat_chain_steps": compat_steps,
            "chain_steps": compat_steps,
            "mrec_diagnostics": diagnostics,
        }
    )

    errors = validate_mrec_trace(out)
    if errors:
        raise ValueError(f"Shuffled MREC trace validation failed for event_id={event_id}: {errors}")
    return out


def _selected_items(row: Mapping[str, Any], *, candidate_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = [dict(step) for step in row.get("mrec_steps") or [] if isinstance(step, Mapping)]
    selected_indices = _int_list(row.get("selected_indices"))
    selected_candidates = [
        dict(candidate)
        for candidate in row.get("selected_candidates") or []
        if isinstance(candidate, Mapping)
    ]
    if not steps:
        steps = [
            _step_from_candidate(_candidate_for_index(candidate_pool, idx, fallback=None), position=pos, candidate_idx=idx)
            for pos, idx in enumerate(selected_indices)
        ]

    items: list[dict[str, Any]] = []
    for pos, step in enumerate(steps):
        fallback_idx = selected_indices[pos] if pos < len(selected_indices) else int(step.get("candidate_idx") or pos)
        candidate_idx = _int_or_default(step.get("selector_candidate_idx"), fallback_idx)
        selected_candidate = (
            selected_candidates[pos]
            if pos < len(selected_candidates)
            else _candidate_for_index(candidate_pool, candidate_idx, fallback=None)
        )
        items.append(
            {
                "candidate_idx": candidate_idx,
                "mrec_step": step,
                "selected_candidate": selected_candidate,
            }
        )
    return items


def _renumber_steps(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cumulative_token_cost = 0
    for idx, raw_step in enumerate(steps, start=1):
        step = dict(raw_step)
        step["step"] = idx
        token_cost = _int_or_default(step.get("token_cost"), 0)
        cumulative_token_cost += token_cost
        trace_state = step.get("trace_state")
        if isinstance(trace_state, Mapping):
            updated_state = dict(trace_state)
            updated_state["selected_count"] = idx
            updated_state["cumulative_token_cost"] = cumulative_token_cost
            step["trace_state"] = updated_state
        out.append(step)
    return out


def _step_from_candidate(candidate: Mapping[str, Any], *, position: int, candidate_idx: int) -> dict[str, Any]:
    return {
        "step": int(position) + 1,
        "operation": "FALLBACK",
        "atom_id": "",
        "atom_text": "",
        "state_before": "U",
        "state_after": "U",
        "cue_text": str(candidate.get("cue_text") or "Verify the main factual claim."),
        "cue_source": "fallback",
        "evidence_id": str(candidate.get("evidence_id") or candidate.get("candidate_uid") or ""),
        "candidate_idx": int(candidate.get("candidate_idx") or candidate_idx),
        "selector_candidate_idx": int(candidate_idx),
        "evidence_text": str(candidate.get("text") or candidate.get("evidence_text") or ""),
        "covered_atom_ids": [],
        "relation": "unknown",
        "directness": "unknown",
        "map_confidence": None,
        "evidence_map_quality_score": None,
        "token_cost": None,
        "transition_reason": "source trace did not include mrec_steps",
    }


def _candidate_for_index(
    candidate_pool: list[dict[str, Any]],
    idx: int,
    *,
    fallback: Any,
) -> dict[str, Any]:
    if 0 <= int(idx) < len(candidate_pool):
        return dict(candidate_pool[int(idx)])
    if isinstance(fallback, Mapping):
        return dict(fallback)
    return {}


def _stable_permutation(length: int, *, seed: int, event_id: str) -> list[int]:
    permutation = list(range(length))
    if length <= 1:
        return permutation
    digest = hashlib.sha256(f"{int(seed)}:{event_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    rng.shuffle(permutation)
    if permutation == list(range(length)):
        permutation = permutation[1:] + permutation[:1]
    return permutation


def _source_selected_ids(row: Mapping[str, Any]) -> list[str]:
    ids = [str(value) for value in row.get("selected_evidence_ids") or []]
    if ids:
        return ids
    return [
        str(step.get("evidence_id") or "")
        for step in row.get("mrec_steps") or []
        if isinstance(step, Mapping)
    ]


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
    n_input_rows: int,
    n_trace_rows: int,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "graph_version": ABLATION_GRAPH_VERSION,
        "mrec_trace_version": MREC_TRACE_VERSION,
        "adaptive_policy": str(args.adaptive_policy),
        "selector_name": str(args.selector_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "shuffle_seed": int(args.seed),
        "shuffle_source_selector_name": str(args.source_selector_name),
        "source_selector_name": str(args.source_selector_name),
        "selection_policy": TRACE_SHUFFLE_SELECTION_POLICY,
        "n_input_rows": int(n_input_rows),
        "n_trace_rows": int(n_trace_rows),
        "diagnostics": dict(diagnostics),
    }


def _summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    step_counts = [len(row.get("mrec_steps") or []) for row in rows]
    order_changed = [
        bool((row.get("mrec_diagnostics") or {}).get("shuffle_order_changed"))
        for row in rows
    ]
    selected_set_preserved = [
        bool((row.get("mrec_diagnostics") or {}).get("shuffle_preserves_selected_set"))
        for row in rows
    ]
    return {
        "n_rows": len(rows),
        "graph_version": ABLATION_GRAPH_VERSION,
        "mrec_trace_version": MREC_TRACE_VERSION,
        "selector_names": _counts(str(row.get("selector_name") or "") for row in rows),
        "selection_policies": _counts(str((row.get("mrec_diagnostics") or {}).get("selection_policy") or "") for row in rows),
        "step_count": _numeric_summary(step_counts),
        "order_changed_count": sum(1 for value in order_changed if value),
        "selected_set_preserved_count": sum(1 for value in selected_set_preserved if value),
    }


def _counts(values: Any) -> dict[str, int]:
    return dict(Counter(value for value in values if value))


def _numeric_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


if __name__ == "__main__":
    raise SystemExit(main())
