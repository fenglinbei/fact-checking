#!/usr/bin/env python3
"""Freeze verifier-visible ordered prefixes as a compact per-event plan.

The source trace keeps one frozen selector order.  That order may be complete
or an explicitly admissible partial order, but an unranked tail is never
synthesized.  A diagnostic verifier build requests its first K items and may
drop only a suffix to satisfy the prompt budget.  This tool verifies that
realization and writes just the final indices/UIDs needed for a second,
truncation-forbidden rebuild.  It avoids copying the large candidate pool into
one trace file per capacity value.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any, TextIO


SCHEMA_VERSION = "capacity_prefix_selection_plan_v0_1"
PROJECTION_VERSION = "verifier_visible_ordered_prefix_plan_v0_1"
TRACE_ORDER_FIELDS = (
    "selector_full_ordered_indices",
    "selector_available_ordered_indices",
    "display_ordered_indices",
)


class PrefixPlanError(ValueError):
    """Raised when a diagnostic build is not a clean ordered-prefix realization."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", required=True)
    parser.add_argument("--source-build", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--requested-prefix-k", required=True, type=int)
    parser.add_argument(
        "--trace-order-field",
        default="selector_full_ordered_indices",
        choices=TRACE_ORDER_FIELDS,
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.requested_prefix_k <= 0:
        parser.error("--requested-prefix-k must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_plan = Path(args.output_plan)
    summary_path = (
        Path(args.summary)
        if args.summary
        else output_plan.with_name(f"{output_plan.stem}.summary.json")
    )
    if output_plan.exists() and not args.overwrite:
        raise PrefixPlanError(
            f"output plan already exists; pass --overwrite to replace it: {output_plan}"
        )
    if summary_path.exists() and not args.overwrite:
        raise PrefixPlanError(
            f"summary already exists; pass --overwrite to replace it: {summary_path}"
        )
    summary = project_prefix_plan(
        source_trace_path=Path(args.source_trace),
        source_build_path=Path(args.source_build),
        output_plan_path=output_plan,
        requested_prefix_k=int(args.requested_prefix_k),
        trace_order_field=str(args.trace_order_field),
        sample_limit=args.sample_limit,
    )
    _write_json_atomic(summary_path, summary)
    print(
        "Projected {rows} rows at requested K={requested} "
        "(exact={exact}, prompt-dropped={dropped}) to {path}".format(
            rows=summary["row_count"],
            requested=summary["requested_prefix_k"],
            exact=summary["exact_policy_k_event_count"],
            dropped=summary["prompt_tail_drop_event_count"],
            path=output_plan,
        )
    )
    return 0


def project_prefix_plan(
    *,
    source_trace_path: Path,
    source_build_path: Path,
    output_plan_path: Path,
    requested_prefix_k: int,
    trace_order_field: str = "selector_full_ordered_indices",
    sample_limit: int | None = None,
) -> dict[str, Any]:
    if requested_prefix_k <= 0:
        raise PrefixPlanError("requested_prefix_k must be positive")
    if trace_order_field not in TRACE_ORDER_FIELDS:
        raise PrefixPlanError(f"unsupported trace_order_field: {trace_order_field}")
    if not source_trace_path.is_file():
        raise PrefixPlanError(f"missing source trace: {source_trace_path}")
    if not source_build_path.is_file():
        raise PrefixPlanError(f"missing source build: {source_build_path}")

    output_plan_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_plan_path.with_name(
        f".{output_plan_path.name}.tmp.{os.getpid()}"
    )
    row_count = 0
    exact_count = 0
    candidate_exhausted_count = 0
    drop_event_count = 0
    drop_evidence_count = 0
    realized_counts: list[int] = []
    event_ids: set[str] = set()
    event_hasher = hashlib.sha256()
    promoted = False
    try:
        with source_trace_path.open(encoding="utf-8") as trace_handle, source_build_path.open(
            encoding="utf-8"
        ) as build_handle, temp_path.open("w", encoding="utf-8") as output_handle:
            trace_rows = _iter_jsonl(trace_handle, artifact=str(source_trace_path))
            build_rows = _iter_jsonl(build_handle, artifact=str(source_build_path))
            while True:
                if sample_limit is not None and row_count >= sample_limit:
                    break
                trace_row = next(trace_rows, None)
                build_row = next(build_rows, None)
                if trace_row is None or build_row is None:
                    if trace_row is not None or build_row is not None:
                        raise PrefixPlanError("source trace/build row counts differ")
                    break
                plan = project_prefix_row(
                    trace_row=trace_row,
                    build_row=build_row,
                    requested_prefix_k=requested_prefix_k,
                    trace_order_field=trace_order_field,
                )
                event_id = str(plan["event_id"])
                if event_id in event_ids:
                    raise PrefixPlanError(f"duplicate event_id={event_id!r}")
                event_ids.add(event_id)
                event_hasher.update(event_id.encode("utf-8"))
                event_hasher.update(b"\0")
                row_count += 1
                realized_counts.append(int(plan["prompt_feasible_prefix_k"]))
                exact_count += int(bool(plan["exact_policy_k"]))
                candidate_exhausted_count += int(bool(plan["candidate_exhausted"]))
                dropped = int(plan["prompt_tail_drop_count"])
                if dropped > 0:
                    drop_event_count += 1
                    drop_evidence_count += dropped
                output_handle.write(
                    json.dumps(plan, ensure_ascii=False, sort_keys=True) + "\n"
                )
        if row_count <= 0:
            raise PrefixPlanError("projection produced no rows")
        temp_path.replace(output_plan_path)
        promoted = True
    finally:
        if not promoted:
            temp_path.unlink(missing_ok=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "projection_version": PROJECTION_VERSION,
        "source_trace": str(source_trace_path),
        "source_trace_sha256": _sha256_file(source_trace_path),
        "source_build": str(source_build_path),
        "source_build_sha256": _sha256_file(source_build_path),
        "output_plan": str(output_plan_path),
        "output_plan_sha256": _sha256_file(output_plan_path),
        "requested_prefix_k": int(requested_prefix_k),
        "trace_order_field": trace_order_field,
        "sample_limit": sample_limit,
        "row_count": row_count,
        "event_id_sequence_sha256": event_hasher.hexdigest(),
        "exact_policy_k_event_count": exact_count,
        "exact_policy_k_event_rate": exact_count / row_count,
        "candidate_exhausted_event_count": candidate_exhausted_count,
        "prompt_tail_drop_event_count": drop_event_count,
        "prompt_tail_drop_event_rate": drop_event_count / row_count,
        "prompt_tail_drop_evidence_count": drop_evidence_count,
        "prompt_feasible_prefix_k": _numeric_summary(realized_counts),
    }


def project_prefix_row(
    *,
    trace_row: Mapping[str, Any],
    build_row: Mapping[str, Any],
    requested_prefix_k: int,
    trace_order_field: str,
) -> dict[str, Any]:
    event_id = _event_id(trace_row, "trace")
    if _event_id(build_row, "build") != event_id:
        raise PrefixPlanError(f"{event_id}: trace/build event_id mismatch")
    if build_row.get("evidence_text_truncated") is not False:
        raise PrefixPlanError(
            f"{event_id}: evidence_text_truncated must be exactly false"
        )
    pool = _mapping_list(trace_row.get("candidate_pool"), f"{event_id}: candidate_pool")
    order = _int_list(trace_row.get(trace_order_field), f"{event_id}: {trace_order_field}")
    if not order:
        raise PrefixPlanError(f"{event_id}: configured trace order is empty")
    if len(order) != len(set(order)) or any(idx < 0 or idx >= len(pool) for idx in order):
        raise PrefixPlanError(f"{event_id}: configured trace order is invalid")
    available = order[:requested_prefix_k]
    if not available:
        raise PrefixPlanError(f"{event_id}: no candidate is available at requested K")

    visible_count = _nonnegative_int(build_row.get("evidence_count"), f"{event_id}: evidence_count")
    if visible_count <= 0:
        raise PrefixPlanError(f"{event_id}: empty verifier-visible prefix is forbidden")
    if visible_count > len(available):
        raise PrefixPlanError(f"{event_id}: visible prefix exceeds requested available prefix")
    for field in (
        "evidence_count_before",
        "prompt_evidence_selected_count_before_prompt_truncation",
    ):
        value = build_row.get(field)
        if value is not None and _nonnegative_int(value, f"{event_id}: {field}") != len(
            available
        ):
            raise PrefixPlanError(
                f"{event_id}: {field} does not equal requested available prefix length"
            )

    pool_uids = [_candidate_uid(candidate, event_id) for candidate in pool]
    pool_evidence_ids = [_evidence_id(candidate, event_id) for candidate in pool]
    expected_uids = [pool_uids[idx] for idx in available]
    expected_evidence_ids = [pool_evidence_ids[idx] for idx in available]
    build_candidates = _mapping_list(
        build_row.get("candidates"), f"{event_id}: build candidates"
    )
    if visible_count > len(build_candidates):
        raise PrefixPlanError(f"{event_id}: evidence_count exceeds build candidates")
    visible_uids = [
        _candidate_uid(candidate, event_id) for candidate in build_candidates[:visible_count]
    ]
    visible_evidence_ids = [
        _evidence_id(candidate, event_id)
        for candidate in build_candidates[:visible_count]
    ]
    if visible_uids != expected_uids[:visible_count]:
        raise PrefixPlanError(
            f"{event_id}: verifier-visible evidence is not the requested ordered prefix"
        )
    if visible_evidence_ids != expected_evidence_ids[:visible_count]:
        raise PrefixPlanError(
            f"{event_id}: verifier-visible evidence IDs are not the requested ordered prefix"
        )

    selected_indices = available[:visible_count]
    return {
        "schema_version": SCHEMA_VERSION,
        "projection_version": PROJECTION_VERSION,
        "event_id": event_id,
        "trace_order_field": trace_order_field,
        "requested_prefix_k": int(requested_prefix_k),
        "available_prefix_k": len(available),
        "prompt_feasible_prefix_k": visible_count,
        "selected_indices": list(selected_indices),
        "selected_candidate_uids": list(visible_uids),
        "selected_evidence_ids": list(visible_evidence_ids),
        "exact_policy_k": visible_count == requested_prefix_k,
        "candidate_exhausted": len(available) < requested_prefix_k,
        "prompt_tail_drop_count": len(available) - visible_count,
        "source_build_was_truncated": bool(build_row.get("was_truncated")),
        "source_build_prompt_token_count": _nonnegative_int(
            build_row.get("prompt_token_count"), f"{event_id}: prompt_token_count"
        ),
    }


def _iter_jsonl(handle: TextIO, *, artifact: str) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrefixPlanError(
                f"invalid JSON at {artifact}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PrefixPlanError(f"{artifact}:{line_number} is not an object")
        yield row


def _event_id(row: Mapping[str, Any], context: str) -> str:
    value = str(row.get("event_id") or "").strip()
    if not value:
        raise PrefixPlanError(f"{context} row has no event_id")
    return value


def _candidate_uid(candidate: Mapping[str, Any], event_id: str) -> str:
    value = str(candidate.get("candidate_uid") or "").strip()
    if not value:
        raise PrefixPlanError(f"{event_id}: candidate has no candidate_uid")
    return value


def _evidence_id(candidate: Mapping[str, Any], event_id: str) -> str:
    value = str(candidate.get("evidence_id") or "").strip()
    if not value:
        raise PrefixPlanError(f"{event_id}: candidate has no evidence_id")
    return value


def _mapping_list(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise PrefixPlanError(f"{context} must be an object array")
    return [dict(item) for item in value]


def _int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise PrefixPlanError(f"{context} must be an integer array")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise PrefixPlanError(f"{context} must contain integers")
        out.append(int(item))
    return out


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrefixPlanError(f"{context} must be a non-negative integer")
    return int(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    promoted = False
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
        promoted = True
    finally:
        if not promoted:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
