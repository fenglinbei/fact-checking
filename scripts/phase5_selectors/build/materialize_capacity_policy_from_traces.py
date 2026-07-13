#!/usr/bin/env python3
"""Extract a per-event capacity policy from validated factorial traces.

Each input trace represents one selector under a fixed controller.  The
selected slate must be an ordered prefix of the configured frozen order
(which may be an admissible partial order); its length is emitted as the policy action consumed by
``sft.capacity_prefix_analysis``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "capacity_policy_from_traces_v0_1"
METADATA_FIELDS = ("factorial_metadata", "factor_metadata", "metadata")
TRACE_ORDER_FIELDS = (
    "selector_available_ordered_indices",
    "selector_full_ordered_indices",
    "display_ordered_indices",
)
TRACE_ORDER_UID_FIELDS = {
    "selector_available_ordered_indices": "selector_available_ordered_candidate_uids",
    "selector_full_ordered_indices": "selector_full_ordered_candidate_uids",
    "display_ordered_indices": "selected_candidate_uids",
}
VERIFIED_STRUCTURE_ONLY_CONTROLLER_CONTRACTS = {
    "ordinal_replay_minmax5_10": (
        "first prefix t>=5 reaching the common exact Kmax=10 ordinal target, else 10"
    ),
}


class CapacityPolicyError(ValueError):
    """Raised when source traces cannot define a valid capacity policy."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-trace",
        action="append",
        required=True,
        metavar="SELECTOR=PATH",
        help="repeat once per selector",
    )
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--expected-controller")
    parser.add_argument(
        "--source-factorial-manifest",
        help=(
            "factorial manifest that binds selector/controller cells to the supplied traces; "
            "required to verify known structure-only controller provenance"
        ),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--trace-order-field",
        default="selector_available_ordered_indices",
        choices=TRACE_ORDER_FIELDS,
    )
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not str(args.policy_id).strip():
        parser.error("--policy-id must not be empty")
    if args.min_k <= 0:
        parser.error("--min-k must be positive")
    if args.max_k < args.min_k:
        parser.error("--max-k must be at least --min-k")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    try:
        args.selector_traces = _parse_selector_traces(args.selector_trace)
    except CapacityPolicyError as exc:
        parser.error(str(exc))
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = materialize_capacity_policy(
        selector_traces=args.selector_traces,
        policy_id=str(args.policy_id),
        output_policy=Path(args.output_policy),
        expected_controller=args.expected_controller,
        source_factorial_manifest=(
            Path(args.source_factorial_manifest)
            if args.source_factorial_manifest
            else None
        ),
        split=str(args.split),
        trace_order_field=str(args.trace_order_field),
        min_k=int(args.min_k),
        max_k=int(args.max_k),
        sample_limit=args.sample_limit,
        overwrite=bool(args.overwrite),
    )
    print(
        "Materialized {assignments} assignments for {selectors} selectors "
        "to {path}".format(
            assignments=summary["assignment_count"],
            selectors=summary["selector_count"],
            path=summary["output_policy"],
        )
    )
    return 0


def materialize_capacity_policy(
    *,
    selector_traces: Mapping[str, Path | str],
    policy_id: str,
    output_policy: Path,
    expected_controller: str | None = None,
    source_factorial_manifest: Path | None = None,
    split: str = "val",
    trace_order_field: str = "selector_available_ordered_indices",
    min_k: int = 1,
    max_k: int = 10,
    sample_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate selector traces and atomically materialize policy + summary."""

    policy_id = str(policy_id).strip()
    split = str(split).strip()
    if not policy_id:
        raise CapacityPolicyError("policy_id must not be empty")
    if not split:
        raise CapacityPolicyError("split must not be empty")
    if not selector_traces:
        raise CapacityPolicyError("at least one selector trace is required")
    if min_k <= 0 or max_k < min_k:
        raise CapacityPolicyError(f"invalid K range: [{min_k}, {max_k}]")
    if sample_limit is not None and sample_limit <= 0:
        raise CapacityPolicyError("sample_limit must be positive")
    if trace_order_field not in TRACE_ORDER_FIELDS:
        raise CapacityPolicyError(f"unsupported trace_order_field: {trace_order_field}")

    output_policy = Path(output_policy).resolve()
    summary_path = output_policy.with_name(f"{output_policy.stem}.summary.json")
    for path in (output_policy, summary_path):
        if path.exists() and not overwrite:
            raise CapacityPolicyError(
                f"output already exists; pass --overwrite to replace it: {path}"
            )

    normalized_sources: list[tuple[str, Path]] = []
    seen_selectors: set[str] = set()
    for raw_selector, raw_path in selector_traces.items():
        selector = str(raw_selector).strip()
        if not selector:
            raise CapacityPolicyError("selector level must not be empty")
        if selector in seen_selectors:
            raise CapacityPolicyError(f"duplicate selector level: {selector}")
        seen_selectors.add(selector)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise CapacityPolicyError(f"missing selector trace: {path}")
        normalized_sources.append((selector, path))

    factorial_contract = _factorial_controller_contract(
        manifest_path=(
            Path(source_factorial_manifest).resolve()
            if source_factorial_manifest is not None
            else None
        ),
        selector_sources=normalized_sources,
        expected_controller=expected_controller,
        split=split,
    )

    assignments: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    reference_events: list[str] | None = None
    total_distribution: Counter[int] = Counter()
    for selector, path in normalized_sources:
        rows, event_ids, distribution = _extract_trace_assignments(
            path=path,
            selector=selector,
            policy_id=policy_id,
            expected_controller=expected_controller,
            trace_order_field=trace_order_field,
            min_k=min_k,
            max_k=max_k,
            sample_limit=sample_limit,
            verify_known_structure_only_controller=bool(
                factorial_contract is not None
                and expected_controller in VERIFIED_STRUCTURE_ONLY_CONTROLLER_CONTRACTS
            ),
        )
        if reference_events is None:
            reference_events = event_ids
        elif event_ids != reference_events:
            mismatch = _first_sequence_mismatch(reference_events, event_ids)
            raise CapacityPolicyError(
                f"selector={selector}: event sequence differs from the first selector "
                f"at position {mismatch}"
            )
        assignments.extend(rows)
        total_distribution.update(distribution)
        source_summaries.append(
            {
                "selector_level": selector,
                "path": str(path),
                "sha256": _sha256_file(path),
                "event_count": len(event_ids),
                "assignment_count": len(rows),
                "k_distribution": _json_distribution(distribution),
            }
        )

    if reference_events is None or not reference_events:
        raise CapacityPolicyError("selector traces produced no assignments")
    if factorial_contract is not None:
        _validate_factorial_event_counts(
            contract=factorial_contract,
            event_count=len(reference_events),
            sample_limit=sample_limit,
        )

    event_sequence_hash = _event_sequence_sha256(reference_events)
    output_policy.parent.mkdir(parents=True, exist_ok=True)
    policy_temp = _temp_path(output_policy)
    summary_temp = _temp_path(summary_path)
    promoted_policy = False
    promoted_summary = False
    try:
        with policy_temp.open("w", encoding="utf-8") as handle:
            for row in assignments:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        policy_sha256 = _sha256_file(policy_temp)
        provenance = _policy_provenance(
            expected_controller=expected_controller,
            split=split,
            factorial_contract=factorial_contract,
        )
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "policy_id": policy_id,
            "output_policy": str(output_policy),
            "output_policy_sha256": policy_sha256,
            "summary_path": str(summary_path),
            "expected_controller": expected_controller,
            "trace_order_field": trace_order_field,
            "split": split,
            "min_k": int(min_k),
            "max_k": int(max_k),
            "sample_limit": sample_limit,
            "selector_count": len(normalized_sources),
            "event_count": len(reference_events),
            "assignment_count": len(assignments),
            "event_id_sequence_sha256": event_sequence_hash,
            "k_distribution": _json_distribution(total_distribution),
            "sources": source_summaries,
            "factorial_manifest": factorial_contract,
            "provenance": {
                **provenance,
                "expected_controller": expected_controller,
                "trace_order_field": trace_order_field,
            },
        }
        with summary_temp.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        policy_temp.replace(output_policy)
        promoted_policy = True
        summary_temp.replace(summary_path)
        promoted_summary = True
    finally:
        if not promoted_policy:
            policy_temp.unlink(missing_ok=True)
        if not promoted_summary:
            summary_temp.unlink(missing_ok=True)
    return summary


def _factorial_controller_contract(
    *,
    manifest_path: Path | None,
    selector_sources: Sequence[tuple[str, Path]],
    expected_controller: str | None,
    split: str,
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    if expected_controller is None:
        raise CapacityPolicyError(
            "--source-factorial-manifest requires --expected-controller"
        )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "baces_factorial_trace_v0_1"
        or str(manifest.get("split") or "") != split
    ):
        raise CapacityPolicyError(
            f"factorial manifest schema/split mismatch: {manifest_path}"
        )
    controller_contracts = manifest.get("controller_contracts")
    if not isinstance(controller_contracts, Mapping):
        raise CapacityPolicyError(
            f"factorial manifest has no controller contracts: {manifest_path}"
        )
    controller_contract = str(controller_contracts.get(expected_controller) or "")
    if not controller_contract:
        raise CapacityPolicyError(
            f"factorial manifest does not declare controller={expected_controller}"
        )
    verified_contract = VERIFIED_STRUCTURE_ONLY_CONTROLLER_CONTRACTS.get(
        expected_controller
    )
    if verified_contract is not None and controller_contract != verified_contract:
        raise CapacityPolicyError(
            f"known controller contract drift for {expected_controller}: {manifest_path}"
        )
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not all(
        isinstance(cell, Mapping) for cell in raw_cells
    ):
        raise CapacityPolicyError(f"factorial manifest has invalid cells: {manifest_path}")
    bound_cells: list[dict[str, Any]] = []
    for selector, trace_path in selector_sources:
        matches = [
            cell
            for cell in raw_cells
            if str(cell.get("selector_level") or "") == selector
            and str(cell.get("controller_level") or "") == expected_controller
            and cell.get("ready") is True
        ]
        if len(matches) != 1:
            raise CapacityPolicyError(
                f"factorial manifest must contain exactly one ready cell for "
                f"{selector}/{expected_controller}"
            )
        cell = matches[0]
        declared_trace = Path(str(cell.get("trace_file") or ""))
        if not declared_trace.is_absolute():
            declared_trace = manifest_path.parent / declared_trace
        declared_trace = declared_trace.resolve()
        if declared_trace != trace_path or not declared_trace.is_file():
            raise CapacityPolicyError(
                f"factorial manifest trace binding mismatch for selector={selector}"
            )
        trace_sha256 = _sha256_file(trace_path)
        declared_trace_sha256 = str(cell.get("trace_sha256") or "")
        if declared_trace_sha256 and declared_trace_sha256 != trace_sha256:
            raise CapacityPolicyError(
                f"factorial manifest trace SHA mismatch for selector={selector}"
            )
        try:
            row_count = int(cell.get("row_count", -1))
        except (TypeError, ValueError) as exc:
            raise CapacityPolicyError(
                f"factorial manifest cell row_count is invalid for selector={selector}"
            ) from exc
        if row_count <= 0:
            raise CapacityPolicyError(
                f"factorial manifest cell is empty for selector={selector}"
            )
        bound_cells.append(
            {
                "cell_id": str(cell.get("cell_id") or ""),
                "selector_level": selector,
                "controller_level": expected_controller,
                "trace_path": str(trace_path),
                "trace_sha256": trace_sha256,
                "manifest_trace_sha256": declared_trace_sha256 or None,
                "manifest_trace_sha256_verified": bool(declared_trace_sha256),
                "row_count": row_count,
            }
        )
    generator_path = Path(__file__).resolve().with_name(
        "build_baces_factorial_traces.py"
    )
    if not generator_path.is_file():
        raise CapacityPolicyError(
            f"factorial generator implementation is missing: {generator_path}"
        )
    try:
        event_count = int(manifest.get("event_count", -1))
    except (TypeError, ValueError) as exc:
        raise CapacityPolicyError(
            f"factorial manifest event_count is invalid: {manifest_path}"
        ) from exc
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "schema_version": str(manifest.get("schema_version") or ""),
        "factorial_version": manifest.get("factorial_version"),
        "split": split,
        "event_count": event_count,
        "controller_level": expected_controller,
        "controller_contract": controller_contract,
        "source_contract": manifest.get("source_contract"),
        "cells": bound_cells,
        "generator_implementation": {
            "path": str(generator_path),
            "sha256": _sha256_file(generator_path),
        },
    }


def _validate_factorial_event_counts(
    *, contract: dict[str, Any], event_count: int, sample_limit: int | None
) -> None:
    source_event_count = int(contract.get("event_count", -1))
    expected_policy_event_count = (
        source_event_count
        if sample_limit is None
        else min(int(sample_limit), source_event_count)
    )
    if event_count != expected_policy_event_count:
        raise CapacityPolicyError(
            "factorial manifest event_count differs from extracted policy events"
        )
    cells = contract.get("cells")
    if not isinstance(cells, list) or any(
        int(cell.get("row_count", -1)) < event_count
        for cell in cells
        if isinstance(cell, Mapping)
    ) or not all(isinstance(cell, Mapping) for cell in cells):
        raise CapacityPolicyError(
            "factorial manifest cell row_count differs from extracted policy events"
        )
    contract["sample_limit"] = sample_limit
    contract["policy_event_count"] = event_count
    for cell in cells:
        cell["policy_event_count"] = event_count


def _policy_provenance(
    *,
    expected_controller: str | None,
    split: str,
    factorial_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    is_verified = bool(
        factorial_contract is not None
        and expected_controller in VERIFIED_STRUCTURE_ONLY_CONTROLLER_CONTRACTS
        and str(factorial_contract.get("controller_contract") or "")
        == VERIFIED_STRUCTURE_ONLY_CONTROLLER_CONTRACTS[expected_controller]
    )
    if is_verified:
        return {
            "verification_status": "verified_known_structure_only_factorial_controller",
            "policy_family": "frozen_factorial_trace_controller",
            "uses_gold": False,
            "uses_verifier_logits": False,
            "fit_split": None,
            "decision_source_split": split,
            "cross_fit_status": "not_applicable_no_fitting",
            "deployable_without_gold": True,
            "deployable_ex_ante": True,
        }
    return {
        "verification_status": "controller_provenance_unknown",
        "policy_family": "trace_selected_count_extraction",
        "uses_gold": None,
        "uses_verifier_logits": None,
        "fit_split": None,
        "decision_source_split": split,
        "cross_fit_status": "not_applicable_extraction_only",
        "deployable_without_gold": False,
        "deployable_ex_ante": False,
    }


def _extract_trace_assignments(
    *,
    path: Path,
    selector: str,
    policy_id: str,
    expected_controller: str | None,
    trace_order_field: str,
    min_k: int,
    max_k: int,
    sample_limit: int | None,
    verify_known_structure_only_controller: bool,
) -> tuple[list[dict[str, Any]], list[str], Counter[int]]:
    rows: list[dict[str, Any]] = []
    event_ids: list[str] = []
    seen_events: set[str] = set()
    distribution: Counter[int] = Counter()
    for line_number, row in _iter_jsonl(path):
        if sample_limit is not None and len(rows) >= sample_limit:
            break
        context = f"{path}:{line_number}"
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            raise CapacityPolicyError(f"{context}: missing event_id")
        if event_id in seen_events:
            raise CapacityPolicyError(f"{context}: duplicate event_id={event_id!r}")
        seen_events.add(event_id)

        _validate_declared_factor(
            row=row,
            field="factor_selector",
            expected=selector,
            context=context,
        )
        _validate_metadata_factors(
            row=row,
            expected_selector=selector,
            expected_controller=expected_controller,
            context=context,
        )
        if expected_controller is not None:
            _validate_declared_factor(
                row=row,
                field="factor_controller",
                expected=expected_controller,
                context=context,
            )

        ordered_indices = _index_list(
            row.get(trace_order_field),
            field=trace_order_field,
            context=context,
        )
        selected = _index_list(
            row.get("selected_indices"),
            field="selected_indices",
            context=context,
        )
        if len(ordered_indices) != len(set(ordered_indices)):
            raise CapacityPolicyError(
                f"{context}: {trace_order_field} contains duplicates"
            )
        if len(selected) != len(set(selected)):
            raise CapacityPolicyError(f"{context}: selected_indices contains duplicates")
        candidate_pool = row.get("candidate_pool")
        if not isinstance(candidate_pool, list) or not all(
            isinstance(candidate, Mapping) for candidate in candidate_pool
        ):
            raise CapacityPolicyError(f"{context}: candidate_pool must be an object array")
        candidate_uids: list[str] = []
        for candidate_index, candidate in enumerate(candidate_pool):
            candidate_uid = str(candidate.get("candidate_uid") or "").strip()
            if not candidate_uid:
                raise CapacityPolicyError(
                    f"{context}: candidate_pool[{candidate_index}] has no candidate_uid"
                )
            candidate_uids.append(candidate_uid)
        if len(candidate_uids) != len(set(candidate_uids)):
            raise CapacityPolicyError(f"{context}: candidate_pool candidate_uids duplicate")
        if any(index >= len(candidate_pool) for index in ordered_indices + selected):
            raise CapacityPolicyError(
                f"{context}: configured/selected index is outside candidate_pool"
            )
        ordered_uids = [candidate_uids[index] for index in ordered_indices]
        declared_order_uid_field = TRACE_ORDER_UID_FIELDS[trace_order_field]
        declared_order_uids = _uid_list(
            row.get(declared_order_uid_field),
            field=declared_order_uid_field,
            context=context,
        )
        if declared_order_uids != ordered_uids:
            raise CapacityPolicyError(
                f"{context}: {declared_order_uid_field} does not bind {trace_order_field}"
            )
        selected_uids = _uid_list(
            row.get("selected_candidate_uids"),
            field="selected_candidate_uids",
            context=context,
        )
        if selected_uids != [candidate_uids[index] for index in selected]:
            raise CapacityPolicyError(
                f"{context}: selected_candidate_uids does not bind selected_indices"
            )
        if selected != ordered_indices[: len(selected)]:
            raise CapacityPolicyError(
                f"{context}: selected_indices is not a prefix of "
                f"{trace_order_field}"
            )
        if verify_known_structure_only_controller:
            _validate_known_structure_only_controller_trace(
                row=row,
                expected_controller=str(expected_controller),
                ordered_indices=ordered_indices,
                selected_indices=selected,
                candidate_uids=candidate_uids,
                context=context,
            )
        selected_k = len(selected)
        if selected_k < min_k or selected_k > max_k:
            raise CapacityPolicyError(
                f"{context}: selected K={selected_k} is outside [{min_k}, {max_k}]"
            )
        if row.get("selected_count") is not None:
            try:
                selected_count = int(row["selected_count"])
            except (TypeError, ValueError) as exc:
                raise CapacityPolicyError(f"{context}: invalid selected_count") from exc
            if selected_count != selected_k:
                raise CapacityPolicyError(
                    f"{context}: selected_count={selected_count} differs from K={selected_k}"
                )

        rows.append(
            {
                "policy_id": policy_id,
                "event_id": event_id,
                "selector_level": selector,
                "selected_k": selected_k,
            }
        )
        event_ids.append(event_id)
        distribution[selected_k] += 1
    if not rows:
        raise CapacityPolicyError(f"selector trace produced no assignments: {path}")
    return rows, event_ids, distribution


def _validate_known_structure_only_controller_trace(
    *,
    row: Mapping[str, Any],
    expected_controller: str,
    ordered_indices: Sequence[int],
    selected_indices: Sequence[int],
    candidate_uids: Sequence[str],
    context: str,
) -> None:
    if expected_controller != "ordinal_replay_minmax5_10":
        raise CapacityPolicyError(
            f"{context}: no invariant validator for controller={expected_controller}"
        )
    metadata = row.get("factorial_metadata")
    if not isinstance(metadata, Mapping):
        raise CapacityPolicyError(
            f"{context}: verified controller requires factorial_metadata"
        )
    expected_contract = VERIFIED_STRUCTURE_ONLY_CONTROLLER_CONTRACTS[
        expected_controller
    ]
    if (
        str(metadata.get("controller_level") or "") != expected_controller
        or str(metadata.get("controller_contract") or "") != expected_contract
        or int(metadata.get("k_min", -1)) != 5
        or int(metadata.get("k_max", -1)) != 10
        or metadata.get("stored_target_resolved_used") is not False
    ):
        raise CapacityPolicyError(
            f"{context}: ordinal-minmax controller metadata contract drift"
        )
    target_state = metadata.get("common_exact_kmax10_target_state")
    if not isinstance(target_state, list) or not target_state:
        raise CapacityPolicyError(
            f"{context}: ordinal-minmax target state must be a non-empty array"
        )
    raw_steps = row.get("baces_display_steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != len(selected_indices) or not all(
        isinstance(step, Mapping) for step in raw_steps
    ):
        raise CapacityPolicyError(
            f"{context}: ordinal-minmax display steps do not bind selected_indices"
        )
    first_resolved_position: int | None = None
    terminal_state: list[Any] | None = None
    for position, (step, candidate_index) in enumerate(
        zip(raw_steps, selected_indices), start=1
    ):
        state_after = step.get("state_after")
        if not isinstance(state_after, list):
            raise CapacityPolicyError(
                f"{context}: display step {position} has no ordinal state"
            )
        if (
            int(step.get("position", -1)) != position
            or int(step.get("candidate_idx", -1)) != candidate_index
            or str(step.get("candidate_uid") or "")
            != candidate_uids[candidate_index]
        ):
            raise CapacityPolicyError(
                f"{context}: display step {position} does not bind the selected candidate"
            )
        terminal_state = list(state_after)
        if position >= 5 and state_after == target_state and first_resolved_position is None:
            first_resolved_position = position
    available_cap = min(10, len(ordered_indices))
    if first_resolved_position is not None:
        expected_selected_count = first_resolved_position
        expected_stop_reason = "ordinal_target_reached"
    else:
        expected_selected_count = available_cap
        expected_stop_reason = "max10" if len(ordered_indices) >= 10 else "pool_exhausted"
    if (
        len(selected_indices) != expected_selected_count
        or str(metadata.get("controller_stop_reason") or "") != expected_stop_reason
    ):
        raise CapacityPolicyError(
            f"{context}: selected prefix violates ordinal-minmax first-hit/limit rule"
        )
    display = row.get("baces_display")
    if (
        not isinstance(display, Mapping)
        or int(display.get("length", -1)) != len(selected_indices)
        or display.get("terminal_state") != terminal_state
    ):
        raise CapacityPolicyError(
            f"{context}: baces_display terminal state/length drift"
        )


def _validate_declared_factor(
    *, row: Mapping[str, Any], field: str, expected: str, context: str
) -> None:
    if field not in row or row[field] is None:
        return
    actual = str(row[field])
    if actual != expected:
        raise CapacityPolicyError(
            f"{context}: {field}={actual!r} does not match {expected!r}"
        )


def _validate_metadata_factors(
    *,
    row: Mapping[str, Any],
    expected_selector: str,
    expected_controller: str | None,
    context: str,
) -> None:
    for metadata_field in METADATA_FIELDS:
        if metadata_field not in row or row[metadata_field] is None:
            continue
        metadata = row[metadata_field]
        if not isinstance(metadata, Mapping):
            raise CapacityPolicyError(f"{context}: {metadata_field} must be an object")
        for selector_field in ("selector_level", "factor_selector"):
            if selector_field in metadata and metadata[selector_field] is not None:
                actual = str(metadata[selector_field])
                if actual != expected_selector:
                    raise CapacityPolicyError(
                        f"{context}: {metadata_field}.{selector_field}={actual!r} "
                        f"does not match {expected_selector!r}"
                    )
        if expected_controller is not None:
            for controller_field in ("controller_level", "factor_controller"):
                if controller_field in metadata and metadata[controller_field] is not None:
                    actual = str(metadata[controller_field])
                    if actual != expected_controller:
                        raise CapacityPolicyError(
                            f"{context}: {metadata_field}.{controller_field}={actual!r} "
                            f"does not match {expected_controller!r}"
                        )


def _index_list(value: Any, *, field: str, context: str) -> list[int]:
    if not isinstance(value, list):
        raise CapacityPolicyError(f"{context}: {field} must be a list")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise CapacityPolicyError(
                f"{context}: {field} must contain nonnegative integers"
            )
        indices.append(item)
    return indices


def _uid_list(value: Any, *, field: str, context: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CapacityPolicyError(
            f"{context}: {field} must contain non-empty strings"
        )
    uids = [str(item) for item in value]
    if len(uids) != len(set(uids)):
        raise CapacityPolicyError(f"{context}: {field} contains duplicates")
    return uids


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CapacityPolicyError(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CapacityPolicyError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CapacityPolicyError(f"JSON artifact must be an object: {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise CapacityPolicyError(f"invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise CapacityPolicyError(f"JSON row must be an object: {path}:{line_number}")
            yield line_number, row


def _parse_selector_traces(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        selector, separator, path = value.partition("=")
        selector = selector.strip()
        path = path.strip()
        if not separator or not selector or not path:
            raise CapacityPolicyError(
                f"invalid --selector-trace {value!r}; expected SELECTOR=PATH"
            )
        if selector in parsed:
            raise CapacityPolicyError(f"duplicate selector level: {selector}")
        parsed[selector] = Path(path)
    return parsed


def _first_sequence_mismatch(left: Sequence[str], right: Sequence[str]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def _event_sequence_sha256(event_ids: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for event_id in event_ids:
        hasher.update(event_id.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_distribution(distribution: Mapping[int, int]) -> dict[str, int]:
    return {str(key): int(distribution[key]) for key in sorted(distribution)}


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


if __name__ == "__main__":
    raise SystemExit(main())
