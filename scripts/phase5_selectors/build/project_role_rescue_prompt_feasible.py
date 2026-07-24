#!/usr/bin/env python3
"""Project role-rescue slates to the exact verifier-visible evidence prefix.

Diagnostic builds may have dropped whole evidence items (``was_truncated``),
but may not have truncated evidence text.  This projector joins rows by
``event_id``, verifies the visible UIDs are a prefix of the intrinsic role
slate, and rewrites that prefix as a self-contained selected-set trace.  A
formal rebuild of the projected trace must then be untruncated and reproduce
the diagnostic prompt text and token IDs exactly.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


SCHEMA_VERSION = "role_rescue_prompt_feasible_trace_v0_1"
PROJECTION_VERSION = "diagnostic_visible_prefix_v0_1"


class ProjectionError(ValueError):
    """Raised when an intrinsic trace and diagnostic build disagree."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-dir", required=True)
    parser.add_argument("--diagnostic-build-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument(
        "--shared-count-cell",
        action="append",
        default=[],
        help=(
            "Cell participating in an event-wise common prompt-feasible count. "
            "Repeat for capacity-matched cells; omitted cells keep their own count."
        ),
    )
    parser.add_argument(
        "--external-cell",
        action="append",
        default=[],
        help="Manifest-only cell materialized by the prepare wrapper (e.g. native_gate_anchor).",
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    if len(set(args.cell + args.external_cell)) != len(args.cell + args.external_cell):
        parser.error("cell IDs must be unique")
    if len(set(args.shared_count_cell)) != len(args.shared_count_cell):
        parser.error("--shared-count-cell values must be unique")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_projection(
        role_dir=Path(args.role_dir),
        diagnostic_build_root=Path(args.diagnostic_build_root),
        output_dir=Path(args.output_dir),
        split=str(args.split),
        cells=list(args.cell) or None,
        shared_count_cells=list(args.shared_count_cell),
        external_cells=list(args.external_cell),
        sample_limit=args.sample_limit,
        overwrite=bool(args.overwrite),
    )
    print(
        f"Projected {manifest['cell_count']} cells x {manifest['event_count']} rows "
        f"to {args.output_dir}"
    )
    return 0


def materialize_projection(
    *,
    role_dir: Path,
    diagnostic_build_root: Path,
    output_dir: Path,
    split: str,
    cells: Sequence[str] | None = None,
    shared_count_cells: Sequence[str] = (),
    external_cells: Sequence[str] = (),
    sample_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_manifest = _read_json(role_dir / "manifest.json")
    source_cells = _source_cells(source_manifest)
    requested = list(cells) if cells else list(source_cells)
    missing = [cell for cell in requested if cell not in source_cells]
    if missing:
        raise ProjectionError(f"cells missing from role manifest: {missing}")
    if len(set(requested + list(external_cells))) != len(requested) + len(external_cells):
        raise ProjectionError("projected and external cells must be disjoint and unique")
    unknown_shared = [cell for cell in shared_count_cells if cell not in requested]
    if unknown_shared:
        raise ProjectionError(
            f"shared-count cells must also be projected: {unknown_shared}"
        )

    shared_visible_counts: dict[str, int] = {}
    if shared_count_cells:
        per_cell_counts = {
            cell: _load_visible_count_index(
                diagnostic_build_root / cell / "build" / f"build_{split}.jsonl"
            )
            for cell in shared_count_cells
        }
        reference_events = set(next(iter(per_cell_counts.values())))
        for cell, counts in per_cell_counts.items():
            if set(counts) != reference_events:
                raise ProjectionError(
                    f"{cell}: shared-count diagnostic event set differs"
                )
        shared_visible_counts = {
            event_id: min(counts[event_id] for counts in per_cell_counts.values())
            for event_id in reference_events
        }
        if any(count <= 0 for count in shared_visible_counts.values()):
            raise ProjectionError("shared prompt-feasible evidence count must be positive")

    declared_rows = _nonnegative_int(source_manifest.get("row_count"), "manifest row_count")
    expected_rows = min(declared_rows, sample_limit or declared_rows)
    if expected_rows <= 0:
        raise ProjectionError("source manifest has no rows")
    if output_dir.exists() and not overwrite:
        raise ProjectionError(f"output exists; pass --overwrite: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise ProjectionError(f"staging directory exists: {staging}")
    staging.mkdir(parents=True)
    promoted = False
    try:
        manifest_cells: list[dict[str, Any]] = []
        common_event_sha: str | None = None
        for cell_id in requested:
            source_meta = source_cells[cell_id]
            trace_value = str(source_meta.get("trace_file") or source_meta.get("trace") or "")
            if not trace_value:
                raise ProjectionError(f"{cell_id}: manifest has no trace path")
            source_trace = Path(trace_value)
            if not source_trace.is_absolute():
                # Builder manifests historically stored either a cwd-relative
                # full path or a path relative to the manifest directory.
                source_trace = (
                    source_trace
                    if source_trace.is_file()
                    else role_dir / source_trace
                )
            source_build = (
                diagnostic_build_root / cell_id / "build" / f"build_{split}.jsonl"
            )
            cell_dir = staging / cell_id
            cell_dir.mkdir(parents=True)
            output_trace = cell_dir / f"selection_trace_{split}.jsonl"
            summary = project_cell(
                cell_id=cell_id,
                source_trace_path=source_trace,
                source_build_path=source_build,
                output_trace_path=output_trace,
                sample_limit=sample_limit,
                visible_count_caps=(
                    shared_visible_counts if cell_id in shared_count_cells else None
                ),
            )
            if summary["row_count"] != expected_rows:
                raise ProjectionError(
                    f"{cell_id}: rows={summary['row_count']}, expected={expected_rows}"
                )
            event_sha = str(summary["event_id_sequence_sha256"])
            if common_event_sha is None:
                common_event_sha = event_sha
            elif event_sha != common_event_sha:
                raise ProjectionError(f"{cell_id}: event order differs from other cells")
            _write_json(cell_dir / "summary.json", summary)
            manifest_cells.append(
                {
                    "cell_id": cell_id,
                    "trace_file": f"{cell_id}/selection_trace_{split}.jsonl",
                    "summary_file": f"{cell_id}/summary.json",
                    "row_count": expected_rows,
                    "ready": True,
                    "projection_kind": "role_prompt_feasible",
                }
            )
        for cell_id in external_cells:
            manifest_cells.append(
                {
                    "cell_id": str(cell_id),
                    "row_count": expected_rows,
                    "ready": True,
                    "projection_kind": "external_prepare_cell",
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "projection_version": PROJECTION_VERSION,
            "split": split,
            "source_role_dir": str(role_dir),
            "source_diagnostic_build_root": str(diagnostic_build_root),
            "output_dir": str(output_dir),
            "source_event_count": declared_rows,
            "event_count": expected_rows,
            "event_id_sequence_sha256": common_event_sha,
            "shared_count_cells": list(shared_count_cells),
            "shared_count_policy": (
                "event_wise_min_diagnostic_visible_count"
                if shared_count_cells
                else "independent_diagnostic_visible_count"
            ),
            "cell_count": len(manifest_cells),
            "all_ready": bool(manifest_cells),
            "cells": manifest_cells,
            "realization_contract": {
                "join_key": "event_id",
                "diagnostic_was_truncated_allowed": True,
                "diagnostic_evidence_text_truncated_allowed": False,
                "visible_identity": "build.candidates[:evidence_count].candidate_uid",
                "formal_rebuild_required": {
                    "was_truncated": False,
                    "evidence_text_truncated": False,
                    "prompt_equals_diagnostic_when_shared_count_unchanged": True,
                    "capacity_matched_cells_share_evidence_count": bool(
                        shared_count_cells
                    ),
                },
            },
        }
        _write_json(staging / "manifest.json", manifest)
        _promote(staging, output_dir, overwrite=overwrite)
        promoted = True
        return manifest
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def project_cell(
    *,
    cell_id: str,
    source_trace_path: Path,
    source_build_path: Path,
    output_trace_path: Path,
    sample_limit: int | None = None,
    visible_count_caps: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if not source_trace_path.is_file():
        raise ProjectionError(f"missing source trace: {source_trace_path}")
    build_by_event = _load_build_index(source_build_path)
    output_trace_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_trace_path.with_name(f".{output_trace_path.name}.tmp.{os.getpid()}")
    counts_before: list[int] = []
    counts_after: list[int] = []
    drop_events = 0
    drop_total = 0
    event_ids: list[str] = []
    seen: set[str] = set()
    promoted = False
    try:
        with source_trace_path.open(encoding="utf-8") as source, temp.open(
            "w", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if sample_limit is not None and len(event_ids) >= sample_limit:
                    break
                row = _json_object(line, f"{source_trace_path}:{line_number}")
                event_id = _event_id(row, "trace")
                if event_id in seen:
                    raise ProjectionError(f"{cell_id}: duplicate trace event_id {event_id}")
                seen.add(event_id)
                build = build_by_event.get(event_id)
                if build is None:
                    raise ProjectionError(f"{cell_id}:{event_id}: missing diagnostic build row")
                projected = project_row(
                    trace_row=row,
                    build_row=build,
                    source_trace_path=source_trace_path,
                    source_build_path=source_build_path,
                    visible_count_cap=(
                        visible_count_caps.get(event_id)
                        if visible_count_caps is not None
                        else None
                    ),
                )
                before = int(projected["realization_metadata"]["intrinsic_selected_count"])
                after = int(projected["selected_count"])
                dropped = before - after
                counts_before.append(before)
                counts_after.append(after)
                drop_events += int(dropped > 0)
                drop_total += dropped
                event_ids.append(event_id)
                output.write(json.dumps(projected, ensure_ascii=False, sort_keys=True) + "\n")
        if not event_ids:
            raise ProjectionError(f"{cell_id}: source trace has no rows")
        if sample_limit is None and seen != set(build_by_event):
            raise ProjectionError(
                f"{cell_id}: trace/build event sets differ "
                f"(trace={len(seen)}, build={len(build_by_event)})"
            )
        temp.replace(output_trace_path)
        promoted = True
    finally:
        if not promoted:
            temp.unlink(missing_ok=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "projection_version": PROJECTION_VERSION,
        "cell_id": cell_id,
        "source_trace": str(source_trace_path),
        "source_build": str(source_build_path),
        "output_trace": str(output_trace_path),
        "row_count": len(event_ids),
        "unique_event_count": len(seen),
        "event_id_sequence_sha256": _event_sequence_sha(event_ids),
        "projected_drop_event_count": drop_events,
        "projected_drop_evidence_count": drop_total,
        "intrinsic_selected_count_mean": _mean(counts_before),
        "prompt_feasible_selected_count_mean": _mean(counts_after),
        "all_rows_projected": True,
    }


def project_row(
    *,
    trace_row: Mapping[str, Any],
    build_row: Mapping[str, Any],
    source_trace_path: Path | str = "",
    source_build_path: Path | str = "",
    visible_count_cap: int | None = None,
) -> dict[str, Any]:
    event_id = _event_id(trace_row, "trace")
    if _event_id(build_row, "build") != event_id:
        raise ProjectionError(f"{event_id}: trace/build event mismatch")
    if build_row.get("evidence_text_truncated") is not False:
        raise ProjectionError(f"{event_id}: evidence_text_truncated must be exactly false")
    diagnostic_visible_count = _nonnegative_int(
        build_row.get("evidence_count"), "evidence_count"
    )
    visible_count = diagnostic_visible_count
    if visible_count_cap is not None:
        visible_count = min(
            visible_count,
            _nonnegative_int(visible_count_cap, "visible_count_cap"),
        )
    if visible_count <= 0:
        raise ProjectionError(f"{event_id}: empty visible evidence is forbidden")

    pool = _mapping_list(trace_row.get("candidate_pool"), f"{event_id}: candidate_pool")
    selected_indices, selected_uids = _selected_slate(trace_row, pool, event_id)
    build_candidates = _mapping_list(build_row.get("candidates"), f"{event_id}: build candidates")
    if visible_count > len(build_candidates) or visible_count > len(selected_indices):
        raise ProjectionError(f"{event_id}: visible count exceeds available slate")
    visible_uids = [_candidate_uid(item, event_id) for item in build_candidates[:visible_count]]
    if visible_uids != selected_uids[:visible_count]:
        raise ProjectionError(f"{event_id}: visible UIDs are not selected prefix")

    prompt = str(build_row.get("prompt") or "")
    prompt_ids = build_row.get("prompt_input_ids")
    if not prompt or not isinstance(prompt_ids, list) or not prompt_ids:
        raise ProjectionError(f"{event_id}: diagnostic prompt surface is missing")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in prompt_ids):
        raise ProjectionError(f"{event_id}: prompt_input_ids must be integers")

    final_source_indices = selected_indices[:visible_count]
    final_pool: list[dict[str, Any]] = []
    for local_idx, source_idx in enumerate(final_source_indices):
        candidate = deepcopy(pool[source_idx])
        candidate["candidate_idx"] = local_idx
        candidate["selector_candidate_idx"] = local_idx
        candidate["selector_pool_rank"] = local_idx
        final_pool.append(candidate)
    local_indices = list(range(visible_count))
    final_steps = _project_steps(trace_row.get("mrec_steps"), final_pool, event_id)
    final_scores = _project_scores(trace_row.get("candidate_scores"), final_pool)
    intrinsic_role = deepcopy(dict(trace_row.get("role_rescue_metadata") or {}))
    visible_source_indices = [
        int(candidate.get("source_candidate_idx", source_idx))
        for candidate, source_idx in zip(final_pool, final_source_indices)
    ]
    realized = dict(intrinsic_role.get("realized_atom_role_slots") or {})
    visible_slots = {
        slot: deepcopy(value)
        for slot, value in realized.items()
        if isinstance(value, Mapping)
        and int(value.get("source_candidate_idx", -1)) in set(visible_source_indices)
    }

    projected = deepcopy(dict(trace_row))
    source_selector = str(trace_row.get("selector_name") or "role_rescue")
    projected.update(
        {
            "schema_version": SCHEMA_VERSION,
            "graph_version": SCHEMA_VERSION,
            "mrec_trace_version": SCHEMA_VERSION,
            "source_role_rescue_schema_version": str(trace_row.get("schema_version") or ""),
            "selector_name": source_selector + "__prompt_feasible",
            "mrec_selector_name": source_selector + "__prompt_feasible",
            "realization_policy": PROJECTION_VERSION,
            "candidate_pool": final_pool,
            "candidate_scores": final_scores,
            "selector_ordered_indices": local_indices,
            "display_ordered_indices": local_indices,
            "selector_available_ordered_indices": local_indices,
            "selector_full_ordered_indices": local_indices,
            "selected_indices": local_indices,
            "selected_candidates": deepcopy(final_pool),
            "selected_candidate_uids": visible_uids,
            "selected_evidence_ids": [str(item.get("evidence_id") or "") for item in final_pool],
            "selected_keys": visible_uids,
            "selected_count": visible_count,
            "mrec_steps": final_steps,
            "intrinsic_role_rescue_metadata": intrinsic_role,
        }
    )
    active_role = deepcopy(intrinsic_role)
    active_role.update(
        {
            "prompt_feasible_selected_count": visible_count,
            "prompt_feasible_selected_source_indices": visible_source_indices,
            "prompt_feasible_selected_candidate_uids": visible_uids,
            "visible_realized_atom_role_slots": visible_slots,
            "dropped_realized_atom_role_slots": {
                slot: deepcopy(value) for slot, value in realized.items() if slot not in visible_slots
            },
        }
    )
    projected["role_rescue_metadata"] = active_role
    pool_meta = dict(projected.get("candidate_pool_metadata") or {})
    pool_meta.update(
        {
            "projection_schema": PROJECTION_VERSION,
            "prompt_feasible_selected_count": visible_count,
        }
    )
    projected["candidate_pool_metadata"] = pool_meta
    params = dict(projected.get("params") or {})
    params.update({"prompt_evidence_policy": "selected_set", "realization_policy": PROJECTION_VERSION})
    projected["params"] = params
    projected["realization_metadata"] = {
        "projection_version": PROJECTION_VERSION,
        "source_trace_path": str(source_trace_path),
        "source_build_path": str(source_build_path),
        "intrinsic_selected_count": len(selected_indices),
        "prompt_feasible_selected_count": visible_count,
        "prompt_tail_drop_count": len(selected_indices) - visible_count,
        "diagnostic_was_truncated": bool(build_row.get("was_truncated")),
        "diagnostic_evidence_text_truncated": False,
        "diagnostic_visible_count": diagnostic_visible_count,
        "shared_prompt_feasible_count": visible_count,
        "shared_count_tail_drop_count": diagnostic_visible_count - visible_count,
        "diagnostic_surface_matches_projection": (
            diagnostic_visible_count == visible_count
        ),
        "diagnostic_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "diagnostic_prompt_input_ids_sha256": _json_sha(prompt_ids),
        "visible_prefix_verified": True,
        "formal_rebuild_requires_lossless_surface": True,
    }
    return projected


def _load_build_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ProjectionError(f"missing diagnostic build: {path}")
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _json_object(line, f"{path}:{line_number}")
            event_id = _event_id(row, "build")
            if event_id in output:
                raise ProjectionError(f"duplicate diagnostic event_id {event_id}")
            output[event_id] = row
    return output


def _load_visible_count_index(path: Path) -> dict[str, int]:
    rows = _load_build_index(path)
    counts: dict[str, int] = {}
    for event_id, row in rows.items():
        if row.get("evidence_text_truncated") is not False:
            raise ProjectionError(
                f"{event_id}: shared-count source has evidence text truncation"
            )
        count = _nonnegative_int(row.get("evidence_count"), "evidence_count")
        if count <= 0:
            raise ProjectionError(f"{event_id}: shared-count source is empty")
        counts[event_id] = count
    return counts


def _selected_slate(
    row: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], event_id: str
) -> tuple[list[int], list[str]]:
    indices = _int_list(row.get("selected_indices"), f"{event_id}: selected_indices")
    if indices != list(range(len(pool))):
        raise ProjectionError(f"{event_id}: role trace must expose selected-only local pool")
    pool_uids = [_candidate_uid(candidate, event_id) for candidate in pool]
    if len(pool_uids) != len(set(pool_uids)):
        raise ProjectionError(f"{event_id}: duplicate candidate UID")
    for field in ("selector_ordered_indices", "display_ordered_indices"):
        if _int_list(row.get(field), f"{event_id}: {field}") != indices:
            raise ProjectionError(f"{event_id}: selected index aliases disagree")
    aliases = row.get("selected_candidate_uids")
    if not isinstance(aliases, list) or [str(value) for value in aliases] != pool_uids:
        raise ProjectionError(f"{event_id}: selected UID alias disagrees")
    if int(row.get("selected_count", -1)) != len(indices):
        raise ProjectionError(f"{event_id}: selected_count disagrees")
    return indices, pool_uids


def _project_steps(value: Any, pool: Sequence[Mapping[str, Any]], event_id: str) -> list[dict[str, Any]]:
    steps = _mapping_list(value, f"{event_id}: mrec_steps")
    if len(steps) < len(pool):
        raise ProjectionError(f"{event_id}: insufficient mrec_steps")
    output: list[dict[str, Any]] = []
    for idx, candidate in enumerate(pool):
        step = deepcopy(steps[idx])
        uid = _candidate_uid(candidate, event_id)
        if str(step.get("candidate_uid") or uid) != uid:
            raise ProjectionError(f"{event_id}: mrec step UID mismatch")
        step.update(
            {
                "step": idx + 1,
                "candidate_idx": idx,
                "selector_candidate_idx": idx,
                "candidate_uid": uid,
            }
        )
        output.append(step)
    return output


def _project_scores(value: Any, pool: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scores = [dict(item) for item in value or [] if isinstance(item, Mapping)]
    by_uid = {str(item.get("candidate_uid") or ""): item for item in scores}
    output: list[dict[str, Any]] = []
    for idx, candidate in enumerate(pool):
        uid = str(candidate.get("candidate_uid") or "")
        score = deepcopy(by_uid.get(uid, scores[idx] if idx < len(scores) else {}))
        score.update({"candidate_idx": idx, "candidate_uid": uid, "selector_selected_step": idx})
        output.append(score)
    return output


def _source_cells(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("cells")
    if isinstance(raw, Mapping):
        return {str(cell): dict(meta) for cell, meta in raw.items() if isinstance(meta, Mapping)}
    if isinstance(raw, list):
        output = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise ProjectionError("manifest cells must be objects")
            cell_id = str(item.get("cell_id") or "")
            if not cell_id or cell_id in output:
                raise ProjectionError("manifest cell IDs must be unique and non-empty")
            output[cell_id] = dict(item)
        return output
    raise ProjectionError("role manifest has no cells")


def _mapping_list(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ProjectionError(f"{context} must be an object array")
    return [dict(item) for item in value]


def _int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ProjectionError(f"{context} must be an integer array")
    return [int(item) for item in value]


def _nonnegative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProjectionError(f"{context} must be a non-negative integer")
    return int(value)


def _candidate_uid(candidate: Mapping[str, Any], event_id: str) -> str:
    uid = str(candidate.get("candidate_uid") or "").strip()
    if not uid:
        raise ProjectionError(f"{event_id}: candidate has no UID")
    return uid


def _event_id(row: Mapping[str, Any], context: str) -> str:
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        raise ProjectionError(f"{context} row has no event_id")
    return event_id


def _json_object(text: str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProjectionError(f"invalid JSON at {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProjectionError(f"{context} is not an object")
    return value


def _event_sequence_sha(event_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for event_id in event_ids:
        digest.update(event_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProjectionError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProjectionError(f"{path} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _promote(staging: Path, output: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if output.exists():
        if not overwrite:
            raise ProjectionError(f"refusing to replace {output}")
        backup = output.with_name(f".{output.name}.backup.{os.getpid()}")
        output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
