#!/usr/bin/env python3
"""Project factorial controller slates to verifier-visible prompt-feasible traces.

The source factorial trace remains the intrinsic selector/controller artifact.
This projection consumes the corresponding verifier build, verifies that its
visible evidence is exactly a non-empty prefix of the controller slate, and
materializes that prefix as an explicit trace.  Rebuilding the projected trace
with ``selected_set`` plus ``--forbid-prompt-truncation`` must therefore be
lossless.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.baces_objective import (  # noqa: E402
    BacesCandidate,
    BacesProblem,
    evaluate_display,
    padded_auc,
)
from scripts.phase5_selectors.build.build_baces_factorial_traces import (  # noqa: E402
    K_MAX,
    _display_steps,
    _mrec_cue_steps,
)


SCHEMA_VERSION = "baces_factorial_prompt_feasible_trace_v0_2"
PROJECTION_VERSION = "verifier_visible_prefix_projection_v0_2"


class ProjectionError(ValueError):
    """Raised when source traces and realized verifier rows do not align."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-dir", required=True)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Project only this manifest cell. Repeat for multiple cells.",
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Transactionally replace an existing output directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sample_limit is not None and args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")
    if len(set(args.cell)) != len(args.cell):
        parser.error("--cell values must be unique")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not args.overwrite:
        raise ProjectionError(
            f"output directory already exists; pass --overwrite to replace it: {output_dir}"
        )
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging_dir.exists():
        raise ProjectionError(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    promoted = False
    try:
        cell_count, event_count = _materialize(
            args,
            output_dir=staging_dir,
            logical_output_dir=output_dir,
        )
        _promote_output_tree(
            staging_dir=staging_dir,
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
        )
        promoted = True
    finally:
        if not promoted:
            shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Projected {cell_count} cells x {event_count} rows to {output_dir}")
    return 0


def _materialize(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    logical_output_dir: Path,
) -> tuple[int, int]:
    factorial_dir = Path(args.factorial_dir)
    build_root = Path(args.build_root)
    source_manifest = _read_json(factorial_dir / "manifest.json")
    manifest_by_id, source_event_count = _validate_source_manifest(source_manifest)
    requested = list(args.cell) if args.cell else list(manifest_by_id)
    missing = [cell for cell in requested if cell not in manifest_by_id]
    if missing:
        raise ProjectionError(f"requested cells missing from manifest: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    expected_rows = min(
        source_event_count,
        args.sample_limit if args.sample_limit is not None else source_event_count,
    )
    event_sequence_sha256: str | None = None
    for cell_id in requested:
        source_cell = manifest_by_id[cell_id]
        relative_trace = str(source_cell.get("trace_file") or "")
        if not relative_trace:
            raise ProjectionError(f"manifest cell {cell_id!r} has no trace_file")
        source_trace = factorial_dir / relative_trace
        source_build = build_root / cell_id / "build" / f"build_{args.split}.jsonl"
        cell_dir = output_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        output_trace = cell_dir / f"selection_trace_{args.split}.jsonl"
        summary = project_cell(
            cell_id=cell_id,
            source_trace_path=source_trace,
            source_build_path=source_build,
            output_trace_path=output_trace,
            sample_limit=args.sample_limit,
        )
        summary["output_trace"] = str(
            logical_output_dir / cell_id / f"selection_trace_{args.split}.jsonl"
        )
        _write_json(cell_dir / "summary.json", summary)
        if int(summary["row_count"]) != expected_rows:
            raise ProjectionError(
                f"{cell_id}: projected row_count={summary['row_count']} differs "
                f"from manifest/sample expectation {expected_rows}"
            )
        cell_event_sha256 = str(summary["event_id_sequence_sha256"])
        if event_sequence_sha256 is None:
            event_sequence_sha256 = cell_event_sha256
        elif cell_event_sha256 != event_sequence_sha256:
            raise ProjectionError(
                f"{cell_id}: event ID sequence differs from the other projected cells"
            )
        cells.append(
            {
                "cell_id": cell_id,
                "selector_level": source_cell.get("selector_level"),
                "controller_level": source_cell.get("controller_level"),
                "trace_file": f"{cell_id}/selection_trace_{args.split}.jsonl",
                "summary_file": f"{cell_id}/summary.json",
                "row_count": int(summary["row_count"]),
                "ready": bool(summary["all_rows_projected"])
                and int(summary["row_count"]) == expected_rows,
                "projected_drop_event_count": int(
                    summary["projected_drop_event_count"]
                ),
                "projected_drop_evidence_count": int(
                    summary["projected_drop_evidence_count"]
                ),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "projection_version": PROJECTION_VERSION,
        "split": str(args.split),
        "sample_limit": args.sample_limit,
        "source_factorial_dir": str(factorial_dir),
        "source_build_root": str(build_root),
        "output_dir": str(logical_output_dir),
        "source_event_count": source_event_count,
        "event_count": expected_rows,
        "event_id_sequence_sha256": event_sequence_sha256,
        "cell_count": len(cells),
        "all_ready": all(bool(cell["ready"]) for cell in cells),
        "selector_levels": source_manifest.get("selector_levels"),
        "controller_levels": source_manifest.get("controller_levels"),
        "selector_contracts": source_manifest.get("selector_contracts"),
        "controller_contracts": source_manifest.get("controller_contracts"),
        "realization_contract": {
            "policy": PROJECTION_VERSION,
            "visible_identity": "build.candidates[:build.evidence_count].candidate_uid",
            "required_relation": "visible_uids == controller_selected_uids[:K_visible]",
            "evidence_text_truncated_allowed": False,
            "empty_evidence_allowed": False,
            "projected_rebuild_must_use": {
                "selection_mode": "trace",
                "trace_prompt_style": "mrec_min",
                "prompt_evidence_policy": "selected_set",
                "forbid_prompt_truncation": True,
            },
        },
        "cells": cells,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return len(cells), expected_rows


def project_cell(
    *,
    cell_id: str,
    source_trace_path: Path,
    source_build_path: Path,
    output_trace_path: Path,
    sample_limit: int | None,
) -> dict[str, Any]:
    if not source_trace_path.is_file():
        raise ProjectionError(f"missing source trace: {source_trace_path}")
    if not source_build_path.is_file():
        raise ProjectionError(f"missing source build: {source_build_path}")
    temp_path = output_trace_path.with_name(
        f".{output_trace_path.name}.tmp.{os.getpid()}"
    )
    row_count = 0
    drop_events = 0
    drop_evidence = 0
    max_drop = 0
    pre_counts: list[int] = []
    final_counts: list[int] = []
    event_ids_seen: set[str] = set()
    event_id_hasher = hashlib.sha256()
    promoted = False
    try:
        with ExitStack() as stack:
            trace_handle = stack.enter_context(source_trace_path.open(encoding="utf-8"))
            build_handle = stack.enter_context(source_build_path.open(encoding="utf-8"))
            output_handle = stack.enter_context(temp_path.open("w", encoding="utf-8"))
            trace_rows = _iter_jsonl(trace_handle, artifact=str(source_trace_path))
            build_rows = _iter_jsonl(build_handle, artifact=str(source_build_path))
            while True:
                if sample_limit is not None and row_count >= sample_limit:
                    break
                trace_row = next(trace_rows, None)
                build_row = next(build_rows, None)
                if trace_row is None or build_row is None:
                    if trace_row is not None or build_row is not None:
                        raise ProjectionError(
                            f"{cell_id}: trace/build row counts differ"
                        )
                    break
                trace_cell_id = str(trace_row.get("factorial_cell_id") or "").strip()
                if trace_cell_id != cell_id:
                    raise ProjectionError(
                        f"{cell_id}: trace factorial_cell_id={trace_cell_id!r} disagrees"
                    )
                projected = project_row(
                    trace_row=trace_row,
                    build_row=build_row,
                    source_trace_path=source_trace_path,
                    source_build_path=source_build_path,
                )
                event_id = _event_id(projected, "projected trace")
                if event_id in event_ids_seen:
                    raise ProjectionError(f"{cell_id}: duplicate event_id {event_id!r}")
                event_ids_seen.add(event_id)
                event_id_hasher.update(event_id.encode("utf-8"))
                event_id_hasher.update(b"\0")
                pre_count = int(projected["realization_metadata"]["controller_selected_count"])
                final_count = int(projected["selected_count"])
                dropped = pre_count - final_count
                row_count += 1
                pre_counts.append(pre_count)
                final_counts.append(final_count)
                if dropped > 0:
                    drop_events += 1
                    drop_evidence += dropped
                    max_drop = max(max_drop, dropped)
                output_handle.write(
                    json.dumps(projected, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temp_path.replace(output_trace_path)
        promoted = True
    finally:
        if not promoted:
            temp_path.unlink(missing_ok=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "projection_version": PROJECTION_VERSION,
        "cell_id": cell_id,
        "source_trace": str(source_trace_path),
        "source_build": str(source_build_path),
        "output_trace": str(output_trace_path),
        "sample_limit": sample_limit,
        "row_count": row_count,
        "all_rows_projected": row_count > 0,
        "unique_event_count": len(event_ids_seen),
        "event_id_sequence_sha256": event_id_hasher.hexdigest(),
        "projected_drop_event_count": drop_events,
        "projected_drop_event_rate": drop_events / row_count if row_count else 0.0,
        "projected_drop_evidence_count": drop_evidence,
        "projected_drop_evidence_max": max_drop,
        "controller_selected_count_mean": _mean(pre_counts),
        "prompt_feasible_selected_count_mean": _mean(final_counts),
    }


def project_row(
    *,
    trace_row: Mapping[str, Any],
    build_row: Mapping[str, Any],
    source_trace_path: Path | str,
    source_build_path: Path | str,
) -> dict[str, Any]:
    event_id = _event_id(trace_row, "trace")
    if _event_id(build_row, "build") != event_id:
        raise ProjectionError(f"{event_id}: trace/build event_id mismatch")
    if build_row.get("evidence_text_truncated") is not False:
        raise ProjectionError(
            f"{event_id}: build evidence_text_truncated must be exactly false"
        )
    evidence_count = _nonnegative_int(
        build_row.get("evidence_count"), f"{event_id}: evidence_count"
    )
    if evidence_count <= 0:
        raise ProjectionError(f"{event_id}: empty visible evidence is forbidden")

    pool = _mapping_list(trace_row.get("candidate_pool"), f"{event_id}: candidate_pool")
    pre_indices, pre_uids = _validated_selected_slate(
        trace_row,
        pool=pool,
        event_id=event_id,
    )
    build_candidates = _mapping_list(
        build_row.get("candidates"), f"{event_id}: build candidates"
    )
    if evidence_count > len(build_candidates):
        raise ProjectionError(f"{event_id}: evidence_count exceeds build candidates")
    visible_candidates = build_candidates[:evidence_count]
    visible_uids = [_candidate_uid(candidate, event_id) for candidate in visible_candidates]
    if visible_uids != pre_uids[:evidence_count]:
        raise ProjectionError(
            f"{event_id}: visible UIDs are not the controller-selected prefix"
        )
    if evidence_count > len(pre_uids):
        raise ProjectionError(f"{event_id}: visible count exceeds controller slate")
    for field in (
        "evidence_count_before",
        "prompt_evidence_selected_count_before_prompt_truncation",
    ):
        value = build_row.get(field)
        if value is not None and _nonnegative_int(value, f"{event_id}:{field}") != len(
            pre_indices
        ):
            raise ProjectionError(
                f"{event_id}: {field} does not equal controller slate length"
            )

    final_indices = pre_indices[:evidence_count]
    problem = _problem_from_trace(trace_row, pool=pool, event_id=event_id)
    display = evaluate_display(problem, visible_uids)
    target_state = _target_state(trace_row, atom_count=len(problem.atom_ids), event_id=event_id)
    claim_atoms = _mapping_list(trace_row.get("claim_atoms"), f"{event_id}: claim_atoms")
    steps = _display_steps(
        display,
        final_indices,
        pool,
        problem,
        target_state=target_state,
    )
    mrec_steps = _mrec_cue_steps(
        display_steps=steps,
        pool=pool,
        problem=problem,
        claim_atoms=claim_atoms,
        target_state=target_state,
    )

    projected = copy.deepcopy(dict(trace_row))
    source_schema = str(trace_row.get("schema_version") or "")
    source_selector = str(trace_row.get("selector_name") or "")
    projected.update(
        {
            "schema_version": SCHEMA_VERSION,
            "graph_version": SCHEMA_VERSION,
            "mrec_trace_version": SCHEMA_VERSION,
            "source_factorial_schema_version": source_schema,
            "selector_name": source_selector + "__prompt_feasible",
            "mrec_selector_name": source_selector + "__prompt_feasible",
            "realization_policy": PROJECTION_VERSION,
            "controller_selected_indices": list(pre_indices),
            "controller_selected_candidate_uids": list(pre_uids),
            "controller_selected_count": len(pre_indices),
            "controller_selected_token_cost": int(
                trace_row.get("selected_token_cost") or 0
            ),
            "ordered_indices": list(final_indices),
            "ordered_candidate_uids": list(visible_uids),
            "ordered_candidates": [copy.deepcopy(dict(pool[idx])) for idx in final_indices],
            "selector_ordered_indices": list(final_indices),
            "display_ordered_indices": list(final_indices),
            "selected_indices": list(final_indices),
            "selected_candidates": [copy.deepcopy(dict(pool[idx])) for idx in final_indices],
            "selected_candidate_uids": list(visible_uids),
            "selected_keys": list(visible_uids),
            "selected_candidate_keys": [
                str(pool[idx].get("candidate_key") or "") for idx in final_indices
            ],
            "selected_evidence_ids": [
                str(pool[idx].get("evidence_id") or "") for idx in final_indices
            ],
            "selected_count": len(final_indices),
            "selected_token_cost": int(display.token_cost),
            "baces_display_steps": steps,
            "mrec_steps": mrec_steps,
            "baces_display": {
                "terminal_state": list(display.state),
                "terminal_utility": int(display.utility),
                "acquisition_time": int(display.acquisition_time),
                "token_cost": int(display.token_cost),
                "length": int(display.length),
                "padded_auc_horizon10": int(padded_auc(display, K_MAX)),
            },
        }
    )
    selected_step = {idx: rank for rank, idx in enumerate(final_indices)}
    scores = projected.get("candidate_scores")
    if isinstance(scores, list):
        for score in scores:
            if not isinstance(score, dict):
                continue
            idx = score.get("candidate_idx")
            score["prompt_feasible_selected_step"] = selected_step.get(idx, -1)

    realization_metadata = {
        "projection_version": PROJECTION_VERSION,
        "source_trace_path": str(source_trace_path),
        "source_build_path": str(source_build_path),
        "controller_selected_count": len(pre_indices),
        "prompt_feasible_selected_count": len(final_indices),
        "prompt_tail_drop_count": len(pre_indices) - len(final_indices),
        "source_build_was_truncated": bool(build_row.get("was_truncated")),
        "source_build_evidence_text_truncated": False,
        "source_build_prompt_token_count": int(build_row.get("prompt_token_count") or 0),
        "visible_prefix_verified": True,
    }
    projected["realization_metadata"] = realization_metadata
    for metadata_field in ("factorial_metadata", "factor_metadata"):
        raw_metadata = projected.get(metadata_field)
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.update(realization_metadata)
        projected[metadata_field] = metadata
    params = dict(projected.get("params") or {})
    params["realization_policy"] = PROJECTION_VERSION
    projected["params"] = params
    return projected


def _validate_source_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    """Validate that the source describes one complete factorial grid."""

    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ProjectionError("source factorial manifest has no cells")
    selector_levels = _unique_nonempty_strings(
        manifest.get("selector_levels"), "source selector_levels"
    )
    controller_levels = _unique_nonempty_strings(
        manifest.get("controller_levels"), "source controller_levels"
    )
    expected_ids = {
        f"{selector}__{controller}"
        for selector in selector_levels
        for controller in controller_levels
    }
    declared_cell_count = _nonnegative_int(
        manifest.get("cell_count"), "source cell_count"
    )
    if declared_cell_count != len(raw_cells):
        raise ProjectionError(
            "source cell_count differs from the number of manifest cells"
        )
    if len(raw_cells) != len(expected_ids):
        raise ProjectionError(
            "source manifest is not a complete selector x controller grid"
        )
    if manifest.get("all_ready") is not True:
        raise ProjectionError("source factorial manifest is not all_ready")
    event_count = _nonnegative_int(manifest.get("event_count"), "source event_count")
    if event_count <= 0:
        raise ProjectionError("source event_count must be positive")

    manifest_by_id: dict[str, Mapping[str, Any]] = {}
    coordinates: set[tuple[str, str]] = set()
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            raise ProjectionError(f"source manifest cell {index} is not an object")
        cell_id = str(raw_cell.get("cell_id") or "").strip()
        selector = str(raw_cell.get("selector_level") or "").strip()
        controller = str(raw_cell.get("controller_level") or "").strip()
        if not cell_id or not selector or not controller:
            raise ProjectionError(f"source manifest cell {index} has empty identity fields")
        if cell_id in manifest_by_id:
            raise ProjectionError(f"source manifest has duplicate cell_id {cell_id!r}")
        coordinate = (selector, controller)
        if coordinate in coordinates:
            raise ProjectionError(
                f"source manifest has duplicate factor coordinate {coordinate!r}"
            )
        if selector not in selector_levels or controller not in controller_levels:
            raise ProjectionError(f"source manifest cell {cell_id!r} has unknown factors")
        if cell_id != f"{selector}__{controller}":
            raise ProjectionError(
                f"source manifest cell_id {cell_id!r} disagrees with its factors"
            )
        if raw_cell.get("ready") is not True:
            raise ProjectionError(f"source manifest cell {cell_id!r} is not ready")
        row_count = _nonnegative_int(
            raw_cell.get("row_count"), f"source cell {cell_id}: row_count"
        )
        if row_count != event_count:
            raise ProjectionError(
                f"source cell {cell_id!r} row_count={row_count} differs "
                f"from event_count={event_count}"
            )
        if not str(raw_cell.get("trace_file") or "").strip():
            raise ProjectionError(f"source manifest cell {cell_id!r} has no trace_file")
        manifest_by_id[cell_id] = raw_cell
        coordinates.add(coordinate)

    actual_ids = set(manifest_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ProjectionError(
            f"source factorial grid mismatch: missing={missing}, extra={extra}"
        )
    return manifest_by_id, event_count


def _validated_selected_slate(
    trace_row: Mapping[str, Any],
    *,
    pool: Sequence[Mapping[str, Any]],
    event_id: str,
) -> tuple[list[int], list[str]]:
    """Require every source-trace selected-slate alias to describe one slate."""

    pool_uids = [_candidate_uid(candidate, event_id) for candidate in pool]
    if len(pool_uids) != len(set(pool_uids)):
        raise ProjectionError(f"{event_id}: candidate_pool has duplicate UIDs")
    selected_indices = _int_list(
        trace_row.get("selected_indices"), f"{event_id}: selected_indices"
    )
    if len(selected_indices) != len(set(selected_indices)) or any(
        idx < 0 or idx >= len(pool) for idx in selected_indices
    ):
        raise ProjectionError(f"{event_id}: selected indices are invalid")
    for field in (
        "ordered_indices",
        "selector_ordered_indices",
        "display_ordered_indices",
    ):
        alias = _int_list(trace_row.get(field), f"{event_id}: {field}")
        if alias != selected_indices:
            raise ProjectionError(
                f"{event_id}: selected index aliases disagree at {field}"
            )

    selected_uids = [pool_uids[idx] for idx in selected_indices]
    for field in ("selected_candidate_uids", "ordered_candidate_uids", "selected_keys"):
        alias = _string_list(trace_row.get(field), f"{event_id}: {field}")
        if alias != selected_uids:
            raise ProjectionError(
                f"{event_id}: selected UID aliases disagree at {field}"
            )
    for field in ("selected_candidates", "ordered_candidates"):
        candidates = _mapping_list(trace_row.get(field), f"{event_id}: {field}")
        alias = [_candidate_uid(candidate, event_id) for candidate in candidates]
        if alias != selected_uids:
            raise ProjectionError(
                f"{event_id}: selected candidate aliases disagree at {field}"
            )
    selected_count = _nonnegative_int(
        trace_row.get("selected_count"), f"{event_id}: selected_count"
    )
    if selected_count != len(selected_indices):
        raise ProjectionError(f"{event_id}: selected_count disagrees with selected slate")
    return selected_indices, selected_uids


def _problem_from_trace(
    trace_row: Mapping[str, Any],
    *,
    pool: Sequence[Mapping[str, Any]],
    event_id: str,
) -> BacesProblem:
    claim_atoms = _mapping_list(trace_row.get("claim_atoms"), f"{event_id}: claim_atoms")
    atom_ids = tuple(
        str(atom.get("atom_id") or atom.get("node_id") or f"A{idx + 1}")
        for idx, atom in enumerate(claim_atoms)
    )
    if not atom_ids or len(atom_ids) != len(set(atom_ids)):
        raise ProjectionError(f"{event_id}: invalid claim atom IDs")
    candidates: list[BacesCandidate] = []
    for idx, raw in enumerate(pool):
        uid = _candidate_uid(raw, event_id)
        q = _int_list(raw.get("baces_q"), f"{event_id}:{uid}:baces_q")
        if len(q) != len(atom_ids) or any(level not in (0, 1, 2) for level in q):
            raise ProjectionError(f"{event_id}:{uid}: invalid baces_q")
        cost = _nonnegative_int(
            raw.get("mrec_token_cost"), f"{event_id}:{uid}:mrec_token_cost"
        )
        candidates.append(BacesCandidate(key=uid, q=tuple(q), cost=cost, uid=uid))
    return BacesProblem(
        candidates=tuple(candidates),
        weights=tuple(1 for _ in atom_ids),
        k_max=K_MAX,
        token_budget=None,
        atom_ids=atom_ids,
    )


def _target_state(
    trace_row: Mapping[str, Any], *, atom_count: int, event_id: str
) -> tuple[int, ...]:
    metadata = trace_row.get("factorial_metadata")
    raw = (
        metadata.get("common_exact_kmax10_target_state")
        if isinstance(metadata, Mapping)
        else None
    )
    state = _int_list(raw, f"{event_id}: common target state")
    if len(state) != atom_count or any(level not in (0, 1, 2) for level in state):
        raise ProjectionError(f"{event_id}: invalid common target state")
    return tuple(state)


def _iter_jsonl(handle: TextIO, *, artifact: str) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProjectionError(f"invalid JSON in {artifact}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ProjectionError(f"{artifact}:{line_number} is not an object")
        yield row


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProjectionError(f"{path} is not a JSON object")
    return payload


def _promote_output_tree(
    *, staging_dir: Path, output_dir: Path, overwrite: bool
) -> None:
    """Promote one complete staging tree, restoring the old tree on failure."""

    backup_dir: Path | None = None
    if output_dir.exists():
        if not overwrite:
            raise ProjectionError(f"refusing to replace existing output: {output_dir}")
        backup_dir = output_dir.with_name(f".{output_dir.name}.backup.{os.getpid()}")
        if backup_dir.exists():
            raise ProjectionError(f"backup directory already exists: {backup_dir}")
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _event_id(row: Mapping[str, Any], context: str) -> str:
    value = str(row.get("event_id") or "").strip()
    if not value:
        raise ProjectionError(f"{context} row has no event_id")
    return value


def _candidate_uid(candidate: Mapping[str, Any], event_id: str) -> str:
    value = str(candidate.get("candidate_uid") or "").strip()
    if not value:
        raise ProjectionError(f"{event_id}: candidate has no candidate_uid")
    return value


def _mapping_list(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ProjectionError(f"{context} must be an object array")
    return [dict(item) for item in value]


def _unique_nonempty_strings(value: Any, context: str) -> list[str]:
    output = _string_list(value, context)
    if not output or any(not item for item in output):
        raise ProjectionError(f"{context} must contain non-empty strings")
    if len(output) != len(set(output)):
        raise ProjectionError(f"{context} must contain unique values")
    return output


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProjectionError(f"{context} must be a string array")
    return [str(item) for item in value]


def _int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise ProjectionError(f"{context} must be an integer array")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ProjectionError(f"{context} must contain integers")
        output.append(int(item))
    return output


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionError(f"{context} must be a non-negative integer")
    return int(value)


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
