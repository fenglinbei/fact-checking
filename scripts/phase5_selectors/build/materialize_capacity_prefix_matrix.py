#!/usr/bin/env python3
"""Validate final prefix builds and freeze a matrix-runner manifest."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


SCHEMA_VERSION = "baces_capacity_prefix_matrix_v0_1"
GATE_SCHEMA_VERSION = "capacity_prefix_integrity_gate_v0_1"


class CapacityMatrixError(ValueError):
    """Raised when a planned prefix lattice and final verifier builds diverge."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise CapacityMatrixError(
            f"output directory already exists; pass --overwrite: {output_dir}"
        )
    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise CapacityMatrixError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    promoted = False
    try:
        manifest = materialize_capacity_matrix(
            plan_manifest_path=Path(args.plan_manifest),
            build_root=Path(args.build_root),
            output_dir=staging,
            logical_output_dir=output_dir,
            split=str(args.split),
        )
        _promote_directory(staging, output_dir, overwrite=bool(args.overwrite))
        promoted = True
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)
    print(
        f"Validated {manifest['cell_count']} prefix cells x "
        f"{manifest['event_count']} events at {output_dir}"
    )
    return 0


def materialize_capacity_matrix(
    *,
    plan_manifest_path: Path,
    build_root: Path,
    output_dir: Path,
    logical_output_dir: Path,
    split: str,
) -> dict[str, Any]:
    plan_manifest = _read_json(plan_manifest_path)
    if plan_manifest.get("all_ready") is not True:
        raise CapacityMatrixError("prefix plan manifest is not all_ready=true")
    if str(plan_manifest.get("split") or "") != split:
        raise CapacityMatrixError("prefix plan split disagrees with requested split")
    selectors = _unique_strings(plan_manifest.get("selector_levels"), "selector_levels")
    controllers = _unique_strings(
        plan_manifest.get("controller_levels"), "controller_levels"
    )
    capacities = _int_list(plan_manifest.get("capacity_values"), "capacity_values")
    if controllers != [_capacity_level(k) for k in capacities]:
        raise CapacityMatrixError("controller_levels do not match capacity_values")
    raw_cells = plan_manifest.get("cells")
    if not isinstance(raw_cells, list) or not all(
        isinstance(cell, Mapping) for cell in raw_cells
    ):
        raise CapacityMatrixError("prefix plan manifest cells must be an object array")
    expected_ids = {
        _cell_id(selector, k) for selector in selectors for k in capacities
    }
    cells_by_id: dict[str, dict[str, Any]] = {}
    for raw_cell in raw_cells:
        cell = dict(raw_cell)
        cell_id = str(cell.get("cell_id") or "")
        if not cell_id or cell_id in cells_by_id:
            raise CapacityMatrixError(f"duplicate or empty plan cell_id={cell_id!r}")
        cells_by_id[cell_id] = cell
    if set(cells_by_id) != expected_ids:
        raise CapacityMatrixError(
            "prefix plan grid mismatch: "
            f"missing={sorted(expected_ids - set(cells_by_id))}, "
            f"extra={sorted(set(cells_by_id) - expected_ids)}"
        )
    capacity_policy = str(plan_manifest.get("capacity_policy") or "")
    source_controller = str(plan_manifest.get("source_controller") or "")
    if not capacity_policy or not source_controller:
        raise CapacityMatrixError(
            "prefix plan manifest must declare capacity_policy and source_controller"
        )
    for selector in selectors:
        for k in capacities:
            cell_id = _cell_id(selector, k)
            cell = cells_by_id[cell_id]
            if (
                str(cell.get("selector_level") or "") != selector
                or str(cell.get("controller_level") or "") != _capacity_level(k)
                or isinstance(cell.get("capacity_k"), bool)
                or not isinstance(cell.get("capacity_k"), int)
                or cell.get("capacity_k") != k
                or str(cell.get("capacity_policy") or "") != capacity_policy
                or str(cell.get("source_order_cell") or "")
                != f"{selector}__{source_controller}"
            ):
                raise CapacityMatrixError(
                    f"plan cell metadata disagrees with grid contract: {cell_id}"
                )

    plan_root = plan_manifest_path.parent
    plan_rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell_id in sorted(expected_ids):
        cell = cells_by_id[cell_id]
        plan_path = plan_root / str(cell.get("plan_file") or "")
        if not plan_path.is_file():
            raise CapacityMatrixError(f"missing cell plan: {plan_path}")
        expected_sha = str(cell.get("plan_sha256") or "")
        if expected_sha and _sha256_file(plan_path) != expected_sha:
            raise CapacityMatrixError(f"cell plan SHA mismatch: {cell_id}")
        rows = _load_jsonl(plan_path)
        if len(rows) != int(cell.get("row_count", -1)):
            raise CapacityMatrixError(f"cell plan row count mismatch: {cell_id}")
        plan_rows_by_cell[cell_id] = rows

    gate = _validate_plan_lattice(
        selectors=selectors,
        capacities=capacities,
        plan_rows_by_cell=plan_rows_by_cell,
        capacity_policy=capacity_policy,
        source_controller=source_controller,
    )
    expected_event_sha = str(gate["event_id_sequence_sha256"])
    expected_event_count = int(gate["event_count"])
    declared_event_count = int(plan_manifest.get("event_count", -1))
    if declared_event_count != expected_event_count:
        raise CapacityMatrixError("plan manifest event_count disagrees with cell plans")

    formal_cells: list[dict[str, Any]] = []
    common_label_schema: str | None = None
    for selector in selectors:
        for k in capacities:
            cell_id = _cell_id(selector, k)
            source_cell = cells_by_id[cell_id]
            build_path = build_root / cell_id / "build" / f"build_{split}.jsonl"
            if not build_path.is_file():
                raise CapacityMatrixError(f"missing final build: {build_path}")
            build_rows = _load_jsonl(build_path)
            plan_rows = plan_rows_by_cell[cell_id]
            if len(build_rows) != expected_event_count:
                raise CapacityMatrixError(f"final build row count mismatch: {cell_id}")
            cell_event_hasher = hashlib.sha256()
            exact_count = 0
            evidence_total = 0
            token_total = 0
            for index, (plan, build) in enumerate(zip(plan_rows, build_rows)):
                event_id = str(plan.get("event_id") or "")
                if str(build.get("event_id") or "") != event_id:
                    raise CapacityMatrixError(
                        f"{cell_id}[{index}]: plan/build event mismatch"
                    )
                cell_event_hasher.update(event_id.encode("utf-8"))
                cell_event_hasher.update(b"\0")
                _validate_final_build_row(
                    cell_id=cell_id,
                    capacity_k=k,
                    plan=plan,
                    build=build,
                    row_index=index,
                )
                label_schema = str(build.get("label_schema") or "")
                if common_label_schema is None:
                    common_label_schema = label_schema
                elif label_schema != common_label_schema:
                    raise CapacityMatrixError("final builds have inconsistent label schemas")
                exact_count += int(bool(plan.get("exact_policy_k")))
                evidence_total += int(build.get("evidence_count") or 0)
                token_total += int(build.get("prompt_token_count") or 0)
            if cell_event_hasher.hexdigest() != expected_event_sha:
                raise CapacityMatrixError(f"final build event sequence mismatch: {cell_id}")
            formal_cells.append(
                {
                    "cell_id": cell_id,
                    "selector_level": selector,
                    "controller_level": _capacity_level(k),
                    "capacity_k": k,
                    "capacity_policy": source_cell.get("capacity_policy"),
                    "source_order_cell": source_cell.get("source_order_cell"),
                    "plan_file": str(
                        (plan_root / str(source_cell.get("plan_file") or "")).resolve()
                    ),
                    "plan_sha256": source_cell.get("plan_sha256"),
                    "build_file": str(build_path.resolve()),
                    "build_sha256": _sha256_file(build_path),
                    "row_count": expected_event_count,
                    "event_id_sequence_sha256": expected_event_sha,
                    "exact_policy_k_event_count": exact_count,
                    "exact_policy_k_event_rate": exact_count / expected_event_count,
                    "mean_realized_k": evidence_total / expected_event_count,
                    "mean_prompt_token_count": token_total / expected_event_count,
                    "ready": True,
                }
            )

    gate.update(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "passed": True,
            "final_build_cell_count": len(formal_cells),
            "final_build_row_count": len(formal_cells) * expected_event_count,
            "checks": [
                "same event order across selectors and K",
                "same frozen configured-order prefix within selector",
                "adjacent K adds at most one suffix item or plateaus",
                "candidate UID and evidence ID identity",
                "final build selected_set replay",
                "final build contains no prompt or evidence-text truncation",
            ],
        }
    )
    _write_json(output_dir / "prefix_integrity_gate.json", gate)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "matrix_kind": "ordered_prompt_feasible_prefix_lattice",
        "split": split,
        "label_schema": common_label_schema,
        "selector_levels": selectors,
        "controller_levels": controllers,
        "capacity_values": capacities,
        "capacity_policy": plan_manifest.get("capacity_policy"),
        "source_controller": plan_manifest.get("source_controller"),
        "source_plan_manifest": str(plan_manifest_path.resolve()),
        "source_plan_manifest_sha256": _sha256_file(plan_manifest_path),
        "build_root": str(build_root.resolve()),
        "output_dir": str(logical_output_dir),
        "event_count": expected_event_count,
        "event_id_sequence_sha256": expected_event_sha,
        "cell_count": len(formal_cells),
        "all_ready": True,
        "prefix_integrity_gate": "prefix_integrity_gate.json",
        "prefix_integrity_gate_sha256": _sha256_file(
            output_dir / "prefix_integrity_gate.json"
        ),
        "cells": formal_cells,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_plan_lattice(
    *,
    selectors: Sequence[str],
    capacities: Sequence[int],
    plan_rows_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    capacity_policy: str,
    source_controller: str,
) -> dict[str, Any]:
    event_ids: list[str] | None = None
    event_hasher = hashlib.sha256()
    adjacency_checks = 0
    plateau_checks = 0
    trace_order_field_by_selector: dict[str, str] = {}
    for selector in selectors:
        selector_rows = [plan_rows_by_cell[_cell_id(selector, k)] for k in capacities]
        selector_event_ids = [str(row.get("event_id") or "") for row in selector_rows[0]]
        if not selector_event_ids or len(selector_event_ids) != len(set(selector_event_ids)):
            raise CapacityMatrixError(f"{selector}: empty or duplicate event IDs")
        for rows in selector_rows[1:]:
            if [str(row.get("event_id") or "") for row in rows] != selector_event_ids:
                raise CapacityMatrixError(f"{selector}: K cells have different event order")
        if event_ids is None:
            event_ids = selector_event_ids
            for event_id in event_ids:
                event_hasher.update(event_id.encode("utf-8"))
                event_hasher.update(b"\0")
        elif selector_event_ids != event_ids:
            raise CapacityMatrixError("selectors have different event order")

        for row_index in range(len(selector_event_ids)):
            previous_indices: list[int] | None = None
            previous_uids: list[str] | None = None
            previous_evidence_ids: list[str] | None = None
            feasible_max: int | None = None
            for k, rows in zip(capacities, selector_rows):
                row = rows[row_index]
                controller_level = _capacity_level(k)
                trace_order_field = str(row.get("trace_order_field") or "")
                if (
                    str(row.get("selector_level") or "") != selector
                    or str(row.get("controller_level") or "") != controller_level
                    or str(row.get("capacity_policy") or "") != capacity_policy
                    or str(row.get("source_controller") or "") != source_controller
                    or not trace_order_field
                ):
                    raise CapacityMatrixError(
                        f"{selector}:{selector_event_ids[row_index]}: plan row metadata "
                        "disagrees with its cell"
                    )
                previous_trace_order_field = trace_order_field_by_selector.setdefault(
                    selector, trace_order_field
                )
                if previous_trace_order_field != trace_order_field:
                    raise CapacityMatrixError(
                        f"{selector}: trace_order_field changes across the prefix lattice"
                    )
                if int(row.get("requested_prefix_k", -1)) != k:
                    raise CapacityMatrixError(
                        f"{selector}:{selector_event_ids[row_index]}: cell/requested K mismatch"
                    )
                indices = _int_list(row.get("selected_indices"), "selected_indices")
                uids = _string_list(row.get("selected_candidate_uids"), "selected_candidate_uids")
                evidence_ids = _string_list(
                    row.get("selected_evidence_ids"), "selected_evidence_ids"
                )
                if not (len(indices) == len(uids) == len(evidence_ids)):
                    raise CapacityMatrixError("plan prefix identity lengths disagree")
                current_feasible_max = int(row.get("feasible_max_prefix_k", -1))
                if feasible_max is None:
                    feasible_max = current_feasible_max
                elif current_feasible_max != feasible_max:
                    raise CapacityMatrixError("feasible max K changes across one event lattice")
                if previous_indices is not None:
                    if not (
                        indices[: len(previous_indices)] == previous_indices
                        and uids[: len(previous_uids or [])] == previous_uids
                        and evidence_ids[: len(previous_evidence_ids or [])]
                        == previous_evidence_ids
                    ):
                        raise CapacityMatrixError("adjacent capacity cells are not strict prefixes")
                    delta = len(indices) - len(previous_indices)
                    if delta not in (0, 1):
                        raise CapacityMatrixError("adjacent capacity changes by more than one")
                    adjacency_checks += 1
                    plateau_checks += int(delta == 0)
                previous_indices = indices
                previous_uids = uids
                previous_evidence_ids = evidence_ids

    return {
        "event_count": len(event_ids or []),
        "event_id_sequence_sha256": event_hasher.hexdigest(),
        "adjacent_prefix_check_count": adjacency_checks,
        "plateau_check_count": plateau_checks,
        "trace_order_fields": dict(sorted(trace_order_field_by_selector.items())),
    }


def _validate_final_build_row(
    *,
    cell_id: str,
    capacity_k: int,
    plan: Mapping[str, Any],
    build: Mapping[str, Any],
    row_index: int,
) -> None:
    context = f"{cell_id}[{row_index}]"
    if bool(build.get("was_truncated")) or bool(build.get("evidence_text_truncated")):
        raise CapacityMatrixError(f"{context}: final build is truncated")
    if build.get("prompt_add_special_tokens") is not False:
        raise CapacityMatrixError(f"{context}: prompt_add_special_tokens must be false")
    if build.get("preserve_prompt_prefix") is not True:
        raise CapacityMatrixError(f"{context}: preserve_prompt_prefix must be true")
    if str(build.get("prompt_evidence_policy") or "") != "selected_set":
        raise CapacityMatrixError(f"{context}: final build did not use selected_set")
    selected_indices = _int_list(plan.get("selected_indices"), f"{context}:indices")
    selected_uids = _string_list(
        plan.get("selected_candidate_uids"), f"{context}:uids"
    )
    selected_evidence_ids = _string_list(
        plan.get("selected_evidence_ids"), f"{context}:evidence_ids"
    )
    evidence_count = int(build.get("evidence_count", -1))
    if evidence_count != len(selected_indices) or evidence_count > capacity_k:
        raise CapacityMatrixError(f"{context}: final evidence count disagrees with plan")
    candidates = build.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < evidence_count:
        raise CapacityMatrixError(f"{context}: final build candidates are incomplete")
    actual_uids = [str(candidate.get("candidate_uid") or "") for candidate in candidates[:evidence_count]]
    actual_evidence_ids = [
        str(candidate.get("evidence_id") or "") for candidate in candidates[:evidence_count]
    ]
    if actual_uids != selected_uids or actual_evidence_ids != selected_evidence_ids:
        raise CapacityMatrixError(f"{context}: final candidate identities disagree with plan")
    selector_trace = build.get("selector_trace")
    if not isinstance(selector_trace, Mapping):
        raise CapacityMatrixError(f"{context}: final build has no selector_trace")
    if _int_list(selector_trace.get("selected_indices"), f"{context}:selector_trace") != selected_indices:
        raise CapacityMatrixError(f"{context}: selector_trace indices disagree with plan")
    replayed_plan = selector_trace.get("selection_plan")
    if not isinstance(replayed_plan, Mapping):
        raise CapacityMatrixError(f"{context}: selector_trace has no replayed selection plan")
    if int(replayed_plan.get("requested_prefix_k", -1)) != capacity_k:
        raise CapacityMatrixError(f"{context}: replayed plan requested K mismatch")
    if int(build.get("prompt_token_count", 0)) != len(build.get("prompt_input_ids") or []):
        raise CapacityMatrixError(f"{context}: prompt token count mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CapacityMatrixError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CapacityMatrixError(f"{path} is not a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CapacityMatrixError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise CapacityMatrixError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _unique_strings(value: Any, context: str) -> list[str]:
    output = _string_list(value, context)
    if not output or len(output) != len(set(output)) or any(not item for item in output):
        raise CapacityMatrixError(f"{context} must contain unique non-empty strings")
    return output


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CapacityMatrixError(f"{context} must be a string array")
    return [str(item) for item in value]


def _int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise CapacityMatrixError(f"{context} must be an integer array")
    return [int(item) for item in value]


def _capacity_level(k: int) -> str:
    return f"prefix_k{k:02d}"


def _cell_id(selector: str, k: int) -> str:
    return f"{selector}__{_capacity_level(k)}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _promote_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise CapacityMatrixError(f"refusing to replace existing output: {target}")
        backup = target.with_name(f".{target.name}.old.{os.getpid()}")
        if backup.exists():
            raise CapacityMatrixError(f"backup directory already exists: {backup}")
        target.replace(backup)
    try:
        staging.replace(target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
