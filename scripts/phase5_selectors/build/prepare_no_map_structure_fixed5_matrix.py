#!/usr/bin/env python3
"""Freeze a LIAR-RAW N_fixed5/S_fixed5 validation matrix on 1,234 events."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

try:
    from .prepare_structure_mechanism_gate import (
        MechanismGateError, _comparison_stats, _copy_indexed_rows,
        _event_sequence_sha256, _index_build, _promote_directory,
        _sha256_file, _validate_common_rows, _write_json,
    )
except ImportError:
    from prepare_structure_mechanism_gate import (  # type: ignore[no-redef]
        MechanismGateError, _comparison_stats, _copy_indexed_rows,
        _event_sequence_sha256, _index_build, _promote_directory,
        _sha256_file, _validate_common_rows, _write_json,
    )


EXPECTED_CELLS = ("N_fixed5", "S_fixed5")
EXPECTED_K = 5
EXPECTED_EVENTS = 1234


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MechanismGateError(f"expected JSON object: {path}")
    return payload


def prepare(
    *,
    no_map_build: Path,
    no_map_build_report: Path,
    structure_build: Path,
    source_matrix_manifest: Path,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    source_manifest = _read_json(source_matrix_manifest)
    if (
        source_manifest.get("schema_version") != "structure-mechanism-gate-matrix-v0.1"
        or source_manifest.get("split") != "val"
        or int(source_manifest.get("expected_k", -1)) != EXPECTED_K
        or int(source_manifest.get("event_count", -1)) != EXPECTED_EVENTS
        or source_manifest.get("all_ready") is not True
    ):
        raise MechanismGateError("source structure matrix violates frozen val/K5/n=1234 contract")
    source_s = [
        cell for cell in source_manifest.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_id") == "stateful__fixed5"
    ]
    if len(source_s) != 1:
        raise MechanismGateError("source structure matrix lacks exactly one stateful__fixed5 cell")
    expected_s_sha = str(source_s[0].get("build_sha256") or "")
    if _sha256_file(structure_build) != expected_s_sha:
        raise MechanismGateError("S_fixed5 build SHA differs from frozen source matrix")

    report = _read_json(no_map_build_report)
    policy = report.get("prompt_evidence") or {}
    val_report = (report.get("splits") or {}).get("val") or {}
    if (
        report.get("val_only") is not True
        or sorted(report.get("built_splits") or []) != ["val"]
        or report.get("selection_mode") != "trace"
        or report.get("trace_order_field") != "display_ordered_indices"
        or policy.get("policy") != "fixed_topk"
        or int(policy.get("min_evidence_count", -1)) != EXPECTED_K
        or int(policy.get("max_evidence_count", -1)) != EXPECTED_K
        or int(val_report.get("n_rows", -1)) != 1274
    ):
        raise MechanismGateError("N_fixed5 build report is not val-only fixed_topk K=5")

    indexes = {
        "N_fixed5": _index_build(no_map_build, expected_k=EXPECTED_K, cell_id="N_fixed5"),
        "S_fixed5": _index_build(structure_build, expected_k=EXPECTED_K, cell_id="S_fixed5"),
    }
    reference = indexes["S_fixed5"]
    if len(reference.event_order) != EXPECTED_EVENTS:
        raise MechanismGateError("S_fixed5 source does not contain exactly 1,234 rows")
    event_ids = [
        event_id for event_id in reference.event_order
        if event_id in indexes["N_fixed5"].eligible_event_ids
    ]
    if len(event_ids) != EXPECTED_EVENTS:
        raise MechanismGateError(
            "N_fixed5 is not exact-K/non-truncated on all 1,234 frozen S events"
        )
    label_schema = _validate_common_rows(
        indexes=indexes,
        event_ids=event_ids,
        stateful_id="S_fixed5",
        expected_k=EXPECTED_K,
    )
    if label_schema != "liar6":
        raise MechanismGateError(f"unexpected label schema: {label_schema}")
    event_sha = _event_sequence_sha256(event_ids)
    source_event_sha = str(source_manifest.get("event_id_sequence_sha256") or "")
    if event_sha != source_event_sha:
        raise MechanismGateError("N/S event order differs from frozen 1,234-event source")
    difference = _comparison_stats(indexes["N_fixed5"], indexes["S_fixed5"], event_ids)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise MechanismGateError(f"staging directory exists: {staging}")
    staging.mkdir(parents=True)
    promoted = False
    try:
        cells = []
        for cell_id in EXPECTED_CELLS:
            target = staging / cell_id / "build/build_val.jsonl"
            _copy_indexed_rows(indexes[cell_id], event_ids, target)
            sha = _sha256_file(target)
            cells.append(
                {
                    "cell_id": cell_id,
                    "cell_kind": "prompt_policy",
                    "prompt_policy": "no_map" if cell_id == "N_fixed5" else "stateful_structure",
                    "ready": True,
                    "row_count": EXPECTED_EVENTS,
                    "event_id_sequence_sha256": event_sha,
                    "build_file": str((output_dir / cell_id / "build/build_val.jsonl").resolve()),
                    "build_sha256": sha,
                    "build_sha": sha,
                }
            )
        audit = {
            "schema_version": "no-map-structure-fixed5-input-difference-audit-v0.1",
            "status": "complete",
            "passed": True,
            "split": "val",
            "expected_k": EXPECTED_K,
            "event_count": EXPECTED_EVENTS,
            "event_id_sequence_sha256": event_sha,
            "cells": {"no_map": "N_fixed5", "structure": "S_fixed5"},
            "input_difference": difference,
            "source_builds": {
                "N_fixed5": {"path": str(no_map_build.resolve()), "sha256": indexes["N_fixed5"].file_sha256, "source_rows": len(indexes["N_fixed5"].event_order)},
                "S_fixed5": {"path": str(structure_build.resolve()), "sha256": indexes["S_fixed5"].file_sha256, "source_rows": len(indexes["S_fixed5"].event_order)},
            },
            "source_structure_matrix": {
                "path": str(source_matrix_manifest.resolve()),
                "sha256": _sha256_file(source_matrix_manifest),
            },
            "no_map_fixed5_build_report": {
                "path": str(no_map_build_report.resolve()),
                "sha256": _sha256_file(no_map_build_report),
            },
            "training_vs_diagnostic_policy": {
                "V_N_training": "natural minmax K=5..10",
                "diagnostic_inputs": "strict fixed_topk K=5",
                "natural_k_build_reused_as_fixed_k": False,
            },
            "standard_clean_results_audit_slot_mutated": False,
        }
        _write_json(staging / "audit.json", audit)
        manifest = {
            "schema_version": "no-map-structure-fixed5-matrix-v0.1",
            "matrix_kind": "no_map_structure_matched_verifier_crossover",
            "split": "val",
            "label_schema": "liar6",
            "expected_k": EXPECTED_K,
            "event_count": EXPECTED_EVENTS,
            "event_id_sequence_sha256": event_sha,
            "cell_count": 2,
            "all_ready": True,
            "checkpoint_contract": {"checkpoint": "checkpoint-800", "selection": "fixed_step", "split": "val", "test_allowed": False, "best_alias_allowed": False},
            "audit_file": str((output_dir / "audit.json").resolve()),
            "audit_sha256": _sha256_file(staging / "audit.json"),
            "cells": cells,
        }
        _write_json(staging / "manifest.json", manifest)
        _promote_directory(staging, output_dir, force=force)
        promoted = True
        return manifest
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-map-build", type=Path, required=True)
    parser.add_argument("--no-map-build-report", type=Path, required=True)
    parser.add_argument("--structure-build", type=Path, required=True)
    parser.add_argument("--source-matrix-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(
        no_map_build=args.no_map_build,
        no_map_build_report=args.no_map_build_report,
        structure_build=args.structure_build,
        source_matrix_manifest=args.source_matrix_manifest,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(
        "[no-map-fixed5-prepare] "
        f"cells=N_fixed5,S_fixed5 events={manifest['event_count']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
