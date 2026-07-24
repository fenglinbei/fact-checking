from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.phase5_selectors.build.prepare_no_map_structure_fixed5_matrix import prepare


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(event_id: str, prefix: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "claim": f"claim {event_id}",
        "target": "Label: A",
        "target_token_count": 1,
        "label_schema": "liar6",
        "gold_id": 0,
        "gold_label": "pants-fire",
        "label": "pants-fire",
        "evidence_count": 5,
        "was_truncated": False,
        "evidence_text_truncated": False,
        "prompt_input_ids": [1, 2, 3, 4, 5],
        "prompt_token_count": 5,
        "candidates": [
            {"candidate_uid": f"{prefix}-{event_id}-{index}", "text": f"text {index}"}
            for index in range(5)
        ],
    }


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_freezes_exact_1234_support_and_audits_n_s_difference(tmp_path: Path) -> None:
    s_build = tmp_path / "s.jsonl"
    n_build = tmp_path / "n.jsonl"
    event_ids = [f"e-{index}" for index in range(1234)]
    _jsonl(s_build, [_row(event_id, "s") for event_id in event_ids])
    _jsonl(n_build, [_row(event_id, "n") for event_id in event_ids] + [_row(f"extra-{index}", "n") for index in range(40)])
    report = tmp_path / "build_report.json"
    report.write_text(
        json.dumps(
            {
                "val_only": True,
                "built_splits": ["val"],
                "selection_mode": "trace",
                "trace_order_field": "display_ordered_indices",
                "prompt_evidence": {"policy": "fixed_topk", "min_evidence_count": 5, "max_evidence_count": 5},
                "splits": {"val": {"n_rows": 1274}},
            }
        ) + "\n",
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "structure-mechanism-gate-matrix-v0.1",
                "split": "val", "expected_k": 5, "event_count": 1234,
                "event_id_sequence_sha256": hashlib.sha256(b"".join(event.encode() + b"\0" for event in event_ids)).hexdigest(),
                "all_ready": True,
                "cells": [{"cell_id": "stateful__fixed5", "build_sha256": _sha(s_build)}],
            }
        ) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "matrix"
    manifest = prepare(
        no_map_build=n_build,
        no_map_build_report=report,
        structure_build=s_build,
        source_matrix_manifest=source_manifest,
        output_dir=output,
        force=False,
    )
    assert [cell["cell_id"] for cell in manifest["cells"]] == ["N_fixed5", "S_fixed5"]
    assert manifest["event_count"] == 1234
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["input_difference"]["event_count"] == 1234
    assert audit["input_difference"]["visible_sequence_difference_rate"] == 1.0
    assert audit["training_vs_diagnostic_policy"]["natural_k_build_reused_as_fixed_k"] is False
    assert audit["standard_clean_results_audit_slot_mutated"] is False
