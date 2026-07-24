from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.analyze.summarize_no_map_structure_fixed5_crossover import summarize


CELLS = ("N_fixed5", "S_fixed5")
EVENT_SHA = "a" * 64


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_source(root: Path) -> None:
    audit_path = root / "audit.json"
    _write(
        audit_path,
        {
            "schema_version": "no-map-structure-fixed5-input-difference-audit-v0.1",
            "status": "complete", "passed": True, "standard_clean_results_audit_slot_mutated": False,
            "input_difference": {"event_count": 1234, "visible_sequence_difference_rate": 0.5},
        },
    )
    _write(
        root / "manifest.json",
        {
            "schema_version": "no-map-structure-fixed5-matrix-v0.1", "all_ready": True,
            "split": "val", "expected_k": 5, "event_count": 1234,
            "audit_sha256": _sha(audit_path), "cells": [{"cell_id": cell} for cell in CELLS],
        },
    )


def _verifier(root: Path, *, matrix_sha: str, adapter: str, n: float, s: float) -> None:
    input_path = root / "input/manifest.json"
    _write(input_path, {"status": "complete", "split": "val", "cell_count": 2, "event_count": 1234, "event_id_sequence_sha256": EVENT_SHA, "matrix_manifest_sha256": matrix_sha, "cells": [{"cell_id": cell} for cell in CELLS]})
    raw_path = root / "raw_logits/manifest.json"
    _write(raw_path, {"status": "complete", "split": "val", "input_manifest_sha256": _sha(input_path), "checkpoint": {"checkpoint_name": "checkpoint-800", "adapter_sha256": adapter, "run_dir": str(root / "train")}})
    _write(root / "materialized/matrix_manifest.json", {"status": "complete", "split": "val", "diagnostic_only": True, "cell_count": 2, "input_manifest_sha256": _sha(input_path), "raw_logits_manifest_sha256": _sha(raw_path), "cells": [{"cell_id": "N_fixed5", "macro_f1": n, "accuracy": 0.4, "eval_ce_loss": 1.5, "num_samples": 1234}, {"cell_id": "S_fixed5", "macro_f1": s, "accuracy": 0.4, "eval_ce_loss": 1.5, "num_samples": 1234}]})


def test_summary_uses_explicit_n_s_semantics_and_exact_contrasts(tmp_path: Path) -> None:
    source = tmp_path / "matrix"
    _matrix_source(source)
    matrix_sha = _sha(source / "manifest.json")
    _verifier(tmp_path / "vn", matrix_sha=matrix_sha, adapter="1" * 64, n=0.31, s=0.32)
    _verifier(tmp_path / "vs", matrix_sha=matrix_sha, adapter="2" * 64, n=0.33, s=0.36)
    result = summarize(verifier_n_dir=tmp_path / "vn", verifier_s_dir=tmp_path / "vs", matrix_dir=source)
    assert result["prompt_cells"] == {"N": "N_fixed5", "S": "S_fixed5"}
    assert result["macro_f1_matrix"] == {"V_N": {"N": 0.31, "S": 0.32}, "V_S": {"N": 0.33, "S": 0.36}}
    assert result["contrasts"]["matched_mean_minus_crossed_mean"] == pytest.approx(0.01)
    assert result["contrasts"]["difference_in_differences"] == pytest.approx(0.02)
    assert result["interpretation_contract"]["natural_minmax_k_reused_as_fixed_k"] is False
