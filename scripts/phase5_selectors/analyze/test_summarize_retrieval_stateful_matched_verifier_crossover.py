from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.analyze.summarize_matched_verifier_crossover import (
    CrossoverSummaryError,
)
from scripts.phase5_selectors.analyze.summarize_retrieval_stateful_matched_verifier_crossover import (
    summarize_crossover,
)


CELLS = ("retrieval__fixed5", "stateful__fixed5")
EVENT_SHA = "a" * 64
MATRIX_SHA = "b" * 64


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(
    root: Path,
    *,
    adapter_sha: str,
    r_macro_f1: float,
    s_macro_f1: float,
    checkpoint: str = "checkpoint-800",
    event_sha: str = EVENT_SHA,
) -> None:
    input_path = root / "input" / "manifest.json"
    _write_json(
        input_path,
        {
            "status": "complete",
            "split": "val",
            "cell_count": 2,
            "event_count": 3,
            "event_id_sequence_sha256": event_sha,
            "matrix_manifest_sha256": MATRIX_SHA,
            "cells": [{"cell_id": cell_id} for cell_id in CELLS],
        },
    )
    raw_path = root / "raw_logits" / "manifest.json"
    _write_json(
        raw_path,
        {
            "status": "complete",
            "split": "val",
            "input_manifest_sha256": _sha(input_path),
            "checkpoint": {
                "checkpoint_name": checkpoint,
                "adapter_sha256": adapter_sha,
                "run_dir": str(root / "train"),
            },
        },
    )
    result_path = root / "materialized" / "matrix_manifest.json"
    _write_json(
        result_path,
        {
            "status": "complete",
            "split": "val",
            "diagnostic_only": True,
            "cell_count": 2,
            "input_manifest_sha256": _sha(input_path),
            "raw_logits_manifest_sha256": _sha(raw_path),
            "cells": [
                {
                    "cell_id": "retrieval__fixed5",
                    "macro_f1": r_macro_f1,
                    "accuracy": 0.4,
                    "eval_ce_loss": 1.5,
                    "num_samples": 3,
                },
                {
                    "cell_id": "stateful__fixed5",
                    "macro_f1": s_macro_f1,
                    "accuracy": 0.4,
                    "eval_ce_loss": 1.5,
                    "num_samples": 3,
                },
            ],
        },
    )


def test_summarize_crossover_emits_fixed_retrieval_stateful_contrasts(
    tmp_path: Path,
) -> None:
    verifier_r = tmp_path / "verifier_r"
    verifier_s = tmp_path / "verifier_s"
    _matrix(
        verifier_r,
        adapter_sha="1" * 64,
        r_macro_f1=0.37,
        s_macro_f1=0.35,
    )
    _matrix(
        verifier_s,
        adapter_sha="2" * 64,
        r_macro_f1=0.34,
        s_macro_f1=0.38,
    )

    summary = summarize_crossover(
        verifier_r_dir=verifier_r,
        verifier_s_dir=verifier_s,
    )

    assert summary["split"] == "val"
    assert summary["checkpoint"] == "checkpoint-800"
    assert summary["macro_f1_matrix"] == {
        "V_R": {"R": 0.37, "S": 0.35},
        "V_S": {"R": 0.34, "S": 0.38},
    }
    assert summary["contrasts"]["matched_mean_minus_crossed_mean"] == pytest.approx(
        0.03
    )
    assert summary["contrasts"]["difference_in_differences"] == pytest.approx(0.06)


def test_summarize_crossover_rejects_different_common_support(tmp_path: Path) -> None:
    verifier_r = tmp_path / "verifier_r"
    verifier_s = tmp_path / "verifier_s"
    _matrix(
        verifier_r,
        adapter_sha="1" * 64,
        r_macro_f1=0.37,
        s_macro_f1=0.35,
    )
    _matrix(
        verifier_s,
        adapter_sha="2" * 64,
        r_macro_f1=0.34,
        s_macro_f1=0.38,
        event_sha="c" * 64,
    )

    with pytest.raises(CrossoverSummaryError, match="same frozen validation input"):
        summarize_crossover(verifier_r_dir=verifier_r, verifier_s_dir=verifier_s)


def test_summarize_crossover_rejects_best_alias(tmp_path: Path) -> None:
    verifier_r = tmp_path / "verifier_r"
    verifier_s = tmp_path / "verifier_s"
    _matrix(
        verifier_r,
        adapter_sha="1" * 64,
        r_macro_f1=0.37,
        s_macro_f1=0.35,
        checkpoint="best",
    )
    _matrix(
        verifier_s,
        adapter_sha="2" * 64,
        r_macro_f1=0.34,
        s_macro_f1=0.38,
    )

    with pytest.raises(CrossoverSummaryError, match="checkpoint-800"):
        summarize_crossover(verifier_r_dir=verifier_r, verifier_s_dir=verifier_s)
