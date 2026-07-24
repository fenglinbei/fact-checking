from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.analyze.summarize_matched_verifier_crossover import (
    CrossoverSummaryError,
)
from scripts.phase5_selectors.analyze.summarize_vo_retrieval_stateful_diagnostic import (
    _write_markdown,
    summarize_diagnostic,
)


CELLS = ("retrieval__fixed5", "stateful__fixed5")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(root: Path, *, checkpoint: str = "checkpoint-800") -> None:
    input_path = root / "input" / "manifest.json"
    _write_json(
        input_path,
        {
            "status": "complete",
            "split": "val",
            "cell_count": 2,
            "event_count": 3,
            "event_id_sequence_sha256": "a" * 64,
            "matrix_manifest_sha256": "b" * 64,
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
                "adapter_sha256": "1" * 64,
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
                    "macro_f1": 0.35,
                    "accuracy": 0.40,
                    "eval_ce_loss": 1.50,
                    "num_samples": 3,
                },
                {
                    "cell_id": "stateful__fixed5",
                    "macro_f1": 0.38,
                    "accuracy": 0.42,
                    "eval_ce_loss": 1.40,
                    "num_samples": 3,
                },
            ],
        },
    )


def test_summarize_vo_diagnostic_emits_explicit_s_minus_r_json_and_markdown(
    tmp_path: Path,
) -> None:
    verifier_o = tmp_path / "verifier_o"
    _matrix(verifier_o)

    summary = summarize_diagnostic(verifier_o_dir=verifier_o)

    assert summary["checkpoint"] == "checkpoint-800"
    assert summary["metrics"] == {
        "V_O_on_R": {
            "macro_f1": 0.35,
            "accuracy": 0.40,
            "eval_ce_loss": 1.50,
            "num_samples": 3,
        },
        "V_O_on_S": {
            "macro_f1": 0.38,
            "accuracy": 0.42,
            "eval_ce_loss": 1.40,
            "num_samples": 3,
        },
    }
    assert summary["primary_contrast"] == {
        "name": "V_O(S)-V_O(R)",
        "metric": "macro_f1",
        "value": pytest.approx(0.03),
        "higher_is_better": True,
    }
    assert summary["contrasts"]["V_O(S)-V_O(R)_accuracy"] == pytest.approx(0.02)

    markdown_path = tmp_path / "summary.md"
    _write_markdown(markdown_path, summary)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "V_O(S)-V_O(R) Macro-F1: `+0.030000`" in markdown
    assert "R: retrieval order" in markdown
    assert "S: structure-induced" in markdown


def test_summarize_vo_diagnostic_rejects_best_alias(tmp_path: Path) -> None:
    verifier_o = tmp_path / "verifier_o"
    _matrix(verifier_o, checkpoint="best")

    with pytest.raises(CrossoverSummaryError, match="checkpoint-800"):
        summarize_diagnostic(verifier_o_dir=verifier_o)
