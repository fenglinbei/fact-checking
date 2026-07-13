from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sft import paired_factorial_inference as paired


LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]


def _synthetic_matrix(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    matrix_dir = run_root / "materialized"
    raw_dir = run_root / "raw_logits"
    raw_dir.mkdir(parents=True)
    scoring_fingerprint = "synthetic-scoring"
    raw_manifest = {
        "schema_version": "synthetic-raw",
        "status": "complete",
        "num_labels": len(LABELS),
        "labels": LABELS,
        "scoring_fingerprint": scoring_fingerprint,
    }
    paired._write_json(raw_dir / "manifest.json", raw_manifest)

    event_ids = [f"event-{index}" for index in range(18)]
    gold_ids = np.asarray([index % len(LABELS) for index in range(len(event_ids))])
    cells = []
    fixed_payloads: dict[str, tuple[np.ndarray, list[str], np.ndarray, np.ndarray]] = {}
    for selector_index, selector in enumerate(paired.SELECTOR_LEVELS):
        for controller_index, controller in enumerate(paired.CONTROLLER_LEVELS):
            cell_id = f"{selector}__{controller}"
            pred_ids = gold_ids.copy()
            flip_stride = 3 + ((selector_index + controller_index) % 4)
            flip_indices = np.arange(selector_index + controller_index, len(event_ids), flip_stride)
            pred_ids[flip_indices] = (pred_ids[flip_indices] + 1 + selector_index % 2) % len(
                LABELS
            )
            evidence = np.full(len(event_ids), 5.0 + controller_index)
            tokens = np.full(len(event_ids), 500.0 + 50.0 * controller_index)
            prompt_hashes = [f"prompt:{cell_id}:{event_id}" for event_id in event_ids]

            if controller == "fixed5":
                fixed_payloads[selector] = (
                    pred_ids.copy(),
                    list(prompt_hashes),
                    evidence.copy(),
                    tokens.copy(),
                )
            if controller == "ordinal_replay_minmax5_10" and selector in {
                "baces_exact",
                "ordinal_coverage_greedy",
            }:
                pred_ids, prompt_hashes, evidence, tokens = fixed_payloads[selector]

            prediction_rows = [
                {
                    "sample_idx": sample_idx,
                    "event_id": event_id,
                    "cell_id": cell_id,
                    "selector_level": selector,
                    "controller_level": controller,
                    "gold_id": int(gold_ids[sample_idx]),
                    "gold_label": LABELS[int(gold_ids[sample_idx])],
                    "pred_id": int(pred_ids[sample_idx]),
                    "pred_label": LABELS[int(pred_ids[sample_idx])],
                    "scoring_fingerprint": scoring_fingerprint,
                    "evidence_count": float(evidence[sample_idx]),
                    "prompt_token_count": float(tokens[sample_idx]),
                    "prompt_input_ids_sha256": prompt_hashes[sample_idx],
                }
                for sample_idx, event_id in enumerate(event_ids)
            ]
            predictions_path = matrix_dir / "cells" / cell_id / "label_token" / "val_predictions.jsonl"
            paired._write_jsonl(predictions_path, prediction_rows)
            point, _ = paired._point_metrics(
                gold_ids, np.asarray(pred_ids), n_labels=len(LABELS)
            )
            metrics = {
                "num_samples": len(event_ids),
                "parse_error_rate": 0.0,
                "accuracy": point["accuracy"],
                "macro_f1": point["macro_f1"],
                "selector_level": selector,
                "controller_level": controller,
            }
            metrics_path = predictions_path.with_name("metrics.json")
            paired._write_json(metrics_path, metrics)
            cells.append(
                {
                    "cell_id": cell_id,
                    "selector_level": selector,
                    "controller_level": controller,
                    "num_samples": len(event_ids),
                    "accuracy": point["accuracy"],
                    "macro_f1": point["macro_f1"],
                    "parse_error_rate": 0.0,
                    "metrics_file": str(metrics_path.relative_to(matrix_dir)),
                    "metrics_sha256": paired._sha256_file(metrics_path),
                    "predictions_file": str(predictions_path.relative_to(matrix_dir)),
                    "predictions_sha256": paired._sha256_file(predictions_path),
                }
            )

    gate_path = matrix_dir / "equivalence_gate.json"
    paired._write_json(gate_path, {"status": "passed", "passed": True})
    matrix_manifest = {
        "schema_version": "synthetic-matrix-v0",
        "status": "complete",
        "diagnostic_only": False,
        "split": "val",
        "cell_count": len(cells),
        "raw_logits_manifest": str(raw_dir / "manifest.json"),
        "raw_logits_manifest_sha256": paired._sha256_file(raw_dir / "manifest.json"),
        "raw_logits_scoring_fingerprint": scoring_fingerprint,
        "raw_logits_execution_fingerprint": "synthetic-execution",
        "checkpoint": {"adapter_sha256": "synthetic-adapter"},
        "equivalence_gate": gate_path.name,
        "equivalence_gate_sha256": paired._sha256_file(gate_path),
        "cells": cells,
    }
    matrix_manifest_path = matrix_dir / "matrix_manifest.json"
    paired._write_json(matrix_manifest_path, matrix_manifest)
    return matrix_manifest_path, run_root / "paired_inference"


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = paired.holm_adjust([0.01, 0.03, 0.04])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_exact_mcnemar_and_stable_seed() -> None:
    assert paired._mcnemar_exact_p(3, 0) == pytest.approx(0.25)
    assert paired._mcnemar_exact_p(0, 0) == 1.0
    assert paired._stable_seed(7, "family", "comparison", "permutation") == paired._stable_seed(
        7, "family", "comparison", "permutation"
    )
    assert paired._stable_seed(7, "family", "comparison", "permutation") != paired._stable_seed(
        7, "family", "other", "permutation"
    )


def test_materializer_publishes_formal_atomic_artifact(tmp_path: Path) -> None:
    matrix_manifest_path, output_dir = _synthetic_matrix(tmp_path)

    result = paired.materialize_paired_inference(
        matrix_manifest_path=matrix_manifest_path,
        output_dir=output_dir,
        bootstrap_samples=40,
        permutation_samples=40,
        seed=20260713,
    )

    assert result["status"] == "complete"
    assert result["comparison_count"] == 33
    assert result["sample_count"] == 18
    assert result["alignment"]["all_cells_exactly_aligned"] is True
    assert len(result["sanity_checks"]) == 2
    assert not list(output_dir.parent.glob(f".{output_dir.name}.tmp.*"))
    for artifact in result["artifacts"].values():
        path = output_dir / artifact["path"]
        assert paired._sha256_file(path) == artifact["sha256"]

    comparison_rows = [
        json.loads(line)
        for line in (output_dir / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    identical = next(
        row
        for row in comparison_rows
        if row["comparison_id"]
        == "baces_exact__ordinal_replay_minmax5_10_minus_fixed5"
    )
    for metric in paired.METRIC_NAMES:
        payload = identical["metrics"][metric]
        assert payload["delta_a_minus_b"] == 0.0
        assert payload["ordinary_paired_bootstrap"]["ci"] == {
            "low": 0.0,
            "high": 0.0,
        }
        assert payload["paired_permutation"]["p_value"] == 1.0

    reused = paired.materialize_paired_inference(
        matrix_manifest_path=matrix_manifest_path,
        output_dir=output_dir,
        bootstrap_samples=40,
        permutation_samples=40,
        seed=20260713,
    )
    assert reused["created_at"] == result["created_at"]

    comparisons_sha = paired._sha256_file(output_dir / "comparisons.jsonl")
    regenerated = paired.materialize_paired_inference(
        matrix_manifest_path=matrix_manifest_path,
        output_dir=output_dir,
        bootstrap_samples=40,
        permutation_samples=40,
        seed=20260713,
        force=True,
    )
    assert paired._sha256_file(output_dir / "comparisons.jsonl") == comparisons_sha
    assert regenerated["alignment"] == result["alignment"]


def test_materializer_rejects_prediction_drift(tmp_path: Path) -> None:
    matrix_manifest_path, output_dir = _synthetic_matrix(tmp_path)
    paired.materialize_paired_inference(
        matrix_manifest_path=matrix_manifest_path,
        output_dir=output_dir,
        bootstrap_samples=10,
        permutation_samples=10,
    )
    matrix = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    prediction_path = matrix_manifest_path.parent / matrix["cells"][0]["predictions_file"]
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(paired.PairedInferenceError, match="Prediction SHA mismatch"):
        paired.materialize_paired_inference(
            matrix_manifest_path=matrix_manifest_path,
            output_dir=output_dir,
            bootstrap_samples=10,
            permutation_samples=10,
        )
