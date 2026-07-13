from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sft.label_token_matrix_infer import (
    MatrixValidationError,
    RAW_LOGITS_SCHEMA_VERSION,
    _checkpoint_provenance,
    _event_sequence_sha256,
    _load_jsonl,
    _native_batch1_losses,
    _prompt_ids_sha256,
    _sha256_file,
    build_cell_metrics_from_raw_logits,
    compare_native_equivalence,
    fanout_matrix,
    prepare_matrix,
    restore_gathered_logits,
)
from sft.label_token_trainer import _compute_label_token_losses


def _row(
    event_id: str,
    prompt_ids: list[int],
    *,
    gold_id: int,
    evidence_count: int = 1,
) -> dict:
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    letters = ["A", "B", "C", "D", "E", "F"]
    prompt = f"prompt:{event_id}:{','.join(str(value) for value in prompt_ids)}"
    return {
        "event_id": event_id,
        "claim": f"claim:{event_id}",
        "prompt": prompt,
        "target": f"Label: {letters[gold_id]}",
        "prompt_input_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "target_token_count": 3,
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "was_truncated": False,
        "evidence_text_truncated": False,
        "evidence_count": evidence_count,
        "gold_id": gold_id,
        "gold_label": labels[gold_id],
        "gold_explain": f"explain:{event_id}",
        "label_schema": "liar6",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _matrix_fixture(tmp_path: Path, *, conflict: bool = False) -> tuple[Path, Path, Path]:
    build_root = tmp_path / "build-root"
    output_dir = tmp_path / "output"
    cell_a = "selector_a__controller"
    cell_b = "selector_b__controller"
    rows_a = [_row("e0", [1, 2], gold_id=0), _row("e1", [3], gold_id=1)]
    rows_b = [
        _row("e0", [1, 2], gold_id=1 if conflict else 0, evidence_count=2),
        _row("e1", [4], gold_id=1),
    ]
    _write_jsonl(build_root / cell_a / "build" / "build_val.jsonl", rows_a)
    _write_jsonl(build_root / cell_b / "build" / "build_val.jsonl", rows_b)
    manifest = {
        "schema_version": "synthetic-factorial",
        "all_ready": True,
        "cell_count": 2,
        "event_count": 2,
        "event_id_sequence_sha256": _event_sequence_sha256(["e0", "e1"]),
        "selector_levels": ["selector_a", "selector_b"],
        "controller_levels": ["controller"],
        "cells": [
            {
                "cell_id": cell_a,
                "selector_level": "selector_a",
                "controller_level": "controller",
                "ready": True,
                "row_count": 2,
            },
            {
                "cell_id": cell_b,
                "selector_level": "selector_b",
                "controller_level": "controller",
                "ready": True,
                "row_count": 2,
            },
        ],
    }
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, build_root, output_dir


def test_prepare_deduplicates_inputs_without_cross_cell_sample_collision(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)

    manifest = prepare_matrix(
        matrix_manifest_path=manifest_path,
        build_root=build_root,
        output_dir=output_dir,
        split="val",
        label_prefix="Label:",
    )

    assert manifest["reference_count"] == 4
    assert manifest["unique_prompt_count"] == 3
    assert manifest["duplicate_reference_count"] == 1
    mappings_a = _load_jsonl(output_dir / "input" / "cells" / "selector_a__controller.jsonl")
    mappings_b = _load_jsonl(output_dir / "input" / "cells" / "selector_b__controller.jsonl")
    assert [row["cell_sample_idx"] for row in mappings_a] == [0, 1]
    assert [row["cell_sample_idx"] for row in mappings_b] == [0, 1]
    assert mappings_a[0]["unique_idx"] == mappings_b[0]["unique_idx"]
    assert mappings_a[1]["unique_idx"] != mappings_b[1]["unique_idx"]


def test_prepare_rejects_same_input_ids_with_conflicting_gold_metadata(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path, conflict=True)

    with pytest.raises(MatrixValidationError, match="conflicting strict metadata"):
        prepare_matrix(
            matrix_manifest_path=manifest_path,
            build_root=build_root,
            output_dir=output_dir,
            split="val",
            label_prefix="Label:",
        )


def test_prepare_rejects_mixed_label_schema_when_manifest_omits_schema(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)
    path = build_root / "selector_b__controller" / "build" / "build_val.jsonl"
    rows = _load_jsonl(path)
    rows[1].update(
        {
            "label_schema": "rawfc3",
            "gold_id": 0,
            "gold_label": "false",
            "target": "Label: A",
        }
    )
    _write_jsonl(path, rows)

    with pytest.raises(MatrixValidationError, match="matrix schema"):
        prepare_matrix(
            matrix_manifest_path=manifest_path,
            build_root=build_root,
            output_dir=output_dir,
            split="val",
            label_prefix="Label:",
        )


def test_prepare_rejects_duplicate_factor_coordinates(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = dict(manifest["cells"][0])
    duplicate["cell_id"] = "selector_a__duplicate_controller_coordinate"
    manifest["cells"].append(duplicate)
    manifest["cell_count"] = len(manifest["cells"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="duplicate selector/controller"):
        prepare_matrix(
            matrix_manifest_path=manifest_path,
            build_root=build_root,
            output_dir=output_dir,
            split="val",
            label_prefix="Label:",
        )


def test_prepare_rejects_build_drift_from_matrix_manifest(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cells"][0]["build_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="source build SHA disagrees"):
        prepare_matrix(
            matrix_manifest_path=manifest_path,
            build_root=build_root,
            output_dir=output_dir,
            split="val",
            label_prefix="Label:",
        )


def test_capacity_cell_metadata_is_preserved_through_fanout(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for cell in manifest["cells"]:
        cell.update(
            {
                "capacity_k": 3,
                "capacity_policy": "strict_ordered_prefix",
                "source_order_cell": f"{cell['selector_level']}__canonical",
            }
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prepared = prepare_matrix(
        matrix_manifest_path=manifest_path,
        build_root=build_root,
        output_dir=output_dir,
        split="val",
        label_prefix="Label:",
    )
    cell = prepared["cells"][0]
    assert cell["capacity_k"] == 3
    assert cell["capacity_policy"] == "strict_ordered_prefix"
    assert cell["source_order_cell"] == "selector_a__canonical"

    unique_rows = _load_jsonl(output_dir / "input" / "unique_rows.jsonl")
    mappings = _load_jsonl(output_dir / "input" / cell["mapping_file"])
    logits = np.zeros((len(unique_rows), 6), dtype=np.float32)
    logits[:, 0] = 1.0
    gold = np.asarray([int(row["gold_id"]) for row in unique_rows], dtype=np.int64)
    raw_manifest = {
        "labels": ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
        "letter_order": ["A", "B", "C", "D", "E", "F"],
        "label_schema": "liar6",
        "label_token_meta": {"label_prefix": "Label:"},
        "checkpoint": {"checkpoint_name": "best"},
        "split": "val",
        "scoring_fingerprint": "fingerprint",
        "loss_replay_contract": {
            "native_eval_batch_size": 1,
            "ordinal_loss": {"enabled": False},
        },
    }
    _, metrics, records = build_cell_metrics_from_raw_logits(
        cell=cell,
        mappings=mappings,
        unique_rows=unique_rows,
        raw_logits=logits,
        raw_gold_ids=gold,
        raw_manifest=raw_manifest,
    )

    assert metrics["capacity_k"] == 3
    assert metrics["capacity_policy"] == "strict_ordered_prefix"
    assert metrics["source_order_cell"] == "selector_a__canonical"
    assert all(record["capacity_k"] == 3 for record in records)


def test_restore_gathered_logits_sorts_and_validates_distributed_duplicates() -> None:
    indices, logits, gold, duplicate_count = restore_gathered_logits(
        sample_indices=np.asarray([2, 0, -1, 1, 0]),
        label_logits=np.asarray(
            [[2.0, 0.0], [0.0, 1.0], [9.0, 9.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        gold_ids=np.asarray([1, 0, -1, 1, 0]),
        expected_count=3,
    )

    assert indices.tolist() == [0, 1, 2]
    assert logits.tolist() == [[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]
    assert gold.tolist() == [0, 1, 1]
    assert duplicate_count == 1

    with pytest.raises(MatrixValidationError, match="conflicting logits"):
        restore_gathered_logits(
            sample_indices=np.asarray([0, 0]),
            label_logits=np.asarray([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32),
            gold_ids=np.asarray([0, 0]),
            expected_count=1,
            atol=0.0,
            rtol=0.0,
        )

    with pytest.raises(MatrixValidationError, match="conflicting argmax"):
        restore_gathered_logits(
            sample_indices=np.asarray([0, 0]),
            label_logits=np.asarray([[1.0, 1.000001], [1.000001, 1.0]], dtype=np.float32),
            gold_ids=np.asarray([0, 0]),
            expected_count=1,
            atol=1e-5,
            rtol=1e-5,
        )

    with pytest.raises(MatrixValidationError, match="NaN or Inf"):
        restore_gathered_logits(
            sample_indices=np.asarray([0]),
            label_logits=np.asarray([[np.nan, 0.0]], dtype=np.float32),
            gold_ids=np.asarray([0]),
            expected_count=1,
        )


def test_cell_fanout_keeps_local_indices_and_uses_lowest_argmax_tie(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)
    prepare_matrix(
        matrix_manifest_path=manifest_path,
        build_root=build_root,
        output_dir=output_dir,
        split="val",
        label_prefix="Label:",
    )
    prepared = json.loads((output_dir / "input" / "manifest.json").read_text())
    unique_rows = _load_jsonl(output_dir / "input" / "unique_rows.jsonl")
    raw_logits = np.zeros((len(unique_rows), 6), dtype=np.float32)
    raw_gold = np.asarray([int(row["gold_id"]) for row in unique_rows], dtype=np.int64)
    raw_logits[:, 0] = 1.0
    raw_logits[:, 1] = 1.0  # numpy/torch argmax must choose label 0 on ties.
    raw_manifest = {
        "labels": ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
        "letter_order": ["A", "B", "C", "D", "E", "F"],
        "label_schema": "liar6",
        "label_token_meta": {"label_prefix": "Label:"},
        "checkpoint": {"checkpoint_name": "best"},
        "split": "val",
        "scoring_fingerprint": "fingerprint",
        "loss_replay_contract": {
            "native_eval_batch_size": 1,
            "ordinal_loss": {
                "enabled": True,
                "alpha": 0.2,
                "alpha_warmup_ratio": 0.3,
                "normalize_distance": True,
            },
        },
    }
    cell = prepared["cells"][1]
    mappings = _load_jsonl(output_dir / "input" / cell["mapping_file"])

    eval_metrics, metrics, records = build_cell_metrics_from_raw_logits(
        cell=cell,
        mappings=mappings,
        unique_rows=unique_rows,
        raw_logits=raw_logits,
        raw_gold_ids=raw_gold,
        raw_manifest=raw_manifest,
    )

    assert [record["sample_idx"] for record in records] == [0, 1]
    assert [record["pred_id"] for record in records] == [0, 0]
    assert all(record["raw_output"] == "Label: A" for record in records)
    assert metrics["num_samples"] == 2
    assert eval_metrics["confusion_matrix"].sum() == 2


def test_fanout_publishes_one_atomic_diagnostic_result_directory(tmp_path: Path) -> None:
    manifest_path, build_root, output_dir = _matrix_fixture(tmp_path)
    prepare_matrix(
        matrix_manifest_path=manifest_path,
        build_root=build_root,
        output_dir=output_dir,
        split="val",
        label_prefix="Label:",
    )
    input_manifest_path = output_dir / "input" / "manifest.json"
    unique_rows = _load_jsonl(output_dir / "input" / "unique_rows.jsonl")
    raw_dir = output_dir / "raw_logits"
    raw_dir.mkdir()
    logits = np.zeros((len(unique_rows), 6), dtype=np.float32)
    logits[:, 0] = 1.0
    gold = np.asarray([int(row["gold_id"]) for row in unique_rows], dtype=np.int64)
    indices = np.arange(len(unique_rows), dtype=np.int64)
    logits_path = raw_dir / "raw_label_logits.npz"
    with logits_path.open("wb") as handle:
        np.savez_compressed(handle, label_logits=logits, gold_ids=gold, unique_indices=indices)
    index_path = raw_dir / "raw_logits_index.jsonl"
    _write_jsonl(
        index_path,
        [
            {
                "unique_idx": idx,
                "prompt_cache_key": row["prompt_cache_key"],
                "gold_id": int(row["gold_id"]),
            }
            for idx, row in enumerate(unique_rows)
        ],
    )
    raw_manifest = {
        "schema_version": RAW_LOGITS_SCHEMA_VERSION,
        "status": "complete",
        "split": "val",
        "num_unique_prompts": len(unique_rows),
        "num_labels": 6,
        "label_schema": "liar6",
        "labels": ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
        "letter_order": ["A", "B", "C", "D", "E", "F"],
        "label_token_meta": {"label_prefix": "Label:"},
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "scoring_fingerprint": "scoring",
        "execution_fingerprint": "execution",
        "execution_contract": {"per_device_eval_batch_size": 1},
        "checkpoint": {"checkpoint_name": "best", "adapter_sha256": "adapter"},
        "raw_logits_file": logits_path.name,
        "raw_logits_sha256": _sha256_file(logits_path),
        "index_file": index_path.name,
        "index_sha256": _sha256_file(index_path),
        "loss_replay_contract": {
            "native_eval_batch_size": 1,
            "ordinal_loss": {"enabled": False},
        },
    }
    (raw_dir / "manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")

    result = fanout_matrix(
        output_dir=output_dir,
        force=False,
        equivalence_gate_cell=None,
        equivalence_gate_predictions=None,
        equivalence_gate_metrics=None,
        equivalence_gate_build=None,
        equivalence_gate_expected_adapter_sha256=None,
        equivalence_gate_reference_contract=None,
        unsafe_skip_equivalence_gate=True,
        classification_atol=1e-12,
        loss_atol=1e-6,
    )

    result_dir = output_dir / "materialized"
    assert result["status"] == "complete"
    assert result["diagnostic_only"] is True
    assert result["cell_count"] == 2
    assert (result_dir / "matrix_manifest.json").is_file()
    assert (result_dir / "factorial_metrics.csv").is_file()
    assert len(list((result_dir / "cells").glob("*/label_token/metrics.json"))) == 2
    assert not list(output_dir.glob(".materialized.tmp.*"))


@pytest.mark.parametrize(
    "ordinal_cfg",
    [
        {"enabled": False},
        {"enabled": True, "alpha": 0.2, "alpha_warmup_ratio": 0.0, "normalize_distance": True},
        {"enabled": True, "alpha": 0.2, "alpha_warmup_ratio": 0.3, "normalize_distance": False},
    ],
)
def test_native_batch1_loss_replay_matches_trainer(ordinal_cfg: dict) -> None:
    logits = np.asarray(
        [[2.0, 1.0, 0.0], [0.1, 0.4, 0.3], [-0.2, 0.5, 1.1]],
        dtype=np.float32,
    )
    gold = np.asarray([0, 1, 2], dtype=np.int64)
    train_cfg = {"label_token_ce": {"ordinal_loss": ordinal_cfg}}
    class_weights = torch.tensor([1.2, 0.7, 2.1], dtype=torch.float32)
    native_rows = [
        _compute_label_token_losses(
            label_logits=torch.from_numpy(logits[idx : idx + 1]),
            gold_ids=torch.tensor([int(gold[idx])]),
            class_weights=class_weights,
            train_cfg=train_cfg,
            global_step=0,
            max_train_steps=100,
        )
        for idx in range(len(gold))
    ]
    replay = _native_batch1_losses(logits, gold, ordinal_cfg=ordinal_cfg)
    expected = tuple(
        float(torch.stack([row[key] for row in native_rows]).mean().item())
        for key in ("loss", "ce_loss", "ordinal_loss")
    )

    assert replay == pytest.approx(expected, abs=5e-7)


def test_native_equivalence_gate_detects_one_prediction_flip(tmp_path: Path) -> None:
    build_rows = [_row("e0", [1, 2], gold_id=0), _row("e1", [3], gold_id=1)]
    reference_build = tmp_path / "build.jsonl"
    _write_jsonl(reference_build, build_rows)
    candidate_records = [
        {
            "sample_idx": idx,
            "event_id": row["event_id"],
            "prompt": row["prompt"],
            "target": row["target"],
            "prompt_input_ids_sha256": _prompt_ids_sha256(row["prompt_input_ids"]),
            "pred_id": idx,
            "pred_label": row["gold_label"],
            "raw_output": row["target"],
            "gold_id": row["gold_id"],
            "gold_label": row["gold_label"],
        }
        for idx, row in enumerate(build_rows)
    ]
    reference_predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(reference_predictions, [dict(record) for record in candidate_records])
    metrics = {
        "label_schema": "liar6",
        "accuracy": 1.0,
        "macro_precision": 1.0 / 3.0,
        "macro_recall": 1.0 / 3.0,
        "macro_f1": 1.0 / 3.0,
        "parse_error_rate": 0.0,
        "true_side_macro_f1": 0.0,
        "checkpoint_selection_score": 1.0 / 3.0,
        "eval_loss": 0.5,
        "eval_ce_loss": 0.5,
        "eval_ordinal_loss": 0.2,
        "per_class": {},
    }
    reference_metrics = tmp_path / "metrics.json"
    reference_metrics.write_text(json.dumps(metrics), encoding="utf-8")

    passing = compare_native_equivalence(
        candidate_records=candidate_records,
        candidate_metrics=metrics,
        reference_predictions_path=reference_predictions,
        reference_metrics_path=reference_metrics,
        reference_build_path=reference_build,
        expected_count=2,
        adapter_sha256="abc",
        expected_adapter_sha256="abc",
    )
    assert passing["passed"] is True
    assert passing["prediction_exact_match_count"] == 2

    wrong_expected_count = compare_native_equivalence(
        candidate_records=candidate_records,
        candidate_metrics=metrics,
        reference_predictions_path=reference_predictions,
        reference_metrics_path=reference_metrics,
        reference_build_path=reference_build,
        expected_count=3,
        adapter_sha256="abc",
        expected_adapter_sha256="abc",
    )
    assert wrong_expected_count["passed"] is False

    flipped = [dict(record) for record in candidate_records]
    flipped[1].update({"pred_id": 0, "pred_label": "pants-fire", "raw_output": "Label: A"})
    failing = compare_native_equivalence(
        candidate_records=flipped,
        candidate_metrics=metrics,
        reference_predictions_path=reference_predictions,
        reference_metrics_path=reference_metrics,
        reference_build_path=reference_build,
        expected_count=2,
        adapter_sha256="abc",
        expected_adapter_sha256="abc",
    )
    assert failing["passed"] is False
    assert failing["prediction_exact_match_count"] == 1


def test_model_weight_content_changes_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"first-weight-content")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    resolved_config = tmp_path / "config.resolved.yaml"
    resolved_config.write_text("sft_train: {}\n", encoding="utf-8")
    context = SimpleNamespace(
        checkpoint_dir=checkpoint,
        checkpoint_name="best",
        is_peft_adapter=False,
        model_name_or_path=str(checkpoint),
        run_dir=tmp_path,
    )

    first = _checkpoint_provenance(context, config_path=str(resolved_config))
    (checkpoint / "model.safetensors").write_bytes(b"second-weight-content")
    second = _checkpoint_provenance(context, config_path=str(resolved_config))

    assert first["model_weight_files"]["model.safetensors"]["sha256"] != second[
        "model_weight_files"
    ]["model.safetensors"]["sha256"]
    assert first["model_identity_sha256"] != second["model_identity_sha256"]


def test_matrix_wrapper_reads_verifier_defaults_from_reference_contract(tmp_path: Path) -> None:
    contract_path = tmp_path / "reference.json"
    contract_path.write_text(
        json.dumps(
            {
                "native_command": [sys.executable],
                "checkpoint": {
                    "run_dir": "contract/verifier/train",
                    "checkpoint": "contract-best",
                    "adapter_sha256": "a" * 64,
                },
                "artifacts": {
                    "inference_config": {"path": "contract/cell/train.resolved.yaml"},
                    "predictions": {"path": "contract/native/val_predictions.jsonl"},
                    "metrics": {"path": "contract/native/metrics.json"},
                    "build": {"path": "contract/cell/build_val.jsonl"},
                },
            }
        ),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "PHASES": "infer",
        "EVAL_NPROC_PER_NODE": "1",
        "PYTHON_BIN": sys.executable,
        "REFERENCE_CONTRACT": str(contract_path),
        "OUTPUT_DIR": str(tmp_path / "output"),
    }

    completed = subprocess.run(
        [
            "bash",
            str(root / "scripts/phase5_selectors/eval/run_baces_deduplicated_raw_logits_matrix.sh"),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "verifier_run=contract/verifier/train checkpoint=contract-best" in completed.stdout
    assert f"expected_adapter_sha256={'a' * 64}" in completed.stdout
    assert "--config contract/cell/train.resolved.yaml" in completed.stdout
    assert "--expected-adapter-sha256 " + "a" * 64 in completed.stdout
