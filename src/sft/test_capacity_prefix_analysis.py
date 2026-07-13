from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sft import capacity_prefix_analysis as capacity
from scripts.phase5_selectors.build.materialize_capacity_policy_from_traces import (
    materialize_capacity_policy,
)


LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]


def _synthetic_prefix_matrix(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "prefix_run"
    input_dir = root / "input"
    raw_dir = root / "raw_logits"
    matrix_dir = root / "materialized"
    input_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    event_ids = [f"event-{index}" for index in range(len(LABELS))]
    gold_ids = list(range(len(LABELS)))
    oracle_ks = [1, 2, 3, 1, 2, 3]
    selector = "baces_exact"
    scoring_fingerprint = "synthetic-prefix-scoring"

    raw_logits: list[np.ndarray] = []
    raw_gold: list[int] = []
    raw_unique_rows: list[dict[str, object]] = []
    unique_by_event_k: dict[tuple[int, int], int] = {}
    prompt_by_event_k: dict[tuple[int, int], tuple[str, str, list[int]]] = {}
    input_cells: list[dict[str, object]] = []
    matrix_cells: list[dict[str, object]] = []
    source_matrix_cells: list[dict[str, object]] = []
    label_prefix = "Label:"
    for requested_k in (1, 2, 3):
        cell_id = f"{selector}__prefix_k{requested_k:02d}"
        controller = f"prefix_k{requested_k:02d}"
        mappings = []
        predictions = []
        source_build_rows = []
        for sample_idx, (event_id, gold_id, oracle_k) in enumerate(
            zip(event_ids, gold_ids, oracle_ks)
        ):
            # Event 0 exhausts at K=2, so requested K=3 aliases the same prompt.
            realized_k = 2 if sample_idx == 0 and requested_k == 3 else requested_k
            source_k = 2 if sample_idx == 0 and requested_k == 3 else requested_k
            token_count = 100 + 20 * realized_k
            key = (sample_idx, source_k)
            if key not in unique_by_event_k:
                values = np.zeros(len(LABELS), dtype=np.float64)
                values[gold_id] = 4.0 - 3.0 * abs(source_k - oracle_k)
                new_unique_idx = len(raw_logits)
                unique_by_event_k[key] = new_unique_idx
                prompt_input_ids = [1000 + new_unique_idx] * token_count
                prompt_hash = capacity._sha256_json(prompt_input_ids)
                prompt_cache_key = capacity._prompt_cache_key(
                    prompt_input_ids, label_prefix
                )
                prompt_by_event_k[key] = (
                    prompt_hash,
                    prompt_cache_key,
                    prompt_input_ids,
                )
                raw_logits.append(values)
                raw_gold.append(gold_id)
                raw_unique_rows.append(
                    {
                        "unique_idx": new_unique_idx,
                        "event_id": event_id,
                        "gold_id": gold_id,
                        "gold_label": LABELS[gold_id],
                        "prompt_cache_key": prompt_cache_key,
                        "prompt_input_ids_sha256": prompt_hash,
                        "evidence_count": realized_k,
                        "prompt_token_count": token_count,
                    }
                )
            unique_idx = unique_by_event_k[key]
            prompt_hash, prompt_cache_key, prompt_input_ids = prompt_by_event_k[key]
            common = {
                "event_id": event_id,
                "gold_id": gold_id,
                "gold_label": LABELS[gold_id],
                "evidence_count": realized_k,
                "prompt_token_count": token_count,
                "prompt_input_ids_sha256": prompt_hash,
            }
            mappings.append(
                {
                    **common,
                    "cell_id": cell_id,
                    "cell_sample_idx": sample_idx,
                    "unique_idx": unique_idx,
                    "prompt_cache_key": prompt_cache_key,
                }
            )
            predictions.append(
                {
                    **common,
                    "cell_id": cell_id,
                    "sample_idx": sample_idx,
                    "selector_level": selector,
                    "controller_level": controller,
                    "raw_logits_unique_idx": unique_idx,
                    "prompt_cache_key": prompt_cache_key,
                    "pred_id": gold_id,
                    "pred_label": LABELS[gold_id],
                    "scoring_fingerprint": scoring_fingerprint,
                }
            )
            source_build_rows.append(
                {
                    "event_id": event_id,
                    "gold_id": gold_id,
                    "evidence_count": realized_k,
                    "prompt_token_count": token_count,
                    "prompt_input_ids": prompt_input_ids,
                }
            )
        mapping_path = input_dir / "cells" / f"{cell_id}.jsonl"
        prediction_path = matrix_dir / "cells" / cell_id / "val_predictions.jsonl"
        capacity._write_jsonl(mapping_path, mappings)
        capacity._write_jsonl(prediction_path, predictions)
        source_build_path = root / "formal" / cell_id / "build" / "build_val.jsonl"
        capacity._write_jsonl(source_build_path, source_build_rows)
        input_cells.append(
            {
                "cell_id": cell_id,
                "selector_level": selector,
                "controller_level": controller,
                "capacity_k": requested_k,
                "row_count": len(event_ids),
                "mapping_file": str(mapping_path.relative_to(input_dir)),
                "mapping_sha256": capacity._sha256_file(mapping_path),
                "source_build": str(source_build_path),
                "source_build_sha256": capacity._sha256_file(source_build_path),
            }
        )
        matrix_cells.append(
            {
                "cell_id": cell_id,
                "selector_level": selector,
                "controller_level": controller,
                "requested_prefix_k": requested_k,
                "predictions_file": str(prediction_path.relative_to(matrix_dir)),
                "predictions_sha256": capacity._sha256_file(prediction_path),
            }
        )
        source_matrix_cells.append(
            {
                "cell_id": cell_id,
                "selector_level": selector,
                "controller_level": controller,
                "capacity_k": requested_k,
                "build_file": str(source_build_path),
                "build_sha256": capacity._sha256_file(source_build_path),
            }
        )

    source_matrix_dir = root / "source_matrix"
    prefix_gate_path = source_matrix_dir / "prefix_integrity_gate.json"
    capacity._write_json(prefix_gate_path, {"passed": True})
    source_matrix_manifest_path = source_matrix_dir / "manifest.json"
    capacity._write_json(
        source_matrix_manifest_path,
        {
            "schema_version": "baces_capacity_prefix_matrix_v0_1",
            "status": "complete",
            "prefix_integrity_gate": prefix_gate_path.name,
            "prefix_integrity_gate_sha256": capacity._sha256_file(prefix_gate_path),
            "cells": source_matrix_cells,
        },
    )

    unique_rows_path = input_dir / "unique_rows.jsonl"
    capacity._write_jsonl(unique_rows_path, raw_unique_rows)
    input_manifest = {
        "schema_version": "synthetic-prefix-input-v0",
        "status": "complete",
        "split": "val",
        "label_prefix": label_prefix,
        "matrix_manifest": str(source_matrix_manifest_path),
        "matrix_manifest_sha256": capacity._sha256_file(source_matrix_manifest_path),
        "matrix_schema_version": "baces_capacity_prefix_matrix_v0_1",
        "unique_prompt_count": len(raw_unique_rows),
        "unique_rows_file": unique_rows_path.name,
        "unique_rows_sha256": capacity._sha256_file(unique_rows_path),
        "cells": input_cells,
    }
    input_manifest_path = input_dir / "manifest.json"
    capacity._write_json(input_manifest_path, input_manifest)

    logits_path = raw_dir / "raw_label_logits.npz"
    with logits_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            label_logits=np.asarray(raw_logits),
            gold_ids=np.asarray(raw_gold, dtype=np.int64),
            unique_indices=np.arange(len(raw_logits), dtype=np.int64),
        )
    index_path = raw_dir / "raw_logits_index.jsonl"
    capacity._write_jsonl(index_path, raw_unique_rows)
    raw_manifest = {
        "schema_version": "synthetic-prefix-logits-v0",
        "status": "complete",
        "split": "val",
        "num_labels": len(LABELS),
        "num_unique_prompts": len(raw_unique_rows),
        "labels": LABELS,
        "input_manifest_sha256": capacity._sha256_file(input_manifest_path),
        "scoring_fingerprint": scoring_fingerprint,
        "execution_fingerprint": "synthetic-execution",
        "raw_logits_file": logits_path.name,
        "raw_logits_sha256": capacity._sha256_file(logits_path),
        "index_file": index_path.name,
        "index_sha256": capacity._sha256_file(index_path),
    }
    raw_manifest_path = raw_dir / "manifest.json"
    capacity._write_json(raw_manifest_path, raw_manifest)

    matrix_manifest = {
        "schema_version": "synthetic-prefix-matrix-v0",
        "status": "complete",
        "split": "val",
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": capacity._sha256_file(input_manifest_path),
        "raw_logits_manifest": str(raw_manifest_path),
        "raw_logits_manifest_sha256": capacity._sha256_file(raw_manifest_path),
        "raw_logits_scoring_fingerprint": scoring_fingerprint,
        "raw_logits_sha256": capacity._sha256_file(logits_path),
        "cells": matrix_cells,
    }
    matrix_manifest_path = matrix_dir / "matrix_manifest.json"
    capacity._write_json(matrix_manifest_path, matrix_manifest)

    policy_rows = [
        {
            "policy_id": "fixed2",
            "event_id": event_id,
            "selector_level": selector,
            "selected_k": 2,
        }
        for event_id in event_ids
    ]
    policy_path = root / "fixed2.jsonl"
    capacity._write_jsonl(policy_path, policy_rows)
    return matrix_manifest_path, policy_path, root / "capacity_analysis"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_balanced_class_weights_normalize_each_class_equally() -> None:
    gold = np.asarray([0, 0, 0, 1])

    weights = capacity.balanced_class_weights(gold, n_labels=2)

    assert weights.tolist() == pytest.approx([2.0 / 3.0, 2.0])
    assert float(np.mean(weights[gold])) == pytest.approx(1.0)


def test_materializer_builds_oracle_regret_and_deduplicates_plateau(tmp_path: Path) -> None:
    matrix_manifest, policy_path, output_dir = _synthetic_prefix_matrix(tmp_path)

    manifest = capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=output_dir,
        policy_paths={"fixed2": policy_path},
    )

    assert manifest["status"] == "complete"
    assert manifest["event_count"] == 6
    assert manifest["selector_count"] == 1
    assert manifest["oracle_row_count"] == 6
    assert manifest["capacity_curve_row_count"] == 6
    assert manifest["policy_regret_row_count"] == 6
    assert not list(output_dir.parent.glob(f".{output_dir.name}.tmp.*"))
    for artifact in manifest["artifacts"].values():
        path = output_dir / artifact["path"]
        assert capacity._sha256_file(path) == artifact["sha256"]

    oracle_rows = _read_jsonl(output_dir / "oracle_events.jsonl")
    assert [row["oracle_requested_k"] for row in oracle_rows] == [1, 2, 3, 1, 2, 3]
    first = oracle_rows[0]
    assert first["capacity_feasible_max_k"] == 2
    assert first["requested_action_count"] == 3
    assert first["unique_prompt_action_count"] == 2

    prefix_rows = _read_jsonl(output_dir / "prefix_scores.jsonl")
    assert len(prefix_rows) == 18
    first_k2 = next(
        row
        for row in prefix_rows
        if row["event_id"] == "event-0" and row["requested_k"] == 2
    )
    assert first_k2["requested_k_aliases"] == [2, 3]
    first_k3 = next(
        row
        for row in prefix_rows
        if row["event_id"] == "event-0" and row["requested_k"] == 3
    )
    assert first_k3["canonical_requested_k"] == 2
    assert first_k3["is_duplicate_prompt_alias"] is True

    curve_rows = _read_jsonl(output_dir / "capacity_curves.jsonl")
    full_k3 = next(
        row
        for row in curve_rows
        if row["support"] == "full_n_deployable" and row["requested_k"] == 3
    )
    strict_k3 = next(
        row
        for row in curve_rows
        if row["support"] == "strict_full_grid_common_support"
        and row["requested_k"] == 3
    )
    assert full_k3["sample_count"] == 6
    assert full_k3["exact_support_count"] == 5
    assert full_k3["mean_realized_k"] == pytest.approx(17.0 / 6.0)
    assert strict_k3["sample_count"] == 5
    assert strict_k3["exact_support_rate"] == 1.0

    regret_rows = _read_jsonl(output_dir / "policy_regret.jsonl")
    relations = [row["capacity_relation"] for row in regret_rows]
    assert relations.count("underfill") == 2
    assert relations.count("capacity_correct") == 2
    assert relations.count("overfill") == 2
    assert all(float(row["objective_regret"]) >= 0.0 for row in regret_rows)

    aggregate = json.loads((output_dir / "aggregate.json").read_text(encoding="utf-8"))
    selector = aggregate["selectors"][0]
    policy = aggregate["policies"][0]
    assert selector["full_grid_exact_support_count"] == 5
    assert selector["full_grid_exact_support_rate"] == pytest.approx(5.0 / 6.0)
    assert policy["underfill_rate"] == pytest.approx(1.0 / 3.0)
    assert policy["capacity_correct_rate"] == pytest.approx(1.0 / 3.0)
    assert policy["overfill_rate"] == pytest.approx(1.0 / 3.0)
    assert policy["verification_status"] == "unverified_external_policy"
    assert policy["deployable_without_gold"] is False
    assert manifest["policy_sources"][0]["uses_gold"] is None
    assert len(aggregate["capacity_optima"]) == 2
    assert {row["support"] for row in aggregate["capacity_optima"]} == {
        "full_n_deployable",
        "strict_full_grid_common_support",
    }

    reused = capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=output_dir,
        policy_paths={"fixed2": policy_path},
    )
    assert reused["created_at"] == manifest["created_at"]


def test_materializer_can_generate_every_fixed_capacity_policy(tmp_path: Path) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    fixed_output = output_dir.with_name("capacity_analysis_fixed")

    manifest = capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=fixed_output,
        include_fixed_policies=True,
    )

    assert manifest["policy_regret_row_count"] == 18
    assert [source["policy_id"] for source in manifest["policy_sources"]] == [
        "fixed_k01",
        "fixed_k02",
        "fixed_k03",
    ]
    assert all(
        source["verification_status"] == "verified_generated_fixed_policy"
        and source["deployable_without_gold"] is True
        and source["uses_gold"] is False
        for source in manifest["policy_sources"]
    )
    aggregate = json.loads((fixed_output / "aggregate.json").read_text(encoding="utf-8"))
    assert {row["policy_id"] for row in aggregate["policies"]} == {
        "fixed_k01",
        "fixed_k02",
        "fixed_k03",
    }


def test_materializer_rejects_policy_k_outside_prefix_grid(tmp_path: Path) -> None:
    matrix_manifest, policy_path, output_dir = _synthetic_prefix_matrix(tmp_path)
    rows = _read_jsonl(policy_path)
    rows[0]["selected_k"] = 4
    capacity._write_jsonl(policy_path, rows)

    with pytest.raises(capacity.CapacityAnalysisError, match="unsupported K"):
        capacity.materialize_capacity_analysis(
            matrix_manifest_path=matrix_manifest,
            output_dir=output_dir,
            policy_paths={"fixed2": policy_path},
        )


def test_strict_capacity_curve_uses_one_cross_selector_common_support() -> None:
    event_ids = ["event-0", "event-1"]
    observations: dict[str, dict[int, list[capacity.PrefixObservation]]] = {}
    unique_idx = 0
    for selector in ("selector-a", "selector-b"):
        observations[selector] = {}
        for requested_k in (1, 2):
            rows = []
            for sample_idx, event_id in enumerate(event_ids):
                realized_k = requested_k
                if selector == "selector-b" and requested_k == 2 and sample_idx == 1:
                    realized_k = 1
                rows.append(
                    capacity.PrefixObservation(
                        selector_level=selector,
                        event_id=event_id,
                        sample_idx=sample_idx,
                        requested_k=requested_k,
                        realized_k=realized_k,
                        prompt_token_count=100 + 10 * realized_k,
                        prompt_hash=f"prompt-{selector}-{sample_idx}-{realized_k}",
                        unique_idx=unique_idx,
                        gold_id=sample_idx,
                        logits=np.asarray([2.0, 0.0])
                        if sample_idx == 0
                        else np.asarray([0.0, 2.0]),
                    )
                )
                unique_idx += 1
            observations[selector][requested_k] = rows

    curve_rows = capacity._capacity_curve_rows(
        observations=observations,
        labels=["false", "true"],
        event_ids=event_ids,
    )
    strict_rows = [
        row
        for row in curve_rows
        if row["support"] == "strict_full_grid_common_support"
    ]

    assert len(strict_rows) == 4
    assert {row["sample_count"] for row in strict_rows} == {1}
    assert len({row["support_event_id_sequence_sha256"] for row in strict_rows}) == 1
    assert strict_rows[0]["support_event_id_sequence_sha256"] == capacity._event_sequence_sha256(
        ["event-0"]
    )


def test_token_penalty_is_used_by_global_capacity_objective(tmp_path: Path) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    penalized_output = output_dir.with_name("capacity_analysis_penalized")

    capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=penalized_output,
        token_penalty_per_1k=100.0,
    )

    curves = _read_jsonl(penalized_output / "capacity_curves.jsonl")
    full_rows = [row for row in curves if row["support"] == "full_n_deployable"]
    assert all(
        row["objective_mean"]
        == pytest.approx(row["class_balanced_nll_mean"] + row["token_penalty_mean"])
        for row in full_rows
    )
    aggregate = json.loads(
        (penalized_output / "aggregate.json").read_text(encoding="utf-8")
    )
    full_optimum = next(
        row
        for row in aggregate["capacity_optima"]
        if row["support"] == "full_n_deployable"
    )
    assert full_optimum["objective_optimal_k"] == 1
    assert full_optimum["objective_optimal_value"] == pytest.approx(
        min(row["objective_mean"] for row in full_rows)
    )


def test_trace_policy_sidecar_is_verified_and_propagated(tmp_path: Path) -> None:
    matrix_manifest, policy_path, output_dir = _synthetic_prefix_matrix(tmp_path)
    trace_path = policy_path.with_name("source_trace.jsonl")
    pool = [{"candidate_uid": f"u{index}"} for index in range(3)]
    order = [0, 1]
    trace_rows = []
    for index in range(len(LABELS)):
        trace_rows.append(
            {
                "event_id": f"event-{index}",
                "factor_selector": "baces_exact",
                "factor_controller": "ordinal_replay_minmax5_10",
                "factorial_metadata": {
                    "selector_level": "baces_exact",
                    "controller_level": "ordinal_replay_minmax5_10",
                    "controller_contract": (
                        "first prefix t>=5 reaching the common exact Kmax=10 ordinal "
                        "target, else 10"
                    ),
                    "k_min": 5,
                    "k_max": 10,
                    "stored_target_resolved_used": False,
                    "common_exact_kmax10_target_state": [99],
                    "controller_stop_reason": "pool_exhausted",
                },
                "candidate_pool": pool,
                "selector_available_ordered_indices": order,
                "selector_available_ordered_candidate_uids": [
                    "u0",
                    "u1",
                ],
                "selected_indices": [0, 1],
                "selected_candidate_uids": ["u0", "u1"],
                "selected_count": 2,
                "baces_display_steps": [
                    {
                        "position": 1,
                        "candidate_idx": 0,
                        "candidate_uid": "u0",
                        "state_after": [1],
                    },
                    {
                        "position": 2,
                        "candidate_idx": 1,
                        "candidate_uid": "u1",
                        "state_after": [2],
                    },
                ],
                "baces_display": {"length": 2, "terminal_state": [2]},
            }
        )
    capacity._write_jsonl(trace_path, trace_rows)
    factorial_manifest = trace_path.with_name("factorial_manifest.json")
    capacity._write_json(
        factorial_manifest,
        {
            "schema_version": "baces_factorial_trace_v0_1",
            "factorial_version": "test",
            "split": "val",
            "event_count": len(LABELS),
            "controller_contracts": {
                "ordinal_replay_minmax5_10": (
                    "first prefix t>=5 reaching the common exact Kmax=10 ordinal "
                    "target, else 10"
                )
            },
            "source_contract": {"coverage_and_pool": "features"},
            "cells": [
                {
                    "cell_id": "baces_exact__ordinal_replay_minmax5_10",
                    "selector_level": "baces_exact",
                    "controller_level": "ordinal_replay_minmax5_10",
                    "trace_file": trace_path.name,
                    "row_count": len(LABELS),
                    "ready": True,
                }
            ],
        },
    )
    materialize_capacity_policy(
        selector_traces={"baces_exact": trace_path},
        policy_id="fixed2",
        output_policy=policy_path,
        expected_controller="ordinal_replay_minmax5_10",
        source_factorial_manifest=factorial_manifest,
        split="val",
        min_k=1,
        max_k=3,
        overwrite=True,
    )

    manifest = capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=output_dir,
        policy_paths={"fixed2": policy_path},
    )

    source = manifest["policy_sources"][0]
    assert source["verification_status"] == "verified_trace_policy_sidecar"
    assert source["declared_verification_status"] == (
        "verified_known_structure_only_factorial_controller"
    )
    assert source["uses_gold"] is False
    assert source["uses_verifier_logits"] is False
    assert source["deployable_without_gold"] is True
    assert source["deployable_ex_ante"] is True
    assert source["source_traces"][0]["path"] == str(trace_path.resolve())
    aggregate = json.loads((output_dir / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["policies"][0]["verification_status"] == (
        "verified_trace_policy_sidecar"
    )
    assert aggregate["policies"][0]["deployable_without_gold"] is True
    assert aggregate["policies"][0]["deployable_ex_ante"] is True


def test_trace_policy_sidecar_with_unknown_controller_provenance_is_not_deployable(
    tmp_path: Path,
) -> None:
    matrix_manifest, policy_path, output_dir = _synthetic_prefix_matrix(tmp_path)
    trace_path = policy_path.with_name("unverified_source_trace.jsonl")
    trace_rows = []
    for index in range(len(LABELS)):
        trace_rows.append(
            {
                "event_id": f"event-{index}",
                "candidate_pool": [
                    {"candidate_uid": "u0"},
                    {"candidate_uid": "u1"},
                    {"candidate_uid": "u2"},
                ],
                "selector_available_ordered_indices": [0, 1, 2],
                "selector_available_ordered_candidate_uids": ["u0", "u1", "u2"],
                "selected_indices": [0, 1],
                "selected_candidate_uids": ["u0", "u1"],
                "selected_count": 2,
            }
        )
    capacity._write_jsonl(trace_path, trace_rows)
    materialize_capacity_policy(
        selector_traces={"baces_exact": trace_path},
        policy_id="fixed2",
        output_policy=policy_path,
        min_k=1,
        max_k=3,
        overwrite=True,
    )

    manifest = capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=output_dir,
        policy_paths={"fixed2": policy_path},
    )

    source = manifest["policy_sources"][0]
    assert source["verification_status"] == (
        "trace_policy_sidecar_integrity_verified_provenance_unknown"
    )
    assert source["uses_gold"] is None
    assert source["uses_verifier_logits"] is None
    assert source["deployable_without_gold"] is False
    assert source["deployable_ex_ante"] is False


def test_rejects_rehashed_raw_index_row_permutation(tmp_path: Path) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    matrix = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    raw_manifest_path = Path(matrix["raw_logits_manifest"])
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    index_path = raw_manifest_path.parent / raw_manifest["index_file"]
    rows = _read_jsonl(index_path)
    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["unique_idx"] = 0
    rows[1]["unique_idx"] = 1
    capacity._write_jsonl(index_path, rows)
    raw_manifest["index_sha256"] = capacity._sha256_file(index_path)
    capacity._write_json(raw_manifest_path, raw_manifest)
    matrix["raw_logits_manifest_sha256"] = capacity._sha256_file(raw_manifest_path)
    capacity._write_json(matrix_manifest, matrix)

    with pytest.raises(
        capacity.CapacityAnalysisError,
        match="Raw index/prepared unique-row event_id mismatch",
    ):
        capacity.materialize_capacity_analysis(
            matrix_manifest_path=matrix_manifest,
            output_dir=output_dir,
        )


def test_rejects_prediction_factor_mismatch_even_when_rehashed(tmp_path: Path) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    matrix = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    cell = matrix["cells"][0]
    prediction_path = matrix_manifest.parent / cell["predictions_file"]
    rows = _read_jsonl(prediction_path)
    rows[0]["selector_level"] = "wrong_selector"
    capacity._write_jsonl(prediction_path, rows)
    cell["predictions_sha256"] = capacity._sha256_file(prediction_path)
    capacity._write_json(matrix_manifest, matrix)

    with pytest.raises(capacity.CapacityAnalysisError, match="identity mismatch"):
        capacity.materialize_capacity_analysis(
            matrix_manifest_path=matrix_manifest,
            output_dir=output_dir,
        )


def test_rejects_same_prompt_identity_with_different_resources(tmp_path: Path) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    matrix = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    cell = next(
        cell for cell in matrix["cells"] if cell["controller_level"] == "prefix_k03"
    )
    prediction_path = matrix_manifest.parent / cell["predictions_file"]
    predictions = _read_jsonl(prediction_path)
    predictions[0]["prompt_token_count"] += 1
    capacity._write_jsonl(prediction_path, predictions)
    cell["predictions_sha256"] = capacity._sha256_file(prediction_path)

    input_manifest_path = Path(matrix["input_manifest"])
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    input_cell = next(
        cell for cell in input_manifest["cells"] if cell["controller_level"] == "prefix_k03"
    )
    mapping_path = input_manifest_path.parent / input_cell["mapping_file"]
    mappings = _read_jsonl(mapping_path)
    mappings[0]["prompt_token_count"] += 1
    capacity._write_jsonl(mapping_path, mappings)
    input_cell["mapping_sha256"] = capacity._sha256_file(mapping_path)
    capacity._write_json(input_manifest_path, input_manifest)
    matrix["input_manifest_sha256"] = capacity._sha256_file(input_manifest_path)

    raw_manifest_path = Path(matrix["raw_logits_manifest"])
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_manifest["input_manifest_sha256"] = matrix["input_manifest_sha256"]
    capacity._write_json(raw_manifest_path, raw_manifest)
    matrix["raw_logits_manifest_sha256"] = capacity._sha256_file(raw_manifest_path)
    capacity._write_json(matrix_manifest, matrix)

    with pytest.raises(
        capacity.CapacityAnalysisError,
        match="Mapping/unique-row identity mismatch",
    ):
        capacity.materialize_capacity_analysis(
            matrix_manifest_path=matrix_manifest,
            output_dir=output_dir,
        )


def test_existing_analysis_binds_implementation_sha(tmp_path: Path) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    capacity.materialize_capacity_analysis(
        matrix_manifest_path=matrix_manifest,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation"]["sha256"] = "0" * 64
    capacity._write_json(manifest_path, manifest)

    with pytest.raises(capacity.CapacityAnalysisError, match="incompatible"):
        capacity.materialize_capacity_analysis(
            matrix_manifest_path=matrix_manifest,
            output_dir=output_dir,
        )


def test_rejects_coordinated_mapping_repoint_against_frozen_source_build(
    tmp_path: Path,
) -> None:
    matrix_manifest, _, output_dir = _synthetic_prefix_matrix(tmp_path)
    matrix = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    input_manifest_path = Path(matrix["input_manifest"])
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    unique_rows = _read_jsonl(input_manifest_path.parent / input_manifest["unique_rows_file"])
    k1_cell = next(
        cell for cell in input_manifest["cells"] if cell["controller_level"] == "prefix_k01"
    )
    k2_cell = next(
        cell for cell in input_manifest["cells"] if cell["controller_level"] == "prefix_k02"
    )
    k1_mapping = _read_jsonl(input_manifest_path.parent / k1_cell["mapping_file"])[0]
    replacement = unique_rows[k1_mapping["unique_idx"]]
    k2_mapping_path = input_manifest_path.parent / k2_cell["mapping_file"]
    k2_mappings = _read_jsonl(k2_mapping_path)
    for field in (
        "unique_idx",
        "prompt_cache_key",
        "prompt_input_ids_sha256",
        "evidence_count",
        "prompt_token_count",
    ):
        k2_mappings[0][field] = replacement[field]
    capacity._write_jsonl(k2_mapping_path, k2_mappings)
    k2_cell["mapping_sha256"] = capacity._sha256_file(k2_mapping_path)
    capacity._write_json(input_manifest_path, input_manifest)
    matrix["input_manifest_sha256"] = capacity._sha256_file(input_manifest_path)

    matrix_k2 = next(
        cell for cell in matrix["cells"] if cell["controller_level"] == "prefix_k02"
    )
    prediction_path = matrix_manifest.parent / matrix_k2["predictions_file"]
    predictions = _read_jsonl(prediction_path)
    predictions[0]["raw_logits_unique_idx"] = replacement["unique_idx"]
    for field in (
        "prompt_cache_key",
        "prompt_input_ids_sha256",
        "evidence_count",
        "prompt_token_count",
    ):
        predictions[0][field] = replacement[field]
    capacity._write_jsonl(prediction_path, predictions)
    matrix_k2["predictions_sha256"] = capacity._sha256_file(prediction_path)

    raw_manifest_path = Path(matrix["raw_logits_manifest"])
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_manifest["input_manifest_sha256"] = matrix["input_manifest_sha256"]
    capacity._write_json(raw_manifest_path, raw_manifest)
    matrix["raw_logits_manifest_sha256"] = capacity._sha256_file(raw_manifest_path)
    capacity._write_json(matrix_manifest, matrix)

    with pytest.raises(
        capacity.CapacityAnalysisError,
        match="Source-build/mapping identity mismatch",
    ):
        capacity.materialize_capacity_analysis(
            matrix_manifest_path=matrix_manifest,
            output_dir=output_dir,
        )
