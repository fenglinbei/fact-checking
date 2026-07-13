from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from sft import capacity_prefix_analysis as capacity
from sft import capacity_prefix_paired_inference as inference
from sft import paired_factorial_inference as paired


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPO_ROOT / "configs/validation/baces_capacity_contrast_registry_v0_1.json"
)
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]


def _action(
    action_id: str,
    *,
    selector: str = "baces_exact",
    pred_ids: np.ndarray | None = None,
    raw_nll: np.ndarray | None = None,
    size: int = 12,
) -> inference.ActionData:
    if pred_ids is None:
        pred_ids = np.arange(size, dtype=np.int64) % len(LABELS)
    if raw_nll is None:
        raw_nll = np.linspace(0.5, 1.5, size)
    return inference.ActionData(
        action_id=action_id,
        kind="fixed_capacity",
        selector_level=selector,
        requested_k=np.full(size, 5, dtype=np.int64),
        realized_k=np.full(size, 5.0),
        prompt_token_count=np.full(size, 500.0),
        prompt_hashes=tuple(f"prompt-{index}" for index in range(size)),
        pred_ids=np.asarray(pred_ids, dtype=np.int64),
        raw_nll=np.asarray(raw_nll, dtype=np.float64),
        source={"kind": "synthetic"},
    )


def test_frozen_registry_separates_current_val_from_future_test() -> None:
    registry, specs = inference._load_registry(REGISTRY_PATH)

    assert registry["registration_timing"]["current_validation"] == (
        "post_hoc_rule_based_after_result_inspection"
    )
    assert len({spec.comparison_id for spec in specs}) == 9
    assert len(specs) == 14
    primary = [
        spec
        for spec in specs
        if spec.family_id == "prospective_capacity_primary"
        and spec.support_id == inference.FULL_SUPPORT
    ]
    assert {spec.comparison_id for spec in primary} == {
        "baces_exact__k05_minus_k01",
        "baces_exact__k05_minus_k10",
    }
    assert all(spec.current_validation_role == "retrospective_diagnostic" for spec in primary)
    assert all(
        spec.future_held_out_role
        == "eligible_prospective_fixed_verifier_confirmatory_with_external_contract"
        for spec in primary
    )
    assert all(not spec.selection_uses_observed_validation_outcome for spec in primary)
    assert not any("k07" in spec.comparison_id or "k08" in spec.comparison_id for spec in specs)


def test_claim_gate_never_promotes_from_split_name() -> None:
    _, specs = inference._load_registry(REGISTRY_PATH)
    spec = next(
        value
        for value in specs
        if value.comparison_id == "baces_exact__k05_minus_k01"
        and value.support_id == inference.FULL_SUPPORT
    )

    validation = inference._claim_gate(spec=spec, evaluation_split="val")
    future_named_test = inference._claim_gate(spec=spec, evaluation_split="test")

    assert validation == (
        "retrospective_diagnostic",
        False,
        "retrospective_inspected_validation",
    )
    assert future_named_test == (
        "diagnostic_no_external_preinference_contract",
        False,
        "automatic_promotion_disabled_external_contract_required",
    )


def test_class_balanced_bootstrap_missing_class_fails_closed() -> None:
    with pytest.raises(
        inference.CapacityPairedInferenceError,
        match="changed its registered class support",
    ):
        inference._require_complete_class_bootstrap(
            support_id=inference.FULL_SUPPORT,
            ordinary_missing=1,
            stratified_missing=0,
        )


def test_class_balanced_bootstrap_recomputes_within_replicate_class_means() -> None:
    gold = np.asarray([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    values = np.asarray(
        [[0.0, 2.0, 10.0, 14.0], [0.0, 0.0, 0.0, 20.0], [1.0, 2.0, 3.0, 4.0]]
    )

    balanced, missing = inference._class_balanced_batch_mean(
        gold_batch=gold, value_batch=values, n_labels=2
    )

    assert balanced == pytest.approx([6.5, 10.0, 2.5])
    assert missing == 1


def test_shared_bootstrap_and_degenerate_permutation_preserve_pairing() -> None:
    size = 12
    gold = np.arange(size, dtype=np.int64) % len(LABELS)
    support = inference.SupportData(
        support_id=inference.FULL_SUPPORT,
        indices=np.arange(size, dtype=np.int64),
        event_ids=tuple(f"event-{index}" for index in range(size)),
        gold_ids=gold,
        event_id_sequence_sha256="synthetic",
    )
    action_a = _action("a", size=size)
    action_b = _action("b", size=size)

    bootstrap, _ = inference.bootstrap_action_scores(
        support=support,
        actions=[action_a, action_b],
        n_labels=len(LABELS),
        n_resamples=25,
        seed=7,
        stratified=True,
    )
    for metric_name in inference.METRIC_NAMES:
        assert np.array_equal(bootstrap[metric_name][:, 0], bootstrap[metric_name][:, 1])

    null = inference.paired_permutation_null(
        support=support,
        a_action=action_a,
        b_action=action_b,
        n_labels=len(LABELS),
        n_resamples=25,
        seed=11,
    )
    assert np.array_equal(null, np.zeros_like(null))
    summary = paired._permutation_summary(
        null[:, 0], observed_delta=0.0, n_resamples=25, seed=11
    )
    assert summary["p_value"] == 1.0


def test_analysis_contract_requires_formal_native_equivalence_gate(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "materialized"
    matrix_dir.mkdir()
    gate_path = matrix_dir / "equivalence_gate.json"
    paired._write_json(gate_path, {"passed": True, "status": "passed"})
    matrix_path = matrix_dir / "matrix_manifest.json"
    matrix = {
        "status": "complete",
        "diagnostic_only": False,
        "equivalence_gate": gate_path.name,
        "equivalence_gate_sha256": paired._sha256_file(gate_path),
    }
    paired._write_json(matrix_path, matrix)
    analysis_path = tmp_path / "capacity_analysis" / "manifest.json"
    analysis = {
        "schema_version": capacity.SCHEMA_VERSION,
        "status": "complete",
        "source": {
            "matrix_manifest": str(matrix_path),
            "matrix_manifest_sha256": paired._sha256_file(matrix_path),
        },
        "implementation": {
            "path": str(Path(capacity.__file__).resolve()),
            "sha256": paired._sha256_file(Path(capacity.__file__).resolve()),
        },
        "artifacts": {},
    }
    paired._write_json(analysis_path, analysis)

    assert inference._verify_analysis_contract(
        analysis_manifest_path=analysis_path,
        matrix_manifest_path=matrix_path,
    )["status"] == "complete"

    paired._write_json(gate_path, {"passed": False, "status": "failed"})
    with pytest.raises(
        inference.CapacityPairedInferenceError, match="equivalence gate"
    ):
        inference._verify_analysis_contract(
            analysis_manifest_path=analysis_path,
            matrix_manifest_path=matrix_path,
        )


def _synthetic_prefix_source() -> tuple[
    dict[str, object],
    list[str],
    list[str],
    np.ndarray,
    dict[str, dict[int, list[capacity.PrefixObservation]]],
    dict[str, str],
]:
    event_ids = [f"event-{index}" for index in range(60)]
    gold_ids = np.arange(len(event_ids), dtype=np.int64) % len(LABELS)
    observations: dict[str, dict[int, list[capacity.PrefixObservation]]] = {}
    for selector_index, selector in enumerate(("baces_exact", "learned_marginal")):
        observations[selector] = {}
        for requested_k in range(1, 11):
            rows = []
            for sample_idx, event_id in enumerate(event_ids):
                gold_id = int(gold_ids[sample_idx])
                pred_id = gold_id
                stride = 3 + ((requested_k + selector_index) % 4)
                if (sample_idx + requested_k + selector_index) % stride == 0:
                    pred_id = (gold_id + 1) % len(LABELS)
                logits = np.zeros(len(LABELS), dtype=np.float64)
                logits[pred_id] = 1.0 + 0.01 * requested_k
                rows.append(
                    capacity.PrefixObservation(
                        selector_level=selector,
                        event_id=event_id,
                        sample_idx=sample_idx,
                        requested_k=requested_k,
                        realized_k=requested_k,
                        prompt_token_count=200 + requested_k * 10 + selector_index,
                        prompt_hash=f"{selector}:{requested_k}:{event_id}",
                        unique_idx=sample_idx,
                        gold_id=gold_id,
                        logits=logits,
                    )
                )
            observations[selector][requested_k] = rows
    return (
        {"status": "complete", "split": "val"},
        list(LABELS),
        event_ids,
        gold_ids,
        observations,
        {"kind": "synthetic"},
    )


def _synthetic_curve_rows(
    source: tuple[
        dict[str, object],
        list[str],
        list[str],
        np.ndarray,
        dict[str, dict[int, list[capacity.PrefixObservation]]],
        dict[str, str],
    ]
) -> dict[tuple[str, str, int], dict[str, object]]:
    _, labels, event_ids, gold_ids, observations, _ = source
    actions = inference._build_fixed_actions(
        observations=observations, event_ids=event_ids
    )
    indices = np.arange(len(event_ids), dtype=np.int64)
    event_hash = capacity._event_sequence_sha256(event_ids)
    supports = {
        support_id: inference.SupportData(
            support_id=support_id,
            indices=indices,
            event_ids=tuple(event_ids),
            gold_ids=gold_ids,
            event_id_sequence_sha256=event_hash,
        )
        for support_id in (inference.FULL_SUPPORT, inference.STRICT_SUPPORT)
    }
    rows: dict[tuple[str, str, int], dict[str, object]] = {}
    for action in actions.values():
        requested_k = int(action.requested_k[0])
        for support_id, support in supports.items():
            point, _ = inference._point_metrics(
                action=action, support=support, n_labels=len(labels)
            )
            rows[(action.selector_level, support_id, requested_k)] = {
                "selector_level": action.selector_level,
                "support": support_id,
                "requested_k": requested_k,
                "sample_count": len(event_ids),
                "support_event_id_sequence_sha256": event_hash,
                "macro_f1": point["macro_f1"],
                "accuracy": point["accuracy"],
                "class_balanced_nll_mean": point["class_balanced_nll_mean"],
                "raw_nll_mean": point["raw_nll_mean"],
                "mean_realized_k": point["evidence_count_mean"],
                "mean_prompt_token_count": point["prompt_token_count_mean"],
            }
    return rows


def test_materializer_freezes_diagnostic_registry_and_reuses_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic_prefix_source()
    curve_rows = _synthetic_curve_rows(source)
    matrix_path = tmp_path / "matrix_manifest.json"
    analysis_path = tmp_path / "capacity_analysis_manifest.json"
    registry_path = tmp_path / "registry.json"
    paired._write_json(matrix_path, {"synthetic": True})
    paired._write_json(analysis_path, {"synthetic": True})
    shutil.copyfile(REGISTRY_PATH, registry_path)

    monkeypatch.setattr(
        inference,
        "_verify_analysis_contract",
        lambda **kwargs: {"split": "val", "policy_sources": []},
    )
    monkeypatch.setattr(inference, "_load_curve_rows", lambda **kwargs: curve_rows)
    monkeypatch.setattr(capacity, "_load_prefix_source", lambda path: source)

    fixed_actions = inference._build_fixed_actions(
        observations=source[4], event_ids=source[2]
    )
    policy_actions = {
        "policy::ordinal_replay_minmax5_10::baces_exact": inference.ActionData(
            **{
                **fixed_actions["fixed::baces_exact::k05"].__dict__,
                "action_id": "policy::ordinal_replay_minmax5_10::baces_exact",
                "kind": "trace_policy",
            }
        ),
        "policy::ordinal_replay_minmax5_10::learned_marginal": inference.ActionData(
            **{
                **fixed_actions["fixed::learned_marginal::k05"].__dict__,
                "action_id": "policy::ordinal_replay_minmax5_10::learned_marginal",
                "kind": "trace_policy",
            }
        ),
    }
    monkeypatch.setattr(
        inference,
        "_build_policy_actions",
        lambda **kwargs: (policy_actions, []),
    )

    output_dir = tmp_path / "paired"
    result = inference.materialize_capacity_paired_inference(
        matrix_manifest_path=matrix_path,
        capacity_analysis_manifest_path=analysis_path,
        contrast_registry_path=registry_path,
        output_dir=output_dir,
        bootstrap_samples=20,
        permutation_samples=20,
        seed=20260713,
    )

    assert result["status"] == "complete"
    assert result["registered_contrast_count"] == 9
    assert result["comparison_support_row_count"] == 14
    rows = [json.loads(line) for line in (output_dir / "comparisons.jsonl").read_text().splitlines()]
    assert not any(row["valid_for_confirmatory_claim"] for row in rows)
    identical = next(
        row
        for row in rows
        if row["comparison_id"] == "baces_exact__ordinal_minmax_minus_k05"
    )
    for metric_name in inference.METRIC_NAMES:
        metric = identical["metrics"][metric_name]
        assert metric["delta_a_minus_b"] == 0.0
        assert metric["ordinary_paired_bootstrap"]["ci_delta_a_minus_b"] == {
            "low": 0.0,
            "high": 0.0,
        }
        assert metric["paired_permutation"]["p_value"] == 1.0
    for artifact in result["artifacts"].values():
        path = output_dir / artifact["path"]
        assert paired._sha256_file(path) == artifact["sha256"]
    assert not list(output_dir.parent.glob(f".{output_dir.name}.tmp.*"))

    reused = inference.materialize_capacity_paired_inference(
        matrix_manifest_path=matrix_path,
        capacity_analysis_manifest_path=analysis_path,
        contrast_registry_path=registry_path,
        output_dir=output_dir,
        bootstrap_samples=20,
        permutation_samples=20,
        seed=20260713,
    )
    assert reused["created_at"] == result["created_at"]

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    drifted_manifest = json.loads(json.dumps(manifest))
    drifted_manifest["artifacts"].pop("report.md")
    paired._write_json(manifest_path, drifted_manifest)
    with pytest.raises(
        inference.CapacityPairedInferenceError, match="artifact key set drift"
    ):
        inference.materialize_capacity_paired_inference(
            matrix_manifest_path=matrix_path,
            capacity_analysis_manifest_path=analysis_path,
            contrast_registry_path=registry_path,
            output_dir=output_dir,
            bootstrap_samples=20,
            permutation_samples=20,
            seed=20260713,
        )
    paired._write_json(manifest_path, manifest)

    registry_path.write_text(registry_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(
        inference.CapacityPairedInferenceError, match="incompatible"
    ):
        inference.materialize_capacity_paired_inference(
            matrix_manifest_path=matrix_path,
            capacity_analysis_manifest_path=analysis_path,
            contrast_registry_path=registry_path,
            output_dir=output_dir,
            bootstrap_samples=20,
            permutation_samples=20,
            seed=20260713,
        )
