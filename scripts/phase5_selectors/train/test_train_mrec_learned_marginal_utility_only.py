from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fact_checking.selectors import mrec_learned_marginal as mrec
from scripts.phase5_selectors.train import train_mrec_learned_marginal_utility_only as module


def _features(candidate_idx: int, step: int) -> dict[str, float]:
    out = {name: 0.0 for name in mrec.FEATURE_NAMES}
    out["resolution_delta"] = float(candidate_idx + step / 10.0)
    out["new_atom_coverage"] = float((candidate_idx + 1) % 3) / 3.0
    out["cost_ratio"] = float(candidate_idx) / 10.0
    return out


def _event_rows(
    event_id: str,
    *,
    split: str,
    run_fingerprint: str,
    gold_label: str = "false",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected_by_step = {0: 1, 1: 0, 2: 2}
    deltas_by_step = {
        0: {0: 0.2, 1: 0.9, 2: 0.3},
        1: {0: 0.8, 2: 0.1},
        2: {2: 0.4},
    }
    prefix_by_step = {0: [], 1: [1], 2: [1, 0]}
    for step, deltas in deltas_by_step.items():
        for candidate_idx, delta in deltas.items():
            rows.append(
                {
                    "event_id": event_id,
                    "split": split,
                    "step": step,
                    "prefix_indices": list(prefix_by_step[step]),
                    "candidate_idx": candidate_idx,
                    "selector_candidate_idx": candidate_idx,
                    "delta_margin": delta,
                    "utility_selected": candidate_idx == selected_by_step[step],
                    "mrec_features": _features(candidate_idx, step),
                    "rollin_policy": module.ROLLIN_POLICY,
                    "reward_source": module.REWARD_SOURCE,
                    "run_fingerprint": run_fingerprint,
                    "teacher_fingerprint": "teacher-v1",
                    "scoring_fingerprint": "scoring-v1",
                    "candidate_pool_fingerprint": f"pool-{event_id}",
                    "gold_label": gold_label,
                    "teacher_order": [2, 1, 0],
                    "oracle_ordered_keys": ["poison"],
                }
            )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_happy_path_accepts_multiple_shards_and_writes_strict_contract(tmp_path: Path) -> None:
    train_0 = tmp_path / "train-0.jsonl"
    train_1 = tmp_path / "train-1.jsonl"
    val_0 = tmp_path / "val-0.jsonl"
    output_dir = tmp_path / "weights"
    _write_rows(train_0, _event_rows("train-a", split="train", run_fingerprint="run-train-0"))
    _write_rows(train_1, _event_rows("train-b", split="train", run_fingerprint="run-train-1"))
    _write_rows(val_0, _event_rows("val-a", split="val", run_fingerprint="run-val-0"))

    assert module.main(
        [
            "--train-reward-input",
            str(train_0),
            str(train_1),
            "--val-reward-input",
            str(val_0),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "2",
        ]
    ) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = json.loads((output_dir / "weights.json").read_text(encoding="utf-8"))
    train_metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((output_dir / "val_metrics.json").read_text(encoding="utf-8"))
    assert manifest["training_supervision"] == "verifier_utility_only"
    assert manifest["compute_device"] == "cpu"
    assert manifest["params"]["objective"] == "winner_vs_rest_pairwise_logistic"
    assert manifest["params"]["listwise_weight"] == 0.0
    assert manifest["params"]["huber_weight"] == 0.0
    assert manifest["params"]["prior_weight"] == 0.0
    assert manifest["params"]["expected_rollout_steps"] == 5
    assert manifest["train_input_audit"]["input_count"] == 2
    assert manifest["train_input_audit"]["expected_rollout_steps"] == 5
    assert set(manifest["train_input_audit"]["run_fingerprints_by_input"].values()) == {
        "run-train-0",
        "run-train-1",
    }
    assert weights["metadata"]["supervision_mode"] == "verifier_utility_only"
    assert weights["metadata"]["initialized_from"] == "equal_weight_neutral_v0_1"
    assert weights["metadata"]["gold_label_read_count"] == 0
    assert weights["metadata"]["teacher_structure_read_count"] == 0
    assert weights["metadata"]["oracle_read_row_count"] == 0
    assert train_metrics["pair_count"] == 6
    assert val_metrics["pair_count"] == 3
    assert train_metrics["supervision_fingerprint"]
    assert val_metrics["supervision_fingerprint"]

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        module.main(
            [
                "--train-reward-input",
                str(train_0),
                "--val-reward-input",
                str(val_0),
                "--output-dir",
                str(output_dir),
            ]
        )


def test_missing_candidate_at_later_step_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = _event_rows("train-a", split="train", run_fingerprint="run-train")
    rows = [row for row in rows if not (row["step"] == 1 and row["candidate_idx"] == 2)]
    _write_rows(path, rows)

    with pytest.raises(ValueError, match="incomplete candidate coverage"):
        module.load_and_validate_utility_inputs([path], expected_split="train")


def test_missing_complete_tail_step_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = _event_rows("train-a", split="train", run_fingerprint="run-train")
    rows = [row for row in rows if row["step"] != 2]
    _write_rows(path, rows)

    with pytest.raises(ValueError, match="incomplete rollout step coverage"):
        module.load_and_validate_utility_inputs([path], expected_split="train")


def test_wrong_prefix_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = _event_rows("train-a", split="train", run_fingerprint="run-train")
    for row in rows:
        if row["step"] == 1:
            row["prefix_indices"] = [0]
    _write_rows(path, rows)

    with pytest.raises(ValueError, match="wrong prefix"):
        module.load_and_validate_utility_inputs([path], expected_split="train")


def test_test_split_is_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    _write_rows(path, _event_rows("test-a", split="test", run_fingerprint="run-test"))

    with pytest.raises(ValueError, match="test reward rows are forbidden"):
        module.load_and_validate_utility_inputs([path], expected_split="train")


def test_wrong_utility_selected_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = _event_rows("train-a", split="train", run_fingerprint="run-train")
    for row in rows:
        if row["step"] == 0:
            row["utility_selected"] = row["candidate_idx"] == 2
    _write_rows(path, rows)

    with pytest.raises(ValueError, match="wrong utility_selected"):
        module.load_and_validate_utility_inputs([path], expected_split="train")


def test_gold_and_structure_poison_do_not_change_weights(tmp_path: Path) -> None:
    clean_train = tmp_path / "clean-train.jsonl"
    poison_train = tmp_path / "poison-train.jsonl"
    clean_val = tmp_path / "clean-val.jsonl"
    poison_val = tmp_path / "poison-val.jsonl"
    clean_train_rows = _event_rows("train-a", split="train", run_fingerprint="run-train", gold_label="false")
    clean_val_rows = _event_rows("val-a", split="val", run_fingerprint="run-val", gold_label="false")
    poison_train_rows = copy.deepcopy(clean_train_rows)
    poison_val_rows = copy.deepcopy(clean_val_rows)
    for row in [*poison_train_rows, *poison_val_rows]:
        row["gold_label"] = "true"
        row["teacher_order"] = [0, 1, 2]
        row["oracle_ordered_keys"] = ["different-poison"]
    _write_rows(clean_train, clean_train_rows)
    _write_rows(poison_train, poison_train_rows)
    _write_rows(clean_val, clean_val_rows)
    _write_rows(poison_val, poison_val_rows)

    clean_output = tmp_path / "clean-weights"
    poison_output = tmp_path / "poison-weights"
    common = ["--epochs", "3", "--learning-rate", "0.05"]
    assert module.main(
        [
            "--train-reward-input",
            str(clean_train),
            "--val-reward-input",
            str(clean_val),
            "--output-dir",
            str(clean_output),
            *common,
        ]
    ) == 0
    assert module.main(
        [
            "--train-reward-input",
            str(poison_train),
            "--val-reward-input",
            str(poison_val),
            "--output-dir",
            str(poison_output),
            *common,
        ]
    ) == 0

    clean_weights = json.loads((clean_output / "weights.json").read_text(encoding="utf-8"))
    poison_weights = json.loads((poison_output / "weights.json").read_text(encoding="utf-8"))
    assert poison_weights == clean_weights


def test_utility_pair_and_same_tiny_structure_pair_share_exact_fitter_result(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = _event_rows("train-a", split="train", run_fingerprint="run-train")
    rows = [row for row in rows if row["step"] == 0 and row["candidate_idx"] in {0, 1}]
    _write_rows(path, rows)
    data = module.load_and_validate_utility_inputs(
        [path], expected_split="train", expected_rollout_steps=1
    )

    utility_weights, utility_metrics = mrec.fit_pairwise_marginal_scorer(
        data.positive_features,
        data.negative_features,
        initial_weights=mrec.initial_neutral_learned_marginal_weights(),
        epochs=4,
        learning_rate=0.05,
        metadata={"source": "shared"},
    )
    structure_weights, structure_metrics = mrec.fit_pairwise_marginal_scorer(
        [data.positive_features[0]],
        [data.negative_features[0]],
        initial_weights=mrec.initial_neutral_learned_marginal_weights(),
        epochs=4,
        learning_rate=0.05,
        metadata={"source": "shared"},
    )

    assert utility_metrics == structure_metrics
    assert utility_weights.to_json_dict() == structure_weights.to_json_dict()
