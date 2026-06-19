from __future__ import annotations

import json
from pathlib import Path

from sft.checkpoint_selection import (
    checkpoint_selection_score,
    macro_f1_bootstrap_se_from_records,
    metric_value,
    select_macro_f1_checkpoint,
    select_one_standard_error_checkpoint,
)


def test_checkpoint_selection_score_is_full_val_macro_f1_regardless_of_configured_metric() -> None:
    metrics = {
        "accuracy": 0.40,
        "macro_f1": 0.50,
        "ordinal_mae_norm": 0.25,
        "per_class": {
            "mostly-true": {"f1": 0.60},
            "true": {"f1": 0.80},
        },
    }
    train_cfg = {
        "label_token_ce": {
            "early_stopping_metric": "macro_f1_plus_true_side_plus_mae",
            "true_side_metric_weight": 0.5,
            "mae_metric_weight": 0.3,
        }
    }

    assert checkpoint_selection_score(metrics, train_cfg) == 0.50


def test_metric_value_uses_macro_f1_for_checkpoint_selection_score_with_legacy_fallbacks() -> None:
    assert (
        metric_value(
            {"macro_f1": 0.9, "checkpoint_selection_score": 0.8, "selection_score": 0.7},
            "checkpoint_selection_score",
        )
        == 0.9
    )
    assert (
        metric_value({"checkpoint_selection_score": 0.8, "selection_score": 0.7}, "checkpoint_selection_score")
        == 0.8
    )
    assert metric_value({"selection_score": 0.7}, "checkpoint_selection_score") == 0.7
    assert metric_value({"macro_f1": 0.6}, "checkpoint_selection_score") == 0.6


def test_one_standard_error_rule_selects_earliest_checkpoint_within_best_se() -> None:
    candidates = [
        {"checkpoint": "checkpoint-100", "step": 100, "macro_f1": 0.610, "macro_f1_se": 0.020},
        {"checkpoint": "checkpoint-200", "step": 200, "macro_f1": 0.625, "macro_f1_se": 0.015},
        {"checkpoint": "checkpoint-300", "step": 300, "macro_f1": 0.630, "macro_f1_se": 0.020},
    ]

    selected = select_one_standard_error_checkpoint(candidates)

    assert selected["checkpoint"] == "checkpoint-100"
    assert selected["one_se_threshold"] == 0.610
    assert selected["one_se_best_checkpoint"] == "checkpoint-300"


def test_macro_f1_checkpoint_selects_highest_macro_f1_then_earliest_step() -> None:
    candidates = [
        {"checkpoint": "checkpoint-300", "step": 300, "macro_f1": 0.63},
        {"checkpoint": "checkpoint-200", "step": 200, "macro_f1": 0.63},
        {"checkpoint": "checkpoint-100", "step": 100, "macro_f1": 0.62},
    ]

    selected = select_macro_f1_checkpoint(candidates)

    assert selected["checkpoint"] == "checkpoint-200"


def test_macro_f1_bootstrap_se_from_prediction_records_is_deterministic() -> None:
    records = [
        {"sample_idx": idx, "pred_id": pred, "gold_id": gold}
        for idx, (pred, gold) in enumerate(
            [
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 1),
                (2, 2),
                (2, 0),
            ]
        )
    ]

    first = macro_f1_bootstrap_se_from_records(records, labels=["false", "half", "true"], n_bootstrap=200, seed=7)
    second = macro_f1_bootstrap_se_from_records(records, labels=["false", "half", "true"], n_bootstrap=200, seed=7)

    assert first == second
    assert first > 0.0


def test_loadable_step_metrics_can_drive_one_se_selection(tmp_path: Path) -> None:
    case_root = tmp_path / "run"
    for step, macro_f1 in ((100, 0.61), (200, 0.625), (300, 0.63)):
        step_dir = case_root / "eval" / f"step-{step}"
        step_dir.mkdir(parents=True)
        (step_dir / "metrics.json").write_text(
            json.dumps({"macro_f1": macro_f1, "macro_f1_se": 0.02, "checkpoint_selection_score": macro_f1 + 0.1}),
            encoding="utf-8",
        )

    selected = select_one_standard_error_checkpoint(case_root)

    assert selected["checkpoint"] == "checkpoint-100"
