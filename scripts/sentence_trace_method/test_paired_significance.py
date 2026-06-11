from __future__ import annotations

import pytest

from scripts.sentence_trace_method import paired_significance


def test_mcnemar_exact_uses_discordant_paired_counts() -> None:
    old_correct = [False, False, False, True]
    new_correct = [True, True, True, True]

    result = paired_significance.mcnemar_exact(old_correct, new_correct)

    assert result["new_correct_old_wrong"] == 3
    assert result["old_correct_new_wrong"] == 0
    assert result["discordant"] == 3
    assert result["p_value_two_sided"] == pytest.approx(0.25)


def test_compute_metrics_recomputes_macro_true_side_and_selection() -> None:
    gold = [0, 1, 0, 1]
    pred = [0, 1, 1, 1]

    metrics = paired_significance.compute_metrics(
        gold,
        pred,
        label_count=2,
        true_side_label_ids=[1],
        true_side_weight=0.5,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["true_side_macro_f1"] == pytest.approx(0.8)
    assert metrics["selection_score"] == pytest.approx(((2 / 3 + 0.8) / 2) + 0.5 * 0.8)


def test_compute_metrics_supports_calibrated_selection_with_ordinal_mae() -> None:
    gold = [0, 1, 0, 1]
    pred = [0, 1, 1, 1]

    metrics = paired_significance.compute_metrics(
        gold,
        pred,
        label_count=2,
        true_side_label_ids=[1],
        true_side_weight=0.5,
        early_stopping_metric="macro_f1_plus_true_side_plus_mae",
        mae_metric_weight=0.3,
    )

    assert metrics["ordinal_mae_norm"] == pytest.approx(0.25)
    assert metrics["selection_score"] == pytest.approx(((2 / 3 + 0.8) / 2) + 0.5 * 0.8 + 0.3 * 0.75)


def test_align_predictions_rejects_mismatched_gold_ids() -> None:
    old_rows = [
        {"sample_idx": 0, "gold_id": 1, "pred_id": 1},
        {"sample_idx": 1, "gold_id": 0, "pred_id": 1},
    ]
    new_rows = [
        {"sample_idx": 0, "gold_id": 1, "pred_id": 1},
        {"sample_idx": 1, "gold_id": 1, "pred_id": 1},
    ]

    with pytest.raises(ValueError, match="gold_id mismatch"):
        paired_significance.align_predictions(old_rows, new_rows)
