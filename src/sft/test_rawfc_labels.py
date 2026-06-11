from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sft.eval import build_eval_metrics
from sft.eval import deduplicate_by_sample_idx
from sft.label_token_trainer import _apply_label_logit_adjust, _compute_label_token_losses, _selection_score
from sft.metrics import _build_confusion_matrix
from sft.parser import _parse_label_id


RAWFC_LABELS = ["false", "half", "true"]


def test_rawfc_parser_accepts_only_schema_letters() -> None:
    assert _parse_label_id("Label: A", label_schema="rawfc3") == 0
    assert _parse_label_id("Label: B", label_schema="rawfc3") == 1
    assert _parse_label_id("Label: C", label_schema="rawfc3") == 2
    assert _parse_label_id("Label: D", label_schema="rawfc3") == -1
    assert _parse_label_id("Label: E", label_schema="rawfc3") == -1
    assert _parse_label_id("Label: F", label_schema="rawfc3") == -1
    assert _parse_label_id("Label: half", label_schema="rawfc3") == 1


def test_rawfc_metrics_use_three_labels_plus_parse_error_column() -> None:
    pred_ids = np.asarray([0, 1, 2, -1], dtype=np.int64)
    gold_ids = np.asarray([0, 1, 2, 2], dtype=np.int64)

    matrix, labels = _build_confusion_matrix(pred_ids, gold_ids, labels=RAWFC_LABELS)
    metrics = build_eval_metrics(pred_ids, gold_ids, labels=RAWFC_LABELS)

    assert matrix.shape == (3, 4)
    assert labels == ["false", "half", "true", "parse_error"]
    assert metrics["confusion_matrix"].shape == (3, 4)
    assert set(metrics["per_class"]) == set(RAWFC_LABELS)


def test_eval_deduplicates_gathered_sample_indices_before_metrics() -> None:
    pred_ids = np.asarray([1, 0, 1, 2], dtype=np.int64)
    gold_ids = np.asarray([1, 0, 0, 2], dtype=np.int64)
    sample_indices = np.asarray([1, 0, 0, 2], dtype=np.int64)

    dedup_pred, dedup_gold, dedup_indices = deduplicate_by_sample_idx(pred_ids, gold_ids, sample_indices)

    assert dedup_indices.tolist() == [0, 1, 2]
    assert dedup_pred.tolist() == [0, 1, 2]
    assert dedup_gold.tolist() == [0, 1, 2]


def test_focus_label_selection_score_adds_focus_f1_weight() -> None:
    metrics = {
        "macro_f1": 0.50,
        "per_class": {
            "false": {"f1": 0.40},
            "half": {"f1": 0.60},
            "true": {"f1": 0.50},
        },
    }
    train_cfg = {
        "label_token_ce": {
            "early_stopping_metric": "macro_f1_plus_focus_label",
            "focus_label": "half",
            "focus_metric_weight": 0.3,
        }
    }

    assert abs(_selection_score(metrics, train_cfg) - 0.68) < 1e-12


def test_ordinal_loss_disabled_matches_weighted_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]], dtype=torch.float32)
    gold_ids = torch.tensor([0, 2], dtype=torch.long)
    class_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    train_cfg = {
        "label_token_ce": {
            "ordinal_loss": {
                "enabled": False,
                "alpha": 0.2,
            }
        }
    }

    losses = _compute_label_token_losses(
        label_logits=logits,
        gold_ids=gold_ids,
        class_weights=class_weights,
        train_cfg=train_cfg,
    )

    expected = F.cross_entropy(logits, gold_ids, weight=class_weights)
    assert torch.allclose(losses["loss"], expected)
    assert torch.allclose(losses["ce_loss"], expected)
    assert float(losses["ordinal_loss"]) == 0.0


def test_coverage_auxiliary_loss_is_added_when_enabled() -> None:
    label_logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32)
    coverage_logits = torch.tensor([[0.0, 3.0, -1.0]], dtype=torch.float32)
    gold_ids = torch.tensor([0], dtype=torch.long)
    coverage_gold_ids = torch.tensor([2], dtype=torch.long)
    class_weights = torch.ones(3, dtype=torch.float32)
    coverage_class_weights = torch.ones(3, dtype=torch.float32)
    train_cfg = {
        "label_token_ce": {"ordinal_loss": {"enabled": False}},
        "coverage_label_token": {"enabled": True, "loss_weight": 0.4},
    }

    losses = _compute_label_token_losses(
        label_logits=label_logits,
        gold_ids=gold_ids,
        class_weights=class_weights,
        train_cfg=train_cfg,
        coverage_label_logits=coverage_logits,
        coverage_gold_ids=coverage_gold_ids,
        coverage_class_weights=coverage_class_weights,
    )

    expected_truth = F.cross_entropy(label_logits, gold_ids)
    expected_coverage = F.cross_entropy(coverage_logits, coverage_gold_ids)
    assert torch.allclose(losses["ce_loss"], expected_truth)
    assert torch.allclose(losses["coverage_ce_loss"], expected_coverage)
    assert torch.allclose(losses["loss"], expected_truth + 0.4 * expected_coverage)


def test_label_logit_adjust_changes_prediction_logits_only() -> None:
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    logit_adjust_cfg = {
        "enabled": True,
        "tau": 1.0,
        "log_priors": [float(np.log(0.05)), float(np.log(0.95))],
    }

    adjusted = _apply_label_logit_adjust(logits, logit_adjust_cfg)

    assert int(adjusted.argmax(dim=-1).item()) == 0
    losses = _compute_label_token_losses(
        label_logits=logits,
        gold_ids=torch.tensor([1], dtype=torch.long),
        class_weights=torch.ones(2, dtype=torch.float32),
        train_cfg={"label_token_ce": {"ordinal_loss": {"enabled": False}}},
    )
    expected = F.cross_entropy(logits, torch.tensor([1], dtype=torch.long))
    assert torch.allclose(losses["loss"], expected)


def test_ordinal_loss_penalizes_farther_label_probability_more() -> None:
    gold_ids = torch.tensor([0], dtype=torch.long)
    class_weights = torch.ones(3, dtype=torch.float32)
    train_cfg = {
        "label_token_ce": {
            "ordinal_loss": {
                "enabled": True,
                "alpha": 1.0,
                "normalize_distance": False,
            }
        }
    }
    near_wrong_logits = torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float32)
    far_wrong_logits = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float32)

    near_losses = _compute_label_token_losses(
        label_logits=near_wrong_logits,
        gold_ids=gold_ids,
        class_weights=class_weights,
        train_cfg=train_cfg,
    )
    far_losses = _compute_label_token_losses(
        label_logits=far_wrong_logits,
        gold_ids=gold_ids,
        class_weights=class_weights,
        train_cfg=train_cfg,
    )

    assert float(far_losses["ordinal_loss"]) > float(near_losses["ordinal_loss"])
