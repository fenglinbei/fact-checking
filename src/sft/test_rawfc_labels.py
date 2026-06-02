from __future__ import annotations

import numpy as np

from sft.eval import build_eval_metrics
from sft.eval import deduplicate_by_sample_idx
from sft.label_token_trainer import _selection_score
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
