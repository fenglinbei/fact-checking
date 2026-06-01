from __future__ import annotations

import numpy as np

from sft.eval import build_eval_metrics
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
