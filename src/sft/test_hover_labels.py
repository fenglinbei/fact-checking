from __future__ import annotations

import numpy as np

from sft.eval import build_eval_metrics
from sft.metrics import _build_confusion_matrix
from sft.parser import _parse_label_id


HOVER_LABELS = ["supported", "not_supported"]


def test_hover_parser_accepts_schema_letters_and_names() -> None:
    assert _parse_label_id("Label: A", label_schema="hover2") == 0
    assert _parse_label_id("Label: B", label_schema="hover2") == 1
    assert _parse_label_id("Label: C", label_schema="hover2") == -1
    assert _parse_label_id("Label: supported", label_schema="hover2") == 0
    assert _parse_label_id("Label: NOT_SUPPORTED", label_schema="hover2") == 1
    assert _parse_label_id("Label: not supported", label_schema="hover2") == 1


def test_hover_metrics_use_two_labels_plus_parse_error_column() -> None:
    pred_ids = np.asarray([0, 1, -1], dtype=np.int64)
    gold_ids = np.asarray([0, 1, 1], dtype=np.int64)

    matrix, labels = _build_confusion_matrix(pred_ids, gold_ids, labels=HOVER_LABELS)
    metrics = build_eval_metrics(pred_ids, gold_ids, labels=HOVER_LABELS)

    assert matrix.shape == (2, 3)
    assert labels == ["supported", "not_supported", "parse_error"]
    assert metrics["confusion_matrix"].shape == (2, 3)
    assert set(metrics["per_class"]) == set(HOVER_LABELS)
