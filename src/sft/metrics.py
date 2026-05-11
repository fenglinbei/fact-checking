import numpy as np
from fact_checking.data.constants import LABELS

def _compute_classification_metrics(
    pred_ids: np.ndarray, gold_ids: np.ndarray, *, labels: list[str] | None = None
) -> dict[str, float | dict[str, dict[str, float]]]:
    _labels = labels if labels is not None else LABELS
    eps = 1e-12
    per_class: dict[str, dict[str, float]] = {}
    p_list: list[float] = []
    r_list: list[float] = []
    f1_list: list[float] = []
    for label_id, label in enumerate(_labels):
        tp = float(np.sum((pred_ids == label_id) & (gold_ids == label_id)))
        fp = float(np.sum((pred_ids == label_id) & (gold_ids != label_id)))
        fn = float(np.sum((pred_ids != label_id) & (gold_ids == label_id)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        p_list.append(precision)
        r_list.append(recall)
        f1_list.append(f1)
    macro_p = float(np.mean(p_list))
    macro_r = float(np.mean(r_list))
    macro_f1 = float(np.mean(f1_list))
    parse_error_rate = float(np.mean(pred_ids < 0))
    accuracy = float(np.mean(pred_ids == gold_ids))
    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "parse_error_rate": parse_error_rate,
        "per_class": per_class,
    }

def _build_confusion_matrix(
    pred_ids: np.ndarray, gold_ids: np.ndarray, *, labels: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    _labels = labels if labels is not None else LABELS
    labels_with_parse = _labels + ["parse_error"]
    mat = np.zeros((len(_labels), len(labels_with_parse)), dtype=np.int64)
    for g, p in zip(gold_ids.tolist(), pred_ids.tolist()):
        pred_idx = p if p >= 0 else len(_labels)
        mat[g, pred_idx] += 1
    return mat, labels_with_parse