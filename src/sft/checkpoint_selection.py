from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from sft.metrics import _compute_classification_metrics


def true_side_macro_f1(metrics: Mapping[str, Any]) -> float:
    per_class = metrics.get("per_class", {}) or {}
    values: list[float] = []
    for label in ("mostly-true", "true"):
        label_metrics = per_class.get(label, {}) if isinstance(per_class, Mapping) else {}
        if isinstance(label_metrics, Mapping):
            values.append(float(label_metrics.get("f1", 0.0)))
    return float(np.mean(values)) if values else 0.0


def checkpoint_selection_score(metrics: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> float:
    _ = train_cfg
    return float(metrics["macro_f1"])


def metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    key = str(metric)
    if key == "checkpoint_selection_score":
        value = metrics.get("macro_f1", metrics.get("checkpoint_selection_score", metrics.get("selection_score")))
    else:
        value = metrics.get(key)
    if value is None:
        raise KeyError(f"metric {metric!r} is missing")
    return float(value)


def macro_f1_bootstrap_se_from_records(
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    labels: list[str],
    n_bootstrap: int,
    seed: int,
) -> float:
    records = [record for record in prediction_records if int(record.get("gold_id", -1)) >= 0]
    if len(records) <= 1 or int(n_bootstrap) <= 1:
        return 0.0
    pred_ids = np.asarray([int(record["pred_id"]) for record in records], dtype=np.int64)
    gold_ids = np.asarray([int(record["gold_id"]) for record in records], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, len(records), size=len(records))
        metrics = _compute_classification_metrics(pred_ids[sample_idx], gold_ids[sample_idx], labels=labels)
        values.append(float(metrics["macro_f1"]))
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def select_macro_f1_checkpoint(candidates: Iterable[Mapping[str, Any]] | str | Path) -> dict[str, Any]:
    rows = _candidate_rows(candidates)
    if not rows:
        raise ValueError("No checkpoint candidates were found.")
    return dict(sorted(rows, key=lambda row: (-float(row["macro_f1"]), int(row["step"])))[0])


def select_one_standard_error_checkpoint(candidates: Iterable[Mapping[str, Any]] | str | Path) -> dict[str, Any]:
    rows = _candidate_rows(candidates)
    if not rows:
        raise ValueError("No checkpoint candidates were found.")
    best = sorted(rows, key=lambda row: (-float(row["macro_f1"]), int(row["step"])))[0]
    threshold = float(best["macro_f1"]) - float(best.get("macro_f1_se", 0.0))
    eligible = [row for row in rows if float(row["macro_f1"]) >= threshold]
    selected = dict(sorted(eligible, key=lambda row: int(row["step"]))[0])
    selected["one_se_best_checkpoint"] = str(best["checkpoint"])
    selected["one_se_best_macro_f1"] = float(best["macro_f1"])
    selected["one_se_best_macro_f1_se"] = float(best.get("macro_f1_se", 0.0))
    selected["one_se_threshold"] = float(threshold)
    return selected


def load_step_metric_candidates(case_root: str | Path) -> list[dict[str, Any]]:
    root = Path(case_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((root / "eval").glob("step-*/metrics.json")):
        match = re.fullmatch(r"step-(\d+)", metrics_path.parent.name)
        if not match:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        step = int(match.group(1))
        row = {
            "checkpoint": f"checkpoint-{step}",
            "step": step,
            "metrics_path": str(metrics_path),
            "macro_f1": float(metrics["macro_f1"]),
            "checkpoint_selection_score": metric_value(metrics, "checkpoint_selection_score"),
        }
        if metrics.get("macro_f1_se") is not None:
            row["macro_f1_se"] = float(metrics["macro_f1_se"])
        else:
            predictions_path = metrics_path.parent / "val_predictions.jsonl"
            if predictions_path.exists():
                records = [
                    json.loads(line)
                    for line in predictions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                row["macro_f1_se"] = macro_f1_bootstrap_se_from_records(
                    records,
                    labels=_labels_from_metrics(metrics),
                    n_bootstrap=1000,
                    seed=20260620 + step,
                )
            else:
                row["macro_f1_se"] = 0.0
        rows.append(row)
    return rows


def _candidate_rows(candidates: Iterable[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(candidates, str | Path):
        return load_step_metric_candidates(candidates)
    return [dict(row) for row in candidates]


def _labels_from_metrics(metrics: Mapping[str, Any]) -> list[str]:
    per_class = metrics.get("per_class", {}) or {}
    if isinstance(per_class, Mapping) and per_class:
        return [str(label) for label in per_class.keys()]
    return ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
