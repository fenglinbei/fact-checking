from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from sft.metrics import _compute_classification_metrics


_EVIDENCE_ARM_ALIASES = {
    "evitrace": "evitrace",
    "evi_trace": "evitrace",
    "evi-trace": "evitrace",
    "evi": "evitrace",
    "s4": "s4",
    "control": "s4",
    "source_score": "s4",
    "source-score": "s4",
}
_CANONICAL_EVIDENCE_ARMS = ("evitrace", "s4")


def normalize_evidence_arm(value: object) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return _EVIDENCE_ARM_ALIASES.get(normalized, normalized)


def arm_balanced_metrics_from_records(
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    labels: list[str],
) -> dict[str, Any]:
    records = [dict(record) for record in prediction_records]
    invalid: dict[str, Any] = {"arm_balanced_valid": False}
    if not records:
        return {**invalid, "arm_balanced_reason": "no_prediction_records"}

    assignment_ids = [str(record.get("assignment_id") or "").strip() for record in records]
    nonempty_assignment_ids = {assignment_id for assignment_id in assignment_ids if assignment_id}
    if nonempty_assignment_ids and (
        len(nonempty_assignment_ids) != 1 or any(not assignment_id for assignment_id in assignment_ids)
    ):
        return {**invalid, "arm_balanced_reason": "inconsistent_assignment_id"}

    by_event: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        event_id = str(record.get("event_id") or "").strip()
        evidence_arm = normalize_evidence_arm(record.get("evidence_arm"))
        if not event_id:
            return {**invalid, "arm_balanced_reason": "missing_event_id"}
        if evidence_arm not in _CANONICAL_EVIDENCE_ARMS:
            return {**invalid, "arm_balanced_reason": "invalid_evidence_arm"}
        record["evidence_arm"] = evidence_arm
        by_event.setdefault(event_id, []).append(record)

    for event_records in by_event.values():
        if len(event_records) != 2:
            return {**invalid, "arm_balanced_reason": "event_not_exactly_paired"}
        if {str(record["evidence_arm"]) for record in event_records} != set(_CANONICAL_EVIDENCE_ARMS):
            return {**invalid, "arm_balanced_reason": "event_missing_arm"}
        if len({int(record.get("gold_id", -1)) for record in event_records}) != 1:
            return {**invalid, "arm_balanced_reason": "paired_gold_mismatch"}

    result: dict[str, Any] = {
        "arm_balanced_valid": True,
        "arm_balanced_num_events": len(by_event),
    }
    arm_mean_ces: list[float] = []
    for arm in _CANONICAL_EVIDENCE_ARMS:
        arm_records = [record for record in records if record["evidence_arm"] == arm]
        pred_ids = np.asarray([int(record["pred_id"]) for record in arm_records], dtype=np.int64)
        gold_ids = np.asarray([int(record["gold_id"]) for record in arm_records], dtype=np.int64)
        arm_metrics = _compute_classification_metrics(pred_ids, gold_ids, labels=labels)
        result[f"macro_f1_{arm}"] = float(arm_metrics["macro_f1"])

        ce_values = np.asarray(
            [float(record.get("ce_loss", float("nan"))) for record in arm_records],
            dtype=np.float64,
        )
        mean_ce = float(np.mean(ce_values)) if len(ce_values) and np.isfinite(ce_values).all() else float("nan")
        result[f"mean_ce_{arm}"] = mean_ce
        arm_mean_ces.append(mean_ce)

    result["arm_balanced_macro_f1"] = float(
        np.mean([result["macro_f1_evitrace"], result["macro_f1_s4"]])
    )
    result["arm_balanced_mean_ce"] = (
        float(np.mean(arm_mean_ces)) if all(math.isfinite(value) for value in arm_mean_ces) else float("nan")
    )
    return result


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
    arm_score = metrics.get("arm_balanced_macro_f1")
    if bool(metrics.get("arm_balanced_valid", False)) and arm_score is not None:
        arm_score = float(arm_score)
        if math.isfinite(arm_score):
            return arm_score
    return float(metrics["macro_f1"])


def metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    key = str(metric)
    if key == "checkpoint_selection_score":
        if bool(metrics.get("arm_balanced_valid", False)) and metrics.get("arm_balanced_macro_f1") is not None:
            value = metrics["arm_balanced_macro_f1"]
        else:
            value = metrics.get("macro_f1", metrics.get("checkpoint_selection_score", metrics.get("selection_score")))
    else:
        value = metrics.get(key)
    if value is None:
        raise KeyError(f"metric {metric!r} is missing")
    return float(value)


def checkpoint_tiebreak_mean_ce(metrics: Mapping[str, Any]) -> float:
    if bool(metrics.get("arm_balanced_valid", False)):
        value = metrics.get("arm_balanced_mean_ce")
    else:
        value = metrics.get("eval_ce_loss")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return numeric if math.isfinite(numeric) else float("inf")


def checkpoint_candidate_is_better(
    *,
    score: float,
    mean_ce: float,
    step: int,
    best_score: float,
    best_mean_ce: float,
    best_step: int | None,
) -> bool:
    score = float(score)
    best_score = float(best_score)
    if score > best_score and not math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12):
        return True
    if score < best_score and not math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12):
        return False

    normalized_mean_ce = float(mean_ce) if math.isfinite(float(mean_ce)) else float("inf")
    normalized_best_mean_ce = (
        float(best_mean_ce) if math.isfinite(float(best_mean_ce)) else float("inf")
    )
    if normalized_mean_ce < normalized_best_mean_ce and not math.isclose(
        normalized_mean_ce,
        normalized_best_mean_ce,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return True
    if normalized_mean_ce > normalized_best_mean_ce and not math.isclose(
        normalized_mean_ce,
        normalized_best_mean_ce,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    return best_step is None or int(step) < int(best_step)


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
    return dict(
        sorted(
            rows,
            key=lambda row: (
                -float(row["macro_f1"]),
                checkpoint_tiebreak_mean_ce(row),
                int(row["step"]),
            ),
        )[0]
    )


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
        for key in (
            "arm_balanced_valid",
            "macro_f1_evitrace",
            "macro_f1_s4",
            "arm_balanced_macro_f1",
            "mean_ce_evitrace",
            "mean_ce_s4",
            "arm_balanced_mean_ce",
            "eval_ce_loss",
        ):
            if metrics.get(key) is not None:
                row[key] = metrics[key]
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
