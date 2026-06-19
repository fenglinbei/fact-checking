#!/usr/bin/env python3
"""Paired significance checks for label-token prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_TRUE_SIDE_LABEL_IDS = [4, 5]
METRIC_NAMES = ("accuracy", "macro_f1", "true_side_macro_f1", "checkpoint_selection_score")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}") from exc
    return rows


def align_predictions(
    old_rows: Sequence[dict[str, Any]], new_rows: Sequence[dict[str, Any]]
) -> tuple[list[int], list[int], list[int]]:
    old_by_idx = _rows_by_sample_idx(old_rows, label="old")
    new_by_idx = _rows_by_sample_idx(new_rows, label="new")
    if set(old_by_idx) != set(new_by_idx):
        missing_new = sorted(set(old_by_idx) - set(new_by_idx))[:5]
        missing_old = sorted(set(new_by_idx) - set(old_by_idx))[:5]
        raise ValueError(f"sample_idx mismatch: missing_new={missing_new} missing_old={missing_old}")

    gold: list[int] = []
    old_pred: list[int] = []
    new_pred: list[int] = []
    for sample_idx in sorted(old_by_idx):
        old_row = old_by_idx[sample_idx]
        new_row = new_by_idx[sample_idx]
        old_gold = int(old_row["gold_id"])
        new_gold = int(new_row["gold_id"])
        if old_gold != new_gold:
            raise ValueError(f"gold_id mismatch for sample_idx={sample_idx}: old={old_gold} new={new_gold}")
        gold.append(old_gold)
        old_pred.append(int(old_row["pred_id"]))
        new_pred.append(int(new_row["pred_id"]))
    return gold, old_pred, new_pred


def compute_metrics(
    gold: Sequence[int],
    pred: Sequence[int],
    *,
    label_count: int,
    true_side_label_ids: Sequence[int],
    true_side_weight: float,
    early_stopping_metric: str = "macro_f1_plus_true_side",
    mae_metric_weight: float = 0.3,
) -> dict[str, float]:
    gold_arr = np.asarray(gold, dtype=np.int64)
    pred_arr = np.asarray(pred, dtype=np.int64)
    if gold_arr.shape != pred_arr.shape:
        raise ValueError(f"gold/pred length mismatch: {len(gold_arr)} != {len(pred_arr)}")

    f1_values: list[float] = []
    per_class_f1: dict[int, float] = {}
    eps = 1e-12
    for label_id in range(label_count):
        tp = float(np.sum((pred_arr == label_id) & (gold_arr == label_id)))
        fp = float(np.sum((pred_arr == label_id) & (gold_arr != label_id)))
        fn = float(np.sum((pred_arr != label_id) & (gold_arr == label_id)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        per_class_f1[label_id] = f1
        f1_values.append(f1)

    true_side_values = [per_class_f1[label_id] for label_id in true_side_label_ids]
    macro_f1 = float(np.mean(f1_values))
    true_side_macro_f1 = float(np.mean(true_side_values)) if true_side_values else 0.0
    valid_mask = (pred_arr >= 0) & (gold_arr >= 0)
    if bool(np.any(valid_mask)):
        ordinal_mae = float(np.mean(np.abs(pred_arr[valid_mask].astype(np.float64) - gold_arr[valid_mask].astype(np.float64))))
        ordinal_mae_norm = ordinal_mae / max(label_count - 1, 1)
    else:
        ordinal_mae = float("nan")
        ordinal_mae_norm = float("nan")

    _ = (early_stopping_metric, true_side_weight, mae_metric_weight)
    checkpoint_selection_score = macro_f1

    return {
        "accuracy": float(np.mean(pred_arr == gold_arr)),
        "macro_f1": macro_f1,
        "true_side_macro_f1": true_side_macro_f1,
        "ordinal_mae": ordinal_mae,
        "ordinal_mae_norm": ordinal_mae_norm,
        "checkpoint_selection_score": checkpoint_selection_score,
    }


def mcnemar_exact(old_correct: Iterable[bool], new_correct: Iterable[bool]) -> dict[str, int | float]:
    old_arr = np.asarray(list(old_correct), dtype=bool)
    new_arr = np.asarray(list(new_correct), dtype=bool)
    if old_arr.shape != new_arr.shape:
        raise ValueError(f"old/new correctness length mismatch: {len(old_arr)} != {len(new_arr)}")
    new_correct_old_wrong = int(np.sum(new_arr & ~old_arr))
    old_correct_new_wrong = int(np.sum(old_arr & ~new_arr))
    discordant = new_correct_old_wrong + old_correct_new_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        k = min(new_correct_old_wrong, old_correct_new_wrong)
        tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "new_correct_old_wrong": new_correct_old_wrong,
        "old_correct_new_wrong": old_correct_new_wrong,
        "discordant": discordant,
        "p_value_two_sided": float(p_value),
    }


def bootstrap_delta_ci(
    gold: Sequence[int],
    old_pred: Sequence[int],
    new_pred: Sequence[int],
    *,
    label_count: int,
    true_side_label_ids: Sequence[int],
    true_side_weight: float,
    early_stopping_metric: str,
    mae_metric_weight: float,
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float | list[float] | int]]:
    rng = np.random.default_rng(seed)
    gold_arr = np.asarray(gold, dtype=np.int64)
    old_arr = np.asarray(old_pred, dtype=np.int64)
    new_arr = np.asarray(new_pred, dtype=np.int64)
    n = len(gold_arr)
    deltas = {metric: np.empty(n_resamples, dtype=np.float64) for metric in METRIC_NAMES}
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        old_metrics = compute_metrics(
            gold_arr[idx],
            old_arr[idx],
            label_count=label_count,
            true_side_label_ids=true_side_label_ids,
            true_side_weight=true_side_weight,
            early_stopping_metric=early_stopping_metric,
            mae_metric_weight=mae_metric_weight,
        )
        new_metrics = compute_metrics(
            gold_arr[idx],
            new_arr[idx],
            label_count=label_count,
            true_side_label_ids=true_side_label_ids,
            true_side_weight=true_side_weight,
            early_stopping_metric=early_stopping_metric,
            mae_metric_weight=mae_metric_weight,
        )
        for metric in METRIC_NAMES:
            deltas[metric][i] = new_metrics[metric] - old_metrics[metric]

    return {
        metric: {
            "samples": int(n_resamples),
            "ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
        }
        for metric, values in deltas.items()
    }


def paired_randomization_p_values(
    gold: Sequence[int],
    old_pred: Sequence[int],
    new_pred: Sequence[int],
    *,
    label_count: int,
    true_side_label_ids: Sequence[int],
    true_side_weight: float,
    early_stopping_metric: str,
    mae_metric_weight: float,
    observed_delta: dict[str, float],
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    gold_arr = np.asarray(gold, dtype=np.int64)
    old_arr = np.asarray(old_pred, dtype=np.int64)
    new_arr = np.asarray(new_pred, dtype=np.int64)
    n = len(gold_arr)
    extreme_counts = {metric: 0 for metric in METRIC_NAMES}
    eps = 1e-15
    for _ in range(n_resamples):
        swap = rng.random(n) < 0.5
        perm_old = np.where(swap, new_arr, old_arr)
        perm_new = np.where(swap, old_arr, new_arr)
        old_metrics = compute_metrics(
            gold_arr,
            perm_old,
            label_count=label_count,
            true_side_label_ids=true_side_label_ids,
            true_side_weight=true_side_weight,
            early_stopping_metric=early_stopping_metric,
            mae_metric_weight=mae_metric_weight,
        )
        new_metrics = compute_metrics(
            gold_arr,
            perm_new,
            label_count=label_count,
            true_side_label_ids=true_side_label_ids,
            true_side_weight=true_side_weight,
            early_stopping_metric=early_stopping_metric,
            mae_metric_weight=mae_metric_weight,
        )
        for metric in METRIC_NAMES:
            delta = new_metrics[metric] - old_metrics[metric]
            if abs(delta) + eps >= abs(observed_delta[metric]):
                extreme_counts[metric] += 1

    return {
        metric: {
            "samples": int(n_resamples),
            "p_value_two_sided": float((count + 1) / (n_resamples + 1)),
        }
        for metric, count in extreme_counts.items()
    }


def compare_predictions(
    *,
    name: str,
    old_predictions: Path,
    new_predictions: Path,
    label_count: int,
    true_side_label_ids: Sequence[int],
    true_side_weight: float,
    early_stopping_metric: str,
    mae_metric_weight: float,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    gold, old_pred, new_pred = align_predictions(read_jsonl(old_predictions), read_jsonl(new_predictions))
    old_metrics = compute_metrics(
        gold,
        old_pred,
        label_count=label_count,
        true_side_label_ids=true_side_label_ids,
        true_side_weight=true_side_weight,
        early_stopping_metric=early_stopping_metric,
        mae_metric_weight=mae_metric_weight,
    )
    new_metrics = compute_metrics(
        gold,
        new_pred,
        label_count=label_count,
        true_side_label_ids=true_side_label_ids,
        true_side_weight=true_side_weight,
        early_stopping_metric=early_stopping_metric,
        mae_metric_weight=mae_metric_weight,
    )
    delta = {metric: new_metrics[metric] - old_metrics[metric] for metric in METRIC_NAMES}
    old_correct = np.asarray(old_pred, dtype=np.int64) == np.asarray(gold, dtype=np.int64)
    new_correct = np.asarray(new_pred, dtype=np.int64) == np.asarray(gold, dtype=np.int64)
    return {
        "name": name,
        "n": len(gold),
        "old_predictions": str(old_predictions),
        "new_predictions": str(new_predictions),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "delta": delta,
        "mcnemar": mcnemar_exact(old_correct, new_correct),
        "bootstrap": bootstrap_delta_ci(
            gold,
            old_pred,
            new_pred,
            label_count=label_count,
            true_side_label_ids=true_side_label_ids,
            true_side_weight=true_side_weight,
            early_stopping_metric=early_stopping_metric,
            mae_metric_weight=mae_metric_weight,
            n_resamples=bootstrap_samples,
            seed=seed,
        ),
        "paired_randomization": paired_randomization_p_values(
            gold,
            old_pred,
            new_pred,
            label_count=label_count,
            true_side_label_ids=true_side_label_ids,
            true_side_weight=true_side_weight,
            early_stopping_metric=early_stopping_metric,
            mae_metric_weight=mae_metric_weight,
            observed_delta=delta,
            n_resamples=randomization_samples,
            seed=seed + 100_000,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        nargs=3,
        action="append",
        metavar=("NAME", "OLD_PREDICTIONS", "NEW_PREDICTIONS"),
        required=True,
        help="Paired comparison. Delta direction is NEW - OLD.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label-count", type=int, default=6)
    parser.add_argument("--true-side-label-id", type=int, action="append", default=None)
    parser.add_argument("--true-side-weight", type=float, default=0.5)
    parser.add_argument(
        "--early-stopping-metric",
        default="macro_f1_plus_true_side",
        choices=[
            "macro_f1",
            "true_side_macro_f1",
            "accuracy",
            "macro_f1_plus_true_side",
            "macro_f1_plus_true_side_plus_mae",
        ],
    )
    parser.add_argument("--mae-metric-weight", type=float, default=0.3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--randomization-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    true_side_label_ids = args.true_side_label_id or list(DEFAULT_TRUE_SIDE_LABEL_IDS)
    comparisons = []
    for index, (name, old_path, new_path) in enumerate(args.comparison):
        comparisons.append(
            compare_predictions(
                name=name,
                old_predictions=Path(old_path),
                new_predictions=Path(new_path),
                label_count=args.label_count,
                true_side_label_ids=true_side_label_ids,
                true_side_weight=args.true_side_weight,
                early_stopping_metric=args.early_stopping_metric,
                mae_metric_weight=args.mae_metric_weight,
                bootstrap_samples=args.bootstrap_samples,
                randomization_samples=args.randomization_samples,
                seed=args.seed + index * 10_000,
            )
        )

    payload = {
        "method": {
            "delta_direction": "new - old",
            "accuracy_test": "two-sided exact McNemar on paired correctness disagreements",
            "metric_ci": "paired percentile bootstrap over sample indices",
            "metric_p_value": "two-sided paired approximate randomization by sample-level prediction swaps",
        },
        "settings": {
            "label_count": int(args.label_count),
            "true_side_label_ids": list(true_side_label_ids),
            "true_side_weight": float(args.true_side_weight),
            "early_stopping_metric": str(args.early_stopping_metric),
            "mae_metric_weight": float(args.mae_metric_weight),
            "bootstrap_samples": int(args.bootstrap_samples),
            "randomization_samples": int(args.randomization_samples),
            "seed": int(args.seed),
        },
        "comparisons": comparisons,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_json)
    return 0


def _rows_by_sample_idx(rows: Sequence[dict[str, Any]], *, label: str) -> dict[int, dict[str, Any]]:
    by_idx: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_idx = int(row["sample_idx"])
        if sample_idx in by_idx:
            raise ValueError(f"duplicate sample_idx in {label}: {sample_idx}")
        by_idx[sample_idx] = row
    return by_idx


def _checkpoint_selection_score(
    *,
    metric: str,
    accuracy: float,
    macro_f1: float,
    true_side_macro_f1: float,
    ordinal_mae_norm: float,
    true_side_weight: float,
    mae_metric_weight: float,
) -> float:
    _ = (metric, accuracy, true_side_macro_f1, ordinal_mae_norm, true_side_weight, mae_metric_weight)
    return macro_f1


if __name__ == "__main__":
    raise SystemExit(main())
