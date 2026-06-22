#!/usr/bin/env python3
"""Diagnose a validation/test macro-F1 gap from label-token prediction files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prediction rows loaded from {path}")
    return rows


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    gold = np.asarray([int(row["gold_id"]) for row in rows], dtype=np.int64)
    pred = np.asarray([int(row["pred_id"]) for row in rows], dtype=np.int64)
    return gold, pred


def confusion_matrix(gold: np.ndarray, pred: np.ndarray, n_labels: int) -> np.ndarray:
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    for gold_id, pred_id in zip(gold.tolist(), pred.tolist()):
        if 0 <= gold_id < n_labels and 0 <= pred_id < n_labels:
            matrix[gold_id, pred_id] += 1
    return matrix


def per_class_from_matrix(matrix: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_counts = matrix.sum(axis=0)
    gold_counts = matrix.sum(axis=1)
    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        pred_count = int(pred_counts[idx])
        gold_count = int(gold_counts[idx])
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label_id": idx,
                "label": label,
                "gold_count": gold_count,
                "pred_count": pred_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_labels: int) -> float:
    matrix = confusion_matrix(gold, pred, n_labels)
    per_class = per_class_from_matrix(matrix, [str(i) for i in range(n_labels)])
    return float(np.mean([row["f1"] for row in per_class]))


def ordinal_summary(gold: np.ndarray, pred: np.ndarray, n_labels: int) -> dict[str, Any]:
    signed = pred - gold
    dist = np.abs(signed)
    counts = Counter(int(x) for x in dist.tolist())
    return {
        "mean_abs_distance": float(np.mean(dist)),
        "median_abs_distance": float(np.median(dist)),
        "p90_abs_distance": float(np.quantile(dist, 0.90)),
        "mean_signed_distance": float(np.mean(signed)),
        "distance_counts": {str(i): int(counts.get(i, 0)) for i in range(n_labels)},
        "extreme_error_rate_abs_ge_3": float(np.mean(dist >= 3)),
    }


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"low": float(lo), "high": float(hi)}


def bootstrap_independent_gap(
    *,
    val_gold: np.ndarray,
    val_pred: np.ndarray,
    test_gold: np.ndarray,
    test_pred: np.ndarray,
    n_labels: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    val_n = len(val_gold)
    test_n = len(test_gold)
    val_scores = np.empty(n_bootstrap, dtype=np.float64)
    test_scores = np.empty(n_bootstrap, dtype=np.float64)
    gaps = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        val_idx = rng.integers(0, val_n, size=val_n)
        test_idx = rng.integers(0, test_n, size=test_n)
        val_score = macro_f1(val_gold[val_idx], val_pred[val_idx], n_labels)
        test_score = macro_f1(test_gold[test_idx], test_pred[test_idx], n_labels)
        val_scores[idx] = val_score
        test_scores[idx] = test_score
        gaps[idx] = val_score - test_score
    return {
        "val_macro_f1_ci": percentile_interval(val_scores),
        "test_macro_f1_ci": percentile_interval(test_scores),
        "gap_ci": percentile_interval(gaps),
        "gap_probability_gt_0": float(np.mean(gaps > 0.0)),
        "gap_samples": gaps,
        "val_samples": val_scores,
        "test_samples": test_scores,
    }


def bootstrap_label_matched_validation(
    *,
    val_gold: np.ndarray,
    val_pred: np.ndarray,
    test_gold: np.ndarray,
    n_labels: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    val_indices_by_label = [np.flatnonzero(val_gold == label_id) for label_id in range(n_labels)]
    test_counts = np.bincount(test_gold, minlength=n_labels)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sampled_chunks: list[np.ndarray] = []
        for label_id, count in enumerate(test_counts.tolist()):
            source = val_indices_by_label[label_id]
            if len(source) == 0 and count:
                raise ValueError(f"Validation has no samples for label_id={label_id}")
            if count:
                sampled_chunks.append(rng.choice(source, size=count, replace=True))
        sampled_idx = np.concatenate(sampled_chunks)
        samples[idx] = macro_f1(val_gold[sampled_idx], val_pred[sampled_idx], n_labels)
    return samples


def with_prefix(prefix: str | Path, suffix: str) -> Path:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    return prefix_path.with_name(prefix_path.name + suffix)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *[int(x) for x in matrix[idx].tolist()]])


def write_bootstrap_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    n_rows = len(next(iter(columns.values())))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for idx in range(n_rows):
            writer.writerow([float(columns[name][idx]) for name in names])


def render_markdown(summary: dict[str, Any], per_class_rows: list[dict[str, Any]], labels: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Val/Test Gap Bootstrap Diagnostic")
    lines.append("")
    lines.append(f"- val predictions: `{summary['inputs']['val_predictions']}`")
    lines.append(f"- test predictions: `{summary['inputs']['test_predictions']}`")
    lines.append(f"- bootstrap samples: {summary['bootstrap']['n_bootstrap']}")
    lines.append(f"- seed: {summary['bootstrap']['seed']}")
    lines.append("")
    lines.append("## Macro-F1")
    lines.append("")
    lines.append("| split | macro_f1 | 95% bootstrap CI | n |")
    lines.append("|---|---:|---:|---:|")
    for split in ("val", "test"):
        ci = summary["bootstrap"][f"{split}_macro_f1_ci"]
        lines.append(
            f"| {split} | {summary['macro_f1'][split]:.6f} | "
            f"[{ci['low']:.6f}, {ci['high']:.6f}] | {summary['n'][split]} |"
        )
    gap_ci = summary["bootstrap"]["gap_ci"]
    lines.append("")
    lines.append(
        f"Observed val-test gap: {summary['macro_f1']['gap_val_minus_test']:.6f}; "
        f"independent bootstrap 95% CI [{gap_ci['low']:.6f}, {gap_ci['high']:.6f}]."
    )
    lines.append("")
    lines.append("## Per-Class")
    lines.append("")
    lines.append("| label | val P | val R | val F1 | val pred | test P | test R | test F1 | test pred | F1 gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in per_class_rows:
        lines.append(
            f"| {row['label']} | {row['val_precision']:.4f} | {row['val_recall']:.4f} | "
            f"{row['val_f1']:.4f} | {row['val_pred_count']} | {row['test_precision']:.4f} | "
            f"{row['test_recall']:.4f} | {row['test_f1']:.4f} | {row['test_pred_count']} | "
            f"{row['f1_gap_val_minus_test']:.4f} |"
        )
    lines.append("")
    lines.append("## Ordinal Error")
    lines.append("")
    lines.append("| split | mean abs | median abs | p90 abs | mean signed | abs>=3 rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ("val", "test"):
        item = summary["ordinal_error"][split]
        lines.append(
            f"| {split} | {item['mean_abs_distance']:.4f} | {item['median_abs_distance']:.4f} | "
            f"{item['p90_abs_distance']:.4f} | {item['mean_signed_distance']:.4f} | "
            f"{item['extreme_error_rate_abs_ge_3']:.4f} |"
        )
    lines.append("")
    lines.append("## Label-Matched Validation Bootstrap")
    lines.append("")
    lm = summary["label_matched_validation_bootstrap"]
    lines.append(
        f"Validation resampled to test label counts has macro-F1 "
        f"mean={lm['mean']:.6f}, 95% CI=[{lm['ci']['low']:.6f}, {lm['ci']['high']:.6f}]."
    )
    lines.append(
        f"Observed test macro-F1 percentile in that distribution: {lm['test_macro_f1_percentile']:.4f}."
    )
    lines.append("")
    lines.append("## Confusion Matrices")
    for split in ("val", "test"):
        lines.append("")
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| gold\\pred | " + " | ".join(labels) + " |")
        lines.append("|---" + "|---:" * len(labels) + "|")
        matrix = summary["confusion_matrix"][split]
        for label, row in zip(labels, matrix):
            lines.append("| " + label + " | " + " | ".join(str(x) for x in row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    labels = list(args.labels)
    n_labels = len(labels)
    val_rows = read_jsonl(args.val_predictions)
    test_rows = read_jsonl(args.test_predictions)
    val_gold, val_pred = arrays(val_rows)
    test_gold, test_pred = arrays(test_rows)

    val_matrix = confusion_matrix(val_gold, val_pred, n_labels)
    test_matrix = confusion_matrix(test_gold, test_pred, n_labels)
    val_per_class = per_class_from_matrix(val_matrix, labels)
    test_per_class = per_class_from_matrix(test_matrix, labels)

    per_class_rows: list[dict[str, Any]] = []
    for val_row, test_row in zip(val_per_class, test_per_class):
        label = str(val_row["label"])
        precision_gap = float(val_row["precision"]) - float(test_row["precision"])
        recall_gap = float(val_row["recall"]) - float(test_row["recall"])
        f1_gap = float(val_row["f1"]) - float(test_row["f1"])
        per_class_rows.append(
            {
                "label_id": int(val_row["label_id"]),
                "label": label,
                "val_gold_count": int(val_row["gold_count"]),
                "val_pred_count": int(val_row["pred_count"]),
                "val_precision": float(val_row["precision"]),
                "val_recall": float(val_row["recall"]),
                "val_f1": float(val_row["f1"]),
                "test_gold_count": int(test_row["gold_count"]),
                "test_pred_count": int(test_row["pred_count"]),
                "test_precision": float(test_row["precision"]),
                "test_recall": float(test_row["recall"]),
                "test_f1": float(test_row["f1"]),
                "precision_gap_val_minus_test": precision_gap,
                "recall_gap_val_minus_test": recall_gap,
                "f1_gap_val_minus_test": f1_gap,
                "macro_gap_contribution": f1_gap / n_labels,
            }
        )

    rng = np.random.default_rng(int(args.seed))
    independent_bootstrap = bootstrap_independent_gap(
        val_gold=val_gold,
        val_pred=val_pred,
        test_gold=test_gold,
        test_pred=test_pred,
        n_labels=n_labels,
        n_bootstrap=int(args.bootstrap_samples),
        rng=rng,
    )
    label_matched_samples = bootstrap_label_matched_validation(
        val_gold=val_gold,
        val_pred=val_pred,
        test_gold=test_gold,
        n_labels=n_labels,
        n_bootstrap=int(args.bootstrap_samples),
        rng=rng,
    )

    val_macro = macro_f1(val_gold, val_pred, n_labels)
    test_macro = macro_f1(test_gold, test_pred, n_labels)
    label_matched_ci = percentile_interval(label_matched_samples)
    summary = {
        "inputs": {
            "val_predictions": str(args.val_predictions),
            "test_predictions": str(args.test_predictions),
        },
        "labels": labels,
        "n": {"val": int(len(val_gold)), "test": int(len(test_gold))},
        "macro_f1": {
            "val": val_macro,
            "test": test_macro,
            "gap_val_minus_test": val_macro - test_macro,
        },
        "per_class": per_class_rows,
        "confusion_matrix": {
            "val": val_matrix.astype(int).tolist(),
            "test": test_matrix.astype(int).tolist(),
        },
        "ordinal_error": {
            "val": ordinal_summary(val_gold, val_pred, n_labels),
            "test": ordinal_summary(test_gold, test_pred, n_labels),
        },
        "bootstrap": {
            "n_bootstrap": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "val_macro_f1_ci": independent_bootstrap["val_macro_f1_ci"],
            "test_macro_f1_ci": independent_bootstrap["test_macro_f1_ci"],
            "gap_ci": independent_bootstrap["gap_ci"],
            "gap_probability_gt_0": independent_bootstrap["gap_probability_gt_0"],
        },
        "label_matched_validation_bootstrap": {
            "test_label_counts": {
                labels[idx]: int(count)
                for idx, count in enumerate(np.bincount(test_gold, minlength=n_labels).tolist())
            },
            "mean": float(np.mean(label_matched_samples)),
            "ci": label_matched_ci,
            "test_macro_f1": test_macro,
            "test_macro_f1_percentile": float(np.mean(label_matched_samples <= test_macro)),
            "gap_mean_val_matched_minus_test": float(np.mean(label_matched_samples - test_macro)),
            "gap_ci_val_matched_minus_test": percentile_interval(label_matched_samples - test_macro),
        },
    }

    prefix = Path(args.output_prefix)
    summary_path = with_prefix(prefix, "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        with_prefix(prefix, "_per_class.csv"),
        per_class_rows,
        [
            "label_id",
            "label",
            "val_gold_count",
            "val_pred_count",
            "val_precision",
            "val_recall",
            "val_f1",
            "test_gold_count",
            "test_pred_count",
            "test_precision",
            "test_recall",
            "test_f1",
            "precision_gap_val_minus_test",
            "recall_gap_val_minus_test",
            "f1_gap_val_minus_test",
            "macro_gap_contribution",
        ],
    )
    write_matrix_csv(with_prefix(prefix, "_val_confusion_matrix.csv"), val_matrix, labels)
    write_matrix_csv(with_prefix(prefix, "_test_confusion_matrix.csv"), test_matrix, labels)
    write_bootstrap_csv(
        with_prefix(prefix, "_bootstrap_macro_f1.csv"),
        {
            "val_macro_f1": independent_bootstrap["val_samples"],
            "test_macro_f1": independent_bootstrap["test_samples"],
            "gap_val_minus_test": independent_bootstrap["gap_samples"],
            "label_matched_val_macro_f1": label_matched_samples,
        },
    )
    with_prefix(prefix, ".md").write_text(render_markdown(summary, per_class_rows, labels), encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"macro_gap={summary['macro_f1']['gap_val_minus_test']:.6f}")
    print(
        "gap_ci=[{low:.6f}, {high:.6f}]".format(
            low=summary["bootstrap"]["gap_ci"]["low"],
            high=summary["bootstrap"]["gap_ci"]["high"],
        )
    )


if __name__ == "__main__":
    main()
