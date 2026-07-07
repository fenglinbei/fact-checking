#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


CHUNKING_CASES = ("abc", "sentence", "sentwin1", "semantic07", "report")
POLICIES = ("top5", "budget")
LIAR3_LABELS = ("False", "Half-True", "True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize selector-free atom-union S4 chunking ablation.")
    parser.add_argument("--output-root", default="outputs/sentence_trace_method")
    parser.add_argument("--analysis-dir", default="outputs/analysis/chunking_ablation_atom_union_s4")
    parser.add_argument("--chunking-cases", default=",".join(CHUNKING_CASES))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--lora-suffix", default="_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir)
    chunking_cases = _split_csv(args.chunking_cases)
    policies = _split_csv(args.policies)
    splits = _split_csv(args.splits)

    rows: list[dict[str, Any]] = []
    for chunking_case in chunking_cases:
        for policy in policies:
            for split in splits:
                rows.append(
                    summarize_case(
                        output_root=output_root,
                        chunking_case=chunking_case,
                        policy=policy,
                        split=split,
                        checkpoint=str(args.checkpoint),
                        lora_suffix=str(args.lora_suffix),
                    )
                )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_json = analysis_dir / "summary.json"
    summary_csv = analysis_dir / "summary.csv"
    summary_md = analysis_dir / "summary.md"
    summary_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, rows)
    print(f"Wrote {len(rows)} rows to {summary_json}")
    print(f"CSV: {summary_csv}")
    print(f"Markdown: {summary_md}")
    return 0


def summarize_case(
    *,
    output_root: Path,
    chunking_case: str,
    policy: str,
    split: str,
    checkpoint: str,
    lora_suffix: str,
) -> dict[str, Any]:
    case_name = _case_name(chunking_case, policy)
    build_root = output_root / case_name
    eval_root = output_root / f"{case_name}{lora_suffix}"
    build_report_path = build_root / "build" / "build_report.json"
    metrics_path, predictions_path = _eval_artifact_paths(eval_root, split=split, checkpoint=checkpoint)
    calibration = _load_calibration(output_root, build_root, chunking_case=chunking_case, policy=policy)

    row: dict[str, Any] = {
        "chunking_case": str(chunking_case),
        "policy": str(policy),
        "split": str(split),
        "checkpoint": str(checkpoint),
        "case_name": case_name,
        "lora_case_name": f"{case_name}{lora_suffix}",
        "build_report_path": str(build_report_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "status": "ok",
        "calibrated_budget": calibration.get("selected_budget"),
        "calibration_target_prompt_mean": calibration.get("target_prompt_mean"),
        "calibration_selected_prompt_mean": calibration.get("selected_prompt_mean"),
    }

    if build_report_path.exists():
        build_report = _read_json(build_report_path)
        split_report = dict((build_report.get("splits") or {}).get(split) or {})
        prompt_stats = dict(split_report.get("prompt_token_count") or {})
        evidence_stats = dict(split_report.get("evidence_count") or {})
        row.update(
            {
                "prompt_token_mean": _float_or_none(prompt_stats.get("mean")),
                "prompt_token_p95": _float_or_none(prompt_stats.get("p95")),
                "prompt_token_max": _float_or_none(prompt_stats.get("max")),
                "evidence_count_mean": _float_or_none(evidence_stats.get("mean")),
                "evidence_count_min": _float_or_none(evidence_stats.get("min")),
                "evidence_count_max": _float_or_none(evidence_stats.get("max")),
                "truncation_rate": _float_or_none(split_report.get("prompt_truncation_rate")),
            }
        )
    else:
        row["status"] = "missing_build_report"

    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        row.update(
            {
                "accuracy_6class": _float_or_none(metrics.get("accuracy")),
                "macro_f1_6class": _float_or_none(metrics.get("macro_f1")),
                "macro_precision_6class": _float_or_none(metrics.get("macro_precision")),
                "macro_recall_6class": _float_or_none(metrics.get("macro_recall")),
                "checkpoint_selection_score": _float_or_none(metrics.get("checkpoint_selection_score")),
            }
        )
    else:
        row["status"] = _append_status(row["status"], "missing_metrics")

    if predictions_path.exists():
        predictions = _read_jsonl(predictions_path)
        three_class = compute_three_class_metrics(predictions)
        row.update(
            {
                "accuracy_3class": _float_or_none(three_class.get("accuracy")),
                "macro_f1_3class": _float_or_none(three_class.get("macro_f1")),
                "macro_precision_3class": _float_or_none(three_class.get("macro_precision")),
                "macro_recall_3class": _float_or_none(three_class.get("macro_recall")),
                "n_predictions": len(predictions),
            }
        )
    else:
        row["status"] = _append_status(row["status"], "missing_predictions")

    return row


def compute_three_class_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    gold: list[str] = []
    pred: list[str] = []
    for row in rows:
        try:
            gold.append(_collapse_liar_label(str(row.get("gold_label") or "")))
            pred.append(_collapse_liar_label(str(row.get("pred_label") or "")))
        except ValueError:
            continue
    total = len(gold)
    if total == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {
                label: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                for label in LIAR3_LABELS
            },
        }

    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in LIAR3_LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(sum(1 for g, p in zip(gold, pred) if g == p) / total),
        "macro_precision": float(sum(precisions) / len(precisions)),
        "macro_recall": float(sum(recalls) / len(recalls)),
        "macro_f1": float(sum(f1s) / len(f1s)),
        "per_class": per_class,
    }


def _case_name(chunking_case: str, policy: str) -> str:
    policy_suffix = "top5" if str(policy) == "top5" else "budget_promptmatched"
    return f"liar_raw__ministral3_8b__chunk_{chunking_case}_s4_union_{policy_suffix}_plain"


def _eval_artifact_paths(eval_root: Path, *, split: str, checkpoint: str) -> tuple[Path, Path]:
    nested = eval_root / "eval" / split / checkpoint / "label_token"
    legacy = eval_root / "eval" / split / checkpoint
    nested_metrics = nested / "metrics.json"
    legacy_metrics = legacy / "metrics.json"
    metrics = nested_metrics if nested_metrics.exists() or not legacy_metrics.exists() else legacy_metrics
    predictions_dir = nested if metrics == nested_metrics else legacy
    return metrics, predictions_dir / f"{split}_predictions.jsonl"


def _load_calibration(
    output_root: Path,
    build_root: Path,
    *,
    chunking_case: str,
    policy: str,
) -> dict[str, Any]:
    if str(policy) != "budget":
        return {}
    candidates = [
        build_root / "build" / "prompt_budget_calibration.json",
        output_root / "_calibration" / f"chunk_{chunking_case}_s4_union_budget_promptmatched.json",
    ]
    for path in candidates:
        if path.exists():
            return _read_json(path)
    return {}


def _collapse_liar_label(label: str) -> str:
    normalized = " ".join(str(label).strip().lower().replace("_", "-").split())
    if normalized in {"pants-fire", "false", "barely-true"}:
        return "False"
    if normalized in {"half-true", "half true", "half"}:
        return "Half-True"
    if normalized in {"mostly-true", "mostly true", "true"}:
        return "True"
    raise ValueError(f"unknown LIAR label: {label!r}")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").replace(" ", ",").split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_status(current: str, item: str) -> str:
    if current == "ok":
        return item
    return f"{current};{item}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "chunking_case",
        "policy",
        "split",
        "status",
        "accuracy_6class",
        "macro_f1_6class",
        "accuracy_3class",
        "macro_f1_3class",
        "prompt_token_mean",
        "prompt_token_p95",
        "evidence_count_mean",
        "truncation_rate",
        "calibrated_budget",
        "case_name",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Selector-Free Atom-Union S4 Chunking Ablation",
        "",
        "| chunking | policy | split | status | acc6 | macroF1_6 | acc3 | macroF1_3 | prompt_mean | ev_mean | B |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {chunking_case} | {policy} | {split} | {status} | {acc6} | {f16} | {acc3} | {f13} | {prompt} | {ev} | {budget} |".format(
                chunking_case=row.get("chunking_case", ""),
                policy=row.get("policy", ""),
                split=row.get("split", ""),
                status=row.get("status", ""),
                acc6=_fmt(row.get("accuracy_6class")),
                f16=_fmt(row.get("macro_f1_6class")),
                acc3=_fmt(row.get("accuracy_3class")),
                f13=_fmt(row.get("macro_f1_3class")),
                prompt=_fmt(row.get("prompt_token_mean")),
                ev=_fmt(row.get("evidence_count_mean")),
                budget="" if row.get("calibrated_budget") is None else str(row.get("calibrated_budget")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
