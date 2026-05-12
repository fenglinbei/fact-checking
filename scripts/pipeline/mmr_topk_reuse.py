#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.config import load_yaml, save_yaml
from fact_checking.data.io import load_jsonl


SUMMARY_COLUMNS = [
    "top_k",
    "num_samples",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "parse_error_rate",
    "avg_evidence_count",
    "avg_evidence_count_before",
    "truncation_rate",
    "avg_prompt_tokens",
    "max_prompt_tokens",
    "build_run_dir",
    "metrics_path",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_base_train_dir(base_run_dir: Path) -> Path:
    if (base_run_dir / "best").exists():
        return base_run_dir
    train_dir = base_run_dir / "train"
    if train_dir.exists():
        return train_dir
    raise FileNotFoundError(
        f"Cannot resolve base train dir from {base_run_dir}. "
        "Pass BASE_RUN_DIR as either a pipeline run dir or its train dir."
    )


def _resolve_base_config(base_run_dir: Path, base_train_dir: Path) -> Path:
    candidates = [
        base_run_dir / "configs" / "train.resolved.yaml",
        base_train_dir / "config.resolved.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find base train config. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _build_outputs(topk_run_dir: Path) -> dict[str, str]:
    manifest_path = topk_run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    outputs = manifest.get("phases", {}).get("build", {}).get("outputs", {})
    required = {"train", "val", "test"}
    missing = sorted(required - set(outputs))
    if missing:
        raise KeyError(f"{manifest_path} is missing build outputs for: {missing}")
    return {split: str(outputs[split]) for split in sorted(required)}


def prepare_config(args: argparse.Namespace) -> None:
    base_run_dir = Path(args.base_run_dir).resolve()
    topk_run_dir = Path(args.topk_run_dir).resolve()
    base_train_dir = _resolve_base_train_dir(base_run_dir)
    base_config_path = _resolve_base_config(base_run_dir, base_train_dir)
    build_outputs = _build_outputs(topk_run_dir)

    cfg = load_yaml(base_config_path)
    cfg["output_dir"] = str(base_train_dir)
    cfg["data"] = {
        "train_candidates": build_outputs["train"],
        "val_candidates": build_outputs["val"],
        "test_candidates": build_outputs["test"],
    }

    output_config = Path(args.output_config).resolve()
    save_yaml(cfg, output_config)
    print(f"[mmr_topk_reuse] wrote config: {output_config}", flush=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evidence_stats(test_candidates: Path) -> dict[str, str]:
    rows = load_jsonl(test_candidates)
    evidence_counts = [float(row.get("evidence_count", 0) or 0) for row in rows]
    before_counts = [
        float(row.get("evidence_count_before", row.get("evidence_count", 0)) or 0)
        for row in rows
    ]
    prompt_tokens = [float(row.get("prompt_token_count", 0) or 0) for row in rows]
    truncated = [1.0 if bool(row.get("was_truncated", False)) else 0.0 for row in rows]
    return {
        "avg_evidence_count": f"{_mean(evidence_counts):.6f}",
        "avg_evidence_count_before": f"{_mean(before_counts):.6f}",
        "truncation_rate": f"{_mean(truncated):.6f}",
        "avg_prompt_tokens": f"{_mean(prompt_tokens):.6f}",
        "max_prompt_tokens": f"{max(prompt_tokens) if prompt_tokens else 0.0:.0f}",
    }


def _format_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key, 0.0)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.12g}"


def summarize(args: argparse.Namespace) -> None:
    topk_run_dir = Path(args.topk_run_dir).resolve()
    manifest = _read_json(topk_run_dir / "manifest.json")
    build_outputs = _build_outputs(topk_run_dir)
    infer_phase = manifest.get("phases", {}).get("infer", {})
    artifacts = infer_phase.get("artifacts", {}) or {}
    metrics_path = Path(str(artifacts.get("metrics_path", "")))
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot find infer metrics path from manifest: {metrics_path}")

    metrics = _read_json(metrics_path)
    row = {
        "top_k": str(int(args.top_k)),
        "num_samples": str(int(metrics.get("num_samples", 0))),
        "accuracy": _format_metric(metrics, "accuracy"),
        "macro_precision": _format_metric(metrics, "macro_precision"),
        "macro_recall": _format_metric(metrics, "macro_recall"),
        "macro_f1": _format_metric(metrics, "macro_f1"),
        "parse_error_rate": _format_metric(metrics, "parse_error_rate"),
        "build_run_dir": str(topk_run_dir),
        "metrics_path": str(metrics_path),
        **_evidence_stats(Path(build_outputs["test"])),
    }

    summary_path = Path(args.summary_csv).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_topk: dict[str, dict[str, str]] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            for old_row in csv.DictReader(f):
                rows_by_topk[str(old_row.get("top_k", ""))] = {
                    key: str(old_row.get(key, "")) for key in SUMMARY_COLUMNS
                }
    rows_by_topk[row["top_k"]] = row

    rows = sorted(rows_by_topk.values(), key=lambda item: int(item["top_k"]))
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[mmr_topk_reuse] updated summary: {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Helpers for top_k sweep with reused inference checkpoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-config", help="Create a train config pointing to a top_k build.")
    prepare.add_argument("--base-run-dir", required=True)
    prepare.add_argument("--topk-run-dir", required=True)
    prepare.add_argument("--output-config", required=True)
    prepare.set_defaults(func=prepare_config)

    summary = subparsers.add_parser("summarize", help="Append or update one top_k row in the summary CSV.")
    summary.add_argument("--top-k", type=int, required=True)
    summary.add_argument("--topk-run-dir", required=True)
    summary.add_argument("--summary-csv", required=True)
    summary.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
