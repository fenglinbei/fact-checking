#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_LORA_METRICS = (
    "outputs/runs/b3_selector_trace_full_pipeline/"
    "v0_6c_rule_step_adaptive5_10_best/infer/val/best/6f87c75c6353/api/metrics.json"
)
DEFAULT_FULLFT_METRICS = (
    "outputs/runs/b3_selector_trace_full_pipeline/"
    "v0_6c_rule_step_adaptive5_10_fullft_best/infer/val/best/fe24e176c7c7/api/metrics.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize v0.6c local verifier vs DeepSeek V4 Flash API metrics.")
    p.add_argument("--run-root", default="outputs/runs/b3_selector_trace_full_pipeline/v0_6c_rule_step_adaptive5_10_deepseek_v4_flash")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--no-thinking-checkpoint", default="no_thinking")
    p.add_argument("--thinking-checkpoint", default="thinking_high")
    p.add_argument("--lora-metrics", default=DEFAULT_LORA_METRICS)
    p.add_argument("--fullft-metrics", default=DEFAULT_FULLFT_METRICS)
    p.add_argument("--json-name", default="deepseek_v4_flash_comparison_summary.json")
    p.add_argument("--csv-name", default="deepseek_v4_flash_comparison_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _row_from_metrics(
            name="v0.6c LoRA",
            kind="local",
            mode="label_token_ce_lora",
            metrics_path=Path(args.lora_metrics),
            usage_path=None,
        ),
        _row_from_metrics(
            name="v0.6c FullFT",
            kind="local",
            mode="label_token_ce_fullft",
            metrics_path=Path(args.fullft_metrics),
            usage_path=None,
        ),
    ]

    no_thinking_metrics = _latest_metrics(run_root, split=str(args.split), checkpoint=str(args.no_thinking_checkpoint))
    thinking_metrics = _latest_metrics(run_root, split=str(args.split), checkpoint=str(args.thinking_checkpoint))
    rows.extend(
        [
            _row_from_metrics(
                name="DeepSeek V4 Flash no-thinking",
                kind="api",
                mode="no_thinking",
                metrics_path=no_thinking_metrics,
                usage_path=no_thinking_metrics.parent / "usage_summary.json",
            ),
            _row_from_metrics(
                name="DeepSeek V4 Flash thinking-high",
                kind="api",
                mode="thinking_high",
                metrics_path=thinking_metrics,
                usage_path=thinking_metrics.parent / "usage_summary.json",
            ),
        ]
    )

    json_path = output_dir / str(args.json_name)
    csv_path = output_dir / str(args.csv_name)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote CSV summary: {csv_path}")


def _latest_metrics(run_root: Path, *, split: str, checkpoint: str) -> Path:
    pattern_root = run_root / "infer" / split / checkpoint
    candidates = sorted(
        pattern_root.glob("*/api/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No metrics.json found under {pattern_root}")
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _row_from_metrics(
    *,
    name: str,
    kind: str,
    mode: str,
    metrics_path: Path,
    usage_path: Path | None,
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    usage = _load_json(usage_path) if usage_path is not None and usage_path.exists() else {}
    return {
        "name": name,
        "kind": kind,
        "mode": mode,
        "metrics_path": str(metrics_path),
        "usage_summary_path": str(usage_path) if usage_path else "",
        "num_samples": int(metrics.get("num_samples") or 0),
        "num_expected": int(metrics.get("num_expected") or metrics.get("num_samples") or 0),
        "accuracy": float(metrics.get("accuracy") or 0.0),
        "macro_precision": float(metrics.get("macro_precision") or 0.0),
        "macro_recall": float(metrics.get("macro_recall") or 0.0),
        "macro_f1": float(metrics.get("macro_f1") or 0.0),
        "parse_error_rate": float(metrics.get("parse_error_rate") or 0.0),
        "prompt_tokens_total": _usage_total(usage, "prompt_tokens"),
        "completion_tokens_total": _usage_total(usage, "completion_tokens"),
        "total_tokens_total": _usage_total(usage, "total_tokens"),
        "reasoning_tokens_total": float(usage.get("reasoning_tokens_total") or 0.0),
        "n_success": int(usage.get("n_success") or 0),
        "n_parse_errors": int(usage.get("n_parse_errors") or 0),
        "n_missing_predictions": int(usage.get("n_missing_predictions") or 0),
        "n_error_rows": int(usage.get("n_error_rows") or 0),
    }


def _usage_total(usage: dict[str, Any], key: str) -> float:
    value = usage.get(key)
    if isinstance(value, dict):
        return float(value.get("total") or 0.0)
    return 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "kind",
        "mode",
        "num_samples",
        "num_expected",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "parse_error_rate",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens_total",
        "reasoning_tokens_total",
        "n_success",
        "n_parse_errors",
        "n_missing_predictions",
        "n_error_rows",
        "metrics_path",
        "usage_summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
