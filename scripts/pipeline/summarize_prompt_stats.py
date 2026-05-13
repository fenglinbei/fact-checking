#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sft.prompting.stats import flatten_prompt_statistics


METADATA_COLUMNS = ["run_name", "phase", "run_dir", "prompt_stats_path"]
PREFERRED_OVERRIDE_COLUMNS = ["build.retrieval.top_k", "build.retrieval.mmr_lambda"]
SELECTED_MARKDOWN_METRICS = [
    "train.prompt_token_count.mean",
    "train.prompt_token_count.p95",
    "train.prompt_token_count.max",
    "train.evidence_truncation.mean_evidence_count",
    "train.evidence_truncation.truncation_rate",
    "val.prompt_token_count.mean",
    "val.prompt_token_count.p95",
    "val.prompt_token_count.max",
    "val.evidence_truncation.mean_evidence_count",
    "val.evidence_truncation.truncation_rate",
]
LENGTH_STATS = {"min", "p50", "p90", "p95", "p99", "max", "mean"}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _coerce_scalar(raw: str) -> int | float | bool | str:
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value) or re.fullmatch(
        r"[+-]?\d+[eE][+-]?\d+",
        value,
    ):
        return float(value)
    return value


def _parse_run_overrides(run_name: str) -> dict[str, int | float | bool | str]:
    slug = run_name.rsplit("__", 1)[0]
    overrides: dict[str, int | float | bool | str] = {}
    for item in slug.split(","):
        if "-" not in item:
            continue
        key, value = item.rsplit("-", 1)
        if key:
            overrides[key] = _coerce_scalar(value)
    return overrides


def _sort_metric_key(metric: str) -> tuple[int, str, int, str, str]:
    parts = metric.split(".")
    split = parts[0] if parts else ""
    section = parts[1] if len(parts) > 1 else ""
    section_order = {
        "max_length": 0,
        "prompt_token_count": 1,
        "target_token_count": 2,
        "evidence_truncation": 3,
        "snapshot_category_counts": 4,
    }.get(section, 99)
    return (SPLIT_ORDER.get(split, 99), split, section_order, section, metric)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, complex):
        if math.isfinite(float(value)):
            return float(value)
    return None


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _prompt_stats_paths(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.rglob("prompt_stats.json")
        if path.parent.name == "prompt_stats"
    )


def collect_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    paths = _prompt_stats_paths(run_root)
    if not paths:
        raise FileNotFoundError(f"No prompt_stats.json files found under {run_root}")

    rows: list[dict[str, Any]] = []
    override_keys: set[str] = set()
    metric_keys: set[str] = set()
    for path in paths:
        phase_dir = path.parent.parent
        run_dir = phase_dir.parent
        stats = _read_json(path)
        overrides = _parse_run_overrides(run_dir.name)
        flat = flatten_prompt_statistics(stats, prefix="", separator=".")
        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "phase": phase_dir.name,
            "run_dir": str(run_dir),
            "prompt_stats_path": str(path),
            **overrides,
            **flat,
        }
        rows.append(row)
        override_keys.update(overrides.keys())
        metric_keys.update(flat.keys())

    ordered_override_keys = [
        key for key in PREFERRED_OVERRIDE_COLUMNS if key in override_keys
    ] + sorted(key for key in override_keys if key not in PREFERRED_OVERRIDE_COLUMNS)
    ordered_metric_keys = sorted(metric_keys, key=_sort_metric_key)
    return rows, ordered_override_keys, ordered_metric_keys


def _sort_rows(rows: list[dict[str, Any]], x_key: str) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, float | str, str]:
        x_value = row.get(x_key)
        numeric = _as_float(x_value)
        if numeric is not None:
            return (0, numeric, str(row.get("run_name", "")))
        return (1, str(x_value or row.get("run_name", "")), str(row.get("run_name", "")))

    return sorted(rows, key=sort_key)


def write_wide_csv(
    rows: list[dict[str, Any]],
    columns: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in columns})


def write_long_csv(
    rows: list[dict[str, Any]],
    metadata_columns: list[str],
    override_columns: list[str],
    metric_columns: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = metadata_columns + override_columns + ["split", "metric", "value"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            base = {key: _format_cell(row.get(key)) for key in metadata_columns + override_columns}
            for metric in metric_columns:
                split, _, metric_name = metric.partition(".")
                writer.writerow(
                    {
                        **base,
                        "split": split,
                        "metric": metric_name,
                        "value": _format_cell(row.get(metric)),
                    }
                )


def _markdown_table(headers: list[str], table_rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    rows: list[dict[str, Any]],
    override_columns: list[str],
    metric_columns: list[str],
    output_path: Path,
    *,
    wide_csv: Path,
    long_csv: Path,
    plot_path: Path,
) -> None:
    selected_metrics = [key for key in SELECTED_MARKDOWN_METRICS if key in metric_columns]
    selected_columns = ["run_name"] + [
        key for key in override_columns if key in PREFERRED_OVERRIDE_COLUMNS
    ] + selected_metrics
    table_rows = [
        [_format_cell(row.get(column)) for column in selected_columns]
        for row in rows
    ]
    text = "\n".join(
        [
            "# Prompt Stats Summary",
            "",
            f"- Wide table: `{wide_csv.name}`",
            f"- Long table: `{long_csv.name}`",
            f"- Line chart: `{plot_path.name}`",
            "",
            _markdown_table(selected_columns, table_rows),
            "",
        ]
    )
    output_path.write_text(text, encoding="utf-8")


def _metric_group(metric: str) -> str:
    parts = metric.split(".")
    if len(parts) < 2:
        return "other"
    split, section = parts[0], parts[1]
    stat = parts[-1]
    if section == "max_length":
        return f"{split}: config"
    if section in {"prompt_token_count", "target_token_count"}:
        if stat in LENGTH_STATS:
            return f"{split}: {section}"
        if stat.endswith("rate"):
            return f"{split}: rates"
        return f"{split}: counts"
    if section == "evidence_truncation":
        if stat.endswith("rate"):
            return f"{split}: rates"
        if stat.endswith("evidence_count"):
            return f"{split}: evidence_count"
        return f"{split}: counts"
    if section == "snapshot_category_counts":
        if stat.endswith("rate"):
            return f"{split}: rates"
        return f"{split}: counts"
    return f"{split}: other"


def _group_sort_key(group: str) -> tuple[int, int, str]:
    split, _, suffix = group.partition(": ")
    suffix_order = {
        "config": 0,
        "prompt_token_count": 1,
        "target_token_count": 2,
        "evidence_count": 3,
        "counts": 4,
        "rates": 5,
        "other": 6,
    }.get(suffix, 99)
    return (SPLIT_ORDER.get(split, 99), suffix_order, group)


def _plot_label(metric: str) -> str:
    parts = metric.split(".")
    if len(parts) <= 1:
        return metric
    return ".".join(parts[1:])


def write_line_plot(
    rows: list[dict[str, Any]],
    metric_columns: list[str],
    output_path: Path,
    *,
    x_key: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric_metrics = [
        metric
        for metric in metric_columns
        if any(_as_float(row.get(metric)) is not None for row in rows)
    ]
    groups: dict[str, list[str]] = {}
    for metric in numeric_metrics:
        groups.setdefault(_metric_group(metric), []).append(metric)
    ordered_groups = sorted(groups.items(), key=lambda item: _group_sort_key(item[0]))

    numeric_x = [_as_float(row.get(x_key)) for row in rows]
    if all(value is not None for value in numeric_x):
        x_values = [float(value) for value in numeric_x if value is not None]
        x_labels = [_format_cell(row.get(x_key)) for row in rows]
        x_axis_label = x_key
    else:
        x_values = list(range(len(rows)))
        x_labels = [str(row.get(x_key) or row.get("run_name", "")) for row in rows]
        x_axis_label = x_key if any(row.get(x_key) is not None for row in rows) else "run"

    ncols = 2
    nrows = max(1, math.ceil(len(ordered_groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 3.8 * nrows), squeeze=False)
    flat_axes = [axis for row_axes in axes for axis in row_axes]
    for axis, (group_name, metrics) in zip(flat_axes, ordered_groups):
        for metric in sorted(metrics, key=_sort_metric_key):
            y_values: list[float] = []
            x_for_metric: list[float] = []
            for x_value, row in zip(x_values, rows):
                y_value = _as_float(row.get(metric))
                if y_value is None:
                    continue
                x_for_metric.append(x_value)
                y_values.append(y_value)
            if y_values:
                axis.plot(x_for_metric, y_values, marker="o", linewidth=1.6, label=_plot_label(metric))
        axis.set_title(group_name)
        axis.set_xlabel(x_axis_label)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
        if len(x_values) <= 20:
            axis.set_xticks(x_values)
            axis.set_xticklabels(x_labels, rotation=30, ha="right")

    for axis in flat_axes[len(ordered_groups):]:
        axis.axis("off")

    fig.suptitle("Prompt stats across runs", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prompt_stats.json files under a sweep run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_root", help="Run root containing child run directories.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated tables and plot.")
    parser.add_argument("--x-key", default="build.retrieval.top_k", help="Column used as the line-chart x axis.")
    parser.add_argument("--wide-csv", default="prompt_stats_summary.csv")
    parser.add_argument("--long-csv", default="prompt_stats_summary_long.csv")
    parser.add_argument("--markdown", default="prompt_stats_summary.md")
    parser.add_argument("--plot", default="prompt_stats_line_chart.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root

    rows, override_columns, metric_columns = collect_rows(run_root)
    rows = _sort_rows(rows, args.x_key)
    wide_columns = METADATA_COLUMNS + override_columns + metric_columns

    wide_csv = output_dir / args.wide_csv
    long_csv = output_dir / args.long_csv
    markdown = output_dir / args.markdown
    plot_path = output_dir / args.plot

    write_wide_csv(rows, wide_columns, wide_csv)
    write_long_csv(rows, METADATA_COLUMNS, override_columns, metric_columns, long_csv)
    write_line_plot(rows, metric_columns, plot_path, x_key=args.x_key)
    write_markdown_summary(
        rows,
        override_columns,
        metric_columns,
        markdown,
        wide_csv=wide_csv,
        long_csv=long_csv,
        plot_path=plot_path,
    )

    print(f"[prompt_stats] rows={len(rows)}")
    print(f"[prompt_stats] wide_csv={wide_csv}")
    print(f"[prompt_stats] long_csv={long_csv}")
    print(f"[prompt_stats] markdown={markdown}")
    print(f"[prompt_stats] plot={plot_path}")


if __name__ == "__main__":
    main()
