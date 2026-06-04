#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

METADATA_COLUMNS = [
    "source_root",
    "run_name",
    "split",
    "checkpoint",
    "infer_id",
    "duplicate_rank",
    "duplicate_count",
    "run_dir",
    "metrics_path",
    "modified_time",
]
PROMPT_METADATA_COLUMNS = ["prompt_stats_found", "prompt_stats_path"]
PREFERRED_OVERRIDE_COLUMNS = [
    "build.retrieval.top_k",
    "build.retrieval.mmr_lambda",
    "build.retrieval.selection_method",
]
SENSITIVITY_COLUMNS = [
    "sensitivity.theta_s",
    "sensitivity.theta_r",
    "sensitivity.lambda_low",
    "sensitivity.gating_mode",
    "sensitivity.epsilon",
]
PREFERRED_METRIC_COLUMNS = [
    "num_samples",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "parse_error_rate",
]
PLOT_METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "parse_error_rate"]
PROMPT_MEAN_COLUMN = "prompt.train.prompt_token_count.mean"
MAIN_PLOT_COLUMNS = PLOT_METRIC_COLUMNS + [PROMPT_MEAN_COLUMN]
PROMPT_STAT_COLUMNS = [
    "prompt.train.prompt_token_count.mean",
    "prompt.train.prompt_token_count.p50",
    "prompt.train.prompt_token_count.p90",
    "prompt.train.prompt_token_count.p95",
    "prompt.train.prompt_token_count.p99",
    "prompt.train.prompt_token_count.max",
    "prompt.train.prompt_token_count.overflow_rate",
    "prompt.train.evidence_truncation.truncation_rate",
    "prompt.train.evidence_truncation.mean_evidence_count",
    "prompt.val.prompt_token_count.mean",
    "prompt.val.prompt_token_count.p50",
    "prompt.val.prompt_token_count.p90",
    "prompt.val.prompt_token_count.p95",
    "prompt.val.prompt_token_count.p99",
    "prompt.val.prompt_token_count.max",
    "prompt.val.prompt_token_count.overflow_rate",
    "prompt.val.evidence_truncation.truncation_rate",
    "prompt.val.evidence_truncation.mean_evidence_count",
]
CLOSE_METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
CLOSE_ABS_MARGIN = 0.006
LABEL_ORDER = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
PER_CLASS_METRICS = ["precision", "recall", "f1"]
SOURCE_LABELS = {
    "b3_mmr_topk_sweep_1024": "b3 top_k sweep",
    "heuristic_lambda_mmr": "heuristic lambda",
    "heuristic_lambda_mmr_fullft": "heuristic lambda full-ft",
    "mmr_sensitivity_gated": "sensitivity gated",
    "mmr_topk_sweep_infer": "reuse top_k sweep",
    "reranker_only": "reranker only",
}
SINGLE_SOURCE_MARKERS = {
    "heuristic_lambda_mmr": "D",
    "heuristic_lambda_mmr_fullft": "*",
    "mmr_sensitivity_gated": "P",
    "reranker_only": "X",
}
SINGLE_SOURCE_X_JITTER = {
    "heuristic_lambda_mmr": -0.24,
    "heuristic_lambda_mmr_fullft": -0.48,
    "mmr_sensitivity_gated": 0.0,
    "reranker_only": 0.24,
}
SINGLE_SOURCE_LABEL_OFFSET = {
    "heuristic_lambda_mmr": (-54, 18),
    "heuristic_lambda_mmr_fullft": (-70, 42),
    "mmr_sensitivity_gated": (10, -32),
    "reranker_only": (12, 30),
}
SINGLE_SOURCE_SHORT_LABELS = {
    "heuristic_lambda_mmr": "H",
    "heuristic_lambda_mmr_fullft": "HF",
    "mmr_sensitivity_gated": "G",
    "reranker_only": "R",
}


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
    overrides.update(_parse_sensitivity_slug(slug))
    return overrides


def _parse_slug_value(raw: str) -> int | float | bool | str:
    return _coerce_scalar(raw.replace("p", "."))


def _parse_sensitivity_slug(slug: str) -> dict[str, int | float | bool | str]:
    pattern = re.compile(
        r"^ts(?P<theta_s>[0-9p.]+)_tr(?P<theta_r>[0-9p.]+)_ll(?P<lambda_low>[0-9p.]+)_"
        r"(?P<gating_mode>[A-Za-z0-9-]+)(?:_eps(?P<epsilon>[0-9p.]+))?$"
    )
    match = pattern.fullmatch(slug)
    if match is None:
        return {}
    values: dict[str, int | float | bool | str] = {
        "sensitivity.theta_s": _parse_slug_value(match.group("theta_s")),
        "sensitivity.theta_r": _parse_slug_value(match.group("theta_r")),
        "sensitivity.lambda_low": _parse_slug_value(match.group("lambda_low")),
        "sensitivity.gating_mode": match.group("gating_mode"),
    }
    epsilon = match.group("epsilon")
    if epsilon is not None:
        values["sensitivity.epsilon"] = _parse_slug_value(epsilon)
    return values


def _load_experiment_retrieval_defaults(source_name: str) -> dict[str, int | float | bool | str]:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "experiment" / f"{source_name}.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    retrieval = config.get("build", {}).get("retrieval", {})
    if not isinstance(retrieval, dict):
        return {}
    defaults: dict[str, int | float | bool | str] = {}
    for key in ("top_k", "mmr_lambda", "selection_method"):
        value = retrieval.get(key)
        if isinstance(value, (str, int, float, bool)):
            defaults[f"build.retrieval.{key}"] = _coerce_scalar(str(value))
    return defaults


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, complex)


def _flatten_numeric_metrics(
    value: Any,
    *,
    prefix: str,
    out: dict[str, float],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric_metrics(child, prefix=child_prefix, out=out)
        return
    if _is_numeric(value):
        numeric = float(value)
        if math.isfinite(numeric):
            out[prefix] = numeric


def _get_nested_float(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if not _is_numeric(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _read_prompt_stats(run_dir: Path) -> tuple[dict[str, float], Path, bool]:
    path = run_dir / "prompt_stats" / "prompt_stats.json"
    if not path.exists():
        path = run_dir / "train" / "prompt_stats" / "prompt_stats.json"
    if not path.exists():
        return {}, path, False
    data = _read_json(path)
    values: dict[str, float] = {}
    for split in ("train", "val"):
        for key in ("mean", "p50", "p90", "p95", "p99", "max", "overflow_rate"):
            value = _get_nested_float(data, (split, "prompt_token_count", key))
            if value is not None:
                values[f"prompt.{split}.prompt_token_count.{key}"] = value
        for key in ("truncation_rate", "mean_evidence_count"):
            value = _get_nested_float(data, (split, "evidence_truncation", key))
            if value is not None:
                values[f"prompt.{split}.evidence_truncation.{key}"] = value
    return values, path, True


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


def _metric_sort_key(metric: str) -> tuple[int, int, str]:
    if metric in PREFERRED_METRIC_COLUMNS:
        return (0, PREFERRED_METRIC_COLUMNS.index(metric), metric)
    parts = metric.split(".")
    if len(parts) == 3 and parts[0] == "per_class":
        label = parts[1]
        score_name = parts[2]
        label_order = LABEL_ORDER.index(label) if label in LABEL_ORDER else len(LABEL_ORDER)
        score_order = PER_CLASS_METRICS.index(score_name) if score_name in PER_CLASS_METRICS else 99
        return (1, label_order * 10 + score_order, metric)
    return (2, 0, metric)


def _metrics_paths(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.rglob("metrics.json")
        if path.parent.name == "api" and "infer" in path.parts
    )


def _extract_path_metadata(path: Path, run_root: Path) -> dict[str, str]:
    relative = path.relative_to(run_root)
    parts = relative.parts
    try:
        infer_idx = parts.index("infer")
    except ValueError as exc:
        raise ValueError(f"Cannot locate infer segment in {path}") from exc
    if infer_idx < 1 or infer_idx + 4 >= len(parts):
        raise ValueError(f"Unexpected infer metrics path shape: {path}")
    run_name = parts[0]
    return {
        "run_name": run_name,
        "split": parts[infer_idx + 1],
        "checkpoint": parts[infer_idx + 2],
        "infer_id": parts[infer_idx + 3],
        "run_dir": str(run_root / run_name),
    }


def collect_rows(
    run_roots: list[Path],
    *,
    splits: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    override_keys: set[str] = set()
    metric_keys: set[str] = set()
    prompt_keys: set[str] = set()
    for run_root in run_roots:
        paths = _metrics_paths(run_root)
        if not paths:
            raise FileNotFoundError(f"No infer api metrics.json files found under {run_root}")
        experiment_defaults = _load_experiment_retrieval_defaults(run_root.name)
        matched_paths = 0
        for path in paths:
            metadata = _extract_path_metadata(path, run_root)
            if splits is not None and metadata["split"] not in splits:
                continue
            matched_paths += 1
            metrics = _read_json(path)
            flat_metrics: dict[str, float] = {}
            _flatten_numeric_metrics(metrics, prefix="", out=flat_metrics)
            stat = path.stat()
            overrides = _parse_run_overrides(metadata["run_name"])
            for key, value in experiment_defaults.items():
                overrides.setdefault(key, value)
            prompt_stats, prompt_stats_path, prompt_stats_found = _read_prompt_stats(Path(metadata["run_dir"]))
            row: dict[str, Any] = {
                "source_root": run_root.name,
                **metadata,
                "duplicate_rank": 1,
                "duplicate_count": 1,
                "metrics_path": str(path),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "prompt_stats_path": str(prompt_stats_path),
                "prompt_stats_found": prompt_stats_found,
                "_mtime": stat.st_mtime,
                **overrides,
                **flat_metrics,
                **prompt_stats,
            }
            rows.append(row)
            override_keys.update(overrides.keys())
            metric_keys.update(flat_metrics.keys())
            prompt_keys.update(prompt_stats.keys())
        if splits is not None and matched_paths == 0:
            split_list = ", ".join(sorted(splits))
            raise FileNotFoundError(f"No infer api metrics.json files matching split={split_list} under {run_root}")

    _annotate_duplicates(rows)
    ordered_override_keys = [
        key for key in PREFERRED_OVERRIDE_COLUMNS if key in override_keys
    ] + [
        key for key in SENSITIVITY_COLUMNS if key in override_keys
    ] + sorted(key for key in override_keys if key not in PREFERRED_OVERRIDE_COLUMNS)
    ordered_override_keys = list(dict.fromkeys(ordered_override_keys))
    ordered_metric_keys = sorted(metric_keys, key=_metric_sort_key)
    ordered_prompt_keys = [key for key in PROMPT_STAT_COLUMNS if key in prompt_keys]
    return rows, ordered_override_keys, ordered_metric_keys, ordered_prompt_keys


def _duplicate_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_root"),
        row.get("build.retrieval.top_k"),
        row.get("build.retrieval.mmr_lambda"),
        row.get("split"),
        row.get("checkpoint"),
    )


def _annotate_duplicates(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_duplicate_group_key(row), []).append(row)
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: (float(row.get("_mtime", 0.0)), str(row["metrics_path"])))
        for rank, row in enumerate(ordered, start=1):
            row["duplicate_rank"] = rank
            row["duplicate_count"] = len(ordered)


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[str, float, str, str, int]:
        top_k = row.get("build.retrieval.top_k")
        numeric_top_k = float(top_k) if isinstance(top_k, (int, float)) else math.inf
        return (
            str(row.get("source_root", "")),
            numeric_top_k,
            str(row.get("split", "")),
            str(row.get("checkpoint", "")),
            int(row.get("duplicate_rank", 0)),
        )

    return sorted(rows, key=sort_key)


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_group: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = _duplicate_group_key(row)
        current = latest_by_group.get(key)
        if current is None or (float(row.get("_mtime", 0.0)), str(row["metrics_path"])) > (
            float(current.get("_mtime", 0.0)),
            str(current["metrics_path"]),
        ):
            latest_by_group[key] = row
    return _sort_rows(list(latest_by_group.values()))


def write_csv(rows: list[dict[str, Any]], columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in columns})


def write_prompt_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns = [
        "source_root",
        "run_name",
        "split",
        "checkpoint",
        "infer_id",
        "build.retrieval.top_k",
        "build.retrieval.mmr_lambda",
        "macro_f1",
        "prompt_stats_found",
        "prompt_stats_path",
        *PROMPT_STAT_COLUMNS,
    ]
    write_csv(rows, columns, output_path)


def _markdown_table(headers: list[str], table_rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _duplicate_summary_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_duplicate_group_key(row), []).append(row)
    summary_rows: list[list[str]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        if len(group_rows) <= 1:
            continue
        source_root, top_k, mmr_lambda, split, checkpoint = key
        infer_ids = ", ".join(str(row["infer_id"]) for row in _sort_rows(group_rows))
        summary_rows.append(
            [
                _format_cell(source_root),
                _format_cell(top_k),
                _format_cell(mmr_lambda),
                _format_cell(split),
                _format_cell(checkpoint),
                str(len(group_rows)),
                infer_ids,
            ]
        )
    return summary_rows


def _row_label(row: dict[str, Any]) -> str:
    source = str(row.get("source_root", ""))
    run_name = str(row.get("run_name", ""))
    if source == "mmr_sensitivity_gated" and run_name:
        return f"{source}/{run_name}"
    return source


def _near_best_summary_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    summary_rows: list[list[str]] = []
    for metric in CLOSE_METRIC_COLUMNS:
        metric_rows = [
            row
            for row in rows
            if isinstance(row.get(metric), (int, float)) and isinstance(row.get("build.retrieval.top_k"), (int, float))
        ]
        if not metric_rows:
            continue
        best = max(metric_rows, key=lambda row: (float(row[metric]), -float(row["build.retrieval.top_k"])))
        best_value = float(best[metric])
        near_rows = [
            row
            for row in metric_rows
            if row is not best and 0 <= best_value - float(row[metric]) <= CLOSE_ABS_MARGIN
        ]
        if not near_rows:
            continue
        near_cells = []
        for row in sorted(near_rows, key=lambda item: (best_value - float(item[metric]), _row_label(item))):
            value = float(row[metric])
            delta = best_value - value
            near_cells.append(
                f"{_row_label(row)} top_k={_format_cell(row.get('build.retrieval.top_k'))}: "
                f"{_format_cell(value)} (delta={delta:.6f})"
            )
        summary_rows.append(
            [
                metric,
                f"{_row_label(best)} top_k={_format_cell(best.get('build.retrieval.top_k'))}: "
                f"{_format_cell(best_value)}",
                "; ".join(near_cells),
                f"abs delta <= {CLOSE_ABS_MARGIN:g}",
            ]
        )
    return summary_rows


def _metric_leader_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    leader_rows: list[list[str]] = []
    for metric in CLOSE_METRIC_COLUMNS:
        metric_rows = [
            row
            for row in rows
            if isinstance(row.get(metric), (int, float)) and isinstance(row.get("build.retrieval.top_k"), (int, float))
        ]
        ordered = sorted(
            metric_rows,
            key=lambda row: (float(row[metric]), -float(row["build.retrieval.top_k"]), _row_label(row)),
            reverse=True,
        )
        if not ordered:
            continue
        best = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        best_value = float(best[metric])
        runner_value = float(runner_up[metric]) if runner_up is not None else None
        leader_rows.append(
            [
                metric,
                f"{_row_label(best)} top_k={_format_cell(best.get('build.retrieval.top_k'))}: "
                f"{_format_cell(best_value)}",
                (
                    f"{_row_label(runner_up)} top_k={_format_cell(runner_up.get('build.retrieval.top_k'))}: "
                    f"{_format_cell(runner_value)}"
                    if runner_up is not None and runner_value is not None
                    else ""
                ),
                f"{best_value - runner_value:.6f}" if runner_value is not None else "",
            ]
        )
    return leader_rows


def _sensitivity_detail_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    detail_rows: list[list[str]] = []
    for row in rows:
        if not any(key in row for key in SENSITIVITY_COLUMNS):
            continue
        detail_rows.append(
            [
                _format_cell(row.get("source_root")),
                _format_cell(row.get("run_name")),
                _format_cell(row.get("build.retrieval.top_k")),
                _format_cell(row.get("build.retrieval.mmr_lambda")),
                _format_cell(row.get("sensitivity.theta_s")),
                _format_cell(row.get("sensitivity.theta_r")),
                _format_cell(row.get("sensitivity.lambda_low")),
                _format_cell(row.get("sensitivity.gating_mode")),
                _format_cell(row.get("sensitivity.epsilon")),
                _format_cell(row.get("accuracy")),
                _format_cell(row.get("macro_f1")),
            ]
        )
    return detail_rows


def _missing_prompt_stats_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    seen: set[tuple[str, str, str]] = set()
    missing_rows: list[list[str]] = []
    for row in rows:
        if row.get("prompt_stats_found"):
            continue
        key = (
            str(row.get("source_root", "")),
            str(row.get("run_name", "")),
            str(row.get("prompt_stats_path", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        missing_rows.append(
            [
                _format_cell(row.get("source_root")),
                _format_cell(row.get("run_name")),
                _format_cell(row.get("build.retrieval.top_k")),
                _format_cell(row.get("prompt_stats_path")),
            ]
        )
    return missing_rows


def write_markdown_summary(
    *,
    all_rows: list[dict[str, Any]],
    latest: list[dict[str, Any]],
    output_path: Path,
    all_csv: Path,
    latest_csv: Path,
    prompt_csv: Path,
    plot_path: Path,
    prompt_plot_path: Path,
) -> None:
    headers = [
        "source_root",
        "top_k",
        "mmr_lambda",
        "split",
        "checkpoint",
        "infer_id",
        "num_samples",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "parse_error_rate",
    ]
    table_rows = [
        [
            _format_cell(row.get("source_root")),
            _format_cell(row.get("build.retrieval.top_k")),
            _format_cell(row.get("build.retrieval.mmr_lambda")),
            _format_cell(row.get("split")),
            _format_cell(row.get("checkpoint")),
            _format_cell(row.get("infer_id")),
            _format_cell(row.get("num_samples")),
            _format_cell(row.get("accuracy")),
            _format_cell(row.get("macro_precision")),
            _format_cell(row.get("macro_recall")),
            _format_cell(row.get("macro_f1")),
            _format_cell(row.get("parse_error_rate")),
        ]
        for row in latest
    ]
    duplicate_rows = _duplicate_summary_rows(all_rows)
    leader_rows = _metric_leader_rows(latest)
    near_best_rows = _near_best_summary_rows(latest)
    sensitivity_rows = _sensitivity_detail_rows(latest)
    missing_prompt_rows = _missing_prompt_stats_rows(latest)

    lines = [
        "# Overall Run Analysis",
        "",
        "## Infer Metrics Summary",
        "",
        f"- All API metric records: `{all_csv.name}`",
        f"- Latest record per source/top_k/split/checkpoint: `{latest_csv.name}`",
        f"- Line chart: `{plot_path.name}`",
        f"- Prompt statistics table: `{prompt_csv.name}`",
        f"- Prompt statistics chart: `{prompt_plot_path.name}`",
        f"- Total records: {len(all_rows)}",
        f"- Latest rows: {len(latest)}",
        "",
        "## Included Artifacts",
        "",
        "- Overall infer metrics table: `infer_metrics_summary_latest.csv`",
        "- Overall infer metrics chart: `infer_metrics_line_chart.png`",
        "- Prompt statistics table: `prompt_stats_summary.csv`",
        "- Prompt statistics chart: `prompt_stats_line_chart.png`",
        "- b3 1024 top_k=0..8 test table: `b3_mmr_topk_test_curves_1024/test_metrics_top_k_0_8.csv`",
        "- b3 1024 top_k=0..8 test chart: `b3_mmr_topk_test_curves_1024/test_metrics_top_k_0_8.png`",
        "",
        "## Latest Records",
        "",
        _markdown_table(headers, table_rows),
    ]
    if leader_rows:
        lines.extend(
            [
                "",
                "## Metric Leaders",
                "",
                _markdown_table(["metric", "best", "runner_up", "delta"], leader_rows),
            ]
        )
    if near_best_rows:
        lines.extend(
            [
                "",
                "## Near-Best Metrics",
                "",
                _markdown_table(
                    ["metric", "best", "near records", "criterion"],
                    near_best_rows,
                ),
            ]
        )
    if sensitivity_rows:
        lines.extend(
            [
                "",
                "## Sensitivity-Gated Details",
                "",
                _markdown_table(
                    [
                        "source_root",
                        "run_name",
                        "top_k",
                        "mmr_lambda",
                        "theta_s",
                        "theta_r",
                        "lambda_low",
                        "gating_mode",
                        "epsilon",
                        "accuracy",
                        "macro_f1",
                    ],
                    sensitivity_rows,
                ),
            ]
        )
    if missing_prompt_rows:
        lines.extend(
            [
                "",
                "## Missing Prompt Stats",
                "",
                "These runs are missing `prompt_stats/prompt_stats.json`; prompt-stat panels skip them.",
                "",
                _markdown_table(["source_root", "run_name", "top_k", "expected_path"], missing_prompt_rows),
            ]
        )
    if duplicate_rows:
        lines.extend(
            [
                "",
                "## Duplicate Groups",
                "",
                _markdown_table(
                    ["source_root", "top_k", "mmr_lambda", "split", "checkpoint", "count", "infer_ids"],
                    duplicate_rows,
                ),
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_row_for_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    metric_rows = [
        row
        for row in rows
        if isinstance(row.get("build.retrieval.top_k"), (int, float)) and isinstance(row.get(metric), (int, float))
    ]
    if not metric_rows:
        return None
    if metric == "parse_error_rate":
        return min(metric_rows, key=lambda row: (float(row[metric]), float(row["build.retrieval.top_k"])))
    return max(metric_rows, key=lambda row: (float(row[metric]), -float(row["build.retrieval.top_k"])))


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


def _metric_title(metric: str) -> str:
    if metric == PROMPT_MEAN_COLUMN:
        return "avg_prompt_tokens_train"
    return metric


def write_line_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        top_k = row.get("build.retrieval.top_k")
        if not isinstance(top_k, (int, float)):
            continue
        by_source.setdefault(str(row.get("source_root", "")), []).append(row)

    ncols = 2
    nrows = math.ceil(len(MAIN_PLOT_COLUMNS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2 * ncols, 3.7 * nrows), squeeze=False)
    flat_axes = [axis for row_axes in axes for axis in row_axes]
    for axis, metric in zip(flat_axes, MAIN_PLOT_COLUMNS):
        all_metric_rows = [
            row
            for row in rows
            if isinstance(row.get("build.retrieval.top_k"), (int, float)) and isinstance(row.get(metric), (int, float))
        ]
        overall_best = _best_row_for_metric(all_metric_rows, metric) if metric in CLOSE_METRIC_COLUMNS else None
        if overall_best is not None:
            best_y = float(overall_best[metric])
            axis.axhspan(
                best_y - CLOSE_ABS_MARGIN,
                best_y,
                color="#f2c94c",
                alpha=0.12,
                label=f"near-best band (delta <= {CLOSE_ABS_MARGIN:g})",
                zorder=0,
            )

        for source, source_rows in sorted(by_source.items()):
            ordered = [
                row
                for row in sorted(source_rows, key=lambda item: float(item["build.retrieval.top_k"]))
                if isinstance(row.get(metric), (int, float))
            ]
            if not ordered:
                continue
            label = _source_label(source)
            x_values = [float(row["build.retrieval.top_k"]) for row in ordered]
            y_values = [float(row[metric]) for row in ordered]
            if len(ordered) == 1:
                row = ordered[0]
                base_x = float(row["build.retrieval.top_k"])
                visual_x = base_x + SINGLE_SOURCE_X_JITTER.get(source, 0.0)
                y_value = float(row[metric])
                marker = SINGLE_SOURCE_MARKERS.get(source, "s")
                (point,) = axis.plot(
                    [visual_x],
                    [y_value],
                    marker=marker,
                    markersize=8.5,
                    linestyle="None",
                    label=f"{label} (top_k={_format_cell(base_x)})",
                    zorder=5,
                )
                color = point.get_color()
                axis.axhline(y_value, color=color, linestyle=":", linewidth=1.2, alpha=0.58, zorder=1)
                if metric != "parse_error_rate":
                    offset = SINGLE_SOURCE_LABEL_OFFSET.get(source, (6, 10))
                    short_label = SINGLE_SOURCE_SHORT_LABELS.get(source, label)
                    if metric in CLOSE_METRIC_COLUMNS:
                        axis.annotate(
                            f"{short_label} {y_value:.4f}",
                            xy=(visual_x, y_value),
                            xytext=offset,
                            textcoords="offset points",
                            fontsize=7,
                            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.78},
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.7},
                            zorder=6,
                        )
                    if overall_best is not None and 0 <= float(overall_best[metric]) - y_value <= CLOSE_ABS_MARGIN:
                        axis.scatter(
                            [visual_x],
                            [y_value],
                            s=155,
                            facecolors="none",
                            edgecolors="black",
                            linewidth=0.9,
                            zorder=6,
                        )
                continue

            (line,) = axis.plot(x_values, y_values, marker="o", linewidth=1.8, label=label, zorder=2)

            best = _best_row_for_metric(ordered, metric)
            if best is None:
                continue
            best_x = float(best["build.retrieval.top_k"])
            best_y = float(best[metric])
            color = line.get_color()
            if metric in CLOSE_METRIC_COLUMNS:
                axis.axhline(best_y, color=color, linestyle="--", linewidth=1.2, alpha=0.45, zorder=1)
                axis.scatter(
                    [best_x],
                    [best_y],
                    marker="*",
                    s=140,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    label=f"{label} best",
                    zorder=4,
                )
        axis.set_title(_metric_title(metric))
        axis.set_xlabel("build.retrieval.top_k")
        axis.grid(True, alpha=0.25)
        if metric == "parse_error_rate" and all(abs(float(row[metric])) < 1e-12 for row in all_metric_rows):
            axis.set_ylim(-0.01, 0.05)
        axis.legend(fontsize=7.5)
        all_top_k = sorted(
            {
                float(row["build.retrieval.top_k"])
                for row in rows
                if isinstance(row.get("build.retrieval.top_k"), (int, float))
            }
        )
        if len(all_top_k) <= 24:
            axis.set_xticks(all_top_k)
            axis.set_xticklabels([_format_cell(int(x) if x.is_integer() else x) for x in all_top_k], rotation=30)

    for axis in flat_axes[len(MAIN_PLOT_COLUMNS):]:
        axis.axis("off")

    fig.suptitle(
        f"Infer test metrics and prompt length by top_k (near-best band delta <= {CLOSE_ABS_MARGIN:g})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_prompt_panel(axis: Any, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        top_k = row.get("build.retrieval.top_k")
        if isinstance(top_k, (int, float)) and isinstance(row.get(metric), (int, float)):
            by_source.setdefault(str(row.get("source_root", "")), []).append(row)
    for source, source_rows in sorted(by_source.items()):
        label = _source_label(source)
        ordered = sorted(source_rows, key=lambda row: float(row["build.retrieval.top_k"]))
        x_values = [float(row["build.retrieval.top_k"]) for row in ordered]
        y_values = [float(row[metric]) for row in ordered]
        if len(ordered) == 1:
            base_x = x_values[0]
            visual_x = base_x + SINGLE_SOURCE_X_JITTER.get(source, 0.0)
            marker = SINGLE_SOURCE_MARKERS.get(source, "s")
            axis.plot(
                [visual_x],
                y_values,
                marker=marker,
                markersize=8,
                linestyle="None",
                label=f"{label} (top_k={_format_cell(base_x)})",
                zorder=4,
            )
        else:
            axis.plot(x_values, y_values, marker="o", linewidth=1.8, label=label, zorder=2)
    axis.set_title(title)
    axis.set_xlabel("build.retrieval.top_k")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7.5)
    all_top_k = sorted(
        {
            float(row["build.retrieval.top_k"])
            for row in rows
            if isinstance(row.get("build.retrieval.top_k"), (int, float))
        }
    )
    if len(all_top_k) <= 24:
        axis.set_xticks(all_top_k)
        axis.set_xticklabels([_format_cell(int(x) if x.is_integer() else x) for x in all_top_k], rotation=30)


def write_prompt_stats_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("macro_f1", "macro_f1"),
        ("prompt.train.prompt_token_count.mean", "train avg prompt tokens"),
        ("prompt.val.prompt_token_count.mean", "val avg prompt tokens"),
        ("prompt.train.prompt_token_count.p90", "train p90 prompt tokens"),
        ("prompt.train.prompt_token_count.p99", "train p99 prompt tokens"),
        ("prompt.train.evidence_truncation.truncation_rate", "train truncation rate"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16.4, 11.1), squeeze=False)
    flat_axes = [axis for row_axes in axes for axis in row_axes]
    for axis, (metric, title) in zip(flat_axes, panels):
        _plot_prompt_panel(axis, rows, metric, title)
    fig.suptitle("Prompt statistics by top_k", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize infer/*/*/*/api/metrics.json files under one or more run roots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_roots", nargs="+", help="Run roots containing child run directories.")
    parser.add_argument("--output-dir", default="outputs/runs/overall_run_analysis")
    parser.add_argument("--all-csv", default="infer_metrics_summary_all.csv")
    parser.add_argument("--latest-csv", default="infer_metrics_summary_latest.csv")
    parser.add_argument("--prompt-csv", default="prompt_stats_summary.csv")
    parser.add_argument("--markdown", default="infer_metrics_summary.md")
    parser.add_argument("--plot", default="infer_metrics_line_chart.png")
    parser.add_argument("--prompt-plot", default="prompt_stats_line_chart.png")
    parser.add_argument("--split", action="append", help="Only include this infer split. Repeat for multiple splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_roots = [Path(path).resolve() for path in args.run_roots]
    output_dir = Path(args.output_dir).resolve()

    splits = set(args.split) if args.split else None
    rows, override_columns, metric_columns, prompt_columns = collect_rows(run_roots, splits=splits)
    rows = _sort_rows(rows)
    latest = latest_rows(rows)
    columns = METADATA_COLUMNS + PROMPT_METADATA_COLUMNS + override_columns + metric_columns + prompt_columns

    all_csv = output_dir / args.all_csv
    latest_csv = output_dir / args.latest_csv
    prompt_csv = output_dir / args.prompt_csv
    markdown = output_dir / args.markdown
    plot_path = output_dir / args.plot
    prompt_plot_path = output_dir / args.prompt_plot
    write_csv(rows, columns, all_csv)
    write_csv(latest, columns, latest_csv)
    write_prompt_csv(latest, prompt_csv)
    write_line_plot(latest, plot_path)
    write_prompt_stats_plot(latest, prompt_plot_path)
    write_markdown_summary(
        all_rows=rows,
        latest=latest,
        output_path=markdown,
        all_csv=all_csv,
        latest_csv=latest_csv,
        prompt_csv=prompt_csv,
        plot_path=plot_path,
        prompt_plot_path=prompt_plot_path,
    )

    print(f"[infer_metrics] records={len(rows)}")
    print(f"[infer_metrics] latest_rows={len(latest)}")
    print(f"[infer_metrics] all_csv={all_csv}")
    print(f"[infer_metrics] latest_csv={latest_csv}")
    print(f"[infer_metrics] prompt_csv={prompt_csv}")
    print(f"[infer_metrics] markdown={markdown}")
    print(f"[infer_metrics] plot={plot_path}")
    print(f"[infer_metrics] prompt_plot={prompt_plot_path}")


if __name__ == "__main__":
    main()
