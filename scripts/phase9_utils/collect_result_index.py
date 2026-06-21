#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_COLUMNS = [
    "num_samples",
    "accuracy",
    "macro_f1",
    "true_side_macro_f1",
    "checkpoint_selection_score",
    "parse_error_rate",
    "eval_loss",
    "ordinal_mae",
]

CSV_COLUMNS = [
    "source_root",
    "artifact_kind",
    "metric_scope",
    "relative_path",
    "run_dir",
    "split",
    "checkpoint",
    "eval_kind",
    "step",
    "training_status",
    "size_bytes",
    "mtime",
    "sha256",
    *METRIC_COLUMNS,
]


@dataclass(frozen=True)
class IndexPaths:
    csv_path: Path
    jsonl_path: Path
    md_path: Path


def collect_inventory(outputs_root: Path) -> list[dict[str, Any]]:
    outputs_root = outputs_root.resolve()
    complete_roots, resume_roots = _collect_training_status_roots(outputs_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(outputs_root).parts
        if _is_excluded_artifact_path(rel_parts):
            continue
        artifact_kind = _artifact_kind(path)
        if artifact_kind is None:
            continue
        run_dir = _guess_run_dir(outputs_root, path)
        row: dict[str, Any] = {
            "source_root": rel_parts[0] if rel_parts else "",
            "artifact_kind": artifact_kind,
            "metric_scope": _metric_scope(path, rel_parts, artifact_kind),
            "relative_path": _display_path(outputs_root, path),
            "run_dir": _display_path(outputs_root, run_dir),
            "split": "",
            "checkpoint": "",
            "eval_kind": "",
            "step": "",
            "training_status": _training_status(run_dir, complete_roots, resume_roots),
            "size_bytes": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "sha256": _sha256(path),
        }
        row.update(_path_context(rel_parts))
        row.update(_metric_values(path))
        for column in CSV_COLUMNS:
            row.setdefault(column, "")
        rows.append(row)
    return rows


def _is_excluded_artifact_path(rel_parts: tuple[str, ...]) -> bool:
    if not rel_parts:
        return False
    if rel_parts[0] in {"cache", "logs"}:
        return True
    return any(
        part in {"_cache_build", "_raw_sources", "latest_state"} or part.startswith("checkpoint-")
        for part in rel_parts
    )


def write_index(rows: list[dict[str, Any]], output_dir: Path, *, stamp: str | None = None) -> IndexPaths:
    stamp = stamp or datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stamp}_outputs_metric_inventory.csv"
    jsonl_path = output_dir / f"{stamp}_outputs_metric_inventory.jsonl"
    md_path = output_dir / f"{stamp}_outputs_metric_inventory.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    md_path.write_text(_render_markdown(rows, stamp=stamp, csv_path=csv_path, jsonl_path=jsonl_path), encoding="utf-8")
    return IndexPaths(csv_path=csv_path, jsonl_path=jsonl_path, md_path=md_path)


def _artifact_kind(path: Path) -> str | None:
    name = path.name
    if name == "confusion_matrix.json":
        return "confusion_json"
    if name in {"metrics.json", "selection_metrics.json", "selector_metrics.json"}:
        return "metrics_json"
    if name.endswith(".json") and "metrics" in name:
        return "metrics_json"
    if name.endswith(".json") and "summary" in name:
        return "summary_json"
    if name.endswith(".json") and "comparison" in name:
        return "comparison_json"
    if name == "training_complete.json":
        return "training_complete"
    if name == "manifest.json":
        return "manifest"
    if name.endswith(".resolved.yaml"):
        return "resolved_yaml"
    return None


def _collect_training_status_roots(outputs_root: Path) -> tuple[set[Path], set[Path]]:
    complete_roots: set[Path] = set()
    resume_roots: set[Path] = set()
    for path in outputs_root.rglob("training_complete.json"):
        if path.parent.name == "train":
            complete_roots.add(path.parent.parent.resolve())
        else:
            complete_roots.add(path.parent.resolve())
    for path in outputs_root.rglob("latest_state"):
        if path.is_dir() and path.parent.name == "train":
            resume_roots.add(path.parent.parent.resolve())
    return complete_roots, resume_roots


def _guess_run_dir(outputs_root: Path, path: Path) -> Path:
    rel_parts = path.relative_to(outputs_root).parts
    boundary_indexes = [idx for idx, part in enumerate(rel_parts) if part in {"eval", "infer", "train"}]
    if boundary_indexes:
        boundary = min(boundary_indexes)
        if boundary > 0:
            return outputs_root.joinpath(*rel_parts[:boundary]).resolve()
    return path.parent.resolve()


def _display_path(outputs_root: Path, path: Path) -> str:
    try:
        return path.relative_to(outputs_root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _metric_scope(path: Path, rel_parts: tuple[str, ...], artifact_kind: str) -> str:
    rel = "/" + "/".join(rel_parts)
    if path.name == "metrics.json":
        if any(marker in rel for marker in ("/eval/val/", "/eval/test/", "/infer/val/", "/infer/test/")):
            return "final_named"
        if "/eval/step-" in rel or "/train/eval/step-" in rel:
            return "step_curve"
        return "other"
    if artifact_kind in {"metrics_json", "summary_json", "comparison_json"}:
        return "selector_or_summary"
    return ""


def _path_context(rel_parts: tuple[str, ...]) -> dict[str, str]:
    out = {"split": "", "checkpoint": "", "eval_kind": "", "step": ""}
    for token in ("eval", "infer"):
        if token not in rel_parts:
            continue
        idx = rel_parts.index(token)
        next_part = rel_parts[idx + 1] if idx + 1 < len(rel_parts) else ""
        if next_part.startswith("step-"):
            out["step"] = next_part
            return out
        if next_part in {"train", "val", "test"}:
            out["split"] = next_part
            if idx + 2 < len(rel_parts):
                out["checkpoint"] = rel_parts[idx + 2]
            if idx + 3 < len(rel_parts):
                out["eval_kind"] = rel_parts[idx + 3]
            return out
    return out


def _training_status(run_dir: Path, complete_roots: set[Path], resume_roots: set[Path]) -> str:
    resolved = run_dir.resolve()
    if resolved in complete_roots:
        return "complete"
    if resolved in resume_roots:
        return "resume_state_present"
    return "unknown"


def _metric_values(path: Path) -> dict[str, Any]:
    values = {column: "" for column in METRIC_COLUMNS}
    if path.suffix != ".json":
        return values
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return values
    if not isinstance(payload, dict):
        return values
    for column in METRIC_COLUMNS:
        if column == "checkpoint_selection_score":
            value = payload.get("checkpoint_selection_score", payload.get("selection_score"))
        else:
            value = payload.get(column)
        if isinstance(value, bool):
            values[column] = int(value)
        elif isinstance(value, int):
            values[column] = value
        elif isinstance(value, float) and math.isfinite(value):
            values[column] = value
    return values


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_markdown(rows: list[dict[str, Any]], *, stamp: str, csv_path: Path, jsonl_path: Path) -> str:
    by_root = Counter(str(row.get("source_root", "")) for row in rows)
    by_kind = Counter(str(row.get("artifact_kind", "")) for row in rows)
    by_scope = Counter(str(row.get("metric_scope", "")) for row in rows if row.get("metric_scope"))
    bytes_by_root: dict[str, int] = defaultdict(int)
    for row in rows:
        bytes_by_root[str(row.get("source_root", ""))] += int(row.get("size_bytes") or 0)

    lines = [
        f"# {stamp} Outputs Metric Inventory",
        "",
        "This file is generated by `scripts/phase9_utils/collect_result_index.py`.",
        "",
        f"- CSV: `{csv_path.as_posix()}`",
        f"- JSONL: `{jsonl_path.as_posix()}`",
        f"- rows: `{len(rows)}`",
        "",
        "## By Source Root",
        "",
        "| source_root | rows | size MiB |",
        "|---|---:|---:|",
    ]
    for root, count in sorted(by_root.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{root}` | {count} | {bytes_by_root[root] / 1024 / 1024:.2f} |")

    lines.extend(["", "## By Artifact Kind", "", "| artifact_kind | rows |", "|---|---:|"])
    for kind, count in sorted(by_kind.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{kind}` | {count} |")

    lines.extend(["", "## By Metric Scope", "", "| metric_scope | rows |", "|---|---:|"])
    for scope, count in sorted(by_scope.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{scope}` | {count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `final_named` means metrics under explicit `eval/{val,test}/...` or `infer/{val,test}/...` paths.",
            "- `step_curve` means training-time validation metrics under `eval/step-*` or `train/eval/step-*`.",
            "- `training_status=complete` is based on `train/training_complete.json`; `resume_state_present` is based on `train/latest_state/`.",
            "- This inventory stores lightweight metadata and paths for Git review; large predictions, checkpoints, caches, and logs are intentionally outside this index.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect lightweight result metrics into Git-friendly indexes.")
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument("--output-dir", default=Path("docs/Z-cross-cutting"), type=Path)
    parser.add_argument("--stamp", default=None, help="Filename stamp, for example 20260615.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_inventory(args.outputs_root)
    paths = write_index(rows, args.output_dir, stamp=args.stamp)
    print(f"rows={len(rows)}")
    print(f"csv={paths.csv_path}")
    print(f"jsonl={paths.jsonl_path}")
    print(f"md={paths.md_path}")


if __name__ == "__main__":
    main()
