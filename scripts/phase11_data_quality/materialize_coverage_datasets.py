#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COVERED = "covered"
WEAK_COVERED = "weak_covered"
UNCOVERED = "uncovered"
VALID_LABELS = {COVERED, WEAK_COVERED, UNCOVERED}
DEFAULT_COVERAGE_VERSION = "source_coverage_v2"
SPLITS = ("train", "val", "test")
POLICIES = {
    "all": {COVERED, WEAK_COVERED, UNCOVERED},
    "covered": {COVERED},
    "covered_weak": {COVERED, WEAK_COVERED},
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_dir: Path
    coverage_dir: Path
    id_key: str
    explain_key: str
    evidence_key: str


DATASET_SPECS = {
    "liar_raw": DatasetSpec(
        name="liar_raw",
        raw_dir=Path("data/raw/LIAR-RAW"),
        coverage_dir=Path("outputs/data_quality/source_coverage/liar_raw"),
        id_key="event_id",
        explain_key="explain",
        evidence_key="reports",
    ),
    "rawfc": DatasetSpec(
        name="rawfc",
        raw_dir=Path("data/raw/RAWFC"),
        coverage_dir=Path("outputs/data_quality/source_coverage/rawfc"),
        id_key="id",
        explain_key="explanation",
        evidence_key="evidence",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize raw datasets with coverage labels and filtered subsets.")
    p.add_argument("--dataset", default="all", choices=["all", *DATASET_SPECS.keys()])
    p.add_argument("--raw-dir", default=None, help="Override raw split directory for one dataset.")
    p.add_argument("--coverage-dir", default=None, help="Override coverage sidecar directory for one dataset.")
    p.add_argument(
        "--output-root",
        default=None,
        help="Root containing <dataset>/<policy>/<split>.json outputs.",
    )
    p.add_argument("--coverage-version", default=DEFAULT_COVERAGE_VERSION)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--policies", nargs="+", default=list(POLICIES.keys()), choices=POLICIES.keys())
    p.add_argument("--sample-limit", type=int, default=None, help="Optional raw split prefix for smoke tests.")
    p.add_argument("--event-id", action="append", default=None, help="Optional id filter; may be repeated.")
    p.add_argument("--allow-partial", action="store_true", help="Skip raw rows missing a coverage record.")
    p.add_argument("--indent", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "all" and (args.raw_dir or args.coverage_dir):
        raise SystemExit("--raw-dir/--coverage-dir can only be used with a single --dataset.")
    datasets = list(DATASET_SPECS) if args.dataset == "all" else [str(args.dataset)]
    summaries: dict[str, Any] = {}
    output_root = Path(args.output_root or f"data/processed/coverage/{args.coverage_version}")
    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        raw_dir = Path(args.raw_dir) if args.raw_dir else spec.raw_dir
        coverage_dir = Path(args.coverage_dir) if args.coverage_dir else spec.coverage_dir
        summary = materialize_dataset(
            spec=spec,
            raw_dir=raw_dir,
            coverage_dir=coverage_dir,
            output_root=output_root,
            coverage_version=str(args.coverage_version),
            splits=list(args.splits),
            policies=list(args.policies),
            strict=not bool(args.allow_partial),
            sample_limit=args.sample_limit,
            event_ids=set(str(x) for x in args.event_id) if args.event_id else None,
            indent=int(args.indent),
        )
        summaries[dataset] = summary
        summary_path = output_root / dataset / "materialization_summary.json"
        save_json(summary, summary_path, indent=int(args.indent))
        print(f"Wrote materialization summary: {summary_path}")
    if len(summaries) > 1:
        combined_path = output_root / "materialization_summary.json"
        save_json(summaries, combined_path, indent=int(args.indent))
        print(f"Wrote combined materialization summary: {combined_path}")


def materialize_dataset(
    *,
    spec: DatasetSpec,
    raw_dir: Path,
    coverage_dir: Path,
    output_root: Path,
    coverage_version: str,
    splits: list[str],
    policies: list[str],
    strict: bool,
    sample_limit: int | None,
    event_ids: set[str] | None,
    indent: int,
) -> dict[str, Any]:
    dataset_summary: dict[str, Any] = {
        "dataset": spec.name,
        "raw_dir": str(raw_dir),
        "coverage_dir": str(coverage_dir),
        "output_root": str(output_root),
        "coverage_version": coverage_version,
        "strict": bool(strict),
        "sample_limit": sample_limit,
        "event_id_filter": sorted(event_ids) if event_ids else [],
        "splits": {},
    }
    for split in splits:
        split_summary = materialize_split(
            spec=spec,
            split=split,
            raw_path=raw_dir / f"{split}.json",
            coverage_path=coverage_dir / f"source_coverage_{split}.jsonl",
            output_root=output_root,
            coverage_version=coverage_version,
            policies=policies,
            strict=strict,
            sample_limit=sample_limit,
            event_ids=event_ids,
            indent=indent,
        )
        dataset_summary["splits"][split] = split_summary
    return dataset_summary


def materialize_split(
    *,
    spec: DatasetSpec,
    split: str,
    raw_path: Path,
    coverage_path: Path,
    output_root: Path,
    coverage_version: str,
    policies: list[str],
    strict: bool,
    sample_limit: int | None,
    event_ids: set[str] | None,
    indent: int,
) -> dict[str, Any]:
    raw_rows = load_json_list(raw_path)
    raw_filter_active = bool(event_ids) or sample_limit is not None
    if event_ids:
        raw_rows = [row for row in raw_rows if raw_id(row, spec) in event_ids]
    if sample_limit is not None:
        raw_rows = raw_rows[: int(sample_limit)]
    coverage_by_id = load_coverage_rows(coverage_path, split=split)
    raw_ids = [raw_id(row, spec) for row in raw_rows]
    duplicate_raw_ids = sorted(find_duplicates(raw_ids))
    if duplicate_raw_ids:
        raise ValueError(f"Duplicate raw ids in {raw_path}: {duplicate_raw_ids[:10]}")
    missing = [event_id for event_id in raw_ids if event_id not in coverage_by_id]
    extra = sorted(set(coverage_by_id) - set(raw_ids))
    if strict and missing:
        raise ValueError(f"Missing coverage rows for {len(missing)} raw samples in {raw_path}: {missing[:10]}")
    if strict and extra and not raw_filter_active:
        raise ValueError(f"Coverage rows not present in raw split {raw_path}: {extra[:10]}")

    annotated_rows: list[dict[str, Any]] = []
    skipped_missing = 0
    for raw_row in raw_rows:
        event_id = raw_id(raw_row, spec)
        coverage_row = coverage_by_id.get(event_id)
        if coverage_row is None:
            skipped_missing += 1
            continue
        annotated_rows.append(annotate_raw_row(raw_row, coverage_row, coverage_path=coverage_path, version=coverage_version))

    policy_summaries: dict[str, Any] = {}
    for policy in policies:
        allowed = POLICIES[policy]
        output_rows = [row for row in annotated_rows if str(row.get("coverage_label")) in allowed]
        output_path = output_root / spec.name / policy / f"{split}.json"
        save_json(output_rows, output_path, indent=indent)
        policy_summaries[policy] = {
            "output": str(output_path),
            "n_rows": len(output_rows),
            "coverage_counts": count_by(output_rows, "coverage_label"),
            "gold_label_counts": count_by(output_rows, "label"),
            "gold_by_coverage": cross_count(output_rows, row_key="label", col_key="coverage_label"),
            "filtered_out": len(annotated_rows) - len(output_rows),
        }
        print(f"Wrote {spec.name} {policy} {split}: {output_path} ({len(output_rows)} rows)")

    return {
        "raw_path": str(raw_path),
        "coverage_path": str(coverage_path),
        "raw_rows": len(raw_rows),
        "annotated_rows": len(annotated_rows),
        "skipped_missing_coverage": skipped_missing,
        "coverage_counts": count_by(annotated_rows, "coverage_label"),
        "gold_label_counts": count_by(annotated_rows, "label"),
        "gold_by_coverage": cross_count(annotated_rows, row_key="label", col_key="coverage_label"),
        "policies": policy_summaries,
    }


def annotate_raw_row(raw_row: dict[str, Any], coverage_row: dict[str, Any], *, coverage_path: Path, version: str) -> dict[str, Any]:
    label = str(coverage_row.get("coverage_label") or "")
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid coverage_label={label!r} for event_id={coverage_row.get('event_id')!r}")
    retrieval = coverage_row.get("retrieval") if isinstance(coverage_row.get("retrieval"), dict) else {}
    annotated = copy.deepcopy(raw_row)
    annotated["coverage_label"] = label
    annotated["coverage_score"] = coverage_row.get("coverage_score")
    annotated["coverage_version"] = version
    annotated["coverage"] = {
        "rule_label": coverage_row.get("rule_coverage_label"),
        "embedding_score": retrieval.get("best_embedding"),
        "bm25_score": retrieval.get("best_bm25"),
        "lexical_score": retrieval.get("best_lexical"),
        "critical_missing": coverage_row.get("critical_missing") or [],
        "llm_judgment": coverage_row.get("llm_judgment") if isinstance(coverage_row.get("llm_judgment"), dict) else {},
        "source_sidecar": str(coverage_path),
    }
    return annotated


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at {path}[{idx}]")
        rows.append(item)
    return rows


def load_coverage_rows(path: Path, *, split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected coverage object in {path}:{line_no}")
            row_split = str(row.get("split") or "")
            if row_split and row_split != split:
                raise ValueError(f"Coverage split mismatch in {path}:{line_no}: {row_split!r} != {split!r}")
            event_id = str(row.get("event_id") or "")
            if not event_id:
                raise ValueError(f"Missing event_id in {path}:{line_no}")
            if event_id in rows:
                raise ValueError(f"Duplicate coverage row for event_id={event_id!r} in {path}:{line_no}")
            label = str(row.get("coverage_label") or "")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid coverage_label={label!r} in {path}:{line_no}")
            rows[event_id] = row
    return rows


def raw_id(row: dict[str, Any], spec: DatasetSpec) -> str:
    if spec.id_key not in row:
        raise ValueError(f"Missing {spec.id_key!r} in {spec.name} raw row")
    return str(row[spec.id_key])


def find_duplicates(values: list[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(key))] += 1
    return dict(sorted(counts.items()))


def cross_count(rows: list[dict[str, Any]], *, row_key: str, col_key: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = {}
    for row in rows:
        row_value = str(row.get(row_key))
        col_value = str(row.get(col_key))
        out.setdefault(row_value, Counter())[col_value] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(out.items())}


def save_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
