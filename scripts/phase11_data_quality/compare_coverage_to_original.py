#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COVERED = "covered"
WEAK_COVERED = "weak_covered"
UNCOVERED = "uncovered"
VALID_LABELS = {COVERED, WEAK_COVERED, UNCOVERED}
SPLITS = ("train", "val", "test")
POLICIES = ("all", "covered", "covered_weak")
DEFAULT_COVERAGE_VERSION = "source_coverage_v2_flash"
DEFAULT_OUTPUT_BASE = Path("outputs/data_quality/source_coverage_flash")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_dir: Path
    coverage_dir: Path
    id_key: str


DATASET_SPECS = {
    "liar_raw": DatasetSpec(
        name="liar_raw",
        raw_dir=Path("data/raw/LIAR-RAW"),
        coverage_dir=DEFAULT_OUTPUT_BASE / "liar_raw",
        id_key="event_id",
    ),
    "rawfc": DatasetSpec(
        name="rawfc",
        raw_dir=Path("data/raw/RAWFC"),
        coverage_dir=DEFAULT_OUTPUT_BASE / "rawfc",
        id_key="id",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original raw datasets against coverage-labeled outputs.")
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_SPECS.keys()])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--raw-dir", default=None, help="Override raw split directory for one dataset.")
    parser.add_argument("--coverage-dir", default=None, help="Override coverage sidecar directory for one dataset.")
    parser.add_argument("--processed-root", default=None, help="Root containing <dataset>/<policy>/<split>.json.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--coverage-version", default=DEFAULT_COVERAGE_VERSION)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--allow-partial", dest="strict", action="store_false")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "all" and (args.raw_dir or args.coverage_dir):
        raise SystemExit("--raw-dir/--coverage-dir can only be used with a single --dataset.")

    datasets = list(DATASET_SPECS) if args.dataset == "all" else [str(args.dataset)]
    processed_root = Path(args.processed_root or f"data/processed/coverage/{args.coverage_version}")
    root_output_dir = Path(args.output_dir) if args.output_dir else default_root_output_dir(args)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        raw_dir = Path(args.raw_dir) if args.raw_dir else spec.raw_dir
        coverage_dir = Path(args.coverage_dir) if args.coverage_dir else spec.coverage_dir
        output_dir = root_output_dir / dataset if args.dataset == "all" else root_output_dir
        summary = compare_dataset(
            spec=spec,
            raw_dir=raw_dir,
            coverage_dir=coverage_dir,
            processed_root=processed_root,
            output_dir=output_dir,
            coverage_version=str(args.coverage_version),
            splits=list(args.splits),
            strict=bool(args.strict),
            indent=int(args.indent),
        )
        summaries[dataset] = summary

    combined = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coverage_version": str(args.coverage_version),
        "processed_root": str(processed_root),
        "strict": bool(args.strict),
        "datasets": summaries,
    }
    if len(summaries) > 1:
        save_json(combined, root_output_dir / "original_coverage_diff_summary.json", indent=int(args.indent))
        (root_output_dir / "original_coverage_diff_summary.md").write_text(render_combined_markdown(combined), encoding="utf-8")
        print(f"Wrote combined original diff summary: {root_output_dir}")


def default_root_output_dir(args: argparse.Namespace) -> Path:
    if str(args.dataset) == "all":
        return DEFAULT_OUTPUT_BASE / "original_diff"
    spec = DATASET_SPECS[str(args.dataset)]
    coverage_dir = Path(args.coverage_dir) if args.coverage_dir else spec.coverage_dir
    return coverage_dir / "original_diff"


def compare_dataset(
    *,
    spec: DatasetSpec,
    raw_dir: Path,
    coverage_dir: Path,
    processed_root: Path,
    output_dir: Path,
    coverage_version: str,
    splits: list[str],
    strict: bool,
    indent: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": spec.name,
        "coverage_version": coverage_version,
        "raw_dir": str(raw_dir),
        "coverage_dir": str(coverage_dir),
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "strict": bool(strict),
        "splits": {},
    }
    all_diff_rows: list[dict[str, Any]] = []
    for split in splits:
        split_summary, diff_rows = compare_split(
            spec=spec,
            split=split,
            raw_path=raw_dir / f"{split}.json",
            coverage_path=coverage_dir / f"source_coverage_{split}.jsonl",
            processed_dataset_dir=processed_root / spec.name,
            output_dir=output_dir,
            strict=strict,
        )
        summary["splits"][split] = split_summary
        all_diff_rows.extend(diff_rows)
    summary["overall"] = summarize_diff_rows(all_diff_rows)
    save_json(summary, output_dir / "original_coverage_diff_summary.json", indent=indent)
    (output_dir / "original_coverage_diff_summary.md").write_text(render_dataset_markdown(summary), encoding="utf-8")
    print(f"Wrote original coverage diff summary: {output_dir}")
    return summary


def compare_split(
    *,
    spec: DatasetSpec,
    split: str,
    raw_path: Path,
    coverage_path: Path,
    processed_dataset_dir: Path,
    output_dir: Path,
    strict: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_rows = load_json_list(raw_path)
    coverage_by_id = load_coverage_rows(coverage_path, split=split)
    processed_all_by_id = load_processed_policy(processed_dataset_dir / "all" / f"{split}.json", spec=spec)
    covered_ids = set(load_processed_policy(processed_dataset_dir / "covered" / f"{split}.json", spec=spec))
    covered_weak_ids = set(load_processed_policy(processed_dataset_dir / "covered_weak" / f"{split}.json", spec=spec))

    raw_ids = [raw_id(row, spec) for row in raw_rows]
    duplicate_raw_ids = sorted(find_duplicates(raw_ids))
    duplicate_processed_all_ids = sorted(find_duplicates(list(processed_all_by_id)))
    missing_coverage = [event_id for event_id in raw_ids if event_id not in coverage_by_id]
    extra_coverage = sorted(set(coverage_by_id) - set(raw_ids))
    missing_processed_all = [event_id for event_id in raw_ids if event_id not in processed_all_by_id]
    extra_processed_all = sorted(set(processed_all_by_id) - set(raw_ids))
    problems = {
        "duplicate_raw_ids": duplicate_raw_ids,
        "duplicate_processed_all_ids": duplicate_processed_all_ids,
        "missing_coverage": missing_coverage,
        "extra_coverage": extra_coverage,
        "missing_processed_all": missing_processed_all,
        "extra_processed_all": extra_processed_all,
    }
    if strict:
        first_problem = next((name for name, values in problems.items() if values), "")
        if first_problem:
            raise ValueError(f"{spec.name}/{split} failed strict alignment check: {first_problem}={problems[first_problem][:10]}")

    diff_rows: list[dict[str, Any]] = []
    skipped_missing = 0
    for raw_row in raw_rows:
        event_id = raw_id(raw_row, spec)
        coverage_row = coverage_by_id.get(event_id)
        processed_row = processed_all_by_id.get(event_id)
        if coverage_row is None or processed_row is None:
            skipped_missing += 1
            if strict:
                continue
            diff_rows.append(missing_diff_row(spec=spec, split=split, raw_row=raw_row, coverage_path=coverage_path, in_covered=event_id in covered_ids, in_covered_weak=event_id in covered_weak_ids))
            continue
        diff_rows.append(
            build_diff_row(
                spec=spec,
                split=split,
                raw_row=raw_row,
                coverage_row=coverage_row,
                processed_row=processed_row,
                coverage_path=coverage_path,
                in_covered=event_id in covered_ids,
                in_covered_weak=event_id in covered_weak_ids,
            )
        )

    output_path = output_dir / f"case_coverage_diff_{split}.jsonl"
    write_jsonl(diff_rows, output_path)
    split_summary = summarize_diff_rows(diff_rows)
    split_summary.update(
        {
            "split": split,
            "raw_path": str(raw_path),
            "coverage_path": str(coverage_path),
            "processed_all_path": str(processed_dataset_dir / "all" / f"{split}.json"),
            "case_diff_path": str(output_path),
            "raw_rows": len(raw_rows),
            "sidecar_rows": len(coverage_by_id),
            "processed_all_rows": len(processed_all_by_id),
            "covered_rows": len(covered_ids),
            "covered_weak_rows": len(covered_weak_ids),
            "skipped_missing": skipped_missing,
            "alignment": {key: {"count": len(value), "examples": value[:10]} for key, value in problems.items()},
        }
    )
    return split_summary, diff_rows


def build_diff_row(
    *,
    spec: DatasetSpec,
    split: str,
    raw_row: dict[str, Any],
    coverage_row: dict[str, Any],
    processed_row: dict[str, Any],
    coverage_path: Path,
    in_covered: bool,
    in_covered_weak: bool,
) -> dict[str, Any]:
    event_id = raw_id(raw_row, spec)
    label = str(coverage_row.get("coverage_label") or "")
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid coverage_label={label!r} for {spec.name}/{split}/{event_id}")
    row = {
        "dataset": spec.name,
        "split": split,
        "event_id": event_id,
        "claim": str(raw_row.get("claim") or processed_row.get("claim") or coverage_row.get("claim") or ""),
        "label": raw_row.get("label"),
        "coverage_label": label,
        "coverage_score": coverage_row.get("coverage_score"),
        "weak_score": coverage_row.get("weak_score"),
        "decision_source": coverage_row.get("decision_source"),
        "critical_missing": coverage_row.get("critical_missing") or [],
        "in_all": True,
        "in_covered": bool(in_covered),
        "in_covered_weak": bool(in_covered_weak),
        "top_evidence_preview": top_evidence_preview(coverage_row),
        "source_sidecar": str(coverage_path),
    }
    if spec.id_key != "event_id":
        row[spec.id_key] = raw_row.get(spec.id_key)
    return row


def missing_diff_row(
    *,
    spec: DatasetSpec,
    split: str,
    raw_row: dict[str, Any],
    coverage_path: Path,
    in_covered: bool,
    in_covered_weak: bool,
) -> dict[str, Any]:
    event_id = raw_id(raw_row, spec)
    row = {
        "dataset": spec.name,
        "split": split,
        "event_id": event_id,
        "claim": str(raw_row.get("claim") or ""),
        "label": raw_row.get("label"),
        "coverage_label": "missing",
        "coverage_score": None,
        "weak_score": None,
        "decision_source": "missing",
        "critical_missing": [],
        "in_all": False,
        "in_covered": bool(in_covered),
        "in_covered_weak": bool(in_covered_weak),
        "top_evidence_preview": [],
        "source_sidecar": str(coverage_path),
    }
    if spec.id_key != "event_id":
        row[spec.id_key] = raw_row.get(spec.id_key)
    return row


def summarize_diff_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = count_by(rows, "coverage_label")
    gold_counts = count_by(rows, "label")
    retained_covered = sum(1 for row in rows if row.get("in_covered"))
    retained_covered_weak = sum(1 for row in rows if row.get("in_covered_weak"))
    return {
        "n_cases": len(rows),
        "coverage_counts": label_counts,
        "gold_label_counts": gold_counts,
        "gold_by_coverage": cross_count(rows, row_key="label", col_key="coverage_label"),
        "decision_source_counts": count_by(rows, "decision_source"),
        "retention": {
            "covered": retention_summary(rows, key="in_covered"),
            "covered_weak": retention_summary(rows, key="in_covered_weak"),
        },
        "gold_label_retention": gold_label_retention(rows),
        "critical_missing_top": critical_missing_top(rows),
    }


def retention_summary(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    retained = sum(1 for row in rows if row.get(key))
    total = len(rows)
    return {
        "retained": retained,
        "filtered_out": total - retained,
        "retention_rate": round(float(retained / max(total, 1)), 6),
    }


def gold_label_retention(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("label")), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for label, label_rows in sorted(by_label.items()):
        out[label] = {
            "total": len(label_rows),
            "covered": retention_summary(label_rows, key="in_covered"),
            "covered_weak": retention_summary(label_rows, key="in_covered_weak"),
        }
    return out


def critical_missing_top(rows: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in row.get("critical_missing") or []:
            if item is not None:
                counts[str(item)] += 1
    return [{"anchor": key, "count": count} for key, count in counts.most_common(limit)]


def top_evidence_preview(coverage_row: dict[str, Any], *, limit: int = 3, text_limit: int = 240) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for item in list(coverage_row.get("top_evidence") or [])[: max(limit, 0)]:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        preview.append(
            {
                "rank": item.get("rank"),
                "report_id": item.get("report_id"),
                "sent_idx": item.get("sent_idx"),
                "text": truncate(str(item.get("text") or ""), text_limit),
                "bm25": scores.get("bm25"),
                "lexical": scores.get("lexical"),
                "embedding": scores.get("embedding"),
                "hybrid": scores.get("hybrid"),
                "anchor_hits": item.get("anchor_hits") if isinstance(item.get("anchor_hits"), dict) else {},
            }
        )
    return preview


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at {path}[{idx}]")
        out.append(item)
    return out


def load_coverage_rows(path: Path, *, split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
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


def load_processed_policy(path: Path, *, spec: DatasetSpec) -> dict[str, dict[str, Any]]:
    rows = load_json_list(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        event_id = raw_id(row, spec)
        if event_id in out:
            raise ValueError(f"Duplicate processed id={event_id!r} in {path}")
        out[event_id] = row
    return out


def raw_id(row: dict[str, Any], spec: DatasetSpec) -> str:
    if spec.id_key not in row:
        raise ValueError(f"Missing {spec.id_key!r} in {spec.name} row")
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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def render_combined_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Original vs Coverage-Labeled Dataset Diff",
        "",
        f"- coverage_version: `{summary.get('coverage_version')}`",
        f"- processed_root: `{summary.get('processed_root')}`",
        f"- strict: `{summary.get('strict')}`",
        "",
        "## Datasets",
        "",
        "| dataset | cases | covered | weak_covered | uncovered | covered retention | covered_weak retention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, payload in sorted((summary.get("datasets") or {}).items()):
        overall = payload.get("overall") or {}
        counts = overall.get("coverage_counts") or {}
        retention = overall.get("retention") or {}
        lines.append(
            "| {dataset} | {cases} | {covered} | {weak} | {uncovered} | {covered_ret} | {weak_ret} |".format(
                dataset=dataset,
                cases=overall.get("n_cases", 0),
                covered=counts.get(COVERED, 0),
                weak=counts.get(WEAK_COVERED, 0),
                uncovered=counts.get(UNCOVERED, 0),
                covered_ret=fmt_rate((retention.get("covered") or {}).get("retention_rate")),
                weak_ret=fmt_rate((retention.get("covered_weak") or {}).get("retention_rate")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_dataset_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Original vs Coverage-Labeled Dataset Diff: {summary.get('dataset')}",
        "",
        f"- coverage_version: `{summary.get('coverage_version')}`",
        f"- raw_dir: `{summary.get('raw_dir')}`",
        f"- coverage_dir: `{summary.get('coverage_dir')}`",
        f"- processed_root: `{summary.get('processed_root')}`",
        f"- strict: `{summary.get('strict')}`",
        "",
        "## Split Summary",
        "",
        "| split | raw | sidecar | all | covered | covered_weak | covered retention | covered_weak retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in sorted((summary.get("splits") or {}).items()):
        retention = payload.get("retention") or {}
        lines.append(
            "| {split} | {raw} | {sidecar} | {all_rows} | {covered} | {covered_weak} | {covered_ret} | {weak_ret} |".format(
                split=split,
                raw=payload.get("raw_rows", 0),
                sidecar=payload.get("sidecar_rows", 0),
                all_rows=payload.get("processed_all_rows", 0),
                covered=payload.get("covered_rows", 0),
                covered_weak=payload.get("covered_weak_rows", 0),
                covered_ret=fmt_rate((retention.get("covered") or {}).get("retention_rate")),
                weak_ret=fmt_rate((retention.get("covered_weak") or {}).get("retention_rate")),
            )
        )
    overall = summary.get("overall") or {}
    lines.extend(
        [
            "",
            "## Overall Coverage Counts",
            "",
            "| label | count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted((overall.get("coverage_counts") or {}).items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Top Critical Missing Anchors", "", "| anchor | count |", "|---|---:|"])
    for item in overall.get("critical_missing_top") or []:
        lines.append(f"| {item.get('anchor')} | {item.get('count')} |")
    lines.append("")
    return "\n".join(lines)


def fmt_rate(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def truncate(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
