#!/usr/bin/env python3
"""Compare rendered verifier prompts between two prompt evidence policies."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = (
    "prompt_evidence_selected_count_before_prompt_truncation",
    "evidence_count",
    "evidence_count_before",
    "prompt_evidence_selected_token_cost",
    "prompt_token_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-build-dir", required=True)
    parser.add_argument("--right-build-dir", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-prompt-pairs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_dir = Path(args.left_build_dir)
    right_dir = Path(args.right_build_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]

    report_lines = [
        f"# Prompt Evidence Policy Comparison: {args.left_name} vs {args.right_name}",
        "",
        f"- Left: `{left_dir}`",
        f"- Right: `{right_dir}`",
        f"- Splits: `{', '.join(splits)}`",
        "",
    ]

    for split in splits:
        left_rows = _read_rows(left_dir / f"build_{split}.jsonl")
        right_rows = _read_rows(right_dir / f"build_{split}.jsonl")
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        report_lines.extend(
            _render_split_report(
                split=split,
                left_rows=left_rows,
                right_rows=right_rows,
                left_name=str(args.left_name),
                right_name=str(args.right_name),
                output_dir=split_dir,
                max_prompt_pairs=max(0, int(args.max_prompt_pairs)),
            )
        )

    report_path = output_dir / "comparison_report.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {report_path}")
    return 0


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event_id = str(row.get("event_id") or "")
            if event_id:
                rows[event_id] = row
    return rows


def _render_split_report(
    *,
    split: str,
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
    output_dir: Path,
    max_prompt_pairs: int,
) -> list[str]:
    common_ids = sorted(set(left_rows) & set(right_rows))
    left_only = sorted(set(left_rows) - set(right_rows))
    right_only = sorted(set(right_rows) - set(left_rows))

    lines = [
        f"## {split}",
        "",
        f"- Common rows: {len(common_ids)}",
        f"- {left_name} only: {len(left_only)}",
        f"- {right_name} only: {len(right_only)}",
        "",
        "### Capacity And Variance",
        "",
        "| field | policy | count | mean | variance | std | p50 | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field in NUMERIC_FIELDS:
        for name, rows in ((left_name, left_rows), (right_name, right_rows)):
            values = [_numeric(rows[event_id].get(field)) for event_id in common_ids]
            summary = _summary(values)
            lines.append(
                "| {field} | {name} | {count:.0f} | {mean:.3f} | {variance:.3f} | "
                "{std:.3f} | {p50:.3f} | {p90:.3f} | {max:.3f} |".format(
                    field=field,
                    name=name,
                    **summary,
                )
            )
    lines.extend(["", "### Paired Deltas", ""])
    lines.extend(
        [
            "| field | mean_delta(left-right) | variance_delta(left-right) | p50_delta | p90_delta | min_delta | max_delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for field in NUMERIC_FIELDS:
        deltas = [
            _numeric(left_rows[event_id].get(field)) - _numeric(right_rows[event_id].get(field))
            for event_id in common_ids
        ]
        summary = _summary(deltas)
        lines.append(
            "| {field} | {mean:.3f} | {variance:.3f} | {p50:.3f} | {p90:.3f} | {min:.3f} | {max:.3f} |".format(
                field=field,
                **summary,
            )
        )

    lines.extend(["", "### Stop Reasons", ""])
    lines.extend(
        [
            f"- {left_name}: `{dict(_counter(left_rows, common_ids, 'prompt_evidence_stop_reason'))}`",
            f"- {right_name}: `{dict(_counter(right_rows, common_ids, 'prompt_evidence_stop_reason'))}`",
            "",
            "### Truncation",
            "",
            f"- {left_name}: {_rate(left_rows, common_ids, 'was_truncated'):.4f}",
            f"- {right_name}: {_rate(right_rows, common_ids, 'was_truncated'):.4f}",
            "",
            "### Rendered Prompt Pairs",
            "",
        ]
    )

    examples = _select_examples(left_rows, right_rows, common_ids, max_prompt_pairs)
    if not examples:
        lines.append("- No common rows available.")
        return lines + [""]

    lines.append("| event_id | selected_delta | prompt_token_delta | left_prompt | right_prompt |")
    lines.append("|---|---:|---:|---|---|")
    for event_id in examples:
        left_prompt_path = output_dir / f"{_safe_name(event_id)}__{_safe_name(left_name)}.txt"
        right_prompt_path = output_dir / f"{_safe_name(event_id)}__{_safe_name(right_name)}.txt"
        left_prompt_path.write_text(str(left_rows[event_id].get("prompt") or ""), encoding="utf-8")
        right_prompt_path.write_text(str(right_rows[event_id].get("prompt") or ""), encoding="utf-8")
        selected_delta = (
            _numeric(left_rows[event_id].get("prompt_evidence_selected_count_before_prompt_truncation"))
            - _numeric(right_rows[event_id].get("prompt_evidence_selected_count_before_prompt_truncation"))
        )
        prompt_token_delta = (
            _numeric(left_rows[event_id].get("prompt_token_count"))
            - _numeric(right_rows[event_id].get("prompt_token_count"))
        )
        lines.append(
            f"| `{event_id}` | {selected_delta:.0f} | {prompt_token_delta:.0f} | "
            f"[{left_prompt_path.name}]({split_relative(left_prompt_path, output_dir.parent)}) | "
            f"[{right_prompt_path.name}]({split_relative(right_prompt_path, output_dir.parent)}) |"
        )
    lines.append("")
    return lines


def _select_examples(
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    common_ids: list[str],
    limit: int,
) -> list[str]:
    ranked = []
    for event_id in common_ids:
        selected_delta = (
            _numeric(left_rows[event_id].get("prompt_evidence_selected_count_before_prompt_truncation"))
            - _numeric(right_rows[event_id].get("prompt_evidence_selected_count_before_prompt_truncation"))
        )
        prompt_token_delta = (
            _numeric(left_rows[event_id].get("prompt_token_count"))
            - _numeric(right_rows[event_id].get("prompt_token_count"))
        )
        ranked.append((abs(selected_delta), abs(prompt_token_delta), event_id))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [event_id for _, _, event_id in ranked[:limit]]


def _counter(rows: dict[str, dict[str, Any]], event_ids: list[str], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for event_id in event_ids:
        counter[str(rows[event_id].get(field) or "")] += 1
    return counter


def _rate(rows: dict[str, dict[str, Any]], event_ids: list[str], field: str) -> float:
    if not event_ids:
        return 0.0
    return float(sum(1 for event_id in event_ids if bool(rows[event_id].get(field))) / len(event_ids))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
        }
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    return {
        "count": float(count),
        "min": float(ordered[0]),
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "max": float(ordered[-1]),
        "mean": float(mean),
        "variance": float(variance),
        "std": float(math.sqrt(variance)),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_name(value: str) -> str:
    rendered = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            rendered.append(char)
        else:
            rendered.append("_")
    return "".join(rendered).strip("_") or "row"


def split_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
