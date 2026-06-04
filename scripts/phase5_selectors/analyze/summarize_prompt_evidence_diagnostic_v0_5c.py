#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = (
    "outputs/selectors/evidence_map_selector/"
    "v0_5c_val_prompt_evidence_diagnostic"
)
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
REQUESTED_CASE_IDS = ["4855.json", "11447.json", "10443.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize v0.5c prompt x evidence verifier diagnostics."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--primary-metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy", "selection_score", "true_side_macro_f1"],
    )
    parser.add_argument("--case-limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = _collect_eval_rows(output_dir=output_dir, split=str(args.split))
    build_reports = _collect_build_reports(output_dir=output_dir)
    build_rows = _collect_build_rows(output_dir=output_dir, split=str(args.split))
    paired_deltas = _paired_prompt_deltas(build_rows)
    label_rows, prediction_bundles = _collect_label_shift_rows(
        output_dir=output_dir,
        split=str(args.split),
        build_rows=build_rows,
    )
    truncation_rows = _truncation_rows(build_reports)
    decisions = _decision_labels(
        eval_rows=eval_rows,
        build_reports=build_reports,
        primary_metric=str(args.primary_metric),
    )

    comparison_json = {
        "status": "completed" if eval_rows or build_reports else "no_artifacts_found",
        "output_dir": str(output_dir),
        "split": str(args.split),
        "primary_metric": str(args.primary_metric),
        "n_eval_rows": len(eval_rows),
        "decision_labels": decisions,
        "eval_rows": eval_rows,
        "build_reports": build_reports,
    }
    _write_json(analysis_dir / "comparison_table.json", comparison_json)
    _write_csv(analysis_dir / "comparison_table.csv", eval_rows, _comparison_headers())
    _write_csv(analysis_dir / "truncation_report.csv", truncation_rows, _truncation_headers())
    _write_csv(analysis_dir / "label_shift_report.csv", label_rows, _label_shift_headers())
    _write_jsonl(analysis_dir / "paired_prompt_delta_by_event.jsonl", paired_deltas)
    (analysis_dir / "case_studies.md").write_text(
        _render_case_studies(
            build_rows=build_rows,
            prediction_bundles=prediction_bundles,
            paired_deltas=paired_deltas,
            case_limit=int(args.case_limit),
        ),
        encoding="utf-8",
    )
    (analysis_dir / "analysis_summary.md").write_text(
        _render_markdown(
            output_dir=output_dir,
            split=str(args.split),
            primary_metric=str(args.primary_metric),
            eval_rows=eval_rows,
            build_reports=build_reports,
            truncation_rows=truncation_rows,
            decisions=decisions,
        ),
        encoding="utf-8",
    )
    print(f"Wrote v0.5c diagnostic summary: {analysis_dir / 'analysis_summary.md'}")


def _collect_eval_rows(*, output_dir: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_root = output_dir / "eval"
    for metrics_path in sorted(eval_root.glob("*/*/*/metrics.json")):
        rel = metrics_path.relative_to(eval_root).parts
        if len(rel) != 4:
            continue
        evidence_source, prompt_style, checkpoint, _ = rel
        metrics = _read_json(metrics_path)
        confusion = _read_json(metrics_path.parent / "confusion_matrix.json")
        predictions = _read_jsonl(metrics_path.parent / f"{split}_predictions.jsonl")
        pred_counts = Counter(str(row.get("pred_label") or "") for row in predictions)
        gold_counts = Counter(str(row.get("gold_label") or "") for row in predictions)
        row: dict[str, Any] = {
            "evidence_source": evidence_source,
            "prompt_style": prompt_style,
            "checkpoint": checkpoint,
            "metrics_path": str(metrics_path),
            "predictions_path": str(metrics_path.parent / f"{split}_predictions.jsonl"),
            "num_samples": _as_int(metrics.get("num_samples")),
            "accuracy": _as_float(metrics.get("accuracy")),
            "macro_precision": _as_float(metrics.get("macro_precision")),
            "macro_recall": _as_float(metrics.get("macro_recall")),
            "macro_f1": _as_float(metrics.get("macro_f1")),
            "true_side_macro_f1": _as_float(metrics.get("true_side_macro_f1")),
            "selection_score": _as_float(metrics.get("selection_score")),
            "parse_error_rate": _as_float(metrics.get("parse_error_rate")),
            "eval_loss": _as_float(metrics.get("eval_loss")),
            "prediction_label_distribution": dict(pred_counts),
            "gold_label_distribution": dict(gold_counts),
            "confusion_matrix_path": str(metrics_path.parent / "confusion_matrix.json")
            if confusion
            else "",
        }
        per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
        for label in LABELS:
            label_metrics = per_class.get(label, {}) if isinstance(per_class, dict) else {}
            row[f"f1_{label}"] = _as_float(label_metrics.get("f1")) if isinstance(label_metrics, dict) else None
        rows.append(row)
    return sorted(rows, key=lambda row: (row["evidence_source"], row["prompt_style"], row["checkpoint"]))


def _collect_build_reports(*, output_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    root = output_dir / "verifier_data"
    report_paths = list(root.glob("*/*/build_report.json"))
    report_paths.extend(root.glob("*/*/build/build_report.json"))
    for report_path in sorted(set(report_paths)):
        if report_path.parent.name == "build":
            evidence_source = report_path.parents[2].name
            prompt_style = report_path.parents[1].name
        else:
            evidence_source = report_path.parents[1].name
            prompt_style = report_path.parent.name
        key = f"{evidence_source}/{prompt_style}"
        report = _read_json(report_path)
        reports[key] = {
            "evidence_source": evidence_source,
            "prompt_style": prompt_style,
            "path": str(report_path),
            "n_rows": report.get("n_rows"),
            "prompt_token_count": report.get("prompt_token_count") or {},
            "target_token_count": report.get("target_token_count") or {},
            "evidence_count": report.get("evidence_count") or {},
            "evidence_count_before": report.get("evidence_count_before") or {},
            "was_truncated_rate": _as_float(report.get("was_truncated_rate")),
            "evidence_dropped_rate": _as_float(report.get("evidence_dropped_rate")),
            "evidence_text_truncated_rate": _as_float(report.get("evidence_text_truncated_rate")),
            "overflow_after_rate": _as_float(report.get("overflow_after_rate")),
            "selection_metrics": report.get("selection_metrics") or {},
            "map_annotation_status": report.get("map_annotation_status") or {},
        }
    return reports


def _collect_build_rows(*, output_dir: Path, split: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for build_path in sorted((output_dir / "verifier_data").glob(f"*/*/build_{split}.jsonl")):
        rows[(build_path.parents[1].name, build_path.parent.name)] = _read_jsonl(build_path)
    return rows


def _paired_prompt_deltas(build_rows: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (source, style), rows in build_rows.items():
        for row in rows:
            by_source[source].setdefault(str(row.get("event_id") or ""), {})[style] = row
    deltas: list[dict[str, Any]] = []
    for source, by_event in sorted(by_source.items()):
        for event_id, styles in sorted(by_event.items()):
            plain = styles.get("plain_original")
            for style, row in sorted(styles.items()):
                if style == "plain_original" or plain is None:
                    continue
                plain_trace = plain.get("selector_trace") or {}
                row_trace = row.get("selector_trace") or {}
                deltas.append(
                    {
                        "event_id": event_id,
                        "evidence_source": source,
                        "prompt_style": style,
                        "prompt_token_delta_vs_plain": int(row.get("prompt_token_count") or 0)
                        - int(plain.get("prompt_token_count") or 0),
                        "evidence_count_delta_vs_plain": int(row.get("evidence_count") or 0)
                        - int(plain.get("evidence_count") or 0),
                        "plain_prompt_token_count": int(plain.get("prompt_token_count") or 0),
                        "style_prompt_token_count": int(row.get("prompt_token_count") or 0),
                        "plain_evidence_count": int(plain.get("evidence_count") or 0),
                        "style_evidence_count": int(row.get("evidence_count") or 0),
                        "plain_was_truncated": bool(plain.get("was_truncated")),
                        "style_was_truncated": bool(row.get("was_truncated")),
                        "selected_texts_identical_before_truncation": list(
                            plain_trace.get("selected_texts_before") or []
                        )
                        == list(row_trace.get("selected_texts_before") or []),
                        "selected_keys_identical_before_truncation": list(
                            plain_trace.get("selected_keys_before") or []
                        )
                        == list(row_trace.get("selected_keys_before") or []),
                    }
                )
    return deltas


def _collect_label_shift_rows(
    *,
    output_dir: Path,
    split: str,
    build_rows: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, dict[str, Any]]]]:
    label_rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    eval_root = output_dir / "eval"
    for pred_path in sorted(eval_root.glob(f"*/*/*/{split}_predictions.jsonl")):
        rel = pred_path.relative_to(eval_root).parts
        if len(rel) != 4:
            continue
        evidence_source, prompt_style, checkpoint, _ = rel
        build = build_rows.get((evidence_source, prompt_style), [])
        by_event: dict[str, dict[str, Any]] = {}
        for record in _read_jsonl(pred_path):
            sample_idx = _as_int(record.get("sample_idx"))
            event_id = ""
            if sample_idx is not None and 0 <= sample_idx < len(build):
                event_id = str(build[sample_idx].get("event_id") or "")
                record = {**record, "event_id": event_id, "claim": build[sample_idx].get("claim", "")}
            if event_id:
                by_event[event_id] = record
        bundles[(evidence_source, prompt_style, checkpoint)] = by_event
        pred_counts = Counter(str(row.get("pred_label") or "") for row in by_event.values())
        gold_counts = Counter(str(row.get("gold_label") or "") for row in by_event.values())
        base = {
            "evidence_source": evidence_source,
            "prompt_style": prompt_style,
            "checkpoint": checkpoint,
            "n_predictions": len(by_event),
        }
        for label in LABELS:
            base[f"pred_{label}"] = pred_counts.get(label, 0)
            base[f"gold_{label}"] = gold_counts.get(label, 0)
            base[f"pred_rate_{label}"] = pred_counts.get(label, 0) / max(len(by_event), 1)
        label_rows.append(base)
    return label_rows, bundles


def _truncation_rows(build_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for report in build_reports.values():
        by_source[str(report["evidence_source"])][str(report["prompt_style"])] = report
    for source, styles in sorted(by_source.items()):
        plain = styles.get("plain_original")
        for style, report in sorted(styles.items()):
            prompt_stats = report.get("prompt_token_count") or {}
            evidence_stats = report.get("evidence_count") or {}
            row = {
                "evidence_source": source,
                "prompt_style": style,
                "n_rows": report.get("n_rows"),
                "prompt_token_mean": _stat(prompt_stats, "mean"),
                "prompt_token_p95": _stat(prompt_stats, "p95"),
                "prompt_token_max": _stat(prompt_stats, "max"),
                "target_token_mean": _stat(report.get("target_token_count") or {}, "mean"),
                "target_token_max": _stat(report.get("target_token_count") or {}, "max"),
                "evidence_count_mean": _stat(evidence_stats, "mean"),
                "evidence_count_min": _stat(evidence_stats, "min"),
                "evidence_count_before_mean": _stat(report.get("evidence_count_before") or {}, "mean"),
                "was_truncated_rate": report.get("was_truncated_rate"),
                "evidence_dropped_rate": report.get("evidence_dropped_rate"),
                "evidence_text_truncated_rate": report.get("evidence_text_truncated_rate"),
                "overflow_after_rate": report.get("overflow_after_rate"),
                "plain_vs_map_token_delta": None,
                "plain_vs_map_evidence_count_delta": None,
            }
            if plain and style != "plain_original":
                row["plain_vs_map_token_delta"] = _stat(prompt_stats, "mean") - _stat(
                    plain.get("prompt_token_count") or {}, "mean"
                )
                row["plain_vs_map_evidence_count_delta"] = _stat(evidence_stats, "mean") - _stat(
                    plain.get("evidence_count") or {}, "mean"
                )
            selection_metrics = report.get("selection_metrics") or {}
            for key in (
                "recall@5",
                "jaccard@5",
                "top1_match",
                "oracle_rank_ndcg@5",
                "weighted_atom_coverage@5",
                "direct_or_partial_map_rate@5",
                "background_only_map_rate@5",
            ):
                row[key] = _as_float(selection_metrics.get(key))
            rows.append(row)
    return rows


def _decision_labels(
    *,
    eval_rows: list[dict[str, Any]],
    build_reports: dict[str, dict[str, Any]],
    primary_metric: str,
) -> list[str]:
    decisions: set[str] = set()
    row_by_key = {
        (row["evidence_source"], row["prompt_style"], row["checkpoint"]): row
        for row in eval_rows
    }
    for checkpoint in sorted({row["checkpoint"] for row in eval_rows}):
        plain = row_by_key.get(("oracle_top5", "plain_original", checkpoint))
        mapped = row_by_key.get(("oracle_top5", "map_full", checkpoint))
        if plain and mapped and _metric(mapped, primary_metric) <= _metric(plain, primary_metric) - 0.05:
            decisions.add("PROMPT_OOD")
    for row in eval_rows:
        if row["evidence_source"] == "oracle_top5":
            continue
        oracle = row_by_key.get(("oracle_top5", row["prompt_style"], row["checkpoint"]))
        if oracle and _metric(row, primary_metric) <= _metric(oracle, primary_metric) - 0.05:
            decisions.add("EVIDENCE_GAP")
    by_source_checkpoint: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in eval_rows:
        by_source_checkpoint[(row["evidence_source"], row["checkpoint"])][row["prompt_style"]] = row
    for styles in by_source_checkpoint.values():
        plain = styles.get("plain_original")
        mapped = styles.get("map_full")
        if plain and mapped and _metric(mapped, primary_metric) <= _metric(plain, primary_metric) - 0.05:
            decisions.add("RENDERING_GAP")
    for source in {report["evidence_source"] for report in build_reports.values()}:
        plain = build_reports.get(f"{source}/plain_original")
        mapped = build_reports.get(f"{source}/map_full")
        if not plain or not mapped:
            continue
        evidence_delta = _stat(mapped.get("evidence_count") or {}, "mean") - _stat(
            plain.get("evidence_count") or {}, "mean"
        )
        trunc_delta = float(mapped.get("was_truncated_rate") or 0.0) - float(plain.get("was_truncated_rate") or 0.0)
        if evidence_delta <= -1.0 or trunc_delta >= 0.20:
            decisions.add("TRUNCATION_GAP")
    by_source_style: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in eval_rows:
        by_source_style[(row["evidence_source"], row["prompt_style"])].append(_metric(row, primary_metric))
    for values in by_source_style.values():
        finite = [value for value in values if math.isfinite(value)]
        if finite and max(finite) - min(finite) > 0.03:
            decisions.add("CHECKPOINT_SENSITIVITY")
    return sorted(decisions) or ["PENDING_EVAL"]


def _render_markdown(
    *,
    output_dir: Path,
    split: str,
    primary_metric: str,
    eval_rows: list[dict[str, Any]],
    build_reports: dict[str, dict[str, Any]],
    truncation_rows: list[dict[str, Any]],
    decisions: list[str],
) -> str:
    lines = [
        "# v0.5c Prompt x Evidence Diagnostic",
        "",
        f"- output_dir: `{output_dir}`",
        f"- split: `{split}`",
        f"- primary_metric: `{primary_metric}`",
        f"- n_eval_rows: `{len(eval_rows)}`",
        f"- decision_labels: `{', '.join(decisions)}`",
        "",
        "## Classification Matrix",
        "",
    ]
    if eval_rows:
        lines.extend(
            [
                "| evidence_source | prompt_style | checkpoint | acc | macro-F1 | true-side F1 | selection | n |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(eval_rows, key=lambda item: (item["checkpoint"], item["evidence_source"], item["prompt_style"])):
            lines.append(
                "| {source} | {style} | {ckpt} | {acc:.4f} | {f1:.4f} | {tf1:.4f} | {sel:.4f} | {n} |".format(
                    source=row["evidence_source"],
                    style=row["prompt_style"],
                    ckpt=row["checkpoint"],
                    acc=_metric(row, "accuracy"),
                    f1=_metric(row, "macro_f1"),
                    tf1=_metric(row, "true_side_macro_f1"),
                    sel=_metric(row, "selection_score"),
                    n=row.get("num_samples", ""),
                )
            )
    else:
        lines.append("(no eval metrics found yet)")
    lines.extend(["", "## Prompt Delta", ""])
    prompt_delta_rows = _prompt_delta_rows(eval_rows=eval_rows, build_reports=build_reports, primary_metric=primary_metric)
    if prompt_delta_rows:
        lines.extend(
            [
                "| evidence_source | checkpoint | map-plain primary | map-plain acc | map-plain tokens | map-plain evidence_count |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prompt_delta_rows:
            lines.append(
                "| {source} | {ckpt} | {metric:.4f} | {acc:.4f} | {tok:.2f} | {ev:.2f} |".format(
                    source=row["evidence_source"],
                    ckpt=row["checkpoint"],
                    metric=float(row["primary_delta"]),
                    acc=float(row["accuracy_delta"]),
                    tok=float(row["token_delta"]),
                    ev=float(row["evidence_count_delta"]),
                )
            )
    else:
        lines.append("(needs both plain_original and map_full rows)")
    lines.extend(["", "## Selector Gap To Oracle", ""])
    selector_gap_rows = _selector_gap_rows(eval_rows=eval_rows, primary_metric=primary_metric)
    if selector_gap_rows:
        lines.extend(
            [
                "| evidence_source | prompt_style | checkpoint | selector-oracle primary | selector-oracle acc |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in selector_gap_rows:
            lines.append(
                "| {source} | {style} | {ckpt} | {metric:.4f} | {acc:.4f} |".format(
                    source=row["evidence_source"],
                    style=row["prompt_style"],
                    ckpt=row["checkpoint"],
                    metric=float(row["primary_delta"]),
                    acc=float(row["accuracy_delta"]),
                )
            )
    else:
        lines.append("(needs oracle_top5 and selector eval rows)")
    lines.extend(["", "## Truncation And Evidence Count", ""])
    if truncation_rows:
        lines.extend(
            [
                "| evidence_source | prompt_style | token_mean | token_p95 | evidence_mean | trunc_rate | dropped_rate | token_delta_vs_plain | evidence_delta_vs_plain |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in truncation_rows:
            lines.append(
                "| {source} | {style} | {tok:.2f} | {p95:.2f} | {ev:.2f} | {tr:.4f} | {drop:.4f} | {td} | {ed} |".format(
                    source=row["evidence_source"],
                    style=row["prompt_style"],
                    tok=float(row.get("prompt_token_mean") or 0.0),
                    p95=float(row.get("prompt_token_p95") or 0.0),
                    ev=float(row.get("evidence_count_mean") or 0.0),
                    tr=float(row.get("was_truncated_rate") or 0.0),
                    drop=float(row.get("evidence_dropped_rate") or 0.0),
                    td=_fmt_optional(row.get("plain_vs_map_token_delta")),
                    ed=_fmt_optional(row.get("plain_vs_map_evidence_count_delta")),
                )
            )
    else:
        lines.append("(no build reports found yet)")
    lines.extend(
        [
            "",
            "## Route Decision",
            "",
            _decision_sentence(decisions),
            "",
        ]
    )
    return "\n".join(lines)


def _render_case_studies(
    *,
    build_rows: dict[tuple[str, str], list[dict[str, Any]]],
    prediction_bundles: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    paired_deltas: list[dict[str, Any]],
    case_limit: int,
) -> str:
    candidates: list[tuple[str, str]] = []
    for event_id in REQUESTED_CASE_IDS:
        candidates.append((event_id, "fixed_case"))
    for delta in sorted(paired_deltas, key=lambda row: int(row.get("prompt_token_delta_vs_plain") or 0), reverse=True):
        event_id = str(delta.get("event_id") or "")
        if event_id and all(event_id != item[0] for item in candidates):
            candidates.append((event_id, "large_prompt_delta"))
        if len(candidates) >= case_limit:
            break
    for bundle_key, bundle in prediction_bundles.items():
        source, style, checkpoint = bundle_key
        if style != "plain_original":
            continue
        mapped = prediction_bundles.get((source, "map_full", checkpoint), {})
        for event_id, pred in bundle.items():
            map_pred = mapped.get(event_id)
            if not map_pred:
                continue
            if _is_correct(pred) and not _is_correct(map_pred) and all(event_id != item[0] for item in candidates):
                candidates.append((event_id, "plain_correct_map_wrong"))
            if len(candidates) >= case_limit:
                break
        if len(candidates) >= case_limit:
            break

    lines = ["# v0.5c Prompt x Evidence Case Studies", ""]
    if not candidates:
        lines.extend(["(no build rows found)", ""])
        return "\n".join(lines)
    for event_id, reason in candidates[:case_limit]:
        payload = _case_payload(event_id, build_rows, prediction_bundles)
        if not payload:
            continue
        lines.extend(_case_lines(event_id, reason, payload))
    return "\n".join(lines)


def _case_payload(
    event_id: str,
    build_rows: dict[tuple[str, str], list[dict[str, Any]]],
    prediction_bundles: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    build_matches: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in build_rows.items():
        for row in rows:
            if str(row.get("event_id") or "") == event_id:
                build_matches[key] = row
                break
    if not build_matches:
        return None
    pred_matches: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, bundle in prediction_bundles.items():
        if event_id in bundle:
            pred_matches[key] = bundle[event_id]
    first = next(iter(build_matches.values()))
    return {"claim": first.get("claim", ""), "gold_label": first.get("gold_label", ""), "build": build_matches, "pred": pred_matches}


def _case_lines(event_id: str, reason: str, payload: dict[str, Any]) -> list[str]:
    lines = [
        f"## {event_id}",
        "",
        f"- reason: `{reason}`",
        f"- gold_label: `{payload.get('gold_label', '')}`",
        f"- claim: {payload.get('claim', '')}",
        "",
    ]
    if payload.get("pred"):
        lines.extend(
            [
                "### Predictions",
                "",
                "| evidence_source | prompt_style | checkpoint | pred | correct |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for (source, style, checkpoint), pred in sorted(payload["pred"].items()):
            lines.append(
                f"| {source} | {style} | {checkpoint} | {pred.get('pred_label', '')} | {str(_is_correct(pred)).lower()} |"
            )
        lines.append("")
    lines.extend(
        [
            "### Prompt Stats",
            "",
            "| evidence_source | prompt_style | prompt_tokens | evidence_count | before | truncated |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for (source, style), row in sorted(payload["build"].items()):
        lines.append(
            "| {source} | {style} | {tokens} | {count} | {before} | {trunc} |".format(
                source=source,
                style=style,
                tokens=row.get("prompt_token_count", ""),
                count=row.get("evidence_count", ""),
                before=row.get("evidence_count_before", ""),
                trunc=str(bool(row.get("was_truncated"))).lower(),
            )
        )
    source_style, row = next(iter(sorted(payload["build"].items())))
    lines.extend(["", "### Selected Evidence", ""])
    for idx, candidate in enumerate(row.get("candidates") or [], start=1):
        lines.append(
            "- {idx}. {relation}/{directness} atoms={atoms} text={text}".format(
                idx=idx,
                relation=candidate.get("map_relation", ""),
                directness=candidate.get("map_directness", ""),
                atoms=candidate.get("covered_atom_ids", []),
                text=str(candidate.get("text") or "")[:260],
            )
        )
    lines.extend(["", "### Prompt Excerpt", "", "```text", str(row.get("prompt") or "")[:1200], "```", ""])
    return lines


def _prompt_delta_rows(
    *,
    eval_rows: list[dict[str, Any]],
    build_reports: dict[str, dict[str, Any]],
    primary_metric: str,
) -> list[dict[str, Any]]:
    row_by_key = {
        (row["evidence_source"], row["prompt_style"], row["checkpoint"]): row
        for row in eval_rows
    }
    out: list[dict[str, Any]] = []
    for source in sorted({row["evidence_source"] for row in eval_rows}):
        for checkpoint in sorted({row["checkpoint"] for row in eval_rows}):
            plain = row_by_key.get((source, "plain_original", checkpoint))
            mapped = row_by_key.get((source, "map_full", checkpoint))
            if not plain or not mapped:
                continue
            plain_report = build_reports.get(f"{source}/plain_original", {})
            mapped_report = build_reports.get(f"{source}/map_full", {})
            out.append(
                {
                    "evidence_source": source,
                    "checkpoint": checkpoint,
                    "primary_delta": _metric(mapped, primary_metric) - _metric(plain, primary_metric),
                    "accuracy_delta": _metric(mapped, "accuracy") - _metric(plain, "accuracy"),
                    "token_delta": _stat(mapped_report.get("prompt_token_count") or {}, "mean")
                    - _stat(plain_report.get("prompt_token_count") or {}, "mean"),
                    "evidence_count_delta": _stat(mapped_report.get("evidence_count") or {}, "mean")
                    - _stat(plain_report.get("evidence_count") or {}, "mean"),
                }
            )
    return out


def _selector_gap_rows(*, eval_rows: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    row_by_key = {
        (row["evidence_source"], row["prompt_style"], row["checkpoint"]): row
        for row in eval_rows
    }
    out: list[dict[str, Any]] = []
    for row in eval_rows:
        if row["evidence_source"] == "oracle_top5":
            continue
        oracle = row_by_key.get(("oracle_top5", row["prompt_style"], row["checkpoint"]))
        if not oracle:
            continue
        out.append(
            {
                "evidence_source": row["evidence_source"],
                "prompt_style": row["prompt_style"],
                "checkpoint": row["checkpoint"],
                "primary_delta": _metric(row, primary_metric) - _metric(oracle, primary_metric),
                "accuracy_delta": _metric(row, "accuracy") - _metric(oracle, "accuracy"),
            }
        )
    return out


def _decision_sentence(decisions: list[str]) -> str:
    if decisions == ["PENDING_EVAL"]:
        return "v0.5c build artifacts are present, but classification eval is still pending; run the 20-job matrix before route selection."
    if "PROMPT_OOD" in decisions:
        return "v0.5b 的低分至少包含 prompt OOD / rendering gap；下一步优先进入 map_minimal prompt ablation 或 map-aware verifier training。"
    if "EVIDENCE_GAP" in decisions:
        return "v0.5b 的低分主要指向 selector evidence gap；下一步优先进入 learned map-feature fusion 或 set-level utility distillation。"
    if "TRUNCATION_GAP" in decisions:
        return "v0.5b 的低分包含 truncation gap；下一步优先压缩 map prompt 并保留 evidence text。"
    return "v0.5c 未触发单一强路线标签；请结合 case_studies.md 与 paired prompt deltas 做人工复核。"


def _comparison_headers() -> list[str]:
    return [
        "evidence_source",
        "prompt_style",
        "checkpoint",
        "num_samples",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "true_side_macro_f1",
        "selection_score",
        "parse_error_rate",
        "eval_loss",
        *[f"f1_{label}" for label in LABELS],
        "metrics_path",
        "predictions_path",
    ]


def _truncation_headers() -> list[str]:
    return [
        "evidence_source",
        "prompt_style",
        "n_rows",
        "prompt_token_mean",
        "prompt_token_p95",
        "prompt_token_max",
        "target_token_mean",
        "target_token_max",
        "evidence_count_mean",
        "evidence_count_min",
        "evidence_count_before_mean",
        "was_truncated_rate",
        "evidence_dropped_rate",
        "evidence_text_truncated_rate",
        "overflow_after_rate",
        "plain_vs_map_token_delta",
        "plain_vs_map_evidence_count_delta",
        "recall@5",
        "jaccard@5",
        "top1_match",
        "oracle_rank_ndcg@5",
        "weighted_atom_coverage@5",
        "direct_or_partial_map_rate@5",
        "background_only_map_rate@5",
    ]


def _label_shift_headers() -> list[str]:
    return [
        "evidence_source",
        "prompt_style",
        "checkpoint",
        "n_predictions",
        *[f"pred_{label}" for label in LABELS],
        *[f"gold_{label}" for label in LABELS],
        *[f"pred_rate_{label}" for label in LABELS],
    ]


def _is_correct(record: dict[str, Any]) -> bool:
    return str(record.get("pred_label") or "") == str(record.get("gold_label") or "")


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(out):
        return float("-inf")
    return out


def _stat(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key) if isinstance(stats, dict) else None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out):
        return 0.0
    return out


def _fmt_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
