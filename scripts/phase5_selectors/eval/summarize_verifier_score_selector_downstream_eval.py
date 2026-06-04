#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "outputs/selectors/verifier_score_selector/b3_oracle_direct_v0/downstream_verifier_eval"
DEFAULT_DIRECT_VERIFIER_RUN_DIR = (
    "outputs/oracle_direct_verifier/stage2_sentence/train/"
    "b3_oracle_sentence_direct_verifier_1024_20260519-200709"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize downstream classifier F1 for verifier-score selector traces."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--primary-metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy", "selection_score", "true_side_macro_f1"],
    )
    parser.add_argument("--direct-verifier-run-dir", default=DEFAULT_DIRECT_VERIFIER_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    eval_rows = _collect_eval_rows(output_dir)
    build_reports = _collect_build_reports(output_dir)
    prior = _collect_original_checkpoint_metrics(Path(args.direct_verifier_run_dir))
    rows_sorted = sorted(
        eval_rows,
        key=lambda row: (
            -_metric(row, args.primary_metric),
            -_metric(row, "macro_f1"),
            -_metric(row, "accuracy"),
            str(row.get("selector_slug") or ""),
            str(row.get("checkpoint") or ""),
        ),
    )
    payload = {
        "status": "completed" if rows_sorted else "no_eval_metrics_found",
        "output_dir": str(output_dir),
        "split": str(args.split),
        "primary_metric": str(args.primary_metric),
        "n_eval_rows": int(len(rows_sorted)),
        "best_overall": rows_sorted[0] if rows_sorted else None,
        "best_by_selector": _best_by(rows_sorted, key="selector_slug", metric=str(args.primary_metric)),
        "best_by_checkpoint": _best_by(rows_sorted, key="checkpoint", metric=str(args.primary_metric)),
        "eval_rows": rows_sorted,
        "build_reports": build_reports,
        "oracle_direct_checkpoint_prior": prior,
        "notes": [
            "Rows evaluate final LIAR-RAW classification with fixed oracle-direct verifier checkpoints.",
            "Selector traces are converted to verifier prompts by build_trace_verifier_data.py.",
            "Primary deployment comparison should use macro_f1 and parse_error_rate, not selector overlap alone.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "downstream_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_summary.md").write_text(_render_markdown(payload), encoding="utf-8")
    if rows_sorted:
        best = rows_sorted[0]
        print(
            "Best downstream verifier eval: "
            f"selector={best['selector_slug']} checkpoint={best['checkpoint']} "
            f"{args.primary_metric}={_metric(best, args.primary_metric):.4f} "
            f"accuracy={_metric(best, 'accuracy'):.4f}"
        )
    else:
        print(f"No downstream eval metrics found under {output_dir}/eval")


def _collect_eval_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((output_dir / "eval").glob("*/*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append(
                {
                    "selector_slug": metrics_path.parents[1].name,
                    "checkpoint": metrics_path.parent.name,
                    "metrics_path": str(metrics_path),
                    "error": f"failed_to_read_metrics: {exc}",
                }
            )
            continue
        rows.append(
            {
                "selector_slug": metrics_path.parents[1].name,
                "checkpoint": metrics_path.parent.name,
                "metrics_path": str(metrics_path),
                "num_samples": _as_int(metrics.get("num_samples")),
                "accuracy": _as_float(metrics.get("accuracy")),
                "macro_precision": _as_float(metrics.get("macro_precision")),
                "macro_recall": _as_float(metrics.get("macro_recall")),
                "macro_f1": _as_float(metrics.get("macro_f1")),
                "true_side_macro_f1": _as_float(metrics.get("true_side_macro_f1")),
                "selection_score": _as_float(metrics.get("selection_score")),
                "parse_error_rate": _as_float(metrics.get("parse_error_rate")),
                "eval_loss": _as_float(metrics.get("eval_loss")),
            }
        )
    return rows


def _collect_build_reports(output_dir: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    report_paths = list((output_dir / "verifier_data").glob("*/build_report.json"))
    report_paths.extend((output_dir / "verifier_data").glob("*/build/build_report.json"))
    for report_path in sorted(set(report_paths)):
        selector_slug = report_path.parent.parent.name if report_path.parent.name == "build" else report_path.parent.name
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            reports[selector_slug] = {"path": str(report_path), "error": str(exc)}
            continue
        split_reports = report.get("splits") if isinstance(report.get("splits"), dict) else {}
        val_report = split_reports.get("val") or next(iter(split_reports.values()), {})
        metrics = val_report.get("selection_metrics") if isinstance(val_report, dict) else {}
        reports[selector_slug] = {
            "path": str(report_path),
            "expected_selector_name": report.get("expected_selector_name"),
            "n_rows": _as_int(val_report.get("n_rows")),
            "prompt_truncation_rate": _as_float(val_report.get("prompt_truncation_rate")),
            "prompt_token_count": val_report.get("prompt_token_count"),
            "evidence_count": val_report.get("evidence_count"),
            "selection_recall@5": _as_float(metrics.get("recall@5") if isinstance(metrics, dict) else None),
            "selection_jaccard@5": _as_float(metrics.get("jaccard@5") if isinstance(metrics, dict) else None),
            "selection_top1_match": _as_float(metrics.get("top1_match") if isinstance(metrics, dict) else None),
            "selection_oracle_rank_ndcg@5": _as_float(
                metrics.get("oracle_rank_ndcg@5") if isinstance(metrics, dict) else None
            ),
        }
    return reports


def _collect_original_checkpoint_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((run_dir / "eval").glob("step-*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        step_name = metrics_path.parent.name
        rows.append(
            {
                "checkpoint": step_name.replace("step-", "checkpoint-"),
                "step": step_name,
                "metrics_path": str(metrics_path),
                "accuracy": _as_float(metrics.get("accuracy")),
                "macro_f1": _as_float(metrics.get("macro_f1")),
                "true_side_macro_f1": _as_float(metrics.get("true_side_macro_f1")),
                "selection_score": _as_float(metrics.get("selection_score")),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -_metric(row, "selection_score"),
            -_metric(row, "macro_f1"),
            str(row.get("checkpoint") or ""),
        ),
    )


def _best_by(rows: list[dict[str, Any]], *, key: str, metric: str) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        current = best.get(value)
        if current is None or (
            _metric(row, metric),
            _metric(row, "macro_f1"),
            _metric(row, "accuracy"),
        ) > (
            _metric(current, metric),
            _metric(current, "macro_f1"),
            _metric(current, "accuracy"),
        ):
            best[value] = row
    return best


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Verifier-Score Selector Downstream Eval",
        "",
        f"- status: `{payload['status']}`",
        f"- output_dir: `{payload['output_dir']}`",
        f"- primary_metric: `{payload['primary_metric']}`",
        f"- n_eval_rows: `{payload['n_eval_rows']}`",
        "",
    ]
    best = payload.get("best_overall")
    if best:
        lines.extend(
            [
                "## Best Overall",
                "",
                (
                    f"- selector: `{best['selector_slug']}`; checkpoint: `{best['checkpoint']}`; "
                    f"accuracy={_metric(best, 'accuracy'):.4f}; macro_f1={_metric(best, 'macro_f1'):.4f}; "
                    f"true_side_macro_f1={_metric(best, 'true_side_macro_f1'):.4f}; "
                    f"parse_error_rate={_metric(best, 'parse_error_rate'):.4f}"
                ),
                "",
            ]
        )

    build_reports = payload.get("build_reports") or {}
    lines.extend(
        [
            "## Eval Rows",
            "",
            "| selector | checkpoint | acc | macro-F1 | true-side F1 | selection score | parse err | sel jaccard | n |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("eval_rows", []):
        report = build_reports.get(str(row.get("selector_slug") or ""), {})
        lines.append(
            "| {selector} | {checkpoint} | {acc:.4f} | {f1:.4f} | {true_f1:.4f} | "
            "{sel:.4f} | {perr:.4f} | {sel_jac:.4f} | {n} |".format(
                selector=row.get("selector_slug", ""),
                checkpoint=row.get("checkpoint", ""),
                acc=_metric(row, "accuracy"),
                f1=_metric(row, "macro_f1"),
                true_f1=_metric(row, "true_side_macro_f1"),
                sel=_metric(row, "selection_score"),
                perr=_metric(row, "parse_error_rate"),
                sel_jac=_metric(report, "selection_jaccard@5"),
                n=row.get("num_samples", ""),
            )
        )
    prior = payload.get("oracle_direct_checkpoint_prior") or []
    if prior:
        lines.extend(
            [
                "",
                "## Original Checkpoint Prior",
                "",
                "| checkpoint | acc | macro-F1 | true-side F1 | selection score |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prior[:8]:
            lines.append(
                "| {checkpoint} | {acc:.4f} | {f1:.4f} | {true_f1:.4f} | {sel:.4f} |".format(
                    checkpoint=row.get("checkpoint", ""),
                    acc=_metric(row, "accuracy"),
                    f1=_metric(row, "macro_f1"),
                    true_f1=_metric(row, "true_side_macro_f1"),
                    sel=_metric(row, "selection_score"),
                )
            )
    return "\n".join(lines) + "\n"


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
