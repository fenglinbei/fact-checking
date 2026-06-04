#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5b_val_map_verifier"
DEFAULT_DIRECT_VERIFIER_RUN_DIR = (
    "outputs/oracle_direct_verifier/stage2_sentence/train/"
    "b3_oracle_sentence_direct_verifier_1024_20260519-200709"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize v0.5b map-aware verifier checkpoint/selector evaluation metrics."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--primary-metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy", "selection_score", "true_side_macro_f1"],
    )
    parser.add_argument(
        "--direct-verifier-run-dir",
        default=DEFAULT_DIRECT_VERIFIER_RUN_DIR,
        help="Optional original oracle-direct verifier run dir, used only to report prior val checkpoint metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = _collect_eval_rows(output_dir=output_dir)
    build_reports = _collect_build_reports(output_dir=output_dir)
    checkpoint_prior = _collect_original_checkpoint_metrics(Path(args.direct_verifier_run_dir))

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            -_metric(row, args.primary_metric),
            -_metric(row, "macro_f1"),
            -_metric(row, "accuracy"),
            str(row.get("selector_slug") or ""),
            str(row.get("checkpoint") or ""),
        ),
    )
    best = rows_sorted[0] if rows_sorted else None
    best_by_selector = _best_by(rows, key="selector_slug", metric=args.primary_metric)
    best_by_checkpoint = _best_by(rows, key="checkpoint", metric=args.primary_metric)

    payload = {
        "status": "completed" if rows else "no_eval_metrics_found",
        "output_dir": str(output_dir),
        "split": str(args.split),
        "primary_metric": str(args.primary_metric),
        "n_eval_rows": len(rows),
        "best_overall": best,
        "best_by_selector": best_by_selector,
        "best_by_checkpoint": best_by_checkpoint,
        "eval_rows": rows_sorted,
        "build_reports": build_reports,
        "oracle_direct_checkpoint_prior": checkpoint_prior,
        "notes": [
            "v0.5b is an eval-only diagnostic over existing oracle-direct verifier checkpoints.",
            "The best checkpoint on original oracle-direct validation is only a prior; map-aware prompts must be judged by v0.5b eval rows.",
            "Prefer macro_f1 unless the downstream report uses selection_score or true-side emphasis.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_summary.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    if best:
        print(
            "Best v0.5b checkpoint: "
            f"selector={best['selector_slug']} checkpoint={best['checkpoint']} "
            f"{args.primary_metric}={_metric(best, args.primary_metric):.4f} "
            f"accuracy={_metric(best, 'accuracy'):.4f}"
        )
    else:
        print(f"No v0.5b eval metrics found under {output_dir}/eval")
    print(f"Wrote summary: {output_dir / 'analysis_summary.md'}")


def _collect_eval_rows(*, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_root = output_dir / "eval"
    for metrics_path in sorted(eval_root.glob("*/*/metrics.json")):
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


def _collect_build_reports(*, output_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    report_paths = list((output_dir / "verifier_data").glob("*/build_report.json"))
    report_paths.extend((output_dir / "verifier_data").glob("*/build/build_report.json"))
    for report_path in sorted(set(report_paths)):
        selector_slug = report_path.parent.parent.name if report_path.parent.name == "build" else report_path.parent.name
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            out[selector_slug] = {"error": str(exc), "path": str(report_path)}
            continue
        split_reports = report.get("splits") if isinstance(report.get("splits"), dict) else None
        if split_reports:
            val_report = split_reports.get("val") or next(iter(split_reports.values()))
            out[selector_slug] = {
                "path": str(report_path),
                "expected_selector_name": report.get("expected_selector_name"),
                "n_rows": val_report.get("n_rows"),
                "prompt_truncation_rate": val_report.get("prompt_truncation_rate"),
                "prompt_token_count": val_report.get("prompt_token_count"),
                "evidence_count": val_report.get("evidence_count"),
            }
        else:
            out[selector_slug] = {
                "path": str(report_path),
                "expected_selector_name": report.get("expected_selector_name"),
                "n_rows": report.get("n_rows"),
                "prompt_token_count": report.get("prompt_token_count"),
                "evidence_count": report.get("evidence_count"),
            }
    return out


def _collect_original_checkpoint_metrics(run_dir: Path) -> list[dict[str, Any]]:
    eval_root = run_dir / "eval"
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(eval_root.glob("step-*/metrics.json")):
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
    lines: list[str] = [
        "# v0.5b Map-Aware Verifier Checkpoint Diagnostic",
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
                    f"selection_score={_metric(best, 'selection_score'):.4f}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Eval Rows",
            "",
            "| selector | checkpoint | acc | macro-F1 | true-side F1 | selection | parse err | n |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("eval_rows", []):
        lines.append(
            "| {selector} | {checkpoint} | {acc:.4f} | {f1:.4f} | {true_f1:.4f} | {sel:.4f} | {perr:.4f} | {n} |".format(
                selector=row.get("selector_slug", ""),
                checkpoint=row.get("checkpoint", ""),
                acc=_metric(row, "accuracy"),
                f1=_metric(row, "macro_f1"),
                true_f1=_metric(row, "true_side_macro_f1"),
                sel=_metric(row, "selection_score"),
                perr=_metric(row, "parse_error_rate"),
                n=row.get("num_samples", ""),
            )
        )
    prior = payload.get("oracle_direct_checkpoint_prior") or []
    if prior:
        lines.extend(
            [
                "",
                "## Original Oracle-Direct Checkpoint Prior",
                "",
                "| checkpoint | original acc | original macro-F1 | original true-side F1 | original selection |",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use the v0.5b rows above for the actual map-aware checkpoint choice.",
            "- If no v0.5b rows exist yet, the original oracle-direct prior favors `checkpoint-600` / `best`.",
            "- Treat this as val diagnostic only until train-side map artifacts are built and a held-out test is run.",
            "",
        ]
    )
    return "\n".join(lines)


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(out):
        return float("-inf")
    return out


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


if __name__ == "__main__":
    main()
