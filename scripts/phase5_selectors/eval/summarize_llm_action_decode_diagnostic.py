#!/usr/bin/env python3
"""Summarize decode-time permutation/calibration diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "recall@5",
    "jaccard@5",
    "top1_match",
    "oracle_rank_ndcg@5",
    "pairwise_order_acc@5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LLM action decode diagnostic eval dirs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-name", default="raw")
    parser.add_argument("--accept-jaccard-lift", type=float, default=0.015)
    parser.add_argument("--accept-jaccard", type=float, default=0.275)
    parser.add_argument("--accept-recall", type=float, default=0.40)
    parser.add_argument("--max-ndcg-drop", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or input_dir)
    metrics_paths = sorted(input_dir.glob("*/selection_metrics.json"))
    if not metrics_paths:
        raise ValueError(f"No selection_metrics.json found under {input_dir}.")

    payloads = [(path.parent.name, _read_json(path), path) for path in metrics_paths]
    raw_payload = _find_raw_payload(payloads, raw_name=str(args.raw_name))
    raw_selector = raw_payload.get("selector", {})
    raw_jaccard = _metric(raw_selector, "jaccard@5")
    raw_ndcg = _metric(raw_selector, "oracle_rank_ndcg@5")
    hybrid = raw_payload.get("controls", {}).get("hybrid_score_top5", {})
    reference = _saved_score_reference(raw_payload)

    rows: list[dict[str, Any]] = []
    for name, payload, path in payloads:
        selector = payload.get("selector", {})
        row = {
            "name": name,
            "path": str(path),
            "decode_strategy": payload.get("decode_strategy"),
            "num_permutations": payload.get("num_permutations"),
            "aggregation": payload.get("aggregation"),
            "calibration_mode": payload.get("calibration_mode"),
            "calibration_alpha": payload.get("calibration_alpha"),
        }
        for key in METRIC_KEYS:
            row[key] = _metric(selector, key)
            row[f"delta_vs_raw_{key}"] = row[key] - _metric(raw_selector, key)
            row[f"delta_vs_hybrid_{key}"] = row[key] - _metric(hybrid, key)
            if reference:
                row[f"delta_vs_saved_score_{key}"] = row[key] - _metric(reference, key)
        row["passes_jaccard_lift"] = row["delta_vs_raw_jaccard@5"] >= float(args.accept_jaccard_lift)
        row["passes_absolute_gate"] = (
            row["jaccard@5"] >= float(args.accept_jaccard)
            and row["recall@5"] >= float(args.accept_recall)
        )
        row["ndcg_drop_vs_raw"] = raw_ndcg - row["oracle_rank_ndcg@5"]
        row["adoptable_decode_fix"] = (
            (row["passes_jaccard_lift"] or row["passes_absolute_gate"])
            and row["ndcg_drop_vs_raw"] <= float(args.max_ndcg_drop)
        )
        rows.append(row)

    summary = {
        "input_dir": str(input_dir),
        "raw_name": str(args.raw_name),
        "acceptance": {
            "accept_jaccard_lift": float(args.accept_jaccard_lift),
            "accept_jaccard": float(args.accept_jaccard),
            "accept_recall": float(args.accept_recall),
            "max_ndcg_drop": float(args.max_ndcg_drop),
        },
        "raw_metrics": {key: _metric(raw_selector, key) for key in METRIC_KEYS},
        "hybrid_metrics": {key: _metric(hybrid, key) for key in METRIC_KEYS},
        "saved_score_reference": {key: _metric(reference, key) for key in METRIC_KEYS} if reference else {},
        "rows": rows,
        "diagnosis": _diagnosis(rows, min_lift=float(args.accept_jaccard_lift)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison_table.json", summary)
    _write_markdown(output_dir / "analysis.md", summary)
    print(f"Wrote comparison table: {output_dir / 'comparison_table.json'}")
    print(f"Wrote analysis: {output_dir / 'analysis.md'}")


def _find_raw_payload(payloads: list[tuple[str, dict[str, Any], Path]], *, raw_name: str) -> dict[str, Any]:
    for name, payload, _path in payloads:
        if name == raw_name:
            return payload
    for _name, payload, _path in payloads:
        if payload.get("decode_strategy") == "raw":
            return payload
    raise ValueError("Could not find a raw decode metrics payload.")


def _saved_score_reference(payload: dict[str, Any]) -> dict[str, Any]:
    refs = payload.get("reference_metrics") or {}
    return dict(refs.get("single_margin_step0_static", {}).get("metrics") or {})


def _diagnosis(rows: list[dict[str, Any]], *, min_lift: float) -> dict[str, Any]:
    calibration_rows = [row for row in rows if row.get("decode_strategy") == "calibrated"]
    permutation_rows = [
        row
        for row in rows
        if row.get("decode_strategy") in {"permutation", "permutation_calibrated"}
    ]
    best_calibration = max(calibration_rows, key=lambda row: row["delta_vs_raw_jaccard@5"], default=None)
    best_permutation = max(permutation_rows, key=lambda row: row["delta_vs_raw_jaccard@5"], default=None)
    calibration_lift = float(best_calibration["delta_vs_raw_jaccard@5"]) if best_calibration else 0.0
    permutation_lift = float(best_permutation["delta_vs_raw_jaccard@5"]) if best_permutation else 0.0

    if permutation_lift >= min_lift and calibration_lift < min_lift:
        conclusion = "position_or_candidate_order_bias_dominant"
    elif calibration_lift >= min_lift and permutation_lift < min_lift:
        conclusion = "action_label_prior_dominant"
    elif calibration_lift >= min_lift and permutation_lift >= min_lift:
        conclusion = "decode_bias_likely_both_label_and_order"
    else:
        conclusion = "no_decode_time_fix_signal"
    return {
        "conclusion": conclusion,
        "best_calibration": best_calibration,
        "best_permutation": best_permutation,
        "calibration_jaccard_lift": calibration_lift,
        "permutation_jaccard_lift": permutation_lift,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# LLM Action Decode Diagnostic",
        "",
        f"- diagnosis: {summary['diagnosis']['conclusion']}",
        f"- raw Jaccard@5: {summary['raw_metrics'].get('jaccard@5', math.nan):.4f}",
        f"- hybrid Jaccard@5: {summary['hybrid_metrics'].get('jaccard@5', math.nan):.4f}",
    ]
    saved = summary.get("saved_score_reference") or {}
    if saved:
        lines.append(f"- saved-score Jaccard@5: {saved.get('jaccard@5', math.nan):.4f}")
    lines.extend(
        [
            "",
            "| name | decode | recall@5 | jaccard@5 | delta raw J | ndcg@5 | delta raw NDCG | pairwise | adoptable |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {name} | {decode} | {recall:.4f} | {jaccard:.4f} | {dj:.4f} | {ndcg:.4f} | {dn:.4f} | {pairwise:.4f} | {adoptable} |".format(
                name=row["name"],
                decode=row.get("decode_strategy"),
                recall=float(row.get("recall@5", math.nan)),
                jaccard=float(row.get("jaccard@5", math.nan)),
                dj=float(row.get("delta_vs_raw_jaccard@5", math.nan)),
                ndcg=float(row.get("oracle_rank_ndcg@5", math.nan)),
                dn=float(row.get("delta_vs_raw_oracle_rank_ndcg@5", math.nan)),
                pairwise=float(row.get("pairwise_order_acc@5", math.nan)),
                adoptable="yes" if row.get("adoptable_decode_fix") else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, math.nan))
    except (TypeError, ValueError):
        return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
