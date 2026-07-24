#!/usr/bin/env python3
"""Validate and summarize the fixed-step retrieval/stateful verifier crossover."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__:
    from .summarize_matched_verifier_crossover import (
        CrossoverSummaryError,
        _load_verifier_matrix,
        _write_json,
    )
else:
    from summarize_matched_verifier_crossover import (
        CrossoverSummaryError,
        _load_verifier_matrix,
        _write_json,
    )


SCHEMA_VERSION = "retrieval-stateful-matched-verifier-crossover-summary-v0.1"
EXPECTED_CHECKPOINT = "checkpoint-800"
EXPECTED_SPLIT = "val"
EXPECTED_CELLS = ("retrieval__fixed5", "stateful__fixed5")


def summarize_crossover(*, verifier_r_dir: Path, verifier_s_dir: Path) -> dict[str, Any]:
    verifier_r = _load_verifier_matrix(
        verifier_r_dir,
        verifier_id="V_R",
        expected_cells=EXPECTED_CELLS,
    )
    verifier_s = _load_verifier_matrix(
        verifier_s_dir,
        verifier_id="V_S",
        expected_cells=EXPECTED_CELLS,
    )
    for key in (
        "event_count",
        "event_id_sequence_sha256",
        "source_matrix_manifest_sha256",
    ):
        if verifier_r[key] != verifier_s[key]:
            raise CrossoverSummaryError(
                f"V_R and V_S do not share the same frozen validation input: {key}"
            )

    r_on_r = verifier_r["metrics"]["retrieval__fixed5"]["macro_f1"]
    r_on_s = verifier_r["metrics"]["stateful__fixed5"]["macro_f1"]
    s_on_r = verifier_s["metrics"]["retrieval__fixed5"]["macro_f1"]
    s_on_s = verifier_s["metrics"]["stateful__fixed5"]["macro_f1"]
    matched_mean = (r_on_r + s_on_s) / 2.0
    crossed_mean = (r_on_s + s_on_r) / 2.0

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support",
        "split": EXPECTED_SPLIT,
        "checkpoint": EXPECTED_CHECKPOINT,
        "event_count": verifier_r["event_count"],
        "event_id_sequence_sha256": verifier_r["event_id_sequence_sha256"],
        "prompt_cells": {
            "R": "retrieval__fixed5",
            "S": "stateful__fixed5",
        },
        "verifiers": {"V_R": verifier_r, "V_S": verifier_s},
        "macro_f1_matrix": {
            "V_R": {"R": r_on_r, "S": r_on_s},
            "V_S": {"R": s_on_r, "S": s_on_s},
        },
        "contrasts": {
            "prompt_S_minus_R_under_V_R": r_on_s - r_on_r,
            "prompt_S_minus_R_under_V_S": s_on_s - s_on_r,
            "verifier_V_S_minus_V_R_on_R": s_on_r - r_on_r,
            "verifier_V_S_minus_V_R_on_S": s_on_s - r_on_s,
            "matched_mean": matched_mean,
            "crossed_mean": crossed_mean,
            "matched_mean_minus_crossed_mean": matched_mean - crossed_mean,
            "difference_in_differences": (s_on_s - s_on_r) - (r_on_s - r_on_r),
        },
        "interpretation_contract": {
            "positive_matched_mean_minus_crossed_mean": (
                "directional evidence of aggregate retrieval/stateful ordering and "
                "verifier-training-distribution compatibility; it does not by itself "
                "show that both matched pairs improve"
            ),
            "causal_claim_allowed": False,
            "significance_included": False,
        },
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    matrix = summary["macro_f1_matrix"]
    contrasts = summary["contrasts"]
    lines = [
        "# Retrieval/stateful matched-verifier crossover",
        "",
        f"- split: `{summary['split']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- frozen common support: `{summary['event_count']}` events",
        "",
        "| verifier | R prompt | S prompt | S - R |",
        "|---|---:|---:|---:|",
        (
            f"| V_R | {matrix['V_R']['R']:.6f} | {matrix['V_R']['S']:.6f} | "
            f"{contrasts['prompt_S_minus_R_under_V_R']:+.6f} |"
        ),
        (
            f"| V_S | {matrix['V_S']['R']:.6f} | {matrix['V_S']['S']:.6f} | "
            f"{contrasts['prompt_S_minus_R_under_V_S']:+.6f} |"
        ),
        "",
        (
            "Matched mean minus crossed mean: "
            f"`{contrasts['matched_mean_minus_crossed_mean']:+.6f}`."
        ),
        "",
        "This artifact reports directional validation evidence only; it does not include a significance test.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-r-dir", required=True)
    parser.add_argument("--verifier-s-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_crossover(
        verifier_r_dir=Path(args.verifier_r_dir),
        verifier_s_dir=Path(args.verifier_s_dir),
    )
    _write_json(Path(args.output_json), summary)
    if args.output_md:
        _write_markdown(Path(args.output_md), summary)
    print(
        "[retrieval-stateful-matched-verifier-crossover] "
        f"checkpoint={EXPECTED_CHECKPOINT} split={EXPECTED_SPLIT} "
        f"matched-minus-crossed="
        f"{summary['contrasts']['matched_mean_minus_crossed_mean']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
