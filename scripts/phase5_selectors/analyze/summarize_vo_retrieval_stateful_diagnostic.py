#!/usr/bin/env python3
"""Validate and summarize V_O checkpoint-800 on frozen retrieval/stateful inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__:
    from .summarize_matched_verifier_crossover import (
        _load_verifier_matrix,
        _write_json,
    )
else:
    from summarize_matched_verifier_crossover import (
        _load_verifier_matrix,
        _write_json,
    )


SCHEMA_VERSION = "vo-retrieval-stateful-diagnostic-summary-v0.1"
EXPECTED_CHECKPOINT = "checkpoint-800"
EXPECTED_SPLIT = "val"
EXPECTED_CELLS = ("retrieval__fixed5", "stateful__fixed5")


def summarize_diagnostic(*, verifier_o_dir: Path) -> dict[str, Any]:
    verifier_o = _load_verifier_matrix(
        verifier_o_dir,
        verifier_id="V_O",
        expected_cells=EXPECTED_CELLS,
    )
    metrics_r = verifier_o["metrics"]["retrieval__fixed5"]
    metrics_s = verifier_o["metrics"]["stateful__fixed5"]
    macro_f1_delta = metrics_s["macro_f1"] - metrics_r["macro_f1"]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support_single_verifier",
        "split": EXPECTED_SPLIT,
        "checkpoint": EXPECTED_CHECKPOINT,
        "event_count": verifier_o["event_count"],
        "event_id_sequence_sha256": verifier_o["event_id_sequence_sha256"],
        "input_cells": {
            "R": "retrieval__fixed5",
            "S": "stateful__fixed5",
        },
        "verifier": verifier_o,
        "metrics": {
            "V_O_on_R": metrics_r,
            "V_O_on_S": metrics_s,
        },
        "primary_contrast": {
            "name": "V_O(S)-V_O(R)",
            "metric": "macro_f1",
            "value": macro_f1_delta,
            "higher_is_better": True,
        },
        "contrasts": {
            "V_O(S)-V_O(R)_macro_f1": macro_f1_delta,
            "V_O(S)-V_O(R)_accuracy": (
                metrics_s["accuracy"] - metrics_r["accuracy"]
            ),
            "V_O(S)-V_O(R)_eval_ce_loss": (
                metrics_s["eval_ce_loss"] - metrics_r["eval_ce_loss"]
            ),
        },
        "interpretation_contract": {
            "positive_primary_contrast": (
                "directional validation evidence that the frozen V_O checkpoint scores "
                "the structure-induced S input above retrieval-order R on the same "
                "common-support events"
            ),
            "matched_training_claim_allowed": False,
            "causal_claim_allowed": False,
            "significance_included": False,
        },
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    metrics = summary["metrics"]
    primary = summary["primary_contrast"]
    lines = [
        "# V_O retrieval/stateful diagnostic",
        "",
        f"- split: `{summary['split']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- frozen common support: `{summary['event_count']}` events",
        "",
        "| frozen V_O input | Macro-F1 | Accuracy | CE loss |",
        "|---|---:|---:|---:|",
        (
            f"| R: retrieval order | {metrics['V_O_on_R']['macro_f1']:.6f} | "
            f"{metrics['V_O_on_R']['accuracy']:.6f} | "
            f"{metrics['V_O_on_R']['eval_ce_loss']:.6f} |"
        ),
        (
            f"| S: structure-induced | {metrics['V_O_on_S']['macro_f1']:.6f} | "
            f"{metrics['V_O_on_S']['accuracy']:.6f} | "
            f"{metrics['V_O_on_S']['eval_ce_loss']:.6f} |"
        ),
        "",
        f"**V_O(S)-V_O(R) Macro-F1: `{primary['value']:+.6f}`.**",
        "",
        "This is a directional validation diagnostic for one frozen verifier checkpoint; it is not a significance or causal result.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-o-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_diagnostic(verifier_o_dir=Path(args.verifier_o_dir))
    _write_json(Path(args.output_json), summary)
    if args.output_md:
        _write_markdown(Path(args.output_md), summary)
    print(
        "[vo-retrieval-stateful-diagnostic] "
        f"checkpoint={EXPECTED_CHECKPOINT} split={EXPECTED_SPLIT} "
        f"V_O(S)-V_O(R)={summary['primary_contrast']['value']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
