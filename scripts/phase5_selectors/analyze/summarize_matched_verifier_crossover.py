#!/usr/bin/env python3
"""Validate and summarize the fixed-step structure-only verifier crossover."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "structure-only-matched-verifier-crossover-summary-v0.1"
EXPECTED_CHECKPOINT = "checkpoint-800"
EXPECTED_SPLIT = "val"
EXPECTED_CELLS = ("one_shot__fixed5", "stateful__fixed5")


class CrossoverSummaryError(ValueError):
    """Raised when either matrix violates the fixed crossover contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CrossoverSummaryError(f"missing crossover artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CrossoverSummaryError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise CrossoverSummaryError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CrossoverSummaryError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise CrossoverSummaryError(f"{context} must be a finite number")
    return result


def _load_verifier_matrix(
    root: Path,
    *,
    verifier_id: str,
    expected_cells: tuple[str, ...] = EXPECTED_CELLS,
) -> dict[str, Any]:
    input_path = root / "input" / "manifest.json"
    raw_path = root / "raw_logits" / "manifest.json"
    result_path = root / "materialized" / "matrix_manifest.json"
    input_manifest = _read_json(input_path)
    raw_manifest = _read_json(raw_path)
    result_manifest = _read_json(result_path)

    for context, payload in (
        ("input", input_manifest),
        ("raw logits", raw_manifest),
        ("materialized matrix", result_manifest),
    ):
        if payload.get("status") != "complete":
            raise CrossoverSummaryError(
                f"{verifier_id} {context} is not status=complete"
            )
        if payload.get("split") != EXPECTED_SPLIT:
            raise CrossoverSummaryError(
                f"{verifier_id} {context} split must be {EXPECTED_SPLIT!r}"
            )

    checkpoint = raw_manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise CrossoverSummaryError(f"{verifier_id} raw logits lack checkpoint provenance")
    checkpoint_name = str(checkpoint.get("checkpoint_name") or "")
    if checkpoint_name != EXPECTED_CHECKPOINT:
        raise CrossoverSummaryError(
            f"{verifier_id} checkpoint must be {EXPECTED_CHECKPOINT!r}, got {checkpoint_name!r}"
        )
    adapter_sha256 = str(checkpoint.get("adapter_sha256") or "")
    if len(adapter_sha256) != 64:
        raise CrossoverSummaryError(f"{verifier_id} has no pinned adapter SHA-256")

    input_cells = input_manifest.get("cells")
    result_cells = result_manifest.get("cells")
    if not isinstance(input_cells, list) or not isinstance(result_cells, list):
        raise CrossoverSummaryError(f"{verifier_id} matrix cells must be arrays")
    input_ids = tuple(str(row.get("cell_id") or "") for row in input_cells)
    result_ids = tuple(str(row.get("cell_id") or "") for row in result_cells)
    if input_ids != expected_cells or result_ids != expected_cells:
        raise CrossoverSummaryError(
            f"{verifier_id} must contain exactly the cells in order {expected_cells}; "
            f"input={input_ids}, result={result_ids}"
        )
    if int(input_manifest.get("cell_count", 0)) != 2:
        raise CrossoverSummaryError(f"{verifier_id} input cell_count must be 2")
    if int(result_manifest.get("cell_count", 0)) != 2:
        raise CrossoverSummaryError(f"{verifier_id} result cell_count must be 2")
    if bool(result_manifest.get("diagnostic_only")) is not True:
        raise CrossoverSummaryError(
            f"{verifier_id} crossover fanout must be explicitly diagnostic_only=true"
        )

    input_sha = _sha256(input_path)
    raw_sha = _sha256(raw_path)
    if str(raw_manifest.get("input_manifest_sha256") or "") != input_sha:
        raise CrossoverSummaryError(f"{verifier_id} raw/input manifest SHA mismatch")
    if str(result_manifest.get("input_manifest_sha256") or "") != input_sha:
        raise CrossoverSummaryError(f"{verifier_id} result/input manifest SHA mismatch")
    if str(result_manifest.get("raw_logits_manifest_sha256") or "") != raw_sha:
        raise CrossoverSummaryError(f"{verifier_id} result/raw manifest SHA mismatch")

    event_count = int(input_manifest.get("event_count", 0))
    if event_count <= 0:
        raise CrossoverSummaryError(f"{verifier_id} has no validation events")
    event_sequence_sha256 = str(
        input_manifest.get("event_id_sequence_sha256") or ""
    )
    source_matrix_sha256 = str(
        input_manifest.get("matrix_manifest_sha256") or ""
    )
    if len(event_sequence_sha256) != 64 or len(source_matrix_sha256) != 64:
        raise CrossoverSummaryError(
            f"{verifier_id} input lacks frozen event/source-matrix SHA-256 provenance"
        )
    metrics_by_cell: dict[str, dict[str, Any]] = {}
    for row in result_cells:
        cell_id = str(row["cell_id"])
        sample_count = int(row.get("num_samples", 0))
        if sample_count != event_count:
            raise CrossoverSummaryError(
                f"{verifier_id}/{cell_id} has {sample_count} samples; expected {event_count}"
            )
        metrics_by_cell[cell_id] = {
            "macro_f1": _finite_float(
                row.get("macro_f1"), context=f"{verifier_id}/{cell_id}.macro_f1"
            ),
            "accuracy": _finite_float(
                row.get("accuracy"), context=f"{verifier_id}/{cell_id}.accuracy"
            ),
            "eval_ce_loss": _finite_float(
                row.get("eval_ce_loss"), context=f"{verifier_id}/{cell_id}.eval_ce_loss"
            ),
            "num_samples": sample_count,
        }

    return {
        "verifier_id": verifier_id,
        "root": str(root),
        "checkpoint": checkpoint_name,
        "adapter_sha256": adapter_sha256,
        "run_dir": str(checkpoint.get("run_dir") or ""),
        "event_count": event_count,
        "event_id_sequence_sha256": event_sequence_sha256,
        "source_matrix_manifest_sha256": source_matrix_sha256,
        "metrics": metrics_by_cell,
        "artifacts": {
            "input_manifest": str(input_path),
            "input_manifest_sha256": input_sha,
            "raw_logits_manifest": str(raw_path),
            "raw_logits_manifest_sha256": raw_sha,
            "materialized_manifest": str(result_path),
            "materialized_manifest_sha256": _sha256(result_path),
        },
    }


def summarize_crossover(*, verifier_s_dir: Path, verifier_o_dir: Path) -> dict[str, Any]:
    verifier_s = _load_verifier_matrix(verifier_s_dir, verifier_id="V_S")
    verifier_o = _load_verifier_matrix(verifier_o_dir, verifier_id="V_O")
    for key in (
        "event_count",
        "event_id_sequence_sha256",
        "source_matrix_manifest_sha256",
    ):
        if verifier_s[key] != verifier_o[key]:
            raise CrossoverSummaryError(
                f"V_S and V_O do not share the same frozen validation input: {key}"
            )

    s_on_o = verifier_s["metrics"]["one_shot__fixed5"]["macro_f1"]
    s_on_s = verifier_s["metrics"]["stateful__fixed5"]["macro_f1"]
    o_on_o = verifier_o["metrics"]["one_shot__fixed5"]["macro_f1"]
    o_on_s = verifier_o["metrics"]["stateful__fixed5"]["macro_f1"]
    matched_mean = (s_on_s + o_on_o) / 2.0
    crossed_mean = (s_on_o + o_on_s) / 2.0

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support",
        "split": EXPECTED_SPLIT,
        "checkpoint": EXPECTED_CHECKPOINT,
        "event_count": verifier_s["event_count"],
        "event_id_sequence_sha256": verifier_s["event_id_sequence_sha256"],
        "prompt_cells": {
            "O": "one_shot__fixed5",
            "S": "stateful__fixed5",
        },
        "verifiers": {"V_S": verifier_s, "V_O": verifier_o},
        "macro_f1_matrix": {
            "V_S": {"O": s_on_o, "S": s_on_s},
            "V_O": {"O": o_on_o, "S": o_on_s},
        },
        "contrasts": {
            "prompt_S_minus_O_under_V_S": s_on_s - s_on_o,
            "prompt_S_minus_O_under_V_O": o_on_s - o_on_o,
            "verifier_V_O_minus_V_S_on_O": o_on_o - s_on_o,
            "verifier_V_O_minus_V_S_on_S": o_on_s - s_on_s,
            "matched_mean": matched_mean,
            "crossed_mean": crossed_mean,
            "matched_mean_minus_crossed_mean": matched_mean - crossed_mean,
            "difference_in_differences": (s_on_s - s_on_o) - (o_on_s - o_on_o),
        },
        "interpretation_contract": {
            "positive_matched_mean_minus_crossed_mean": (
                "directional evidence of aggregate prompt-policy/verifier compatibility; "
                "it does not by itself show that both matched pairs improve"
            ),
            "causal_claim_allowed": False,
            "significance_included": False,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    matrix = summary["macro_f1_matrix"]
    contrasts = summary["contrasts"]
    lines = [
        "# Structure-only matched-verifier crossover",
        "",
        f"- split: `{summary['split']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- frozen common support: `{summary['event_count']}` events",
        "",
        "| verifier | O prompt | S prompt | S - O |",
        "|---|---:|---:|---:|",
        (
            f"| V_S | {matrix['V_S']['O']:.6f} | {matrix['V_S']['S']:.6f} | "
            f"{contrasts['prompt_S_minus_O_under_V_S']:+.6f} |"
        ),
        (
            f"| V_O | {matrix['V_O']['O']:.6f} | {matrix['V_O']['S']:.6f} | "
            f"{contrasts['prompt_S_minus_O_under_V_O']:+.6f} |"
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
    parser.add_argument("--verifier-s-dir", required=True)
    parser.add_argument("--verifier-o-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_crossover(
        verifier_s_dir=Path(args.verifier_s_dir),
        verifier_o_dir=Path(args.verifier_o_dir),
    )
    _write_json(Path(args.output_json), summary)
    if args.output_md:
        _write_markdown(Path(args.output_md), summary)
    print(
        "[matched-verifier-crossover] "
        f"checkpoint={EXPECTED_CHECKPOINT} split={EXPECTED_SPLIT} "
        f"matched-minus-crossed="
        f"{summary['contrasts']['matched_mean_minus_crossed_mean']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
