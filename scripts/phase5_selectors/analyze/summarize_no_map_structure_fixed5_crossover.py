#!/usr/bin/env python3
"""Summarize the V_N/V_S x N_fixed5/S_fixed5 checkpoint-800 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from .summarize_matched_verifier_crossover import (
        CrossoverSummaryError,
        _load_verifier_matrix,
    )
except ImportError:
    from summarize_matched_verifier_crossover import (  # type: ignore[no-redef]
        CrossoverSummaryError,
        _load_verifier_matrix,
    )


CELLS = ("N_fixed5", "S_fixed5")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CrossoverSummaryError(f"expected JSON object: {path}")
    return payload


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(*, verifier_n_dir: Path, verifier_s_dir: Path, matrix_dir: Path) -> dict[str, Any]:
    verifier_n = _load_verifier_matrix(verifier_n_dir, verifier_id="V_N", expected_cells=CELLS)
    verifier_s = _load_verifier_matrix(verifier_s_dir, verifier_id="V_S", expected_cells=CELLS)
    for key in ("event_count", "event_id_sequence_sha256", "source_matrix_manifest_sha256"):
        if verifier_n[key] != verifier_s[key]:
            raise CrossoverSummaryError(f"V_N and V_S differ on frozen input provenance: {key}")
    if int(verifier_n["event_count"]) != 1234:
        raise CrossoverSummaryError("diagnostic must use exactly 1,234 frozen validation events")

    manifest_path = matrix_dir / "manifest.json"
    audit_path = matrix_dir / "audit.json"
    manifest = _read(manifest_path)
    audit = _read(audit_path)
    if (
        manifest.get("schema_version") != "no-map-structure-fixed5-matrix-v0.1"
        or manifest.get("all_ready") is not True
        or manifest.get("split") != "val"
        or int(manifest.get("expected_k", -1)) != 5
        or int(manifest.get("event_count", -1)) != 1234
        or [cell.get("cell_id") for cell in manifest.get("cells", [])] != list(CELLS)
        or manifest.get("audit_sha256") != _sha(audit_path)
        or verifier_n["source_matrix_manifest_sha256"] != _sha(manifest_path)
    ):
        raise CrossoverSummaryError("frozen N/S matrix manifest or SHA contract failed")
    if (
        audit.get("schema_version") != "no-map-structure-fixed5-input-difference-audit-v0.1"
        or audit.get("status") != "complete"
        or audit.get("passed") is not True
        or audit.get("standard_clean_results_audit_slot_mutated") is not False
        or int((audit.get("input_difference") or {}).get("event_count", -1)) != 1234
    ):
        raise CrossoverSummaryError("input-difference audit is incomplete")

    n_on_n = verifier_n["metrics"]["N_fixed5"]["macro_f1"]
    n_on_s = verifier_n["metrics"]["S_fixed5"]["macro_f1"]
    s_on_n = verifier_s["metrics"]["N_fixed5"]["macro_f1"]
    s_on_s = verifier_s["metrics"]["S_fixed5"]["macro_f1"]
    matched = (n_on_n + s_on_s) / 2.0
    crossed = (n_on_s + s_on_n) / 2.0
    return {
        "schema_version": "no-map-structure-fixed5-crossover-summary-v0.1",
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support",
        "split": "val",
        "checkpoint": "checkpoint-800",
        "event_count": 1234,
        "event_id_sequence_sha256": verifier_n["event_id_sequence_sha256"],
        "prompt_cells": {"N": "N_fixed5", "S": "S_fixed5"},
        "verifiers": {"V_N": verifier_n, "V_S": verifier_s},
        "macro_f1_matrix": {
            "V_N": {"N": n_on_n, "S": n_on_s},
            "V_S": {"N": s_on_n, "S": s_on_s},
        },
        "contrasts": {
            "prompt_S_minus_N_under_V_N": n_on_s - n_on_n,
            "prompt_S_minus_N_under_V_S": s_on_s - s_on_n,
            "verifier_V_S_minus_V_N_on_N": s_on_n - n_on_n,
            "verifier_V_S_minus_V_N_on_S": s_on_s - n_on_s,
            "matched_mean": matched,
            "crossed_mean": crossed,
            "matched_mean_minus_crossed_mean": matched - crossed,
            "difference_in_differences": (s_on_s - s_on_n) - (n_on_s - n_on_n),
        },
        "input_difference_audit": {
            "path": str(audit_path.resolve()),
            "sha256": _sha(audit_path),
            "metrics": audit["input_difference"],
        },
        "matrix_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha(manifest_path)},
        "interpretation_contract": {
            "diagnostic_only": True,
            "directional_only": True,
            "causal_claim_allowed": False,
            "significance_included": False,
            "natural_minmax_k_reused_as_fixed_k": False,
            "standard_clean_results_audit_slot_mutated": False,
        },
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_md(path: Path, summary: Mapping[str, Any]) -> None:
    matrix = summary["macro_f1_matrix"]
    contrasts = summary["contrasts"]
    text = "\n".join(
        [
            "# No-map / structure fixed-K matched-verifier crossover",
            "",
            "- split: `val`",
            "- checkpoint: `checkpoint-800`",
            "- common support: `1234`",
            "- prompt cells: `N_fixed5`, `S_fixed5`",
            "",
            "| verifier | N_fixed5 | S_fixed5 | S - N |",
            "|---|---:|---:|---:|",
            f"| V_N | {matrix['V_N']['N']:.6f} | {matrix['V_N']['S']:.6f} | {contrasts['prompt_S_minus_N_under_V_N']:+.6f} |",
            f"| V_S | {matrix['V_S']['N']:.6f} | {matrix['V_S']['S']:.6f} | {contrasts['prompt_S_minus_N_under_V_S']:+.6f} |",
            "",
            f"Matched minus crossed: `{contrasts['matched_mean_minus_crossed_mean']:+.6f}`.",
            "",
            "Directional validation diagnostic only; no causal or significance claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-n-dir", type=Path, required=True)
    parser.add_argument("--verifier-s-dir", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    summary = summarize(verifier_n_dir=args.verifier_n_dir, verifier_s_dir=args.verifier_s_dir, matrix_dir=args.matrix_dir)
    _write(args.output_json, summary)
    if args.output_md:
        _write_md(args.output_md, summary)
    print(
        "[no-map-fixed5-crossover] "
        f"matched-minus-crossed={summary['contrasts']['matched_mean_minus_crossed_mean']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
