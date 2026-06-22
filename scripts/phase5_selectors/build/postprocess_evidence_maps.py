#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.evidence_map_selector import attach_evidence_map_annotations, summarize_atom_quality_rows
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5a_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocess v0.5a evidence-map teacher annotations into candidate features.")
    p.add_argument("--candidate-pool", required=True)
    p.add_argument("--annotations", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = Path(args.annotations) if args.annotations else out_dir / f"deepseek_evidence_map_annotations_{args.split}.jsonl"
    rows = read_jsonl(args.candidate_pool)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    annotations = read_jsonl(annotations_path) if annotations_path.exists() else []
    features = attach_evidence_map_annotations(rows, annotations)
    output_path = out_dir / f"candidate_evidence_map_features_{args.split}.jsonl"
    atom_quality_path = out_dir / "atom_quality_summary.json"
    atom_quality_split_path = out_dir / f"atom_quality_summary_{args.split}.json"
    atom_quality_summary = summarize_atom_quality_rows(features)
    write_jsonl(features, output_path)
    save_json(atom_quality_summary, atom_quality_path)
    save_json(atom_quality_summary, atom_quality_split_path)
    manifest: dict[str, Any] = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "candidate_pool": str(args.candidate_pool),
        "annotations": str(annotations_path),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "n_events": len(features),
        "n_annotations": len(annotations),
        "n_candidates": sum(len(row.get("candidates") or []) for row in features),
        "parse_status_counts": _counts(str(row.get("evidence_map_parse_status") or "") for row in features),
        "atom_quality": {
            "total_atoms": int(atom_quality_summary.get("total_atoms") or 0),
            "fragment_atom_count": int(atom_quality_summary.get("fragment_atom_count") or 0),
            "rows_with_fragment_atoms": int(atom_quality_summary.get("rows_with_fragment_atoms") or 0),
            "row_fragment_rate": float(atom_quality_summary.get("row_fragment_rate") or 0.0),
        },
        "outputs": {
            "candidate_features": str(output_path),
            "atom_quality_summary": str(atom_quality_path),
            "atom_quality_summary_split": str(atom_quality_split_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "postprocess_manifest.json")
    save_json(manifest, out_dir / f"postprocess_manifest_{args.split}.json")
    print(f"Wrote evidence-map candidate features: {output_path}")


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


if __name__ == "__main__":
    main()
