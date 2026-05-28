#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.evidence_map_selector import prepare_evidence_map_candidate_rows
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_INPUT = "outputs/selectors/direct_evidence_cross_encoder/v0_4d_val_default_query_fusion/candidate_fusion_scores_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5a_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare v0.5a evidence-map annotation candidate pool from v0.4d fusion rows.")
    p.add_argument("--input-fusion-file", default=DEFAULT_INPUT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--candidate-top-n", type=int, default=20)
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input_fusion_file)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_fusion_file}")
    prepared = prepare_evidence_map_candidate_rows(rows, candidate_top_n=int(args.candidate_top_n))
    output_path = out_dir / f"evidence_map_candidate_pool_{args.split}.jsonl"
    write_jsonl(prepared, output_path)
    manifest: dict[str, Any] = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "input_fusion_file": str(args.input_fusion_file),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "candidate_top_n": int(args.candidate_top_n),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "n_events": len(prepared),
        "n_candidates": sum(len(row.get("candidates") or []) for row in prepared),
        "outputs": {"candidate_pool": str(output_path)},
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "prepare_manifest.json")
    print(f"Wrote evidence-map candidate pool: {output_path}")


if __name__ == "__main__":
    main()
