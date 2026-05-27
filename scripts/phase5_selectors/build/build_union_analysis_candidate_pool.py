#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    build_union_analysis_rows,
    enrich_quality_rows,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_QD_UNION = "outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl"
DEFAULT_TRAIN_QD_UNION = "outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build original Stage2 + QD union analysis candidate pools.")
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--qd-union-pool-jsonl", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    split = str(args.split)
    oracle_results = args.oracle_results or _default_oracle_results(split)
    qd_union_pool = args.qd_union_pool_jsonl or _default_qd_union(split)
    output_dir = Path(args.output_dir or f"outputs/selectors/count_amplified_stance_bucket_selector/v0_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = _load_rows(oracle_results, sample_limit=args.sample_limit)
    qd_rows = _load_rows(qd_union_pool, sample_limit=args.sample_limit)
    if not oracle_rows:
        raise ValueError("No oracle rows loaded.")

    union_rows, summary = build_union_analysis_rows(oracle_rows, qd_rows)
    quality_rows = enrich_quality_rows(union_rows)
    created_at = datetime.now(timezone.utc).isoformat()

    union_path = output_dir / f"union_analysis_candidate_pool_{split}.jsonl"
    quality_path = output_dir / f"candidate_quality_labels_{split}.jsonl"
    manifest_path = output_dir / "build_union_manifest.json"
    write_jsonl(union_rows, union_path)
    write_jsonl(quality_rows, quality_path)
    save_json(
        {
            "status": "completed",
            "created_at": created_at,
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "split": split,
            "oracle_results": str(oracle_results),
            "qd_union_pool_jsonl": str(qd_union_pool),
            "output_dir": str(output_dir),
            "sample_limit": args.sample_limit,
            "outputs": {
                "union_analysis_candidate_pool": str(union_path),
                "candidate_quality_labels": str(quality_path),
            },
            "summary": summary,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        manifest_path,
    )
    print(f"Wrote union analysis pool: {union_path}")
    print(
        "events={events} mean_union_pool={pool:.2f} missing_qd={missing}".format(
            events=summary["n_events"],
            pool=float(summary["mean_union_pool_size"]),
            missing=summary["n_missing_qd_events"],
        )
    )


def _load_rows(path: str, *, sample_limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if sample_limit is not None:
        rows = rows[: int(sample_limit)]
    return rows


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _default_qd_union(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_QD_UNION
    return DEFAULT_VAL_QD_UNION


if __name__ == "__main__":
    main()
