#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.count_amplified_stance_bucket_selector import attach_stance_annotations
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map DeepSeek stance scores to local soft stance buckets.")
    p.add_argument("--candidate-pool", required=True)
    p.add_argument("--teacher-annotations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-stance-buckets", default="3,5,7")
    p.add_argument("--bucket-tau", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.candidate_pool)
    annotations = read_jsonl(args.teacher_annotations)
    if not rows:
        raise ValueError("No candidate pool rows loaded.")
    bucket_values = _parse_bucket_values(args.n_stance_buckets)
    outputs: dict[str, str] = {}
    for n in bucket_values:
        enriched = attach_stance_annotations(
            rows,
            annotations,
            n_stance_buckets=int(n),
            tau=float(args.bucket_tau),
        )
        suffix = "" if int(n) == 3 else f"_n{int(n)}"
        path = out_dir / f"candidate_stance_buckets{suffix}_{args.split}.jsonl"
        write_jsonl(enriched, path)
        outputs[f"n{int(n)}"] = str(path)

    save_json(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "candidate_pool": str(args.candidate_pool),
            "teacher_annotations": str(args.teacher_annotations),
            "split": str(args.split),
            "n_stance_buckets": bucket_values,
            "bucket_tau": float(args.bucket_tau),
            "n_events": len(rows),
            "n_annotations": len(annotations),
            "outputs": outputs,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        out_dir / "postprocess_stance_manifest.json",
    )
    print(f"Wrote stance bucket files under: {out_dir}")
    print("bucket files: " + ", ".join(outputs.values()))


def _parse_bucket_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 2:
            raise ValueError("n_stance_buckets must be at least 2.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("No bucket values provided.")
    return values


if __name__ == "__main__":
    main()
