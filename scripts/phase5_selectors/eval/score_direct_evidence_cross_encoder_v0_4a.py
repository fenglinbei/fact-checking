#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.direct_evidence_cross_encoder import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_VERSION,
    DirectEvidenceCrossEncoderScorer,
    merge_scored_event_rows,
    score_rows_with_scorer,
    select_event_shard,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_BUCKET_FILE = "outputs/selectors/count_amplified_stance_bucket_selector/v0_2_val/candidate_stance_buckets_v02_n7_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/direct_evidence_cross_encoder/v0_4a_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score v0.4a direct evidence with a text-only CrossEncoder reranker.")
    p.add_argument("--candidate-stance-buckets", default=DEFAULT_BUCKET_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mock-scores", action="store_true")
    p.add_argument("--merge-shards", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_jsonl(args.candidate_stance_buckets)
    if args.sample_limit is not None:
        all_rows = all_rows[: int(args.sample_limit)]
    if not all_rows:
        raise ValueError(f"No candidate rows loaded from {args.candidate_stance_buckets}")

    if bool(args.merge_shards):
        _merge_shards(args, all_rows, started_at)
        return

    shard_rows = select_event_shard(
        all_rows,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    output_path = _scored_path(out_dir, split=str(args.split), num_shards=int(args.num_shards), shard_index=int(args.shard_index))
    existing_rows: list[dict[str, Any]] = []
    if bool(args.resume) and output_path.exists():
        existing_rows = read_jsonl(output_path)
        existing_event_ids = {str(row.get("event_id") or "") for row in existing_rows}
        shard_rows_to_score = [row for row in shard_rows if str(row.get("event_id") or "") not in existing_event_ids]
    else:
        shard_rows_to_score = shard_rows

    scorer = None
    if not bool(args.mock_scores):
        scorer = DirectEvidenceCrossEncoderScorer(
            model_name=str(args.model_name),
            max_length=int(args.max_length),
            device=str(args.device),
            instruction=str(args.instruction),
        )
    newly_scored = score_rows_with_scorer(
        shard_rows_to_score,
        scorer,
        batch_size=int(args.batch_size),
        model_name=str(args.model_name),
        prompt_version=str(args.prompt_version),
        instruction=str(args.instruction),
        mock_scores=bool(args.mock_scores),
    )
    merged_rows = _merge_resume_rows(shard_rows, [*existing_rows, *newly_scored])
    write_jsonl(merged_rows, output_path)
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "mode": "score_shard",
        "input": str(args.candidate_stance_buckets),
        "output": str(output_path),
        "split": str(args.split),
        "model_name": str(args.model_name),
        "prompt_version": str(args.prompt_version),
        "instruction": str(args.instruction),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "torch_dtype": str(args.torch_dtype),
        "mock_scores": bool(args.mock_scores),
        "resume": bool(args.resume),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "n_reference_events": len(all_rows),
        "n_shard_events": len(shard_rows),
        "n_newly_scored_events": len(newly_scored),
        "n_output_events": len(merged_rows),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "forbidden_fields_note": "rank/source/oracle/gold metadata is not used in CrossEncoder input; it is retained only for eval and trace.",
    }
    save_json(manifest, _manifest_path(out_dir, split=str(args.split), num_shards=int(args.num_shards), shard_index=int(args.shard_index)))
    print(f"Wrote direct CE scored candidates: {output_path}")
    print(f"events={len(merged_rows)} newly_scored={len(newly_scored)} mock={bool(args.mock_scores)}")


def _merge_shards(args: argparse.Namespace, all_rows: list[dict[str, Any]], started_at: float) -> None:
    out_dir = Path(args.output_dir)
    shard_rows: list[dict[str, Any]] = []
    for idx in range(int(args.num_shards)):
        path = _scored_path(out_dir, split=str(args.split), num_shards=int(args.num_shards), shard_index=idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing shard file: {path}")
        shard_rows.extend(read_jsonl(path))
    merged = merge_scored_event_rows(all_rows, shard_rows)
    output_path = _scored_path(out_dir, split=str(args.split), num_shards=1, shard_index=0)
    write_jsonl(merged, output_path)
    save_json(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "mode": "merge_shards",
            "input": str(args.candidate_stance_buckets),
            "output": str(output_path),
            "split": str(args.split),
            "num_shards": int(args.num_shards),
            "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
            "n_output_events": len(merged),
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        out_dir / "direct_ce_merge_manifest.json",
    )
    print(f"Merged direct CE shards into: {output_path}")
    print(f"events={len(merged)} shards={int(args.num_shards)}")


def _merge_resume_rows(reference_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_scored_event_rows(reference_rows, scored_rows)


def _scored_path(out_dir: Path, *, split: str, num_shards: int, shard_index: int) -> Path:
    if int(num_shards) > 1:
        return out_dir / f"direct_ce_scored_candidates_{split}.shard{int(shard_index):05d}-of-{int(num_shards):05d}.jsonl"
    return out_dir / f"direct_ce_scored_candidates_{split}.jsonl"


def _manifest_path(out_dir: Path, *, split: str, num_shards: int, shard_index: int) -> Path:
    if int(num_shards) > 1:
        return out_dir / f"direct_ce_score_manifest_{split}.shard{int(shard_index):05d}-of-{int(num_shards):05d}.json"
    return out_dir / "direct_ce_score_manifest.json"


if __name__ == "__main__":
    main()
