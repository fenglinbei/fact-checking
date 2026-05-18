#!/usr/bin/env python3
"""Merge sharded oracle search JSONL outputs.

The search runner writes one JSONL per shard:

    oracle_results_<split>.shard-00000-of-00004.jsonl

This utility concatenates those files in shard order, keeps the last row for
duplicate event_ids, and writes a normal oracle_results_<split>.jsonl that can
be consumed by downstream supervision builders.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded oracle JSONL outputs")
    p.add_argument("--input-dir", required=True, help="Directory containing shard JSONL files")
    p.add_argument("--split", default="train", help="Data split name")
    p.add_argument("--num-shards", type=int, default=0, help="Expected shard count; 0 = infer")
    p.add_argument("--output-results", default=None, help="Merged JSONL path")
    p.add_argument("--output-metrics", default=None, help="Merged metrics JSON path")
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def _dedup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    by_event: dict[str, dict[str, Any]] = {}
    for record in records:
        event_id = str(record.get("event_id", ""))
        if not event_id:
            continue
        if event_id not in by_event:
            order.append(event_id)
        by_event[event_id] = record
    return [by_event[event_id] for event_id in order]


def _effective_candidate_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [int(r.get("n_candidates", 0)) for r in records if "n_candidates" in r]
    if not sizes:
        return {}
    arr = np.asarray(sizes, dtype=np.int32)
    return {
        "n_samples": int(len(arr)),
        "min": int(arr.min()),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "n_le_15": int((arr <= 15).sum()),
        "n_le_20": int((arr <= 20).sum()),
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    from sft.metrics import _compute_classification_metrics

    if not records:
        return {"accuracy": 0.0, "n_samples": 0}

    pred = np.asarray([int(r.get("final_prediction", -1)) for r in records], dtype=np.int32)
    gold = np.asarray([int(r.get("gold_id", -1)) for r in records], dtype=np.int32)
    metrics = _compute_classification_metrics(pred, gold)
    correct = int((pred == gold).sum())
    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            serializable[key] = int(value)
        elif isinstance(value, (np.floating,)):
            serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    serializable["accuracy"] = correct / len(records)
    serializable["n_samples"] = len(records)
    return serializable


def _shard_index(path: Path) -> int:
    match = re.search(r"\.shard-(\d+)-of-(\d+)\.jsonl$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    pattern = f"oracle_results_{args.split}.shard-*-of-*.jsonl"
    shard_paths = sorted(input_dir.glob(pattern), key=_shard_index)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched {input_dir / pattern}")
    if args.num_shards and len(shard_paths) != args.num_shards:
        raise RuntimeError(
            f"Expected {args.num_shards} shard files, found {len(shard_paths)}"
        )

    records: list[dict[str, Any]] = []
    for path in shard_paths:
        records.extend(_read_jsonl(path))
    merged = _dedup_records(records)

    output_results = Path(args.output_results) if args.output_results else input_dir / f"oracle_results_{args.split}.jsonl"
    output_metrics = Path(args.output_metrics) if args.output_metrics else input_dir / f"oracle_metrics_{args.split}.json"
    output_results.parent.mkdir(parents=True, exist_ok=True)

    with open(output_results, "w", encoding="utf-8") as fh:
        for record in merged:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = _metrics(merged)
    metrics["split"] = args.split
    metrics["merged_from_shards"] = [str(p) for p in shard_paths]
    metrics["n_input_rows"] = len(records)
    metrics["n_unique_event_ids"] = len(merged)
    metrics["effective_candidate_pool_stats"] = _effective_candidate_stats(merged)
    metrics["output_contract"] = {
        "version": "oracle-results-v3",
        "merged_shards": True,
    }
    with open(output_metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"Merged {len(shard_paths)} shard(s): {len(records)} rows -> {len(merged)} unique rows")
    print(f"Results: {output_results}")
    print(f"Metrics: {output_metrics}")


if __name__ == "__main__":
    main()
