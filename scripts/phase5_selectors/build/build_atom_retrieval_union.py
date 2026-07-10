#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, canonicalize_sentence
from fact_checking.selectors.atom_retrieval_union import (
    AtomUnionSelectionParams,
    build_atom_union_pool_row,
    compute_atom_union_metrics,
    select_atom_union_rules,
)
from fact_checking.selectors.question_decomp_retrieval import oracle_selected_texts_by_event
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json, write_jsonl


DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build baseline top5 + atom-conditioned top15 union-pool diagnostics.")
    p.add_argument("--baseline-jsonl", required=True)
    p.add_argument("--atom-pool-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--selector-top-k", type=int, default=5)
    p.add_argument("--baseline-bonus", type=float, default=0.04)
    p.add_argument("--baseline-rank-weight", type=float, default=0.01)
    p.add_argument("--atom-rrf-weight", type=float, default=1.0)
    p.add_argument("--atom-route-hit-weight", type=float, default=0.004)
    p.add_argument("--atom-max-hybrid-weight", type=float, default=0.01)
    p.add_argument("--chunk-cache-path", default=None)
    p.add_argument("--final-pool-size", type=int, default=None)
    p.add_argument("--union-mmr-lambda", type=float, default=0.70)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_jsonl(args.baseline_jsonl)
    atom_pool_rows = read_jsonl(args.atom_pool_jsonl)
    if args.sample_limit is not None:
        baseline_rows = baseline_rows[: int(args.sample_limit)]
        atom_pool_rows = atom_pool_rows[: int(args.sample_limit)]
    if len(baseline_rows) != len(atom_pool_rows):
        raise ValueError(f"Row count mismatch: baseline={len(baseline_rows)} atom_pool={len(atom_pool_rows)}")
    if args.final_pool_size is not None and int(args.final_pool_size) < 1:
        raise ValueError("--final-pool-size must be >= 1.")
    if args.final_pool_size is not None and not args.chunk_cache_path:
        raise ValueError("--chunk-cache-path is required when --final-pool-size is set.")

    baseline_by_event = {str(row.get("event_id") or ""): row for row in baseline_rows}
    atom_pool_by_event = {str(row.get("event_id") or ""): row for row in atom_pool_rows}
    missing = sorted(set(baseline_by_event) ^ set(atom_pool_by_event))
    if missing:
        raise ValueError(f"Event mismatch between baseline and atom pool, sample={missing[:5]}")

    params = AtomUnionSelectionParams(
        selector_top_k=int(args.selector_top_k),
        baseline_bonus=float(args.baseline_bonus),
        baseline_rank_weight=float(args.baseline_rank_weight),
        atom_rrf_weight=float(args.atom_rrf_weight),
        atom_route_hit_weight=float(args.atom_route_hit_weight),
        atom_max_hybrid_weight=float(args.atom_max_hybrid_weight),
        union_mmr_lambda=float(args.union_mmr_lambda),
    )
    vectors_by_event = _load_candidate_vectors(args.chunk_cache_path) if args.chunk_cache_path else {}
    union_rows: list[dict[str, Any]] = []
    rule_rows: dict[str, list[dict[str, Any]]] = {
        "atom_union_baseline_first_top5": [],
        "atom_union_interleave_top5": [],
        "atom_union_source_score_top5": [],
    }
    trace_rows: list[dict[str, Any]] = []
    for atom_row in atom_pool_rows:
        event_id = str(atom_row.get("event_id") or "")
        baseline_row = baseline_by_event[event_id]
        union_row = build_atom_union_pool_row(
            baseline_row=baseline_row,
            atom_pool_row=atom_row,
            candidate_vectors=vectors_by_event.get(event_id),
            final_pool_size=int(args.final_pool_size) if args.final_pool_size is not None else None,
            params=params,
        )
        selections = select_atom_union_rules(union_row, params=params)
        union_rows.append(union_row)
        for name, selected in selections.items():
            rule_rows[name].append(
                {
                    "event_id": event_id,
                    "claim": union_row.get("claim", ""),
                    "label": union_row.get("label", ""),
                    "gold_label": union_row.get("gold_label", ""),
                    "claim_atoms": union_row.get("claim_atoms") or [],
                    "candidates": selected,
                }
            )
        trace_rows.append(
            {
                "event_id": event_id,
                "claim": union_row.get("claim", ""),
                "label": union_row.get("label", ""),
                "gold_label": union_row.get("gold_label", ""),
                "claim_atoms": union_row.get("claim_atoms") or [],
                "union_pool_size_before_mmr": int(union_row.get("union_pool_size_before_mmr") or 0),
                "union_pool_size": len(union_row.get("candidates") or []),
                "union_mmr_applied": bool(union_row.get("union_mmr_applied")),
                "source_counts": _source_counts(union_row),
                "rule_selected_texts": {
                    name: [candidate.get("text", "") for candidate in selected]
                    for name, selected in selections.items()
                },
            }
        )

    oracle_results = args.oracle_results if args.oracle_results is not None else _default_oracle_results(str(args.split))
    oracle_rows = read_jsonl(oracle_results) if oracle_results and Path(oracle_results).exists() else []
    if args.sample_limit is not None:
        oracle_rows = oracle_rows[: int(args.sample_limit)]
    metrics = compute_atom_union_metrics(
        union_rows=union_rows,
        rule_rows=rule_rows,
        oracle_texts=oracle_selected_texts_by_event(oracle_rows),
    )
    metrics["oracle_metrics_available"] = bool(oracle_rows)

    union_pool_path = out_dir / f"atom_union_candidate_pool_{args.split}.jsonl"
    trace_path = out_dir / f"atom_union_trace_{args.split}.jsonl"
    metrics_path = out_dir / f"atom_union_metrics_{args.split}.json"
    manifest_path = out_dir / f"atom_union_manifest_{args.split}.json"
    write_jsonl(union_pool_path, _json_safe_rows(union_rows))
    write_jsonl(trace_path, _json_safe_rows(trace_rows))
    for name, rows in rule_rows.items():
        write_jsonl(out_dir / f"{name}_{args.split}.jsonl", _json_safe_rows(rows))
    write_json(metrics_path, _json_safe(metrics))
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "split": str(args.split),
        "baseline_jsonl": str(args.baseline_jsonl),
        "atom_pool_jsonl": str(args.atom_pool_jsonl),
        "chunk_cache_path": str(args.chunk_cache_path or ""),
        "final_pool_size": int(args.final_pool_size) if args.final_pool_size is not None else None,
        "oracle_results": str(oracle_results),
        "output_dir": str(out_dir),
        "n_rows": len(union_rows),
        "params": params.__dict__,
        "paths": {
            "atom_union_candidate_pool": str(union_pool_path),
            "atom_union_trace": str(trace_path),
            "atom_union_metrics": str(metrics_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(manifest_path, _json_safe(manifest))

    print(f"Wrote atom union pool: {union_pool_path}")
    print(f"Wrote atom union metrics: {metrics_path}")
    for key in (
        "atom_union_pool",
        "atom_union_baseline_first_top5",
        "atom_union_interleave_top5",
        "atom_union_source_score_top5",
    ):
        row = metrics[key]
        print(
            f"{key} recall={row['oracle_selected_recall@5']:.4f} "
            f"jaccard={row['jaccard@5']:.4f} top1={row['top1_match']:.4f}"
        )


def _source_counts(row: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in row.get("candidates") or []:
        source = str(candidate.get("union_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _load_candidate_vectors(path: str) -> dict[str, dict[str, np.ndarray]]:
    with Path(path).open("rb") as handle:
        samples = pickle.load(handle)
    out: dict[str, dict[str, np.ndarray]] = {}
    for sample in samples:
        if not isinstance(sample, ChunkMMRSample):
            raise TypeError(f"Expected ChunkMMRSample in {path}, got {type(sample)!r}")
        if len(sample.candidates) != int(sample.chunk_emb.shape[0]):
            raise ValueError(
                f"Chunk candidate/vector mismatch for event={sample.event_id}: "
                f"candidates={len(sample.candidates)} vectors={sample.chunk_emb.shape[0]}"
            )
        vectors: dict[str, np.ndarray] = {}
        for idx, candidate in enumerate(sample.candidates):
            key = canonicalize_sentence(str(candidate.get("text") or ""))
            if key:
                vectors.setdefault(key, np.asarray(sample.chunk_emb[idx], dtype=np.float32))
        out[str(sample.event_id)] = vectors
    return out


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in rows]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
