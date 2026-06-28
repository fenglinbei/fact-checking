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
from tqdm.auto import tqdm

from fact_checking.build.candidates import ChunkMMRSample, canonicalize_sentence
from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder
from fact_checking.selectors.atom_conditioned_retrieval import (
    AtomRetrievalParams,
    align_atoms_and_chunks,
    build_atom_baseline_claim_mmr_row,
    build_atom_conditioned_retrieval_row,
    compute_retrieval_metrics,
    oracle_selected_texts_by_event,
)
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json, write_jsonl


DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build atom-conditioned retrieval traces from generated claim atoms.")
    p.add_argument("--claim-atoms-jsonl", required=True)
    p.add_argument("--chunk-cache-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)

    p.add_argument("--embedder-model", default=_default_embedder_model())
    p.add_argument("--device", default="cuda")
    p.add_argument("--embedder-max-length", type=int, default=256)
    p.add_argument("--embedder-batch-size", type=int, default=64)
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])

    p.add_argument("--per-atom-keep", type=int, default=20)
    p.add_argument("--merged-pool-size", type=int, default=15)
    p.add_argument("--selector-top-k", type=int, default=5)
    p.add_argument("--baseline-top-k", type=int, default=None)
    p.add_argument("--rrf-k", type=float, default=60.0)
    p.add_argument("--atom-weight", type=float, default=1.0)
    p.add_argument("--merge-mmr-lambda", type=float, default=0.70)
    p.add_argument("--alpha-dense", type=float, default=0.70)
    p.add_argument("--alpha-lexical", type=float, default=0.20)
    p.add_argument("--alpha-bm25", type=float, default=0.10)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atom_rows = read_jsonl(args.claim_atoms_jsonl)
    if args.sample_limit is not None:
        atom_rows = atom_rows[: int(args.sample_limit)]
    if not atom_rows:
        raise ValueError("No claim-atom rows loaded.")

    chunk_samples = _load_chunk_samples(args.chunk_cache_path)
    pairs = align_atoms_and_chunks(atom_rows, chunk_samples)

    _validate_embedder_model_path(args.embedder_model)
    embedder = TextEmbedder(
        EmbedderConfig(
            model_name=str(args.embedder_model),
            device=str(args.device),
            max_length=int(args.embedder_max_length),
            batch_size=int(args.embedder_batch_size),
            precision=str(args.precision),
        )
    )
    atom_embeddings = _encode_atoms(atom_rows, embedder, disable_progress=bool(args.no_progress))
    params = AtomRetrievalParams(
        per_atom_keep=int(args.per_atom_keep),
        merged_pool_size=int(args.merged_pool_size),
        selector_top_k=int(args.selector_top_k),
        baseline_top_k=int(args.baseline_top_k) if args.baseline_top_k is not None else None,
        rrf_k=float(args.rrf_k),
        atom_weight=float(args.atom_weight),
        merge_mmr_lambda=float(args.merge_mmr_lambda),
        alpha_dense=float(args.alpha_dense),
        alpha_lexical=float(args.alpha_lexical),
        alpha_bm25=float(args.alpha_bm25),
    )

    atom_retrieval_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    iterator = tqdm(pairs, desc="atom retrieval", unit="claim", dynamic_ncols=True, disable=bool(args.no_progress))
    for atom_row, sample in iterator:
        retrieval_row = build_atom_conditioned_retrieval_row(
            atom_row,
            sample,
            atom_embeddings=atom_embeddings,
            params=params,
        )
        baseline_row = build_atom_baseline_claim_mmr_row(sample, params=params)
        _add_baseline_overlap(retrieval_row, baseline_row)
        atom_retrieval_rows.append(retrieval_row)
        baseline_rows.append(baseline_row)
        trace_rows.append(
            {
                "event_id": retrieval_row["event_id"],
                "claim": retrieval_row["claim"],
                "label": retrieval_row.get("label", ""),
                "gold_label": retrieval_row.get("gold_label") or retrieval_row.get("label", ""),
                "claim_atoms": retrieval_row["claim_atoms"],
                "atom_routes": retrieval_row["atom_routes"],
                "baseline_top5_overlap": retrieval_row.get("baseline_top5_overlap", []),
            }
        )
        merged_rows.append(
            {
                "event_id": retrieval_row["event_id"],
                "claim": retrieval_row["claim"],
                "label": retrieval_row.get("label", ""),
                "gold_label": retrieval_row.get("gold_label") or retrieval_row.get("label", ""),
                "claim_atoms": retrieval_row["claim_atoms"],
                "candidates": retrieval_row["merged_candidate_pool"],
            }
        )
        selected_rows.append(
            {
                "event_id": retrieval_row["event_id"],
                "claim": retrieval_row["claim"],
                "label": retrieval_row.get("label", ""),
                "gold_label": retrieval_row.get("gold_label") or retrieval_row.get("label", ""),
                "claim_atoms": retrieval_row["claim_atoms"],
                "candidates": retrieval_row["selected_evidence"],
            }
        )

    oracle_results = args.oracle_results if args.oracle_results is not None else _default_oracle_results(str(args.split))
    oracle_rows = read_jsonl(oracle_results) if oracle_results and Path(oracle_results).exists() else []
    if args.sample_limit is not None:
        oracle_rows = oracle_rows[: int(args.sample_limit)]
    metrics = compute_retrieval_metrics(
        atom_rows=atom_rows,
        atom_retrieval_rows=atom_retrieval_rows,
        baseline_rows=baseline_rows,
        oracle_texts=oracle_selected_texts_by_event(oracle_rows),
    )
    metrics["oracle_metrics_available"] = bool(oracle_rows)

    trace_path = output_dir / f"retrieval_trace_{args.split}.jsonl"
    merged_path = output_dir / f"merged_candidate_pool_{args.split}.jsonl"
    selected_path = output_dir / f"selected_evidence_{args.split}.jsonl"
    baseline_path = output_dir / f"baseline_claim_mmr_selected_{args.split}.jsonl"
    metrics_path = output_dir / "retrieval_metrics.json"
    manifest_path = output_dir / "retrieval_manifest.json"

    write_jsonl(trace_path, _json_safe_rows(trace_rows))
    write_jsonl(merged_path, _json_safe_rows(merged_rows))
    write_jsonl(selected_path, _json_safe_rows(selected_rows))
    write_jsonl(baseline_path, _json_safe_rows(baseline_rows))
    write_json(metrics_path, _json_safe(metrics))
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "split": str(args.split),
        "claim_atoms_jsonl": str(args.claim_atoms_jsonl),
        "chunk_cache_path": str(args.chunk_cache_path),
        "oracle_results": str(oracle_results),
        "output_dir": str(output_dir),
        "n_atom_rows": len(atom_rows),
        "n_chunk_pairs": len(pairs),
        "params": params.__dict__,
        "embedder": {
            "model": str(args.embedder_model),
            "device": str(args.device),
            "max_length": int(args.embedder_max_length),
            "batch_size": int(args.embedder_batch_size),
            "precision": str(args.precision),
        },
        "paths": {
            "retrieval_trace": str(trace_path),
            "merged_candidate_pool": str(merged_path),
            "selected_evidence": str(selected_path),
            "baseline_claim_mmr_selected": str(baseline_path),
            "retrieval_metrics": str(metrics_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(manifest_path, _json_safe(manifest))
    write_json(output_dir / f"retrieval_manifest_{args.split}.json", _json_safe(manifest))

    print(f"Wrote atom retrieval trace: {trace_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(
        "atom_conditioned oracle_pool_recall@15={:.4f} selected_recall@5={:.4f} jaccard@5={:.4f}".format(
            metrics["atom_conditioned_retrieval"]["oracle_pool_recall@15"],
            metrics["atom_conditioned_retrieval"]["oracle_selected_recall@5"],
            metrics["atom_conditioned_retrieval"]["jaccard@5"],
        )
    )
    print(
        "baseline_claim_mmr oracle_pool_recall@15={:.4f} selected_recall@5={:.4f} jaccard@5={:.4f}".format(
            metrics["baseline_claim_mmr"]["oracle_pool_recall@15"],
            metrics["baseline_claim_mmr"]["oracle_selected_recall@5"],
            metrics["baseline_claim_mmr"]["jaccard@5"],
        )
    )


def _load_chunk_samples(path: str | Path) -> list[ChunkMMRSample]:
    with Path(path).open("rb") as fh:
        samples = pickle.load(fh)
    if not isinstance(samples, list):
        raise ValueError(f"Expected chunk cache to contain a list, got {type(samples).__name__}.")
    return samples


def _encode_atoms(
    atom_rows: list[dict[str, Any]],
    embedder: TextEmbedder,
    *,
    disable_progress: bool,
) -> dict[tuple[str, str], np.ndarray]:
    keys: list[tuple[str, str]] = []
    texts: list[str] = []
    for row in atom_rows:
        event_id = str(row.get("event_id") or "")
        for idx, atom in enumerate(row.get("claim_atoms") or [], start=1):
            atom_id = str(atom.get("atom_id") or f"A{idx}")
            keys.append((event_id, atom_id))
            texts.append(str(atom.get("query_rendering") or atom.get("proposition") or atom.get("text") or ""))
    if not texts:
        return {}
    if not disable_progress:
        print(f"[atom-retrieval] Encoding {len(texts)} atom query rendering(s)")
    embeddings = embedder.encode(texts, is_query=True)
    return {key: embeddings[idx] for idx, key in enumerate(keys)}


def _add_baseline_overlap(atom_row: dict[str, Any], baseline_row: dict[str, Any]) -> None:
    baseline = {
        canonicalize_sentence(str(candidate.get("text") or ""))
        for candidate in baseline_row.get("candidates") or []
    }
    overlap: list[str] = []
    for candidate in atom_row.get("selected_evidence") or []:
        text = canonicalize_sentence(str(candidate.get("text") or ""))
        if text and text in baseline:
            overlap.append(text)
    atom_row["baseline_top5_overlap"] = overlap


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _default_embedder_model() -> str:
    primary = Path("/data/models/bge-base-en-v1.5")
    if primary.exists():
        return str(primary)
    fallback = Path("/home/fenglin/project/models/bge-base-en-v1.5")
    if fallback.exists():
        return str(fallback)
    return str(primary)


def _validate_embedder_model_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() and not path.exists():
        raise FileNotFoundError(
            f"Embedder model path does not exist: {value}. "
            "Set EMBEDDER_MODEL to a local model path or pass a Hugging Face repo id."
        )


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
