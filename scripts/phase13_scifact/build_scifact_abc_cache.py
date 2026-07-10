#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.candidates import ChunkMMRSample, canonicalize_sentence
from fact_checking.build.chunking import build_chunking_strategy
from fact_checking.data.io import load_split
from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder
from fact_checking.utils.text import word_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build claim-aware SciFact ABC ChunkMMRSample cache.")
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--claim-atoms-jsonl", required=True)
    parser.add_argument("--corpus-sqlite", default="data/processed/SciFact/scifact_corpus.sqlite")
    parser.add_argument("--output-root", default="outputs/cache/scifact_abc")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--label-schema", default="scifact3")

    parser.add_argument("--embedder-model", default=_default_embedder_model())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedder-max-length", type=int, default=256)
    parser.add_argument("--embedder-batch-size", type=int, default=64)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--claim-doc-top-k", type=int, default=80)
    parser.add_argument("--atom-doc-top-k", type=int, default=40)
    parser.add_argument("--universe-doc-top-k", type=int, default=120)
    parser.add_argument("--fts-row-limit-multiplier", type=int, default=8)

    parser.add_argument("--boundary-mode", default="local_peak")
    parser.add_argument("--boundary-threshold", type=float, default=0.55)
    parser.add_argument("--lambda-std", type=float, default=0.5)
    parser.add_argument("--w-sem", type=float, default=0.75)
    parser.add_argument("--w-rel", type=float, default=0.25)
    parser.add_argument("--max-sent-per-chunk", type=int, default=3)
    parser.add_argument("--max-tokens-per-chunk", type=int, default=150)
    parser.add_argument("--min-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--single-sentence-relevance-threshold", type=float, default=0.55)
    parser.add_argument("--high-rel-threshold", type=float, default=0.70)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    _validate_shard_args(args)

    fingerprint = _fingerprint(args)
    output_dir = Path(args.output_root) / fingerprint
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _shard_suffix(int(args.num_shards), int(args.shard_index))
    cache_path = output_dir / f"{args.split}{suffix}.pkl"
    manifest_path = output_dir / f"{args.split}{suffix}_manifest.json"

    samples = load_split(args.raw_path, dataset=args.dataset, label_schema=args.label_schema)
    if args.sample_limit is not None:
        samples = samples[: int(args.sample_limit)]
    atom_rows = _read_jsonl(Path(args.claim_atoms_jsonl), sample_limit=args.sample_limit)
    atoms_by_event = {str(row.get("event_id") or ""): row for row in atom_rows}
    missing_atoms = [sample.event_id for sample in samples if sample.event_id not in atoms_by_event]
    if missing_atoms:
        raise ValueError(f"Missing claim atoms for {len(missing_atoms)} SciFact claims, sample={missing_atoms[:5]}")
    if bool(args.merge_shards):
        _merge_shards(
            args,
            all_samples=samples,
            output_dir=output_dir,
            fingerprint=fingerprint,
            started_at=started_at,
        )
        return 0
    shard_samples = _select_shard(samples, num_shards=int(args.num_shards), shard_index=int(args.shard_index))
    if not shard_samples:
        raise ValueError(f"No SciFact sample assigned to shard {args.shard_index}/{args.num_shards}.")

    _validate_embedder_model_path(args.embedder_model)

    embedder_cfg = EmbedderConfig(
        model_name=str(args.embedder_model),
        device=str(args.device),
        max_length=int(args.embedder_max_length),
        batch_size=int(args.embedder_batch_size),
        precision=str(args.precision),
    )
    embedder = TextEmbedder(embedder_cfg)
    chunking = build_chunking_strategy(_chunking_cfg(args), _retrieval_cfg(args))

    conn = sqlite3.connect(str(args.corpus_sqlite))
    try:
        chunk_samples: list[ChunkMMRSample] = []
        coverage_rows: list[dict[str, Any]] = []
        candidate_counts: list[int] = []
        doc_counts: list[int] = []
        desc = f"scifact ABC [{args.split}{suffix}]"
        iterator = tqdm(shard_samples, desc=desc, unit="claim", disable=bool(args.no_progress))
        for sample in iterator:
            atom_row = atoms_by_event[sample.event_id]
            route_doc_ids, route_metadata = _retrieval_universe_doc_ids(
                conn,
                claim=sample.claim,
                atoms=atom_row.get("claim_atoms") or [],
                claim_doc_top_k=int(args.claim_doc_top_k),
                atom_doc_top_k=int(args.atom_doc_top_k),
                universe_doc_top_k=int(args.universe_doc_top_k),
                fts_row_limit_multiplier=int(args.fts_row_limit_multiplier),
            )
            abstracts = _load_abstracts(conn, route_doc_ids)
            claim_emb = embedder.encode([sample.claim], is_query=True)[0].astype(np.float32, copy=False)
            candidates = _build_abc_candidates(
                abstracts,
                claim=sample.claim,
                claim_embedding=claim_emb,
                chunking=chunking,
                route_metadata=route_metadata,
            )
            chunk_emb = _encode_candidates(embedder, candidates, claim_emb=claim_emb)
            chunk_samples.append(
                ChunkMMRSample(
                    event_id=sample.event_id,
                    claim=sample.claim,
                    label=sample.label,
                    explain=sample.explain,
                    candidates=candidates,
                    chunk_emb=chunk_emb,
                    claim_emb=claim_emb,
                )
            )
            candidate_counts.append(len(candidates))
            doc_counts.append(len(abstracts))
            coverage_rows.append(_coverage_row(sample, candidates))
    finally:
        conn.close()

    with cache_path.open("wb") as handle:
        pickle.dump(chunk_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "fingerprint": fingerprint,
        "split": str(args.split),
        "raw_path": str(args.raw_path),
        "claim_atoms_jsonl": str(args.claim_atoms_jsonl),
        "corpus_sqlite": str(args.corpus_sqlite),
        "cache_path": str(cache_path),
        "n_samples": len(chunk_samples),
        "n_reference_samples": len(samples),
        "candidate_count": _numeric_summary(candidate_counts),
        "retrieved_doc_count": _numeric_summary(doc_counts),
        "gold_coverage": _coverage_summary(coverage_rows),
        "params": _fingerprint_payload(args),
        "sharding": {
            "mode": "single" if int(args.num_shards) <= 1 else "build_shard",
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "shard_suffix": suffix,
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / f"{args.split}{suffix}_coverage.jsonl", coverage_rows)
    if int(args.num_shards) <= 1:
        root = Path(args.output_root)
        (root / f"latest_{args.split}_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
        (root / f"latest_{args.split}_cache_path.txt").write_text(str(cache_path) + "\n", encoding="utf-8")
    print(f"Wrote SciFact ABC cache: {cache_path}")
    print(f"Wrote SciFact ABC manifest: {manifest_path}")
    return 0


def _read_jsonl(path: Path, *, sample_limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if sample_limit is not None and len(rows) >= int(sample_limit):
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_shards(
    args: argparse.Namespace,
    *,
    all_samples: list[Any],
    output_dir: Path,
    fingerprint: str,
    started_at: float,
) -> None:
    shard_samples: list[ChunkMMRSample] = []
    coverage_rows: list[dict[str, Any]] = []
    for shard_idx in range(int(args.num_shards)):
        suffix = _shard_suffix(int(args.num_shards), shard_idx)
        shard_path = output_dir / f"{args.split}{suffix}.pkl"
        coverage_path = output_dir / f"{args.split}{suffix}_coverage.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing SciFact ABC shard cache: {shard_path}")
        if not coverage_path.exists():
            raise FileNotFoundError(f"Missing SciFact ABC shard coverage: {coverage_path}")
        with shard_path.open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in shard cache {shard_path}, got {type(rows).__name__}")
        shard_samples.extend(rows)
        coverage_rows.extend(_read_jsonl(coverage_path))

    ordered_samples = _merge_by_event_order(
        reference_event_ids=[str(sample.event_id) for sample in all_samples],
        rows=shard_samples,
        event_id_fn=lambda row: str(row.event_id),
        kind="ABC cache samples",
    )
    ordered_coverage = _merge_by_event_order(
        reference_event_ids=[str(sample.event_id) for sample in all_samples],
        rows=coverage_rows,
        event_id_fn=lambda row: str(row.get("event_id") or ""),
        kind="ABC coverage rows",
    )
    candidate_counts = [len(sample.candidates) for sample in ordered_samples]
    doc_counts = [len({str(candidate.get("doc_id") or candidate.get("scifact_doc_id") or "") for candidate in sample.candidates}) for sample in ordered_samples]
    cache_path = output_dir / f"{args.split}.pkl"
    manifest_path = output_dir / f"{args.split}_manifest.json"
    with cache_path.open("wb") as handle:
        pickle.dump(ordered_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_jsonl(output_dir / f"{args.split}_coverage.jsonl", ordered_coverage)
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "mode": "merge_shards",
        "fingerprint": fingerprint,
        "split": str(args.split),
        "raw_path": str(args.raw_path),
        "claim_atoms_jsonl": str(args.claim_atoms_jsonl),
        "corpus_sqlite": str(args.corpus_sqlite),
        "cache_path": str(cache_path),
        "n_samples": len(ordered_samples),
        "candidate_count": _numeric_summary(candidate_counts),
        "retrieved_doc_count": _numeric_summary(doc_counts),
        "gold_coverage": _coverage_summary(ordered_coverage),
        "params": _fingerprint_payload(args),
        "sharding": {
            "mode": "merge_shards",
            "num_shards": int(args.num_shards),
            "shard_paths": [
                str(output_dir / f"{args.split}{_shard_suffix(int(args.num_shards), idx)}.pkl")
                for idx in range(int(args.num_shards))
            ],
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    root = Path(args.output_root)
    (root / f"latest_{args.split}_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
    (root / f"latest_{args.split}_cache_path.txt").write_text(str(cache_path) + "\n", encoding="utf-8")
    print(f"Merged SciFact ABC shards into: {cache_path}")
    print(f"events={len(ordered_samples)} shards={int(args.num_shards)}")


def _select_shard(rows: list[Any], *, num_shards: int, shard_index: int) -> list[Any]:
    n = max(1, int(num_shards))
    idx = int(shard_index)
    return [row for offset, row in enumerate(rows) if offset % n == idx]


def _merge_by_event_order(
    *,
    reference_event_ids: list[str],
    rows: list[Any],
    event_id_fn: Any,
    kind: str,
) -> list[Any]:
    by_event: dict[str, Any] = {}
    duplicates: list[str] = []
    for row in rows:
        event_id = event_id_fn(row)
        if event_id in by_event:
            duplicates.append(event_id)
        by_event[event_id] = row
    if duplicates:
        raise ValueError(f"Duplicate {kind} during shard merge: {duplicates[:10]}")
    missing = [event_id for event_id in reference_event_ids if event_id not in by_event]
    if missing:
        raise ValueError(f"Missing {kind} during shard merge: {missing[:10]}")
    return [by_event[event_id] for event_id in reference_event_ids]


def _validate_shard_args(args: argparse.Namespace) -> None:
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if bool(args.merge_shards) and int(args.num_shards) <= 1:
        raise ValueError("--merge-shards requires --num-shards > 1.")


def _shard_suffix(num_shards: int, shard_index: int) -> str:
    if int(num_shards) <= 1:
        return ""
    return f".shard-{int(shard_index):05d}-of-{int(num_shards):05d}"


def _retrieval_universe_doc_ids(
    conn: sqlite3.Connection,
    *,
    claim: str,
    atoms: list[dict[str, Any]],
    claim_doc_top_k: int,
    atom_doc_top_k: int,
    universe_doc_top_k: int,
    fts_row_limit_multiplier: int,
) -> tuple[list[str], dict[str, Any]]:
    claim_docs = _search_doc_ids(
        conn,
        claim,
        limit=claim_doc_top_k,
        fts_row_limit_multiplier=fts_row_limit_multiplier,
    )
    routes: list[dict[str, Any]] = []
    merged: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(claim_docs, start=1):
        doc_id = str(item["doc_id"])
        merged.setdefault(doc_id, {"doc_id": doc_id, "claim_rank": rank, "atom_hits": []})

    for atom_idx, atom in enumerate(atoms, start=1):
        atom_id = str(atom.get("atom_id") or f"A{atom_idx}")
        query = str(atom.get("query_rendering") or atom.get("proposition") or atom.get("text") or "")
        docs = _search_doc_ids(
            conn,
            query,
            limit=atom_doc_top_k,
            fts_row_limit_multiplier=fts_row_limit_multiplier,
        )
        atom_route = {
            "atom_id": atom_id,
            "query_rendering": query,
            "doc_ids": [str(item["doc_id"]) for item in docs],
        }
        routes.append(atom_route)
        for rank, item in enumerate(docs, start=1):
            doc_id = str(item["doc_id"])
            entry = merged.setdefault(doc_id, {"doc_id": doc_id, "claim_rank": None, "atom_hits": []})
            entry["atom_hits"].append({"atom_id": atom_id, "rank": rank, "score": float(item.get("score", 0.0))})

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            int(item["claim_rank"]) if item.get("claim_rank") is not None else 10**9,
            -len(item.get("atom_hits") or []),
            str(item.get("doc_id") or ""),
        ),
    )
    ordered = ordered[: max(1, int(universe_doc_top_k))]
    doc_ids = [str(item["doc_id"]) for item in ordered]
    return doc_ids, {
        "claim_route_doc_ids": [str(item["doc_id"]) for item in claim_docs],
        "atom_routes": routes,
        "doc_route_metadata": {str(item["doc_id"]): item for item in ordered},
    }


def _search_doc_ids(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    fts_row_limit_multiplier: int,
) -> list[dict[str, Any]]:
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    row_limit = max(int(limit) * max(int(fts_row_limit_multiplier), 1), int(limit))
    try:
        rows = conn.execute(
            """
            SELECT doc_id, bm25(sentence_fts) AS score
            FROM sentence_fts
            WHERE sentence_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, row_limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    best: dict[str, float] = {}
    for doc_id, score in rows:
        key = str(doc_id)
        value = float(score)
        if key not in best or value < best[key]:
            best[key] = value
    ordered = sorted(best.items(), key=lambda item: (item[1], item[0]))[: int(limit)]
    return [{"doc_id": doc_id, "score": score} for doc_id, score in ordered]


def _fts_query(text: str) -> str:
    tokens = []
    for token in word_tokens(text):
        token = "".join(ch for ch in token.lower() if ch.isalnum())
        if len(token) >= 2:
            tokens.append(token)
    tokens = list(dict.fromkeys(tokens))[:24]
    return " OR ".join(tokens)


def _load_abstracts(conn: sqlite3.Connection, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        row = conn.execute(
            "SELECT doc_id, title, abstract_json, structured FROM abstracts WHERE doc_id = ?",
            (str(doc_id),),
        ).fetchone()
        if row is None:
            continue
        out.append(
            {
                "doc_id": str(row[0]),
                "title": str(row[1] or ""),
                "abstract": json.loads(row[2]),
                "structured": bool(row[3]),
            }
        )
    return out


def _build_abc_candidates(
    abstracts: list[dict[str, Any]],
    *,
    claim: str,
    claim_embedding: np.ndarray,
    chunking: Any,
    route_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    route_by_doc = dict(route_metadata.get("doc_route_metadata") or {})
    for doc_order, abstract in enumerate(abstracts, start=1):
        doc_id = str(abstract["doc_id"])
        title = str(abstract.get("title") or "")
        sents = [str(sent).strip() for sent in abstract.get("abstract") or [] if str(sent).strip()]
        if not sents:
            continue
        chunks = chunking.chunks_from_presplit_with_context(
            sents,
            claim=claim,
            claim_embedding=claim_embedding,
        )
        route = dict(route_by_doc.get(doc_id) or {})
        for chunk_idx, chunk in enumerate(chunks, start=1):
            sent_indices = [int(idx) for idx in chunk.sent_indices]
            if not sent_indices:
                continue
            text = str(chunk.text).strip()
            key = f"{doc_id}:{','.join(str(idx) for idx in sent_indices)}:{canonicalize_sentence(text)}"
            if not text or key in seen:
                continue
            seen.add(key)
            metadata = dict(chunk.metadata or {})
            anchor_idx = int(metadata.get("anchor_sent_idx", sent_indices[0]))
            candidates.append(
                {
                    "candidate_uid": f"scifact:{doc_id}:{'-'.join(str(idx) for idx in sent_indices)}",
                    "report_id": doc_id,
                    "doc_id": doc_id,
                    "scifact_doc_id": doc_id,
                    "sent_idx": anchor_idx,
                    "chunk_sent_indices": sent_indices,
                    "scifact_sentence_ids": sent_indices,
                    "text": text,
                    "title": title,
                    "source_report": title or doc_id,
                    "raw_report_order": int(doc_order),
                    "raw_sent_order": int(anchor_idx),
                    "structured_abstract": bool(abstract.get("structured")),
                    "abc_chunk_index": int(chunk_idx),
                    "abc_metadata": metadata,
                    "claim_route_rank": route.get("claim_rank"),
                    "atom_route_hits": route.get("atom_hits") or [],
                    "retrieval_universe_source": "scifact_open_corpus_atom_union",
                }
            )
    return candidates


def _encode_candidates(embedder: TextEmbedder, candidates: list[dict[str, Any]], *, claim_emb: np.ndarray) -> np.ndarray:
    if not candidates:
        return np.zeros((0, int(claim_emb.reshape(-1).shape[0])), dtype=np.float32)
    vectors = embedder.encode([str(candidate.get("text") or "") for candidate in candidates], is_query=False)
    return np.asarray(vectors, dtype=np.float32)


def _coverage_row(sample: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", {}) or {}
    evidence = metadata.get("evidence") or {}
    gold_doc_ids: set[str] = set()
    gold_sentence_keys: set[str] = set()
    if isinstance(evidence, dict):
        for doc_id, rationales in evidence.items():
            doc_key = str(doc_id)
            gold_doc_ids.add(doc_key)
            if isinstance(rationales, list):
                for rationale in rationales:
                    if not isinstance(rationale, dict):
                        continue
                    for sent_idx in rationale.get("sentences") or []:
                        gold_sentence_keys.add(f"{doc_key}:{int(sent_idx)}")
    candidate_doc_ids = {str(candidate.get("scifact_doc_id") or candidate.get("doc_id") or "") for candidate in candidates}
    candidate_sentence_keys = {
        f"{candidate.get('scifact_doc_id') or candidate.get('doc_id')}:{int(sent_idx)}"
        for candidate in candidates
        for sent_idx in (candidate.get("scifact_sentence_ids") or candidate.get("chunk_sent_indices") or [])
        if str(candidate.get("scifact_doc_id") or candidate.get("doc_id") or "")
    }
    return {
        "event_id": sample.event_id,
        "label": sample.label,
        "n_candidates": len(candidates),
        "gold_doc_count": len(gold_doc_ids),
        "gold_sentence_count": len(gold_sentence_keys),
        "hit_gold_doc_count": len(gold_doc_ids & candidate_doc_ids),
        "hit_gold_sentence_count": len(gold_sentence_keys & candidate_sentence_keys),
        "gold_doc_recall": _safe_div(len(gold_doc_ids & candidate_doc_ids), len(gold_doc_ids)),
        "gold_sentence_recall": _safe_div(len(gold_sentence_keys & candidate_sentence_keys), len(gold_sentence_keys)),
    }


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_gold_docs = sum(int(row["gold_doc_count"]) for row in rows)
    total_hit_docs = sum(int(row["hit_gold_doc_count"]) for row in rows)
    total_gold_sentences = sum(int(row["gold_sentence_count"]) for row in rows)
    total_hit_sentences = sum(int(row["hit_gold_sentence_count"]) for row in rows)
    return {
        "row_count": len(rows),
        "rows_with_gold": sum(1 for row in rows if int(row["gold_doc_count"]) > 0),
        "micro_gold_doc_recall": _safe_div(total_hit_docs, total_gold_docs),
        "micro_gold_sentence_recall": _safe_div(total_hit_sentences, total_gold_sentences),
        "full_doc_coverage_count": sum(
            1 for row in rows if int(row["gold_doc_count"]) > 0 and row["hit_gold_doc_count"] == row["gold_doc_count"]
        ),
        "full_sentence_coverage_count": sum(
            1
            for row in rows
            if int(row["gold_sentence_count"]) > 0 and row["hit_gold_sentence_count"] == row["gold_sentence_count"]
        ),
    }


def _numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": len(values),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _chunking_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "strategy": "abc_claim_aware",
        "boundary_mode": str(args.boundary_mode),
        "boundary_threshold": float(args.boundary_threshold),
        "lambda_std": float(args.lambda_std),
        "w_sem": float(args.w_sem),
        "w_rel": float(args.w_rel),
        "max_sent_per_chunk": int(args.max_sent_per_chunk),
        "max_tokens_per_chunk": int(args.max_tokens_per_chunk),
        "min_tokens_per_chunk": int(args.min_tokens_per_chunk),
        "single_sentence_relevance_threshold": float(args.single_sentence_relevance_threshold),
        "high_rel_threshold": float(args.high_rel_threshold),
        "embedder_model": str(args.embedder_model),
        "device": str(args.device),
        "max_length": int(args.embedder_max_length),
        "batch_size": int(args.embedder_batch_size),
        "precision": str(args.precision),
    }


def _retrieval_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "embedder_model": str(args.embedder_model),
        "device": str(args.device),
        "max_length": int(args.embedder_max_length),
        "batch_size": int(args.embedder_batch_size),
        "precision": str(args.precision),
    }


def _fingerprint(args: argparse.Namespace) -> str:
    encoded = json.dumps(_fingerprint_payload(args), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _fingerprint_payload(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "embedder_model",
        "embedder_max_length",
        "precision",
        "claim_doc_top_k",
        "atom_doc_top_k",
        "universe_doc_top_k",
        "boundary_mode",
        "boundary_threshold",
        "lambda_std",
        "w_sem",
        "w_rel",
        "max_sent_per_chunk",
        "max_tokens_per_chunk",
        "min_tokens_per_chunk",
        "single_sentence_relevance_threshold",
        "high_rel_threshold",
        "sample_limit",
    ]
    return {key: getattr(args, key) for key in keys}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


if __name__ == "__main__":
    raise SystemExit(main())
