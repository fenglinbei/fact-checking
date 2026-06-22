from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, canonicalize_sentence
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.selectors.question_decomp_retrieval import (
    _compute_prediction_metrics,
    build_baseline_claim_mmr_row,
    oracle_selected_texts_by_event,
    score_question_against_sample,
)


@dataclass(frozen=True)
class AtomRetrievalParams:
    per_atom_keep: int = 20
    merged_pool_size: int = 15
    selector_top_k: int = 5
    rrf_k: float = 60.0
    atom_weight: float = 1.0
    merge_mmr_lambda: float = 0.70
    alpha_dense: float = 0.70
    alpha_lexical: float = 0.20
    alpha_bm25: float = 0.10


def align_atoms_and_chunks(
    atom_rows: Sequence[dict[str, Any]],
    chunk_samples: Sequence[ChunkMMRSample],
) -> list[tuple[dict[str, Any], ChunkMMRSample]]:
    by_event = {str(sample.event_id): sample for sample in chunk_samples}
    pairs: list[tuple[dict[str, Any], ChunkMMRSample]] = []
    missing: list[str] = []
    for row in atom_rows:
        event_id = str(row.get("event_id") or "")
        sample = by_event.get(event_id)
        if sample is None:
            missing.append(event_id)
            continue
        pairs.append((row, sample))
    if missing:
        raise ValueError(f"Missing {len(missing)} atom event(s) in chunk cache, sample={missing[:5]}")
    return pairs


def score_atom_routes(
    sample: ChunkMMRSample,
    *,
    atom: dict[str, Any],
    atom_embedding: np.ndarray,
    params: AtomRetrievalParams,
) -> list[dict[str, Any]]:
    query_text = _atom_query_text(atom)
    scored = score_question_against_sample(
        sample,
        query_text=query_text,
        query_embedding=atom_embedding,
        alpha_dense=float(params.alpha_dense),
        alpha_lexical=float(params.alpha_lexical),
        alpha_bm25=float(params.alpha_bm25),
    )
    n = int(scored["n"])
    if n == 0:
        return []
    keep = min(int(params.per_atom_keep), n)
    order = np.argsort(-scored["hybrid_scores"])[:keep]
    atom_id = str(atom.get("atom_id") or "")
    routes: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        routes.append(
            {
                "candidate_idx": idx,
                "atom_id": atom_id,
                "proposition": str(atom.get("proposition") or atom.get("text") or ""),
                "query_rendering": query_text,
                "rank": int(rank),
                "dense_score": float(scored["dense_scores"][idx]),
                "lexical_score": float(scored["lexical_scores"][idx]),
                "bm25_score": float(scored["bm25_scores"][idx]),
                "hybrid_score": float(scored["hybrid_scores"][idx]),
            }
        )
    return routes


def build_atom_conditioned_retrieval_row(
    atom_row: dict[str, Any],
    sample: ChunkMMRSample,
    *,
    atom_embeddings: dict[tuple[str, str], np.ndarray],
    params: AtomRetrievalParams,
) -> dict[str, Any]:
    atoms = list(atom_row.get("claim_atoms") or [])
    all_routes: list[dict[str, Any]] = []
    per_atom_routes: list[dict[str, Any]] = []
    event_id = str(atom_row.get("event_id") or sample.event_id)
    for idx, atom in enumerate(atoms, start=1):
        atom_id = str(atom.get("atom_id") or f"A{idx}")
        emb = atom_embeddings[(event_id, atom_id)]
        routes = score_atom_routes(sample, atom=atom, atom_embedding=emb, params=params)
        all_routes.extend(routes)
        per_atom_routes.append(
            {
                "atom_id": atom_id,
                "proposition": str(atom.get("proposition") or atom.get("text") or ""),
                "query_rendering": _atom_query_text(atom),
                "routes": routes,
            }
        )

    pool = merge_atom_routes(sample, all_routes, params=params)
    selected = select_final_candidates_with_mmr(sample, pool, params=params)
    return {
        "event_id": event_id,
        "claim": str(atom_row.get("claim") or sample.claim),
        "label": getattr(sample, "label", ""),
        "gold_label": str(atom_row.get("gold_label") or getattr(sample, "label", "")),
        "claim_atoms": atoms,
        "atom_routes": per_atom_routes,
        "merged_candidate_pool": pool,
        "selected_evidence": selected,
    }


def merge_atom_routes(
    sample: ChunkMMRSample,
    routes: Sequence[dict[str, Any]],
    *,
    params: AtomRetrievalParams,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for route in routes:
        candidate_idx = int(route["candidate_idx"])
        if candidate_idx < 0 or candidate_idx >= len(sample.candidates):
            continue
        candidate = sample.candidates[candidate_idx]
        text = str(candidate.get("text") or "")
        key = canonicalize_sentence(text)
        if not key:
            continue
        rrf_part = float(params.atom_weight) / (float(params.rrf_k) + int(route.get("rank", 0)))
        item = merged.get(key)
        route_copy = dict(route)
        if item is None:
            item = dict(candidate)
            item.update(
                {
                    "text": text,
                    "canonical_text": key,
                    "original_candidate_idx": candidate_idx,
                    "atom_rrf_score": 0.0,
                    "atom_max_route_hybrid": float(route.get("hybrid_score", 0.0)),
                    "atom_route_hit_count": 0,
                    "atom_routes": [],
                    "matched_atom_ids": [],
                }
            )
            merged[key] = item
        item["atom_rrf_score"] = float(item["atom_rrf_score"]) + float(rrf_part)
        item["atom_routes"].append(route_copy)
        item["matched_atom_ids"] = _ordered_unique(str(r.get("atom_id") or "") for r in item["atom_routes"])
        item["atom_route_hit_count"] = len(item["matched_atom_ids"])
        if float(route.get("hybrid_score", 0.0)) > float(item.get("atom_max_route_hybrid", 0.0)):
            item["atom_max_route_hybrid"] = float(route.get("hybrid_score", 0.0))
            item["original_candidate_idx"] = candidate_idx

    pool = list(merged.values())
    pool.sort(
        key=lambda item: (
            -float(item.get("atom_rrf_score", 0.0)),
            -float(item.get("atom_max_route_hybrid", 0.0)),
            -int(item.get("atom_route_hit_count", 0)),
            int(item.get("original_candidate_idx", 10**9)),
        )
    )
    for rank, candidate in enumerate(pool[: int(params.merged_pool_size)], start=1):
        candidate["merge_rank"] = rank
        candidate["atom_pool_rank"] = rank
    return pool[: int(params.merged_pool_size)]


def select_final_candidates_with_mmr(
    sample: ChunkMMRSample,
    pool: Sequence[dict[str, Any]],
    *,
    params: AtomRetrievalParams,
) -> list[dict[str, Any]]:
    if not pool:
        return []
    indices = [int(candidate.get("original_candidate_idx", -1)) for candidate in pool]
    valid_positions = [
        pos
        for pos, idx in enumerate(indices)
        if idx >= 0 and idx < int(sample.chunk_emb.shape[0])
    ]
    if not valid_positions:
        selected = [dict(candidate) for candidate in pool[: int(params.selector_top_k)]]
    else:
        query_scores = np.asarray(
            [float(pool[pos].get("atom_rrf_score", 0.0)) for pos in valid_positions],
            dtype=np.float32,
        )
        vectors = np.asarray([sample.chunk_emb[indices[pos]] for pos in valid_positions], dtype=np.float32)
        local_selected = maximal_marginal_relevance(
            query_scores=query_scores,
            sentence_vectors=vectors,
            top_k=min(int(params.selector_top_k), len(valid_positions)),
            lambda_weight=float(params.merge_mmr_lambda),
        )
        selected = [dict(pool[valid_positions[int(pos)]]) for pos in local_selected]
    for rank, candidate in enumerate(selected, start=1):
        candidate["selection_rank"] = rank
    return selected


def compute_retrieval_metrics(
    *,
    atom_rows: Sequence[dict[str, Any]],
    atom_retrieval_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    oracle_texts: dict[str, set[str]],
) -> dict[str, Any]:
    atom_metrics = _compute_prediction_metrics(
        pool_rows=[
            {
                "event_id": row["event_id"],
                "pool": row.get("merged_candidate_pool") or [],
                "selected": row.get("selected_evidence") or [],
            }
            for row in atom_retrieval_rows
        ],
        oracle_texts=oracle_texts,
        include_pool=True,
    )
    baseline_metrics = _compute_prediction_metrics(
        pool_rows=[
            {
                "event_id": row["event_id"],
                "pool": row.get("candidates") or [],
                "selected": row.get("candidates") or [],
            }
            for row in baseline_rows
        ],
        oracle_texts=oracle_texts,
        include_pool=False,
    )
    atom_counts = [len(row.get("claim_atoms") or []) for row in atom_rows]
    atom_metrics["mean_atoms_per_claim"] = float(np.mean(atom_counts)) if atom_counts else 0.0
    route_counts = [
        len(candidate.get("atom_routes") or [])
        for row in atom_retrieval_rows
        for candidate in (row.get("selected_evidence") or [])
    ]
    atom_metrics["mean_routes_per_selected_candidate"] = float(np.mean(route_counts)) if route_counts else 0.0
    return {
        "atom_conditioned_retrieval": atom_metrics,
        "baseline_claim_mmr": baseline_metrics,
        "delta": _metric_delta(atom_metrics, baseline_metrics),
    }


def _atom_query_text(atom: dict[str, Any]) -> str:
    return str(atom.get("query_rendering") or atom.get("proposition") or atom.get("text") or "")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    keys = [
        "oracle_pool_recall@15",
        "oracle_selected_recall@5",
        "jaccard@5",
        "top1_match",
        "any_oracle_hit@15",
        "all_oracle_hit@15",
    ]
    return {
        key: float(left.get(key, 0.0)) - float(right.get(key, 0.0))
        for key in keys
    }


__all__ = [
    "AtomRetrievalParams",
    "align_atoms_and_chunks",
    "build_atom_conditioned_retrieval_row",
    "build_baseline_claim_mmr_row",
    "compute_retrieval_metrics",
    "merge_atom_routes",
    "oracle_selected_texts_by_event",
    "score_atom_routes",
    "select_final_candidates_with_mmr",
]
