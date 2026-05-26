from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from fact_checking.build.candidates import (
    ChunkMMRSample,
    canonicalize_sentence,
    compute_hybrid_scores,
    minmax_scale,
)
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)


@dataclass(frozen=True)
class RetrievalParams:
    per_question_keep: int = 20
    merged_pool_size: int = 15
    selector_top_k: int = 5
    rrf_k: float = 60.0
    q1_weight: float = 1.2
    other_question_weight: float = 1.0
    merge_mmr_lambda: float = 0.70
    alpha_dense: float = 0.70
    alpha_lexical: float = 0.20
    alpha_bm25: float = 0.10


def align_questions_and_chunks(
    question_rows: Sequence[dict[str, Any]],
    chunk_samples: Sequence[ChunkMMRSample],
) -> list[tuple[dict[str, Any], ChunkMMRSample]]:
    by_event = {str(sample.event_id): sample for sample in chunk_samples}
    pairs: list[tuple[dict[str, Any], ChunkMMRSample]] = []
    missing: list[str] = []
    for row in question_rows:
        event_id = str(row.get("event_id") or "")
        sample = by_event.get(event_id)
        if sample is None:
            missing.append(event_id)
            continue
        pairs.append((row, sample))
    if missing:
        raise ValueError(f"Missing {len(missing)} question event(s) in chunk cache, sample={missing[:5]}")
    return pairs


def score_question_routes(
    sample: ChunkMMRSample,
    *,
    question: dict[str, Any],
    question_embedding: np.ndarray,
    params: RetrievalParams,
) -> list[dict[str, Any]]:
    scored = score_question_against_sample(
        sample,
        query_text=str(question.get("question") or ""),
        query_embedding=question_embedding,
        alpha_dense=float(params.alpha_dense),
        alpha_lexical=float(params.alpha_lexical),
        alpha_bm25=float(params.alpha_bm25),
    )
    n = int(scored["n"])
    if n == 0:
        return []
    keep = min(int(params.per_question_keep), n)
    order = np.argsort(-scored["hybrid_scores"])[:keep]
    routes: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        routes.append(
            {
                "candidate_idx": idx,
                "question_id": str(question.get("id") or f"q{rank}"),
                "question": str(question.get("question") or ""),
                "focus": str(question.get("focus") or "other"),
                "rank": int(rank),
                "dense_score": float(scored["dense_scores"][idx]),
                "lexical_score": float(scored["lexical_scores"][idx]),
                "bm25_score": float(scored["bm25_scores"][idx]),
                "hybrid_score": float(scored["hybrid_scores"][idx]),
            }
        )
    return routes


def score_question_against_sample(
    sample: ChunkMMRSample,
    *,
    query_text: str,
    query_embedding: np.ndarray,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> dict[str, Any]:
    n = min(len(sample.candidates), int(sample.chunk_emb.shape[0]))
    if n == 0:
        return {
            "n": 0,
            "chunk_emb": np.zeros((0, 0), dtype=np.float32),
            "dense_scores": np.zeros((0,), dtype=np.float32),
            "lexical_scores": np.zeros((0,), dtype=np.float32),
            "bm25_scores": np.zeros((0,), dtype=np.float32),
            "hybrid_scores": np.zeros((0,), dtype=np.float32),
        }
    chunk_emb = np.asarray(sample.chunk_emb[:n], dtype=np.float32)
    query_emb = np.asarray(question_vector(query_embedding), dtype=np.float32)
    dense_scores = chunk_emb @ query_emb

    q_ctr, q_len = content_tokens_counter(str(query_text or ""))
    lexical_scores = np.empty(n, dtype=np.float32)
    bm25_scores = np.empty(n, dtype=np.float32)
    for idx, candidate in enumerate(sample.candidates[:n]):
        s_ctr, s_len = content_tokens_counter(str(candidate.get("text", "")))
        lexical_scores[idx] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
        bm25_scores[idx] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

    hybrid_scores = (
        float(alpha_dense) * minmax_scale(dense_scores)
        + float(alpha_lexical) * minmax_scale(lexical_scores)
        + float(alpha_bm25) * minmax_scale(bm25_scores)
    )
    return {
        "n": int(n),
        "chunk_emb": chunk_emb,
        "dense_scores": dense_scores.astype(np.float32, copy=False),
        "lexical_scores": lexical_scores.astype(np.float32, copy=False),
        "bm25_scores": bm25_scores.astype(np.float32, copy=False),
        "hybrid_scores": hybrid_scores.astype(np.float32, copy=False),
    }


def question_vector(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected one query embedding, got shape={arr.shape}.")
        arr = arr[0]
    return arr.reshape(-1)


def build_question_decomp_retrieval_row(
    question_row: dict[str, Any],
    sample: ChunkMMRSample,
    *,
    question_embeddings: dict[tuple[str, str], np.ndarray],
    params: RetrievalParams,
) -> dict[str, Any]:
    questions = list(question_row.get("questions") or [])
    all_routes: list[dict[str, Any]] = []
    per_question_routes: list[dict[str, Any]] = []
    event_id = str(question_row.get("event_id") or sample.event_id)
    for idx, question in enumerate(questions, start=1):
        question_id = str(question.get("id") or f"q{idx}")
        emb = question_embeddings[(event_id, question_id)]
        routes = score_question_routes(sample, question=question, question_embedding=emb, params=params)
        all_routes.extend(routes)
        per_question_routes.append(
            {
                "question_id": question_id,
                "question": str(question.get("question") or ""),
                "focus": str(question.get("focus") or "other"),
                "routes": routes,
            }
        )

    pool = merge_question_routes(sample, all_routes, params=params)
    selected = select_final_candidates_with_mmr(sample, pool, params=params)
    return {
        "event_id": event_id,
        "claim": str(question_row.get("claim") or sample.claim),
        "label": getattr(sample, "label", ""),
        "questions": questions,
        "question_routes": per_question_routes,
        "merged_candidate_pool": pool,
        "selected_evidence": selected,
    }


def merge_question_routes(
    sample: ChunkMMRSample,
    routes: Sequence[dict[str, Any]],
    *,
    params: RetrievalParams,
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
        weight = float(params.q1_weight) if str(route.get("question_id") or "") == "q1" else float(params.other_question_weight)
        rrf_part = weight / (float(params.rrf_k) + int(route.get("rank", 0)))
        item = merged.get(key)
        route_copy = dict(route)
        if item is None:
            item = dict(candidate)
            item.update(
                {
                    "text": text,
                    "canonical_text": key,
                    "original_candidate_idx": candidate_idx,
                    "rrf_score": 0.0,
                    "max_question_hybrid": float(route.get("hybrid_score", 0.0)),
                    "question_hit_count": 0,
                    "question_routes": [],
                }
            )
            merged[key] = item
        item["rrf_score"] = float(item["rrf_score"]) + float(rrf_part)
        item["question_routes"].append(route_copy)
        item["question_hit_count"] = len({str(r.get("question_id") or "") for r in item["question_routes"]})
        if float(route.get("hybrid_score", 0.0)) > float(item.get("max_question_hybrid", 0.0)):
            item["max_question_hybrid"] = float(route.get("hybrid_score", 0.0))
            item["original_candidate_idx"] = candidate_idx

    pool = list(merged.values())
    pool.sort(
        key=lambda item: (
            -float(item.get("rrf_score", 0.0)),
            -float(item.get("max_question_hybrid", 0.0)),
            -int(item.get("question_hit_count", 0)),
            int(item.get("original_candidate_idx", 10**9)),
        )
    )
    for rank, candidate in enumerate(pool[: int(params.merged_pool_size)], start=1):
        candidate["merge_rank"] = rank
    return pool[: int(params.merged_pool_size)]


def select_final_candidates_with_mmr(
    sample: ChunkMMRSample,
    pool: Sequence[dict[str, Any]],
    *,
    params: RetrievalParams,
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
        selected = list(pool[: int(params.selector_top_k)])
    else:
        query_scores = np.asarray([float(pool[pos].get("rrf_score", 0.0)) for pos in valid_positions], dtype=np.float32)
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


def build_baseline_claim_mmr_row(
    sample: ChunkMMRSample,
    *,
    params: RetrievalParams,
) -> dict[str, Any]:
    scored = compute_hybrid_scores(sample, params.alpha_dense, params.alpha_lexical, params.alpha_bm25)
    n = int(scored["n"])
    if n == 0:
        candidates: list[dict[str, Any]] = []
    else:
        selected_indices = maximal_marginal_relevance(
            query_scores=scored["hybrid_scores"],
            sentence_vectors=scored["chunk_emb"],
            top_k=min(int(params.selector_top_k), n),
            lambda_weight=float(params.merge_mmr_lambda),
        )
        candidates = []
        seen: set[str] = set()
        for rank, idx in enumerate(selected_indices, start=1):
            idx = int(idx)
            candidate = dict(sample.candidates[idx])
            key = canonicalize_sentence(str(candidate.get("text") or ""))
            if not key or key in seen:
                continue
            seen.add(key)
            candidate.update(
                {
                    "canonical_text": key,
                    "original_candidate_idx": idx,
                    "dense_score": float(scored["dense_scores"][idx]),
                    "lexical_score": float(scored["lexical_scores"][idx]),
                    "bm25_score": float(scored["bm25_scores"][idx]),
                    "hybrid_score": float(scored["hybrid_scores"][idx]),
                    "selection_rank": len(candidates) + 1,
                    "mmr_rank": rank,
                }
            )
            candidates.append(candidate)
            if len(candidates) >= int(params.selector_top_k):
                break
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "candidates": candidates,
    }


def oracle_selected_texts_by_event(oracle_rows: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in oracle_rows:
        event_id = str(row.get("event_id") or "")
        pool = row.get("candidate_pool") or []
        selected = row.get("selected_indices") or []
        texts: set[str] = set()
        for raw_idx in selected:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(pool):
                text = canonicalize_sentence(str((pool[idx] or {}).get("text") or ""))
                if text:
                    texts.add(text)
        if event_id:
            result[event_id] = texts
    return result


def compute_retrieval_metrics(
    *,
    question_rows: Sequence[dict[str, Any]],
    qd_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    oracle_texts: dict[str, set[str]],
) -> dict[str, Any]:
    qd_metrics = _compute_prediction_metrics(
        pool_rows=[
            {
                "event_id": row["event_id"],
                "pool": row.get("merged_candidate_pool") or [],
                "selected": row.get("selected_evidence") or [],
            }
            for row in qd_rows
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
    q_counts = [len(row.get("questions") or []) for row in question_rows]
    qd_metrics["mean_questions_per_claim"] = float(np.mean(q_counts)) if q_counts else 0.0
    route_counts = [
        len(candidate.get("question_routes") or [])
        for row in qd_rows
        for candidate in (row.get("selected_evidence") or [])
    ]
    qd_metrics["mean_routes_per_selected_candidate"] = float(np.mean(route_counts)) if route_counts else 0.0
    focus_counts = Counter()
    for row in qd_rows:
        for candidate in row.get("selected_evidence") or []:
            for route in candidate.get("question_routes") or []:
                focus_counts[str(route.get("focus") or "other")] += 1
    total_focus = sum(focus_counts.values())
    qd_metrics["question_focus_contribution@5"] = {
        "counts": dict(sorted(focus_counts.items())),
        "share": {
            key: float(value / total_focus) if total_focus else 0.0
            for key, value in sorted(focus_counts.items())
        },
    }
    return {
        "question_decomp": qd_metrics,
        "baseline_claim_mmr": baseline_metrics,
        "delta": _metric_delta(qd_metrics, baseline_metrics),
    }


def _compute_prediction_metrics(
    *,
    pool_rows: Sequence[dict[str, Any]],
    oracle_texts: dict[str, set[str]],
    include_pool: bool,
) -> dict[str, Any]:
    n = 0
    total_oracle = 0
    pool_hits = 0
    selected_hits = 0
    any_hit = 0
    all_hit = 0
    top1_match = 0
    jaccards: list[float] = []
    for row in pool_rows:
        event_id = str(row.get("event_id") or "")
        oracle = set(oracle_texts.get(event_id) or set())
        if not oracle:
            continue
        n += 1
        total_oracle += len(oracle)
        pool = _candidate_text_set(row.get("pool") or [])
        selected = _candidate_text_set(row.get("selected") or [])
        pool_for_recall = pool if include_pool else selected
        pool_hit_set = oracle & pool_for_recall
        selected_hit_set = oracle & selected
        pool_hits += len(pool_hit_set)
        selected_hits += len(selected_hit_set)
        if pool_hit_set:
            any_hit += 1
        if oracle <= pool_for_recall:
            all_hit += 1
        ordered_selected = row.get("selected") or []
        if ordered_selected:
            top1 = canonicalize_sentence(str(ordered_selected[0].get("text") or ""))
            if top1 in oracle:
                top1_match += 1
        union = oracle | selected
        jaccards.append(float(len(oracle & selected) / len(union)) if union else 0.0)
    denom_oracle = max(total_oracle, 1)
    denom_events = max(n, 1)
    return {
        "n_events": int(n),
        "total_oracle_evidence": int(total_oracle),
        "oracle_pool_recall@15": float(pool_hits / denom_oracle),
        "oracle_selected_recall@5": float(selected_hits / denom_oracle),
        "jaccard@5": float(np.mean(jaccards)) if jaccards else 0.0,
        "top1_match": float(top1_match / denom_events),
        "any_oracle_hit@15": float(any_hit / denom_events),
        "all_oracle_hit@15": float(all_hit / denom_events),
    }


def _candidate_text_set(candidates: Iterable[dict[str, Any]]) -> set[str]:
    texts: set[str] = set()
    for candidate in candidates:
        text = canonicalize_sentence(str(candidate.get("text") or ""))
        if text:
            texts.add(text)
    return texts


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
