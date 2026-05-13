from __future__ import annotations

from typing import Any

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, minmax_scale
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)


CHUNK_EMBEDDING_FEATURE_MODE = "chunk_embedding"


def _candidate_count(sample: ChunkMMRSample) -> int:
    return min(len(sample.candidates), int(sample.chunk_emb.shape[0]))


def _resolve_candidate_capacity(samples: list[ChunkMMRSample], candidate_top_k: int | None) -> int:
    if candidate_top_k is not None:
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive when provided.")
        return int(candidate_top_k)

    max_candidates = max((_candidate_count(sample) for sample in samples), default=0)
    return max(1, max_candidates)


def _chunk_scores(
    sample: ChunkMMRSample,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(sample.candidates), int(sample.chunk_emb.shape[0]))
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    chunk_emb = np.asarray(sample.chunk_emb[:n], dtype=np.float32)
    claim_emb = np.asarray(sample.claim_emb, dtype=np.float32).reshape(-1)
    dense_scores = chunk_emb @ claim_emb

    q_ctr, q_len = content_tokens_counter(sample.claim)
    lexical_scores = np.empty(n, dtype=np.float32)
    bm25_scores = np.empty(n, dtype=np.float32)
    for j, candidate in enumerate(sample.candidates[:n]):
        s_ctr, s_len = content_tokens_counter(str(candidate.get("text", "")))
        lexical_scores[j] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
        bm25_scores[j] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

    hybrid_scores = (
        alpha_dense * minmax_scale(dense_scores)
        + alpha_lexical * minmax_scale(lexical_scores)
        + alpha_bm25 * minmax_scale(bm25_scores)
    )
    ranked = np.argsort(-hybrid_scores).astype(np.int64, copy=False)
    return hybrid_scores.astype(np.float32, copy=False), ranked


def chunk_embedding_example(
    sample: ChunkMMRSample,
    *,
    candidate_top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    claim_emb = np.asarray(sample.claim_emb, dtype=np.float32).reshape(-1)
    dim = int(claim_emb.shape[0])
    candidate_emb = np.zeros((candidate_top_k, dim), dtype=np.float32)
    candidate_mask = np.zeros(candidate_top_k, dtype=np.float32)

    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive.")

    _scores, ranked = _chunk_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    keep = ranked[:candidate_top_k]
    if keep.size:
        selected = np.asarray(sample.chunk_emb[keep], dtype=np.float32)
        candidate_emb[: selected.shape[0]] = selected
        candidate_mask[: selected.shape[0]] = 1.0

    return claim_emb, candidate_emb, candidate_mask


def build_chunk_embedding_arrays(
    samples: list[ChunkMMRSample],
    *,
    candidate_top_k: int | None,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("No chunk-MMR samples provided.")

    candidate_capacity = _resolve_candidate_capacity(samples, candidate_top_k)
    claim_list: list[np.ndarray] = []
    candidate_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    count_list: list[int] = []
    event_ids: list[str] = []
    for sample in samples:
        claim_emb, candidate_emb, candidate_mask = chunk_embedding_example(
            sample,
            candidate_top_k=candidate_capacity,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
        )
        claim_list.append(claim_emb)
        candidate_list.append(candidate_emb)
        mask_list.append(candidate_mask)
        count_list.append(_candidate_count(sample))
        event_ids.append(sample.event_id)

    return {
        "event_ids": np.array(event_ids, dtype=object),
        "claim_emb": np.stack(claim_list).astype(np.float32, copy=False),
        "candidate_emb": np.stack(candidate_list).astype(np.float32, copy=False),
        "candidate_mask": np.stack(mask_list).astype(np.float32, copy=False),
        "candidate_counts": np.array(count_list, dtype=np.int64),
    }


def build_matched_chunk_embedding_arrays(
    samples: list[ChunkMMRSample],
    oracle_by_eid: dict[str, dict[str, Any]],
    *,
    candidate_top_k: int | None,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], int]:
    matched: list[ChunkMMRSample] = []
    targets: list[float] = []
    oracle_records: list[dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        oracle_rec = oracle_by_eid.get(sample.event_id)
        if oracle_rec is None:
            skipped += 1
            continue
        matched.append(sample)
        targets.append(float(oracle_rec["oracle_lambda"]))
        oracle_records.append(oracle_rec)

    if not matched:
        raise ValueError("No chunk-MMR samples matched oracle λ values by event_id.")

    arrays = build_chunk_embedding_arrays(
        matched,
        candidate_top_k=candidate_top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
    )
    return arrays, np.array(targets, dtype=np.float32), oracle_records, skipped
