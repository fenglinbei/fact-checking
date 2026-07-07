from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, canonicalize_sentence, compute_hybrid_scores
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.selectors.atom_retrieval_union import (
    AtomUnionSelectionParams,
    rank_atom_union_source_score_candidates,
    select_atom_union_rules,
)


SELECTOR_MECHANISM_GRAPH_VERSION = "selector_mechanism_ablation_v0"
SELECTOR_MECHANISM_ADAPTIVE_POLICY = "fixed_top5"

SELECTOR_MECH_S0_NO_EVIDENCE = "selector_mech_s0_no_evidence"
SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5 = "selector_mech_s1_claim_pool_random_top5"
SELECTOR_MECH_S2_CLAIM_POOL_HYBRID_TOP5 = "selector_mech_s2_claim_pool_hybrid_top5"
SELECTOR_MECH_S3_CLAIM_POOL_HYBRID_MMR_TOP5 = "selector_mech_s3_claim_pool_hybrid_mmr_top5"
SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5 = "selector_mech_s4_atom_union_source_score_top5"
SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED = "selector_mech_s4_atom_union_source_score_ordered"

SELECTOR_MECHANISM_NAMES = (
    SELECTOR_MECH_S0_NO_EVIDENCE,
    SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5,
    SELECTOR_MECH_S2_CLAIM_POOL_HYBRID_TOP5,
    SELECTOR_MECH_S3_CLAIM_POOL_HYBRID_MMR_TOP5,
    SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
    SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED,
)


@dataclass(frozen=True)
class SelectorMechanismParams:
    top_k: int = 5
    claim_pool_top_n: int = 20
    random_seed: int = 0
    merge_mmr_lambda: float = 0.70
    alpha_dense: float = 0.70
    alpha_lexical: float = 0.20
    alpha_bm25: float = 0.10


def build_claim_candidate_pool_row(
    sample: ChunkMMRSample,
    *,
    params: SelectorMechanismParams,
) -> dict[str, Any]:
    scored = compute_hybrid_scores(
        sample,
        float(params.alpha_dense),
        float(params.alpha_lexical),
        float(params.alpha_bm25),
    )
    n = int(scored["n"])
    if n == 0:
        candidates: list[dict[str, Any]] = []
    else:
        ordered = np.argsort(-np.asarray(scored["hybrid_scores"], dtype=np.float32))
        candidates = []
        seen: set[str] = set()
        for rank, raw_idx in enumerate(ordered, start=1):
            raw_idx = int(raw_idx)
            candidate = dict(sample.candidates[raw_idx])
            key = canonicalize_sentence(str(candidate.get("text") or ""))
            if not key or key in seen:
                continue
            seen.add(key)
            candidate.update(
                {
                    "canonical_text": key,
                    "candidate_idx": len(candidates),
                    "selector_candidate_idx": len(candidates),
                    "original_candidate_idx": raw_idx,
                    "dense_score": float(scored["dense_scores"][raw_idx]),
                    "lexical_score": float(scored["lexical_scores"][raw_idx]),
                    "bm25_score": float(scored["bm25_scores"][raw_idx]),
                    "hybrid_score": float(scored["hybrid_scores"][raw_idx]),
                    "hybrid_rank": rank,
                    "_chunk_vector": np.asarray(scored["chunk_emb"][raw_idx], dtype=np.float32).tolist(),
                }
            )
            candidates.append(candidate)
            if len(candidates) >= int(params.claim_pool_top_n):
                break
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "gold_label": sample.label,
        "candidates": candidates,
    }


def build_selector_mechanism_trace_row(
    *,
    claim_pool_row: dict[str, Any] | None,
    union_row: dict[str, Any] | None,
    selector_name: str,
    params: SelectorMechanismParams,
    chunk_mmr_fingerprint: str,
) -> dict[str, Any]:
    if selector_name not in SELECTOR_MECHANISM_NAMES:
        raise ValueError(f"unknown selector mechanism: {selector_name!r}")

    if selector_name in {
        SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
        SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED,
    }:
        if union_row is None:
            raise ValueError(f"{selector_name} requires union_row")
        source_row = union_row
        candidate_pool = _normalize_candidate_pool(union_row.get("candidates") or [])
        if selector_name == SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED:
            selected_candidates = rank_atom_union_source_score_candidates(
                candidate_pool,
                params=AtomUnionSelectionParams(selector_top_k=len(candidate_pool)),
            )
        else:
            selected_candidates = select_atom_union_rules(
                {**union_row, "candidates": candidate_pool},
                params=AtomUnionSelectionParams(selector_top_k=int(params.top_k)),
            )["atom_union_source_score_top5"]
        selected_indices = _indices_for_selected(candidate_pool, selected_candidates)
        candidate_pool = _merge_selected_fields(candidate_pool, selected_indices, selected_candidates)
    else:
        if claim_pool_row is None:
            raise ValueError(f"{selector_name} requires claim_pool_row")
        source_row = claim_pool_row
        candidate_pool = []
        if selector_name != SELECTOR_MECH_S0_NO_EVIDENCE:
            candidate_pool = _normalize_candidate_pool(claim_pool_row.get("candidates") or [])
        selected_indices = _select_claim_pool_indices(candidate_pool, selector_name=selector_name, params=params)
        selected_candidates = [dict(candidate_pool[idx]) for idx in selected_indices]

    candidate_scores = _candidate_scores(candidate_pool, selected_indices)
    selected_candidates = [
        {
            **_strip_private_candidate_fields(candidate),
            "selection_rank": rank + 1,
            "selector_selected_step": rank,
        }
        for rank, candidate in enumerate(selected_candidates)
    ]
    candidate_pool = [_strip_private_candidate_fields(candidate) for candidate in candidate_pool]

    metadata = {
        "chunk_mmr_fingerprint": str(chunk_mmr_fingerprint or ""),
        "selector_name": selector_name,
        "graph_version": SELECTOR_MECHANISM_GRAPH_VERSION,
        "adaptive_policy": _adaptive_policy_for_selector(selector_name),
    }
    return {
        "event_id": source_row.get("event_id", ""),
        "claim": source_row.get("claim", ""),
        "label": source_row.get("label", ""),
        "gold_label": source_row.get("gold_label") or source_row.get("label", ""),
        "claim_atoms": list(source_row.get("claim_atoms") or []),
        "selector_name": selector_name,
        "graph_version": SELECTOR_MECHANISM_GRAPH_VERSION,
        "adaptive_policy": _adaptive_policy_for_selector(selector_name),
        "fingerprint": str(chunk_mmr_fingerprint or ""),
        "chunk_mmr_fingerprint": str(chunk_mmr_fingerprint or ""),
        "candidate_pool_metadata": metadata,
        "candidate_pool": candidate_pool,
        "candidate_scores": candidate_scores,
        "selector_ordered_indices": [int(idx) for idx in selected_indices],
        "selected_indices": [int(idx) for idx in selected_indices],
        "selected_candidates": selected_candidates,
        "selected_keys": [_candidate_key(candidate) for candidate in selected_candidates],
        "oracle_ordered_indices": [],
        "selection_steps": [
            {
                "step": rank + 1,
                "candidate_idx": int(idx),
                "selector_candidate_idx": int(idx),
                "selector_name": selector_name,
            }
            for rank, idx in enumerate(selected_indices)
        ],
    }


def _adaptive_policy_for_selector(selector_name: str) -> str:
    if selector_name == SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED:
        return "source_score_ordered"
    return SELECTOR_MECHANISM_ADAPTIVE_POLICY


def summarize_selector_mechanism_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected_counts = [len(trace.get("selected_indices") or []) for trace in traces]
    pool_counts = [len(trace.get("candidate_pool") or []) for trace in traces]
    selector_names: dict[str, int] = {}
    for trace in traces:
        name = str(trace.get("selector_name") or "")
        selector_names[name] = selector_names.get(name, 0) + 1
    return {
        "n_traces": len(traces),
        "selector_names": selector_names,
        "candidate_pool_size": _summary(pool_counts),
        "selected_count": _summary(selected_counts),
    }


def _select_claim_pool_indices(
    candidate_pool: list[dict[str, Any]],
    *,
    selector_name: str,
    params: SelectorMechanismParams,
) -> list[int]:
    if selector_name == SELECTOR_MECH_S0_NO_EVIDENCE:
        return []
    n = len(candidate_pool)
    if n == 0:
        return []
    limit = min(int(params.top_k), n)
    if selector_name == SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5:
        order = list(range(n))
        rng = np.random.default_rng(int(params.random_seed))
        rng.shuffle(order)
        return order[:limit]
    if selector_name == SELECTOR_MECH_S2_CLAIM_POOL_HYBRID_TOP5:
        return sorted(range(n), key=lambda idx: _hybrid_score(candidate_pool[idx]), reverse=True)[:limit]
    if selector_name == SELECTOR_MECH_S3_CLAIM_POOL_HYBRID_MMR_TOP5:
        valid_positions = [
            pos
            for pos, candidate in enumerate(candidate_pool)
            if _int_or_none(candidate.get("original_candidate_idx")) is not None
        ]
        if not valid_positions:
            return sorted(range(n), key=lambda idx: _hybrid_score(candidate_pool[idx]), reverse=True)[:limit]
        query_scores = np.asarray([_hybrid_score(candidate_pool[pos]) for pos in valid_positions], dtype=np.float32)
        vectors = np.asarray([candidate_pool[pos].get("_chunk_vector") for pos in valid_positions], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(valid_positions):
            return sorted(range(n), key=lambda idx: _hybrid_score(candidate_pool[idx]), reverse=True)[:limit]
        local = maximal_marginal_relevance(
            query_scores=query_scores,
            sentence_vectors=vectors,
            top_k=limit,
            lambda_weight=float(params.merge_mmr_lambda),
        )
        return [int(valid_positions[int(pos)]) for pos in local]
    raise ValueError(f"unknown claim-pool selector: {selector_name!r}")


def _normalize_candidate_pool(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(candidates):
        candidate = dict(raw)
        candidate["candidate_idx"] = idx
        candidate["selector_candidate_idx"] = idx
        candidate.setdefault("canonical_text", _candidate_key(candidate))
        out.append(candidate)
    return out


def _candidate_scores(candidate_pool: list[dict[str, Any]], selected_indices: list[int]) -> list[dict[str, Any]]:
    selected_step = {int(idx): rank for rank, idx in enumerate(selected_indices)}
    scores: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidate_pool):
        score = {
            "candidate_idx": idx,
            "selector_candidate_idx": idx,
            "candidate_uid": str(candidate.get("candidate_uid") or candidate.get("canonical_text") or ""),
            "hybrid_rank": int(candidate.get("hybrid_rank") or idx + 1),
            "dense_score": _float_or_default(candidate.get("dense_score"), 0.0),
            "lexical_score": _float_or_default(candidate.get("lexical_score"), 0.0),
            "bm25_score": _float_or_default(candidate.get("bm25_score"), 0.0),
            "hybrid_score": _float_or_default(candidate.get("hybrid_score"), 0.0),
            "selector_score": _selector_score(candidate),
        }
        if idx in selected_step:
            score["selector_selected_step"] = selected_step[idx]
        scores.append(score)
    return scores


def _strip_private_candidate_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    copied = dict(candidate)
    copied.pop("_chunk_vector", None)
    return copied


def _indices_for_selected(
    candidate_pool: list[dict[str, Any]],
    selected_candidates: Sequence[dict[str, Any]],
) -> list[int]:
    unused: dict[str, list[int]] = {}
    for idx, candidate in enumerate(candidate_pool):
        unused.setdefault(_candidate_key(candidate), []).append(idx)
    selected: list[int] = []
    for candidate in selected_candidates:
        key = _candidate_key(candidate)
        matches = unused.get(key) or []
        if not matches:
            continue
        selected.append(matches.pop(0))
    return selected


def _merge_selected_fields(
    candidate_pool: list[dict[str, Any]],
    selected_indices: list[int],
    selected_candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(candidate) for candidate in candidate_pool]
    for idx, selected in zip(selected_indices, selected_candidates):
        item = dict(merged[idx])
        item.update(dict(selected))
        item["candidate_idx"] = idx
        item["selector_candidate_idx"] = idx
        merged[idx] = item
    return merged


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("canonical_text") or canonicalize_sentence(str(candidate.get("text") or "")))


def _hybrid_score(candidate: dict[str, Any]) -> float:
    return _float_or_default(candidate.get("hybrid_score"), 0.0)


def _selector_score(candidate: dict[str, Any]) -> float:
    for key in ("atom_union_source_score", "selector_score", "hybrid_score"):
        if key in candidate:
            return _float_or_default(candidate.get(key), 0.0)
    return 0.0


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
    }
