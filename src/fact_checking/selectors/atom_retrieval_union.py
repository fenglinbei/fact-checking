from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fact_checking.build.candidates import canonicalize_sentence, minmax_scale
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.selectors.question_decomp_retrieval import _compute_prediction_metrics


ATOM_UNION_POOL_MODES = ("baseline_only", "atom_only", "union_no_mmr", "union_full")


@dataclass(frozen=True)
class AtomUnionSelectionParams:
    selector_top_k: int = 5
    baseline_bonus: float = 0.04
    baseline_rank_weight: float = 0.01
    atom_rrf_weight: float = 1.0
    atom_route_hit_weight: float = 0.004
    atom_max_hybrid_weight: float = 0.01
    union_mmr_lambda: float = 0.70


def build_atom_union_pool_row(
    *,
    baseline_row: dict[str, Any],
    atom_pool_row: dict[str, Any],
    candidate_vectors: Mapping[str, np.ndarray] | None = None,
    final_pool_size: int | None = None,
    pool_mode: str = "union_full",
    params: AtomUnionSelectionParams | None = None,
) -> dict[str, Any]:
    pool_mode = normalize_atom_union_pool_mode(pool_mode)
    selection_params = params or AtomUnionSelectionParams()
    event_id = str(atom_pool_row.get("event_id") or baseline_row.get("event_id") or "")
    claim = str(atom_pool_row.get("claim") or baseline_row.get("claim") or "")
    label = str(atom_pool_row.get("label") or baseline_row.get("label") or "")
    gold_label = str(atom_pool_row.get("gold_label") or baseline_row.get("gold_label") or label)
    merged: dict[str, dict[str, Any]] = {}

    baseline_candidates = baseline_row.get("candidates") or []
    if pool_mode == "atom_only":
        baseline_candidates = []
    for rank, candidate in enumerate(baseline_candidates, start=1):
        key = _candidate_key(candidate)
        if not key:
            continue
        item = dict(candidate)
        item["canonical_text"] = key
        item["from_baseline"] = True
        item["baseline_rank"] = int(candidate.get("selection_rank") or rank)
        item["baseline_hybrid_score"] = candidate.get("hybrid_score")
        item.setdefault("from_atom_route", False)
        item.setdefault("atom_pool_rank", None)
        item.setdefault("atom_rrf_score", 0.0)
        item.setdefault("atom_route_hit_count", 0)
        item.setdefault("atom_max_route_hybrid", 0.0)
        item.setdefault("atom_routes", [])
        item.setdefault("matched_atom_ids", [])
        merged[key] = item

    atom_candidates = atom_pool_row.get("candidates") or atom_pool_row.get("merged_candidate_pool") or []
    if pool_mode == "baseline_only":
        atom_candidates = []
    for rank, candidate in enumerate(atom_candidates, start=1):
        key = _candidate_key(candidate)
        if not key:
            continue
        item = merged.get(key)
        if item is None:
            item = dict(candidate)
            item["canonical_text"] = key
            item["from_baseline"] = False
            item["baseline_rank"] = None
            item["baseline_hybrid_score"] = None
            merged[key] = item
        item["from_atom_route"] = True
        item["atom_pool_rank"] = int(candidate.get("atom_pool_rank") or candidate.get("merge_rank") or rank)
        item["atom_rrf_score"] = float(candidate.get("atom_rrf_score") or 0.0)
        item["atom_route_hit_count"] = int(candidate.get("atom_route_hit_count") or 0)
        item["atom_max_route_hybrid"] = float(candidate.get("atom_max_route_hybrid") or 0.0)
        item["atom_routes"] = list(candidate.get("atom_routes") or [])
        item["matched_atom_ids"] = list(candidate.get("matched_atom_ids") or [])
        item.setdefault("text", candidate.get("text", ""))
        item.setdefault("source_report", candidate.get("source_report"))
        item.setdefault("report_id", candidate.get("report_id"))
        item.setdefault("sent_idx", candidate.get("sent_idx"))
        item.setdefault("chunk_sent_indices", candidate.get("chunk_sent_indices"))

    candidates = list(merged.values())
    if final_pool_size is None:
        candidates.sort(key=_union_pool_sort_key)
    elif pool_mode == "baseline_only":
        candidates.sort(key=lambda candidate: int(candidate.get("baseline_rank") or 10**9))
    elif pool_mode == "atom_only":
        candidates.sort(key=lambda candidate: int(candidate.get("atom_pool_rank") or 10**9))
    else:
        candidates.sort(
            key=lambda candidate: (
                -_atom_union_mmr_relevance(candidate, params=selection_params),
                *_union_pool_sort_key(candidate),
            )
        )
    pool_size_before_mmr = len(candidates)
    mmr_applied = final_pool_size is not None and pool_mode == "union_full"
    if mmr_applied:
        if candidate_vectors is None:
            raise ValueError("candidate_vectors are required when final_pool_size is set.")
        candidates = select_atom_union_mmr(
            candidates,
            candidate_vectors=candidate_vectors,
            top_k=int(final_pool_size),
            params=selection_params,
        )
    elif final_pool_size is not None:
        candidates = candidates[: int(final_pool_size)]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["union_pool_rank"] = rank
        candidate["union_source"] = _union_source(candidate)
    return {
        "event_id": event_id,
        "claim": claim,
        "label": label,
        "gold_label": gold_label,
        "claim_atoms": list(atom_pool_row.get("claim_atoms") or []),
        "pool_mode": pool_mode,
        "union_pool_size_before_mmr": pool_size_before_mmr,
        "union_mmr_applied": mmr_applied,
        "candidates": candidates,
    }


def normalize_atom_union_pool_mode(pool_mode: str) -> str:
    normalized = str(pool_mode or "union_full").strip().lower()
    if normalized == "atom_route_only":
        normalized = "atom_only"
    if normalized not in ATOM_UNION_POOL_MODES:
        raise ValueError(
            f"Unsupported Atom-Union pool_mode={pool_mode!r}; expected one of {ATOM_UNION_POOL_MODES}."
        )
    return normalized


def select_atom_union_mmr(
    candidates: Sequence[dict[str, Any]],
    *,
    candidate_vectors: Mapping[str, np.ndarray],
    top_k: int,
    params: AtomUnionSelectionParams,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    if not rows or int(top_k) <= 0:
        return []

    vectors: list[np.ndarray] = []
    raw_scores: list[float] = []
    for candidate in rows:
        key = _candidate_key(candidate)
        vector = candidate_vectors.get(key)
        if vector is None:
            raise ValueError(f"Missing chunk embedding for Atom-Union candidate: {key[:120]!r}")
        vectors.append(np.asarray(vector, dtype=np.float32))
        raw_scores.append(_atom_union_mmr_relevance(candidate, params=params))

    query_scores = minmax_scale(np.asarray(raw_scores, dtype=np.float32))
    selected_indices = maximal_marginal_relevance(
        query_scores=query_scores,
        sentence_vectors=np.asarray(vectors, dtype=np.float32),
        top_k=min(int(top_k), len(rows)),
        lambda_weight=float(params.union_mmr_lambda),
    )
    selected: list[dict[str, Any]] = []
    for rank, candidate_idx in enumerate(selected_indices, start=1):
        item = rows[int(candidate_idx)]
        item["atom_union_relevance_score"] = float(raw_scores[int(candidate_idx)])
        item["atom_union_relevance_score_normalized"] = float(query_scores[int(candidate_idx)])
        item["atom_union_mmr_rank"] = rank
        selected.append(item)
    return selected


def select_atom_union_rules(
    union_row: dict[str, Any],
    *,
    params: AtomUnionSelectionParams,
) -> dict[str, list[dict[str, Any]]]:
    candidates = list(union_row.get("candidates") or [])
    return {
        "atom_union_baseline_first_top5": _select_baseline_first(candidates, params.selector_top_k),
        "atom_union_interleave_top5": _select_interleave(candidates, params.selector_top_k),
        "atom_union_source_score_top5": _select_source_score(candidates, params),
    }


def rank_atom_union_source_score_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    params: AtomUnionSelectionParams,
) -> list[dict[str, Any]]:
    """Return the full atom-union pool ordered by the S4 source-score rule."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        baseline_rank = candidate.get("baseline_rank")
        baseline_component = 0.0
        if candidate.get("from_baseline"):
            baseline_component += float(params.baseline_bonus)
            if baseline_rank is not None:
                baseline_component += float(params.baseline_rank_weight) / max(float(baseline_rank), 1.0)
        atom_component = float(params.atom_rrf_weight) * float(candidate.get("atom_rrf_score") or 0.0)
        atom_component += float(params.atom_route_hit_weight) * float(candidate.get("atom_route_hit_count") or 0.0)
        atom_component += float(params.atom_max_hybrid_weight) * float(candidate.get("atom_max_route_hybrid") or 0.0)
        item["atom_union_source_score"] = float(baseline_component + atom_component)
        scored.append(item)
    scored.sort(
        key=lambda c: (
            -float(c.get("atom_union_source_score") or 0.0),
            int(c.get("baseline_rank") or 10**9),
            int(c.get("atom_pool_rank") or 10**9),
            int(c.get("union_pool_rank") or 10**9),
        )
    )
    return _ranked_selection(scored, len(scored), "source_score_rank")


def compute_atom_union_metrics(
    *,
    union_rows: Sequence[dict[str, Any]],
    rule_rows: dict[str, Sequence[dict[str, Any]]],
    oracle_texts: dict[str, set[str]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["atom_union_pool"] = _compute_prediction_metrics(
        pool_rows=[
            {
                "event_id": row.get("event_id"),
                "pool": row.get("candidates") or [],
                "selected": row.get("candidates") or [],
            }
            for row in union_rows
        ],
        oracle_texts=oracle_texts,
        include_pool=True,
    )
    for rule_name, rows in rule_rows.items():
        metrics[rule_name] = _compute_prediction_metrics(
            pool_rows=[
                {
                    "event_id": row.get("event_id"),
                    "pool": row.get("candidates") or [],
                    "selected": row.get("candidates") or [],
                }
                for row in rows
            ],
            oracle_texts=oracle_texts,
            include_pool=False,
        )
    return metrics


def _select_baseline_first(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    baseline = [candidate for candidate in candidates if candidate.get("from_baseline")]
    baseline.sort(key=lambda c: int(c.get("baseline_rank") or 10**9))
    atom_only = [
        candidate
        for candidate in candidates
        if not candidate.get("from_baseline") and candidate.get("from_atom_route")
    ]
    atom_only.sort(key=lambda c: int(c.get("atom_pool_rank") or 10**9))
    return _ranked_selection([*baseline, *atom_only], top_k, "baseline_first_rank")


def _select_interleave(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    baseline = [candidate for candidate in candidates if candidate.get("from_baseline")]
    baseline.sort(key=lambda c: int(c.get("baseline_rank") or 10**9))
    atom = [candidate for candidate in candidates if candidate.get("from_atom_route")]
    atom.sort(key=lambda c: int(c.get("atom_pool_rank") or 10**9))
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_len = max(len(baseline), len(atom))
    for idx in range(max_len):
        for source in (baseline, atom):
            if idx >= len(source):
                continue
            key = _candidate_key(source[idx])
            if key and key not in seen:
                ordered.append(source[idx])
                seen.add(key)
            if len(ordered) >= top_k:
                break
        if len(ordered) >= top_k:
            break
    return _ranked_selection(ordered, top_k, "interleave_rank")


def _select_source_score(candidates: list[dict[str, Any]], params: AtomUnionSelectionParams) -> list[dict[str, Any]]:
    return rank_atom_union_source_score_candidates(candidates, params=params)[: int(params.selector_top_k)]


def _ranked_selection(candidates: list[dict[str, Any]], top_k: int, rank_key: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _candidate_key(candidate)
        if not key or key in seen:
            continue
        item = dict(candidate)
        item["selection_rank"] = len(selected) + 1
        item[rank_key] = len(selected) + 1
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def _union_pool_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate.get("from_baseline") else 1,
        int(candidate.get("baseline_rank") or 10**9),
        int(candidate.get("atom_pool_rank") or 10**9),
        -float(candidate.get("atom_rrf_score") or 0.0),
    )


def _atom_union_mmr_relevance(
    candidate: dict[str, Any],
    *,
    params: AtomUnionSelectionParams,
) -> float:
    claim_relevance = float(candidate.get("baseline_hybrid_score") or 0.0)
    atom_relevance = float(candidate.get("atom_max_route_hybrid") or 0.0)
    relevance = max(claim_relevance, atom_relevance)
    relevance += float(params.atom_rrf_weight) * float(candidate.get("atom_rrf_score") or 0.0)
    relevance += float(params.atom_route_hit_weight) * float(candidate.get("atom_route_hit_count") or 0.0)
    return float(relevance)


def _union_source(candidate: dict[str, Any]) -> str:
    if candidate.get("from_baseline") and candidate.get("from_atom_route"):
        return "baseline+atom"
    if candidate.get("from_baseline"):
        return "baseline"
    if candidate.get("from_atom_route"):
        return "atom"
    return "unknown"


def _candidate_key(candidate: dict[str, Any]) -> str:
    key = str(candidate.get("canonical_text") or "")
    if key:
        return key
    return canonicalize_sentence(str(candidate.get("text") or ""))


__all__ = [
    "AtomUnionSelectionParams",
    "build_atom_union_pool_row",
    "compute_atom_union_metrics",
    "rank_atom_union_source_score_candidates",
    "select_atom_union_mmr",
    "select_atom_union_rules",
]
