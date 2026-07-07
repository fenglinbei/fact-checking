from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from fact_checking.build.candidates import canonicalize_sentence
from fact_checking.selectors.question_decomp_retrieval import _compute_prediction_metrics


@dataclass(frozen=True)
class AtomUnionSelectionParams:
    selector_top_k: int = 5
    baseline_bonus: float = 0.04
    baseline_rank_weight: float = 0.01
    atom_rrf_weight: float = 1.0
    atom_route_hit_weight: float = 0.004
    atom_max_hybrid_weight: float = 0.01


def build_atom_union_pool_row(
    *,
    baseline_row: dict[str, Any],
    atom_pool_row: dict[str, Any],
) -> dict[str, Any]:
    event_id = str(atom_pool_row.get("event_id") or baseline_row.get("event_id") or "")
    claim = str(atom_pool_row.get("claim") or baseline_row.get("claim") or "")
    label = str(atom_pool_row.get("label") or baseline_row.get("label") or "")
    gold_label = str(atom_pool_row.get("gold_label") or baseline_row.get("gold_label") or label)
    merged: dict[str, dict[str, Any]] = {}

    for rank, candidate in enumerate(baseline_row.get("candidates") or [], start=1):
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

    for rank, candidate in enumerate(atom_pool_row.get("candidates") or atom_pool_row.get("merged_candidate_pool") or [], start=1):
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
    candidates.sort(key=_union_pool_sort_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["union_pool_rank"] = rank
        candidate["union_source"] = _union_source(candidate)
    return {
        "event_id": event_id,
        "claim": claim,
        "label": label,
        "gold_label": gold_label,
        "claim_atoms": list(atom_pool_row.get("claim_atoms") or []),
        "candidates": candidates,
    }


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
    "select_atom_union_rules",
]
