from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from fact_checking.selectors.evidence_quality import retrieval_score


MAP_SELECTOR_S0_RETRIEVAL_TOP5 = "map_selector_s0_retrieval_top5"
MAP_SELECTOR_S1_MMR_POOL_TOP5 = "map_selector_s1_mmr_pool_top5"
MAP_SELECTOR_S2_MAP_QUALITY_TOP5 = "map_selector_s2_map_quality_top5"
MAP_SELECTOR_S3_WEIGHTED_SET_COVER_TOP5 = "map_selector_s3_weighted_set_cover_top5"
MAP_SELECTOR_S4_MINIMAL_EVIDENCE_GROUP_TOP5 = "map_selector_s4_minimal_evidence_group_top5"
MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5 = "map_selector_s5_fixed_budget_marginal_greedy_top5"
MAP_SELECTOR_NAMES = (
    MAP_SELECTOR_S0_RETRIEVAL_TOP5,
    MAP_SELECTOR_S1_MMR_POOL_TOP5,
    MAP_SELECTOR_S2_MAP_QUALITY_TOP5,
    MAP_SELECTOR_S3_WEIGHTED_SET_COVER_TOP5,
    MAP_SELECTOR_S4_MINIMAL_EVIDENCE_GROUP_TOP5,
    MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5,
)
MAP_SELECTOR_GRAPH_VERSION = "map_selector_ablation_v0"
MAP_SELECTOR_ADAPTIVE_POLICY = "fixed_top5"
MAP_SELECTOR_S5_ADAPTIVE_POLICY = "fixed_budget_marginal_greedy"
DEFAULT_MAP_SELECTOR_FINGERPRINT = "d4cbf7c18126"


@dataclass(frozen=True)
class MapSelectorAblationParams:
    selector_name: str
    top_k: int = 5
    candidate_top_n: int = 20
    chunk_mmr_fingerprint: str = DEFAULT_MAP_SELECTOR_FINGERPRINT


def build_map_selector_ablation_trace(
    row: Mapping[str, Any],
    *,
    params: MapSelectorAblationParams,
) -> dict[str, Any]:
    selector_name = str(params.selector_name)
    if selector_name not in MAP_SELECTOR_NAMES:
        raise ValueError(f"Unsupported map selector ablation: {selector_name}")
    top_k = max(1, int(params.top_k))
    if selector_name == MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5:
        return _build_fixed_budget_marginal_trace(row, params=params, top_k=top_k)
    candidate_pool = _candidate_pool(row, candidate_top_n=int(params.candidate_top_n))
    claim_atoms = _claim_atoms(row)
    selection = _select_candidate_indices(
        candidate_pool,
        selector_name=selector_name,
        top_k=top_k,
        claim_atoms=claim_atoms,
    )
    selected_indices = list(selection["selected_indices"])
    selected_candidates = [candidate_pool[idx] for idx in selected_indices]
    fingerprint = _row_fingerprint(row) or str(params.chunk_mmr_fingerprint or "")
    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "selector_name": selector_name,
        "graph_version": MAP_SELECTOR_GRAPH_VERSION,
        "adaptive_policy": MAP_SELECTOR_ADAPTIVE_POLICY,
        "fingerprint": fingerprint,
        "candidate_pool_metadata": {
            "chunk_mmr_fingerprint": fingerprint,
            "graph_version": MAP_SELECTOR_GRAPH_VERSION,
            "selector_name": selector_name,
            "adaptive_policy": MAP_SELECTOR_ADAPTIVE_POLICY,
            "top_k": top_k,
            "candidate_top_n": int(params.candidate_top_n),
        },
        "candidate_pool": candidate_pool,
        "candidate_scores": _candidate_scores(
            candidate_pool,
            selected_indices=selected_indices,
            selector_name=selector_name,
            top_k=top_k,
        ),
        "selector_ordered_indices": list(selection.get("selector_ordered_indices") or selected_indices),
        "selected_indices": selected_indices,
        "oracle_ordered_indices": _oracle_ordered_indices(list(row.get("oracle_ordered_keys") or []), candidate_pool),
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "selected_evidence_ids": [str(candidate.get("evidence_id") or "") for candidate in selected_candidates],
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected_candidates],
        "selected_candidates": [_candidate_trace_output(candidate) for candidate in selected_candidates],
        "claim_atoms": claim_atoms,
        "params": {
            "top_k": top_k,
            "candidate_top_n": int(params.candidate_top_n),
            "chunk_mmr_fingerprint": fingerprint,
        },
        "chain_summary": {
            "selector_name": selector_name,
            "evidence_ids": [str(candidate.get("evidence_id") or "") for candidate in selected_candidates],
            "candidate_keys": [str(candidate.get("candidate_key") or "") for candidate in selected_candidates],
        },
    }
    trace.update(dict(selection.get("trace_extra") or {}))
    trace.update(_selection_diagnostics(selected_candidates, top_k=top_k))
    return trace


def summarize_map_selector_ablation_traces(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_counts = [len(row.get("selected_indices") or []) for row in rows]
    pool_counts = [len(row.get("candidate_pool") or []) for row in rows]
    return {
        "n_rows": len(rows),
        "graph_version": MAP_SELECTOR_GRAPH_VERSION,
        "selector_names": dict(Counter(str(row.get("selector_name") or "") for row in rows if row.get("selector_name"))),
        "selected_count": _numeric_summary(selected_counts),
        "candidate_pool_count": _numeric_summary(pool_counts),
    }


def _build_fixed_budget_marginal_trace(
    row: Mapping[str, Any],
    *,
    params: MapSelectorAblationParams,
    top_k: int,
) -> dict[str, Any]:
    from fact_checking.selectors.evidence_chain_graph import (
        BudgetedMarginalChainParams,
        build_budgeted_marginal_chain_graph_row,
    )

    fingerprint = _row_fingerprint(row) or str(params.chunk_mmr_fingerprint or "")
    graph_row = build_budgeted_marginal_chain_graph_row(
        dict(row),
        params=BudgetedMarginalChainParams(
            candidate_top_n=int(params.candidate_top_n),
            min_top_k=int(top_k),
            max_top_k=int(top_k),
            chunk_mmr_fingerprint=fingerprint,
        ),
    )
    trace = dict(graph_row.get("selection_trace") or {})
    source_graph_version = str(trace.get("graph_version") or graph_row.get("graph_version") or "")
    source_selector_name = str(trace.get("selector_name") or graph_row.get("selector_name") or "")
    selector_name = MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5

    candidate_pool_metadata = dict(trace.get("candidate_pool_metadata") or {})
    candidate_pool_metadata.update(
        {
            "chunk_mmr_fingerprint": fingerprint,
            "graph_version": MAP_SELECTOR_GRAPH_VERSION,
            "selector_name": selector_name,
            "adaptive_policy": MAP_SELECTOR_S5_ADAPTIVE_POLICY,
            "top_k": int(top_k),
            "candidate_top_n": int(params.candidate_top_n),
            "source_graph_version": source_graph_version,
            "source_selector_name": source_selector_name,
        }
    )
    candidate_scores = [dict(score) for score in trace.get("candidate_scores") or []]
    for score in candidate_scores:
        score["selector_name"] = selector_name
    chain_summary = dict(trace.get("chain_summary") or {})
    chain_summary.update(
        {
            "selector_name": selector_name,
            "source_selector_name": source_selector_name,
            "source_graph_version": source_graph_version,
        }
    )

    trace.update(
        {
            "selector_name": selector_name,
            "graph_version": MAP_SELECTOR_GRAPH_VERSION,
            "adaptive_policy": MAP_SELECTOR_S5_ADAPTIVE_POLICY,
            "fingerprint": fingerprint,
            "candidate_pool_metadata": candidate_pool_metadata,
            "candidate_scores": candidate_scores,
            "chain_summary": chain_summary,
            "min_top_k": int(top_k),
            "max_top_k": int(top_k),
            "top_k": int(top_k),
            "adaptive_evidence_count": len(trace.get("selected_indices") or []),
            "params": {
                "top_k": int(top_k),
                "candidate_top_n": int(params.candidate_top_n),
                "chunk_mmr_fingerprint": fingerprint,
                "source_graph_version": source_graph_version,
                "source_selector_name": source_selector_name,
                "budgeted_marginal_params": dict(graph_row.get("params") or {}),
            },
        }
    )
    trace.update(_selection_diagnostics(list(trace.get("selected_candidates") or []), top_k=top_k))
    return trace


def _candidate_pool(row: Mapping[str, Any], *, candidate_top_n: int) -> list[dict[str, Any]]:
    raw_candidates = [dict(candidate) for candidate in row.get("candidates") or [] if isinstance(candidate, Mapping)]
    if candidate_top_n > 0:
        raw_candidates = raw_candidates[:candidate_top_n]
    event_id = str(row.get("event_id") or "")
    pool: list[dict[str, Any]] = []
    for pool_idx, candidate in enumerate(raw_candidates):
        item = dict(candidate)
        original_idx = _first_int(item.get("original_candidate_idx"), item.get("candidate_idx"), item.get("source_index"))
        if original_idx is None:
            original_idx = pool_idx
        candidate_key = str(item.get("candidate_key") or item.get("canonical_text") or item.get("text") or f"candidate:{pool_idx}")
        evidence_id = str(item.get("evidence_id") or item.get("candidate_uid") or f"E{pool_idx + 1:02d}")
        item["candidate_idx"] = int(pool_idx)
        item["original_candidate_idx"] = int(original_idx)
        item["selector_pool_rank"] = int(pool_idx)
        item["candidate_key"] = candidate_key
        item["candidate_uid"] = str(item.get("candidate_uid") or f"{event_id}:{pool_idx}")
        item["evidence_id"] = evidence_id
        item["text"] = str(item.get("text") or item.get("canonical_text") or "")
        item["retrieval_score"] = float(retrieval_score(item))
        pool.append(item)
    return pool


def _rank_candidate_indices(candidate_pool: Sequence[Mapping[str, Any]], *, selector_name: str) -> list[int]:
    indices = list(range(len(candidate_pool)))
    if selector_name == MAP_SELECTOR_S0_RETRIEVAL_TOP5:
        return sorted(indices, key=lambda idx: _s0_sort_key(candidate_pool[idx]))
    if selector_name == MAP_SELECTOR_S1_MMR_POOL_TOP5:
        return sorted(indices, key=lambda idx: _s1_sort_key(candidate_pool[idx], idx=idx))
    if selector_name == MAP_SELECTOR_S2_MAP_QUALITY_TOP5:
        return sorted(indices, key=lambda idx: _s2_sort_key(candidate_pool[idx]))
    raise ValueError(f"Unsupported map selector ablation: {selector_name}")


def _select_candidate_indices(
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    selector_name: str,
    top_k: int,
    claim_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if selector_name in {
        MAP_SELECTOR_S0_RETRIEVAL_TOP5,
        MAP_SELECTOR_S1_MMR_POOL_TOP5,
        MAP_SELECTOR_S2_MAP_QUALITY_TOP5,
    }:
        ordered_indices = _rank_candidate_indices(candidate_pool, selector_name=selector_name)
        selected_indices = ordered_indices[: min(int(top_k), len(ordered_indices))]
        return {"selected_indices": selected_indices, "selector_ordered_indices": selected_indices}
    if selector_name == MAP_SELECTOR_S3_WEIGHTED_SET_COVER_TOP5:
        return _select_weighted_set_cover_indices(
            candidate_pool,
            top_k=top_k,
            claim_atoms=claim_atoms,
            record_minimal_group=False,
        )
    if selector_name == MAP_SELECTOR_S4_MINIMAL_EVIDENCE_GROUP_TOP5:
        return _select_weighted_set_cover_indices(
            candidate_pool,
            top_k=top_k,
            claim_atoms=claim_atoms,
            record_minimal_group=True,
        )
    raise ValueError(f"Unsupported map selector ablation: {selector_name}")


def _select_weighted_set_cover_indices(
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    claim_atoms: Sequence[Mapping[str, Any]],
    record_minimal_group: bool,
) -> dict[str, Any]:
    atom_weights = _claim_atom_weights(claim_atoms)
    atom_order = {atom_id: idx for idx, atom_id in enumerate(atom_weights)}
    candidate_atom_ids = [
        _candidate_claim_atom_ids(candidate, atom_weights=atom_weights, atom_order=atom_order)
        for candidate in candidate_pool
    ]
    coverable_atom_ids = _sort_atom_ids(
        {atom_id for atom_ids in candidate_atom_ids for atom_id in atom_ids},
        atom_order=atom_order,
    )
    coverable_atom_set = set(coverable_atom_ids)
    selected: list[int] = []
    covered: set[str] = set()
    steps: list[dict[str, Any]] = []

    while len(selected) < int(top_k) and (coverable_atom_set - covered):
        ranked: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidate_pool):
            if idx in selected:
                continue
            new_atom_ids = [atom_id for atom_id in candidate_atom_ids[idx] if atom_id not in covered]
            gain = sum(float(atom_weights.get(atom_id, 1.0)) for atom_id in new_atom_ids)
            if gain <= 0.0:
                continue
            ranked.append(
                {
                    "idx": int(idx),
                    "candidate": candidate,
                    "weighted_new_atom_gain": float(gain),
                    "covered_new_atom_ids": new_atom_ids,
                }
            )
        if not ranked:
            break
        ranked.sort(key=_weighted_set_cover_candidate_sort_key)
        pick = ranked[0]
        idx = int(pick["idx"])
        new_atom_ids = list(pick["covered_new_atom_ids"])
        selected.append(idx)
        covered.update(new_atom_ids)
        steps.append(
            {
                "step": len(selected),
                "candidate_idx": idx,
                "candidate_key": str(candidate_pool[idx].get("candidate_key") or ""),
                "evidence_id": str(candidate_pool[idx].get("evidence_id") or ""),
                "rule": "weighted_set_cover",
                "weighted_new_atom_gain": float(pick["weighted_new_atom_gain"]),
                "covered_new_atom_ids": new_atom_ids,
                "covered_atom_ids_after_step": _sort_atom_ids(covered, atom_order=atom_order),
            }
        )

    minimal_group_indices = list(selected)
    fill_indices = [
        idx
        for idx in _rank_candidate_indices(candidate_pool, selector_name=MAP_SELECTOR_S2_MAP_QUALITY_TOP5)
        if idx not in set(selected)
    ][: max(int(top_k) - len(selected), 0)]
    selected_indices = selected + fill_indices
    trace_extra: dict[str, Any] = {
        "selection_steps": steps,
        "coverable_atom_ids": coverable_atom_ids,
        "covered_atom_ids": _sort_atom_ids(covered, atom_order=atom_order),
        "fixed_budget_fill_indices": fill_indices,
        "weighted_atom_coverage": _weighted_atom_coverage(
            covered,
            coverable_atom_ids=coverable_atom_ids,
            atom_weights=atom_weights,
        ),
    }
    if record_minimal_group:
        trace_extra.update(
            {
                "minimal_group_indices": minimal_group_indices,
                "minimal_group_size": len(minimal_group_indices),
            }
        )
    return {
        "selected_indices": selected_indices,
        "selector_ordered_indices": selected_indices,
        "trace_extra": trace_extra,
    }


def _weighted_set_cover_candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, int, str]:
    candidate = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
    return (
        -_safe_float(row.get("weighted_new_atom_gain"), 0.0),
        -_safe_float(candidate.get("evidence_map_quality_score"), 0.0),
        -_safe_float(candidate.get("retrieval_score"), 0.0),
        -_safe_float(candidate.get("evidence_map_base_score"), 0.0),
        _rank_value(candidate.get("union_pool_rank")),
        str(candidate.get("candidate_key") or ""),
    )


def _claim_atom_weights(claim_atoms: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for idx, atom in enumerate(claim_atoms, start=1):
        atom_id = str(atom.get("atom_id") or atom.get("node_id") or f"A{idx}")
        if not atom_id:
            continue
        weights.setdefault(atom_id, _importance_weight(atom.get("importance", 1.0)))
    return weights


def _candidate_claim_atom_ids(
    candidate: Mapping[str, Any],
    *,
    atom_weights: Mapping[str, float],
    atom_order: Mapping[str, int],
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in candidate.get("covered_atom_ids") or []:
        atom_id = str(value)
        if atom_id not in atom_weights or atom_id in seen:
            continue
        seen.add(atom_id)
        out.append(atom_id)
    return _sort_atom_ids(out, atom_order=atom_order)


def _weighted_atom_coverage(
    covered_atom_ids: Iterable[str],
    *,
    coverable_atom_ids: Sequence[str],
    atom_weights: Mapping[str, float],
) -> dict[str, Any]:
    covered = set(str(atom_id) for atom_id in covered_atom_ids)
    coverable = [str(atom_id) for atom_id in coverable_atom_ids]
    coverable_weight = sum(float(atom_weights.get(atom_id, 1.0)) for atom_id in coverable)
    covered_weight = sum(float(atom_weights.get(atom_id, 1.0)) for atom_id in coverable if atom_id in covered)
    return {
        "coverable_atom_count": len(coverable),
        "covered_atom_count": sum(1 for atom_id in coverable if atom_id in covered),
        "coverable_atom_weight": float(coverable_weight),
        "covered_atom_weight": float(covered_weight),
        "weighted_coverage_rate": float(covered_weight / coverable_weight) if coverable_weight > 0.0 else 0.0,
    }


def _sort_atom_ids(atom_ids: Iterable[Any], *, atom_order: Mapping[str, int]) -> list[str]:
    return sorted({str(atom_id) for atom_id in atom_ids}, key=lambda atom_id: (atom_order.get(atom_id, 10**9), atom_id))


def _importance_weight(value: Any) -> float:
    parsed = _safe_float(value, math.nan)
    if math.isfinite(parsed):
        return float(max(parsed, 0.0))
    label = str(value or "").strip().lower()
    if label in {"critical", "high", "important"}:
        return 1.0
    if label in {"medium", "med"}:
        return 0.6
    if label in {"low", "minor"}:
        return 0.3
    return 1.0


def _s0_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, int, str]:
    return (
        -_safe_float(candidate.get("retrieval_score"), 0.0),
        -_safe_float(candidate.get("evidence_map_base_score"), 0.0),
        _rank_value(candidate.get("union_pool_rank")),
        str(candidate.get("candidate_key") or ""),
    )


def _s1_sort_key(candidate: Mapping[str, Any], *, idx: int) -> tuple[int, int, float, str]:
    mmr_rank = _first_int(candidate.get("mmr_rank"), candidate.get("union_pool_rank"))
    union_rank = _first_int(candidate.get("union_pool_rank"), idx + 1)
    return (
        int(mmr_rank if mmr_rank is not None else idx + 1),
        int(union_rank if union_rank is not None else idx + 1),
        -_safe_float(candidate.get("evidence_map_base_score"), 0.0),
        str(candidate.get("candidate_key") or ""),
    )


def _s2_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        -_safe_float(candidate.get("evidence_map_quality_score"), 0.0),
        -_safe_float(candidate.get("retrieval_score"), 0.0),
        -_safe_float(candidate.get("evidence_map_base_score"), 0.0),
        _rank_value(candidate.get("union_pool_rank")),
        str(candidate.get("candidate_key") or ""),
    )


def _candidate_scores(
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    selected_indices: Sequence[int],
    selector_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    selected_rank_by_idx = {int(idx): rank for rank, idx in enumerate(selected_indices)}
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidate_pool):
        selected_rank = selected_rank_by_idx.get(idx)
        selector_score = float(max(int(top_k) - int(selected_rank), 0)) if selected_rank is not None else 0.0
        original_idx = _first_int(candidate.get("original_candidate_idx"))
        if original_idx is None:
            original_idx = idx
        rows.append(
            {
                "candidate_idx": int(idx),
                "original_candidate_idx": int(original_idx),
                "candidate_uid": str(candidate.get("candidate_uid") or ""),
                "candidate_key": str(candidate.get("candidate_key") or ""),
                "evidence_id": str(candidate.get("evidence_id") or ""),
                "selector_name": selector_name,
                "selector_score": selector_score,
                "selector_selected_step": int(selected_rank) if selected_rank is not None else -1,
                "retrieval_score": _safe_float(candidate.get("retrieval_score"), 0.0),
                "hybrid_score": _safe_float(candidate.get("hybrid_score"), _safe_float(candidate.get("baseline_hybrid_score"), 0.0)),
                "baseline_hybrid_score": _safe_float(candidate.get("baseline_hybrid_score"), 0.0),
                "evidence_map_base_score": _safe_float(candidate.get("evidence_map_base_score"), 0.0),
                "evidence_map_quality_score": _safe_float(candidate.get("evidence_map_quality_score"), 0.0),
                "atom_coverage_score": _safe_float(candidate.get("atom_coverage_score"), 0.0),
                "map_confidence": _safe_float(candidate.get("map_confidence"), 0.0),
                "union_pool_rank": _rank_value(candidate.get("union_pool_rank")),
                "mmr_rank": _rank_value(candidate.get("mmr_rank")),
                "map_relation": str(candidate.get("map_relation") or ""),
                "map_directness": str(candidate.get("map_directness") or ""),
            }
        )
    return rows


def _oracle_ordered_indices(oracle_ordered_keys: Sequence[Any], candidate_pool: Sequence[Mapping[str, Any]]) -> list[int]:
    exact: dict[str, int] = {}
    normalized: dict[str, int] = {}
    for idx, candidate in enumerate(candidate_pool):
        for key in (candidate.get("candidate_key"), candidate.get("text")):
            raw = str(key or "").strip()
            if not raw:
                continue
            exact.setdefault(raw, int(idx))
            normalized.setdefault(_norm_text(raw), int(idx))
    out: list[int] = []
    seen: set[int] = set()
    for raw_key in oracle_ordered_keys:
        key = str(raw_key or "").strip()
        idx = exact.get(key)
        if idx is None:
            idx = normalized.get(_norm_text(key))
        if idx is not None and idx not in seen:
            out.append(int(idx))
            seen.add(int(idx))
    return out


def _candidate_trace_output(candidate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_uid",
        "candidate_key",
        "evidence_id",
        "original_candidate_idx",
        "union_pool_rank",
        "mmr_rank",
        "source_group",
        "source_domain",
        "report_id",
        "sent_idx",
        "chunk_sent_indices",
        "evidence_map_base_score",
        "evidence_map_quality_score",
        "map_confidence",
        "covered_atom_ids",
        "map_relation",
        "map_directness",
        "map_evidence_role",
        "key_spans",
        "duplicate_group",
    )
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["text"] = str(candidate.get("text") or "")
    return out


def _claim_atoms(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = row.get("claim_atoms")
    if values is None:
        evidence_map = row.get("evidence_map")
        if isinstance(evidence_map, Mapping):
            values = evidence_map.get("claim_atoms")
    if values is None:
        values = row.get("atom_nodes")
    return [dict(item) for item in values or [] if isinstance(item, Mapping)]


def _row_fingerprint(row: Mapping[str, Any]) -> str:
    metadata = row.get("candidate_pool_metadata") if isinstance(row.get("candidate_pool_metadata"), Mapping) else {}
    values = [
        row.get("chunk_mmr_fingerprint"),
        row.get("fingerprint"),
        metadata.get("chunk_mmr_fingerprint") if isinstance(metadata, Mapping) else None,
        metadata.get("fingerprint") if isinstance(metadata, Mapping) else None,
    ]
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _selection_diagnostics(selected: Sequence[Mapping[str, Any]], *, top_k: int) -> dict[str, Any]:
    return {
        "selected_count": len(selected),
        "top_k": int(top_k),
        "selected_map_quality_mean": _mean(
            _safe_float(candidate.get("evidence_map_quality_score"), 0.0) for candidate in selected
        ),
        "selected_retrieval_score_mean": _mean(
            _safe_float(candidate.get("retrieval_score"), 0.0) for candidate in selected
        ),
    }


def _numeric_summary(values: Iterable[int | float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0.0}
    return {
        "count": float(len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _rank_value(value: Any) -> int:
    parsed = _first_int(value)
    if parsed is None:
        return 1_000_000
    return int(parsed)


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
