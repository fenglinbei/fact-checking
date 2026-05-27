from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np

from fact_checking.build.candidates import canonicalize_sentence


ORIGINAL_POOL = "original_stage2_pool"
QD_UNION_POOL = "qd_union_pool"


@dataclass(frozen=True)
class RoundtableParams:
    top_k: int = 5
    similarity_threshold: float = 0.72
    min_factions: int = 2
    max_factions: int = 6


def canonical_candidate_key(candidate: dict[str, Any]) -> str:
    return canonicalize_sentence(str(candidate.get("text") or candidate.get("canonical_text") or ""))


def oracle_ordered_keys(oracle_row: dict[str, Any]) -> list[str]:
    pool = list(oracle_row.get("candidate_pool") or [])
    keys: list[str] = []
    seen: set[str] = set()
    for raw_idx in oracle_row.get("selected_indices") or []:
        idx = _safe_int(raw_idx, -1)
        if idx < 0 or idx >= len(pool):
            continue
        key = canonical_candidate_key(pool[idx])
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def normalize_original_candidates(oracle_row: dict[str, Any]) -> list[dict[str, Any]]:
    selected_indices = [_safe_int(idx, -1) for idx in oracle_row.get("selected_indices") or []]
    selected_order = {idx: order for order, idx in enumerate(selected_indices) if idx >= 0}
    score_by_idx = _score_rows_by_candidate_idx(oracle_row.get("candidate_scores") or [])
    candidates: list[dict[str, Any]] = []
    for position, raw_candidate in enumerate(oracle_row.get("candidate_pool") or []):
        candidate = dict(raw_candidate)
        candidate_idx = _safe_int(candidate.get("candidate_idx"), position)
        score_row = dict(score_by_idx.get(candidate_idx) or score_by_idx.get(position) or {})
        source_report = candidate.get("source_report") if isinstance(candidate.get("source_report"), dict) else {}
        text = str(candidate.get("text") or "")
        key = canonicalize_sentence(text)
        selected = candidate_idx in selected_order
        candidates.append(
            {
                "event_id": str(oracle_row.get("event_id") or ""),
                "claim": str(oracle_row.get("claim") or ""),
                "gold_label": str(oracle_row.get("gold_label") or ""),
                "pool_name": ORIGINAL_POOL,
                "pool_position": int(position),
                "candidate_idx": int(candidate_idx),
                "candidate_uid": str(candidate.get("candidate_uid") or score_row.get("candidate_uid") or ""),
                "candidate_key": key,
                "canonical_text": key,
                "text": text,
                "source_index": _nullable_int(candidate.get("source_index", score_row.get("source_index"))),
                "embedding_index": _nullable_int(candidate.get("source_index", score_row.get("source_index"))),
                "report_id": _nullable_int(candidate.get("report_id")),
                "sent_idx": _nullable_int(candidate.get("sent_idx")),
                "chunk_sent_indices": candidate.get("chunk_sent_indices") or [],
                "source_link": str(source_report.get("link") or ""),
                "source_domain": _normalize_domain(str(source_report.get("domain") or source_report.get("link") or "")),
                "hybrid_rank": _nullable_float(score_row.get("hybrid_rank", position)),
                "hybrid_score": _nullable_float(score_row.get("hybrid_score")),
                "dense_score": _nullable_float(score_row.get("dense_score")),
                "lexical_score": _nullable_float(score_row.get("lexical_score")),
                "bm25_score": _nullable_float(score_row.get("bm25_score")),
                "qd_question_ids": [],
                "qd_question_focuses": [],
                "question_coverage_score": 0.0,
                "union_source": "original",
                "from_baseline": True,
                "from_qd": False,
                "oracle_selected": bool(selected),
                "oracle_step": int(selected_order[candidate_idx]) if selected else -1,
            }
        )
    return [_with_roundtable_score(candidate) for candidate in candidates if candidate["candidate_key"]]


def normalize_qd_union_candidates(
    qd_row: dict[str, Any],
    *,
    oracle_key_to_step: dict[str, int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for position, raw_candidate in enumerate(qd_row.get("candidates") or []):
        candidate = dict(raw_candidate)
        text = str(candidate.get("text") or "")
        key = canonical_candidate_key(candidate)
        if not key:
            continue
        source_report = candidate.get("source_report") if isinstance(candidate.get("source_report"), dict) else {}
        q_routes = list(candidate.get("qd_question_routes") or candidate.get("question_routes") or [])
        question_ids = _unique_strings(route.get("question_id") for route in q_routes)
        focuses = _unique_strings(route.get("focus") for route in q_routes)
        selected = key in oracle_key_to_step
        embedding_index = _first_int(
            candidate.get("original_candidate_idx"),
            candidate.get("source_index"),
            candidate.get("candidate_idx"),
        )
        candidates.append(
            {
                "event_id": str(qd_row.get("event_id") or ""),
                "claim": str(qd_row.get("claim") or ""),
                "gold_label": str(qd_row.get("gold_label") or ""),
                "pool_name": QD_UNION_POOL,
                "pool_position": int(position),
                "candidate_idx": _nullable_int(candidate.get("candidate_idx")),
                "candidate_uid": str(candidate.get("candidate_uid") or ""),
                "candidate_key": key,
                "canonical_text": key,
                "text": text,
                "source_index": _nullable_int(candidate.get("source_index")),
                "embedding_index": embedding_index,
                "original_candidate_idx": _nullable_int(candidate.get("original_candidate_idx")),
                "report_id": _nullable_int(candidate.get("report_id")),
                "sent_idx": _nullable_int(candidate.get("sent_idx")),
                "chunk_sent_indices": candidate.get("chunk_sent_indices") or [],
                "source_link": str(source_report.get("link") or ""),
                "source_domain": _normalize_domain(str(source_report.get("domain") or source_report.get("link") or "")),
                "hybrid_rank": _nullable_float(candidate.get("hybrid_rank", candidate.get("baseline_rank"))),
                "hybrid_score": _first_float(
                    candidate.get("hybrid_score"),
                    candidate.get("baseline_hybrid_score"),
                    candidate.get("qd_max_question_hybrid"),
                ),
                "baseline_rank": _nullable_int(candidate.get("baseline_rank")),
                "baseline_hybrid_score": _nullable_float(candidate.get("baseline_hybrid_score")),
                "qd_pool_rank": _nullable_int(candidate.get("qd_pool_rank")),
                "qd_rrf_score": _nullable_float(candidate.get("qd_rrf_score", candidate.get("rrf_score"))),
                "qd_question_hit_count": _safe_int(candidate.get("qd_question_hit_count", candidate.get("question_hit_count")), 0),
                "qd_max_question_hybrid": _nullable_float(candidate.get("qd_max_question_hybrid", candidate.get("max_question_hybrid"))),
                "qd_question_ids": question_ids,
                "qd_question_focuses": focuses,
                "question_coverage_score": min(1.0, len(question_ids) / 3.0),
                "union_source": str(candidate.get("union_source") or _union_source(candidate)),
                "from_baseline": bool(candidate.get("from_baseline")),
                "from_qd": bool(candidate.get("from_qd")),
                "oracle_selected": bool(selected),
                "oracle_step": int(oracle_key_to_step[key]) if selected else -1,
            }
        )
    return [_with_roundtable_score(candidate) for candidate in candidates]


def apply_auxiliary_labels(
    candidates: list[dict[str, Any]],
    *,
    stance_index: dict[tuple[str, str], dict[str, Any]] | None = None,
    aspect_index: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        stance = _lookup_aux_row(item, stance_index or {})
        if stance:
            item["stance_to_claim"] = str(stance.get("stance_label") or "unclear")
            item["stance_confidence"] = _nullable_float(stance.get("stance_confidence"))
            item["support_score"] = _nullable_float(stance.get("support_score"))
            item["refute_score"] = _nullable_float(stance.get("refute_score"))
            item["neutral_score"] = _nullable_float(stance.get("neutral_score"))
        else:
            item["stance_to_claim"] = "unclear"
            item["stance_confidence"] = None
        aspect = _lookup_aux_row(item, aspect_index or {})
        if aspect:
            item["max_aspect_score"] = _nullable_float(aspect.get("max_aspect_score"))
            item["mean_aspect_score"] = _nullable_float(aspect.get("mean_aspect_score"))
            item["n_aspects"] = _safe_int(aspect.get("n_aspects"), 0)
        else:
            item["max_aspect_score"] = None
            item["mean_aspect_score"] = None
            item["n_aspects"] = 0
        out.append(item)
    return out


def build_auxiliary_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        event_id = str(row.get("event_id") or "")
        for key in _aux_keys(row):
            if event_id and key:
                index[(event_id, key)] = dict(row)
    return index


def cluster_factions_for_pool(
    candidates: list[dict[str, Any]],
    *,
    sample: Any | None = None,
    params: RoundtableParams | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params = params or RoundtableParams()
    if not candidates:
        return [], []

    vectors = _candidate_vectors(candidates, sample=sample)
    clusters = _greedy_clusters(
        vectors,
        candidates,
        similarity_threshold=float(params.similarity_threshold),
        min_factions=int(params.min_factions),
        max_factions=int(params.max_factions),
    )

    labeled: list[dict[str, Any]] = [dict(candidate) for candidate in candidates]
    factions: list[dict[str, Any]] = []
    for faction_idx, member_positions in enumerate(clusters, start=1):
        faction_id = f"F{faction_idx}"
        members = [labeled[pos] for pos in member_positions]
        reps = sorted(members, key=_candidate_sort_key)
        representative = reps[0] if reps else None
        for member in members:
            member["faction_id"] = faction_id
            member["role"] = "representative" if representative and member["candidate_key"] == representative["candidate_key"] else "redundant"
        factions.append(_faction_payload(faction_id, members))

    by_key_role = {(member["candidate_key"], member["pool_position"]): member for member in _flatten_members(labeled)}
    out: list[dict[str, Any]] = []
    for candidate in labeled:
        merged = by_key_role.get((candidate["candidate_key"], candidate["pool_position"]), candidate)
        out.append(merged)
    return out, factions


def build_pool_comparison(
    *,
    event_id: str,
    oracle_keys: Sequence[str],
    original_candidates: Sequence[dict[str, Any]],
    qd_candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    original_keys = {candidate["candidate_key"] for candidate in original_candidates if candidate.get("candidate_key")}
    qd_keys = {candidate["candidate_key"] for candidate in qd_candidates if candidate.get("candidate_key")}
    oracle_set = {key for key in oracle_keys if key}
    overlap = original_keys & qd_keys
    qd_only = qd_keys - original_keys
    original_only = original_keys - qd_keys
    preserved = oracle_set & qd_keys
    dropped = oracle_set - qd_keys
    qd_only_oracle = qd_only & oracle_set
    original_only_oracle = original_only & oracle_set
    return {
        "event_id": event_id,
        "original_pool_size": len(original_keys),
        "qd_union_pool_size": len(qd_keys),
        "overlap_count": len(overlap),
        "qd_only_count": len(qd_only),
        "original_only_count": len(original_only),
        "oracle_selected_count": len(oracle_set),
        "oracle_defined_upper_bound": 1.0,
        "oracle_selected_preserved_by_qd_union_count": len(preserved),
        "oracle_selected_dropped_by_qd_union_count": len(dropped),
        "qd_only_oracle_hit_count": len(qd_only_oracle),
        "original_only_oracle_loss_count": len(original_only_oracle),
        "oracle_selected_preserved_by_qd_union": sorted(preserved),
        "oracle_selected_dropped_by_qd_union": sorted(dropped),
        "overlap": sorted(overlap),
        "qd_only": sorted(qd_only),
        "original_only": sorted(original_only),
    }


def select_pool_order(candidates: Sequence[dict[str, Any]], *, top_k: int, selector_name: str) -> list[dict[str, Any]]:
    ordered = [dict(candidate) for candidate in candidates]
    ordered.sort(key=lambda c: int(c.get("pool_position") or 0))
    return _ranked_selection(ordered, top_k=top_k, selector_name=selector_name)


def select_qd_source_score(candidates: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        baseline_rank = item.get("baseline_rank")
        baseline_component = 0.0
        if item.get("from_baseline"):
            baseline_component += 0.04
            if baseline_rank is not None:
                baseline_component += 0.01 / max(float(baseline_rank), 1.0)
        qd_component = float(item.get("qd_rrf_score") or 0.0)
        qd_component += 0.004 * float(item.get("qd_question_hit_count") or 0.0)
        qd_component += 0.01 * float(item.get("qd_max_question_hybrid") or 0.0)
        item["source_score"] = float(item.get("union_source_score") or (baseline_component + qd_component))
        scored.append(item)
    scored.sort(
        key=lambda c: (
            -float(c.get("source_score") or 0.0),
            int(c.get("baseline_rank") or 10**9),
            int(c.get("qd_pool_rank") or 10**9),
            int(c.get("pool_position") or 10**9),
        )
    )
    return _ranked_selection(scored, top_k=top_k, selector_name="qd_union_source_score_top5")


def select_roundtable_topk(
    candidates: Sequence[dict[str, Any]],
    factions: Sequence[dict[str, Any]],
    *,
    top_k: int,
    selector_name: str,
) -> list[dict[str, Any]]:
    by_faction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_faction[str(candidate.get("faction_id") or "")].append(dict(candidate))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered_factions = sorted(
        factions,
        key=lambda faction: (
            -float(faction.get("strength_score") or 0.0),
            str(faction.get("faction_id") or ""),
        ),
    )
    for faction in ordered_factions:
        members = sorted(by_faction.get(str(faction.get("faction_id") or ""), []), key=_candidate_sort_key)
        for member in members:
            key = str(member.get("candidate_key") or "")
            if key and key not in seen:
                selected.append(member)
                seen.add(key)
                break
        if len(selected) >= top_k:
            return _ranked_selection(selected, top_k=top_k, selector_name=selector_name)

    for candidate in sorted([dict(candidate) for candidate in candidates], key=_candidate_sort_key):
        key = str(candidate.get("candidate_key") or "")
        if key and key not in seen:
            selected.append(candidate)
            seen.add(key)
        if len(selected) >= top_k:
            break
    return _ranked_selection(selected, top_k=top_k, selector_name=selector_name)


def selection_metrics(oracle_keys: Sequence[str], selected: Sequence[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    oracle = _dedupe([str(key) for key in oracle_keys if key])[:top_k]
    pred = _dedupe([str(candidate.get("candidate_key") or "") for candidate in selected if candidate.get("candidate_key")])[:top_k]
    oracle_set = set(oracle)
    pred_set = set(pred)
    overlap = oracle_set & pred_set
    union = oracle_set | pred_set
    return {
        "recall@5": float(len(overlap) / max(len(oracle_set), 1)),
        "precision@5": float(len(overlap) / max(len(pred), 1)),
        "jaccard@5": float(len(overlap) / max(len(union), 1)),
        "top1_match": float(bool(oracle and pred and oracle[0] == pred[0])),
        "oracle_rank_ndcg@5": _oracle_rank_ndcg_text(oracle, pred, top_k=top_k),
        "set_overlap": int(len(overlap)),
    }


def build_selection_trace(
    *,
    event_id: str,
    claim: str,
    gold_label: str,
    pool_name: str,
    selector_name: str,
    selected: Sequence[dict[str, Any]],
    oracle_keys: Sequence[str],
    oracle_faction_ids: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    trace = {
        "event_id": event_id,
        "claim": claim,
        "gold_label": gold_label,
        "pool_name": pool_name,
        "selector_name": selector_name,
        "oracle_ordered_keys": list(oracle_keys),
        "oracle_faction_ids": list(oracle_faction_ids),
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected],
        "selected_faction_ids": [str(candidate.get("faction_id") or "") for candidate in selected],
        "selected_source_domains": [str(candidate.get("source_domain") or "") for candidate in selected],
        "selected_candidates": [
            _candidate_output(candidate, include_text=True)
            for candidate in selected
        ],
    }
    trace.update(selection_metrics(oracle_keys, selected, top_k=top_k))
    return trace


def summarize_pool_deltas(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_oracle = sum(int(row.get("oracle_selected_count") or 0) for row in rows)
    total_qd_only = sum(int(row.get("qd_only_count") or 0) for row in rows)
    total_preserved = sum(int(row.get("oracle_selected_preserved_by_qd_union_count") or 0) for row in rows)
    total_dropped = sum(int(row.get("oracle_selected_dropped_by_qd_union_count") or 0) for row in rows)
    total_qd_only_oracle = sum(int(row.get("qd_only_oracle_hit_count") or 0) for row in rows)
    total_original_only_oracle = sum(int(row.get("original_only_oracle_loss_count") or 0) for row in rows)
    return {
        "n_events": len(rows),
        "oracle_defined_upper_bound": 1.0,
        "mean_original_pool_size": _mean(row.get("original_pool_size") for row in rows),
        "mean_qd_union_pool_size": _mean(row.get("qd_union_pool_size") for row in rows),
        "mean_overlap_count": _mean(row.get("overlap_count") for row in rows),
        "mean_qd_only_count": _mean(row.get("qd_only_count") for row in rows),
        "mean_original_only_count": _mean(row.get("original_only_count") for row in rows),
        "qd_union_preserved_oracle_selected_rate": float(total_preserved / max(total_oracle, 1)),
        "qd_union_dropped_oracle_selected_rate": float(total_dropped / max(total_oracle, 1)),
        "qd_only_oracle_hit_rate": float(total_qd_only_oracle / max(total_qd_only, 1)),
        "original_only_oracle_loss_rate": float(total_original_only_oracle / max(total_oracle, 1)),
        "total_oracle_selected": int(total_oracle),
        "total_oracle_preserved_by_qd_union": int(total_preserved),
        "total_oracle_dropped_by_qd_union": int(total_dropped),
    }


def summarize_faction_metrics(
    *,
    faction_rows: Sequence[dict[str, Any]],
    selection_traces: Sequence[dict[str, Any]],
    pool_comparisons: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    pool_payload: dict[str, Any] = {}
    for pool_name in (ORIGINAL_POOL, QD_UNION_POOL):
        summaries = [row.get(pool_name, {}) for row in faction_rows]
        pool_payload[pool_name] = {
            "mean_pool_size": _mean(summary.get("pool_size") for summary in summaries),
            "mean_factions_per_event": _mean(summary.get("n_factions") for summary in summaries),
            "single_faction_collapse_rate": _mean_bool(int(summary.get("n_factions") or 0) <= 1 for summary in summaries),
            "mean_source_domains": _mean(summary.get("n_source_domains") for summary in summaries),
            "mean_stance_entropy": _mean(summary.get("stance_entropy") for summary in summaries),
            "mean_question_coverage": _mean(summary.get("question_coverage") for summary in summaries),
        }

    selector_payload: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in selection_traces:
        grouped[str(trace.get("selector_name") or "")].append(trace)
    for selector_name, traces in sorted(grouped.items()):
        selector_payload[selector_name] = _summarize_selection_traces(traces)

    return {
        "pools": pool_payload,
        "selectors": selector_payload,
        "pool_delta": summarize_pool_deltas(pool_comparisons),
    }


def decision_payload(
    *,
    faction_metrics: dict[str, Any],
    pool_delta_metrics: dict[str, Any],
    min_lift_pp: float = 1.0,
    max_pool_drop_rate_for_go: float = 0.05,
) -> dict[str, Any]:
    selectors = faction_metrics.get("selectors", {})
    original = selectors.get("original_pool_order_top5", {})
    qd_roundtable = selectors.get("roundtable_qd_union_top5", {})
    original_jaccard = float(original.get("jaccard@5", 0.0))
    qd_roundtable_jaccard = float(qd_roundtable.get("jaccard@5", 0.0))
    original_top1 = float(original.get("top1_match", 0.0))
    qd_roundtable_top1 = float(qd_roundtable.get("top1_match", 0.0))
    jaccard_lift_pp = (qd_roundtable_jaccard - original_jaccard) * 100.0
    top1_lift_pp = (qd_roundtable_top1 - original_top1) * 100.0
    qd_drop = float(pool_delta_metrics.get("qd_union_dropped_oracle_selected_rate", 1.0))
    original_factions = faction_metrics.get("pools", {}).get(ORIGINAL_POOL, {}).get("mean_factions_per_event", 0.0)
    qd_factions = faction_metrics.get("pools", {}).get(QD_UNION_POOL, {}).get("mean_factions_per_event", 0.0)
    richer_qd = float(qd_factions) >= float(original_factions) + 0.25
    improves_selection = jaccard_lift_pp >= float(min_lift_pp) or top1_lift_pp >= float(min_lift_pp)
    low_pool_drop = qd_drop <= float(max_pool_drop_rate_for_go)
    if improves_selection and low_pool_drop:
        decision = "go_roundtable_qd_union"
    elif richer_qd or float(pool_delta_metrics.get("qd_union_preserved_oracle_selected_rate", 0.0)) > original_jaccard:
        decision = "analysis_only_qd_pool_better_coverage"
    else:
        decision = "stop_roundtable_v0"
    return {
        "decision": decision,
        "min_lift_pp": float(min_lift_pp),
        "max_pool_drop_rate_for_go": float(max_pool_drop_rate_for_go),
        "roundtable_qd_union_jaccard_lift_pp": float(jaccard_lift_pp),
        "roundtable_qd_union_top1_lift_pp": float(top1_lift_pp),
        "qd_union_dropped_oracle_selected_rate": float(qd_drop),
        "original_mean_factions_per_event": float(original_factions),
        "qd_union_mean_factions_per_event": float(qd_factions),
        "richer_qd_pool": bool(richer_qd),
        "improves_selection": bool(improves_selection),
        "low_pool_drop_for_go": bool(low_pool_drop),
    }


def pool_summary(candidates: Sequence[dict[str, Any]], factions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    domains = {str(candidate.get("source_domain") or "") for candidate in candidates if candidate.get("source_domain")}
    question_ids = {
        str(question_id)
        for candidate in candidates
        for question_id in (candidate.get("qd_question_ids") or [])
        if question_id
    }
    stance_counts = Counter(str(candidate.get("stance_to_claim") or "unclear") for candidate in candidates)
    return {
        "pool_size": len(candidates),
        "n_factions": len(factions),
        "n_source_domains": len(domains),
        "source_domains": sorted(domains),
        "stance_counts": dict(sorted(stance_counts.items())),
        "stance_entropy": _entropy(stance_counts.values()),
        "n_question_ids": len(question_ids),
        "question_coverage": min(1.0, len(question_ids) / 3.0),
        "factions": list(factions),
    }


def candidate_role_rows(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_candidate_output(candidate, include_text=False) for candidate in candidates]


def _score_rows_by_candidate_idx(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        idx = _safe_int(row.get("candidate_idx"), position)
        out[idx] = dict(row)
    return out


def _with_roundtable_score(candidate: dict[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    relevance = _first_float(
        item.get("hybrid_score"),
        item.get("baseline_hybrid_score"),
        item.get("qd_max_question_hybrid"),
        0.0,
    )
    qd_rrf = min(1.0, max(0.0, float(item.get("qd_rrf_score") or 0.0) * 20.0))
    question_coverage = float(item.get("question_coverage_score") or 0.0)
    item["roundtable_score"] = float(0.65 * _clip01(relevance) + 0.20 * question_coverage + 0.15 * qd_rrf)
    return item


def _candidate_vectors(candidates: Sequence[dict[str, Any]], *, sample: Any | None) -> np.ndarray:
    if sample is not None and getattr(sample, "chunk_emb", None) is not None:
        chunk_emb = np.asarray(sample.chunk_emb, dtype=np.float32)
        rows: list[np.ndarray] = []
        for candidate in candidates:
            idx = _nullable_int(candidate.get("embedding_index"))
            if idx is None or idx < 0 or idx >= int(chunk_emb.shape[0]):
                rows = []
                break
            rows.append(np.asarray(chunk_emb[idx], dtype=np.float32))
        if rows:
            return _normalize_matrix(np.stack(rows, axis=0))
    return _hashed_text_vectors([str(candidate.get("text") or "") for candidate in candidates])


def _hashed_text_vectors(texts: Sequence[str], *, dim: int = 256) -> np.ndarray:
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in canonicalize_sentence(text).split():
            vectors[row, hash(token) % dim] += 1.0
    return _normalize_matrix(vectors)


def _normalize_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return arr / norms


def _greedy_clusters(
    vectors: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float,
    min_factions: int,
    max_factions: int,
) -> list[list[int]]:
    n = len(candidates)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    similarity = np.asarray(vectors @ vectors.T, dtype=np.float32)
    order = sorted(range(n), key=lambda idx: _candidate_sort_key(candidates[idx]))
    clusters: list[list[int]] = []
    for idx in order:
        best_cluster = -1
        best_sim = -1.0
        for cluster_idx, cluster in enumerate(clusters):
            sim = float(max(similarity[idx, member] for member in cluster))
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_idx
        if best_cluster >= 0 and best_sim >= float(similarity_threshold):
            clusters[best_cluster].append(idx)
        else:
            clusters.append([idx])

    while len(clusters) > max(1, int(max_factions)):
        left, right = _closest_cluster_pair(clusters, similarity)
        clusters[left].extend(clusters[right])
        del clusters[right]

    while len(clusters) < min(int(min_factions), n):
        largest_idx = max(range(len(clusters)), key=lambda idx: len(clusters[idx]))
        if len(clusters[largest_idx]) <= 1:
            break
        split_member = _least_central_member(clusters[largest_idx], similarity)
        clusters[largest_idx].remove(split_member)
        clusters.append([split_member])

    clusters = [sorted(cluster, key=lambda idx: _candidate_sort_key(candidates[idx])) for cluster in clusters]
    clusters.sort(key=lambda cluster: _candidate_sort_key(candidates[cluster[0]]))
    return clusters


def _closest_cluster_pair(clusters: Sequence[Sequence[int]], similarity: np.ndarray) -> tuple[int, int]:
    best = (0, 1)
    best_sim = -1.0
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            sims = [float(similarity[i, j]) for i in clusters[left] for j in clusters[right]]
            sim = max(sims) if sims else -1.0
            if sim > best_sim:
                best = (left, right)
                best_sim = sim
    return best


def _least_central_member(cluster: Sequence[int], similarity: np.ndarray) -> int:
    best_member = cluster[0]
    best_score = math.inf
    for member in cluster:
        others = [other for other in cluster if other != member]
        score = float(np.mean([similarity[member, other] for other in others])) if others else math.inf
        if score < best_score:
            best_member = member
            best_score = score
    return int(best_member)


def _faction_payload(faction_id: str, members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted([dict(member) for member in members], key=_candidate_sort_key)
    representative = ordered[0] if ordered else {}
    stance_counts = Counter(str(member.get("stance_to_claim") or "unclear") for member in ordered)
    question_ids = sorted(
        {
            str(question_id)
            for member in ordered
            for question_id in (member.get("qd_question_ids") or [])
            if question_id
        }
    )
    focuses = Counter(
        str(focus)
        for member in ordered
        for focus in (member.get("qd_question_focuses") or [])
        if focus
    )
    domains = sorted({str(member.get("source_domain") or "") for member in ordered if member.get("source_domain")})
    report_ids = sorted(
        {
            int(member["report_id"])
            for member in ordered
            if member.get("report_id") is not None
        }
    )
    stance = _majority_label(stance_counts, default="unclear")
    focus_label = _majority_label(focuses, default="retrieval")
    return {
        "faction_id": faction_id,
        "faction_label": f"{focus_label}_{stance}",
        "stance_to_claim": stance,
        "size": len(ordered),
        "representative_candidate_keys": [str(representative.get("candidate_key") or "")] if representative else [],
        "redundant_candidate_keys": [str(member.get("candidate_key") or "") for member in ordered[1:]],
        "covered_question_ids": question_ids,
        "source_domains": domains,
        "report_ids": report_ids,
        "strength_score": _max(member.get("roundtable_score") for member in ordered),
        "coverage_score": min(1.0, len(question_ids) / 3.0),
        "source_diversity": float(len(domains) / max(len(ordered), 1)),
        "conflicting_faction_ids": [],
    }


def add_conflicting_factions(factions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(faction) for faction in factions]
    for left in out:
        conflicts: list[str] = []
        for right in out:
            if left["faction_id"] == right["faction_id"]:
                continue
            if {left.get("stance_to_claim"), right.get("stance_to_claim")} == {"support", "refute"}:
                conflicts.append(str(right["faction_id"]))
        left["conflicting_faction_ids"] = sorted(conflicts)
    return out


def _flatten_members(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(candidate) for candidate in candidates]


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(candidate.get("roundtable_score") or 0.0),
        -float(candidate.get("hybrid_score") or 0.0),
        int(candidate.get("pool_position") or 10**9),
        str(candidate.get("candidate_key") or ""),
    )


def _ranked_selection(candidates: Sequence[dict[str, Any]], *, top_k: int, selector_name: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.get("candidate_key") or "")
        if not key or key in seen:
            continue
        item = dict(candidate)
        item["selector_name"] = selector_name
        item["selection_rank"] = len(selected) + 1
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def _candidate_output(candidate: dict[str, Any], *, include_text: bool) -> dict[str, Any]:
    keys = [
        "event_id",
        "pool_name",
        "candidate_key",
        "candidate_uid",
        "pool_position",
        "selection_rank",
        "candidate_idx",
        "source_index",
        "original_candidate_idx",
        "report_id",
        "source_domain",
        "faction_id",
        "role",
        "stance_to_claim",
        "qd_question_ids",
        "oracle_selected",
        "oracle_step",
        "roundtable_score",
    ]
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["question_ids"] = list(candidate.get("qd_question_ids") or [])
    if include_text:
        out["text"] = candidate.get("text")
    return out


def _summarize_selection_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_events": len(traces),
        "recall@5": _mean(trace.get("recall@5") for trace in traces),
        "precision@5": _mean(trace.get("precision@5") for trace in traces),
        "jaccard@5": _mean(trace.get("jaccard@5") for trace in traces),
        "top1_match": _mean(trace.get("top1_match") for trace in traces),
        "oracle_rank_ndcg@5": _mean(trace.get("oracle_rank_ndcg@5") for trace in traces),
        "mean_source_domains_per_top5": _mean(len(set(trace.get("selected_source_domains") or [])) for trace in traces),
        "source_entropy@5": _mean(_entropy(Counter(trace.get("selected_source_domains") or []).values()) for trace in traces),
        "oracle_faction_recall@5": _mean(_trace_oracle_faction_recall(trace) for trace in traces),
    }


def _trace_oracle_faction_recall(trace: dict[str, Any]) -> float:
    oracle_factions = {str(fid) for fid in (trace.get("oracle_faction_ids") or []) if fid}
    selected_factions = {str(fid) for fid in (trace.get("selected_faction_ids") or []) if fid}
    if not oracle_factions:
        return 0.0
    return float(len(oracle_factions & selected_factions) / max(len(oracle_factions), 1))


def _oracle_rank_ndcg_text(oracle: Sequence[str], pred: Sequence[str], *, top_k: int) -> float:
    rel_by_key = {key: max(int(top_k) - rank, 1) for rank, key in enumerate(oracle[:top_k])}
    dcg = 0.0
    for rank, key in enumerate(pred[:top_k], start=1):
        rel = rel_by_key.get(key, 0)
        dcg += (2.0**rel - 1.0) / math.log2(rank + 1)
    ideal_rels = sorted(rel_by_key.values(), reverse=True)[:top_k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _lookup_aux_row(candidate: dict[str, Any], index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    event_id = str(candidate.get("event_id") or "")
    for key in _aux_keys(candidate):
        row = index.get((event_id, key))
        if row is not None:
            return row
    return None


def _aux_keys(row: dict[str, Any]) -> list[str]:
    keys = [
        str(row.get("candidate_key") or ""),
        canonicalize_sentence(str(row.get("text") or "")),
        str(row.get("candidate_uid") or ""),
    ]
    candidate_idx = row.get("candidate_idx")
    if candidate_idx is not None:
        keys.append(f"candidate_idx:{candidate_idx}")
    source_index = row.get("source_index")
    if source_index is not None:
        keys.append(f"source_index:{source_index}")
    return [key for key in keys if key]


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return (parsed.netloc or parsed.path).lower().strip("/")


def _union_source(candidate: dict[str, Any]) -> str:
    if candidate.get("from_baseline") and candidate.get("from_qd"):
        return "baseline+qd"
    if candidate.get("from_baseline"):
        return "baseline"
    if candidate.get("from_qd"):
        return "qd"
    return "unknown"


def _unique_strings(values: Iterable[Any]) -> list[str]:
    return _dedupe(str(value) for value in values if value is not None and str(value))


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _majority_label(counts: Counter[str], *, default: str) -> str:
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _entropy(counts: Iterable[int]) -> float:
    values = np.asarray([float(value) for value in counts if float(value) > 0.0], dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values / total
    return float(-(probs * np.log2(probs)).sum())


def _mean(values: Iterable[Any]) -> float:
    nums = [_safe_float(value, math.nan) for value in values]
    nums = [value for value in nums if not math.isnan(value)]
    return float(np.mean(nums)) if nums else 0.0


def _mean_bool(values: Iterable[bool]) -> float:
    nums = [1.0 if bool(value) else 0.0 for value in values]
    return float(np.mean(nums)) if nums else 0.0


def _max(values: Iterable[Any]) -> float:
    nums = [_safe_float(value, math.nan) for value in values]
    nums = [value for value in nums if not math.isnan(value)]
    return float(max(nums)) if nums else 0.0


def _clip01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _nullable_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _nullable_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _first_int(*values: Any) -> int | None:
    for value in values:
        out = _nullable_int(value)
        if out is not None:
            return out
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        out = _nullable_float(value)
        if out is not None:
            return out
    return None
