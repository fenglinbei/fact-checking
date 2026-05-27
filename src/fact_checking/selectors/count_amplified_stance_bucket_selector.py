from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fact_checking.selectors.evidence_quality import (
    ORIGINAL_POOL,
    QD_UNION_POOL,
    canonical_candidate_key,
    dedup_key_for_candidate,
    enrich_quality_fields,
    quality_gate,
    question_coverage_score,
    question_route_weight,
    retrieval_score,
    source_domain,
    source_group,
    stable_candidate_uid,
)
from fact_checking.selectors.stance_buckets import (
    derive_stance_fields,
    normalized_entropy,
)


COUNT_AMPLIFIED_SELECTOR = "count_amplified_stance_bucket_top5"
LINEAR_STANCE_SELECTOR = "linear_stance_bucket_count_top5"


@dataclass(frozen=True)
class CountAmplifiedParams:
    top_k: int = 5
    alpha: float = 0.5
    gamma_stance: float = 1.6
    rho: float = 0.6
    tau_c: float = 0.50
    tau_r: float = 0.15
    min_bucket_membership: float | None = None
    n_stance_buckets: int = 3


def build_union_analysis_row(
    oracle_row: dict[str, Any],
    qd_row: dict[str, Any] | None,
) -> dict[str, Any]:
    event_id = str(oracle_row.get("event_id") or (qd_row or {}).get("event_id") or "")
    claim = str(oracle_row.get("claim") or (qd_row or {}).get("claim") or "")
    oracle_keys = oracle_ordered_keys(oracle_row)
    oracle_key_to_step = {key: idx for idx, key in enumerate(oracle_keys)}
    merged: dict[str, dict[str, Any]] = {}
    key_by_text: dict[str, str] = {}

    for position, candidate in enumerate(oracle_row.get("candidate_pool") or []):
        normalized = _normalize_original_candidate(
            oracle_row,
            candidate,
            position=position,
            oracle_key_to_step=oracle_key_to_step,
        )
        _merge_candidate(merged, key_by_text, normalized)

    if qd_row is not None:
        for position, candidate in enumerate(qd_row.get("candidates") or []):
            normalized = _normalize_qd_candidate(
                qd_row,
                candidate,
                position=position,
                oracle_key_to_step=oracle_key_to_step,
            )
            _merge_candidate(merged, key_by_text, normalized)

    candidates = list(merged.values())
    candidates.sort(key=_union_sort_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["union_pool_rank"] = int(rank)
        candidate["union_source"] = _union_source(candidate)
        candidate["source_pools"] = _ordered_source_pools(candidate.get("source_pools") or [])
        candidate["candidate_uid"] = stable_candidate_uid(event_id, str(candidate.get("dedup_key") or candidate.get("candidate_key") or rank))

    return {
        "event_id": event_id,
        "claim": claim,
        "gold_label": str(oracle_row.get("gold_label") or ""),
        "oracle_ordered_keys": oracle_keys,
        "oracle_selected_count": int(len(oracle_keys)),
        "candidates": candidates,
    }


def build_union_analysis_rows(
    oracle_rows: Sequence[dict[str, Any]],
    qd_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qd_by_event = {str(row.get("event_id") or ""): row for row in qd_rows}
    rows: list[dict[str, Any]] = []
    missing_qd: list[str] = []
    for oracle_row in oracle_rows:
        event_id = str(oracle_row.get("event_id") or "")
        qd_row = qd_by_event.get(event_id)
        if qd_row is None:
            missing_qd.append(event_id)
        rows.append(build_union_analysis_row(oracle_row, qd_row))
    summary = {
        "n_events": len(rows),
        "n_missing_qd_events": len(missing_qd),
        "missing_qd_events_sample": missing_qd[:10],
        "mean_union_pool_size": _mean(len(row.get("candidates") or []) for row in rows),
        "mean_original_pool_size": _mean(
            sum(1 for candidate in row.get("candidates") or [] if _is_original_stage2_candidate(candidate))
            for row in rows
        ),
        "mean_qd_pool_size": _mean(
            sum(1 for candidate in row.get("candidates") or [] if candidate.get("from_qd"))
            for row in rows
        ),
    }
    return rows, summary


def enrich_quality_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        claim = str(row.get("claim") or "")
        item = dict(row)
        item["candidates"] = [
            enrich_quality_fields(dict(candidate), claim=claim)
            for candidate in (row.get("candidates") or [])
        ]
        out.append(item)
    return out


def attach_stance_annotations(
    rows: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    *,
    n_stance_buckets: int,
    tau: float = 2.0,
) -> list[dict[str, Any]]:
    annotation_by_uid = {
        str(row.get("candidate_uid") or ""): row
        for row in annotations
        if row.get("candidate_uid")
    }
    annotation_by_key = {
        str(row.get("annotation_key") or ""): row
        for row in annotations
        if row.get("annotation_key")
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        claim = str(row.get("claim") or "")
        item = dict(row)
        candidates: list[dict[str, Any]] = []
        for candidate in row.get("candidates") or []:
            ann = annotation_by_uid.get(str(candidate.get("candidate_uid") or ""))
            if ann is None:
                ann = annotation_by_key.get(str(candidate.get("annotation_key") or ""))
            candidates.append(
                attach_stance_annotation(
                    dict(candidate),
                    annotation=ann,
                    claim=claim,
                    n_stance_buckets=n_stance_buckets,
                    tau=tau,
                )
            )
        item["candidates"] = candidates
        item["n_stance_buckets"] = int(n_stance_buckets)
        item["stance_bucket_names"] = _bucket_names_from_candidates(candidates)
        out.append(item)
    return out


def attach_stance_annotation(
    candidate: dict[str, Any],
    *,
    annotation: dict[str, Any] | None,
    claim: str,
    n_stance_buckets: int,
    tau: float,
) -> dict[str, Any]:
    if annotation:
        teacher = dict(annotation.get("teacher_annotation") or annotation)
        stance_score = _safe_float(teacher.get("stance_score"), 5.5)
        completeness = _safe_float(teacher.get("semantic_completeness"), math.nan)
        semantic_score = None if math.isnan(completeness) else completeness / 10.0
        item = enrich_quality_fields(
            candidate,
            claim=claim,
            semantic_completeness_score=semantic_score,
            annotation_missing=False,
        )
        item["teacher_annotation"] = {
            "stance_score": float(stance_score),
            "semantic_completeness": None if math.isnan(completeness) else float(completeness),
            "stance_score_clamped": bool(teacher.get("stance_score_clamped", False)),
            "semantic_completeness_clamped": bool(teacher.get("semantic_completeness_clamped", False)),
        }
        item["annotation_key"] = str(annotation.get("annotation_key") or item.get("annotation_key") or "")
    else:
        item = enrich_quality_fields(candidate, claim=claim, annotation_missing=True)
        stance_score = 5.5
        item["teacher_annotation"] = {
            "stance_score": None,
            "semantic_completeness": None,
            "stance_score_clamped": False,
            "semantic_completeness_clamped": False,
        }

    item.update(
        derive_stance_fields(
            stance_score=stance_score,
            n_stance_buckets=int(n_stance_buckets),
            tau=float(tau),
        )
    )
    return item


def select_count_amplified_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    params: CountAmplifiedParams,
    selector_name: str = COUNT_AMPLIFIED_SELECTOR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = [dict(candidate) for candidate in candidates]
    bucket_names = _bucket_names_from_candidates(candidates)
    if not bucket_names:
        return [], [], {"bucket_names": [], "effective_counts": {}, "bucket_mass": {}}
    min_membership = (
        1.0 / float(len(bucket_names))
        if params.min_bucket_membership is None
        else float(params.min_bucket_membership)
    )
    dedup_weights = source_dedup_weights(candidates, bucket_names=bucket_names, params=params)
    count_payload = bucket_count_payload(
        candidates,
        bucket_names=bucket_names,
        params=params,
        dedup_weights=dedup_weights,
    )
    selected: list[dict[str, Any]] = []
    slot_trace: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_source_groups: set[str] = set()
    selected_count_by_bucket: Counter[str] = Counter()

    for slot in range(1, int(params.top_k) + 1):
        bucket_scores: dict[str, float] = {}
        for bucket in bucket_names:
            if not _has_eligible_remaining(
                candidates,
                bucket=bucket,
                selected_keys=selected_keys,
                min_membership=min_membership,
                params=params,
            ):
                continue
            mass = float(count_payload["bucket_mass"].get(bucket, 0.0))
            bucket_scores[bucket] = mass / ((1.0 + float(selected_count_by_bucket[bucket])) ** float(params.rho))
        if not bucket_scores:
            break
        chosen_bucket = max(bucket_scores, key=lambda name: (bucket_scores[name], -bucket_names.index(name)))
        candidate = _best_candidate_in_bucket(
            candidates,
            bucket=chosen_bucket,
            selected_keys=selected_keys,
            selected_source_groups=selected_source_groups,
            min_membership=min_membership,
            params=params,
        )
        if candidate is None:
            break
        selected_key = str(candidate.get("candidate_key") or "")
        selected_keys.add(selected_key)
        selected_source_groups.add(str(candidate.get("source_group") or source_group(candidate)))
        selected_count_by_bucket[chosen_bucket] += 1
        item = dict(candidate)
        item["selector_name"] = selector_name
        item["selection_rank"] = int(slot)
        item["selected_stance_bucket"] = chosen_bucket
        item["slot_score"] = float(bucket_scores[chosen_bucket])
        selected.append(item)
        slot_trace.append(
            {
                "slot": int(slot),
                "chosen_stance_bucket": chosen_bucket,
                "bucket_mass": float(count_payload["bucket_mass"].get(chosen_bucket, 0.0)),
                "slot_score": float(bucket_scores[chosen_bucket]),
                "selected_candidate_uid": str(item.get("candidate_uid") or ""),
                "selected_candidate_key": selected_key,
                "selected_text": str(item.get("text") or ""),
                "oracle_selected": bool(item.get("oracle_selected", False)),
                "candidate_score": float(item.get("candidate_score", 0.0)),
            }
        )
    return selected, slot_trace, count_payload


def bucket_count_payload(
    candidates: Sequence[dict[str, Any]],
    *,
    bucket_names: Sequence[str],
    params: CountAmplifiedParams,
    dedup_weights: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    dedup_weights = dedup_weights or source_dedup_weights(candidates, bucket_names=bucket_names, params=params)
    effective: dict[str, float] = {bucket: 0.0 for bucket in bucket_names}
    raw_effective: dict[str, float] = {bucket: 0.0 for bucket in bucket_names}
    eligible_counts: dict[str, int] = {bucket: 0 for bucket in bucket_names}
    for candidate in candidates:
        if not quality_gate(candidate, tau_c=params.tau_c, tau_r=params.tau_r):
            continue
        probs = dict(candidate.get("teacher_stance_probs") or {})
        q_weight = question_route_weight(candidate)
        candidate_uid = str(candidate.get("candidate_uid") or candidate.get("candidate_key") or "")
        for bucket in bucket_names:
            membership = _safe_float(probs.get(bucket), 0.0)
            if membership <= 0.0:
                continue
            raw_effective[bucket] += membership * q_weight
            effective[bucket] += membership * q_weight * float(dedup_weights.get((candidate_uid, bucket), 1.0))
            if membership >= (params.min_bucket_membership or (1.0 / max(len(bucket_names), 1))):
                eligible_counts[bucket] += 1
    mass = {
        bucket: float((effective[bucket] + float(params.alpha)) ** float(params.gamma_stance))
        for bucket in bucket_names
    }
    return {
        "bucket_names": list(bucket_names),
        "effective_counts": effective,
        "raw_effective_counts": raw_effective,
        "bucket_mass": mass,
        "eligible_counts": eligible_counts,
        "params": params.__dict__,
    }


def source_dedup_weights(
    candidates: Sequence[dict[str, Any]],
    *,
    bucket_names: Sequence[str],
    params: CountAmplifiedParams,
) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    candidate_buckets: dict[str, str] = {}
    for candidate in candidates:
        if not quality_gate(candidate, tau_c=params.tau_c, tau_r=params.tau_r):
            continue
        probs = dict(candidate.get("teacher_stance_probs") or {})
        if not probs:
            continue
        bucket = max(bucket_names, key=lambda name: _safe_float(probs.get(name), 0.0))
        candidate_uid = str(candidate.get("candidate_uid") or candidate.get("candidate_key") or "")
        candidate_buckets[candidate_uid] = bucket
        counts[(bucket, source_group(candidate))] += 1

    weights: dict[tuple[str, str], float] = {}
    for candidate in candidates:
        candidate_uid = str(candidate.get("candidate_uid") or candidate.get("candidate_key") or "")
        argmax_bucket = candidate_buckets.get(candidate_uid)
        group = source_group(candidate)
        for bucket in bucket_names:
            if bucket == argmax_bucket:
                n_same = max(int(counts.get((bucket, group), 1)), 1)
                weights[(candidate_uid, bucket)] = 1.0 / math.sqrt(float(n_same))
            else:
                weights[(candidate_uid, bucket)] = 1.0
    return weights


def candidate_score_in_bucket(
    candidate: dict[str, Any],
    *,
    bucket: str,
    selected_source_groups: set[str] | None = None,
) -> float:
    probs = dict(candidate.get("teacher_stance_probs") or {})
    source_bonus = 1.0
    if selected_source_groups is not None and source_group(candidate) in selected_source_groups:
        source_bonus = 0.0
    score = (
        0.30 * retrieval_score(candidate)
        + 0.30 * _safe_float(candidate.get("semantic_completeness_score"), 0.0)
        + 0.20 * _safe_float(probs.get(bucket), 0.0)
        + 0.10 * question_coverage_score(candidate)
        + 0.10 * source_bonus
    )
    return float(score)


def select_order_control(
    candidates: Sequence[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    if mode == "original_pool_order_top5":
        rows = [row for row in rows if _is_original_stage2_candidate(row)]
        rows.sort(key=lambda row: (int(row.get("baseline_rank") or 10**9), int(row.get("union_pool_rank") or 10**9)))
    elif mode == "qd_union_pool_order_top5":
        rows = [row for row in rows if row.get("from_qd")]
        rows.sort(key=lambda row: (int(row.get("qd_pool_rank") or 10**9), int(row.get("union_pool_rank") or 10**9)))
    elif mode == "qd_union_source_score_top5":
        for row in rows:
            row["source_score"] = _source_score(row)
        rows.sort(
            key=lambda row: (
                -float(row.get("source_score") or 0.0),
                int(row.get("baseline_rank") or 10**9),
                int(row.get("qd_pool_rank") or 10**9),
                int(row.get("union_pool_rank") or 10**9),
            )
        )
    elif mode == "completeness_only_top5":
        rows.sort(
            key=lambda row: (
                -float(row.get("semantic_completeness_score") or 0.0),
                -retrieval_score(row),
                int(row.get("union_pool_rank") or 10**9),
            )
        )
    else:
        raise ValueError(f"Unknown control mode: {mode}")
    return _ranked(rows, top_k=top_k, selector_name=mode)


def text_ordered_selection_metrics(
    oracle_ordered_keys: Sequence[str],
    selected: Sequence[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    oracle = _dedupe([str(key) for key in oracle_ordered_keys if key])[:top_k]
    pred = _dedupe([str(candidate.get("candidate_key") or "") for candidate in selected if candidate.get("candidate_key")])[:top_k]
    oracle_set = set(oracle)
    pred_set = set(pred)
    overlap = oracle_set & pred_set
    union = oracle_set | pred_set
    pairwise_acc, pair_count = _pairwise_order_acc(oracle, pred)
    return {
        "set_overlap": int(len(overlap)),
        "recall@5": float(len(overlap) / max(len(oracle_set), 1)),
        "precision@5": float(len(overlap) / max(len(pred), 1)),
        "jaccard@5": float(len(overlap) / max(len(union), 1)),
        "top1_match": float(bool(oracle and pred and oracle[0] == pred[0])),
        "oracle_rank_ndcg@5": _oracle_rank_ndcg(oracle, pred, top_k=top_k),
        "pairwise_order_acc@5": float(pairwise_acc),
        "overlap_pair_count": int(pair_count),
    }


def build_selector_trace(
    row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    selector_name: str,
    top_k: int,
    slot_trace: Sequence[dict[str, Any]] | None = None,
    count_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = list(selected)
    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "selector_name": selector_name,
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected],
        "selected_candidates": [_candidate_output(candidate) for candidate in selected],
        "slot_trace": list(slot_trace or []),
        "count_payload": count_payload or {},
    }
    trace.update(text_ordered_selection_metrics(trace["oracle_ordered_keys"], selected, top_k=top_k))
    trace.update(selection_quality_metrics(selected))
    return trace


def selection_quality_metrics(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(selected)
    if not rows:
        return {
            "mean_semantic_completeness@5": 0.0,
            "source_entropy@5": 0.0,
            "stance_bucket_entropy@5": 0.0,
        }
    source_counts = Counter(source_group(row) for row in rows)
    stance_counts = Counter(str(row.get("stance_bucket_derived") or "") for row in rows)
    return {
        "mean_semantic_completeness@5": float(np.mean([_safe_float(row.get("semantic_completeness_score"), 0.0) for row in rows])),
        "source_entropy@5": normalized_entropy(list(source_counts.values())),
        "stance_bucket_entropy@5": normalized_entropy(list(stance_counts.values())),
    }


def summarize_selector_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace.get("selector_name") or "")].append(dict(trace))
    return {selector: _summarize_trace_group(rows) for selector, rows in sorted(grouped.items())}


def oracle_observation_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flat = [
        candidate
        for row in rows
        for candidate in (row.get("candidates") or [])
    ]
    selected = [candidate for candidate in flat if bool(candidate.get("oracle_selected"))]
    nonselected = [candidate for candidate in flat if not bool(candidate.get("oracle_selected"))]
    selected_completeness = [_safe_float(row.get("semantic_completeness_score"), 0.0) for row in selected]
    nonselected_completeness = [_safe_float(row.get("semantic_completeness_score"), 0.0) for row in nonselected]
    labels = [1 if bool(row.get("oracle_selected")) else 0 for row in flat]
    completeness_values = [_safe_float(row.get("semantic_completeness_score"), 0.0) for row in flat]
    alignment_values: list[float] = []
    selected_alignment: list[float] = []
    pool_alignment: list[float] = []
    for row in rows:
        target = gold_label_to_stance_bucket(str(row.get("gold_label") or ""), n_stance_buckets=int(row.get("n_stance_buckets") or 3))
        if not target:
            continue
        for candidate in row.get("candidates") or []:
            value = _safe_float((candidate.get("teacher_stance_probs") or {}).get(target), 0.0)
            alignment_values.append(value)
            pool_alignment.append(value)
            if bool(candidate.get("oracle_selected")):
                selected_alignment.append(value)
    return {
        "n_candidates": len(flat),
        "n_oracle_selected": len(selected),
        "mean_completeness_oracle_selected": _mean(selected_completeness),
        "mean_completeness_non_selected": _mean(nonselected_completeness),
        "completeness_selected_lift": _mean(selected_completeness) - _mean(nonselected_completeness),
        "completeness_selected_auroc": _roc_auc_score(labels, completeness_values),
        "oracle_selected_stance_label_alignment": _mean(selected_alignment),
        "pool_stance_label_alignment": _mean(pool_alignment),
        "oracle_vs_pool_stance_alignment_lift": _mean(selected_alignment) - _mean(pool_alignment),
        "stance_expected_score_mean": _mean(
            _safe_float(candidate.get("stance_expected_score"), 0.0)
            for candidate in flat
        ),
        "stance_soft_entropy_mean": _mean(
            _safe_float(candidate.get("stance_entropy"), 0.0)
            for candidate in flat
        ),
    }


def gold_label_to_stance_bucket(gold_label: str, *, n_stance_buckets: int) -> str:
    label = str(gold_label or "").strip().lower()
    names = _bucket_names_from_n(n_stance_buckets)
    if not names:
        return ""
    if label in {"pants-fire", "pants_fire", "false"}:
        return names[0]
    if label in {"barely-true", "barely_true", "half-true", "half_true"}:
        return names[len(names) // 2]
    if label in {"mostly-true", "mostly_true", "true"}:
        return names[-1]
    return ""


def oracle_ordered_keys(oracle_row: dict[str, Any]) -> list[str]:
    pool = list(oracle_row.get("candidate_pool") or [])
    keys: list[str] = []
    seen: set[str] = set()
    for raw_idx in oracle_row.get("selected_indices") or []:
        idx = _nullable_int(raw_idx)
        if idx is None or idx < 0 or idx >= len(pool):
            continue
        key = canonical_candidate_key(pool[idx])
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _normalize_original_candidate(
    oracle_row: dict[str, Any],
    raw_candidate: dict[str, Any],
    *,
    position: int,
    oracle_key_to_step: dict[str, int],
) -> dict[str, Any]:
    candidate = dict(raw_candidate)
    event_id = str(oracle_row.get("event_id") or "")
    claim = str(oracle_row.get("claim") or "")
    score_by_idx = _score_rows_by_candidate_idx(oracle_row.get("candidate_scores") or [])
    candidate_idx = _nullable_int(candidate.get("candidate_idx"))
    if candidate_idx is None:
        candidate_idx = int(position)
    score_row = dict(score_by_idx.get(candidate_idx) or score_by_idx.get(position) or {})
    key = canonical_candidate_key(candidate)
    dedup_key = dedup_key_for_candidate(event_id, candidate, source_pool=ORIGINAL_POOL)
    selected = key in oracle_key_to_step
    hybrid_rank = _nullable_int(score_row.get("hybrid_rank"))
    return {
        "event_id": event_id,
        "claim": claim,
        "gold_label": str(oracle_row.get("gold_label") or ""),
        "pool_position": int(position),
        "candidate_key": key,
        "canonical_text": key,
        "candidate_uid": stable_candidate_uid(event_id, dedup_key),
        "dedup_key": dedup_key,
        "source_pools": [ORIGINAL_POOL],
        "text": str(candidate.get("text") or ""),
        "candidate_idx": int(candidate_idx),
        "original_candidate_idx": int(candidate_idx),
        "original_pool_position": int(position),
        "source_index": _nullable_int(candidate.get("source_index", score_row.get("source_index"))),
        "report_id": _nullable_int(candidate.get("report_id")),
        "sent_idx": _nullable_int(candidate.get("sent_idx")),
        "chunk_sent_indices": candidate.get("chunk_sent_indices") or [],
        "source_report": candidate.get("source_report") if isinstance(candidate.get("source_report"), dict) else {},
        "source_domain": source_domain(candidate),
        "from_baseline": True,
        "from_qd": False,
        "from_original_stage2_pool": True,
        "baseline_rank": int(hybrid_rank + 1) if hybrid_rank is not None else int(position + 1),
        "baseline_hybrid_score": _nullable_float(score_row.get("hybrid_score", candidate.get("hybrid_score"))),
        "hybrid_rank": hybrid_rank,
        "hybrid_score": _nullable_float(score_row.get("hybrid_score", candidate.get("hybrid_score"))),
        "dense_score": _nullable_float(score_row.get("dense_score", candidate.get("dense_score"))),
        "lexical_score": _nullable_float(score_row.get("lexical_score", candidate.get("lexical_score"))),
        "bm25_score": _nullable_float(score_row.get("bm25_score", candidate.get("bm25_score"))),
        "qd_pool_rank": None,
        "qd_rrf_score": 0.0,
        "qd_question_hit_count": 0,
        "qd_max_question_hybrid": 0.0,
        "qd_question_routes": [],
        "oracle_selected": bool(selected),
        "oracle_step": int(oracle_key_to_step[key]) if selected else -1,
    }


def _normalize_qd_candidate(
    qd_row: dict[str, Any],
    raw_candidate: dict[str, Any],
    *,
    position: int,
    oracle_key_to_step: dict[str, int],
) -> dict[str, Any]:
    candidate = dict(raw_candidate)
    event_id = str(qd_row.get("event_id") or "")
    key = canonical_candidate_key(candidate)
    dedup_key = dedup_key_for_candidate(event_id, candidate, source_pool=QD_UNION_POOL)
    qd_routes = list(candidate.get("qd_question_routes") or candidate.get("question_routes") or [])
    selected = key in oracle_key_to_step
    qd_pool_rank = _nullable_int(candidate.get("qd_pool_rank", candidate.get("merge_rank")))
    return {
        "event_id": event_id,
        "claim": str(qd_row.get("claim") or ""),
        "gold_label": str(qd_row.get("gold_label") or ""),
        "pool_position": int(position),
        "candidate_key": key,
        "canonical_text": key,
        "candidate_uid": stable_candidate_uid(event_id, dedup_key),
        "dedup_key": dedup_key,
        "source_pools": [QD_UNION_POOL],
        "text": str(candidate.get("text") or ""),
        "candidate_idx": _nullable_int(candidate.get("candidate_idx")),
        "original_candidate_idx": _nullable_int(candidate.get("original_candidate_idx")),
        "qd_candidate_idx": int(position),
        "source_index": _nullable_int(candidate.get("source_index")),
        "report_id": _nullable_int(candidate.get("report_id")),
        "sent_idx": _nullable_int(candidate.get("sent_idx")),
        "chunk_sent_indices": candidate.get("chunk_sent_indices") or [],
        "source_report": candidate.get("source_report") if isinstance(candidate.get("source_report"), dict) else {},
        "source_domain": source_domain(candidate),
        "from_baseline": bool(candidate.get("from_baseline")),
        "from_qd": True,
        "from_original_stage2_pool": False,
        "baseline_rank": _nullable_int(candidate.get("baseline_rank")),
        "baseline_hybrid_score": _nullable_float(candidate.get("baseline_hybrid_score")),
        "hybrid_rank": _nullable_int(candidate.get("hybrid_rank", candidate.get("baseline_rank"))),
        "hybrid_score": _first_float(
            candidate.get("hybrid_score"),
            candidate.get("baseline_hybrid_score"),
            candidate.get("qd_max_question_hybrid"),
            candidate.get("max_question_hybrid"),
        ),
        "dense_score": _nullable_float(candidate.get("dense_score")),
        "lexical_score": _nullable_float(candidate.get("lexical_score")),
        "bm25_score": _nullable_float(candidate.get("bm25_score")),
        "qd_pool_rank": qd_pool_rank if qd_pool_rank is not None else int(position + 1),
        "qd_rrf_score": _safe_float(candidate.get("qd_rrf_score", candidate.get("rrf_score")), 0.0),
        "qd_question_hit_count": int(_safe_float(candidate.get("qd_question_hit_count", candidate.get("question_hit_count")), 0.0)),
        "qd_max_question_hybrid": _safe_float(candidate.get("qd_max_question_hybrid", candidate.get("max_question_hybrid")), 0.0),
        "qd_question_routes": qd_routes,
        "oracle_selected": bool(selected),
        "oracle_step": int(oracle_key_to_step[key]) if selected else -1,
    }


def _merge_candidate(
    merged: dict[str, dict[str, Any]],
    key_by_text: dict[str, str],
    candidate: dict[str, Any],
) -> None:
    dedup_key = str(candidate.get("dedup_key") or "")
    text_key = str(candidate.get("candidate_key") or "")
    existing_key = dedup_key if dedup_key in merged else key_by_text.get(text_key)
    if existing_key is None:
        merged[dedup_key] = dict(candidate)
        if text_key:
            key_by_text[text_key] = dedup_key
        return
    item = merged[existing_key]
    item["source_pools"] = _ordered_source_pools([*(item.get("source_pools") or []), *(candidate.get("source_pools") or [])])
    for flag in ("from_baseline", "from_qd", "from_original_stage2_pool", "oracle_selected"):
        item[flag] = bool(item.get(flag)) or bool(candidate.get(flag))
    if int(candidate.get("oracle_step", -1)) >= 0:
        current = int(item.get("oracle_step", -1))
        item["oracle_step"] = int(candidate["oracle_step"]) if current < 0 else min(current, int(candidate["oracle_step"]))
    _merge_min_int(item, candidate, "baseline_rank")
    _merge_min_int(item, candidate, "qd_pool_rank")
    _merge_min_int(item, candidate, "original_pool_position")
    _merge_min_int(item, candidate, "source_index")
    _merge_min_int(item, candidate, "original_candidate_idx")
    _merge_max_float(item, candidate, "baseline_hybrid_score")
    _merge_max_float(item, candidate, "hybrid_score")
    _merge_max_float(item, candidate, "qd_rrf_score")
    _merge_max_float(item, candidate, "qd_max_question_hybrid")
    routes = [*(item.get("qd_question_routes") or []), *(candidate.get("qd_question_routes") or [])]
    item["qd_question_routes"] = _dedupe_routes(routes)
    item["qd_question_hit_count"] = max(
        int(item.get("qd_question_hit_count") or 0),
        int(candidate.get("qd_question_hit_count") or 0),
        len({str(route.get("question_id") or "") for route in item["qd_question_routes"] if isinstance(route, dict)}),
    )
    for key in (
        "text",
        "canonical_text",
        "candidate_key",
        "source_report",
        "report_id",
        "sent_idx",
        "chunk_sent_indices",
        "source_domain",
    ):
        if not item.get(key) and candidate.get(key):
            item[key] = candidate[key]


def _best_candidate_in_bucket(
    candidates: Sequence[dict[str, Any]],
    *,
    bucket: str,
    selected_keys: set[str],
    selected_source_groups: set[str],
    min_membership: float,
    params: CountAmplifiedParams,
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        key = str(candidate.get("candidate_key") or "")
        if not key or key in selected_keys:
            continue
        if not quality_gate(candidate, tau_c=params.tau_c, tau_r=params.tau_r):
            continue
        probs = dict(candidate.get("teacher_stance_probs") or {})
        if _safe_float(probs.get(bucket), 0.0) < float(min_membership):
            continue
        item = dict(candidate)
        item["candidate_score"] = candidate_score_in_bucket(
            item,
            bucket=bucket,
            selected_source_groups=selected_source_groups,
        )
        eligible.append(item)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row.get("candidate_score") or 0.0),
            float(row.get("semantic_completeness_score") or 0.0),
            -int(row.get("union_pool_rank") or 10**9),
            str(row.get("candidate_key") or ""),
        ),
    )


def _has_eligible_remaining(
    candidates: Sequence[dict[str, Any]],
    *,
    bucket: str,
    selected_keys: set[str],
    min_membership: float,
    params: CountAmplifiedParams,
) -> bool:
    for candidate in candidates:
        key = str(candidate.get("candidate_key") or "")
        if not key or key in selected_keys:
            continue
        if not quality_gate(candidate, tau_c=params.tau_c, tau_r=params.tau_r):
            continue
        if _safe_float((candidate.get("teacher_stance_probs") or {}).get(bucket), 0.0) >= float(min_membership):
            return True
    return False


def _ranked(candidates: Sequence[dict[str, Any]], *, top_k: int, selector_name: str) -> list[dict[str, Any]]:
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


def _source_score(candidate: dict[str, Any]) -> float:
    baseline = 0.0
    if candidate.get("from_baseline"):
        baseline += 0.04
        baseline_rank = _nullable_int(candidate.get("baseline_rank"))
        if baseline_rank is not None:
            baseline += 0.01 / max(float(baseline_rank), 1.0)
    qd = _safe_float(candidate.get("qd_rrf_score"), 0.0)
    qd += 0.004 * _safe_float(candidate.get("qd_question_hit_count"), 0.0)
    qd += 0.01 * _safe_float(candidate.get("qd_max_question_hybrid"), 0.0)
    return float(baseline + qd)


def _candidate_output(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_uid",
        "candidate_key",
        "selection_rank",
        "union_pool_rank",
        "source_pools",
        "union_source",
        "from_baseline",
        "from_qd",
        "from_original_stage2_pool",
        "baseline_rank",
        "qd_pool_rank",
        "retrieval_score",
        "semantic_completeness_score",
        "relevance_gate_score",
        "stance_bucket_derived",
        "stance_strength",
        "stance_entropy",
        "source_group",
        "oracle_selected",
        "oracle_step",
        "selected_stance_bucket",
        "candidate_score",
        "text",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def _summarize_trace_group(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "recall@5",
        "precision@5",
        "jaccard@5",
        "top1_match",
        "oracle_rank_ndcg@5",
        "pairwise_order_acc@5",
        "mean_semantic_completeness@5",
        "source_entropy@5",
        "stance_bucket_entropy@5",
    ]
    out: dict[str, Any] = {"n_claims": len(traces)}
    weights = np.asarray([float(trace.get("overlap_pair_count", 0)) for trace in traces], dtype=np.float64)
    for key in metric_keys:
        values = np.asarray([float(trace.get(key, 0.0)) for trace in traces], dtype=np.float64)
        if key == "pairwise_order_acc@5" and weights.sum() > 0:
            out[key] = float((values * weights).sum() / weights.sum())
        else:
            out[key] = float(values.mean()) if values.size else 0.0
    return out


def _pairwise_order_acc(oracle: Sequence[str], pred: Sequence[str]) -> tuple[float, int]:
    oracle_pos = {key: pos for pos, key in enumerate(oracle)}
    pred_pos = {key: pos for pos, key in enumerate(pred)}
    overlap = [key for key in oracle if key in pred_pos]
    if len(overlap) < 2:
        return 0.0, 0
    total = 0
    correct = 0
    for i, left in enumerate(overlap):
        for right in overlap[i + 1 :]:
            total += 1
            if (oracle_pos[left] < oracle_pos[right]) == (pred_pos[left] < pred_pos[right]):
                correct += 1
    return float(correct / max(total, 1)), int(total)


def _oracle_rank_ndcg(oracle: Sequence[str], pred: Sequence[str], *, top_k: int) -> float:
    rel_by_key = {key: max(int(top_k) - rank, 1) for rank, key in enumerate(oracle[:top_k])}
    dcg = 0.0
    for rank, key in enumerate(pred[:top_k], start=1):
        rel = rel_by_key.get(key, 0)
        dcg += (2.0**rel - 1.0) / math.log2(rank + 1)
    ideal = sorted(rel_by_key.values(), reverse=True)[:top_k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _roc_auc_score(labels: Sequence[int], values: Sequence[float]) -> float:
    pairs = [(float(value), int(label)) for label, value in zip(labels, values)]
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = sum(1 for _, label in pairs if label == 0)
    if positives == 0 or negatives == 0:
        return 0.0
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in pairs[idx:end] if label == 1)
        idx = end
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _bucket_names_from_candidates(candidates: Sequence[dict[str, Any]]) -> list[str]:
    for candidate in candidates:
        probs = candidate.get("teacher_stance_probs")
        if isinstance(probs, dict) and probs:
            return list(probs.keys())
    return []


def _bucket_names_from_n(n_stance_buckets: int) -> list[str]:
    try:
        from fact_checking.selectors.stance_buckets import bucket_names

        return bucket_names(int(n_stance_buckets))
    except Exception:
        return []


def _score_rows_by_candidate_idx(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        idx = _nullable_int(row.get("candidate_idx"))
        out[int(position if idx is None else idx)] = dict(row)
    return out


def _union_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate.get("from_baseline") else 1,
        int(candidate.get("baseline_rank") or 10**9),
        int(candidate.get("qd_pool_rank") or 10**9),
        int(candidate.get("union_pool_rank") or 10**9),
        str(candidate.get("candidate_key") or ""),
    )


def _union_source(candidate: dict[str, Any]) -> str:
    if candidate.get("from_baseline") and candidate.get("from_qd"):
        return "baseline+qd"
    if candidate.get("from_baseline"):
        return "baseline"
    if candidate.get("from_qd"):
        return "qd"
    return "unknown"


def _is_original_stage2_candidate(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("from_original_stage2_pool")) or ORIGINAL_POOL in set(candidate.get("source_pools") or [])


def _ordered_source_pools(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    for pool in (ORIGINAL_POOL, QD_UNION_POOL):
        if pool in values:
            ordered.append(pool)
    for value in values:
        if value not in ordered:
            ordered.append(str(value))
    return ordered


def _dedupe_routes(routes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for route in routes:
        if not isinstance(route, dict):
            continue
        key = (str(route.get("question_id") or ""), int(_safe_float(route.get("rank"), -1.0)))
        if key not in seen:
            out.append(dict(route))
            seen.add(key)
    return out


def _merge_min_int(item: dict[str, Any], candidate: dict[str, Any], key: str) -> None:
    left = _nullable_int(item.get(key))
    right = _nullable_int(candidate.get(key))
    if right is None:
        return
    item[key] = right if left is None else min(left, right)


def _merge_max_float(item: dict[str, Any], candidate: dict[str, Any], key: str) -> None:
    left = _nullable_float(item.get(key))
    right = _nullable_float(candidate.get(key))
    if right is None:
        return
    item[key] = right if left is None else max(left, right)


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _nullable_float(value)
        if parsed is not None:
            return parsed
    return None


def _nullable_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _nullable_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _safe_float(value: Any, default: float) -> float:
    parsed = _nullable_float(value)
    return float(default) if parsed is None else float(parsed)


def _mean(values: Sequence[float] | Any) -> float:
    vals = [float(value) for value in values if value is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0
