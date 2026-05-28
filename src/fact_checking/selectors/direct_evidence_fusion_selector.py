from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    COUNT_AMPLIFIED_SELECTOR,
    CountAmplifiedParams,
    select_count_amplified_topk,
    select_order_control,
    selection_quality_metrics,
    summarize_selector_traces,
    text_ordered_selection_metrics,
)
from fact_checking.selectors.direct_evidence_cross_encoder import (
    DIRECT_CE_TEXT_ONLY_SELECTOR,
    candidate_level_metrics as direct_ce_candidate_level_metrics,
    score_sanity_summary,
    select_direct_ce_topk,
)
from fact_checking.selectors.evidence_quality import retrieval_score, source_group
from fact_checking.selectors.oracle_likelihood_constrained_selector import (
    FORBIDDEN_FEATURE_FIELDS,
    ORACLE_LIKELIHOOD_SELECTOR,
    LogisticParams,
    average_precision_score,
    binary_log_loss,
    build_feature_rows,
    calibration_bins,
    cross_fit_score_rows,
    feature_importance_rows,
    labels_array,
    roc_auc_score,
    select_likelihood_topk,
    stance_region,
)


FUSION_REFIT_SELECTOR = "fusion_refit_all_features_plus_direct_ce_top5"
FUSION_Z_PREFIX = "fusion_z_lambda"
FUSION_RANK_PREFIX = "fusion_rank_lambda"
DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
DIRECT_CE_FUSION_FEATURES = (
    "direct_ce_score",
    "direct_ce_raw_score",
    "direct_ce_event_z",
    "direct_ce_rank_recip",
)


@dataclass(frozen=True)
class FusionDecision:
    decision: str
    best_selector: str
    best_jaccard: float
    best_recall: float
    baseline_jaccard: float
    baseline_recall: float
    delta_jaccard: float
    delta_recall: float
    best_lambda: float | None
    best_family: str
    refit_direct_ce_mean_abs_weight: float
    passes_go_gate: bool


def merge_oracle_and_direct_ce_rows(
    oracle_rows: Sequence[dict[str, Any]],
    direct_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    oracle_by_event = _rows_by_event(oracle_rows, label="oracle")
    direct_by_event = _rows_by_event(direct_rows, label="direct_ce")
    if set(oracle_by_event) != set(direct_by_event):
        missing_direct = sorted(set(oracle_by_event) - set(direct_by_event))[:10]
        missing_oracle = sorted(set(direct_by_event) - set(oracle_by_event))[:10]
        raise ValueError(f"Event set mismatch: missing_direct={missing_direct}, missing_oracle={missing_oracle}")
    out: list[dict[str, Any]] = []
    for event_id in sorted(oracle_by_event):
        oracle_row = oracle_by_event[event_id]
        direct_row = direct_by_event[event_id]
        oracle_candidates = _candidates_by_key(oracle_row, label="oracle")
        direct_candidates = _candidates_by_key(direct_row, label="direct_ce")
        if set(oracle_candidates) != set(direct_candidates):
            missing_direct = sorted(set(oracle_candidates) - set(direct_candidates))[:10]
            missing_oracle = sorted(set(direct_candidates) - set(oracle_candidates))[:10]
            raise ValueError(
                f"Candidate set mismatch for event_id={event_id}: "
                f"missing_direct={missing_direct}, missing_oracle={missing_oracle}"
            )
        item = dict(oracle_row)
        merged_candidates: list[dict[str, Any]] = []
        for candidate in oracle_row.get("candidates") or []:
            key = _candidate_key(event_id, candidate)
            direct = direct_candidates[key]
            merged = dict(candidate)
            for field in (
                "direct_ce_raw_score",
                "direct_ce_score",
                "direct_ce_model",
                "direct_ce_prompt_version",
                "direct_ce_prompt_mode",
                "direct_ce_score_normalization",
                "direct_ce_score_source",
            ):
                if field not in direct:
                    raise ValueError(f"Missing {field} in direct CE candidate key={key}")
                merged[field] = direct[field]
            merged_candidates.append(merged)
        item["candidates"] = merged_candidates
        out.append(item)
    add_fusion_candidate_features(out, lambdas=DEFAULT_LAMBDAS)
    return out


def add_fusion_candidate_features(rows: Sequence[dict[str, Any]], *, lambdas: Sequence[float]) -> None:
    all_oracle_logits = [
        _safe_float(candidate.get("oracle_likelihood_logit"), _logit(_safe_float(candidate.get("oracle_likelihood_score"), 0.0)))
        for row in rows
        for candidate in row.get("candidates") or []
    ]
    all_direct_raw = [
        _safe_float(candidate.get("direct_ce_raw_score"), 0.0)
        for row in rows
        for candidate in row.get("candidates") or []
    ]
    oracle_mean, oracle_std = _mean_std(all_oracle_logits)
    direct_mean, direct_std = _mean_std(all_direct_raw)
    for row in rows:
        candidates = [candidate for candidate in row.get("candidates") or []]
        direct_values = [_safe_float(candidate.get("direct_ce_raw_score"), 0.0) for candidate in candidates]
        direct_event_mean, direct_event_std = _mean_std(direct_values)
        oracle_rank_order = sorted(candidates, key=_oracle_rank_sort_key)
        direct_rank_order = sorted(candidates, key=_direct_rank_sort_key)
        oracle_rank_recip = {_selection_key(candidate): 1.0 / float(idx + 1) for idx, candidate in enumerate(oracle_rank_order)}
        direct_rank_recip = {_selection_key(candidate): 1.0 / float(idx + 1) for idx, candidate in enumerate(direct_rank_order)}
        for candidate in candidates:
            oracle_logit = _safe_float(
                candidate.get("oracle_likelihood_logit"),
                _logit(_safe_float(candidate.get("oracle_likelihood_score"), 0.0)),
            )
            direct_raw = _safe_float(candidate.get("direct_ce_raw_score"), 0.0)
            candidate["oracle_likelihood_logit_z"] = (oracle_logit - oracle_mean) / oracle_std
            candidate["direct_ce_raw_z"] = (direct_raw - direct_mean) / direct_std
            candidate["direct_ce_event_z"] = (direct_raw - direct_event_mean) / direct_event_std
            key = _selection_key(candidate)
            candidate["oracle_likelihood_rank_recip"] = float(oracle_rank_recip.get(key, 0.0))
            candidate["direct_ce_rank_recip"] = float(direct_rank_recip.get(key, 0.0))
            for lam in lambdas:
                tag = lambda_tag(lam)
                z_score = (1.0 - float(lam)) * float(candidate["oracle_likelihood_logit_z"]) + float(lam) * float(candidate["direct_ce_raw_z"])
                rank_score = (1.0 - float(lam)) * float(candidate["oracle_likelihood_rank_recip"]) + float(lam) * float(candidate["direct_ce_rank_recip"])
                candidate[f"{FUSION_Z_PREFIX}_{tag}_score"] = float(z_score)
                candidate[f"{FUSION_RANK_PREFIX}_{tag}_score"] = float(rank_score)


def build_fusion_feature_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    base_rows, base_feature_names = build_feature_rows(rows)
    direct_by_key = {
        _candidate_key(str(row.get("event_id") or ""), candidate): candidate
        for row in rows
        for candidate in row.get("candidates") or []
    }
    feature_names = list(base_feature_names) + list(DIRECT_CE_FUSION_FEATURES)
    forbidden = FORBIDDEN_FEATURE_FIELDS & set(feature_names)
    if forbidden:
        raise ValueError(f"Forbidden fusion features: {sorted(forbidden)}")
    out: list[dict[str, Any]] = []
    for row in base_rows:
        item = dict(row)
        features = dict(row.get("features") or {})
        candidate = direct_by_key.get(_row_key(row))
        if candidate is None:
            raise KeyError(f"Missing direct CE candidate features for key={_row_key(row)}")
        for name in DIRECT_CE_FUSION_FEATURES:
            features[name] = _safe_float(candidate.get(name), 0.0)
        item["features"] = features
        out.append(item)
    return out, feature_names


def attach_refit_scores(
    rows: Sequence[dict[str, Any]],
    scored_feature_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored_by_key = {_row_key(row): row for row in scored_feature_rows}
    out: list[dict[str, Any]] = []
    for event_row in rows:
        item = dict(event_row)
        candidates: list[dict[str, Any]] = []
        for candidate in event_row.get("candidates") or []:
            c = dict(candidate)
            key = _candidate_key(str(event_row.get("event_id") or ""), c)
            scored = scored_by_key.get(key)
            if scored is None:
                raise KeyError(f"Missing fusion refit score for key={key}")
            c["fusion_refit_score"] = _safe_float(scored.get("oracle_likelihood_score"), 0.0)
            c["fusion_refit_logit"] = _safe_float(scored.get("oracle_likelihood_logit"), 0.0)
            c["fusion_refit_fold"] = int(_safe_float(scored.get("oracle_likelihood_fold"), 0.0))
            candidates.append(c)
        item["candidates"] = candidates
        out.append(item)
    return out


def run_refit_fusion(
    rows: Sequence[dict[str, Any]],
    *,
    folds: int,
    params: LogisticParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    feature_rows, feature_names = build_fusion_feature_rows(rows)
    scored_feature_rows, models, fold_records = cross_fit_score_rows(
        feature_rows,
        feature_names,
        folds=int(folds),
        params=params,
        objective="pointwise",
    )
    scored_rows = attach_refit_scores(rows, scored_feature_rows)
    importance = feature_importance_rows(models, feature_names)
    return scored_rows, models, fold_records, feature_names, importance


def build_all_fusion_traces(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    lambdas: Sequence[float],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        traces.extend(_control_traces(row, top_k=top_k))
        traces.append(
            build_fusion_trace(
                row,
                select_likelihood_topk(candidates, top_k=top_k, selector_name=ORACLE_LIKELIHOOD_SELECTOR),
                selector_name=ORACLE_LIKELIHOOD_SELECTOR,
                top_k=top_k,
            )
        )
        traces.append(
            build_fusion_trace(
                row,
                select_direct_ce_topk(candidates, top_k=top_k, selector_name=DIRECT_CE_TEXT_ONLY_SELECTOR),
                selector_name=DIRECT_CE_TEXT_ONLY_SELECTOR,
                top_k=top_k,
            )
        )
        for lam in lambdas:
            tag = lambda_tag(lam)
            for family, score_field in (
                (FUSION_Z_PREFIX, f"{FUSION_Z_PREFIX}_{tag}_score"),
                (FUSION_RANK_PREFIX, f"{FUSION_RANK_PREFIX}_{tag}_score"),
            ):
                selector_name = f"{family}_{tag}_top5"
                if abs(float(lam)) < 1e-12:
                    selected = _metadata_for_fusion_selection(
                        select_likelihood_topk(candidates, top_k=top_k, selector_name=selector_name),
                        score_field=score_field,
                        selector_name=selector_name,
                        origin=selector_name.replace("_top5", ""),
                        lambda_value=float(lam),
                    )
                else:
                    selected = select_fusion_topk(
                        candidates,
                        score_field=score_field,
                        top_k=top_k,
                        selector_name=selector_name,
                        origin=selector_name.replace("_top5", ""),
                        lambda_value=float(lam),
                    )
                traces.append(
                    build_fusion_trace(
                        row,
                        selected,
                        selector_name=selector_name,
                        top_k=top_k,
                    )
                )
        traces.append(
            build_fusion_trace(
                row,
                select_fusion_topk(
                    candidates,
                    score_field="fusion_refit_score",
                    top_k=top_k,
                    selector_name=FUSION_REFIT_SELECTOR,
                    origin="fusion_refit",
                    lambda_value=None,
                ),
                selector_name=FUSION_REFIT_SELECTOR,
                top_k=top_k,
            )
        )
    return traces


def select_fusion_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    score_field: str,
    top_k: int,
    selector_name: str,
    origin: str,
    lambda_value: float | None,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    rows.sort(key=lambda candidate: _fusion_sort_key(candidate, score_field))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _selection_key(row)
        if not key or key in seen:
            continue
        item = dict(row)
        item["selector_name"] = selector_name
        item["selection_origin"] = origin
        item["selection_rank"] = len(selected) + 1
        item["fusion_score"] = _safe_float(item.get(score_field), 0.0)
        item["fusion_score_field"] = score_field
        if lambda_value is not None:
            item["fusion_lambda"] = float(lambda_value)
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def _metadata_for_fusion_selection(
    selected: Sequence[dict[str, Any]],
    *,
    score_field: str,
    selector_name: str,
    origin: str,
    lambda_value: float,
) -> list[dict[str, Any]]:
    out = []
    for idx, candidate in enumerate(selected):
        item = dict(candidate)
        item["selector_name"] = selector_name
        item["selection_origin"] = origin
        item["selection_rank"] = int(idx + 1)
        item["fusion_score"] = _safe_float(item.get(score_field), 0.0)
        item["fusion_score_field"] = score_field
        item["fusion_lambda"] = float(lambda_value)
        out.append(item)
    return out


def build_fusion_trace(
    row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    selector_name: str,
    top_k: int,
) -> dict[str, Any]:
    selected = list(selected)
    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "selector_name": str(selector_name),
        "oracle_ordered_keys": list(row.get("oracle_ordered_keys") or []),
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected],
        "selected_candidates": [_candidate_output(candidate) for candidate in selected],
        "slot_trace": [
            {
                "slot": int(candidate.get("selection_rank") or idx + 1),
                "candidate_uid": str(candidate.get("candidate_uid") or ""),
                "candidate_key": str(candidate.get("candidate_key") or ""),
                "selection_origin": str(candidate.get("selection_origin") or ""),
                "oracle_selected": bool(candidate.get("oracle_selected")),
                "fusion_score": _safe_float(candidate.get("fusion_score"), 0.0),
                "oracle_likelihood_score": _safe_float(candidate.get("oracle_likelihood_score"), 0.0),
                "direct_ce_score": _safe_float(candidate.get("direct_ce_score"), 0.0),
                "source_group": source_group(candidate),
                "stance_region": stance_region(candidate),
            }
            for idx, candidate in enumerate(selected)
        ],
    }
    trace.update(text_ordered_selection_metrics(trace["oracle_ordered_keys"], selected, top_k=top_k))
    trace.update(selection_quality_metrics(selected))
    trace["mean_fusion_score@5"] = _mean(_safe_float(candidate.get("fusion_score"), 0.0) for candidate in selected)
    trace["mean_oracle_likelihood_score@5"] = _mean(_safe_float(candidate.get("oracle_likelihood_score"), 0.0) for candidate in selected)
    trace["mean_direct_ce_score@5"] = _mean(_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in selected)
    return trace


def summarize_fusion_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_selector_traces(traces)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace.get("selector_name") or "")].append(dict(trace))
    for selector, rows in grouped.items():
        selected = [candidate for trace in rows for candidate in trace.get("selected_candidates") or []]
        item = summary.setdefault(selector, {"n_claims": len(rows)})
        item["mean_fusion_score@5"] = _mean(_safe_float(candidate.get("fusion_score"), 0.0) for candidate in selected)
        item["mean_oracle_likelihood_score@5"] = _mean(_safe_float(candidate.get("oracle_likelihood_score"), 0.0) for candidate in selected)
        item["mean_direct_ce_score@5"] = _mean(_safe_float(candidate.get("direct_ce_score"), 0.0) for candidate in selected)
        item["hit_rate@5"] = _mean(1.0 if bool(candidate.get("oracle_selected")) else 0.0 for candidate in selected)
    return summary


def fusion_diagnostics(
    rows: Sequence[dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    selector_metrics: dict[str, Any],
    *,
    feature_importance: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    flat = [dict(candidate) for row in rows for candidate in row.get("candidates") or []]
    metrics_by_score = {
        "oracle_likelihood_score": candidate_metrics_for_field(flat, "oracle_likelihood_score"),
        "direct_ce_score": candidate_metrics_for_field(flat, "direct_ce_score"),
        "fusion_refit_score": candidate_metrics_for_field(flat, "fusion_refit_score"),
    }
    best_selector, best_metrics = best_fusion_selector(selector_metrics)
    best_score_field = _selector_score_field(best_selector)
    if best_score_field:
        metrics_by_score["best_fusion_score"] = candidate_metrics_for_field(flat, best_score_field)
    increments = event_level_increments(traces, baseline_selector=ORACLE_LIKELIHOOD_SELECTOR)
    direct_weights = [
        dict(row)
        for row in feature_importance
        if str(row.get("feature") or "") in DIRECT_CE_FUSION_FEATURES
    ]
    return {
        "candidate_metrics": metrics_by_score,
        "best_fusion_selector": best_selector,
        "best_fusion_metrics": best_metrics,
        "event_level_increment_vs_v0_3_1": increments,
        "direct_ce_feature_importance": direct_weights,
        "direct_ce_score_sanity": score_sanity_summary(rows),
    }


def decide_fusion(
    selector_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> FusionDecision:
    baseline = selector_metrics.get(ORACLE_LIKELIHOOD_SELECTOR, {})
    best_selector, best_metrics = best_fusion_selector(selector_metrics)
    baseline_jaccard = _safe_float(baseline.get("jaccard@5"), 0.0)
    baseline_recall = _safe_float(baseline.get("recall@5"), 0.0)
    best_jaccard = _safe_float(best_metrics.get("jaccard@5"), 0.0)
    best_recall = _safe_float(best_metrics.get("recall@5"), 0.0)
    delta_jaccard = best_jaccard - baseline_jaccard
    delta_recall = best_recall - baseline_recall
    best_lambda = selector_lambda(best_selector)
    best_family = selector_family(best_selector)
    direct_weights = diagnostics.get("direct_ce_feature_importance") or []
    direct_weight = max((_safe_float(row.get("mean_abs_weight"), 0.0) for row in direct_weights), default=0.0)
    go = bool(delta_jaccard >= 0.005 and delta_recall >= -0.001 and best_selector != ORACLE_LIKELIHOOD_SELECTOR)
    if go:
        decision = "go_direct_ce_light_fusion_v0_4d"
    elif (best_lambda is not None and abs(best_lambda) < 1e-12) or delta_jaccard < 0.002:
        decision = "no_go_direct_ce_no_increment_v0_4d"
    elif best_family == "refit" and direct_weight < 1e-3:
        decision = "no_go_refit_ignored_direct_ce_v0_4d"
    else:
        decision = "analysis_small_direct_ce_increment_v0_4d"
    return FusionDecision(
        decision=decision,
        best_selector=best_selector,
        best_jaccard=best_jaccard,
        best_recall=best_recall,
        baseline_jaccard=baseline_jaccard,
        baseline_recall=baseline_recall,
        delta_jaccard=delta_jaccard,
        delta_recall=delta_recall,
        best_lambda=best_lambda,
        best_family=best_family,
        refit_direct_ce_mean_abs_weight=direct_weight,
        passes_go_gate=go,
    )


def candidate_metrics_for_field(candidates: Sequence[dict[str, Any]], score_field: str) -> dict[str, Any]:
    labels = np.asarray([1.0 if candidate.get("oracle_selected") else 0.0 for candidate in candidates], dtype=np.float32)
    raw_scores = np.asarray([_safe_float(candidate.get(score_field), 0.0) for candidate in candidates], dtype=np.float32)
    probs, transform = _probability_scores(raw_scores)
    return {
        "score_field": score_field,
        "probability_transform": transform,
        "n_rows": int(len(candidates)),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "auroc": roc_auc_score(labels, raw_scores),
        "auprc": average_precision_score(labels, raw_scores),
        "brier": float(np.mean((probs - labels) ** 2)) if labels.size else 0.0,
        "log_loss": binary_log_loss(labels, probs),
        "calibration_bins": calibration_bins(labels, probs, n_bins=10),
    }


def event_level_increments(
    traces: Sequence[dict[str, Any]],
    *,
    baseline_selector: str,
) -> dict[str, Any]:
    by_event: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for trace in traces:
        by_event[str(trace.get("event_id") or "")][str(trace.get("selector_name") or "")] = dict(trace)
    rows: dict[str, dict[str, Any]] = {}
    for selector in sorted({str(trace.get("selector_name") or "") for trace in traces}):
        if selector == baseline_selector:
            continue
        win = loss = tie = 0
        deltas: list[float] = []
        for per_event in by_event.values():
            if baseline_selector not in per_event or selector not in per_event:
                continue
            delta = _safe_float(per_event[selector].get("jaccard@5"), 0.0) - _safe_float(per_event[baseline_selector].get("jaccard@5"), 0.0)
            deltas.append(delta)
            if delta > 0:
                win += 1
            elif delta < 0:
                loss += 1
            else:
                tie += 1
        rows[selector] = {
            "win": int(win),
            "loss": int(loss),
            "tie": int(tie),
            "mean_delta_jaccard@5": _mean(deltas),
        }
    return rows


def best_fusion_selector(selector_metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = {
        selector: metrics
        for selector, metrics in selector_metrics.items()
        if selector.startswith(FUSION_Z_PREFIX) or selector.startswith(FUSION_RANK_PREFIX) or selector == FUSION_REFIT_SELECTOR
    }
    if not candidates:
        return ORACLE_LIKELIHOOD_SELECTOR, selector_metrics.get(ORACLE_LIKELIHOOD_SELECTOR, {})
    selector, metrics = max(
        candidates.items(),
        key=lambda item: (
            _safe_float(item[1].get("jaccard@5"), 0.0),
            _safe_float(item[1].get("recall@5"), 0.0),
            _safe_float(item[1].get("oracle_rank_ndcg@5"), 0.0),
            item[0],
        ),
    )
    return selector, dict(metrics)


def lambda_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p") if text else "0"


def parse_lambdas(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
        return [float(part) for part in raw]
    return [float(item) for item in value]


def selector_lambda(selector_name: str) -> float | None:
    parts = str(selector_name).split("_")
    if "lambda" not in parts:
        return None
    idx = parts.index("lambda")
    if idx + 1 >= len(parts):
        return None
    token = parts[idx + 1]
    try:
        return float(token.replace("m", "-").replace("p", "."))
    except ValueError:
        return None


def selector_family(selector_name: str) -> str:
    if selector_name.startswith(FUSION_Z_PREFIX):
        return "z"
    if selector_name.startswith(FUSION_RANK_PREFIX):
        return "rank"
    if selector_name == FUSION_REFIT_SELECTOR:
        return "refit"
    return "other"


def validate_lambda_zero_reproduces_baseline(
    traces: Sequence[dict[str, Any]],
) -> None:
    by_event: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for trace in traces:
        by_event[str(trace.get("event_id") or "")][str(trace.get("selector_name") or "")] = dict(trace)
    for selector in (f"{FUSION_Z_PREFIX}_0_top5", f"{FUSION_RANK_PREFIX}_0_top5"):
        mismatches = []
        for event_id, per_event in by_event.items():
            baseline = per_event.get(ORACLE_LIKELIHOOD_SELECTOR)
            fusion = per_event.get(selector)
            if not baseline or not fusion:
                continue
            if list(baseline.get("selected_keys") or []) != list(fusion.get("selected_keys") or []):
                mismatches.append(event_id)
        if mismatches:
            raise AssertionError(f"{selector} did not reproduce {ORACLE_LIKELIHOOD_SELECTOR}; examples={mismatches[:10]}")


def _control_traces(row: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    traces = [
        build_fusion_trace(
            row,
            select_order_control(candidates, mode=name, top_k=top_k),
            selector_name=name,
            top_k=top_k,
        )
        for name in ("original_pool_order_top5", "qd_union_pool_order_top5", "qd_union_source_score_top5")
    ]
    count_selected, _, _ = select_count_amplified_topk(
        candidates,
        params=CountAmplifiedParams(
            top_k=top_k,
            alpha=0.5,
            gamma_stance=0.8,
            rho=2.0,
            ambiguous_bucket_penalty=0.6,
            use_directness_scoring=True,
            adaptive_polar_quota=True,
            tau_polar_ready=0.8,
            max_forced_polar_slots=2,
            tau_c=0.50,
            tau_r=0.15,
            n_stance_buckets=int(row.get("n_stance_buckets") or _infer_n_buckets(candidates) or 7),
        ),
        selector_name=COUNT_AMPLIFIED_SELECTOR,
    )
    traces.append(
        build_fusion_trace(
            row,
            count_selected,
            selector_name=COUNT_AMPLIFIED_SELECTOR,
            top_k=top_k,
        )
    )
    return traces


def _selector_score_field(selector_name: str) -> str:
    if selector_name == FUSION_REFIT_SELECTOR:
        return "fusion_refit_score"
    if selector_name.startswith(FUSION_Z_PREFIX) or selector_name.startswith(FUSION_RANK_PREFIX):
        return f"{selector_name.removesuffix('_top5')}_score"
    return ""


def _probability_scores(scores: np.ndarray) -> tuple[np.ndarray, str]:
    if scores.size and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
        clipped = np.clip(scores.astype(np.float64), -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32), "sigmoid"
    return scores.astype(np.float32), "identity"


def _fusion_sort_key(candidate: dict[str, Any], score_field: str) -> tuple[float, float, float, int, str]:
    return (
        -_safe_float(candidate.get(score_field), 0.0),
        -_safe_float(candidate.get("oracle_likelihood_score"), 0.0),
        -_safe_float(candidate.get("direct_ce_raw_score"), 0.0),
        int(_safe_float(candidate.get("union_pool_rank"), 10**9)),
        str(candidate.get("candidate_key") or ""),
    )


def _oracle_rank_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        -_safe_float(candidate.get("oracle_likelihood_score"), 0.0),
        -retrieval_score(candidate),
        -_safe_float(candidate.get("semantic_completeness_score"), 0.0),
        int(_safe_float(candidate.get("union_pool_rank"), 10**9)),
        str(candidate.get("candidate_key") or ""),
    )


def _direct_rank_sort_key(candidate: dict[str, Any]) -> tuple[float, str]:
    return (-_safe_float(candidate.get("direct_ce_score"), 0.0), str(candidate.get("candidate_key") or ""))


def _candidate_output(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_uid",
        "candidate_key",
        "selection_rank",
        "selection_origin",
        "union_pool_rank",
        "source_pools",
        "from_baseline",
        "from_qd",
        "baseline_rank",
        "qd_pool_rank",
        "retrieval_score",
        "semantic_completeness_score",
        "stance_bucket_derived",
        "stance_region",
        "source_group",
        "oracle_selected",
        "oracle_step",
        "oracle_likelihood_score",
        "oracle_likelihood_logit",
        "direct_ce_raw_score",
        "direct_ce_score",
        "direct_ce_event_z",
        "direct_ce_rank_recip",
        "fusion_score",
        "fusion_score_field",
        "fusion_lambda",
        "fusion_refit_score",
        "fusion_refit_fold",
        "direct_evidence_score",
        "claim_specificity_score",
        "background_only_score",
        "key_fact_overlap_score",
        "evidence_role",
    ]
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["source_group"] = source_group(candidate)
    out["stance_region"] = stance_region(candidate)
    return out


def _rows_by_event(rows: Sequence[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            raise ValueError(f"Missing event_id in {label} rows")
        if event_id in out:
            raise ValueError(f"Duplicate event_id={event_id} in {label} rows")
        out[event_id] = dict(row)
    return out


def _candidates_by_key(row: dict[str, Any], *, label: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    event_id = str(row.get("event_id") or "")
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in row.get("candidates") or []:
        key = _candidate_key(event_id, candidate)
        if key in out:
            raise ValueError(f"Duplicate candidate key={key} in {label} rows")
        out[key] = dict(candidate)
    return out


def _candidate_key(event_id: str, candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (str(event_id), str(candidate.get("candidate_uid") or ""), str(candidate.get("candidate_key") or ""))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("event_id") or ""), str(row.get("candidate_uid") or ""), str(row.get("candidate_key") or ""))


def _selection_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_key") or candidate.get("candidate_uid") or "")


def _infer_n_buckets(candidates: Sequence[dict[str, Any]]) -> int:
    for candidate in candidates:
        probs = candidate.get("teacher_stance_probs")
        if isinstance(probs, dict) and probs:
            return len(probs)
    return 0


def _mean(values: Sequence[float] | Any) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else 0.0


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if vals.size == 0:
        return 0.0, 1.0
    std = float(vals.std())
    return float(vals.mean()), std if std >= 1e-8 else 1.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _logit(score: float) -> float:
    clipped = min(max(float(score), 1e-6), 1.0 - 1e-6)
    return float(math.log(clipped / (1.0 - clipped)))
