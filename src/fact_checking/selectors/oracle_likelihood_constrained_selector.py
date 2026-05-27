from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    selection_quality_metrics,
    summarize_selector_traces,
    text_ordered_selection_metrics,
)
from fact_checking.selectors.evidence_quality import retrieval_score, source_group


ORACLE_LIKELIHOOD_SELECTOR = "oracle_likelihood_top5"
SOURCE_DIVERSE_ORACLE_LIKELIHOOD_SELECTOR = "source_diverse_oracle_likelihood_top5"
STAGE2_ANCHOR1_ORACLE_LIKELIHOOD_SELECTOR = "stage2_anchor1_plus_oracle_likelihood_top5"
PRIMARY_SELECTOR = "stage2_anchor2_plus_oracle_likelihood_constrained_top5"

FORBIDDEN_FEATURE_FIELDS = {
    "oracle_selected",
    "oracle_step",
    "gold_label",
    "event_id",
    "claim",
    "text",
    "candidate_key",
    "candidate_uid",
    "canonical_text",
}

FEATURE_SET_ALL = "all_features"
FEATURE_SET_PROVENANCE_RANK = "provenance_rank_only"
FEATURE_SET_ALL_MINUS_PROVENANCE = "all_minus_provenance"
FEATURE_SET_TEACHER_DIRECTNESS_STANCE = "teacher_directness_stance_only"
FEATURE_SET_RETRIEVAL_QUALITY = "retrieval_quality_only"
FEATURE_SET_CHOICES = [
    FEATURE_SET_ALL,
    FEATURE_SET_PROVENANCE_RANK,
    FEATURE_SET_ALL_MINUS_PROVENANCE,
    FEATURE_SET_TEACHER_DIRECTNESS_STANCE,
    FEATURE_SET_RETRIEVAL_QUALITY,
]

OBJECTIVE_POINTWISE = "pointwise"
OBJECTIVE_PAIRWISE = "pairwise"
OBJECTIVE_CHOICES = [OBJECTIVE_POINTWISE, OBJECTIVE_PAIRWISE]

PROVENANCE_RANK_FEATURES = {
    "from_baseline",
    "from_qd",
    "from_both",
    "original_only",
    "qd_only",
    "union_pool_rank_recip",
    "baseline_rank_recip",
    "qd_pool_rank_recip",
    "qd_rrf_score",
    "qd_question_hit_count",
    "qd_question_hit_count_log",
    "qd_max_question_hybrid",
    "same_source_pool_count_log",
    "same_source_pool_fraction",
    "same_stance_region_pool_count_log",
    "same_stance_region_pool_fraction",
}

TEACHER_DIRECTNESS_STANCE_FEATURES = {
    "semantic_completeness_score",
    "direct_evidence_score",
    "claim_specificity_score",
    "key_fact_overlap_score",
    "background_only_score",
    "claim_directness_score",
    "role_evidence_score",
    "stance_expected_score",
    "stance_entropy",
    "stance_region_oppose",
    "stance_region_ambiguous",
    "stance_region_support",
}

RETRIEVAL_QUALITY_FEATURES = {
    "retrieval_score",
    "semantic_completeness_score",
    "claim_lexical_f1",
    "question_route_weight",
    "question_coverage_score",
}


@dataclass(frozen=True)
class LogisticParams:
    epochs: int = 800
    lr: float = 0.05
    l2: float = 1e-4
    patience: int = 80
    eval_every: int = 10
    seed: int = 20260527
    dev_fraction: float = 0.1


@dataclass(frozen=True)
class ConstrainedSelectionParams:
    top_k: int = 5
    anchor_k: int = 2
    source_penalty: float = 0.10
    stance_region_penalty: float = 0.04


def default_feature_names(bucket_names: Sequence[str]) -> list[str]:
    names = [
        "retrieval_score",
        "union_pool_rank_recip",
        "baseline_rank_recip",
        "qd_pool_rank_recip",
        "qd_rrf_score",
        "qd_question_hit_count",
        "qd_question_hit_count_log",
        "qd_max_question_hybrid",
        "from_baseline",
        "from_qd",
        "from_both",
        "original_only",
        "qd_only",
        "semantic_completeness_score",
        "claim_lexical_f1",
        "direct_evidence_score",
        "claim_specificity_score",
        "key_fact_overlap_score",
        "background_only_score",
        "claim_directness_score",
        "role_evidence_score",
        "stance_expected_score",
        "stance_entropy",
        "question_route_weight",
        "question_coverage_score",
        "same_source_pool_count_log",
        "same_source_pool_fraction",
        "same_stance_region_pool_count_log",
        "same_stance_region_pool_fraction",
        "stance_region_oppose",
        "stance_region_ambiguous",
        "stance_region_support",
    ]
    for bucket in bucket_names:
        names.append(f"stance_prob_{bucket}")
    validate_feature_names(names)
    return names


def validate_feature_names(feature_names: Sequence[str]) -> None:
    forbidden = FORBIDDEN_FEATURE_FIELDS & set(feature_names)
    if forbidden:
        raise ValueError(f"Forbidden model features: {sorted(forbidden)}")


def normalize_feature_set(feature_set: str) -> str:
    value = str(feature_set or FEATURE_SET_ALL).strip()
    if value not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature_set={feature_set!r}; expected one of {FEATURE_SET_CHOICES}")
    return value


def normalize_objective(objective: str) -> str:
    value = str(objective or OBJECTIVE_POINTWISE).strip()
    if value not in OBJECTIVE_CHOICES:
        raise ValueError(f"Unknown objective={objective!r}; expected one of {OBJECTIVE_CHOICES}")
    return value


def feature_names_for_set(all_feature_names: Sequence[str], feature_set: str) -> list[str]:
    names = list(all_feature_names)
    validate_feature_names(names)
    selected_feature_set = normalize_feature_set(feature_set)
    if selected_feature_set == FEATURE_SET_ALL:
        selected = names
    elif selected_feature_set == FEATURE_SET_PROVENANCE_RANK:
        selected = [
            name
            for name in names
            if name in PROVENANCE_RANK_FEATURES
        ]
    elif selected_feature_set == FEATURE_SET_ALL_MINUS_PROVENANCE:
        selected = [
            name
            for name in names
            if name not in PROVENANCE_RANK_FEATURES
        ]
    elif selected_feature_set == FEATURE_SET_TEACHER_DIRECTNESS_STANCE:
        selected = [
            name
            for name in names
            if name in TEACHER_DIRECTNESS_STANCE_FEATURES or name.startswith("stance_prob_")
        ]
    elif selected_feature_set == FEATURE_SET_RETRIEVAL_QUALITY:
        selected = [
            name
            for name in names
            if name in RETRIEVAL_QUALITY_FEATURES
        ]
    else:
        selected = []
    validate_feature_names(selected)
    if not selected:
        raise ValueError(f"Feature set {selected_feature_set!r} produced no features.")
    return selected


def filter_feature_rows(feature_rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> list[dict[str, Any]]:
    names = list(feature_names)
    validate_feature_names(names)
    out: list[dict[str, Any]] = []
    for row in feature_rows:
        item = dict(row)
        features = row.get("features") or {}
        item["features"] = {name: float(features.get(name, 0.0)) for name in names}
        out.append(item)
    return out


def infer_bucket_names(candidate_rows: Sequence[dict[str, Any]]) -> list[str]:
    for row in candidate_rows:
        names = row.get("stance_bucket_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        for candidate in row.get("candidates") or []:
            probs = candidate.get("teacher_stance_probs")
            if isinstance(probs, dict) and probs:
                return [str(name) for name in probs]
    return []


def build_feature_rows(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    bucket_names = infer_bucket_names(candidate_rows)
    names = list(feature_names or default_feature_names(bucket_names))
    validate_feature_names(names)
    rows: list[dict[str, Any]] = []
    for event_row in candidate_rows:
        event_id = str(event_row.get("event_id") or "")
        candidates = [dict(candidate) for candidate in event_row.get("candidates") or []]
        source_counts = Counter(source_group(candidate) for candidate in candidates)
        region_counts = Counter(stance_region(candidate) for candidate in candidates)
        pool_size = max(len(candidates), 1)
        for candidate in candidates:
            region = stance_region(candidate)
            group = source_group(candidate)
            features = candidate_features(
                candidate,
                bucket_names=bucket_names,
                source_count=int(source_counts[group]),
                stance_region_count=int(region_counts[region]),
                pool_size=pool_size,
            )
            rows.append(
                {
                    "event_id": event_id,
                    "claim": str(event_row.get("claim") or ""),
                    "gold_label": str(event_row.get("gold_label") or ""),
                    "oracle_ordered_keys": list(event_row.get("oracle_ordered_keys") or []),
                    "candidate_uid": str(candidate.get("candidate_uid") or ""),
                    "candidate_key": str(candidate.get("candidate_key") or ""),
                    "label": int(bool(candidate.get("oracle_selected"))),
                    "oracle_selected": bool(candidate.get("oracle_selected")),
                    "oracle_step": int(_safe_float(candidate.get("oracle_step"), -1)),
                    "source_group": group,
                    "stance_region": region,
                    "union_pool_rank": int(_safe_float(candidate.get("union_pool_rank"), 10**9)),
                    "baseline_rank": _nullable_int(candidate.get("baseline_rank")),
                    "qd_pool_rank": _nullable_int(candidate.get("qd_pool_rank")),
                    "features": {name: float(features.get(name, 0.0)) for name in names},
                }
            )
    return rows, names


def candidate_features(
    candidate: dict[str, Any],
    *,
    bucket_names: Sequence[str],
    source_count: int,
    stance_region_count: int,
    pool_size: int,
) -> dict[str, float]:
    route_count = _safe_float(candidate.get("qd_question_hit_count"), 0.0)
    from_baseline = bool(candidate.get("from_baseline"))
    from_qd = bool(candidate.get("from_qd"))
    features = {
        "retrieval_score": retrieval_score(candidate),
        "union_pool_rank_recip": _reciprocal_rank(candidate.get("union_pool_rank")),
        "baseline_rank_recip": _reciprocal_rank(candidate.get("baseline_rank")),
        "qd_pool_rank_recip": _reciprocal_rank(candidate.get("qd_pool_rank")),
        "qd_rrf_score": _safe_float(candidate.get("qd_rrf_score"), 0.0),
        "qd_question_hit_count": route_count,
        "qd_question_hit_count_log": math.log1p(max(route_count, 0.0)),
        "qd_max_question_hybrid": _safe_float(candidate.get("qd_max_question_hybrid"), 0.0),
        "from_baseline": _bool_float(from_baseline),
        "from_qd": _bool_float(from_qd),
        "from_both": _bool_float(from_baseline and from_qd),
        "original_only": _bool_float(from_baseline and not from_qd),
        "qd_only": _bool_float(from_qd and not from_baseline),
        "semantic_completeness_score": _safe_float(candidate.get("semantic_completeness_score"), 0.0),
        "claim_lexical_f1": _safe_float(candidate.get("claim_lexical_f1"), 0.0),
        "direct_evidence_score": _safe_float(candidate.get("direct_evidence_score"), 0.0),
        "claim_specificity_score": _safe_float(candidate.get("claim_specificity_score"), 0.0),
        "key_fact_overlap_score": _safe_float(candidate.get("key_fact_overlap_score"), 0.0),
        "background_only_score": _safe_float(candidate.get("background_only_score"), 0.0),
        "claim_directness_score": _safe_float(candidate.get("claim_directness_score"), 0.0),
        "role_evidence_score": _safe_float(candidate.get("role_evidence_score"), 0.0),
        "stance_expected_score": _safe_float(candidate.get("stance_expected_score"), 0.0),
        "stance_entropy": _safe_float(candidate.get("stance_entropy"), 0.0),
        "question_route_weight": _safe_float(candidate.get("question_route_weight"), 0.0),
        "question_coverage_score": _safe_float(candidate.get("question_coverage_score"), 0.0),
        "same_source_pool_count_log": math.log1p(max(int(source_count), 0)),
        "same_source_pool_fraction": float(max(int(source_count), 0) / max(int(pool_size), 1)),
        "same_stance_region_pool_count_log": math.log1p(max(int(stance_region_count), 0)),
        "same_stance_region_pool_fraction": float(max(int(stance_region_count), 0) / max(int(pool_size), 1)),
    }
    region = stance_region(candidate)
    features["stance_region_oppose"] = _bool_float(region == "oppose")
    features["stance_region_ambiguous"] = _bool_float(region == "ambiguous")
    features["stance_region_support"] = _bool_float(region == "support")
    probs = candidate.get("teacher_stance_probs") if isinstance(candidate.get("teacher_stance_probs"), dict) else {}
    for bucket in bucket_names:
        features[f"stance_prob_{bucket}"] = _safe_float(probs.get(bucket), 0.0)
    return features


def feature_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        features = row.get("features") or {}
        for j, name in enumerate(feature_names):
            x[i, j] = _safe_float(features.get(name), 0.0)
    return x


def labels_array(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1.0 if int(row.get("label") or 0) else 0.0 for row in rows], dtype=np.float32)


def train_logistic(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    params: LogisticParams,
) -> dict[str, Any]:
    if not train_rows:
        raise ValueError("No train rows provided.")
    dev_rows = list(dev_rows or train_rows)
    x_train_raw = feature_matrix(train_rows, feature_names)
    x_dev_raw = feature_matrix(dev_rows, feature_names)
    y_train = labels_array(train_rows)
    y_dev = labels_array(dev_rows)
    sample_weight = balanced_sample_weights(y_train)
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train_raw - mean) / std
    x_dev = (x_dev_raw - mean) / std
    rng = np.random.default_rng(int(params.seed))
    weights = rng.normal(0.0, 0.001, size=x_train.shape[1]).astype(np.float32)
    bias = _initial_bias(y_train)
    best_weights = weights.copy()
    best_bias = bias
    best_score = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    weight_sum = max(float(sample_weight.sum()), 1e-8)
    for epoch in range(1, int(params.epochs) + 1):
        logits = x_train @ weights + bias
        probs = sigmoid(logits)
        error = (probs - y_train) * sample_weight
        grad_w = (x_train.T @ error) / weight_sum + float(params.l2) * weights
        grad_b = float(error.sum() / weight_sum)
        weights -= float(params.lr) * grad_w.astype(np.float32)
        bias -= float(params.lr) * grad_b
        if epoch == 1 or epoch % int(params.eval_every) == 0:
            train_scores = sigmoid(x_train @ weights + bias)
            dev_scores = sigmoid(x_dev @ weights + bias)
            record = {
                "epoch": int(epoch),
                "train_loss": weighted_bce_loss(y_train, train_scores, sample_weight),
                "train_auroc": roc_auc_score(y_train, train_scores),
                "train_auprc": average_precision_score(y_train, train_scores),
                "dev_loss": weighted_bce_loss(y_dev, dev_scores, balanced_sample_weights(y_dev)),
                "dev_auroc": roc_auc_score(y_dev, dev_scores),
                "dev_auprc": average_precision_score(y_dev, dev_scores),
            }
            history.append(record)
            score = float(record["dev_auprc"])
            if score > best_score + 1e-6:
                best_score = score
                best_weights = weights.copy()
                best_bias = float(bias)
                stale = 0
            else:
                stale += int(params.eval_every)
                if stale >= int(params.patience):
                    break
    return {
        "weights": best_weights.astype(np.float32),
        "bias": float(best_bias),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": list(feature_names),
        "history": history,
        "params": params.__dict__,
        "objective": OBJECTIVE_POINTWISE,
    }


def train_pairwise_logistic(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    params: LogisticParams,
) -> dict[str, Any]:
    if not train_rows:
        raise ValueError("No train rows provided.")
    dev_rows = list(dev_rows or train_rows)
    x_train_raw = feature_matrix(train_rows, feature_names)
    x_dev_raw = feature_matrix(dev_rows, feature_names)
    y_train = labels_array(train_rows)
    y_dev = labels_array(dev_rows)
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train_raw - mean) / std
    x_dev = (x_dev_raw - mean) / std
    train_pairs = pairwise_diff_matrix(train_rows, x_train)
    dev_pairs = pairwise_diff_matrix(dev_rows, x_dev)
    if dev_pairs.shape[0] == 0:
        dev_pairs = train_pairs
        dev_rows = list(train_rows)
        x_dev = x_train
        y_dev = y_train
    if train_pairs.shape[0] == 0:
        raise ValueError("Pairwise objective requires at least one positive-negative candidate pair in a train event.")
    rng = np.random.default_rng(int(params.seed))
    weights = rng.normal(0.0, 0.001, size=x_train.shape[1]).astype(np.float32)
    best_weights = weights.copy()
    best_score = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    n_pairs = max(int(train_pairs.shape[0]), 1)
    for epoch in range(1, int(params.epochs) + 1):
        margins = train_pairs @ weights
        grad_factor = -sigmoid(-margins).astype(np.float32)
        grad_w = (train_pairs.T @ grad_factor) / float(n_pairs) + float(params.l2) * weights
        weights -= float(params.lr) * grad_w.astype(np.float32)
        if epoch == 1 or epoch % int(params.eval_every) == 0:
            train_logits = x_train @ weights
            dev_logits = x_dev @ weights
            record = {
                "epoch": int(epoch),
                "train_pairwise_loss": pairwise_logistic_loss(train_pairs, weights, l2=float(params.l2)),
                "train_pairwise_acc": pairwise_accuracy(train_rows, train_logits),
                "train_auroc": roc_auc_score(y_train, train_logits),
                "train_auprc": average_precision_score(y_train, train_logits),
                "dev_pairwise_loss": pairwise_logistic_loss(dev_pairs, weights, l2=float(params.l2)),
                "dev_pairwise_acc": pairwise_accuracy(dev_rows, dev_logits),
                "dev_auroc": roc_auc_score(y_dev, dev_logits),
                "dev_auprc": average_precision_score(y_dev, dev_logits),
            }
            history.append(record)
            score = float(record["dev_pairwise_acc"])
            if score > best_score + 1e-6:
                best_score = score
                best_weights = weights.copy()
                stale = 0
            else:
                stale += int(params.eval_every)
                if stale >= int(params.patience):
                    break
    return {
        "weights": best_weights.astype(np.float32),
        "bias": 0.0,
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": list(feature_names),
        "history": history,
        "params": params.__dict__,
        "objective": OBJECTIVE_PAIRWISE,
    }


def score_feature_rows(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    x = feature_matrix(rows, feature_names)
    x = (x - np.asarray(model["feature_mean"], dtype=np.float32)) / np.asarray(model["feature_std"], dtype=np.float32)
    logits = x @ np.asarray(model["weights"], dtype=np.float32) + float(model.get("bias", 0.0))
    scores = sigmoid(logits)
    return scores.astype(np.float32), logits.astype(np.float32)


def train_model_for_objective(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    params: LogisticParams,
    objective: str,
) -> dict[str, Any]:
    selected_objective = normalize_objective(objective)
    if selected_objective == OBJECTIVE_PAIRWISE:
        return train_pairwise_logistic(train_rows, dev_rows, feature_names, params=params)
    return train_logistic(train_rows, dev_rows, feature_names, params=params)


def cross_fit_score_rows(
    feature_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    folds: int,
    params: LogisticParams,
    objective: str = OBJECTIVE_POINTWISE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(row) for row in feature_rows]
    selected_objective = normalize_objective(objective)
    fold_events = split_event_ids_kfold(rows, folds=folds, seed=int(params.seed))
    scored: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for fold_idx, heldout_events in enumerate(fold_events):
        train_events = sorted({str(row["event_id"]) for row in rows} - set(heldout_events))
        inner_train_events, inner_dev_events = split_train_dev_events(
            train_events,
            dev_fraction=float(params.dev_fraction),
            seed=int(params.seed) + fold_idx + 1,
        )
        train_rows = [row for row in rows if str(row["event_id"]) in inner_train_events]
        dev_rows = [row for row in rows if str(row["event_id"]) in inner_dev_events]
        heldout_rows = [row for row in rows if str(row["event_id"]) in heldout_events]
        model = train_model_for_objective(train_rows, dev_rows, feature_names, params=params, objective=selected_objective)
        scores, logits = score_feature_rows(heldout_rows, feature_names, model)
        model_record = dict(model)
        model_record["fold"] = int(fold_idx)
        model_record["train_event_ids"] = sorted(inner_train_events)
        model_record["dev_event_ids"] = sorted(inner_dev_events)
        model_record["heldout_event_ids"] = sorted(heldout_events)
        models.append(model_record)
        fold_records.append(
            {
                "fold": int(fold_idx),
                "n_train_events": len(inner_train_events),
                "n_dev_events": len(inner_dev_events),
                "n_heldout_events": len(heldout_events),
                "n_train_rows": len(train_rows),
                "n_dev_rows": len(dev_rows),
                "n_heldout_rows": len(heldout_rows),
                "objective": selected_objective,
                "heldout_metrics": candidate_level_metrics(_attach_row_scores(heldout_rows, scores, logits, fold_idx)),
            }
        )
        scored.extend(_attach_row_scores(heldout_rows, scores, logits, fold_idx))
    scored.sort(key=lambda row: (str(row.get("event_id") or ""), int(row.get("union_pool_rank") or 10**9), str(row.get("candidate_key") or "")))
    return scored, models, fold_records


def train_heldout_score_rows(
    train_feature_rows: Sequence[dict[str, Any]],
    eval_feature_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    params: LogisticParams,
    objective: str = OBJECTIVE_POINTWISE,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    selected_objective = normalize_objective(objective)
    train_events = sorted({str(row["event_id"]) for row in train_feature_rows})
    inner_train_events, inner_dev_events = split_train_dev_events(
        train_events,
        dev_fraction=float(params.dev_fraction),
        seed=int(params.seed),
    )
    train_rows = [row for row in train_feature_rows if str(row["event_id"]) in inner_train_events]
    dev_rows = [row for row in train_feature_rows if str(row["event_id"]) in inner_dev_events]
    model = train_model_for_objective(train_rows, dev_rows, feature_names, params=params, objective=selected_objective)
    scores, logits = score_feature_rows(eval_feature_rows, feature_names, model)
    scored = _attach_row_scores(eval_feature_rows, scores, logits, 0)
    metadata = {
        "n_train_events": len(inner_train_events),
        "n_dev_events": len(inner_dev_events),
        "n_eval_events": len({str(row["event_id"]) for row in eval_feature_rows}),
        "n_train_rows": len(train_rows),
        "n_dev_rows": len(dev_rows),
        "n_eval_rows": len(eval_feature_rows),
        "objective": selected_objective,
    }
    return scored, model, metadata


def split_event_ids_kfold(rows: Sequence[dict[str, Any]], *, folds: int, seed: int) -> list[set[str]]:
    event_ids = sorted({str(row.get("event_id") or "") for row in rows if row.get("event_id")})
    if not event_ids:
        raise ValueError("No event ids available for fold split.")
    n_folds = max(1, min(int(folds), len(event_ids)))
    rng = random.Random(int(seed))
    rng.shuffle(event_ids)
    out = [set() for _ in range(n_folds)]
    for idx, event_id in enumerate(event_ids):
        out[idx % n_folds].add(event_id)
    return out


def split_train_dev_events(
    event_ids: Sequence[str],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    ids = sorted({str(event_id) for event_id in event_ids if str(event_id)})
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    if len(ids) <= 1:
        return set(ids), set(ids)
    n_dev = max(1, int(round(len(ids) * float(dev_fraction))))
    n_dev = min(n_dev, len(ids) - 1)
    dev = set(ids[:n_dev])
    train = set(ids[n_dev:])
    return train, dev


def attach_scores_to_candidate_rows(
    candidate_rows: Sequence[dict[str, Any]],
    scored_feature_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored_by_key = {
        _row_key(row): row
        for row in scored_feature_rows
    }
    out: list[dict[str, Any]] = []
    for event_row in candidate_rows:
        item = dict(event_row)
        candidates: list[dict[str, Any]] = []
        for candidate in event_row.get("candidates") or []:
            c = dict(candidate)
            key = _candidate_key(str(event_row.get("event_id") or ""), c)
            scored = scored_by_key.get(key)
            if scored is None:
                raise KeyError(f"Missing oracle-likelihood score for candidate key={key}")
            c["oracle_likelihood_score"] = float(scored.get("oracle_likelihood_score") or 0.0)
            c["oracle_likelihood_logit"] = float(scored.get("oracle_likelihood_logit") or 0.0)
            c["oracle_likelihood_fold"] = int(scored.get("oracle_likelihood_fold") or 0)
            c["oracle_likelihood_features"] = dict(scored.get("features") or {})
            c["stance_region"] = str(scored.get("stance_region") or stance_region(c))
            candidates.append(c)
        item["candidates"] = candidates
        out.append(item)
    return out


def select_likelihood_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    top_k: int,
    selector_name: str = ORACLE_LIKELIHOOD_SELECTOR,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    rows.sort(key=_likelihood_sort_key, reverse=True)
    return _ranked_unique(rows, top_k=top_k, selector_name=selector_name, origin="learned_rank")


def select_constrained_likelihood_topk(
    candidates: Sequence[dict[str, Any]],
    *,
    params: ConstrainedSelectionParams,
    selector_name: str = PRIMARY_SELECTOR,
) -> list[dict[str, Any]]:
    pool = [dict(candidate) for candidate in candidates]
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    stance_counts: Counter[str] = Counter()
    if int(params.anchor_k) > 0:
        anchors = [candidate for candidate in pool if bool(candidate.get("from_baseline"))]
        anchors.sort(key=lambda row: (int(_safe_float(row.get("baseline_rank"), 10**9)), int(_safe_float(row.get("union_pool_rank"), 10**9))))
        for candidate in anchors:
            if len(selected) >= int(params.anchor_k):
                break
            key = _selection_key(candidate)
            if key in selected_keys:
                continue
            item = dict(candidate)
            item["selection_origin"] = "stage2_anchor"
            item["oracle_likelihood_adjusted_score"] = float(item.get("oracle_likelihood_score") or 0.0)
            _append_selected(item, selected, selected_keys, source_counts, stance_counts, selector_name)
    while len(selected) < int(params.top_k):
        best: dict[str, Any] | None = None
        for candidate in pool:
            key = _selection_key(candidate)
            if not key or key in selected_keys:
                continue
            item = dict(candidate)
            same_source = source_counts[source_group(item)]
            same_stance = stance_counts[stance_region(item)]
            adjusted = (
                _safe_float(item.get("oracle_likelihood_score"), 0.0)
                - float(params.source_penalty) * float(same_source)
                - float(params.stance_region_penalty) * float(same_stance)
            )
            item["selection_origin"] = "learned_fill"
            item["oracle_likelihood_adjusted_score"] = float(adjusted)
            item["same_source_selected_count"] = int(same_source)
            item["same_stance_region_selected_count"] = int(same_stance)
            if best is None or _adjusted_sort_key(item) > _adjusted_sort_key(best):
                best = item
        if best is None:
            break
        _append_selected(best, selected, selected_keys, source_counts, stance_counts, selector_name)
    return selected


def build_oracle_likelihood_trace(
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
        "selector_name": selector_name,
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
                "oracle_likelihood_score": _safe_float(candidate.get("oracle_likelihood_score"), 0.0),
                "oracle_likelihood_adjusted_score": _safe_float(candidate.get("oracle_likelihood_adjusted_score"), 0.0),
                "stance_region": stance_region(candidate),
                "source_group": source_group(candidate),
            }
            for idx, candidate in enumerate(selected)
        ],
    }
    trace.update(text_ordered_selection_metrics(trace["oracle_ordered_keys"], selected, top_k=top_k))
    trace.update(selection_quality_metrics(selected))
    trace["mean_oracle_likelihood_score@5"] = _mean(_safe_float(candidate.get("oracle_likelihood_score"), 0.0) for candidate in selected)
    return trace


def summarize_oracle_likelihood_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_selector_traces(traces)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace.get("selector_name") or "")].append(dict(trace))
    for selector, rows in grouped.items():
        scores = []
        anchor_hits = []
        fill_hits = []
        anchor_counts = []
        fill_counts = []
        collapse = []
        for trace in rows:
            selected = list(trace.get("selected_candidates") or [])
            scores.extend(_safe_float(candidate.get("oracle_likelihood_score"), 0.0) for candidate in selected)
            anchor = [candidate for candidate in selected if str(candidate.get("selection_origin") or "") == "stage2_anchor"]
            fill = [
                candidate
                for candidate in selected
                if str(candidate.get("selection_origin") or "") in {"learned_rank", "learned_fill"}
            ]
            anchor_counts.append(len(anchor))
            fill_counts.append(len(fill))
            anchor_hits.extend(1.0 if bool(candidate.get("oracle_selected")) else 0.0 for candidate in anchor)
            fill_hits.extend(1.0 if bool(candidate.get("oracle_selected")) else 0.0 for candidate in fill)
            buckets = [str(candidate.get("stance_bucket_derived") or "") for candidate in selected if candidate.get("stance_bucket_derived")]
            if buckets:
                counts = Counter(buckets)
                collapse.append(float(max(counts.values()) == len(buckets)))
        item = summary.setdefault(selector, {"n_claims": len(rows)})
        item["mean_oracle_likelihood_score@5"] = _mean(scores)
        item["mean_anchor_count@5"] = _mean(anchor_counts)
        item["mean_learned_fill_count@5"] = _mean(fill_counts)
        item["anchor_hit_rate@5"] = _mean(anchor_hits)
        item["learned_fill_hit_rate@5"] = _mean(fill_hits)
        item["single_bucket_collapse_rate@5"] = _mean(collapse)
    return summary


def oracle_likelihood_diagnostics(
    scored_rows: Sequence[dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    *,
    primary_selector: str = PRIMARY_SELECTOR,
) -> dict[str, Any]:
    flat_candidates = [
        dict(candidate)
        for row in scored_rows
        for candidate in row.get("candidates") or []
    ]
    labels = np.asarray([1.0 if candidate.get("oracle_selected") else 0.0 for candidate in flat_candidates], dtype=np.float32)
    scores = np.asarray([_safe_float(candidate.get("oracle_likelihood_score"), 0.0) for candidate in flat_candidates], dtype=np.float32)
    primary_selected = [
        candidate
        for trace in traces
        if str(trace.get("selector_name") or "") == primary_selector
        for candidate in trace.get("selected_candidates") or []
    ]
    selected_scores = [score for label, score in zip(labels, scores) if label > 0.0]
    nonselected_scores = [score for label, score in zip(labels, scores) if label <= 0.0]
    return {
        "candidate_level": candidate_level_metrics(_score_rows_from_arrays(labels, scores)),
        "oracle_selected_score_mean": _mean(selected_scores),
        "non_oracle_selected_score_mean": _mean(nonselected_scores),
        "oracle_selected_score_lift": _mean(selected_scores) - _mean(nonselected_scores),
        "primary_selected_score_mean": _mean(_safe_float(candidate.get("oracle_likelihood_score"), 0.0) for candidate in primary_selected),
        "primary_role_composition": _composition(primary_selected, "evidence_role"),
        "pool_role_composition": _composition(flat_candidates, "evidence_role"),
        "primary_selection_origin_composition": _composition(primary_selected, "selection_origin"),
        "primary_stance_region_composition": _composition(primary_selected, "stance_region"),
        "pool_stance_region_composition": _composition(flat_candidates, "stance_region"),
    }


def candidate_level_metrics(scored_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([float(row.get("label") or 0.0) for row in scored_rows], dtype=np.float32)
    scores = np.asarray([_safe_float(row.get("oracle_likelihood_score"), 0.0) for row in scored_rows], dtype=np.float32)
    return {
        "n_rows": int(len(scored_rows)),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)) if labels.size else 0.0,
        "log_loss": binary_log_loss(labels, scores),
        "calibration_bins": calibration_bins(labels, scores, n_bins=10),
    }


def save_logistic_model(
    path: str | Path,
    *,
    model: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=np.asarray(model["weights"], dtype=np.float32),
        bias=np.asarray([float(model.get("bias", 0.0))], dtype=np.float32),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(model["feature_std"], dtype=np.float32),
        feature_names=np.asarray(list(model.get("feature_names") or []), dtype=object),
        metadata_json=np.array(json.dumps(metadata or {}, ensure_ascii=False), dtype=object),
    )


def feature_importance_rows(models: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> list[dict[str, Any]]:
    if not models:
        return []
    weights = np.vstack([np.asarray(model["weights"], dtype=np.float32) for model in models])
    rows = []
    for idx, feature in enumerate(feature_names):
        values = weights[:, idx]
        rows.append(
            {
                "feature": str(feature),
                "mean_weight": float(values.mean()),
                "mean_abs_weight": float(np.abs(values).mean()),
                "std_weight": float(values.std()),
            }
        )
    rows.sort(key=lambda row: float(row["mean_abs_weight"]), reverse=True)
    return rows


def weighted_bce_loss(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    scores = np.clip(scores.astype(np.float64), 1e-6, 1.0 - 1e-6)
    labels = labels.astype(np.float64)
    weights = weights.astype(np.float64)
    loss = -(labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores))
    return float((loss * weights).sum() / max(float(weights.sum()), 1e-8))


def binary_log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return weighted_bce_loss(labels, scores, np.ones_like(labels, dtype=np.float32))


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.float32)
    n = max(int(labels.shape[0]), 1)
    positives = float(labels.sum())
    negatives = float(n - positives)
    weights = np.ones(n, dtype=np.float32)
    if positives > 0.0:
        weights[labels > 0.0] = n / (2.0 * positives)
    if negatives > 0.0:
        weights[labels <= 0.0] = n / (2.0 * negatives)
    return weights


def pairwise_diff_matrix(rows: Sequence[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    by_event: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_event[str(row.get("event_id") or "")].append(idx)
    diffs: list[np.ndarray] = []
    labels = labels_array(rows)
    for indexes in by_event.values():
        positives = [idx for idx in indexes if labels[idx] > 0.0]
        negatives = [idx for idx in indexes if labels[idx] <= 0.0]
        for pos_idx in positives:
            for neg_idx in negatives:
                diffs.append(x[pos_idx] - x[neg_idx])
    if not diffs:
        return np.zeros((0, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32)
    return np.vstack(diffs).astype(np.float32)


def pairwise_logistic_loss(pair_diffs: np.ndarray, weights: np.ndarray, *, l2: float) -> float:
    if pair_diffs.shape[0] == 0:
        return 0.0
    margins = pair_diffs @ weights
    loss = np.logaddexp(0.0, -margins.astype(np.float64)).mean()
    reg = 0.5 * float(l2) * float(np.sum(weights.astype(np.float64) ** 2))
    return float(loss + reg)


def pairwise_accuracy(rows: Sequence[dict[str, Any]], scores: Sequence[float] | np.ndarray) -> float:
    by_event: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_event[str(row.get("event_id") or "")].append((float(score), int(row.get("label") or 0)))
    hits = 0.0
    total = 0
    for items in by_event.values():
        positives = [score for score, label in items if label > 0]
        negatives = [score for score, label in items if label <= 0]
        for pos_score in positives:
            for neg_score in negatives:
                if pos_score > neg_score:
                    hits += 1.0
                elif pos_score == neg_score:
                    hits += 0.5
                total += 1
    if total == 0:
        return 0.0
    return float(hits / total)


def roc_auc_score(labels: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    pairs = [(float(score), int(label > 0.0)) for label, score in zip(labels, scores)]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
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
        rank_sum += avg_rank * sum(label for _, label in pairs[idx:end])
        idx = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision_score(labels: Sequence[float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    pairs = sorted(((float(score), int(label > 0.0)) for label, score in zip(labels, scores)), reverse=True)
    positives = sum(label for _, label in pairs)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return float(precision_sum / positives)


def calibration_bins(labels: np.ndarray, scores: np.ndarray, *, n_bins: int) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    labels = labels.astype(np.float32)
    scores = scores.astype(np.float32)
    for idx in range(int(n_bins)):
        low = idx / float(n_bins)
        high = (idx + 1) / float(n_bins)
        if idx == int(n_bins) - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        count = int(mask.sum())
        if count:
            bins.append(
                {
                    "bin": int(idx),
                    "low": float(low),
                    "high": float(high),
                    "count": count,
                    "mean_score": float(scores[mask].mean()),
                    "positive_rate": float(labels[mask].mean()),
                }
            )
        else:
            bins.append({"bin": int(idx), "low": float(low), "high": float(high), "count": 0, "mean_score": 0.0, "positive_rate": 0.0})
    return bins


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def stance_region(candidate: dict[str, Any]) -> str:
    region = str(candidate.get("stance_region") or "").strip().lower()
    if region in {"oppose", "ambiguous", "support"}:
        return region
    bucket = str(candidate.get("stance_bucket_derived") or "")
    if "oppose" in bucket:
        return "oppose"
    if "support" in bucket:
        return "support"
    return "ambiguous"


def _attach_row_scores(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    logits: Sequence[float],
    fold: int,
) -> list[dict[str, Any]]:
    out = []
    for row, score, logit in zip(rows, scores, logits):
        item = dict(row)
        item["oracle_likelihood_score"] = float(score)
        item["oracle_likelihood_logit"] = float(logit)
        item["oracle_likelihood_fold"] = int(fold)
        out.append(item)
    return out


def _score_rows_from_arrays(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"label": int(label > 0.0), "oracle_likelihood_score": float(score)}
        for label, score in zip(labels, scores)
    ]


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
        "relevance_gate_score",
        "stance_bucket_derived",
        "stance_region",
        "stance_entropy",
        "source_group",
        "oracle_selected",
        "oracle_step",
        "oracle_likelihood_score",
        "oracle_likelihood_adjusted_score",
        "oracle_likelihood_fold",
        "same_source_selected_count",
        "same_stance_region_selected_count",
        "claim_specificity_score",
        "direct_evidence_score",
        "background_only_score",
        "key_fact_overlap_score",
        "evidence_role",
        "role_evidence_score",
        "claim_directness_score",
        "text",
    ]
    out = {key: candidate.get(key) for key in keys if key in candidate}
    out["source_group"] = source_group(candidate)
    out["stance_region"] = stance_region(candidate)
    return out


def _append_selected(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    selected_keys: set[str],
    source_counts: Counter[str],
    stance_counts: Counter[str],
    selector_name: str,
) -> None:
    key = _selection_key(item)
    if not key:
        return
    item["selector_name"] = selector_name
    item["selection_rank"] = len(selected) + 1
    item["source_group"] = source_group(item)
    item["stance_region"] = stance_region(item)
    selected.append(item)
    selected_keys.add(key)
    source_counts[source_group(item)] += 1
    stance_counts[stance_region(item)] += 1


def _ranked_unique(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    selector_name: str,
    origin: str,
) -> list[dict[str, Any]]:
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
        item["oracle_likelihood_adjusted_score"] = float(item.get("oracle_likelihood_score") or 0.0)
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def _likelihood_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        _safe_float(candidate.get("oracle_likelihood_score"), 0.0),
        retrieval_score(candidate),
        _safe_float(candidate.get("semantic_completeness_score"), 0.0),
        -int(_safe_float(candidate.get("union_pool_rank"), 10**9)),
        str(candidate.get("candidate_key") or ""),
    )


def _adjusted_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, int, str]:
    return (
        _safe_float(candidate.get("oracle_likelihood_adjusted_score"), 0.0),
        _safe_float(candidate.get("oracle_likelihood_score"), 0.0),
        retrieval_score(candidate),
        _safe_float(candidate.get("semantic_completeness_score"), 0.0),
        -int(_safe_float(candidate.get("union_pool_rank"), 10**9)),
        str(candidate.get("candidate_key") or ""),
    )


def _selection_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_key") or candidate.get("candidate_uid") or "")


def _candidate_key(event_id: str, candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (str(event_id), str(candidate.get("candidate_uid") or ""), str(candidate.get("candidate_key") or ""))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("event_id") or ""), str(row.get("candidate_uid") or ""), str(row.get("candidate_key") or ""))


def _composition(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [str(row.get(key) or "") for row in rows]
    if key == "stance_region":
        values = [stance_region(row) for row in rows]
    counts = Counter(value for value in values if value)
    total = max(sum(counts.values()), 1)
    return {
        value: {"count": int(count), "fraction": float(count / total)}
        for value, count in sorted(counts.items())
    }


def _reciprocal_rank(value: Any) -> float:
    rank = _nullable_int(value)
    if rank is None or rank <= 0 or rank >= 10**9:
        return 0.0
    return float(1.0 / max(rank, 1))


def _nullable_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed


def _initial_bias(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    rate = float(np.clip(labels.mean(), 1e-4, 1.0 - 1e-4))
    return float(math.log(rate / (1.0 - rate)))


def _bool_float(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _mean(values: Sequence[float] | Any) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else 0.0
