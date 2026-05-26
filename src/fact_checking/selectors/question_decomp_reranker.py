from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np

from fact_checking.build.candidates import canonicalize_sentence
from fact_checking.retrieval.text_utils import content_tokens_counter, lexical_overlap_f1_from_counters
from fact_checking.selectors.question_decomp_retrieval import _compute_prediction_metrics


FOCI = ("overall", "quantity", "attribution", "entity", "comparison", "policy", "time", "causal", "other")


@dataclass(frozen=True)
class PairwiseRerankerParams:
    top_k: int = 5
    val_fraction: float = 0.2
    seed: int = 20260526
    epochs: int = 500
    lr: float = 0.05
    l2: float = 1e-4
    patience: int = 50
    eval_every: int = 10


def default_feature_names() -> list[str]:
    names = [
        "from_baseline",
        "from_qd",
        "from_both",
        "source_baseline_only",
        "source_qd_only",
        "baseline_rank_inv",
        "baseline_rank_norm",
        "baseline_missing",
        "baseline_hybrid_score",
        "baseline_dense_score",
        "baseline_lexical_score",
        "baseline_bm25_score",
        "qd_pool_rank_inv",
        "qd_pool_rank_norm",
        "qd_missing",
        "qd_rrf_score",
        "qd_question_hit_count",
        "qd_max_question_hybrid",
        "qd_min_route_rank_inv",
        "qd_mean_route_rank_inv",
        "qd_has_q1_route",
        "union_pool_rank_inv",
        "evidence_token_len_log",
        "claim_evidence_lexical_f1",
        "candidate_sent_idx_log_inv",
        "candidate_chunk_sent_count",
        "has_source_report",
        "domain_present",
        "domain_is_factcheck_like",
        "domain_is_news_like",
        "domain_is_social_or_forum_like",
        "domain_is_gov_edu_org",
        "domain_is_blog_like",
        "domain_is_reference_like",
        "domain_is_unknown",
        "same_report_union_candidate_count",
        "same_report_qd_candidate_count",
        "same_report_baseline_candidate_count",
        "same_report_has_baseline",
        "same_report_max_qd_rrf",
        "same_report_min_qd_rank_inv",
        "same_report_focus_count",
        "same_report_question_count",
        "same_report_candidate_share",
    ]
    for focus in FOCI:
        names.append(f"qd_focus_count_{focus}")
        names.append(f"qd_focus_has_{focus}")
    return names


def build_feature_rows(
    union_rows: Sequence[dict[str, Any]],
    *,
    oracle_texts: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for union_row in union_rows:
        event_id = str(union_row.get("event_id") or "")
        claim = str(union_row.get("claim") or "")
        candidates = list(union_row.get("candidates") or [])
        report_stats = _report_stats(candidates)
        oracle = set(oracle_texts.get(event_id) or set())
        for candidate in candidates:
            key = _candidate_key(candidate)
            features = candidate_features(candidate, claim=claim, candidates=candidates, report_stats=report_stats)
            rows.append(
                {
                    "event_id": event_id,
                    "claim": claim,
                    "candidate_key": key,
                    "text": str(candidate.get("text") or ""),
                    "label": 1 if key and key in oracle else 0,
                    "features": features,
                    "union_source": str(candidate.get("union_source") or ""),
                    "from_baseline": bool(candidate.get("from_baseline")),
                    "from_qd": bool(candidate.get("from_qd")),
                    "baseline_rank": candidate.get("baseline_rank"),
                    "qd_pool_rank": candidate.get("qd_pool_rank"),
                }
            )
    return rows


def candidate_features(
    candidate: dict[str, Any],
    *,
    claim: str,
    candidates: Sequence[dict[str, Any]],
    report_stats: dict[str, dict[str, Any]],
) -> dict[str, float]:
    baseline_rank = _optional_float(candidate.get("baseline_rank"))
    qd_rank = _optional_float(candidate.get("qd_pool_rank"))
    union_rank = _optional_float(candidate.get("union_pool_rank"))
    routes = list(candidate.get("qd_question_routes") or candidate.get("question_routes") or [])
    focus_counts = Counter(str(route.get("focus") or "other") for route in routes)
    route_ranks = [_safe_float(route.get("rank"), 0.0) for route in routes if _safe_float(route.get("rank"), 0.0) > 0]
    text = str(candidate.get("text") or "")
    report_id = _report_id(candidate)
    stats = report_stats.get(report_id, {})
    domain = _extract_domain(candidate)
    domain_flags = _domain_flags(domain)
    q_ctr, q_len = content_tokens_counter(claim)
    s_ctr, s_len = content_tokens_counter(text)
    features: dict[str, float] = {
        "from_baseline": _bool(candidate.get("from_baseline")),
        "from_qd": _bool(candidate.get("from_qd")),
        "from_both": _bool(candidate.get("from_baseline") and candidate.get("from_qd")),
        "source_baseline_only": _bool(candidate.get("from_baseline") and not candidate.get("from_qd")),
        "source_qd_only": _bool(candidate.get("from_qd") and not candidate.get("from_baseline")),
        "baseline_rank_inv": _rank_inv(baseline_rank),
        "baseline_rank_norm": _rank_norm(baseline_rank, 5.0),
        "baseline_missing": _bool(baseline_rank is None),
        "baseline_hybrid_score": _safe_float(candidate.get("baseline_hybrid_score"), 0.0),
        "baseline_dense_score": _safe_float(candidate.get("dense_score"), 0.0) if candidate.get("from_baseline") else 0.0,
        "baseline_lexical_score": _safe_float(candidate.get("lexical_score"), 0.0) if candidate.get("from_baseline") else 0.0,
        "baseline_bm25_score": _safe_float(candidate.get("bm25_score"), 0.0) if candidate.get("from_baseline") else 0.0,
        "qd_pool_rank_inv": _rank_inv(qd_rank),
        "qd_pool_rank_norm": _rank_norm(qd_rank, 15.0),
        "qd_missing": _bool(qd_rank is None),
        "qd_rrf_score": _safe_float(candidate.get("qd_rrf_score"), 0.0),
        "qd_question_hit_count": _safe_float(candidate.get("qd_question_hit_count"), 0.0),
        "qd_max_question_hybrid": _safe_float(candidate.get("qd_max_question_hybrid"), 0.0),
        "qd_min_route_rank_inv": _rank_inv(min(route_ranks) if route_ranks else None),
        "qd_mean_route_rank_inv": _rank_inv(float(sum(route_ranks) / len(route_ranks)) if route_ranks else None),
        "qd_has_q1_route": _bool(any(str(route.get("question_id") or "") == "q1" for route in routes)),
        "union_pool_rank_inv": _rank_inv(union_rank),
        "evidence_token_len_log": math.log1p(float(s_len)),
        "claim_evidence_lexical_f1": float(lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)),
        "candidate_sent_idx_log_inv": 1.0 / math.log2(_safe_float(candidate.get("sent_idx"), 0.0) + 2.0),
        "candidate_chunk_sent_count": float(len(candidate.get("chunk_sent_indices") or []) or 1),
        "has_source_report": _bool(bool(candidate.get("source_report"))),
        "same_report_union_candidate_count": _safe_float(stats.get("union_count"), 0.0),
        "same_report_qd_candidate_count": _safe_float(stats.get("qd_count"), 0.0),
        "same_report_baseline_candidate_count": _safe_float(stats.get("baseline_count"), 0.0),
        "same_report_has_baseline": _bool(stats.get("baseline_count", 0) > 0),
        "same_report_max_qd_rrf": _safe_float(stats.get("max_qd_rrf"), 0.0),
        "same_report_min_qd_rank_inv": _rank_inv(_optional_float(stats.get("min_qd_rank"))),
        "same_report_focus_count": _safe_float(stats.get("focus_count"), 0.0),
        "same_report_question_count": _safe_float(stats.get("question_count"), 0.0),
        "same_report_candidate_share": _safe_float(stats.get("union_count"), 0.0) / max(float(len(candidates)), 1.0),
    }
    features.update(domain_flags)
    for focus in FOCI:
        count = float(focus_counts.get(focus, 0))
        features[f"qd_focus_count_{focus}"] = count
        features[f"qd_focus_has_{focus}"] = _bool(count > 0)
    return features


def split_event_ids(rows: Sequence[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    event_ids = sorted({str(row.get("event_id") or "") for row in rows if row.get("event_id")})
    rng = random.Random(int(seed))
    rng.shuffle(event_ids)
    n_val = max(1, int(round(len(event_ids) * float(val_fraction)))) if len(event_ids) > 1 else 0
    val = set(event_ids[:n_val])
    train = set(event_ids[n_val:])
    if not train and val:
        moved = sorted(val)[0]
        val.remove(moved)
        train.add(moved)
    return train, val


def feature_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        features = row.get("features") or {}
        for j, name in enumerate(feature_names):
            x[i, j] = _safe_float(features.get(name), 0.0)
    return x


def train_pairwise_logistic(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    params: PairwiseRerankerParams,
) -> dict[str, Any]:
    x_train_raw = feature_matrix(train_rows, feature_names)
    x_val_raw = feature_matrix(val_rows, feature_names)
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std if len(val_rows) else x_val_raw
    train_pairs = _pairwise_diffs(train_rows, x_train)
    val_pairs = _pairwise_diffs(val_rows, x_val) if len(val_rows) else np.zeros((0, x_train.shape[1]), dtype=np.float32)
    if train_pairs.shape[0] == 0:
        raise ValueError("No positive/negative pairs available for pairwise training.")
    weights = np.zeros(x_train.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_score = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(params.epochs) + 1):
        margins = train_pairs @ weights
        probs = 1.0 / (1.0 + np.exp(np.clip(margins, -50.0, 50.0)))
        grad = -(train_pairs.T @ probs) / max(float(train_pairs.shape[0]), 1.0)
        grad += float(params.l2) * weights
        weights -= float(params.lr) * grad.astype(np.float32)
        if epoch % int(params.eval_every) == 0 or epoch == 1:
            train_acc = pairwise_accuracy(train_pairs, weights)
            val_acc = pairwise_accuracy(val_pairs, weights) if val_pairs.shape[0] else 0.0
            record = {"epoch": epoch, "train_pairwise_acc": train_acc, "val_pairwise_acc": val_acc}
            history.append(record)
            score = val_acc if val_pairs.shape[0] else train_acc
            if score > best_score + 1e-6:
                best_score = score
                best_weights = weights.copy()
                stale = 0
            else:
                stale += int(params.eval_every)
                if stale >= int(params.patience):
                    break
    return {
        "weights": best_weights,
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "history": history,
        "n_train_pairs": int(train_pairs.shape[0]),
        "n_val_pairs": int(val_pairs.shape[0]),
        "train_pairwise_acc": pairwise_accuracy(train_pairs, best_weights),
        "val_pairwise_acc": pairwise_accuracy(val_pairs, best_weights) if val_pairs.shape[0] else 0.0,
    }


def score_rows(rows: Sequence[dict[str, Any]], feature_names: Sequence[str], model: dict[str, Any]) -> np.ndarray:
    x = feature_matrix(rows, feature_names)
    x = (x - model["feature_mean"]) / model["feature_std"]
    return (x @ model["weights"]).astype(np.float32)


def pairwise_metrics_for_rows(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    model: dict[str, Any],
) -> dict[str, Any]:
    x = feature_matrix(rows, feature_names)
    x = (x - model["feature_mean"]) / model["feature_std"]
    pairs = _pairwise_diffs(rows, x)
    return {
        "n_pairs": int(pairs.shape[0]),
        "pairwise_acc": pairwise_accuracy(pairs, model["weights"]),
    }


def save_pairwise_logistic_model(
    path: str,
    *,
    model: dict[str, Any],
    feature_names: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    np.savez(
        path,
        weights=np.asarray(model["weights"], dtype=np.float32),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(model["feature_std"], dtype=np.float32),
        feature_names=np.asarray(list(feature_names), dtype=object),
        metadata_json=np.array(metadata or {}, dtype=object),
    )


def load_pairwise_logistic_model(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    feature_names = [str(item) for item in data["feature_names"].tolist()]
    metadata_raw = data["metadata_json"].tolist() if "metadata_json" in data else {}
    if isinstance(metadata_raw, dict):
        metadata = metadata_raw
    elif isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}
    return {
        "weights": data["weights"].astype(np.float32),
        "feature_mean": data["feature_mean"].astype(np.float32),
        "feature_std": data["feature_std"].astype(np.float32),
        "feature_names": feature_names,
        "metadata": metadata,
    }


def build_selected_rows(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    top_k: int,
    mode: str,
    baseline_anchor_k: int = 0,
) -> list[dict[str, Any]]:
    by_event: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_event[str(row["event_id"])].append((row, float(score)))
    selected_rows: list[dict[str, Any]] = []
    for event_id, items in sorted(by_event.items()):
        claim = str(items[0][0].get("claim") or "") if items else ""
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        if baseline_anchor_k > 0:
            anchors = [item for item in items if item[0].get("from_baseline")]
            anchors.sort(key=lambda item: int(item[0].get("baseline_rank") or 10**9))
            for row, score in anchors[:baseline_anchor_k]:
                _append_selected(selected, seen, row, score, mode)
        remaining = sorted(items, key=lambda item: (-item[1], int(item[0].get("baseline_rank") or 10**9), int(item[0].get("qd_pool_rank") or 10**9)))
        for row, score in remaining:
            if len(selected) >= int(top_k):
                break
            _append_selected(selected, seen, row, score, mode)
        selected_rows.append({"event_id": event_id, "claim": claim, "candidates": selected})
    return selected_rows


def evaluate_selected_rows(
    selected_rows: Sequence[dict[str, Any]],
    *,
    oracle_texts: dict[str, set[str]],
) -> dict[str, Any]:
    return _compute_prediction_metrics(
        pool_rows=[{"event_id": row.get("event_id"), "pool": row.get("candidates") or [], "selected": row.get("candidates") or []} for row in selected_rows],
        oracle_texts=oracle_texts,
        include_pool=False,
    )


def source_composition(selected_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    total = 0
    for row in selected_rows:
        for candidate in row.get("candidates") or []:
            total += 1
            counts[str(candidate.get("union_source") or "unknown")] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "share": {key: float(value / total) if total else 0.0 for key, value in sorted(counts.items())},
    }


def feature_importance(feature_names: Sequence[str], weights: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"feature": name, "weight": float(weight)}
        for name, weight in sorted(zip(feature_names, weights), key=lambda item: abs(float(item[1])), reverse=True)
    ]


def pairwise_accuracy(pair_diffs: np.ndarray, weights: np.ndarray) -> float:
    if pair_diffs.shape[0] == 0:
        return 0.0
    margins = pair_diffs @ weights
    return float((margins > 0).mean())


def _pairwise_diffs(rows: Sequence[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    by_event: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_event[str(row["event_id"])].append(idx)
    diffs: list[np.ndarray] = []
    for indices in by_event.values():
        positives = [idx for idx in indices if int(rows[idx].get("label") or 0) == 1]
        negatives = [idx for idx in indices if int(rows[idx].get("label") or 0) == 0]
        for pos in positives:
            for neg in negatives:
                diffs.append(x[pos] - x[neg])
    if not diffs:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    return np.asarray(diffs, dtype=np.float32)


def _append_selected(selected: list[dict[str, Any]], seen: set[str], row: dict[str, Any], score: float, mode: str) -> None:
    key = str(row.get("candidate_key") or canonicalize_sentence(str(row.get("text") or "")))
    if not key or key in seen:
        return
    selected.append(
        {
            "text": row.get("text", ""),
            "canonical_text": key,
            "selection_rank": len(selected) + 1,
            "model_score": float(score),
            "selection_mode": mode,
            "union_source": row.get("union_source"),
            "from_baseline": bool(row.get("from_baseline")),
            "from_qd": bool(row.get("from_qd")),
            "baseline_rank": row.get("baseline_rank"),
            "qd_pool_rank": row.get("qd_pool_rank"),
        }
    )
    seen.add(key)


def _report_stats(candidates: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[_report_id(candidate)].append(candidate)
    stats: dict[str, dict[str, Any]] = {}
    for report_id, rows in grouped.items():
        routes = [route for row in rows for route in (row.get("qd_question_routes") or [])]
        focus = {str(route.get("focus") or "other") for route in routes}
        questions = {str(route.get("question_id") or "") for route in routes if route.get("question_id")}
        qd_ranks = [_safe_float(row.get("qd_pool_rank"), 0.0) for row in rows if _safe_float(row.get("qd_pool_rank"), 0.0) > 0]
        stats[report_id] = {
            "union_count": len(rows),
            "qd_count": sum(1 for row in rows if row.get("from_qd")),
            "baseline_count": sum(1 for row in rows if row.get("from_baseline")),
            "max_qd_rrf": max([_safe_float(row.get("qd_rrf_score"), 0.0) for row in rows] or [0.0]),
            "min_qd_rank": min(qd_ranks) if qd_ranks else None,
            "focus_count": len(focus),
            "question_count": len(questions),
        }
    return stats


def _domain_flags(domain: str) -> dict[str, float]:
    d = domain.lower()
    has = bool(d)
    factcheck = any(token in d for token in ("politifact", "factcheck", "snopes", "truthorfiction", "fullfact", "checkyourfact"))
    news = any(token in d for token in ("cnn", "nytimes", "washingtonpost", "wsj", "reuters", "apnews", "npr", "bbc", "foxnews", "nbcnews", "cbsnews", "abcnews", "usatoday", "theguardian", "latimes"))
    social = any(token in d for token in ("facebook", "twitter", "x.com", "reddit", "stackoverflow", "question-it", "overcoder", "quora", "forum"))
    gov_edu_org = d.endswith(".gov") or ".gov." in d or d.endswith(".edu") or ".edu." in d or d.endswith(".org") or ".org." in d
    blog = any(token in d for token in ("blogspot", "wordpress", "medium.com", "substack"))
    reference = any(token in d for token in ("wikipedia", "britannica", "investopedia", "jamanetwork", "heritage.org"))
    return {
        "domain_present": _bool(has),
        "domain_is_factcheck_like": _bool(factcheck),
        "domain_is_news_like": _bool(news),
        "domain_is_social_or_forum_like": _bool(social),
        "domain_is_gov_edu_org": _bool(gov_edu_org),
        "domain_is_blog_like": _bool(blog),
        "domain_is_reference_like": _bool(reference),
        "domain_is_unknown": _bool(not has),
    }


def _extract_domain(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_report") or {}
    raw = str(source.get("domain") or source.get("link") or "")
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return (parsed.netloc or parsed.path).lower().strip("/")


def _report_id(candidate: dict[str, Any]) -> str:
    value = candidate.get("report_id")
    if value is not None:
        return str(value)
    source = candidate.get("source_report") or {}
    return str(source.get("report_id") or _extract_domain(candidate) or "")


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("canonical_text") or canonicalize_sentence(str(candidate.get("text") or "")))


def _rank_inv(value: float | None) -> float:
    if value is None or value <= 0:
        return 0.0
    return 1.0 / float(value)


def _rank_norm(value: float | None, max_rank: float) -> float:
    if value is None or value <= 0:
        return 0.0
    return max(0.0, (float(max_rank) + 1.0 - float(value)) / float(max_rank))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0
