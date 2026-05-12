from __future__ import annotations

import re

import numpy as np

from fact_checking.build.candidates import PreMMRSample, minmax_scale
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)

FEATURE_NAMES: list[str] = [
    # F_c (claim)
    "claim_token_count",
    "claim_emb_norm",
    "claim_entity_density",
    "claim_has_number",
    "claim_word_count",
    # F_C (candidate pool)
    "n_sentences",
    "n_unique_domains",
    "n_unique_reports",
    "score_mean",
    "score_std",
    "score_max",
    "score_min",
    "score_entropy",
    "mean_pairwise_sim",
    "score_concentration",
    # F_{c,C} (interaction)
    "top1_score",
    "top1_top5_gap",
    "top5_mean",
    "score_gini",
    "dense_lexical_corr",
    "max_sim_to_claim_top10",
    "min_sim_to_claim_top10",
    "diversity_top10",
    # Additional claim surface features
    "claim_has_percent",
    "claim_has_year",
    "claim_has_quote",
    "claim_has_negation",
    "claim_has_comparison",
    "claim_has_superlative",
    # Hybrid score distribution
    "score_q10",
    "score_q25",
    "score_q50",
    "score_q75",
    "score_q90",
    "score_iqr",
    "top1_top2_gap",
    "top3_mean",
    "top10_mean",
    "top10_mass",
    # Dense / lexical / BM25 agreement
    "dense_mean",
    "dense_std",
    "dense_top1",
    "lexical_mean",
    "lexical_std",
    "lexical_top1",
    "bm25_mean",
    "bm25_std",
    "bm25_top1",
    "dense_bm25_corr",
    "lexical_bm25_corr",
    "dense_lexical_top10_overlap",
    "dense_bm25_top10_overlap",
    "lexical_bm25_top10_overlap",
    # Redundancy among high-scoring candidates
    "top10_pairwise_std",
    "top10_pairwise_max",
    "top10_pairwise_min",
    "top10_sim_gt_0_80",
    "top10_sim_gt_0_90",
    "top20_pairwise_mean",
    # Approximate MMR sensitivity on top hybrid candidates
    "mmr_rel_lambda_0_00",
    "mmr_rel_lambda_0_30",
    "mmr_rel_lambda_0_50",
    "mmr_rel_lambda_0_70",
    "mmr_rel_lambda_1_00",
    "mmr_sim_lambda_0_00",
    "mmr_sim_lambda_0_30",
    "mmr_sim_lambda_0_50",
    "mmr_sim_lambda_0_70",
    "mmr_sim_lambda_1_00",
    "mmr_overlap_lambda_0_00_1_00",
    "mmr_overlap_lambda_0_30_0_70",
    "mmr_rel_range",
    "mmr_sim_range",
]


def _compute_hybrid_scores(
    pre: PreMMRSample,
    alpha_dense: float = 0.70,
    alpha_lexical: float = 0.20,
    alpha_bm25: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sent_texts = [d["text"] for d in pre.sentences]
    n = len(sent_texts)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, empty

    dense_scores = pre.sent_emb @ pre.claim_emb

    q_ctr, q_len = content_tokens_counter(pre.claim)
    lexical_scores = np.empty(n, dtype=np.float32)
    bm25_scores = np.empty(n, dtype=np.float32)
    for j, s in enumerate(sent_texts):
        s_ctr, s_len = content_tokens_counter(s)
        lexical_scores[j] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
        bm25_scores[j] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

    dense_scaled = minmax_scale(dense_scores)
    lexical_scaled = minmax_scale(lexical_scores)
    bm25_scaled = minmax_scale(bm25_scores)
    hybrid = alpha_dense * dense_scaled + alpha_lexical * lexical_scaled + alpha_bm25 * bm25_scaled
    return hybrid, dense_scaled, lexical_scaled, bm25_scaled


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2 or a.std() <= 1e-8 or b.std() <= 1e-8:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if np.isnan(value) else value


def _gini(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_v) / (n * np.sum(sorted_v)) - (n + 1) / n))


def _entropy(scores: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    shifted = scores - scores.max()
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _topk_overlap(scores_a: np.ndarray, scores_b: np.ndarray, k: int = 10) -> float:
    n = min(len(scores_a), len(scores_b), k)
    if n <= 0:
        return 0.0
    top_a = set(np.argsort(-scores_a)[:n].tolist())
    top_b = set(np.argsort(-scores_b)[:n].tolist())
    return float(len(top_a & top_b) / n)


def _pairwise_values(emb: np.ndarray) -> np.ndarray:
    n = len(emb)
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    pw = emb @ emb.T
    mask = ~np.eye(n, dtype=bool)
    return pw[mask].astype(np.float32, copy=False)


def _mean_pairwise(emb: np.ndarray) -> float:
    values = _pairwise_values(emb)
    return float(values.mean()) if values.size else 0.0


def _mmr_select_from_similarity(
    query_scores: np.ndarray,
    similarity: np.ndarray,
    top_k: int,
    lambda_weight: float,
) -> list[int]:
    n_items = int(query_scores.shape[0])
    if n_items == 0 or top_k <= 0:
        return []
    if top_k >= n_items:
        return [int(i) for i in np.argsort(-query_scores)]

    selected: list[int] = [int(np.argmax(query_scores))]
    candidate_mask = np.ones(n_items, dtype=bool)
    candidate_mask[selected[0]] = False
    max_sim_to_selected = np.zeros(n_items, dtype=np.float32)

    while len(selected) < top_k:
        last_selected = selected[-1]
        np.maximum(max_sim_to_selected, similarity[last_selected, :], out=max_sim_to_selected)
        mmr_scores = lambda_weight * query_scores - (1.0 - lambda_weight) * max_sim_to_selected
        mmr_scores[~candidate_mask] = -np.inf
        best_idx = int(np.argmax(mmr_scores))
        selected.append(best_idx)
        candidate_mask[best_idx] = False

    return selected


def _selection_overlap(a: list[int], b: list[int]) -> float:
    denom = min(len(a), len(b))
    if denom <= 0:
        return 0.0
    return float(len(set(a) & set(b)) / denom)


def extract_features(
    pre: PreMMRSample,
    alpha_dense: float = 0.70,
    alpha_lexical: float = 0.20,
    alpha_bm25: float = 0.10,
) -> np.ndarray:
    n = len(pre.sentences)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    hybrid, dense_scaled, lexical_scaled, bm25_scaled = _compute_hybrid_scores(
        pre, alpha_dense, alpha_lexical, alpha_bm25
    )

    words = pre.claim.split()
    claim_lower = pre.claim.lower()
    claim_token_count = float(content_tokens_counter(pre.claim)[1])
    claim_emb_norm = float(np.linalg.norm(pre.claim_emb))
    upper_words = sum(1 for w in words if w[0].isupper()) if words else 0
    claim_entity_density = upper_words / max(len(words), 1)
    claim_has_number = float(bool(re.search(r"\d", pre.claim)))
    claim_word_count = float(len(words))
    claim_has_percent = float(bool(re.search(r"[%]|percent|percentage", claim_lower)))
    claim_has_year = float(bool(re.search(r"\b(?:19|20)\d{2}\b", pre.claim)))
    claim_has_quote = float(bool(re.search(r"""["'“”‘’]""", pre.claim)))
    claim_has_negation = float(bool(re.search(r"\b(?:no|not|never|n't|none|without|neither|nor)\b", claim_lower)))
    claim_has_comparison = float(bool(re.search(
        r"\b(?:more|less|fewer|higher|lower|greater|smaller|larger|than|compared|increase|decrease)\b",
        claim_lower,
    )))
    claim_has_superlative = float(bool(re.search(
        r"\b(?:most|least|highest|lowest|largest|smallest|best|worst|first|last)\b",
        claim_lower,
    )))

    domains = {d.get("domain", "") for d in pre.sentences if d.get("domain")}
    reports = {d.get("report_id", "") for d in pre.sentences}

    sorted_scores = np.sort(hybrid)[::-1]
    top3_sum = float(sorted_scores[:3].sum())
    total_sum = float(sorted_scores.sum()) + 1e-12

    top10_idx = np.argsort(-hybrid)[:min(10, n)]
    top10_emb = pre.sent_emb[top10_idx]
    claim_sims_top10 = top10_emb @ pre.claim_emb
    top10_pairwise = _pairwise_values(top10_emb)
    diversity_top10 = float(top10_pairwise.mean()) if top10_pairwise.size else 0.0
    top10_pairwise_std = float(top10_pairwise.std()) if top10_pairwise.size else 0.0
    top10_pairwise_max = float(top10_pairwise.max()) if top10_pairwise.size else 0.0
    top10_pairwise_min = float(top10_pairwise.min()) if top10_pairwise.size else 0.0
    top10_sim_gt_0_80 = float((top10_pairwise > 0.80).mean()) if top10_pairwise.size else 0.0
    top10_sim_gt_0_90 = float((top10_pairwise > 0.90).mean()) if top10_pairwise.size else 0.0

    top20_idx = np.argsort(-hybrid)[:min(20, n)]
    top20_pairwise_mean = _mean_pairwise(pre.sent_emb[top20_idx])

    top5_scores = sorted_scores[:min(5, n)]
    top10_scores = sorted_scores[:min(10, n)]
    top1_top5_gap = float(sorted_scores[0] - top5_scores[-1]) if len(top5_scores) > 1 else 0.0
    top1_top2_gap = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0

    score_q10, score_q25, score_q50, score_q75, score_q90 = np.quantile(
        hybrid,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )
    dense_lexical_corr = _safe_corr(dense_scaled, lexical_scaled)
    dense_bm25_corr = _safe_corr(dense_scaled, bm25_scaled)
    lexical_bm25_corr = _safe_corr(lexical_scaled, bm25_scaled)

    if n > 10:
        sample_idx = np.random.default_rng(42).choice(n, size=min(n, 200), replace=False)
        sample_emb = pre.sent_emb[sample_idx]
        mean_pw_sim = _mean_pairwise(sample_emb)
    else:
        mean_pw_sim = _mean_pairwise(pre.sent_emb)

    mmr_lambdas = [0.00, 0.30, 0.50, 0.70, 1.00]
    mmr_top_idx = np.argsort(-hybrid)[:min(30, n)]
    mmr_scores = hybrid[mmr_top_idx]
    mmr_emb = pre.sent_emb[mmr_top_idx]
    mmr_sim = mmr_emb @ mmr_emb.T if len(mmr_top_idx) else np.zeros((0, 0), dtype=np.float32)
    mmr_top_k = min(10, len(mmr_top_idx))
    mmr_selected: dict[float, list[int]] = {}
    mmr_rel_values: list[float] = []
    mmr_sim_values: list[float] = []
    for lam in mmr_lambdas:
        selected = _mmr_select_from_similarity(mmr_scores, mmr_sim, mmr_top_k, lam)
        mmr_selected[lam] = selected
        if selected:
            mmr_rel_values.append(float(mmr_scores[selected].mean()))
            mmr_sim_values.append(_mean_pairwise(mmr_emb[selected]))
        else:
            mmr_rel_values.append(0.0)
            mmr_sim_values.append(0.0)

    features = np.array([
        claim_token_count,
        claim_emb_norm,
        claim_entity_density,
        claim_has_number,
        claim_word_count,
        float(n),
        float(len(domains)),
        float(len(reports)),
        float(hybrid.mean()),
        float(hybrid.std()),
        float(hybrid.max()),
        float(hybrid.min()),
        _entropy(hybrid),
        mean_pw_sim,
        top3_sum / total_sum,
        float(sorted_scores[0]),
        top1_top5_gap,
        float(top5_scores.mean()),
        _gini(hybrid),
        dense_lexical_corr,
        float(claim_sims_top10.max()),
        float(claim_sims_top10.min()),
        diversity_top10,
        claim_has_percent,
        claim_has_year,
        claim_has_quote,
        claim_has_negation,
        claim_has_comparison,
        claim_has_superlative,
        float(score_q10),
        float(score_q25),
        float(score_q50),
        float(score_q75),
        float(score_q90),
        float(score_q75 - score_q25),
        top1_top2_gap,
        float(sorted_scores[:min(3, n)].mean()),
        float(top10_scores.mean()),
        float(top10_scores.sum() / total_sum),
        float(dense_scaled.mean()),
        float(dense_scaled.std()),
        float(dense_scaled.max()),
        float(lexical_scaled.mean()),
        float(lexical_scaled.std()),
        float(lexical_scaled.max()),
        float(bm25_scaled.mean()),
        float(bm25_scaled.std()),
        float(bm25_scaled.max()),
        dense_bm25_corr,
        lexical_bm25_corr,
        _topk_overlap(dense_scaled, lexical_scaled, 10),
        _topk_overlap(dense_scaled, bm25_scaled, 10),
        _topk_overlap(lexical_scaled, bm25_scaled, 10),
        top10_pairwise_std,
        top10_pairwise_max,
        top10_pairwise_min,
        top10_sim_gt_0_80,
        top10_sim_gt_0_90,
        top20_pairwise_mean,
        *mmr_rel_values,
        *mmr_sim_values,
        _selection_overlap(mmr_selected[0.00], mmr_selected[1.00]),
        _selection_overlap(mmr_selected[0.30], mmr_selected[0.70]),
        float(max(mmr_rel_values) - min(mmr_rel_values)),
        float(max(mmr_sim_values) - min(mmr_sim_values)),
    ], dtype=np.float32)

    return features


def extract_features_batch(
    pre_samples: list[PreMMRSample],
    alpha_dense: float = 0.70,
    alpha_lexical: float = 0.20,
    alpha_bm25: float = 0.10,
) -> np.ndarray:
    return np.stack([
        extract_features(pre, alpha_dense, alpha_lexical, alpha_bm25)
        for pre in pre_samples
    ])
