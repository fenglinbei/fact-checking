from __future__ import annotations

import re
from dataclasses import dataclass

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


def extract_features(
    pre: PreMMRSample,
    alpha_dense: float = 0.70,
    alpha_lexical: float = 0.20,
    alpha_bm25: float = 0.10,
) -> np.ndarray:
    n = len(pre.sentences)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    hybrid, dense_scaled, lexical_scaled, _ = _compute_hybrid_scores(
        pre, alpha_dense, alpha_lexical, alpha_bm25
    )

    words = pre.claim.split()
    claim_token_count = float(content_tokens_counter(pre.claim)[1])
    claim_emb_norm = float(np.linalg.norm(pre.claim_emb))
    upper_words = sum(1 for w in words if w[0].isupper()) if words else 0
    claim_entity_density = upper_words / max(len(words), 1)
    claim_has_number = float(bool(re.search(r"\d", pre.claim)))
    claim_word_count = float(len(words))

    domains = {d.get("domain", "") for d in pre.sentences if d.get("domain")}
    reports = {d.get("report_id", "") for d in pre.sentences}

    sorted_scores = np.sort(hybrid)[::-1]
    top3_sum = float(sorted_scores[:3].sum())
    total_sum = float(sorted_scores.sum()) + 1e-12

    top10_idx = np.argsort(-hybrid)[:min(10, n)]
    top10_emb = pre.sent_emb[top10_idx]
    claim_sims_top10 = top10_emb @ pre.claim_emb
    if len(top10_idx) > 1:
        pw = top10_emb @ top10_emb.T
        mask = ~np.eye(len(top10_idx), dtype=bool)
        diversity_top10 = float(pw[mask].mean())
    else:
        diversity_top10 = 0.0

    top5_scores = sorted_scores[:min(5, n)]
    top1_top5_gap = float(sorted_scores[0] - top5_scores[-1]) if len(top5_scores) > 1 else 0.0

    if n > 1 and dense_scaled.std() > 1e-8 and lexical_scaled.std() > 1e-8:
        dense_lexical_corr = float(np.corrcoef(dense_scaled, lexical_scaled)[0, 1])
        if np.isnan(dense_lexical_corr):
            dense_lexical_corr = 0.0
    else:
        dense_lexical_corr = 0.0

    if n > 10:
        sample_idx = np.random.default_rng(42).choice(n, size=min(n, 200), replace=False)
        sample_emb = pre.sent_emb[sample_idx]
        pw_sample = sample_emb @ sample_emb.T
        mask = ~np.eye(len(sample_idx), dtype=bool)
        mean_pw_sim = float(pw_sample[mask].mean())
    else:
        pw = pre.sent_emb @ pre.sent_emb.T
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            mean_pw_sim = float(pw[mask].mean())
        else:
            mean_pw_sim = 0.0

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
