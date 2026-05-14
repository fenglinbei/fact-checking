"""Feature extraction for soft-label RL-MMR lambda policies."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample
from fact_checking.retrieval.mmr import maximal_marginal_relevance
from fact_checking.retrieval.text_utils import content_tokens_counter
from fact_checking.rl_mmr.sensitivity import (
    compute_pairwise_sim,
    jaccard,
    kendall_tau,
    mean_pairwise_sim,
)


DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

POOL_FEATURE_NAMES: list[str] = [
    "n_candidates",
    "log_n_candidates",
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "score_entropy",
    "score_gini",
    "top1_top2_gap",
    "top5_score_std",
    "mean_pairwise_sim",
    "max_pairwise_sim",
    "score_q10",
    "score_q50",
    "score_q90",
    "score_iqr",
]

INTERVENTIONAL_FEATURE_NAMES: list[str] = [
    "jaccard_0p10_0p70",
    "jaccard_0p30_0p70",
    "jaccard_0p10_0p90",
    "sens_0p30_0p70",
    "mean_rel_0p70_minus_0p30",
    "mean_red_0p70_minus_0p30",
    "kendall_tau_0p30_0p70",
    "n_selected_changes",
    "pool_redundancy",
    "selected_redundancy_0p30",
    "selected_redundancy_0p70",
    "selected_redundancy_0p90",
    "mean_rel_0p30",
    "mean_rel_0p70",
    "mean_rel_0p90",
]

CLAIM_FEATURE_NAMES: list[str] = [
    "claim_token_count",
    "claim_word_count",
    "entity_count",
    "number_count",
    "time_expression_count",
    "negation_flag",
    "comparison_flag",
    "superlative_flag",
    "percent_flag",
    "quote_flag",
]

SOFT_LABEL_FEATURE_NAMES: list[str] = (
    POOL_FEATURE_NAMES + INTERVENTIONAL_FEATURE_NAMES + CLAIM_FEATURE_NAMES
)


def _lambda_key(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _nearest_lambda(value: float, grid: tuple[float, ...]) -> float:
    if not grid:
        return float(value)
    return min((float(x) for x in grid), key=lambda x: abs(x - value))


def _entropy(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    shifted = scores - float(scores.max())
    expv = np.exp(shifted)
    total = float(expv.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    probs = expv / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _gini(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    sorted_v = np.sort(np.asarray(values, dtype=np.float64))
    total = float(sorted_v.sum())
    if abs(total) < 1e-12:
        return 0.0
    n = sorted_v.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_v) / (n * total)) - ((n + 1.0) / n))


def _max_pairwise_sim(indices: list[int], sim: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 2 or sim.size == 0:
        return 0.0
    sub = sim[np.ix_(idx, idx)]
    iu = np.triu_indices(sub.shape[0], k=1)
    if iu[0].size == 0:
        return 0.0
    return float(sub[iu].max())


def _claim_surface_features(claim: str) -> dict[str, float]:
    words = claim.split()
    claim_lower = claim.lower()
    entity_count = sum(1 for word in words if word[:1].isupper())
    number_matches = re.findall(r"\d+(?:[,.]\d+)*", claim)
    time_matches = re.findall(
        r"\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
        claim_lower,
    )
    return {
        "claim_token_count": float(content_tokens_counter(claim)[1]),
        "claim_word_count": float(len(words)),
        "entity_count": float(entity_count),
        "number_count": float(len(number_matches)),
        "time_expression_count": float(len(time_matches)),
        "negation_flag": float(bool(re.search(r"\b(?:no|not|never|n't|none|without|neither|nor)\b", claim_lower))),
        "comparison_flag": float(bool(re.search(
            r"\b(?:more|less|fewer|higher|lower|greater|smaller|larger|than|compared|increase|decrease)\b",
            claim_lower,
        ))),
        "superlative_flag": float(bool(re.search(
            r"\b(?:most|least|highest|lowest|largest|smallest|best|worst|first|last)\b",
            claim_lower,
        ))),
        "percent_flag": float(bool(re.search(r"[%]|percent|percentage", claim_lower))),
        "quote_flag": float(bool(re.search(r"""["'“”‘’]""", claim))),
    }


def _selection_stats(
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    lambda_grid: tuple[float, ...],
    top_k: int,
) -> dict[str, Any]:
    n = int(hybrid_scores.shape[0])
    if n == 0:
        return {
            "selections": {},
            "mean_rel": {},
            "selected_redundancy": {},
            "sim": np.zeros((0, 0), dtype=np.float32),
        }

    effective_k = min(int(top_k), n)
    sim = compute_pairwise_sim(chunk_emb)
    selections: dict[float, list[int]] = {}
    mean_rel: dict[float, float] = {}
    selected_redundancy: dict[float, float] = {}
    for lam in lambda_grid:
        lam_f = float(lam)
        selected = maximal_marginal_relevance(
            query_scores=hybrid_scores,
            sentence_vectors=chunk_emb,
            top_k=effective_k,
            lambda_weight=lam_f,
        )
        selections[lam_f] = [int(i) for i in selected]
        mean_rel[lam_f] = float(hybrid_scores[selected].mean()) if selected else 0.0
        selected_redundancy[lam_f] = mean_pairwise_sim(selected, sim)
    return {
        "selections": selections,
        "mean_rel": mean_rel,
        "selected_redundancy": selected_redundancy,
        "sim": sim,
    }


def extract_soft_label_features(
    chunk_sample: ChunkMMRSample,
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    top_k: int = 5,
) -> dict[str, float | list[int]]:
    """Return tabular features for a single ``ChunkMMRSample``.

    The caller is expected to pass the same ``hybrid_scores`` and ``chunk_emb``
    returned by ``fact_checking.build.candidates.compute_hybrid_scores`` so the
    policy sees the same retrieval geometry as the production MMR phase.
    """
    grid = tuple(float(x) for x in lambda_grid)
    scores = np.asarray(hybrid_scores, dtype=np.float32).reshape(-1)
    emb = np.asarray(chunk_emb, dtype=np.float32)
    n = int(scores.shape[0])

    features: dict[str, float | list[int]] = {name: 0.0 for name in SOFT_LABEL_FEATURE_NAMES}
    features.update(_claim_surface_features(str(chunk_sample.claim)))

    features["n_candidates"] = float(n)
    features["log_n_candidates"] = float(np.log1p(n))
    if n == 0:
        return features

    sorted_scores = np.sort(scores)[::-1]
    top5 = sorted_scores[: min(5, n)]
    q10, q50, q90 = np.quantile(scores, [0.10, 0.50, 0.90])

    sim = compute_pairwise_sim(emb)
    pool_indices = list(range(n))
    features.update({
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_entropy": _entropy(scores),
        "score_gini": _gini(scores),
        "top1_top2_gap": float(sorted_scores[0] - sorted_scores[1]) if n >= 2 else 0.0,
        "top5_score_std": float(top5.std()) if top5.size >= 2 else 0.0,
        "mean_pairwise_sim": mean_pairwise_sim(pool_indices, sim),
        "max_pairwise_sim": _max_pairwise_sim(pool_indices, sim),
        "score_q10": float(q10),
        "score_q50": float(q50),
        "score_q90": float(q90),
        "score_iqr": float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25)),
    })

    stats = _selection_stats(scores, emb, grid, top_k)
    selections: dict[float, list[int]] = stats["selections"]
    mean_rel: dict[float, float] = stats["mean_rel"]
    selected_redundancy: dict[float, float] = stats["selected_redundancy"]

    lam_01 = _nearest_lambda(0.1, grid)
    lam_03 = _nearest_lambda(0.3, grid)
    lam_07 = _nearest_lambda(0.7, grid)
    lam_09 = _nearest_lambda(0.9, grid)

    s01 = selections.get(lam_01, [])
    s03 = selections.get(lam_03, [])
    s07 = selections.get(lam_07, [])
    s09 = selections.get(lam_09, [])
    unique_selection_sets = {tuple(selections[lam]) for lam in sorted(selections)}

    features.update({
        "jaccard_0p10_0p70": float(jaccard(s01, s07)),
        "jaccard_0p30_0p70": float(jaccard(s03, s07)),
        "jaccard_0p10_0p90": float(jaccard(s01, s09)),
        "sens_0p30_0p70": float(1.0 - jaccard(s03, s07)),
        "mean_rel_0p70_minus_0p30": float(mean_rel.get(lam_07, 0.0) - mean_rel.get(lam_03, 0.0)),
        "mean_red_0p70_minus_0p30": float(
            selected_redundancy.get(lam_07, 0.0) - selected_redundancy.get(lam_03, 0.0)
        ),
        "kendall_tau_0p30_0p70": float(kendall_tau(s03, s07)),
        "n_selected_changes": float(max(0, len(unique_selection_sets) - 1)),
        "pool_redundancy": mean_pairwise_sim(pool_indices, sim),
        "selected_redundancy_0p30": float(selected_redundancy.get(lam_03, 0.0)),
        "selected_redundancy_0p70": float(selected_redundancy.get(lam_07, 0.0)),
        "selected_redundancy_0p90": float(selected_redundancy.get(lam_09, 0.0)),
        "mean_rel_0p30": float(mean_rel.get(lam_03, 0.0)),
        "mean_rel_0p70": float(mean_rel.get(lam_07, 0.0)),
        "mean_rel_0p90": float(mean_rel.get(lam_09, 0.0)),
        f"S_{_lambda_key(lam_03)}": s03,
        f"S_{_lambda_key(lam_07)}": s07,
    })
    return features


def feature_dict_to_vector(
    features: dict[str, float | list[int]],
    feature_names: list[str] | None = None,
) -> np.ndarray:
    names = SOFT_LABEL_FEATURE_NAMES if feature_names is None else feature_names
    return np.array([float(features.get(name, 0.0)) for name in names], dtype=np.float32)
