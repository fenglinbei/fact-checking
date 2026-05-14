"""Sensitivity-gated MMR: numpy-only helpers.

Stateless implementation of the features and gating decision described in
``todo/RL_MMR_ordered_experiment_plan.md`` §5. All functions are pure so they
are trivial to unit-test and reuse from future experiments (soft-label, DPO).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fact_checking.retrieval.mmr import maximal_marginal_relevance


GATE_LOW = "low_lambda"
GATE_BASE = "base_lambda"
GATE_FLOOR_BLOCKED = "relevance_floor_blocked"
GATE_TRIVIAL = "trivial_pool"


# ---------------------------------------------------------------------------
# Basic similarity / set helpers
# ---------------------------------------------------------------------------


def compute_pairwise_sim(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. ``emb`` should be (near) L2-normalized."""
    if emb.ndim != 2 or emb.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return (emb @ emb.T).astype(np.float32, copy=False)


def mean_pairwise_sim(indices: list[int] | np.ndarray, sim: np.ndarray) -> float:
    """Mean of the upper-triangle of ``sim[indices][:, indices]``.

    Returns 0.0 when fewer than 2 indices are supplied.
    """
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size < 2 or sim.size == 0:
        return 0.0
    sub = sim[np.ix_(idx, idx)]
    iu = np.triu_indices(sub.shape[0], k=1)
    if iu[0].size == 0:
        return 0.0
    return float(sub[iu].mean())


def jaccard(a: list[int] | np.ndarray, b: list[int] | np.ndarray) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def kendall_tau(rank_a: list[int] | np.ndarray, rank_b: list[int] | np.ndarray) -> float:
    """Kendall tau on the intersection of two ordered selections.

    Items only in one of the two rankings contribute neither concordant nor
    discordant pairs. Returns 1.0 when fewer than 2 shared items.
    """
    ra = list(int(x) for x in rank_a)
    rb = list(int(x) for x in rank_b)
    shared = [x for x in ra if x in rb]
    if len(shared) < 2:
        return 1.0
    pos_a = {item: i for i, item in enumerate(ra)}
    pos_b = {item: i for i, item in enumerate(rb)}
    concordant = 0
    discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            x, y = shared[i], shared[j]
            da = pos_a[x] - pos_a[y]
            db = pos_b[x] - pos_b[y]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def sensitivity_features(
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    *,
    top_k: int,
    lambdas: tuple[float, float, float] = (0.3, 0.7, 1.0),
    pool_redundancy_topn: int | None = 32,
) -> dict[str, Any]:
    """Compute sensitivity / redundancy / score-distribution features for one sample.

    Args:
        hybrid_scores: ``[N]`` query relevance (the same hybrid score that the
            production MMR phase uses).
        chunk_emb: ``[N, D]``, must be L2-normalized for cosine similarity.
        top_k: number of evidence items to select.
        lambdas: (lambda_low, lambda_base, lambda_probe). Order matters: the
            returned ``S_low / S_base / S_probe`` follow this order.
        pool_redundancy_topn: cap on the pool used for ``pool_redundancy``.
            ``None`` uses the full pool.
    """
    n = int(hybrid_scores.shape[0])
    lambda_low, lambda_base, lambda_probe = lambdas

    if n == 0:
        return _empty_features(lambdas, top_k)

    effective_k = min(int(top_k), n)
    s_low = maximal_marginal_relevance(hybrid_scores, chunk_emb, effective_k, lambda_low)
    s_base = maximal_marginal_relevance(hybrid_scores, chunk_emb, effective_k, lambda_base)
    s_probe = maximal_marginal_relevance(hybrid_scores, chunk_emb, effective_k, lambda_probe)

    sens_low_base = 1.0 - jaccard(s_low, s_base)
    sens_base_probe = 1.0 - jaccard(s_base, s_probe)
    sens_low_probe = 1.0 - jaccard(s_low, s_probe)

    sim = compute_pairwise_sim(chunk_emb)

    pool_indices: list[int]
    if pool_redundancy_topn is not None and pool_redundancy_topn > 0 and n > pool_redundancy_topn:
        pool_indices = np.argsort(-hybrid_scores)[:pool_redundancy_topn].tolist()
    else:
        pool_indices = list(range(n))
    pool_redundancy = mean_pairwise_sim(pool_indices, sim)
    max_pool_redundancy = _max_pairwise_sim(pool_indices, sim)

    selected_redundancy_low = mean_pairwise_sim(s_low, sim)
    selected_redundancy_base = mean_pairwise_sim(s_base, sim)

    mean_rel_low = float(hybrid_scores[s_low].mean()) if s_low else 0.0
    mean_rel_base = float(hybrid_scores[s_base].mean()) if s_base else 0.0

    top1_change = int(bool(s_low and s_base and s_low[0] != s_base[0]))
    overlap_size = len(set(s_low) & set(s_base))
    kendall_low_base = kendall_tau(s_low, s_base)

    score_entropy, top1_top2_gap, top5_score_std = _score_distribution_stats(hybrid_scores)

    return {
        "n_candidates": n,
        "top_k": effective_k,
        "lambda_low": float(lambda_low),
        "lambda_base": float(lambda_base),
        "lambda_probe": float(lambda_probe),
        "S_low": [int(i) for i in s_low],
        "S_base": [int(i) for i in s_base],
        "S_probe": [int(i) for i in s_probe],
        "sens_low_base": float(sens_low_base),
        "sens_base_probe": float(sens_base_probe),
        "sens_low_probe": float(sens_low_probe),
        "pool_redundancy": float(pool_redundancy),
        "max_pool_redundancy": float(max_pool_redundancy),
        "selected_redundancy_low": float(selected_redundancy_low),
        "selected_redundancy_base": float(selected_redundancy_base),
        "mean_rel_low": mean_rel_low,
        "mean_rel_base": mean_rel_base,
        "kendall_low_base": float(kendall_low_base),
        "top1_change": top1_change,
        "overlap_size": int(overlap_size),
        "score_entropy": float(score_entropy),
        "top1_top2_gap": float(top1_top2_gap),
        "top5_score_std": float(top5_score_std),
    }


def _empty_features(lambdas: tuple[float, float, float], top_k: int) -> dict[str, Any]:
    lambda_low, lambda_base, lambda_probe = lambdas
    return {
        "n_candidates": 0,
        "top_k": int(top_k),
        "lambda_low": float(lambda_low),
        "lambda_base": float(lambda_base),
        "lambda_probe": float(lambda_probe),
        "S_low": [],
        "S_base": [],
        "S_probe": [],
        "sens_low_base": 0.0,
        "sens_base_probe": 0.0,
        "sens_low_probe": 0.0,
        "pool_redundancy": 0.0,
        "max_pool_redundancy": 0.0,
        "selected_redundancy_low": 0.0,
        "selected_redundancy_base": 0.0,
        "mean_rel_low": 0.0,
        "mean_rel_base": 0.0,
        "kendall_low_base": 1.0,
        "top1_change": 0,
        "overlap_size": 0,
        "score_entropy": 0.0,
        "top1_top2_gap": 0.0,
        "top5_score_std": 0.0,
    }


def _max_pairwise_sim(indices: list[int], sim: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 2 or sim.size == 0:
        return 0.0
    sub = sim[np.ix_(idx, idx)]
    iu = np.triu_indices(sub.shape[0], k=1)
    if iu[0].size == 0:
        return 0.0
    return float(sub[iu].max())


def _score_distribution_stats(scores: np.ndarray) -> tuple[float, float, float]:
    n = int(scores.shape[0])
    if n == 0:
        return 0.0, 0.0, 0.0
    shifted = scores - scores.max()
    expv = np.exp(shifted)
    Z = float(expv.sum())
    if Z <= 0.0:
        entropy = 0.0
    else:
        p = expv / Z
        p_safe = np.where(p > 0, p, 1.0)
        entropy = float(-(p * np.log(p_safe)).sum())
    sorted_scores = np.sort(scores)[::-1]
    if n >= 2:
        top1_top2_gap = float(sorted_scores[0] - sorted_scores[1])
    else:
        top1_top2_gap = 0.0
    top5 = sorted_scores[: min(5, n)]
    top5_score_std = float(top5.std()) if top5.size >= 2 else 0.0
    return entropy, top1_top2_gap, top5_score_std


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def relevance_floor_ok(
    hybrid_scores: np.ndarray,
    s_low: list[int],
    s_base: list[int],
    *,
    mode: str = "mean_delta",
    epsilon: float = 0.05,
    p_floor: float = 0.50,
) -> bool:
    """Return True when switching to ``lambda_low`` does not erode relevance.

    Modes:
        ``mean_delta``  — ``mean_rel(S_base) - mean_rel(S_low) <= epsilon``.
        ``min_quantile`` — ``min Rel(S_low) >= percentile(H_pool, p_floor*100)``.
    """
    if not s_low or not s_base:
        return False

    if mode == "mean_delta":
        mean_low = float(hybrid_scores[s_low].mean())
        mean_base = float(hybrid_scores[s_base].mean())
        return (mean_base - mean_low) <= float(epsilon)

    if mode == "min_quantile":
        threshold = float(np.quantile(hybrid_scores, float(p_floor)))
        min_low = float(hybrid_scores[s_low].min())
        return min_low >= threshold

    raise ValueError(f"Unknown relevance_floor mode: {mode!r}")


def gating_decision(
    feats: dict[str, Any],
    hybrid_scores: np.ndarray,
    *,
    gating_mode: str = "conservative",
    theta_s: float = 0.4,
    theta_r: float = 0.4,
    lambda_low: float = 0.3,
    lambda_base: float = 0.7,
    min_n_candidates_for_gate: int = 2,
    relevance_floor_kwargs: dict[str, Any] | None = None,
) -> tuple[float, str, dict[str, Any]]:
    """Apply the gating rule and return ``(chosen_lambda, gate_label, extras)``."""
    extras: dict[str, Any] = {
        "theta_s": float(theta_s),
        "theta_r": float(theta_r),
        "lambda_low_cfg": float(lambda_low),
        "lambda_base_cfg": float(lambda_base),
        "gating_mode": str(gating_mode),
    }

    n = int(feats.get("n_candidates", 0))
    s_low = list(feats.get("S_low", []))
    s_base = list(feats.get("S_base", []))

    if n < int(min_n_candidates_for_gate) or not s_low or not s_base:
        extras["relevance_floor_ok"] = None
        return float(lambda_base), GATE_TRIVIAL, extras

    sens = float(feats.get("sens_low_base", 0.0))
    pool_red = float(feats.get("pool_redundancy", 0.0))

    if not (sens >= float(theta_s) and pool_red >= float(theta_r)):
        extras["relevance_floor_ok"] = None
        return float(lambda_base), GATE_BASE, extras

    if gating_mode == "basic":
        extras["relevance_floor_ok"] = None
        return float(lambda_low), GATE_LOW, extras

    if gating_mode == "conservative":
        rf_kwargs = dict(relevance_floor_kwargs or {})
        ok = relevance_floor_ok(hybrid_scores, s_low, s_base, **rf_kwargs)
        extras["relevance_floor_ok"] = bool(ok)
        if ok:
            return float(lambda_low), GATE_LOW, extras
        return float(lambda_base), GATE_FLOOR_BLOCKED, extras

    raise ValueError(f"Unknown gating_mode: {gating_mode!r}")
