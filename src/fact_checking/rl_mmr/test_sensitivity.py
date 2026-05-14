from __future__ import annotations

import numpy as np

from fact_checking.rl_mmr.sensitivity import (
    GATE_BASE,
    GATE_FLOOR_BLOCKED,
    GATE_LOW,
    GATE_TRIVIAL,
    gating_decision,
    jaccard,
    kendall_tau,
    mean_pairwise_sim,
    relevance_floor_ok,
    sensitivity_features,
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return x / norms


def test_jaccard_basic_cases() -> None:
    assert jaccard([], []) == 1.0
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert abs(jaccard([1, 2, 3], [2, 3, 4]) - (2 / 4)) < 1e-9


def test_mean_pairwise_sim_safe_on_singleton() -> None:
    sim = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
    assert mean_pairwise_sim([0], sim) == 0.0
    assert abs(mean_pairwise_sim([0, 1], sim) - 0.2) < 1e-6
    assert mean_pairwise_sim([], np.zeros((0, 0))) == 0.0


def test_kendall_tau_returns_one_on_singleton_intersection() -> None:
    assert kendall_tau([1, 2, 3], [4, 5]) == 1.0
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_uniform_pool_gives_zero_sensitivity_and_base_gate() -> None:
    """All candidates identical → MMR returns same set regardless of λ → gate = base."""
    n, d = 6, 4
    emb = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    emb = _l2_normalize(emb)
    hybrid = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)

    feats = sensitivity_features(hybrid, emb, top_k=3, lambdas=(0.3, 0.7, 1.0))
    assert feats["sens_low_base"] == 0.0
    assert feats["sens_base_probe"] == 0.0

    chosen, gate, extras = gating_decision(
        feats,
        hybrid,
        gating_mode="basic",
        theta_s=0.1,
        theta_r=0.1,
        lambda_low=0.3,
        lambda_base=0.7,
    )
    assert chosen == 0.7
    assert gate == GATE_BASE


def test_high_sens_high_redundancy_triggers_low_lambda_under_basic() -> None:
    """Cluster A: 3 redundant high-relevance items.
    Cluster B: 2 orthogonal low-relevance items.

    λ=1.0 picks top-3 by score → all cluster A.
    λ=0.7 keeps cluster A because the score gap dominates redundancy.
    λ=0.3 prefers diversity, so the second pick crosses to cluster B.
    """
    emb = np.array(
        [
            [1.00, 0.00],
            [0.99, 0.05],
            [0.97, 0.10],
            [0.00, 1.00],
            [0.05, 0.99],
        ],
        dtype=np.float32,
    )
    emb = _l2_normalize(emb)
    hybrid = np.array([0.95, 0.93, 0.91, 0.40, 0.38], dtype=np.float32)

    feats = sensitivity_features(hybrid, emb, top_k=3, lambdas=(0.3, 0.7, 1.0))
    assert sorted(feats["S_probe"]) == [0, 1, 2]
    assert sorted(feats["S_base"]) == [0, 1, 2]
    assert any(idx >= 3 for idx in feats["S_low"])
    assert feats["sens_low_base"] > 0.0
    assert feats["pool_redundancy"] > 0.0

    chosen, gate, _ = gating_decision(
        feats,
        hybrid,
        gating_mode="basic",
        theta_s=0.2,
        theta_r=0.2,
        lambda_low=0.3,
        lambda_base=0.7,
    )
    assert chosen == 0.3
    assert gate == GATE_LOW


def test_conservative_mode_blocks_low_lambda_when_relevance_drops() -> None:
    """Even when sens & redundancy are high, gate must not flip if S_low's
    mean relevance is much worse than S_base's."""
    emb = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.05],
            [0.98, 0.07],
            [0.0, 1.0],
            [0.05, 0.99],
            [0.07, 0.98],
        ],
        dtype=np.float32,
    )
    emb = _l2_normalize(emb)
    # Cluster A very relevant, cluster B basically irrelevant.
    hybrid = np.array([0.99, 0.97, 0.95, 0.05, 0.04, 0.03], dtype=np.float32)

    feats = sensitivity_features(hybrid, emb, top_k=3, lambdas=(0.3, 0.7, 1.0))
    # Sanity: λ=0.3 still pulls in cluster B (high diversity weight).
    assert any(idx >= 3 for idx in feats["S_low"])
    # mean_rel_low should be much lower than mean_rel_base.
    assert feats["mean_rel_base"] - feats["mean_rel_low"] > 0.2

    chosen, gate, extras = gating_decision(
        feats,
        hybrid,
        gating_mode="conservative",
        theta_s=0.1,
        theta_r=0.1,
        lambda_low=0.3,
        lambda_base=0.7,
        relevance_floor_kwargs={"mode": "mean_delta", "epsilon": 0.05, "p_floor": 0.5},
    )
    assert chosen == 0.7
    assert gate == GATE_FLOOR_BLOCKED
    assert extras["relevance_floor_ok"] is False


def test_trivial_pool_falls_back_to_base() -> None:
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    hybrid = np.array([0.5], dtype=np.float32)
    feats = sensitivity_features(hybrid, emb, top_k=5, lambdas=(0.3, 0.7, 1.0))
    chosen, gate, _ = gating_decision(
        feats,
        hybrid,
        gating_mode="basic",
        theta_s=0.0,
        theta_r=0.0,
        lambda_low=0.3,
        lambda_base=0.7,
        min_n_candidates_for_gate=2,
    )
    assert chosen == 0.7
    assert gate == GATE_TRIVIAL


def test_empty_pool_safe() -> None:
    feats = sensitivity_features(
        np.zeros((0,), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        top_k=5,
    )
    assert feats["n_candidates"] == 0
    assert feats["S_low"] == []
    chosen, gate, _ = gating_decision(
        feats,
        np.zeros((0,), dtype=np.float32),
        gating_mode="conservative",
        lambda_low=0.3,
        lambda_base=0.7,
    )
    assert chosen == 0.7
    assert gate == GATE_TRIVIAL


def test_relevance_floor_min_quantile_mode() -> None:
    # H sorted desc: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]; median = 0.65.
    hybrid = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    # S_low has all items above median.
    assert relevance_floor_ok(
        hybrid, s_low=[0, 1, 2], s_base=[0, 1, 2],
        mode="min_quantile", p_floor=0.5, epsilon=0.0,
    ) is True
    # S_low drags in a below-median item.
    assert relevance_floor_ok(
        hybrid, s_low=[0, 4], s_base=[0, 1, 2],
        mode="min_quantile", p_floor=0.5, epsilon=0.0,
    ) is False
