"""Per-step state feature extraction for DPO step-wise λ policy.

Features are designed so that winner/loser trajectories for the SAME claim
have non-zero differences. Pool-level features (identical within a claim) are
excluded. Previous λ choice is included to give the policy memory.
"""

from __future__ import annotations

import numpy as np

from fact_checking.rl_mmr.sensitivity import compute_pairwise_sim, mean_pairwise_sim

# Step-level features that vary based on which items were already selected.
STEP_FEATURE_NAMES: list[str] = [
    "step_fraction",
    "n_already_selected",
    "mean_rel_selected",
    "mean_red_selected",
    "max_red_in_pool",
    "mean_red_in_pool",
    "last_selected_rel",
    "last_selected_red",
    "remaining_score_entropy",
    "remaining_n",
    "top_mmr_score",
    "mmr_score_gap",
]

# Policy state = step features + previous λ (1 dim).
POLICY_FEATURE_NAMES: list[str] = STEP_FEATURE_NAMES + ["prev_lambda"]


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


def _max_pairwise_sim(indices: list[int], sim: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 2 or sim.size == 0:
        return 0.0
    sub = sim[np.ix_(idx, idx)]
    iu = np.triu_indices(sub.shape[0], k=1)
    if iu[0].size == 0:
        return 0.0
    return float(sub[iu].max())


def extract_step_features(
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    selected_indices: list[int],
    candidate_mask: np.ndarray | None,
    step_idx: int,
    total_steps: int,
    mmr_scores_before: np.ndarray | None = None,
) -> np.ndarray:
    """Extract per-step state features before making the selection at step t.

    Returns float32 array of shape [len(STEP_FEATURE_NAMES)].
    """
    n = int(hybrid_scores.shape[0])
    k = max(int(total_steps), 1)

    if candidate_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(candidate_mask, dtype=bool)

    available = np.where(mask)[0]
    n_available = available.size

    # --- already selected stats ---
    t = len(selected_indices)
    if t > 0:
        mean_rel_selected = float(hybrid_scores[selected_indices].mean())
        if t >= 2:
            sim = compute_pairwise_sim(chunk_emb)
            mean_red_selected = mean_pairwise_sim(selected_indices, sim)
        else:
            mean_red_selected = 0.0
        last_selected_rel = float(hybrid_scores[selected_indices[-1]])
        if t >= 2:
            sim_for_last = chunk_emb @ chunk_emb.T
            last_sim_to_prev = max(float(sim_for_last[selected_indices[-1], i]) for i in selected_indices[:-1])
        else:
            last_sim_to_prev = 0.0
        last_selected_red = last_sim_to_prev
    else:
        mean_rel_selected = 0.0
        mean_red_selected = 0.0
        last_selected_rel = 0.0
        last_selected_red = 0.0

    # --- redundancy in pool ---
    if t > 0 and n_available > 0:
        sim_full = chunk_emb @ chunk_emb.T
        max_red_per_item = np.array([
            max(float(sim_full[i, j]) for j in selected_indices)
            for i in available
        ], dtype=np.float32)
        max_red_in_pool = float(max_red_per_item.max()) if max_red_per_item.size > 0 else 0.0
        mean_red_in_pool = float(max_red_per_item.mean()) if max_red_per_item.size > 0 else 0.0
    else:
        max_red_in_pool = 0.0
        mean_red_in_pool = 0.0

    # --- remaining score stats ---
    if n_available > 0:
        available_scores = hybrid_scores[available]
        remaining_score_entropy = _entropy(available_scores)
        remaining_n = float(n_available)
        if mmr_scores_before is not None:
            mmr_available = mmr_scores_before[available]
        else:
            mmr_available = available_scores.copy()
        sorted_mmr = np.sort(mmr_available)[::-1]
        top_mmr_score = float(sorted_mmr[0])
        mmr_score_gap = float(sorted_mmr[0] - sorted_mmr[1]) if n_available >= 2 else 0.0
    else:
        remaining_score_entropy = 0.0
        remaining_n = 0.0
        top_mmr_score = 0.0
        mmr_score_gap = 0.0

    feats = {
        "step_fraction": float(step_idx) / float(k),
        "n_already_selected": float(t),
        "mean_rel_selected": mean_rel_selected,
        "mean_red_selected": mean_red_selected,
        "max_red_in_pool": max_red_in_pool,
        "mean_red_in_pool": mean_red_in_pool,
        "last_selected_rel": last_selected_rel,
        "last_selected_red": last_selected_red,
        "remaining_score_entropy": remaining_score_entropy,
        "remaining_n": remaining_n,
        "top_mmr_score": top_mmr_score,
        "mmr_score_gap": mmr_score_gap,
    }
    return np.array([feats[name] for name in STEP_FEATURE_NAMES], dtype=np.float32)


def extract_policy_features(
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    step_records: list[dict],
    total_steps: int,
    lambda_schedule: list[float],
) -> list[np.ndarray]:
    """Extract policy state features for each step of an episode.

    Features: step features (12 dims) + prev_lambda (1 dim) = 13 dims.
    Pool-level features are NOT included — they are identical for all
    trajectories of the same claim and provide zero differentiation.

    Args:
        hybrid_scores: [N] relevance scores.
        chunk_emb: [N, D] normalized embeddings.
        step_records: list of per-step dicts from ``maximal_marginal_relevance_stepwise``.
        total_steps: K.
        lambda_schedule: the full λ schedule for this trajectory.

    Returns:
        List of float32 arrays, each shape [len(POLICY_FEATURE_NAMES)].
    """
    features: list[np.ndarray] = []
    selected: list[int] = []

    for r in step_records:
        step_feats = extract_step_features(
            hybrid_scores=hybrid_scores,
            chunk_emb=chunk_emb,
            selected_indices=list(selected),
            candidate_mask=r.get("candidate_mask_before"),
            step_idx=int(r["step_idx"]),
            total_steps=total_steps,
            mmr_scores_before=r.get("mmr_scores_before"),
        )
        step_idx = int(r["step_idx"])
        if step_idx == 0:
            prev_lambda = -1.0
        else:
            prev_lambda = float(lambda_schedule[step_idx - 1])

        full = np.append(step_feats, prev_lambda).astype(np.float32)
        features.append(full)
        selected.append(int(r["selected_idx"]))

    return features


# ---------------------------------------------------------------------------
# Backward-compat aliases for code that referenced the old names
# ---------------------------------------------------------------------------

POOL_FEATURE_NAMES: list[str] = [
    "n_candidates", "log_n_candidates", "score_mean", "score_std",
    "score_entropy", "top1_top2_gap", "mean_pairwise_sim", "max_pairwise_sim",
]
ALL_FEATURE_NAMES: list[str] = POLICY_FEATURE_NAMES  # new policy features are the "all" features


def extract_pool_features(hybrid_scores: np.ndarray, chunk_emb: np.ndarray) -> np.ndarray:
    """Extract static pool-level features (kept for backward compat)."""
    n = int(hybrid_scores.shape[0])
    if n == 0:
        return np.zeros(len(POOL_FEATURE_NAMES), dtype=np.float32)
    sorted_scores = np.sort(hybrid_scores)[::-1]
    sim = compute_pairwise_sim(chunk_emb)
    pool_indices = list(range(n))
    feats = {
        "n_candidates": float(n),
        "log_n_candidates": float(np.log1p(n)),
        "score_mean": float(hybrid_scores.mean()),
        "score_std": float(hybrid_scores.std()),
        "score_entropy": _entropy(hybrid_scores),
        "top1_top2_gap": float(sorted_scores[0] - sorted_scores[1]) if n >= 2 else 0.0,
        "mean_pairwise_sim": mean_pairwise_sim(pool_indices, sim),
        "max_pairwise_sim": _max_pairwise_sim(pool_indices, sim),
    }
    return np.array([feats[name] for name in POOL_FEATURE_NAMES], dtype=np.float32)


def extract_episode_features(
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    step_records: list[dict],
    total_steps: int,
    lambda_schedule: list[float] | None = None,
) -> list[np.ndarray]:
    """Backward-compat wrapper: uses new policy features.

    If lambda_schedule is not provided, prev_lambda defaults to -1 for all steps.
    """
    sched = list(lambda_schedule) if lambda_schedule else [-1.0] * total_steps
    return extract_policy_features(hybrid_scores, chunk_emb, step_records, total_steps, sched)
