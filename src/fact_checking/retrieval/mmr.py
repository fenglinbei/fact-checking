from __future__ import annotations

import numpy as np


def maximal_marginal_relevance(
    query_scores: np.ndarray,
    sentence_vectors: np.ndarray,
    top_k: int,
    lambda_weight: float = 0.7,
) -> list[int]:
    """Select diverse top-k items with MMR.

    Args:
        query_scores: shape [N], larger is better.
        sentence_vectors: shape [N, D], assumed normalized if using cosine.
        top_k: number of items to keep.
        lambda_weight: query relevance vs diversity tradeoff.
    """
    n_items = int(query_scores.shape[0])
    if n_items == 0 or top_k <= 0:
        return []
    if top_k >= n_items:
        return [int(i) for i in np.argsort(-query_scores)]

    similarity = sentence_vectors @ sentence_vectors.T

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


def maximal_marginal_relevance_stepwise(
    query_scores: np.ndarray,
    sentence_vectors: np.ndarray,
    lambda_weights: list[float],
) -> tuple[list[int], list[dict]]:
    """Step-wise MMR with a per-step lambda weight.

    Args:
        query_scores: shape [N], larger is better.
        sentence_vectors: shape [N, D], assumed normalized if using cosine.
        lambda_weights: per-step lambda values. Length determines how many items
            are selected (``top_k = len(lambda_weights)``).

    Returns:
        selected_indices: list of selected candidate indices (length <= len(lambda_weights)).
        step_records: list of per-step dicts with keys ``step_idx``, ``lambda_val``,
            ``selected_idx``, ``hybrid_score``, ``max_sim_to_selected``, ``mmr_score``,
            ``candidate_mask_before``, ``mmr_scores_before``.
    """
    n_items = int(query_scores.shape[0])
    top_k = len(lambda_weights)
    if n_items == 0 or top_k == 0:
        return [], []

    if top_k >= n_items:
        order = [int(i) for i in np.argsort(-query_scores)]
        records: list[dict] = []
        for t, idx in enumerate(order):
            records.append({
                "step_idx": t,
                "lambda_val": float(lambda_weights[t]) if t < len(lambda_weights) else 0.0,
                "selected_idx": idx,
                "hybrid_score": float(query_scores[idx]),
                "max_sim_to_selected": 0.0,
                "mmr_score": float(query_scores[idx]),
                "candidate_mask_before": None,
                "mmr_scores_before": None,
            })
        return order[:top_k], records[:top_k]

    similarity = sentence_vectors @ sentence_vectors.T

    selected: list[int] = []
    candidate_mask = np.ones(n_items, dtype=bool)
    max_sim_to_selected = np.zeros(n_items, dtype=np.float32)
    step_records: list[dict] = []

    for t in range(top_k):
        lam = float(lambda_weights[t])

        if not selected:
            mmr_scores = query_scores.copy().astype(np.float32)
        else:
            mmr_scores = lam * query_scores - (1.0 - lam) * max_sim_to_selected

        mmr_scores[~candidate_mask] = -np.inf
        best_idx = int(np.argmax(mmr_scores))

        step_records.append({
            "step_idx": t,
            "lambda_val": lam,
            "selected_idx": best_idx,
            "hybrid_score": float(query_scores[best_idx]),
            "max_sim_to_selected": float(max_sim_to_selected[best_idx]),
            "mmr_score": float(mmr_scores[best_idx]),
            "candidate_mask_before": candidate_mask.copy(),
            "mmr_scores_before": mmr_scores.copy(),
        })

        selected.append(best_idx)
        candidate_mask[best_idx] = False
        np.maximum(max_sim_to_selected, similarity[best_idx, :], out=max_sim_to_selected)

    return selected, step_records
