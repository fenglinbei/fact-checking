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
    if n_items == 0:
        return []
    if top_k >= n_items:
        return list(np.argsort(-query_scores))

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
