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

    selected: list[int] = [int(np.argmax(query_scores))]
    candidate_set = set(range(n_items)) - set(selected)
    similarity = sentence_vectors @ sentence_vectors.T

    while len(selected) < top_k and candidate_set:
        best_idx = None
        best_score = -1e9
        for idx in candidate_set:
            max_sim = max(similarity[idx, s] for s in selected)
            mmr_score = lambda_weight * float(query_scores[idx]) - (1.0 - lambda_weight) * float(max_sim)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        candidate_set.remove(best_idx)
    return selected
