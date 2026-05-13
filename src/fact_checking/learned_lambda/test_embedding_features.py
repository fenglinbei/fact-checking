from __future__ import annotations

import numpy as np
import torch

from fact_checking.build.candidates import ChunkMMRSample
from fact_checking.learned_lambda.embedding_features import build_chunk_embedding_arrays
from fact_checking.learned_lambda.predictor import predict_lambdas_for_samples


def _sample(event_id: str, n_candidates: int) -> ChunkMMRSample:
    claim_emb = np.array([1.0, 0.0], dtype=np.float32)
    chunk_emb = np.array(
        [[1.0 - 0.1 * i, 0.1 * i] for i in range(n_candidates)],
        dtype=np.float32,
    )
    return ChunkMMRSample(
        event_id=event_id,
        claim="alpha claim",
        label="true",
        explain="",
        candidates=[
            {"text": f"alpha evidence {i}", "report_id": "r", "sent_idx": i}
            for i in range(n_candidates)
        ],
        chunk_emb=chunk_emb,
        claim_emb=claim_emb,
    )


def test_chunk_embedding_arrays_default_use_full_chunk_pool() -> None:
    samples = [_sample("a", 3), _sample("b", 5)]

    arrays = build_chunk_embedding_arrays(
        samples,
        candidate_top_k=None,
        alpha_dense=1.0,
        alpha_lexical=0.0,
        alpha_bm25=0.0,
    )

    assert arrays["candidate_emb"].shape == (2, 5, 2)
    assert arrays["candidate_mask"].sum(axis=1).tolist() == [3.0, 5.0]
    assert arrays["candidate_counts"].tolist() == [3, 5]


def test_chunk_embedding_arrays_explicit_top_k_still_truncates() -> None:
    samples = [_sample("a", 3), _sample("b", 5)]

    arrays = build_chunk_embedding_arrays(
        samples,
        candidate_top_k=2,
        alpha_dense=1.0,
        alpha_lexical=0.0,
        alpha_bm25=0.0,
    )

    assert arrays["candidate_emb"].shape == (2, 2, 2)
    assert arrays["candidate_mask"].sum(axis=1).tolist() == [2.0, 2.0]
    assert arrays["candidate_counts"].tolist() == [3, 5]


def test_predict_lambdas_chunk_embedding_default_ignores_retrieval_top_k() -> None:
    class ShapeRecorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.candidate_shape: tuple[int, ...] | None = None

        def forward(
            self,
            claim_emb: torch.Tensor,
            candidate_emb: torch.Tensor,
            candidate_mask: torch.Tensor,
        ) -> torch.Tensor:
            self.candidate_shape = tuple(candidate_emb.shape)
            return torch.zeros(claim_emb.shape[0], dtype=torch.float32)

    samples = [_sample("a", 3), _sample("b", 5)]
    model = ShapeRecorder()

    predicted = predict_lambdas_for_samples(
        samples,
        model,  # type: ignore[arg-type]
        stats={
            "feature_mode": "chunk_embedding",
            "candidate_top_k": None,
        },
        retrieval_cfg={
            "top_k": 1,
            "alpha_dense": 1.0,
            "alpha_lexical": 0.0,
            "alpha_bm25": 0.0,
        },
    )

    assert model.candidate_shape == (2, 5, 2)
    assert predicted == {"a": 0.0, "b": 0.0}
