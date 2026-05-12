from __future__ import annotations

import numpy as np

from fact_checking.build.chunking import SemanticChunking
from fact_checking.retrieval.embedder import EmbedderConfig


def test_semantic_chunking_uses_precomputed_embeddings_without_encoding() -> None:
    strategy = SemanticChunking(
        theta=0.8,
        embedder_cfg=EmbedderConfig(model_name="unused"),
    )
    encode_calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> np.ndarray:
        encode_calls.append(texts)
        return np.zeros((len(texts), 2), dtype=np.float32)

    strategy._encode = fake_encode  # type: ignore[method-assign]

    sents = [
        "Alpha topic continues.",
        "Alpha topic also continues.",
        "Different topic starts.",
    ]
    embeddings_by_idx = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
    }

    chunk = strategy.chunk_from_presplit_with_embeddings(sents, 0, embeddings_by_idx)

    assert chunk == "Alpha topic continues. Alpha topic also continues."
    assert encode_calls == []


def test_semantic_chunking_encodes_only_missing_embeddings() -> None:
    strategy = SemanticChunking(
        theta=0.8,
        embedder_cfg=EmbedderConfig(model_name="unused"),
    )
    encode_calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> np.ndarray:
        encode_calls.append(texts)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    strategy._encode = fake_encode  # type: ignore[method-assign]

    sents = [
        "Alpha topic continues.",
        "Short.",
        "Different topic starts.",
    ]
    embeddings_by_idx = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
    }

    chunk = strategy.chunk_from_presplit_with_embeddings(sents, 0, embeddings_by_idx)

    assert chunk == "Alpha topic continues. Short."
    assert encode_calls == [["Short."]]
