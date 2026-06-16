from __future__ import annotations

import numpy as np

from fact_checking.build.chunking import AdjacentBoundaryClaimAwareChunking, SemanticChunking
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


def _abc_strategy(**overrides) -> AdjacentBoundaryClaimAwareChunking:
    kwargs = {
        "embedder_cfg": EmbedderConfig(model_name="unused"),
        "lambda_std": 0.5,
        "min_tokens_per_chunk": 0,
        "max_sent_per_chunk": 3,
        "max_tokens_per_chunk": 100,
    }
    kwargs.update(overrides)
    strategy = AdjacentBoundaryClaimAwareChunking(**kwargs)

    def fake_encode(texts: list[str]) -> np.ndarray:
        raise AssertionError(f"unexpected encoding call: {texts}")

    strategy._encode = fake_encode  # type: ignore[method-assign]
    return strategy


def test_abc_claim_aware_uses_adjacent_boundaries_not_global_similarity() -> None:
    strategy = _abc_strategy()
    sents = [
        "Alpha policy changed.",
        "Completely unrelated bridge.",
        "Alpha policy returned.",
    ]
    embeddings_by_idx = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
        2: np.array([1.0, 0.0], dtype=np.float32),
    }

    chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="alpha policy",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert [chunk.sent_indices for chunk in chunks] == [(0,), (1,), (2,)]


def test_abc_claim_relevance_difference_can_create_boundary_when_semantics_match() -> None:
    strategy = _abc_strategy()
    sents = [
        "Tax revenue increased sharply.",
        "Officials met privately.",
    ]
    embeddings_by_idx = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
    }

    chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="tax revenue",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert [chunk.sent_indices for chunk in chunks] == [(0,), (1,)]
    assert chunks[0].metadata["boundary_right_score"] > 0.0


def test_abc_local_peak_keeps_only_local_boundary_peak() -> None:
    strategy = _abc_strategy()
    sents = [
        "Alpha opening sentence.",
        "Alpha follow up sentence.",
        "Different topic starts here.",
        "Different topic continues here.",
    ]
    embeddings_by_idx = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([0.0, 1.0], dtype=np.float32),
    }

    chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="alpha opening",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert [chunk.sent_indices for chunk in chunks] == [(0, 1), (2, 3)]


def test_abc_hard_cap_splits_long_chunks_by_highest_boundary_score() -> None:
    strategy = _abc_strategy(max_sent_per_chunk=2)
    sents = [
        "Alpha one.",
        "Alpha two.",
        "Alpha three.",
        "Alpha four.",
        "Alpha five.",
    ]
    embeddings_by_idx = {
        idx: np.array([1.0, 0.0], dtype=np.float32)
        for idx in range(len(sents))
    }

    chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="alpha",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert all(len(chunk.sent_indices) <= 2 for chunk in chunks)
    assert [idx for chunk in chunks for idx in chunk.sent_indices] == list(range(len(sents)))


def test_abc_merges_low_relevance_short_chunk_but_keeps_relevant_singleton() -> None:
    strategy = _abc_strategy(min_tokens_per_chunk=3, max_sent_per_chunk=3)
    sents = [
        "Revenue increased strongly.",
        "Aside.",
        "Tax revenue rose.",
    ]
    embeddings_by_idx = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
        2: np.array([1.0, 0.0], dtype=np.float32),
    }

    chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="tax revenue",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert (1,) not in [chunk.sent_indices for chunk in chunks]

    relevant_singleton_chunks = strategy.chunks_from_presplit_with_context(
        [
            "Tax revenue.",
            "Officials discussed matters in a long meeting.",
        ],
        claim="tax revenue",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx={
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    assert (0,) in [chunk.sent_indices for chunk in relevant_singleton_chunks]


def test_abc_anchor_metadata_uses_highest_claim_relevance_sentence() -> None:
    strategy = _abc_strategy(w_rel=0.0)
    sents = [
        "Officials discussed the issue.",
        "Tax revenue increased sharply.",
        "The meeting ended later.",
    ]
    embeddings_by_idx = {
        idx: np.array([1.0, 0.0], dtype=np.float32)
        for idx in range(len(sents))
    }

    chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="tax revenue",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["anchor_sent_idx"] == 1
    assert chunks[0].metadata["anchor_text"] == "Tax revenue increased sharply."


def test_abc_same_report_different_claims_do_not_share_claim_aware_metadata() -> None:
    strategy = _abc_strategy(w_rel=0.0)
    sents = [
        "Tax revenue increased sharply.",
        "Officials met privately.",
    ]
    embeddings_by_idx = {
        idx: np.array([1.0, 0.0], dtype=np.float32)
        for idx in range(len(sents))
    }

    tax_chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="tax revenue",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )
    officials_chunks = strategy.chunks_from_presplit_with_context(
        sents,
        claim="officials met",
        claim_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_sent_idx=embeddings_by_idx,
    )

    assert tax_chunks[0].metadata["anchor_sent_idx"] == 0
    assert officials_chunks[0].metadata["anchor_sent_idx"] == 1
