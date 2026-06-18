from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)
from fact_checking.utils.text import robust_sentence_split, word_tokens


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    sent_indices: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """Produce the evidence text for a candidate sentence within a source report."""

    @abstractmethod
    def chunk(self, content: str, sent_idx: int) -> str: ...

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        """Same as chunk() but accepts pre-split sentences to avoid redundant splitting."""
        return self.chunk(" ".join(sents), sent_idx)

    def chunk_from_presplit_with_embeddings(
        self,
        sents: list[str],
        sent_idx: int,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> str:
        """Same as chunk_from_presplit(), optionally reusing sentence embeddings."""
        del embeddings_by_sent_idx
        return self.chunk_from_presplit(sents, sent_idx)

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        seen: set[str] = set()
        for idx in range(len(sents)):
            text = self.chunk_from_presplit(sents, idx)
            key = " ".join(text.lower().split())
            if not text.strip() or key in seen:
                continue
            seen.add(key)
            records.append(ChunkRecord(text=text, sent_indices=(idx,)))
        return records

    def chunks_from_presplit_with_embeddings(
        self,
        sents: list[str],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> list[ChunkRecord]:
        del embeddings_by_sent_idx
        return self.chunks_from_presplit(sents)

    def chunk_from_presplit_with_context(
        self,
        sents: list[str],
        sent_idx: int,
        *,
        claim: str = "",
        claim_embedding: np.ndarray | None = None,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> str:
        """Same as chunk_from_presplit_with_embeddings(), with optional claim context."""
        del claim, claim_embedding
        return self.chunk_from_presplit_with_embeddings(sents, sent_idx, embeddings_by_sent_idx)

    def chunks_from_presplit_with_context(
        self,
        sents: list[str],
        *,
        claim: str = "",
        claim_embedding: np.ndarray | None = None,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> list[ChunkRecord]:
        """Same as chunks_from_presplit_with_embeddings(), with optional claim context."""
        del claim, claim_embedding
        return self.chunks_from_presplit_with_embeddings(sents, embeddings_by_sent_idx)


class SentenceChunking(ChunkingStrategy):
    """Return the single sentence at sent_idx (the current default behaviour)."""

    def chunk(self, content: str, sent_idx: int) -> str:
        sents = robust_sentence_split(content)
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        return sents[idx]

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        return sents[idx]

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        return [
            ChunkRecord(text=text, sent_indices=(idx,))
            for idx, text in enumerate(sents)
            if text.strip()
        ]


class ContextWindowChunking(ChunkingStrategy):
    """Return a window of 2k+1 sentences centred on sent_idx."""

    def __init__(self, k: int) -> None:
        self.k = int(k)

    def chunk(self, content: str, sent_idx: int) -> str:
        sents = robust_sentence_split(content)
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        left = max(0, idx - self.k)
        right = min(len(sents), idx + self.k + 1)
        return " ".join(sents[pos] for pos in range(left, right))

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        left = max(0, idx - self.k)
        right = min(len(sents), idx + self.k + 1)
        return " ".join(sents[pos] for pos in range(left, right))

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        seen: set[str] = set()
        for idx in range(len(sents)):
            left = max(0, idx - self.k)
            right = min(len(sents), idx + self.k + 1)
            text = " ".join(sents[pos] for pos in range(left, right))
            key = " ".join(text.lower().split())
            if not text.strip() or key in seen:
                continue
            seen.add(key)
            records.append(ChunkRecord(text=text, sent_indices=tuple(range(left, right))))
        return records


class RawChunking(ChunkingStrategy):
    """Use the full report content as a single unit."""

    def chunk(self, content: str, sent_idx: int) -> str:
        return content

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        return " ".join(sents)

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        text = " ".join(sents)
        if not text.strip():
            return []
        return [ChunkRecord(text=text, sent_indices=tuple(range(len(sents))))]


class _SemanticBase(ChunkingStrategy):
    """Shared embedder + caching utilities for semantic chunking strategies."""

    def __init__(self, theta: float, embedder_cfg: EmbedderConfig) -> None:
        self.theta = float(theta)
        self._embedder_cfg = embedder_cfg
        self._embedder: TextEmbedder | None = None
        self._embedder_lock = threading.Lock()
        self._partition_cache: dict[tuple, list[list[int]]] = {}
        self._cache_lock = threading.Lock()

    def _get_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = TextEmbedder(self._embedder_cfg)
        return self._embedder

    def _encode(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        with self._embedder_lock:
            return embedder.encode(texts, is_query=False)

    def _encode_query(self, text: str) -> np.ndarray:
        embedder = self._get_embedder()
        with self._embedder_lock:
            vectors = embedder.encode([text], is_query=True)
        return np.asarray(vectors[0], dtype=np.float32).reshape(-1)

    def _embeddings_for_partition(
        self,
        sents: tuple[str, ...],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        if not embeddings_by_sent_idx:
            return self._encode(list(sents))

        first_vec: np.ndarray | None = None
        normalized: dict[int, np.ndarray] = {}
        for sent_idx, embedding in embeddings_by_sent_idx.items():
            idx = int(sent_idx)
            if idx < 0 or idx >= len(sents):
                continue
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vec.size == 0:
                continue
            if first_vec is None:
                first_vec = vec
            elif vec.shape != first_vec.shape:
                return self._encode(list(sents))
            normalized[idx] = vec

        if first_vec is None:
            return self._encode(list(sents))

        embeddings = np.empty((len(sents), first_vec.shape[0]), dtype=np.float32)
        missing_indices: list[int] = []
        for idx in range(len(sents)):
            vec = normalized.get(idx)
            if vec is None:
                missing_indices.append(idx)
            else:
                embeddings[idx] = vec

        if missing_indices:
            missing_embeddings = self._encode([sents[idx] for idx in missing_indices])
            if (
                missing_embeddings.ndim != 2
                or missing_embeddings.shape[0] != len(missing_indices)
                or missing_embeddings.shape[1] != first_vec.shape[0]
            ):
                return self._encode(list(sents))
            for row_idx, sent_idx in enumerate(missing_indices):
                embeddings[sent_idx] = missing_embeddings[row_idx]

        return embeddings


class SemanticChunking(_SemanticBase):
    """Merge adjacent sentences whose cosine similarity exceeds theta."""

    def _partition(
        self,
        sents: tuple[str, ...],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> list[list[int]]:
        n = len(sents)
        if n == 0:
            return []
        if n == 1:
            return [[0]]
        with self._cache_lock:
            cached = self._partition_cache.get(sents)
        if cached is not None:
            return cached

        embeddings = self._embeddings_for_partition(sents, embeddings_by_sent_idx)
        chunks: list[list[int]] = []
        current = [0]
        for i in range(n - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            if sim > self.theta:
                current.append(i + 1)
            else:
                chunks.append(current)
                current = [i + 1]
        chunks.append(current)

        with self._cache_lock:
            self._partition_cache[sents] = chunks
        return chunks

    def chunk(self, content: str, sent_idx: int) -> str:
        sents = robust_sentence_split(content)
        return self.chunk_from_presplit(sents, sent_idx)

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        chunks = self._partition(tuple(sents))
        for chunk in chunks:
            if idx in chunk:
                return " ".join(sents[i] for i in chunk)
        return sents[idx]

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        return [
            ChunkRecord(text=" ".join(sents[i] for i in chunk), sent_indices=tuple(chunk))
            for chunk in self._partition(tuple(sents))
        ]

    def chunk_from_presplit_with_embeddings(
        self,
        sents: list[str],
        sent_idx: int,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        chunks = self._partition(tuple(sents), embeddings_by_sent_idx)
        for chunk in chunks:
            if idx in chunk:
                return " ".join(sents[i] for i in chunk)
        return sents[idx]

    def chunks_from_presplit_with_embeddings(
        self,
        sents: list[str],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> list[ChunkRecord]:
        return [
            ChunkRecord(text=" ".join(sents[i] for i in chunk), sent_indices=tuple(chunk))
            for chunk in self._partition(tuple(sents), embeddings_by_sent_idx)
        ]


class ContextSemanticChunking(_SemanticBase):
    """Group sentences into non-overlapping windows of k, then merge by similarity."""

    def __init__(self, k: int, theta: float, embedder_cfg: EmbedderConfig) -> None:
        super().__init__(theta=theta, embedder_cfg=embedder_cfg)
        self.k = max(int(k), 1)

    def _windows(self, n: int) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + self.k, n)
            windows.append((start, end))
            start = end
        return windows

    def _partition(
        self,
        sents: tuple[str, ...],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> list[list[int]]:
        n = len(sents)
        if n == 0:
            return []
        with self._cache_lock:
            cached = self._partition_cache.get(sents)
        if cached is not None:
            return cached

        windows = self._windows(n)
        if len(windows) == 1:
            chunks = [list(range(n))]
            with self._cache_lock:
                self._partition_cache[sents] = chunks
            return chunks

        sent_emb = self._embeddings_for_partition(sents, embeddings_by_sent_idx)
        window_emb = np.stack([sent_emb[s:e].mean(axis=0) for s, e in windows], axis=0)
        norms = np.linalg.norm(window_emb, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        window_emb = window_emb / norms

        chunks: list[list[int]] = []
        current = list(range(windows[0][0], windows[0][1]))
        for w_idx in range(len(windows) - 1):
            sim = float(np.dot(window_emb[w_idx], window_emb[w_idx + 1]))
            next_start, next_end = windows[w_idx + 1]
            if sim > self.theta:
                current.extend(range(next_start, next_end))
            else:
                chunks.append(current)
                current = list(range(next_start, next_end))
        chunks.append(current)

        with self._cache_lock:
            self._partition_cache[sents] = chunks
        return chunks

    def chunk(self, content: str, sent_idx: int) -> str:
        sents = robust_sentence_split(content)
        return self.chunk_from_presplit(sents, sent_idx)

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        chunks = self._partition(tuple(sents))
        for chunk in chunks:
            if idx in chunk:
                return " ".join(sents[i] for i in chunk)
        return sents[idx]

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        return [
            ChunkRecord(text=" ".join(sents[i] for i in chunk), sent_indices=tuple(chunk))
            for chunk in self._partition(tuple(sents))
        ]

    def chunk_from_presplit_with_embeddings(
        self,
        sents: list[str],
        sent_idx: int,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        chunks = self._partition(tuple(sents), embeddings_by_sent_idx)
        for chunk in chunks:
            if idx in chunk:
                return " ".join(sents[i] for i in chunk)
        return sents[idx]

    def chunks_from_presplit_with_embeddings(
        self,
        sents: list[str],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> list[ChunkRecord]:
        return [
            ChunkRecord(text=" ".join(sents[i] for i in chunk), sent_indices=tuple(chunk))
            for chunk in self._partition(tuple(sents), embeddings_by_sent_idx)
        ]


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    denom = max(float(np.linalg.norm(va) * np.linalg.norm(vb)), 1e-12)
    return float(np.dot(va, vb) / denom)


def _span_token_count(sents: list[str], start: int, end: int) -> int:
    return len(word_tokens(" ".join(sents[start : end + 1])))


def _starts_with_coref_pronoun(text: str) -> bool:
    tokens = word_tokens(text)
    if not tokens:
        return False
    return tokens[0] in {"he", "she", "they", "it", "this", "that", "these", "those"}


class AdjacentBoundaryClaimAwareChunking(_SemanticBase):
    """Claim-aware adjacent boundary chunking for report-local evidence spans."""

    def __init__(
        self,
        embedder_cfg: EmbedderConfig,
        *,
        boundary_mode: str = "local_peak",
        boundary_threshold: float = 0.55,
        lambda_std: float = 0.5,
        w_sem: float = 0.75,
        w_rel: float = 0.25,
        max_sent_per_chunk: int = 3,
        max_tokens_per_chunk: int = 150,
        min_tokens_per_chunk: int = 20,
        allow_single_sentence_if_relevant: bool = True,
        single_sentence_relevance_threshold: float = 0.55,
        high_rel_threshold: float = 0.70,
        coref_boundary_discount: float = 0.10,
        chunking_method: str = "ABC-claim-aware-v1",
    ) -> None:
        super().__init__(theta=0.0, embedder_cfg=embedder_cfg)
        self.boundary_mode = str(boundary_mode or "local_peak").strip().lower()
        self.boundary_threshold = float(boundary_threshold)
        self.lambda_std = float(lambda_std)
        self.w_sem = float(w_sem)
        self.w_rel = float(w_rel)
        self.max_sent_per_chunk = max(1, int(max_sent_per_chunk))
        self.max_tokens_per_chunk = max(1, int(max_tokens_per_chunk))
        self.min_tokens_per_chunk = max(0, int(min_tokens_per_chunk))
        self.allow_single_sentence_if_relevant = bool(allow_single_sentence_if_relevant)
        self.single_sentence_relevance_threshold = float(single_sentence_relevance_threshold)
        self.high_rel_threshold = float(high_rel_threshold)
        self.coref_boundary_discount = max(0.0, float(coref_boundary_discount))
        self.chunking_method = str(chunking_method or "ABC-claim-aware-v1")

    def chunk(self, content: str, sent_idx: int) -> str:
        sents = robust_sentence_split(content)
        return self.chunk_from_presplit(sents, sent_idx)

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        return self.chunk_from_presplit_with_context(sents, sent_idx)

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        return self.chunks_from_presplit_with_context(sents)

    def chunk_from_presplit_with_embeddings(
        self,
        sents: list[str],
        sent_idx: int,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> str:
        return self.chunk_from_presplit_with_context(
            sents,
            sent_idx,
            embeddings_by_sent_idx=embeddings_by_sent_idx,
        )

    def chunks_from_presplit_with_embeddings(
        self,
        sents: list[str],
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None,
    ) -> list[ChunkRecord]:
        return self.chunks_from_presplit_with_context(
            sents,
            embeddings_by_sent_idx=embeddings_by_sent_idx,
        )

    def chunk_from_presplit_with_context(
        self,
        sents: list[str],
        sent_idx: int,
        *,
        claim: str = "",
        claim_embedding: np.ndarray | None = None,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> str:
        if not sents:
            return ""
        idx = min(max(int(sent_idx), 0), len(sents) - 1)
        chunks = self.chunks_from_presplit_with_context(
            sents,
            claim=claim,
            claim_embedding=claim_embedding,
            embeddings_by_sent_idx=embeddings_by_sent_idx,
        )
        for chunk in chunks:
            if idx in chunk.sent_indices:
                return chunk.text
        return sents[idx]

    def chunks_from_presplit_with_context(
        self,
        sents: list[str],
        *,
        claim: str = "",
        claim_embedding: np.ndarray | None = None,
        embeddings_by_sent_idx: Mapping[int, np.ndarray] | None = None,
    ) -> list[ChunkRecord]:
        clean_sents = [str(sent).strip() for sent in sents]
        clean_sents = [sent for sent in clean_sents if sent]
        n = len(clean_sents)
        if n == 0:
            return []

        sent_emb = self._embeddings_for_partition(tuple(clean_sents), embeddings_by_sent_idx)
        claim_vec = self._claim_embedding(str(claim or ""), claim_embedding)
        rel_scores = self._claim_relevance_scores(clean_sents, sent_emb, str(claim or ""), claim_vec)
        adj_sims = self._adjacent_similarities(sent_emb)
        boundary_scores = self._boundary_scores(clean_sents, adj_sims, rel_scores)
        boundaries = self._decide_boundaries(boundary_scores)
        spans = self._spans_from_boundaries(boundaries, n)
        spans = self._enforce_max_length(spans, boundary_scores, clean_sents)
        spans = self._merge_too_short_spans(spans, adj_sims, rel_scores, clean_sents)
        return [
            self._build_record(clean_sents, start, end, rel_scores, boundary_scores)
            for start, end in spans
        ]

    def _claim_embedding(self, claim: str, claim_embedding: np.ndarray | None) -> np.ndarray | None:
        if claim_embedding is not None:
            vec = np.asarray(claim_embedding, dtype=np.float32).reshape(-1)
            return vec if vec.size else None
        if not claim.strip():
            return None
        return self._encode_query(claim)

    def _claim_relevance_scores(
        self,
        sents: list[str],
        sent_emb: np.ndarray,
        claim: str,
        claim_vec: np.ndarray | None,
    ) -> np.ndarray:
        n = len(sents)
        dense = np.zeros(n, dtype=np.float32)
        if claim_vec is not None:
            for idx in range(n):
                dense[idx] = _cosine(sent_emb[idx], claim_vec)

        q_ctr, q_len = content_tokens_counter(claim)
        lexical = np.zeros(n, dtype=np.float32)
        bm25 = np.zeros(n, dtype=np.float32)
        for idx, sent in enumerate(sents):
            s_ctr, s_len = content_tokens_counter(sent)
            lexical[idx] = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
            bm25[idx] = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)

        return (
            0.70 * _minmax_scale(dense)
            + 0.20 * _minmax_scale(lexical)
            + 0.10 * _minmax_scale(bm25)
        ).astype(np.float32, copy=False)

    def _adjacent_similarities(self, sent_emb: np.ndarray) -> list[float]:
        return [
            _cosine(sent_emb[idx], sent_emb[idx + 1])
            for idx in range(max(0, int(sent_emb.shape[0]) - 1))
        ]

    def _boundary_scores(
        self,
        sents: list[str],
        adj_sims: list[float],
        rel_scores: np.ndarray,
    ) -> list[float]:
        scores: list[float] = []
        for idx, sim in enumerate(adj_sims):
            b_sem = 1.0 - float(sim)
            b_rel = abs(float(rel_scores[idx]) - float(rel_scores[idx + 1]))
            score = self.w_sem * b_sem + self.w_rel * b_rel
            if _starts_with_coref_pronoun(sents[idx + 1]):
                score = max(0.0, score - self.coref_boundary_discount)
            scores.append(float(score))
        return scores

    def _decide_boundaries(self, boundary_scores: list[float]) -> list[bool]:
        if not boundary_scores:
            return []
        scores = np.asarray(boundary_scores, dtype=np.float32)
        if self.boundary_mode == "absolute":
            return [float(score) >= self.boundary_threshold for score in scores]
        if self.boundary_mode != "local_peak":
            raise ValueError(f"Unknown ABC boundary_mode: {self.boundary_mode!r}")

        threshold = float(scores.mean() + self.lambda_std * scores.std())
        boundaries: list[bool] = []
        for idx, score in enumerate(scores):
            left = float(scores[idx - 1]) if idx > 0 else -1.0
            right = float(scores[idx + 1]) if idx < len(scores) - 1 else -1.0
            is_peak = float(score) >= left and float(score) >= right
            boundaries.append(bool(float(score) > 0.0 and is_peak and float(score) >= threshold))
        return boundaries

    @staticmethod
    def _spans_from_boundaries(boundaries: list[bool], n: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        for idx, is_boundary in enumerate(boundaries):
            if is_boundary:
                spans.append((start, idx))
                start = idx + 1
        spans.append((start, n - 1))
        return spans

    def _span_is_too_long(self, sents: list[str], start: int, end: int) -> bool:
        if end - start + 1 > self.max_sent_per_chunk:
            return True
        return _span_token_count(sents, start, end) > self.max_tokens_per_chunk

    def _enforce_max_length(
        self,
        spans: list[tuple[int, int]],
        boundary_scores: list[float],
        sents: list[str],
    ) -> list[tuple[int, int]]:
        final: list[tuple[int, int]] = []
        queue = list(spans)
        while queue:
            start, end = queue.pop(0)
            if start >= end or not self._span_is_too_long(sents, start, end):
                final.append((start, end))
                continue
            candidate_boundaries = list(range(start, end))
            if candidate_boundaries:
                split_at = max(
                    candidate_boundaries,
                    key=lambda idx: float(boundary_scores[idx]) if idx < len(boundary_scores) else 0.0,
                )
            else:
                split_at = start
            queue.insert(0, (split_at + 1, end))
            queue.insert(0, (start, split_at))
        return final

    def _span_can_accept(self, sents: list[str], start: int, end: int) -> bool:
        return (
            end - start + 1 <= self.max_sent_per_chunk
            and _span_token_count(sents, start, end) <= self.max_tokens_per_chunk
        )

    def _span_has_relevant_sentence(self, rel_scores: np.ndarray, start: int, end: int) -> bool:
        if not self.allow_single_sentence_if_relevant:
            return False
        if start < 0 or end >= len(rel_scores):
            return False
        return float(rel_scores[start : end + 1].max()) >= self.single_sentence_relevance_threshold

    def _merge_too_short_spans(
        self,
        spans: list[tuple[int, int]],
        adj_sims: list[float],
        rel_scores: np.ndarray,
        sents: list[str],
    ) -> list[tuple[int, int]]:
        if self.min_tokens_per_chunk <= 0 or len(spans) <= 1:
            return spans

        merged: list[tuple[int, int]] = []
        pending = list(spans)
        idx = 0
        while idx < len(pending):
            start, end = pending[idx]
            token_count = _span_token_count(sents, start, end)
            if token_count >= self.min_tokens_per_chunk or self._span_has_relevant_sentence(rel_scores, start, end):
                merged.append((start, end))
                idx += 1
                continue

            left = merged[-1] if merged else None
            right = pending[idx + 1] if idx + 1 < len(pending) else None
            choices: list[tuple[str, float, float]] = []
            if left is not None and self._span_can_accept(sents, left[0], end):
                sim = float(adj_sims[start - 1]) if start - 1 >= 0 and start - 1 < len(adj_sims) else -1.0
                relevance = float(rel_scores[left[0] : left[1] + 1].max())
                choices.append(("left", sim, relevance))
            if right is not None and self._span_can_accept(sents, start, right[1]):
                sim = float(adj_sims[end]) if end < len(adj_sims) else -1.0
                relevance = float(rel_scores[right[0] : right[1] + 1].max())
                choices.append(("right", sim, relevance))

            if not choices:
                merged.append((start, end))
                idx += 1
                continue

            target = max(choices, key=lambda item: (item[1], item[2], 1 if item[0] == "right" else 0))[0]
            if target == "left":
                left_start, _left_end = merged[-1]
                merged[-1] = (left_start, end)
                idx += 1
            else:
                _right_start, right_end = pending[idx + 1]
                pending[idx + 1] = (start, right_end)
                idx += 1

        return merged

    def _build_record(
        self,
        sents: list[str],
        start: int,
        end: int,
        rel_scores: np.ndarray,
        boundary_scores: list[float],
    ) -> ChunkRecord:
        indices = tuple(range(start, end + 1))
        chunk_rel = rel_scores[start : end + 1]
        anchor_offset = int(chunk_rel.argmax()) if chunk_rel.size else 0
        anchor_idx = start + anchor_offset
        token_count = _span_token_count(sents, start, end)
        metadata: dict[str, Any] = {
            "chunking_method": self.chunking_method,
            "sent_start": int(start),
            "sent_end": int(end),
            "num_sentences": int(end - start + 1),
            "num_tokens": int(token_count),
            "claim_relevance": float(chunk_rel.max()) if chunk_rel.size else 0.0,
            "anchor_sent_idx": int(anchor_idx),
            "anchor_text": sents[anchor_idx] if 0 <= anchor_idx < len(sents) else "",
            "anchor_claim_relevance": float(rel_scores[anchor_idx]) if 0 <= anchor_idx < len(rel_scores) else 0.0,
            "boundary_left_score": float(boundary_scores[start - 1]) if start > 0 and start - 1 < len(boundary_scores) else None,
            "boundary_right_score": float(boundary_scores[end]) if end < len(boundary_scores) else None,
            "has_high_rel_sentence": bool(chunk_rel.size and float(chunk_rel.max()) >= self.high_rel_threshold),
            "token_overflow": bool(start == end and token_count > self.max_tokens_per_chunk),
        }
        return ChunkRecord(
            text=" ".join(sents[idx] for idx in indices),
            sent_indices=indices,
            metadata=metadata,
        )


class AdjacentBoundaryRawfcTightChunking(AdjacentBoundaryClaimAwareChunking):
    """RAWFC-specific tighter ABC preset without changing the base ABC strategy."""

    def __init__(self, embedder_cfg: EmbedderConfig, **kwargs: Any) -> None:
        kwargs.setdefault("lambda_std", 0.35)
        kwargs.setdefault("w_sem", 0.65)
        kwargs.setdefault("w_rel", 0.35)
        kwargs.setdefault("max_sent_per_chunk", 2)
        kwargs.setdefault("max_tokens_per_chunk", 150)
        kwargs.setdefault("min_tokens_per_chunk", 20)
        kwargs.setdefault("allow_single_sentence_if_relevant", True)
        kwargs.setdefault("single_sentence_relevance_threshold", 0.55)
        kwargs.setdefault("high_rel_threshold", 0.70)
        kwargs.setdefault("coref_boundary_discount", 0.10)
        kwargs.setdefault("chunking_method", "ABC-claim-aware-rawfc-tight-v1")
        super().__init__(embedder_cfg, **kwargs)


def _build_semantic_embedder_cfg(
    cfg: dict, retrieval_cfg: dict | None
) -> EmbedderConfig:
    retrieval_cfg = retrieval_cfg or {}
    model_name = cfg.get("embedder_model") or retrieval_cfg.get("embedder_model")
    if not model_name:
        raise ValueError(
            "Semantic chunking requires an embedder_model. "
            "Set retrieval.chunking.embedder_model or retrieval.embedder_model."
        )
    return EmbedderConfig(
        model_name=str(model_name),
        device=str(cfg.get("device", retrieval_cfg.get("device", "cpu"))),
        max_length=int(cfg.get("max_length", retrieval_cfg.get("max_length", 256))),
        batch_size=int(cfg.get("batch_size", retrieval_cfg.get("batch_size", 64))),
        normalize=True,
        precision=str(cfg.get("precision", retrieval_cfg.get("precision", "fp32"))),
    )


def build_chunking_strategy(
    cfg: dict | None, retrieval_cfg: dict | None = None
) -> ChunkingStrategy:
    if cfg is None:
        return SentenceChunking()
    strategy = str(cfg.get("strategy", "sentence")).strip().lower()
    if strategy == "sentence":
        return SentenceChunking()
    if strategy == "ctx_window":
        return ContextWindowChunking(k=int(cfg.get("context_k", 1)))
    if strategy == "raw":
        return RawChunking()
    if strategy == "semantic":
        embedder_cfg = _build_semantic_embedder_cfg(cfg, retrieval_cfg)
        return SemanticChunking(theta=float(cfg.get("theta", 0.7)), embedder_cfg=embedder_cfg)
    if strategy == "ctx_semantic":
        embedder_cfg = _build_semantic_embedder_cfg(cfg, retrieval_cfg)
        return ContextSemanticChunking(
            k=int(cfg.get("context_k", 1)),
            theta=float(cfg.get("theta", 0.7)),
            embedder_cfg=embedder_cfg,
        )
    if strategy == "abc_claim_aware":
        embedder_cfg = _build_semantic_embedder_cfg(cfg, retrieval_cfg)
        return AdjacentBoundaryClaimAwareChunking(
            embedder_cfg=embedder_cfg,
            boundary_mode=str(cfg.get("boundary_mode", "local_peak")),
            boundary_threshold=float(cfg.get("boundary_threshold", 0.55)),
            lambda_std=float(cfg.get("lambda_std", 0.5)),
            w_sem=float(cfg.get("w_sem", 0.75)),
            w_rel=float(cfg.get("w_rel", 0.25)),
            max_sent_per_chunk=int(cfg.get("max_sent_per_chunk", 3)),
            max_tokens_per_chunk=int(cfg.get("max_tokens_per_chunk", 150)),
            min_tokens_per_chunk=int(cfg.get("min_tokens_per_chunk", 20)),
            allow_single_sentence_if_relevant=bool(cfg.get("allow_single_sentence_if_relevant", True)),
            single_sentence_relevance_threshold=float(cfg.get("single_sentence_relevance_threshold", 0.55)),
            high_rel_threshold=float(cfg.get("high_rel_threshold", 0.70)),
            coref_boundary_discount=float(cfg.get("coref_boundary_discount", 0.10)),
        )
    if strategy == "abc_claim_aware_rawfc_tight":
        embedder_cfg = _build_semantic_embedder_cfg(cfg, retrieval_cfg)
        return AdjacentBoundaryRawfcTightChunking(
            embedder_cfg=embedder_cfg,
            boundary_mode=str(cfg.get("boundary_mode", "local_peak")),
            boundary_threshold=float(cfg.get("boundary_threshold", 0.55)),
            lambda_std=float(cfg.get("lambda_std", 0.35)),
            w_sem=float(cfg.get("w_sem", 0.65)),
            w_rel=float(cfg.get("w_rel", 0.35)),
            max_sent_per_chunk=int(cfg.get("max_sent_per_chunk", 2)),
            max_tokens_per_chunk=int(cfg.get("max_tokens_per_chunk", 150)),
            min_tokens_per_chunk=int(cfg.get("min_tokens_per_chunk", 20)),
            allow_single_sentence_if_relevant=bool(cfg.get("allow_single_sentence_if_relevant", True)),
            single_sentence_relevance_threshold=float(cfg.get("single_sentence_relevance_threshold", 0.55)),
            high_rel_threshold=float(cfg.get("high_rel_threshold", 0.70)),
            coref_boundary_discount=float(cfg.get("coref_boundary_discount", 0.10)),
            chunking_method=str(cfg.get("chunking_method", "ABC-claim-aware-rawfc-tight-v1")),
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
