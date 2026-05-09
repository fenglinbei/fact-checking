from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import numpy as np

from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder
from fact_checking.utils.text import robust_sentence_split


class ChunkingStrategy(ABC):
    """Produce the evidence text for a candidate sentence within a source report."""

    @abstractmethod
    def chunk(self, content: str, sent_idx: int) -> str: ...

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        """Same as chunk() but accepts pre-split sentences to avoid redundant splitting."""
        return self.chunk(" ".join(sents), sent_idx)


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


class RawChunking(ChunkingStrategy):
    """Use the full report content as a single unit."""

    def chunk(self, content: str, sent_idx: int) -> str:
        return content

    def chunk_from_presplit(self, sents: list[str], sent_idx: int) -> str:
        return " ".join(sents)


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


class SemanticChunking(_SemanticBase):
    """Merge adjacent sentences whose cosine similarity exceeds theta."""

    def _partition(self, sents: tuple[str, ...]) -> list[list[int]]:
        n = len(sents)
        if n == 0:
            return []
        if n == 1:
            return [[0]]
        with self._cache_lock:
            cached = self._partition_cache.get(sents)
        if cached is not None:
            return cached

        embeddings = self._encode(list(sents))
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

    def _partition(self, sents: tuple[str, ...]) -> list[list[int]]:
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

        sent_emb = self._encode(list(sents))
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
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
