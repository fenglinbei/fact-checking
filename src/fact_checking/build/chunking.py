from __future__ import annotations

from abc import ABC, abstractmethod

from fact_checking.utils.text import robust_sentence_split


class ChunkingStrategy(ABC):
    """Produce the evidence text for a candidate sentence within a source report."""

    @abstractmethod
    def chunk(self, content: str, sent_idx: int) -> str: ...


class SentenceChunking(ChunkingStrategy):
    """Return the single sentence at sent_idx (the current default behaviour)."""

    def chunk(self, content: str, sent_idx: int) -> str:
        sents = robust_sentence_split(content)
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


class RawChunking(ChunkingStrategy):
    """Use the full report content as a single unit. (not yet implemented)"""

    def chunk(self, content: str, sent_idx: int) -> str:
        raise NotImplementedError("RawChunking is not yet implemented.")


class SemanticChunking(ChunkingStrategy):
    """Merge adjacent sentences by similarity threshold. (not yet implemented)"""

    def chunk(self, content: str, sent_idx: int) -> str:
        raise NotImplementedError("SemanticChunking is not yet implemented.")


class ContextSemanticChunking(ChunkingStrategy):
    """Window-based semantic merging. (not yet implemented)"""

    def chunk(self, content: str, sent_idx: int) -> str:
        raise NotImplementedError("ContextSemanticChunking is not yet implemented.")


def build_chunking_strategy(cfg: dict | None) -> ChunkingStrategy:
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
        return SemanticChunking()
    if strategy == "ctx_semantic":
        return ContextSemanticChunking()
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
