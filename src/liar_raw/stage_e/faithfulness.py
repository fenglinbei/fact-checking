from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from liar_raw.utils.text import clean_text, jaccard


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = []
    for chunk in text.split("."):
        chunk = clean_text(chunk)
        if chunk:
            parts.append(chunk + ".")
    return parts


class FaithfulnessFilter:
    def __init__(
        self,
        embedder_model: str,
        semantic_threshold: float = 0.50,
        lexical_jaccard_threshold: float = 0.12,
        device: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(embedder_model, device=device)
        self.semantic_threshold = semantic_threshold
        self.lexical_jaccard_threshold = lexical_jaccard_threshold

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def filter_or_fallback(self, generated: str, evidence_texts: list[str], fallback: str) -> str:
        generated_sents = split_sentences(generated)
        evidence_texts = [clean_text(x) for x in evidence_texts if clean_text(x)]

        if not generated_sents or not evidence_texts:
            return fallback

        gen_embs = self._encode(generated_sents)
        ev_embs = self._encode(evidence_texts)

        kept = []
        for i, sent in enumerate(generated_sents):
            sims = ev_embs @ gen_embs[i]
            max_sem = float(np.max(sims))
            max_lex = max(jaccard(sent, ev) for ev in evidence_texts)
            if max_sem >= self.semantic_threshold or max_lex >= self.lexical_jaccard_threshold:
                kept.append(sent)

        if not kept:
            return fallback
        return " ".join(kept)
