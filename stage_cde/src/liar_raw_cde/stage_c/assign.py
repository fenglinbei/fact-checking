from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from liar_raw_cde.utils.text import cosine_from_numpy


def _extract_sentence(evidence_item: dict[str, Any]) -> str:
    return str(
        evidence_item.get("sentence")
        or evidence_item.get("sent")
        or evidence_item.get("text")
        or ""
    ).strip()


class SubclaimEvidenceAssigner:
    def __init__(
        self,
        embedder_model: str,
        top_k_support_per_subclaim: int = 2,
        top_k_refute_per_subclaim: int = 2,
        device: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(embedder_model, device=device)
        self.top_k_support_per_subclaim = top_k_support_per_subclaim
        self.top_k_refute_per_subclaim = top_k_refute_per_subclaim

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _rank(self, query_emb: np.ndarray, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not items:
            return []

        texts = [_extract_sentence(x) for x in items]
        doc_embs = self.encode(texts)
        scores = [cosine_from_numpy(query_emb, emb) for emb in doc_embs]

        ranked_idx = list(np.argsort([-s for s in scores]))[: min(top_k, len(items))]
        ranked: list[dict[str, Any]] = []
        for idx in ranked_idx:
            item = dict(items[idx])
            item["subclaim_similarity"] = float(scores[idx])
            ranked.append(item)
        return ranked

    def assign(
        self,
        claim: str,
        subclaims: list[str],
        support_evidence: list[dict[str, Any]],
        refute_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not subclaims:
            return []

        claim_emb = self.encode([claim])[0]
        sub_embs = self.encode(subclaims)
        assignments: list[dict[str, Any]] = []
        for subclaim, emb in zip(subclaims, sub_embs):
            assigned_support = self._rank(emb, support_evidence, self.top_k_support_per_subclaim)
            assigned_refute = self._rank(emb, refute_evidence, self.top_k_refute_per_subclaim)
            claim_similarity = cosine_from_numpy(claim_emb, emb)
            assignments.append(
                {
                    "subclaim": subclaim,
                    "subclaim_claim_similarity": float(claim_similarity),
                    "support_evidence": assigned_support,
                    "refute_evidence": assigned_refute,
                }
            )
        return assignments
