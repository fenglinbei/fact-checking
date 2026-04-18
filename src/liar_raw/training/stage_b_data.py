from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from liar_raw import LABEL2ID


class StageBDataset(Dataset):
    def __init__(self, candidates_path: str | Path, top_k: int = 24) -> None:
        self.rows: list[dict[str, Any]] = []
        self.top_k = int(top_k)
        with Path(candidates_path).open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["candidates"] = sorted(
                    row.get("candidates", []),
                    key=lambda x: x.get("hybrid_score", 0.0),
                    reverse=True,
                )[: self.top_k]
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "event_id": row["event_id"],
            "claim": row["claim"],
            "label_id": LABEL2ID[row["label"]],
            "label": row["label"],
            "explain": row.get("explain", ""),
            "candidates": row.get("candidates", []),
        }


class StageBCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        top_k: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.top_k = int(top_k)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        flat_claims: list[str] = []
        flat_sentences: list[str] = []
        dense_scores = torch.zeros(batch_size, self.top_k, dtype=torch.float)
        hybrid_scores = torch.zeros(batch_size, self.top_k, dtype=torch.float)
        candidate_mask = torch.zeros(batch_size, self.top_k, dtype=torch.bool)
        metadata: list[list[dict[str, Any] | None]] = []
        labels = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        event_ids = [item["event_id"] for item in batch]
        claims = [item["claim"] for item in batch]

        for b_idx, item in enumerate(batch):
            claim = item["claim"]
            metas: list[dict[str, Any] | None] = []
            candidates = item["candidates"][: self.top_k]
            for k in range(self.top_k):
                if k < len(candidates):
                    cand = candidates[k]
                    sent = str(cand.get("text", ""))
                    flat_claims.append(claim)
                    flat_sentences.append(sent)
                    dense_scores[b_idx, k] = float(cand.get("dense_score", 0.0))
                    hybrid_scores[b_idx, k] = float(cand.get("hybrid_score", 0.0))
                    candidate_mask[b_idx, k] = bool(sent.strip())
                    metas.append(cand)
                else:
                    flat_claims.append(claim)
                    flat_sentences.append("[PAD]")
                    metas.append(None)
            metadata.append(metas)

        enc = self.tokenizer(
            flat_claims,
            flat_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "event_ids": event_ids,
            "claims": claims,
            "labels": labels,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "candidate_mask": candidate_mask,
            "dense_scores": dense_scores,
            "hybrid_scores": hybrid_scores,
            "metadata": metadata,
        }



def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
