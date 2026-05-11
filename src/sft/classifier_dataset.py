from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from fact_checking.data.io import load_jsonl


class ClassifierDataset(torch.utils.data.Dataset):
    """Dataset for discriminative fact-checking classification.

    Reads the build-stage JSONL where each row contains:
    - claim: str
    - candidates: list[dict] with at least `text` and `hybrid_score` (already sorted desc)
    - gold_id: int in [0, 5] or -1 for unknown labels (filtered out)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        top_k_evidence: int = 16,
        max_length: int = 2048,
        label_map: dict[int, int] | None = None,
    ) -> None:
        rows = load_jsonl(jsonl_path)
        self.rows = [row for row in rows if int(row.get("gold_id", -1)) >= 0]
        self.tokenizer = tokenizer
        self.top_k = int(top_k_evidence)
        self.max_length = int(max_length)
        self.label_map = label_map
        sep = tokenizer.sep_token or tokenizer.eos_token or "\n"
        self.sep = f" {sep} "

    def __len__(self) -> int:
        return len(self.rows)

    def _build_text(self, row: dict[str, Any]) -> str:
        claim = str(row.get("claim", "")).strip()
        cands = row.get("candidates", []) or []
        evidence_texts = [
            str(c.get("text", "")).strip()
            for c in cands[: self.top_k]
            if isinstance(c, dict) and str(c.get("text", "")).strip()
        ]
        if not evidence_texts:
            return claim
        return claim + self.sep + self.sep.join(evidence_texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        text = self._build_text(row)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        gold_id = int(row["gold_id"])
        if self.label_map is not None:
            gold_id = self.label_map[gold_id]
        enc["labels"] = gold_id
        return enc
