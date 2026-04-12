from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(slots=True)
class EmbedderConfig:
    model_name: str
    device: str = "cuda"
    max_length: int = 256
    batch_size: int = 64
    normalize: bool = True


class TextEmbedder:
    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _apply_instruction(self, texts: Iterable[str], is_query: bool) -> list[str]:
        texts = list(texts)
        name = self.cfg.model_name.lower()
        if "bge" in name and is_query:
            return [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        if "e5" in name:
            prefix = "query: " if is_query else "passage: "
            return [prefix + t for t in texts]
        return texts

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-8)
        return summed / denom

    @torch.inference_mode()
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        texts = self._apply_instruction(texts, is_query=is_query)
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = texts[start : start + self.cfg.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            pooled = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
            if self.cfg.normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy())
        if not vectors:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(vectors, axis=0).astype(np.float32)
