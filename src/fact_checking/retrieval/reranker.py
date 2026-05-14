from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class RerankerConfig:
    model_name: str = "/data/models/Qwen3-Reranker-0.6B"
    device: str = "cuda"
    max_length: int = 4096
    batch_size: int = 4
    normalize: bool = True

_PRECISION_MAP: dict[str, torch.dtype | None] = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_QWEN_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_DEFAULT_TASK = "Given a claim, retrieve relevant evidence that supports or refutes the claim"


def _detect_model_type(model_path: str) -> str:
    """Read config.json to determine whether the model is a causal-LM reranker or a
    sequence-classification cross-encoder."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        architectures = cfg.get("architectures", [])
        model_type = cfg.get("model_type", "")
        if any("ForCausalLM" in a for a in architectures) or model_type in ("qwen3", "qwen2", "llama"):
            return "causal"
        if any("ForSequenceClassification" in a for a in architectures):
            return "sequence_classification"
    # Fallback: try loading tokenizer and infer from chat_template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.chat_template is not None and "Qwen" in str(type(tokenizer)):
        return "causal"
    return "sequence_classification"


class CrossEncoderReranker:
    """Cross-encoder reranker that supports two model families:

    - **causal** (Qwen3-Reranker): uses a causal LM with yes/no token log-probabilities
      to produce a relevance score in [0, 1].
    - **sequence_classification** (BGE-reranker): uses a BERT-style classification head,
      optionally applying sigmoid.
    """

    def __init__(self, cfg: RerankerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self._model_type = _detect_model_type(cfg.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

        load_kwargs = {}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16

        if self._model_type == "causal":
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, **load_kwargs,
            ).eval()
            prefix_tokens = self.tokenizer.encode(_QWEN_RERANKER_PREFIX, add_special_tokens=False)
            suffix_tokens = self.tokenizer.encode(_QWEN_RERANKER_SUFFIX, add_special_tokens=False)
            self._causal_overhead = len(prefix_tokens) + len(suffix_tokens)
            self._prefix_tokens = prefix_tokens
            self._suffix_tokens = suffix_tokens
            self._yes_token_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
            self._no_token_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        else:
            self.model = AutoModel.from_pretrained(
                cfg.model_name, **load_kwargs,
            ).eval()

        self.model.to(self.device)

    @torch.inference_mode()
    def score(self, claim: str, candidates: list[str]) -> np.ndarray:
        """Score claim against each candidate text.

        Returns float32 array of shape [len(candidates)], values in [0, 1].
        """
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(claim, c) for c in candidates]
        return self.score_pairs(pairs)

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score a list of (query, document) pairs.

        Returns float32 array of shape [len(pairs)].
        """
        if not pairs:
            return np.zeros((0,), dtype=np.float32)

        if self._model_type == "causal":
            return self._score_causal(pairs)
        return self._score_seq_class(pairs)

    # ------------------------------------------------------------------
    # Causal-LM reranker path (Qwen3-Reranker)
    # ------------------------------------------------------------------

    def _score_causal(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        all_scores: list[float] = []
        for start in range(0, len(pairs), self.cfg.batch_size):
            batch = pairs[start : start + self.cfg.batch_size]
            formatted = [
                f"<Instruct>: {_QWEN_DEFAULT_TASK}\n<Query>: {q}\n<Document>: {d}"
                for q, d in batch
            ]
            enc = self.tokenizer(formatted, padding=False, truncation=True,
                                return_attention_mask=False,
                                max_length=self.cfg.max_length - self._causal_overhead)
            for i, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][i] = self._prefix_tokens + ids + self._suffix_tokens
            padded = self.tokenizer.pad(
                {"input_ids": enc["input_ids"]},
                padding=True, return_tensors="pt",
            )
            padded = {k: v.to(self.device) for k, v in padded.items()}
            logits = self.model(**padded).logits[:, -1, :]  # [bsz, vocab]
            yes_vec = logits[:, self._yes_token_id]
            no_vec = logits[:, self._no_token_id]
            stacked = torch.stack([no_vec, yes_vec], dim=1)
            probs = torch.nn.functional.log_softmax(stacked, dim=1).exp()
            scores = probs[:, 1].cpu().tolist()
            all_scores.extend(scores)
        return np.array(all_scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Sequence-classification path (BGE-reranker)
    # ------------------------------------------------------------------

    def _score_seq_class(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        all_logits: list[np.ndarray] = []
        for start in range(0, len(pairs), self.cfg.batch_size):
            batch = pairs[start : start + self.cfg.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            logits = outputs.logits.squeeze(-1)  # [batch_size]
            if self.cfg.normalize:
                logits = torch.sigmoid(logits)
            all_logits.append(logits.cpu().numpy().astype(np.float32))
        return np.concatenate(all_logits, axis=0)
