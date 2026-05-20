from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fact_checking.oracle_pointwise import build_pointwise_inference_pool
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    candidate_text,
)


@dataclass(frozen=True)
class CrossEncoderSelectorConfig:
    model_dir: str
    device: str = "cuda"
    max_length: int = 384
    batch_size: int = 32
    strict_fingerprint: bool = True
    expected_chunk_mmr_fingerprint: str = EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT


class CrossEncoderSelector:
    def __init__(self, cfg: CrossEncoderSelectorConfig) -> None:
        self.cfg = cfg
        self.model_dir = Path(cfg.model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Cross-encoder selector model not found: {self.model_dir}")
        self.metadata = load_cross_encoder_metadata(self.model_dir)
        self._validate_fingerprint()

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        ).eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def score(self, claim: str, candidates: list[str]) -> np.ndarray:
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        scores: list[np.ndarray] = []
        for start in range(0, len(candidates), int(self.cfg.batch_size)):
            batch_candidates = candidates[start : start + int(self.cfg.batch_size)]
            enc = tokenize_claim_candidate_pairs(
                self.tokenizer,
                [claim] * len(batch_candidates),
                batch_candidates,
                max_length=int(self.cfg.max_length),
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            logits = selector_logits(self.model(**enc).logits)
            scores.append(logits.detach().float().cpu().numpy().astype(np.float32))
        return np.concatenate(scores, axis=0)

    def _validate_fingerprint(self) -> None:
        expected = str(self.cfg.expected_chunk_mmr_fingerprint or "")
        if not expected:
            return
        actual = str(self.metadata.get("chunk_mmr_fingerprint") or "")
        if not actual and self.cfg.strict_fingerprint:
            raise ValueError(
                f"Cross-encoder selector {self.model_dir} has no chunk_mmr_fingerprint metadata; "
                f"expected {expected}."
            )
        if actual and actual != expected and self.cfg.strict_fingerprint:
            raise ValueError(
                f"Cross-encoder selector fingerprint mismatch: expected {expected}, "
                f"got {actual} from {self.model_dir}."
            )


def tokenize_claim_candidate_pairs(
    tokenizer: Any,
    claims: list[str],
    candidate_texts: list[str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    left = [f"Claim: {claim}" for claim in claims]
    right = [f"Evidence: {text}" for text in candidate_texts]
    return tokenizer(
        left,
        right,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )


def selector_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[-1] == 1:
        return logits[:, 0]
    if logits.ndim == 2 and logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    if logits.ndim == 1:
        return logits
    return logits.reshape(logits.shape[0], -1)[:, 0]


def pairwise_selector_loss(
    grouped_scores: list[torch.Tensor],
    selected_indices: list[list[int]],
    *,
    candidate_scores: list[list[dict[str, Any]]] | None = None,
    bce_weight: float = 0.3,
    order_weight: float = 0.5,
    top_k: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair_losses: list[torch.Tensor] = []
    order_losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    n_pairs = 0
    n_order_pairs = 0

    for group_idx, scores in enumerate(grouped_scores):
        selected = [idx for idx in selected_indices[group_idx] if 0 <= idx < scores.numel()]
        selected_set = set(selected)
        if not selected:
            continue

        labels = torch.zeros_like(scores)
        labels[selected] = 1.0
        bce_losses.append(nn.functional.binary_cross_entropy_with_logits(scores, labels))

        negatives = [idx for idx in range(scores.numel()) if idx not in selected_set]
        neg_weights = _negative_weights(
            negatives,
            candidate_scores[group_idx] if candidate_scores else None,
            top_k=top_k,
        )
        for pos_idx in selected:
            for neg_idx, neg_weight in zip(negatives, neg_weights):
                loss = nn.functional.softplus(-(scores[pos_idx] - scores[neg_idx]))
                pair_losses.append(loss * float(neg_weight))
                n_pairs += 1

        for rank_a, idx_a in enumerate(selected):
            for rank_b, idx_b in enumerate(selected[rank_a + 1 :], start=rank_a + 1):
                position_weight = 1.0 / np.log2(rank_a + 2.0)
                loss = nn.functional.softplus(-(scores[idx_a] - scores[idx_b]))
                order_losses.append(loss * float(position_weight))
                n_order_pairs += 1

    device = grouped_scores[0].device if grouped_scores else torch.device("cpu")
    zero = torch.zeros((), device=device)
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    bce_loss = torch.stack(bce_losses).mean() if bce_losses else zero
    order_loss = torch.stack(order_losses).mean() if order_losses else zero
    total = pair_loss + float(bce_weight) * bce_loss + float(order_weight) * order_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "bce_loss": float(bce_loss.detach().cpu()),
        "order_loss": float(order_loss.detach().cpu()),
        "n_pairs": float(n_pairs),
        "n_order_pairs": float(n_order_pairs),
    }


def split_flat_scores(flat_scores: torch.Tensor, group_sizes: list[int]) -> list[torch.Tensor]:
    grouped: list[torch.Tensor] = []
    offset = 0
    for size in group_sizes:
        grouped.append(flat_scores[offset : offset + int(size)])
        offset += int(size)
    return grouped


def score_examples(
    selector: CrossEncoderSelector,
    examples: list[Any],
) -> list[np.ndarray]:
    all_scores: list[np.ndarray] = []
    for example in examples:
        texts = [candidate_text(candidate) for candidate in example.candidates]
        all_scores.append(selector.score(example.claim, texts))
    return all_scores


def select_candidates_cross_encoder(
    sample: Any,
    selector: CrossEncoderSelector,
    *,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    candidate_pool_size: int | None = DEFAULT_CANDIDATE_POOL_SIZE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates, _features, _source_indices, source_candidate_count = build_pointwise_inference_pool(
        sample,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
        candidate_pool_size=candidate_pool_size,
    )
    if not candidates:
        return {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": [],
        }, {
            "event_id": sample.event_id,
            "label": sample.label,
            "top_k": int(top_k),
            "candidate_pool_size": int(candidate_pool_size or 0),
            "n_source_candidates": int(source_candidate_count),
            "n_pool_candidates": 0,
            "selected": [],
            "model_dir": str(selector.model_dir),
        }

    texts = [candidate_text(candidate) for candidate in candidates]
    scores = selector.score(str(sample.claim), texts)
    order = np.argsort(-scores)[: min(int(top_k), len(candidates))]
    selected: list[dict[str, Any]] = []
    trace_selected: list[dict[str, Any]] = []
    for rank, pool_idx in enumerate(order.astype(int).tolist()):
        candidate = dict(candidates[pool_idx])
        score = float(scores[pool_idx])
        candidate.update({
            "cross_encoder_score": score,
            "cross_encoder_rank": int(rank),
            "cross_encoder_candidate_pool_size": int(len(candidates)),
        })
        selected.append(candidate)
        trace_selected.append({
            "rank": int(rank),
            "candidate_pool_rank": int(candidate.get("candidate_pool_rank", pool_idx)),
            "source_index": int(candidate.get("source_index", -1)),
            "cross_encoder_score": score,
            "hybrid_score": float(candidate.get("hybrid_score", 0.0)),
            "text": str(candidate.get("text", "")),
            "report_id": str(candidate.get("report_id") or ""),
        })

    trace = {
        "event_id": sample.event_id,
        "label": sample.label,
        "top_k": int(top_k),
        "candidate_pool_size": int(candidate_pool_size or len(candidates)),
        "n_source_candidates": int(source_candidate_count),
        "n_pool_candidates": int(len(candidates)),
        "score_mean": float(scores.mean()) if scores.size else 0.0,
        "score_max": float(scores.max()) if scores.size else 0.0,
        "selected": trace_selected,
        "model_dir": str(selector.model_dir),
        "metadata": selector.metadata,
    }
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": selected,
    }, trace


def load_cross_encoder_metadata(model_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, dict) else {}


def _negative_weights(
    negatives: list[int],
    candidate_scores: list[dict[str, Any]] | None,
    *,
    top_k: int,
) -> list[float]:
    if not negatives:
        return []
    weights: list[float] = []
    for neg_idx in negatives:
        score = candidate_scores[neg_idx] if candidate_scores and neg_idx < len(candidate_scores) else {}
        weight = 1.0
        try:
            rank = int(score.get("hybrid_rank", score.get("candidate_idx", neg_idx)))
        except (TypeError, ValueError):
            rank = neg_idx
        if rank < int(top_k):
            weight += 1.0
        try:
            hybrid = float(score.get("hybrid_score", 0.0))
        except (TypeError, ValueError):
            hybrid = 0.0
        if hybrid >= 0.75:
            weight += 0.5
        weights.append(weight)
    return weights

