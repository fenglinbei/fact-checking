from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from fact_checking.oracle_pointwise import build_pointwise_inference_pool
from fact_checking.selectors.cross_encoder import tokenize_claim_candidate_pairs
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    candidate_text,
)


LISTWISE_HEAD_FILENAME = "listwise_head.pt"

NUMERIC_FEATURE_NAMES = [
    "hybrid_score",
    "dense_score",
    "lexical_score",
    "bm25_log_norm",
    "hybrid_rank_norm",
    "candidate_idx_norm",
    "sent_idx_norm",
    "source_index_norm",
    "text_token_len_norm",
    "claim_token_overlap",
    "number_overlap",
]

FEATURE_ABLATION_NONE = "none"
FEATURE_ABLATION_NO_RANK_PRIOR = "no_rank_prior"
FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR = "hybrid_score_only_prior"
FEATURE_ABLATION_CONTENT_FEATURES_ONLY = "content_features_only"
FEATURE_ABLATION_TEXT_ONLY = "text_only"
FEATURE_ABLATION_CHOICES = (
    FEATURE_ABLATION_NONE,
    FEATURE_ABLATION_NO_RANK_PRIOR,
    FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR,
    FEATURE_ABLATION_CONTENT_FEATURES_ONLY,
    FEATURE_ABLATION_TEXT_ONLY,
)
_RANK_PRIOR_FEATURES = {"hybrid_rank_norm", "candidate_idx_norm"}
_RETRIEVAL_COMPONENT_FEATURES = {"dense_score", "lexical_score", "bm25_log_norm"}
_CONTENT_FEATURES = {"text_token_len_norm", "claim_token_overlap", "number_overlap"}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ListwiseSelectorConfig:
    model_dir: str
    device: str = "cuda"
    max_length: int = 384
    batch_size: int = 8
    strict_fingerprint: bool = True
    expected_chunk_mmr_fingerprint: str = EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT


class SetAwareListwiseSelectorModel(nn.Module):
    def __init__(
        self,
        encoder_name_or_path: str,
        *,
        numeric_feature_dim: int = len(NUMERIC_FEATURE_NAMES),
        hidden_size: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
        feature_ablation: str = FEATURE_ABLATION_NONE,
        use_rank_embedding: bool | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        encoder_hidden = int(getattr(self.encoder.config, "hidden_size", 768))
        self.numeric_feature_dim = int(numeric_feature_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.dropout = float(dropout)
        self.max_candidates = int(max_candidates)
        self.feature_ablation = normalize_feature_ablation(feature_ablation)
        if use_rank_embedding is None:
            use_rank_embedding = self.feature_ablation == FEATURE_ABLATION_NONE
        self.use_rank_embedding = bool(use_rank_embedding)

        numeric_hidden = max(32, min(128, self.hidden_size // 2))
        self.numeric_projection = nn.Sequential(
            nn.LayerNorm(self.numeric_feature_dim),
            nn.Linear(self.numeric_feature_dim, numeric_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(encoder_hidden + numeric_hidden, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.rank_embedding = (
            nn.Embedding(self.max_candidates + 1, self.hidden_size)
            if self.use_rank_embedding
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(
        self,
        encoded_inputs: dict[str, torch.Tensor],
        *,
        group_sizes: list[int],
        numeric_features: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(**encoded_inputs)
        attention_mask = encoded_inputs.get("attention_mask")
        pair_embeddings = pool_pair_embeddings(outputs, attention_mask)
        numeric = numeric_features.to(device=pair_embeddings.device, dtype=pair_embeddings.dtype)
        numeric_embeddings = self.numeric_projection(numeric)
        item_embeddings = self.item_projection(torch.cat([pair_embeddings, numeric_embeddings], dim=-1))

        padded_items, mask = pad_flat_items(item_embeddings, group_sizes)
        padded_ranks, _ = pad_flat_ranks(
            candidate_ranks.to(device=pair_embeddings.device),
            group_sizes,
            max_candidates=self.max_candidates,
        )
        x = padded_items
        if self.rank_embedding is not None:
            x = x + self.rank_embedding(padded_ranks)
        x = self.set_encoder(x, src_key_padding_mask=~mask)
        scores = self.scorer(self.output_norm(x)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1.0e4)
        return scores, mask

    def selector_head_state_dict(self) -> dict[str, Any]:
        payload = {
            "numeric_projection": self.numeric_projection.state_dict(),
            "item_projection": self.item_projection.state_dict(),
            "set_encoder": self.set_encoder.state_dict(),
            "output_norm": self.output_norm.state_dict(),
            "scorer": self.scorer.state_dict(),
        }
        if self.rank_embedding is not None:
            payload["rank_embedding"] = self.rank_embedding.state_dict()
        return payload

    def load_selector_head_state_dict(self, payload: dict[str, Any]) -> None:
        self.numeric_projection.load_state_dict(payload["numeric_projection"])
        self.item_projection.load_state_dict(payload["item_projection"])
        if self.rank_embedding is not None:
            self.rank_embedding.load_state_dict(payload["rank_embedding"])
        self.set_encoder.load_state_dict(payload["set_encoder"])
        self.output_norm.load_state_dict(payload["output_norm"])
        self.scorer.load_state_dict(payload["scorer"])

    def model_config(self) -> dict[str, Any]:
        return {
            "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
            "numeric_feature_dim": self.numeric_feature_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "max_candidates": self.max_candidates,
            "feature_ablation": self.feature_ablation,
            "dropped_numeric_feature_names": dropped_numeric_feature_names(self.feature_ablation),
            "use_rank_embedding": self.use_rank_embedding,
        }


class ListwiseSelector:
    def __init__(self, cfg: ListwiseSelectorConfig) -> None:
        self.cfg = cfg
        self.model_dir = Path(cfg.model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Listwise selector model not found: {self.model_dir}")
        self.metadata = load_listwise_metadata(self.model_dir)
        self._validate_fingerprint()

        model_cfg = dict(self.metadata.get("model_config") or {})
        feature_ablation = normalize_feature_ablation(
            model_cfg.get("feature_ablation", self.metadata.get("feature_ablation", FEATURE_ABLATION_NONE))
        )
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = SetAwareListwiseSelectorModel(
            str(self.model_dir),
            numeric_feature_dim=int(model_cfg.get("numeric_feature_dim", len(NUMERIC_FEATURE_NAMES))),
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_candidates=int(model_cfg.get("max_candidates", DEFAULT_CANDIDATE_POOL_SIZE)),
            feature_ablation=feature_ablation,
            use_rank_embedding=bool(
                model_cfg.get("use_rank_embedding", feature_ablation == FEATURE_ABLATION_NONE)
            ),
        )
        head_path = self.model_dir / LISTWISE_HEAD_FILENAME
        if not head_path.exists():
            raise FileNotFoundError(f"Listwise selector head not found: {head_path}")
        state = torch.load(head_path, map_location="cpu")
        self.model.load_selector_head_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
        candidate_scores: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        group = ListwiseCandidateGroup(
            claim=str(claim),
            candidates=[dict(item) for item in candidates],
            candidate_scores=_default_candidate_scores(candidates, candidate_scores),
        )
        grouped_scores = forward_listwise_groups(
            self.model,
            self.tokenizer,
            [group],
            device=self.device,
            max_length=int(self.cfg.max_length),
            max_candidates=int(self.metadata.get("max_candidates", DEFAULT_CANDIDATE_POOL_SIZE)),
        )
        return grouped_scores[0].detach().float().cpu().numpy().astype(np.float32)

    def _validate_fingerprint(self) -> None:
        expected = str(self.cfg.expected_chunk_mmr_fingerprint or "")
        if not expected:
            return
        actual = str(self.metadata.get("chunk_mmr_fingerprint") or "")
        if not actual and self.cfg.strict_fingerprint:
            raise ValueError(
                f"Listwise selector {self.model_dir} has no chunk_mmr_fingerprint metadata; "
                f"expected {expected}."
            )
        if actual and actual != expected and self.cfg.strict_fingerprint:
            raise ValueError(
                f"Listwise selector fingerprint mismatch: expected {expected}, "
                f"got {actual} from {self.model_dir}."
            )


@dataclass(frozen=True)
class ListwiseCandidateGroup:
    claim: str
    candidates: list[dict[str, Any]]
    candidate_scores: list[dict[str, Any]]


def forward_listwise_examples(
    model: SetAwareListwiseSelectorModel,
    tokenizer: Any,
    examples: list[Stage2OracleExample],
    *,
    device: torch.device,
    max_length: int,
    max_candidates: int,
) -> list[torch.Tensor]:
    groups = [
        ListwiseCandidateGroup(
            claim=example.claim,
            candidates=example.candidates,
            candidate_scores=example.candidate_scores,
        )
        for example in examples
    ]
    return forward_listwise_groups(
        model,
        tokenizer,
        groups,
        device=device,
        max_length=max_length,
        max_candidates=max_candidates,
    )


def forward_listwise_groups(
    model: SetAwareListwiseSelectorModel,
    tokenizer: Any,
    groups: list[ListwiseCandidateGroup],
    *,
    device: torch.device,
    max_length: int,
    max_candidates: int,
) -> list[torch.Tensor]:
    claims: list[str] = []
    texts: list[str] = []
    group_sizes: list[int] = []
    feature_rows: list[list[float]] = []
    ranks: list[int] = []
    feature_ablation = _model_feature_ablation(model)

    for group in groups:
        group_sizes.append(len(group.candidates))
        for idx, candidate in enumerate(group.candidates):
            score = group.candidate_scores[idx] if idx < len(group.candidate_scores) else {}
            claims.append(group.claim)
            texts.append(candidate_text(candidate))
            feature_rows.append(
                build_numeric_features(
                    group.claim,
                    candidate,
                    score,
                    idx=idx,
                    max_candidates=max_candidates,
                    feature_ablation=feature_ablation,
                )
            )
            ranks.append(
                _safe_int(
                    score.get("hybrid_rank", score.get("candidate_pool_rank", score.get("candidate_idx", idx))),
                    idx,
                )
            )

    enc = tokenize_claim_candidate_pairs(tokenizer, claims, texts, max_length=int(max_length))
    enc = {key: value.to(device) for key, value in enc.items()}
    numeric_features = torch.tensor(feature_rows, dtype=torch.float32, device=device)
    candidate_ranks = torch.tensor(ranks, dtype=torch.long, device=device)
    padded_scores, mask = model(
        enc,
        group_sizes=group_sizes,
        numeric_features=numeric_features,
        candidate_ranks=candidate_ranks,
    )
    return split_padded_scores(padded_scores, mask)


def listwise_selector_loss(
    score_groups: list[torch.Tensor],
    selected_indices: list[list[int]],
    *,
    mask_weight: float = 0.3,
    listmle_weight: float = 1.0,
    order_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_losses: list[torch.Tensor] = []
    listmle_losses: list[torch.Tensor] = []
    order_losses: list[torch.Tensor] = []
    n_order_pairs = 0
    n_list_steps = 0

    for group_idx, scores in enumerate(score_groups):
        selected = [idx for idx in selected_indices[group_idx] if 0 <= idx < scores.numel()]
        if not selected:
            continue

        labels = torch.zeros_like(scores)
        labels[selected] = 1.0
        mask_losses.append(nn.functional.binary_cross_entropy_with_logits(scores, labels))

        remaining = list(range(scores.numel()))
        for rank, idx in enumerate(selected):
            if idx not in remaining:
                continue
            position_weight = 1.0 / np.log2(rank + 2.0)
            remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=scores.device)
            listmle_losses.append(
                (torch.logsumexp(scores.index_select(0, remaining_tensor), dim=0) - scores[idx])
                * float(position_weight)
            )
            remaining = [candidate_idx for candidate_idx in remaining if candidate_idx != idx]
            n_list_steps += 1

        for rank_a, idx_a in enumerate(selected):
            for idx_b in selected[rank_a + 1 :]:
                position_weight = 1.0 / np.log2(rank_a + 2.0)
                order_losses.append(
                    nn.functional.softplus(-(scores[idx_a] - scores[idx_b])) * float(position_weight)
                )
                n_order_pairs += 1

    device = score_groups[0].device if score_groups else torch.device("cpu")
    zero = torch.zeros((), device=device)
    mask_loss = torch.stack(mask_losses).mean() if mask_losses else zero
    listmle_loss = torch.stack(listmle_losses).mean() if listmle_losses else zero
    order_loss = torch.stack(order_losses).mean() if order_losses else zero
    total = (
        float(mask_weight) * mask_loss
        + float(listmle_weight) * listmle_loss
        + float(order_weight) * order_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "mask_loss": float(mask_loss.detach().cpu()),
        "listmle_loss": float(listmle_loss.detach().cpu()),
        "order_loss": float(order_loss.detach().cpu()),
        "n_list_steps": float(n_list_steps),
        "n_order_pairs": float(n_order_pairs),
    }


def build_numeric_features(
    claim: str,
    candidate: dict[str, Any],
    candidate_score: dict[str, Any] | None,
    *,
    idx: int,
    max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
    feature_ablation: str = FEATURE_ABLATION_NONE,
) -> list[float]:
    score = candidate_score or {}
    text = candidate_text(candidate)
    rank = _safe_int(
        score.get("hybrid_rank", score.get("candidate_pool_rank", score.get("candidate_idx", idx))),
        idx,
    )
    candidate_idx = _safe_int(score.get("candidate_idx", candidate.get("candidate_idx", idx)), idx)
    sent_idx = _safe_int(candidate.get("sent_idx", 0), 0)
    source_index = _safe_int(candidate.get("source_index", score.get("source_index", 0)), 0)
    text_tokens = _tokens(text)
    claim_tokens = _content_tokens(claim)
    overlap = len(set(claim_tokens) & set(_content_tokens(text))) / max(len(set(claim_tokens)), 1)
    claim_numbers = set(_NUMBER_RE.findall(claim))
    text_numbers = set(_NUMBER_RE.findall(text))
    number_overlap = len(claim_numbers & text_numbers) / max(len(claim_numbers), 1) if claim_numbers else 0.0
    denom = max(int(max_candidates) - 1, 1)
    features = [
        _safe_float(score.get("hybrid_score", candidate.get("hybrid_score", 0.0)), 0.0),
        _safe_float(score.get("dense_score", candidate.get("dense_score", 0.0)), 0.0),
        _safe_float(score.get("lexical_score", candidate.get("lexical_score", 0.0)), 0.0),
        min(np.log1p(max(_safe_float(score.get("bm25_score", candidate.get("bm25_score", 0.0)), 0.0), 0.0)) / 5.0, 1.0),
        min(max(rank / denom, 0.0), 1.0),
        min(max(candidate_idx / denom, 0.0), 1.0),
        min(max(sent_idx / 100.0, 0.0), 1.0),
        min(max(source_index / 200.0, 0.0), 1.0),
        min(len(text_tokens) / 80.0, 1.0),
        float(min(max(overlap, 0.0), 1.0)),
        float(min(max(number_overlap, 0.0), 1.0)),
    ]
    return apply_numeric_feature_ablation(features, feature_ablation)


def normalize_feature_ablation(feature_ablation: Any) -> str:
    value = str(feature_ablation or FEATURE_ABLATION_NONE).strip().lower()
    if value in {"", "false", "off", "full"}:
        value = FEATURE_ABLATION_NONE
    if value not in FEATURE_ABLATION_CHOICES:
        choices = ", ".join(FEATURE_ABLATION_CHOICES)
        raise ValueError(f"Unknown listwise feature ablation mode: {value!r}; choices: {choices}")
    return value


def dropped_numeric_feature_names(feature_ablation: Any) -> list[str]:
    mode = normalize_feature_ablation(feature_ablation)
    dropped: set[str] = set()
    if mode in {FEATURE_ABLATION_NO_RANK_PRIOR, FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR}:
        dropped.update(_RANK_PRIOR_FEATURES)
    if mode == FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR:
        dropped.update(_RETRIEVAL_COMPONENT_FEATURES)
    if mode == FEATURE_ABLATION_CONTENT_FEATURES_ONLY:
        dropped.update(set(NUMERIC_FEATURE_NAMES) - _CONTENT_FEATURES)
    if mode == FEATURE_ABLATION_TEXT_ONLY:
        dropped.update(NUMERIC_FEATURE_NAMES)
    return [name for name in NUMERIC_FEATURE_NAMES if name in dropped]


def apply_numeric_feature_ablation(features: list[float], feature_ablation: Any) -> list[float]:
    dropped = set(dropped_numeric_feature_names(feature_ablation))
    if not dropped:
        return [float(value) for value in features]
    return [
        0.0
        if idx < len(NUMERIC_FEATURE_NAMES) and NUMERIC_FEATURE_NAMES[idx] in dropped
        else float(value)
        for idx, value in enumerate(features)
    ]


def _model_feature_ablation(model: Any) -> str:
    base_model = getattr(model, "module", model)
    return normalize_feature_ablation(getattr(base_model, "feature_ablation", FEATURE_ABLATION_NONE))


def select_candidates_listwise(
    sample: Any,
    selector: ListwiseSelector,
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

    scores = selector.score(str(sample.claim), candidates, candidates)
    order = np.argsort(-scores)[: min(int(top_k), len(candidates))]
    selected: list[dict[str, Any]] = []
    trace_selected: list[dict[str, Any]] = []
    for rank, pool_idx in enumerate(order.astype(int).tolist()):
        candidate = dict(candidates[pool_idx])
        score = float(scores[pool_idx])
        candidate.update({
            "listwise_score": score,
            "listwise_rank": int(rank),
            "listwise_candidate_pool_size": int(len(candidates)),
        })
        selected.append(candidate)
        trace_selected.append({
            "rank": int(rank),
            "candidate_pool_rank": int(candidate.get("candidate_pool_rank", pool_idx)),
            "source_index": int(candidate.get("source_index", -1)),
            "listwise_score": score,
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


def load_listwise_metadata(model_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, dict) else {}


def pool_pair_embeddings(outputs: Any, attention_mask: torch.Tensor | None) -> torch.Tensor:
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = outputs.last_hidden_state
    if attention_mask is None:
        return hidden[:, 0]
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def pad_flat_items(flat: torch.Tensor, group_sizes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(group_sizes)
    max_group = max([int(size) for size in group_sizes] or [0])
    padded = flat.new_zeros((batch_size, max_group, flat.shape[-1]))
    mask = torch.zeros((batch_size, max_group), dtype=torch.bool, device=flat.device)
    offset = 0
    for row, size in enumerate(group_sizes):
        size = int(size)
        if size > 0:
            padded[row, :size] = flat[offset : offset + size]
            mask[row, :size] = True
        offset += size
    return padded, mask


def pad_flat_ranks(
    ranks: torch.Tensor,
    group_sizes: list[int],
    *,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(group_sizes)
    max_group = max([int(size) for size in group_sizes] or [0])
    padded = torch.full(
        (batch_size, max_group),
        fill_value=int(max_candidates),
        dtype=torch.long,
        device=ranks.device,
    )
    mask = torch.zeros((batch_size, max_group), dtype=torch.bool, device=ranks.device)
    offset = 0
    for row, size in enumerate(group_sizes):
        size = int(size)
        if size > 0:
            padded[row, :size] = ranks[offset : offset + size].clamp(0, int(max_candidates))
            mask[row, :size] = True
        offset += size
    return padded, mask


def split_padded_scores(padded_scores: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    groups: list[torch.Tensor] = []
    for row in range(padded_scores.shape[0]):
        size = int(mask[row].sum().item())
        groups.append(padded_scores[row, :size])
    return groups


def _default_candidate_scores(
    candidates: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if candidate_scores is not None and len(candidate_scores) == len(candidates):
        return [dict(item) for item in candidate_scores]
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        row = dict(candidate)
        row.setdefault("candidate_idx", idx)
        row.setdefault("hybrid_rank", row.get("candidate_pool_rank", idx))
        rows.append(row)
    return rows


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text))]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if len(token) > 2]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
