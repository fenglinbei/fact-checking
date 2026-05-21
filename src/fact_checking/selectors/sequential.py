from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from fact_checking.oracle_pointwise import build_pointwise_inference_pool
from fact_checking.selectors.base import (
    EvidenceSelector,
    SelectorCandidateGroup,
    SelectorPrediction,
    register_selector_type,
)
from fact_checking.selectors.cross_encoder import tokenize_claim_candidate_pairs
from fact_checking.selectors.listwise import pad_flat_items
from fact_checking.selectors.metrics import ordered_selection_metrics
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    Stage2OracleExample,
    candidate_text,
)


SEQUENTIAL_HEAD_FILENAME = "sequential_head.pt"
SEMANTIC_FEATURE_PROFILE_DEEP = "deep"
TARGETED_FEATURE_PROFILE_NONE = "none"
SHALLOW_FEATURE_PROFILE_OFF = "off"
SEMANTIC_FEATURE_PROFILE_CHOICES = (SEMANTIC_FEATURE_PROFILE_DEEP,)
TARGETED_FEATURE_PROFILE_CHOICES = (TARGETED_FEATURE_PROFILE_NONE,)
SHALLOW_FEATURE_PROFILE_CHOICES = (SHALLOW_FEATURE_PROFILE_OFF,)
PROJECTION_MODE_LINEAR = "linear"
PROJECTION_MODE_MLP_RESIDUAL = "mlp_residual"
PROJECTION_MODE_CHOICES = (PROJECTION_MODE_LINEAR, PROJECTION_MODE_MLP_RESIDUAL)
CLAIM_START_MODE_LEARNED = "learned"
CLAIM_START_MODE_CANDIDATE_POOL_MEAN = "candidate_pool_mean"
CLAIM_START_MODE_CHOICES = (CLAIM_START_MODE_LEARNED, CLAIM_START_MODE_CANDIDATE_POOL_MEAN)
CLAIM_FEATURE_MODE_OFF = "off"
CLAIM_FEATURE_MODE_CLAIM_ONLY = "claim_only"
CLAIM_FEATURE_MODE_CHOICES = (CLAIM_FEATURE_MODE_OFF, CLAIM_FEATURE_MODE_CLAIM_ONLY)
SEQUENTIAL_SELECTOR_TYPE = "sequential_pointer"


@dataclass(frozen=True)
class SequentialSelectorConfig:
    model_dir: str
    device: str = "cuda"
    max_length: int = 384
    batch_size: int = 8
    strict_fingerprint: bool = True
    expected_chunk_mmr_fingerprint: str = EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT


@dataclass(frozen=True)
class SequentialForwardOutput:
    context_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    claim_start: torch.Tensor | None = None
    claim_embedding: torch.Tensor | None = None


class DeepInteractionPointerHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        dropout: float = 0.1,
        bilinear_size: int | None = None,
        claim_feature_mode: str = CLAIM_FEATURE_MODE_OFF,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.bilinear_size = int(bilinear_size or hidden_size)
        self.claim_feature_mode = normalize_claim_feature_mode(claim_feature_mode)
        self.start_prefix = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.start_prefix, mean=0.0, std=0.02)
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, self.bilinear_size)
        interaction_dim = self.hidden_size * 4 + 1 + self.bilinear_size
        if self.claim_feature_mode == CLAIM_FEATURE_MODE_CLAIM_ONLY:
            interaction_dim += self.hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(interaction_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def prefix_representation(
        self,
        context_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
        *,
        claim_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        has_prefix = selected_mask.any(dim=1)
        dtype = context_embeddings.dtype
        start = self.start_prefix.to(dtype=dtype).unsqueeze(0)  # [1, H]
        if not has_prefix.any():
            if claim_start is not None:
                return claim_start.to(dtype=dtype) + start
            return start.expand(context_embeddings.shape[0], -1)
        selected = selected_mask.to(dtype=dtype).unsqueeze(-1)
        counts = selected.sum(dim=1).clamp_min(1.0)
        pooled = (context_embeddings * selected).sum(dim=1) / counts
        if claim_start is not None:
            fallback = claim_start.to(dtype=dtype) + start
        else:
            fallback = start.expand_as(pooled)
        return torch.where(has_prefix.unsqueeze(-1), pooled, fallback)

    def score_step(
        self,
        context_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        *,
        claim_start: torch.Tensor | None = None,
        claim_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix = self.prefix_representation(
            context_embeddings, selected_mask, claim_start=claim_start,
        )
        prefix_expanded = prefix.unsqueeze(1).expand_as(context_embeddings)
        product = context_embeddings * prefix_expanded
        difference = torch.abs(context_embeddings - prefix_expanded)
        cosine = F.cosine_similarity(context_embeddings, prefix_expanded, dim=-1).unsqueeze(-1)
        bilinear = self.bilinear(context_embeddings, prefix_expanded)
        feature_parts = [context_embeddings, prefix_expanded, product, difference, cosine, bilinear]
        if self.claim_feature_mode == CLAIM_FEATURE_MODE_CLAIM_ONLY:
            if claim_embedding is None:
                raise ValueError("claim_embedding is required when claim_feature_mode='claim_only'.")
            claim_expanded = (
                claim_embedding.to(dtype=context_embeddings.dtype)
                .unsqueeze(1)
                .expand_as(context_embeddings)
            )
            feature_parts.append(claim_expanded)
        features = torch.cat(feature_parts, dim=-1)
        logits = self.scorer(features).squeeze(-1)
        return mask_step_logits(logits, candidate_mask, selected_mask)

    def teacher_forcing_logits(
        self,
        context_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        selected_indices: list[list[int]],
        *,
        top_k: int,
        claim_start: torch.Tensor | None = None,
        claim_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, max_candidates = candidate_mask.shape
        max_steps = _max_teacher_steps(selected_indices, top_k=top_k)
        selected_mask = torch.zeros(
            (batch_size, max_candidates),
            dtype=torch.bool,
            device=context_embeddings.device,
        )
        step_logits: list[torch.Tensor] = []
        for step in range(max_steps):
            logits = self.score_step(
                context_embeddings,
                candidate_mask,
                selected_mask,
                claim_start=claim_start,
                claim_embedding=claim_embedding,
            )
            step_logits.append(logits)
            for row, indices in enumerate(selected_indices):
                if step < len(indices):
                    idx = int(indices[step])
                    if 0 <= idx < max_candidates:
                        selected_mask[row, idx] = True
        if not step_logits:
            return context_embeddings.new_zeros((batch_size, 0, max_candidates))
        return torch.stack(step_logits, dim=1)

    @torch.inference_mode()
    def greedy_decode(
        self,
        context_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        top_k: int,
        claim_start: torch.Tensor | None = None,
        claim_embedding: torch.Tensor | None = None,
    ) -> list[SelectorPrediction]:
        batch_size, max_candidates = candidate_mask.shape
        selected_mask = torch.zeros(
            (batch_size, max_candidates),
            dtype=torch.bool,
            device=context_embeddings.device,
        )
        ordered: list[list[int]] = [[] for _ in range(batch_size)]
        step_traces: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
        final_scores = torch.zeros(
            (batch_size, max_candidates),
            dtype=context_embeddings.dtype,
            device=context_embeddings.device,
        )

        for step in range(int(top_k)):
            logits = self.score_step(
                context_embeddings,
                candidate_mask,
                selected_mask,
                claim_start=claim_start,
                claim_embedding=claim_embedding,
            )
            next_idx = torch.argmax(logits, dim=1)
            valid = torch.isfinite(logits.gather(1, next_idx.unsqueeze(1)).squeeze(1))
            valid &= logits.gather(1, next_idx.unsqueeze(1)).squeeze(1) > -1.0e3
            for row in range(batch_size):
                remaining = torch.where(candidate_mask[row] & ~selected_mask[row])[0]
                if remaining.numel() == 0 or not bool(valid[row].item()):
                    continue
                idx = int(next_idx[row].item())
                score = float(logits[row, idx].detach().cpu())
                ordered[row].append(idx)
                final_scores[row, idx] = logits[row, idx]
                selected_mask[row, idx] = True
                step_traces[row].append(_step_trace_from_logits(logits[row], remaining, step, idx, score))

        predictions: list[SelectorPrediction] = []
        for row in range(batch_size):
            valid_count = int(candidate_mask[row].sum().item())
            row_scores = final_scores[row, :valid_count].detach().float().cpu().numpy()
            predictions.append(
                SelectorPrediction(
                    ordered_indices=[int(idx) for idx in ordered[row]],
                    scores=[float(x) for x in row_scores.tolist()],
                    step_trace=step_traces[row],
                    metadata={"selector_type": SEQUENTIAL_SELECTOR_TYPE},
                )
            )
        return predictions


class SequentialPointerSelectorModel(nn.Module):
    def __init__(
        self,
        encoder_name_or_path: str,
        *,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        max_candidates: int = DEFAULT_CANDIDATE_POOL_SIZE,
        semantic_feature_profile: str = SEMANTIC_FEATURE_PROFILE_DEEP,
        targeted_feature_profile: str = TARGETED_FEATURE_PROFILE_NONE,
        shallow_feature_profile: str = SHALLOW_FEATURE_PROFILE_OFF,
        projection_mode: str = PROJECTION_MODE_LINEAR,
        projection_hidden_multiplier: int = 2,
        claim_start_mode: str = CLAIM_START_MODE_LEARNED,
        claim_feature_mode: str = CLAIM_FEATURE_MODE_OFF,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        encoder_hidden = int(getattr(self.encoder.config, "hidden_size", 768))
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.dropout = float(dropout)
        self.max_candidates = int(max_candidates)
        self.semantic_feature_profile = normalize_semantic_feature_profile(semantic_feature_profile)
        self.targeted_feature_profile = normalize_targeted_feature_profile(targeted_feature_profile)
        self.shallow_feature_profile = normalize_shallow_feature_profile(shallow_feature_profile)
        self.projection_mode = normalize_projection_mode(projection_mode)
        self.projection_hidden_multiplier = int(projection_hidden_multiplier)
        if self.projection_hidden_multiplier <= 0:
            raise ValueError(
                f"projection_hidden_multiplier must be positive, got {self.projection_hidden_multiplier}"
            )
        self.claim_start_mode = normalize_claim_start_mode(claim_start_mode)
        self.claim_feature_mode = normalize_claim_feature_mode(claim_feature_mode)

        self.proj_residual: nn.Linear | None = None
        if self.projection_mode == PROJECTION_MODE_MLP_RESIDUAL:
            projection_hidden = self.hidden_size * self.projection_hidden_multiplier
            self.item_projection = nn.Sequential(
                nn.Linear(encoder_hidden, projection_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(projection_hidden, self.hidden_size),
            )
            self.proj_residual = nn.Linear(encoder_hidden, self.hidden_size)
        else:
            self.item_projection = nn.Sequential(
                nn.Linear(encoder_hidden, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
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
        self.pointer_head = DeepInteractionPointerHead(
            self.hidden_size,
            dropout=self.dropout,
            claim_feature_mode=self.claim_feature_mode,
        )

    def forward(
        self,
        encoded_inputs: dict[str, torch.Tensor],
        *,
        group_sizes: list[int],
        claim_encoded_inputs: dict[str, torch.Tensor] | None = None,
    ) -> SequentialForwardOutput:
        outputs = self.encoder(**encoded_inputs)
        pair_embeddings = pool_pair_embeddings(outputs, encoded_inputs.get("attention_mask"))
        item_embeddings = self._project_encoder_embeddings(pair_embeddings)
        claim_embedding = None
        if self.claim_feature_mode == CLAIM_FEATURE_MODE_CLAIM_ONLY:
            if claim_encoded_inputs is None:
                raise ValueError("claim_encoded_inputs is required when claim_feature_mode='claim_only'.")
            claim_outputs = self.encoder(**claim_encoded_inputs)
            claim_raw_embeddings = pool_pair_embeddings(
                claim_outputs,
                claim_encoded_inputs.get("attention_mask"),
            )
            claim_embedding = self._project_encoder_embeddings(claim_raw_embeddings)
        padded_items, mask = pad_flat_items(item_embeddings, group_sizes)
        context = self.set_encoder(padded_items, src_key_padding_mask=~mask)
        context = self.output_norm(context)
        claim_start = None
        if self.claim_start_mode == CLAIM_START_MODE_CANDIDATE_POOL_MEAN:
            claim_start = _build_claim_start(item_embeddings, group_sizes)
        return SequentialForwardOutput(
            context_embeddings=context,
            candidate_mask=mask,
            claim_start=claim_start,
            claim_embedding=claim_embedding,
        )

    def _project_encoder_embeddings(self, encoder_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.item_projection(encoder_embeddings)
        if self.projection_mode == PROJECTION_MODE_MLP_RESIDUAL:
            if self.proj_residual is None:
                raise RuntimeError("proj_residual is required for mlp_residual projection mode")
            residual = self.proj_residual(encoder_embeddings)
            return F.gelu(projected + residual)
        return projected

    def selector_head_state_dict(self) -> dict[str, Any]:
        payload = {
            "item_projection": self.item_projection.state_dict(),
            "set_encoder": self.set_encoder.state_dict(),
            "output_norm": self.output_norm.state_dict(),
            "pointer_head": self.pointer_head.state_dict(),
        }
        if self.proj_residual is not None:
            payload["proj_residual"] = self.proj_residual.state_dict()
        return payload

    def load_selector_head_state_dict(self, payload: dict[str, Any]) -> None:
        self.item_projection.load_state_dict(payload["item_projection"])
        if self.proj_residual is not None:
            if "proj_residual" not in payload:
                raise ValueError(
                    "Sequential selector checkpoint is missing proj_residual for "
                    f"projection_mode={self.projection_mode!r}."
                )
            self.proj_residual.load_state_dict(payload["proj_residual"])
        self.set_encoder.load_state_dict(payload["set_encoder"])
        self.output_norm.load_state_dict(payload["output_norm"])
        self.pointer_head.load_state_dict(payload["pointer_head"])

    def model_config(self) -> dict[str, Any]:
        uses_residual = self.projection_mode == PROJECTION_MODE_MLP_RESIDUAL
        deep_interaction_features = [
            "h_i_pair",
            "H_i_ctx",
            "P_t",
            "H_i_ctx * P_t",
            "abs(H_i_ctx - P_t)",
            "cos(H_i_ctx, P_t)",
            "bilinear(H_i_ctx, P_t)",
        ]
        if self.claim_feature_mode == CLAIM_FEATURE_MODE_CLAIM_ONLY:
            deep_interaction_features.append("h_claim")
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "max_candidates": self.max_candidates,
            "semantic_feature_profile": self.semantic_feature_profile,
            "targeted_feature_profile": self.targeted_feature_profile,
            "shallow_feature_profile": self.shallow_feature_profile,
            "projection_mode": self.projection_mode,
            "projection_hidden_multiplier": self.projection_hidden_multiplier,
            "proj_num_layers": 2 if uses_residual else 1,
            "proj_residual": uses_residual,
            "claim_start_mode": self.claim_start_mode,
            "claim_start": self.claim_start_mode,
            "claim_feature_mode": self.claim_feature_mode,
            "deep_interaction_features": deep_interaction_features,
        }


class SequentialSelector:
    selector_type = SEQUENTIAL_SELECTOR_TYPE

    def __init__(self, cfg: SequentialSelectorConfig) -> None:
        self.cfg = cfg
        self.model_dir = Path(cfg.model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Sequential selector model not found: {self.model_dir}")
        self.metadata = load_sequential_metadata(self.model_dir)
        self._validate_fingerprint()
        model_cfg = dict(self.metadata.get("model_config") or {})
        for key in (
            "max_candidates",
            "semantic_feature_profile",
            "targeted_feature_profile",
            "shallow_feature_profile",
            "projection_mode",
            "projection_hidden_multiplier",
            "claim_start_mode",
            "claim_feature_mode",
        ):
            if key not in model_cfg and key in self.metadata:
                model_cfg[key] = self.metadata[key]
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        head_path = self.model_dir / SEQUENTIAL_HEAD_FILENAME
        if not head_path.exists():
            raise FileNotFoundError(f"Sequential selector head not found: {head_path}")
        state = torch.load(head_path, map_location="cpu")
        hidden_size = int(model_cfg.get("hidden_size", 256))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = SequentialPointerSelectorModel(
            str(self.model_dir),
            hidden_size=hidden_size,
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_candidates=int(model_cfg.get("max_candidates", DEFAULT_CANDIDATE_POOL_SIZE)),
            semantic_feature_profile=str(model_cfg.get("semantic_feature_profile", SEMANTIC_FEATURE_PROFILE_DEEP)),
            targeted_feature_profile=str(model_cfg.get("targeted_feature_profile", TARGETED_FEATURE_PROFILE_NONE)),
            shallow_feature_profile=str(model_cfg.get("shallow_feature_profile", SHALLOW_FEATURE_PROFILE_OFF)),
            projection_mode=infer_projection_mode_from_model_config(model_cfg, selector_state=state),
            projection_hidden_multiplier=int(
                model_cfg.get("projection_hidden_multiplier")
                or _infer_projection_hidden_multiplier_from_selector_state(state, hidden_size=hidden_size)
            ),
            claim_start_mode=infer_claim_start_mode_from_model_config(model_cfg),
            claim_feature_mode=infer_claim_feature_mode_from_model_config(
                model_cfg,
                selector_state=state,
                hidden_size=hidden_size,
            ),
        )
        self.model.load_selector_head_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def select(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
        candidate_scores: list[dict[str, Any]] | None = None,
        *,
        top_k: int,
    ) -> SelectorPrediction:
        if not candidates:
            return SelectorPrediction(ordered_indices=[], metadata={"selector_type": self.selector_type})
        group = SelectorCandidateGroup(
            claim=str(claim),
            candidates=[dict(item) for item in candidates],
            candidate_scores=_default_candidate_scores(candidates, candidate_scores),
        )
        predictions = predict_sequential_groups(
            self.model,
            self.tokenizer,
            [group],
            device=self.device,
            max_length=int(self.cfg.max_length),
            top_k=int(top_k),
        )
        return predictions[0]

    def _validate_fingerprint(self) -> None:
        expected = str(self.cfg.expected_chunk_mmr_fingerprint or "")
        if not expected:
            return
        actual = str(self.metadata.get("chunk_mmr_fingerprint") or "")
        if not actual and self.cfg.strict_fingerprint:
            raise ValueError(
                f"Sequential selector {self.model_dir} has no chunk_mmr_fingerprint metadata; "
                f"expected {expected}."
            )
        if actual and actual != expected and self.cfg.strict_fingerprint:
            raise ValueError(
                f"Sequential selector fingerprint mismatch: expected {expected}, "
                f"got {actual} from {self.model_dir}."
            )


def forward_sequential_examples(
    model: nn.Module,
    tokenizer: Any,
    examples: list[Stage2OracleExample],
    *,
    device: torch.device,
    max_length: int,
) -> SequentialForwardOutput:
    groups = [
        SelectorCandidateGroup(
            claim=example.claim,
            candidates=example.candidates,
            candidate_scores=example.candidate_scores,
        )
        for example in examples
    ]
    return forward_sequential_groups(
        model,
        tokenizer,
        groups,
        device=device,
        max_length=max_length,
    )


def forward_sequential_groups(
    model: nn.Module,
    tokenizer: Any,
    groups: list[SelectorCandidateGroup],
    *,
    device: torch.device,
    max_length: int,
) -> SequentialForwardOutput:
    claims: list[str] = []
    texts: list[str] = []
    group_sizes: list[int] = []
    group_claims: list[str] = []
    for group in groups:
        group_claims.append(group.claim)
        group_sizes.append(len(group.candidates))
        for candidate in group.candidates:
            claims.append(group.claim)
            texts.append(candidate_text(candidate))
    enc = tokenize_claim_candidate_pairs(tokenizer, claims, texts, max_length=int(max_length))
    enc = {key: value.to(device) for key, value in enc.items()}
    claim_enc = None
    if _base_model(model).claim_feature_mode == CLAIM_FEATURE_MODE_CLAIM_ONLY:
        claim_enc = tokenize_claims_only(tokenizer, group_claims, max_length=int(max_length))
        claim_enc = {key: value.to(device) for key, value in claim_enc.items()}
    return model(enc, group_sizes=group_sizes, claim_encoded_inputs=claim_enc)


def teacher_forcing_sequential_logits(
    model: nn.Module,
    output: SequentialForwardOutput,
    selected_indices: list[list[int]],
    *,
    top_k: int,
) -> torch.Tensor:
    base_model = _base_model(model)
    return base_model.pointer_head.teacher_forcing_logits(
        output.context_embeddings,
        output.candidate_mask,
        selected_indices,
        top_k=top_k,
        claim_start=output.claim_start,
        claim_embedding=output.claim_embedding,
    )


def sequential_teacher_forcing_loss(
    logits: torch.Tensor,
    selected_indices: list[list[int]],
    *,
    candidate_mask: torch.Tensor | None = None,
    top_k: int = DEFAULT_SELECTOR_TOP_K,
    seq_loss_weight: float = 1.0,
    mask_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,N], got {tuple(logits.shape)}")
    batch_size, steps, max_candidates = logits.shape
    targets = torch.full(
        (batch_size, steps),
        fill_value=-100,
        dtype=torch.long,
        device=logits.device,
    )
    n_steps = 0
    for row, indices in enumerate(selected_indices[:batch_size]):
        for step, idx in enumerate(indices[: min(int(top_k), steps)]):
            idx = int(idx)
            if 0 <= idx < max_candidates:
                targets[row, step] = idx
                n_steps += 1
    if n_steps <= 0:
        zero = logits.sum() * 0.0
        return zero, {"loss": 0.0, "sequence_ce_loss": 0.0, "mask_loss": 0.0, "n_steps": 0.0}

    ce_loss = F.cross_entropy(
        logits.reshape(batch_size * steps, max_candidates),
        targets.reshape(batch_size * steps),
        ignore_index=-100,
    )
    mask_loss = logits.sum() * 0.0
    if mask_loss_weight > 0.0 and candidate_mask is not None:
        labels, valid = remaining_selected_bce_targets(
            selected_indices[:batch_size],
            candidate_mask,
            steps=steps,
            max_candidates=max_candidates,
            top_k=top_k,
        )
        if bool(valid.any().item()):
            mask_loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
    total = float(seq_loss_weight) * ce_loss + float(mask_loss_weight) * mask_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "sequence_ce_loss": float(ce_loss.detach().cpu()),
        "mask_loss": float(mask_loss.detach().cpu()),
        "seq_loss_weight": float(seq_loss_weight),
        "mask_loss_weight": float(mask_loss_weight),
        "n_steps": float(n_steps),
    }


def remaining_selected_bce_targets(
    selected_indices: list[list[int]],
    candidate_mask: torch.Tensor,
    *,
    steps: int,
    max_candidates: int,
    top_k: int = DEFAULT_SELECTOR_TOP_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build order-agnostic BCE labels for remaining oracle-selected candidates.

    At teacher-forced step t, previously selected oracle actions are invalid.
    Every remaining oracle-selected candidate is positive, so the auxiliary loss
    teaches set utility instead of only reweighting the next-action CE target.
    """
    labels = torch.zeros(
        (int(candidate_mask.shape[0]), int(steps), int(max_candidates)),
        dtype=torch.float32,
        device=candidate_mask.device,
    )
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for row, indices in enumerate(selected_indices):
        if row >= labels.shape[0]:
            break
        clean_indices = [
            int(idx)
            for idx in indices[: int(top_k)]
            if 0 <= int(idx) < int(max_candidates)
        ]
        selected_set = set(clean_indices)
        prefix_set: set[int] = set()
        for step in range(int(steps)):
            valid[row, step] = candidate_mask[row]
            for prev in prefix_set:
                valid[row, step, prev] = False
            for idx in selected_set - prefix_set:
                labels[row, step, idx] = 1.0
            if step < len(clean_indices):
                prefix_set.add(clean_indices[step])
    return labels, valid


@torch.inference_mode()
def predict_sequential_examples(
    model: nn.Module,
    tokenizer: Any,
    examples: list[Stage2OracleExample],
    *,
    device: torch.device,
    max_length: int,
    top_k: int,
) -> list[SelectorPrediction]:
    groups = [
        SelectorCandidateGroup(
            claim=example.claim,
            candidates=example.candidates,
            candidate_scores=example.candidate_scores,
        )
        for example in examples
    ]
    return predict_sequential_groups(
        model,
        tokenizer,
        groups,
        device=device,
        max_length=max_length,
        top_k=top_k,
    )


@torch.inference_mode()
def predict_sequential_groups(
    model: nn.Module,
    tokenizer: Any,
    groups: list[SelectorCandidateGroup],
    *,
    device: torch.device,
    max_length: int,
    top_k: int,
) -> list[SelectorPrediction]:
    output = forward_sequential_groups(model, tokenizer, groups, device=device, max_length=max_length)
    base_model = _base_model(model)
    return base_model.pointer_head.greedy_decode(
        output.context_embeddings,
        output.candidate_mask,
        top_k=top_k,
        claim_start=output.claim_start,
        claim_embedding=output.claim_embedding,
    )


def build_sequential_selection_trace(
    example: Stage2OracleExample,
    prediction: SelectorPrediction,
    *,
    selector_name: str = SEQUENTIAL_SELECTOR_TYPE,
    top_k: int = DEFAULT_SELECTOR_TOP_K,
) -> dict[str, Any]:
    ordered = [int(idx) for idx in prediction.ordered_indices[:top_k]]
    metrics = ordered_selection_metrics(example.selected_indices, ordered, top_k=top_k)
    candidate_scores: list[dict[str, Any]] = []
    selected_pos = {idx: pos for pos, idx in enumerate(ordered)}
    for idx, candidate in enumerate(example.candidates):
        base = dict(example.candidate_scores[idx]) if idx < len(example.candidate_scores) else {}
        base["candidate_idx"] = int(idx)
        base["candidate_uid"] = str(candidate.get("candidate_uid") or base.get("candidate_uid") or "")
        if idx in selected_pos:
            base["sequential_selected_step"] = int(selected_pos[idx])
            if idx < len(prediction.scores) and math.isfinite(float(prediction.scores[idx])):
                base["sequential_selected_score"] = float(prediction.scores[idx])
        candidate_scores.append(base)
    trace = {
        "event_id": example.event_id,
        "claim": example.claim,
        "gold_label": example.gold_label,
        "candidate_pool": example.candidates,
        "candidate_scores": candidate_scores,
        "oracle_ordered_indices": [int(idx) for idx in example.selected_indices],
        "selector_ordered_indices": ordered,
        "selector_scores": [float(x) for x in prediction.scores],
        "selector_name": selector_name,
        "fingerprint": example.fingerprint,
        "step_trace": prediction.step_trace,
    }
    trace.update(metrics)
    return trace


def summarize_sequential_step_diagnostics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {"n_claims": 0}
    by_step: dict[int, dict[str, list[float]]] = {}
    first_wrong_steps: list[float] = []
    for trace in traces:
        oracle = [int(idx) for idx in trace.get("oracle_ordered_indices", [])]
        pred = [int(idx) for idx in trace.get("selector_ordered_indices", [])]
        first_wrong = -1
        for step, target in enumerate(oracle):
            correct = float(step < len(pred) and pred[step] == target)
            row = by_step.setdefault(step, {"accuracy": [], "entropy": []})
            row["accuracy"].append(correct)
            if first_wrong < 0 and correct < 1.0:
                first_wrong = step
        if first_wrong < 0 and oracle:
            first_wrong = len(oracle)
        first_wrong_steps.append(float(first_wrong))
        for record in trace.get("step_trace", []) or []:
            try:
                step = int(record.get("step"))
            except (TypeError, ValueError):
                continue
            row = by_step.setdefault(step, {"accuracy": [], "entropy": []})
            if "step_entropy" in record:
                row["entropy"].append(float(record.get("step_entropy", 0.0)))
    return {
        "n_claims": len(traces),
        "by_step": {
            str(step): {
                "accuracy": float(np.mean(values["accuracy"])) if values["accuracy"] else 0.0,
                "entropy": float(np.mean(values["entropy"])) if values["entropy"] else 0.0,
            }
            for step, values in sorted(by_step.items())
        },
        "first_wrong_step_mean": float(np.mean(first_wrong_steps)) if first_wrong_steps else 0.0,
    }


def select_candidates_sequential(
    sample: Any,
    selector: SequentialSelector,
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
            "step_trace": [],
            "model_dir": str(selector.model_dir),
        }

    prediction = selector.select(str(sample.claim), candidates, candidates, top_k=int(top_k))
    selected: list[dict[str, Any]] = []
    trace_selected: list[dict[str, Any]] = []
    for rank, pool_idx in enumerate(prediction.ordered_indices[:top_k]):
        if pool_idx < 0 or pool_idx >= len(candidates):
            continue
        candidate = dict(candidates[pool_idx])
        score = prediction.scores[pool_idx] if pool_idx < len(prediction.scores) else float("nan")
        candidate.update({
            "sequential_score": float(score),
            "sequential_rank": int(rank),
            "sequential_candidate_pool_size": int(len(candidates)),
        })
        selected.append(candidate)
        trace_selected.append({
            "rank": int(rank),
            "candidate_pool_rank": int(candidate.get("candidate_pool_rank", pool_idx)),
            "source_index": int(candidate.get("source_index", -1)),
            "sequential_score": float(score),
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
        "selected": trace_selected,
        "step_trace": prediction.step_trace,
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


def mask_step_logits(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    invalid = ~candidate_mask.bool() | selected_mask.bool()
    return logits.masked_fill(invalid, -1.0e4)


def normalize_semantic_feature_profile(value: Any) -> str:
    profile = str(value or SEMANTIC_FEATURE_PROFILE_DEEP).strip().lower()
    if profile not in SEMANTIC_FEATURE_PROFILE_CHOICES:
        choices = ", ".join(SEMANTIC_FEATURE_PROFILE_CHOICES)
        raise ValueError(f"Unknown semantic_feature_profile: {profile!r}; choices: {choices}")
    return profile


def normalize_targeted_feature_profile(value: Any) -> str:
    profile = str(value or TARGETED_FEATURE_PROFILE_NONE).strip().lower()
    if profile not in TARGETED_FEATURE_PROFILE_CHOICES:
        choices = ", ".join(TARGETED_FEATURE_PROFILE_CHOICES)
        raise ValueError(f"Unknown targeted_feature_profile: {profile!r}; choices: {choices}")
    return profile


def normalize_shallow_feature_profile(value: Any) -> str:
    profile = str(value or SHALLOW_FEATURE_PROFILE_OFF).strip().lower()
    if profile in {"none", "false", "0"}:
        profile = SHALLOW_FEATURE_PROFILE_OFF
    if profile not in SHALLOW_FEATURE_PROFILE_CHOICES:
        choices = ", ".join(SHALLOW_FEATURE_PROFILE_CHOICES)
        raise ValueError(f"Unknown shallow_feature_profile: {profile!r}; choices: {choices}")
    return profile


def normalize_projection_mode(value: Any) -> str:
    mode = str(value or PROJECTION_MODE_LINEAR).strip().lower().replace("-", "_")
    aliases = {
        "deep": PROJECTION_MODE_LINEAR,
        "one_layer": PROJECTION_MODE_LINEAR,
        "single_layer": PROJECTION_MODE_LINEAR,
        "proj1": PROJECTION_MODE_LINEAR,
        "mlp": PROJECTION_MODE_MLP_RESIDUAL,
        "residual": PROJECTION_MODE_MLP_RESIDUAL,
        "proj2": PROJECTION_MODE_MLP_RESIDUAL,
        "proj2_residual": PROJECTION_MODE_MLP_RESIDUAL,
    }
    mode = aliases.get(mode, mode)
    if mode not in PROJECTION_MODE_CHOICES:
        choices = ", ".join(PROJECTION_MODE_CHOICES)
        raise ValueError(f"Unknown projection_mode: {mode!r}; choices: {choices}")
    return mode


def normalize_claim_start_mode(value: Any) -> str:
    mode = str(value or CLAIM_START_MODE_LEARNED).strip().lower().replace("-", "_")
    aliases = {
        "none": CLAIM_START_MODE_LEARNED,
        "false": CLAIM_START_MODE_LEARNED,
        "0": CLAIM_START_MODE_LEARNED,
        "start_prefix": CLAIM_START_MODE_LEARNED,
        "learned_start": CLAIM_START_MODE_LEARNED,
        "mean": CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
        "pool_mean": CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
        "candidate_mean": CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
        "candidate_pool": CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
        "claim_start": CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
    }
    mode = aliases.get(mode, mode)
    if mode not in CLAIM_START_MODE_CHOICES:
        choices = ", ".join(CLAIM_START_MODE_CHOICES)
        raise ValueError(f"Unknown claim_start_mode: {mode!r}; choices: {choices}")
    return mode


def normalize_claim_feature_mode(value: Any) -> str:
    mode = str(value or CLAIM_FEATURE_MODE_OFF).strip().lower().replace("-", "_")
    aliases = {
        "none": CLAIM_FEATURE_MODE_OFF,
        "false": CLAIM_FEATURE_MODE_OFF,
        "0": CLAIM_FEATURE_MODE_OFF,
        "no": CLAIM_FEATURE_MODE_OFF,
        "claim": CLAIM_FEATURE_MODE_CLAIM_ONLY,
        "claim_only_text": CLAIM_FEATURE_MODE_CLAIM_ONLY,
        "claim_text": CLAIM_FEATURE_MODE_CLAIM_ONLY,
        "h_claim": CLAIM_FEATURE_MODE_CLAIM_ONLY,
    }
    mode = aliases.get(mode, mode)
    if mode not in CLAIM_FEATURE_MODE_CHOICES:
        choices = ", ".join(CLAIM_FEATURE_MODE_CHOICES)
        raise ValueError(f"Unknown claim_feature_mode: {mode!r}; choices: {choices}")
    return mode


def infer_projection_mode_from_model_config(
    model_config: dict[str, Any] | None,
    *,
    selector_state: dict[str, Any] | None = None,
) -> str:
    cfg = dict(model_config or {})
    explicit = cfg.get("projection_mode")
    if explicit not in (None, ""):
        return normalize_projection_mode(explicit)
    if _truthy(cfg.get("proj_residual")):
        return PROJECTION_MODE_MLP_RESIDUAL
    try:
        if int(cfg.get("proj_num_layers", 1)) > 1:
            return PROJECTION_MODE_MLP_RESIDUAL
    except (TypeError, ValueError):
        pass
    if selector_state:
        if "proj_residual" in selector_state:
            return PROJECTION_MODE_MLP_RESIDUAL
        item_projection = selector_state.get("item_projection")
        if isinstance(item_projection, dict) and any(str(key).startswith("3.") for key in item_projection):
            return PROJECTION_MODE_MLP_RESIDUAL
    return PROJECTION_MODE_LINEAR


def infer_claim_start_mode_from_model_config(model_config: dict[str, Any] | None) -> str:
    cfg = dict(model_config or {})
    explicit = cfg.get("claim_start_mode")
    if explicit not in (None, ""):
        return normalize_claim_start_mode(explicit)
    if "claim_start" in cfg:
        return normalize_claim_start_mode(cfg.get("claim_start"))
    return CLAIM_START_MODE_LEARNED


def infer_claim_feature_mode_from_model_config(
    model_config: dict[str, Any] | None,
    *,
    selector_state: dict[str, Any] | None = None,
    hidden_size: int = 256,
) -> str:
    cfg = dict(model_config or {})
    explicit = cfg.get("claim_feature_mode")
    if explicit not in (None, ""):
        return normalize_claim_feature_mode(explicit)
    pointer_head = (selector_state or {}).get("pointer_head")
    if isinstance(pointer_head, dict):
        scorer_weight = pointer_head.get("scorer.0.weight")
        shape = getattr(scorer_weight, "shape", None)
        if shape and int(shape[1]) >= (int(hidden_size) * 6 + 1):
            return CLAIM_FEATURE_MODE_CLAIM_ONLY
    return CLAIM_FEATURE_MODE_OFF


def load_sequential_metadata(model_dir: str | Path) -> dict[str, Any]:
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


def tokenize_claims_only(
    tokenizer: Any,
    claims: list[str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        [f"Claim: {claim}" for claim in claims],
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )


def _step_trace_from_logits(
    logits: torch.Tensor,
    remaining: torch.Tensor,
    step: int,
    selected_idx: int,
    selected_score: float,
) -> dict[str, Any]:
    remaining = remaining.to(dtype=torch.long)
    valid_logits = logits.index_select(0, remaining)
    probs = torch.softmax(valid_logits.float(), dim=0)
    entropy = float(-(probs * probs.clamp_min(1e-8).log()).sum().detach().cpu())
    topn = min(5, int(remaining.numel()))
    top_scores, top_pos = torch.topk(valid_logits, k=topn)
    return {
        "step": int(step),
        "selected_index": int(selected_idx),
        "selected_score": float(selected_score),
        "remaining_indices_before_step": [int(x) for x in remaining.detach().cpu().tolist()],
        "step_entropy": entropy,
        "step_logits_topk": [
            {
                "candidate_idx": int(remaining[int(pos.detach().cpu().item())].detach().cpu().item()),
                "logit": float(score.detach().cpu()),
            }
            for score, pos in zip(top_scores, top_pos)
        ],
    }


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


def _build_claim_start(
    item_embeddings: torch.Tensor, group_sizes: list[int]
) -> torch.Tensor | None:
    if not group_sizes:
        return None
    starts = []
    offset = 0
    for size in group_sizes:
        size = int(size)
        if size <= 0:
            starts.append(torch.zeros(item_embeddings.shape[-1], device=item_embeddings.device))
        else:
            starts.append(item_embeddings[offset : offset + size].mean(dim=0))
        offset += size
    return torch.stack(starts, dim=0)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _infer_projection_hidden_multiplier_from_selector_state(
    selector_state: dict[str, Any] | None,
    *,
    hidden_size: int,
) -> int:
    if not selector_state:
        return 2
    item_projection = selector_state.get("item_projection")
    if not isinstance(item_projection, dict) or "3.weight" not in item_projection:
        return 2
    first_weight = item_projection.get("0.weight")
    shape = getattr(first_weight, "shape", None)
    if not shape or int(hidden_size) <= 0:
        return 2
    return max(1, int(shape[0]) // int(hidden_size))


def _base_model(model: nn.Module) -> SequentialPointerSelectorModel:
    return getattr(model, "module", model)


def _max_teacher_steps(selected_indices: list[list[int]], *, top_k: int) -> int:
    max_len = max([len(indices) for indices in selected_indices] or [0])
    return max(0, min(int(top_k), int(max_len)))


register_selector_type(SEQUENTIAL_SELECTOR_TYPE, SequentialSelector)  # type: ignore[arg-type]
register_selector_type("sequential_selector", SequentialSelector)  # type: ignore[arg-type]
