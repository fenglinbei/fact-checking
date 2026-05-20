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


class DeepInteractionPointerHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        dropout: float = 0.1,
        bilinear_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.bilinear_size = int(bilinear_size or hidden_size)
        self.start_prefix = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.start_prefix, mean=0.0, std=0.02)
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, self.bilinear_size)
        interaction_dim = self.hidden_size * 4 + 1 + self.bilinear_size
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
    ) -> torch.Tensor:
        selected = selected_mask.to(dtype=context_embeddings.dtype).unsqueeze(-1)
        counts = selected.sum(dim=1).clamp_min(1.0)
        pooled = (context_embeddings * selected).sum(dim=1) / counts
        has_prefix = selected_mask.any(dim=1).unsqueeze(-1)
        start = self.start_prefix.to(dtype=context_embeddings.dtype).unsqueeze(0)
        return torch.where(has_prefix, pooled, start.expand_as(pooled))

    def score_step(
        self,
        context_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix = self.prefix_representation(context_embeddings, selected_mask)
        prefix_expanded = prefix.unsqueeze(1).expand_as(context_embeddings)
        product = context_embeddings * prefix_expanded
        difference = torch.abs(context_embeddings - prefix_expanded)
        cosine = F.cosine_similarity(context_embeddings, prefix_expanded, dim=-1).unsqueeze(-1)
        bilinear = self.bilinear(context_embeddings, prefix_expanded)
        features = torch.cat(
            [context_embeddings, prefix_expanded, product, difference, cosine, bilinear],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)
        return mask_step_logits(logits, candidate_mask, selected_mask)

    def teacher_forcing_logits(
        self,
        context_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        selected_indices: list[list[int]],
        *,
        top_k: int,
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
            logits = self.score_step(context_embeddings, candidate_mask, selected_mask)
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
            logits = self.score_step(context_embeddings, candidate_mask, selected_mask)
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
        self.pointer_head = DeepInteractionPointerHead(self.hidden_size, dropout=self.dropout)

    def forward(
        self,
        encoded_inputs: dict[str, torch.Tensor],
        *,
        group_sizes: list[int],
    ) -> SequentialForwardOutput:
        outputs = self.encoder(**encoded_inputs)
        pair_embeddings = pool_pair_embeddings(outputs, encoded_inputs.get("attention_mask"))
        item_embeddings = self.item_projection(pair_embeddings)
        padded_items, mask = pad_flat_items(item_embeddings, group_sizes)
        context = self.set_encoder(padded_items, src_key_padding_mask=~mask)
        context = self.output_norm(context)
        return SequentialForwardOutput(context_embeddings=context, candidate_mask=mask)

    def selector_head_state_dict(self) -> dict[str, Any]:
        return {
            "item_projection": self.item_projection.state_dict(),
            "set_encoder": self.set_encoder.state_dict(),
            "output_norm": self.output_norm.state_dict(),
            "pointer_head": self.pointer_head.state_dict(),
        }

    def load_selector_head_state_dict(self, payload: dict[str, Any]) -> None:
        self.item_projection.load_state_dict(payload["item_projection"])
        self.set_encoder.load_state_dict(payload["set_encoder"])
        self.output_norm.load_state_dict(payload["output_norm"])
        self.pointer_head.load_state_dict(payload["pointer_head"])

    def model_config(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "max_candidates": self.max_candidates,
            "semantic_feature_profile": self.semantic_feature_profile,
            "targeted_feature_profile": self.targeted_feature_profile,
            "shallow_feature_profile": self.shallow_feature_profile,
            "deep_interaction_features": [
                "h_i_pair",
                "H_i_ctx",
                "P_t",
                "H_i_ctx * P_t",
                "abs(H_i_ctx - P_t)",
                "cos(H_i_ctx, P_t)",
                "bilinear(H_i_ctx, P_t)",
            ],
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
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = SequentialPointerSelectorModel(
            str(self.model_dir),
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_candidates=int(model_cfg.get("max_candidates", DEFAULT_CANDIDATE_POOL_SIZE)),
            semantic_feature_profile=str(model_cfg.get("semantic_feature_profile", SEMANTIC_FEATURE_PROFILE_DEEP)),
            targeted_feature_profile=str(model_cfg.get("targeted_feature_profile", TARGETED_FEATURE_PROFILE_NONE)),
            shallow_feature_profile=str(model_cfg.get("shallow_feature_profile", SHALLOW_FEATURE_PROFILE_OFF)),
        )
        head_path = self.model_dir / SEQUENTIAL_HEAD_FILENAME
        if not head_path.exists():
            raise FileNotFoundError(f"Sequential selector head not found: {head_path}")
        state = torch.load(head_path, map_location="cpu")
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
    for group in groups:
        group_sizes.append(len(group.candidates))
        for candidate in group.candidates:
            claims.append(group.claim)
            texts.append(candidate_text(candidate))
    enc = tokenize_claim_candidate_pairs(tokenizer, claims, texts, max_length=int(max_length))
    enc = {key: value.to(device) for key, value in enc.items()}
    return model(enc, group_sizes=group_sizes)


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


def _base_model(model: nn.Module) -> SequentialPointerSelectorModel:
    return getattr(model, "module", model)


def _max_teacher_steps(selected_indices: list[list[int]], *, top_k: int) -> int:
    max_len = max([len(indices) for indices in selected_indices] or [0])
    return max(0, min(int(top_k), int(max_len)))


register_selector_type(SEQUENTIAL_SELECTOR_TYPE, SequentialSelector)  # type: ignore[arg-type]
register_selector_type("sequential_selector", SequentialSelector)  # type: ignore[arg-type]
