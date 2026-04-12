from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel

from liar_raw.models.sparsemax import masked_sparsemax


@dataclass(slots=True)
class LatentEvidenceOutput:
    class_logits: torch.Tensor
    ordinal_logits: torch.Tensor
    attention_weights: torch.Tensor
    support_prob: torch.Tensor
    refute_prob: torch.Tensor
    margin: torch.Tensor
    total_evidence: torch.Tensor


class LatentEvidenceOrdinalModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 6,
        dropout: float = 0.1,
        unfreeze_last_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.attention_scorer = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self.support_head = nn.Linear(self.hidden_size, 1)
        self.refute_head = nn.Linear(self.hidden_size, 1)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.hidden_size + 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.ordinal_head = nn.Linear(self.hidden_size, num_classes - 1)

        self._freeze_encoder_except_last_n(unfreeze_last_n_layers)

    def _freeze_encoder_except_last_n(self, unfreeze_last_n_layers: int) -> None:
        if unfreeze_last_n_layers < 0:
            return
        for param in self.encoder.parameters():
            param.requires_grad = False

        candidate_layer_stacks = []
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            candidate_layer_stacks.append(self.encoder.encoder.layer)
        if hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            candidate_layer_stacks.append(self.encoder.transformer.layer)
        if hasattr(self.encoder, "layer"):
            candidate_layer_stacks.append(self.encoder.layer)

        if not candidate_layer_stacks:
            for param in self.encoder.parameters():
                param.requires_grad = True
            return

        layer_stack = candidate_layer_stacks[0]
        n = len(layer_stack)
        start = max(0, n - unfreeze_last_n_layers)
        for layer_idx in range(start, n):
            for param in layer_stack[layer_idx].parameters():
                param.requires_grad = True

        # Try to keep embeddings frozen for stability.
        if hasattr(self.encoder, "pooler"):
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

    @staticmethod
    def _masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-8)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        hybrid_scores: torch.Tensor,
    ) -> LatentEvidenceOutput:
        """Forward.

        Args:
            input_ids: [B*K, L]
            attention_mask: [B*K, L]
            candidate_mask: [B, K]
            hybrid_scores: [B, K], expected in [0, 1] or close to it.
        """
        batch_size, top_k = candidate_mask.shape
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean_pool(enc_out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        pair_repr = pooled.view(batch_size, top_k, self.hidden_size)

        hybrid_feat = hybrid_scores.unsqueeze(-1)
        attn_input = torch.cat([pair_repr, hybrid_feat], dim=-1)
        attn_logits = self.attention_scorer(attn_input).squeeze(-1)
        attention_weights = masked_sparsemax(attn_logits, candidate_mask, dim=-1)

        support_prob = torch.sigmoid(self.support_head(pair_repr).squeeze(-1))
        refute_prob = torch.sigmoid(self.refute_head(pair_repr).squeeze(-1))

        mask_f = candidate_mask.float()
        support_prob = support_prob * mask_f
        refute_prob = refute_prob * mask_f

        aggregated_repr = torch.einsum("bk,bkh->bh", attention_weights, pair_repr)
        support_score = torch.sum(attention_weights * support_prob, dim=-1, keepdim=True)
        refute_score = torch.sum(attention_weights * refute_prob, dim=-1, keepdim=True)
        margin = support_score - refute_score
        total_evidence = support_score + refute_score

        feature_vec = torch.cat(
            [aggregated_repr, support_score, refute_score, margin, total_evidence],
            dim=-1,
        )
        hidden = self.feature_proj(feature_vec)
        class_logits = self.classifier(hidden)
        ordinal_logits = self.ordinal_head(hidden)

        return LatentEvidenceOutput(
            class_logits=class_logits,
            ordinal_logits=ordinal_logits,
            attention_weights=attention_weights,
            support_prob=support_prob,
            refute_prob=refute_prob,
            margin=margin.squeeze(-1),
            total_evidence=total_evidence.squeeze(-1),
        )

    @torch.inference_mode()
    def extract_evidence(
        self,
        output: LatentEvidenceOutput,
        metadata: list[list[dict[str, Any] | None]],
        top_n: int = 3,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        attn = output.attention_weights.detach().cpu()
        support = output.support_prob.detach().cpu()
        refute = output.refute_prob.detach().cpu()
        results: list[dict[str, list[dict[str, Any]]]] = []
        for b_idx, meta_list in enumerate(metadata):
            support_rank = []
            refute_rank = []
            for k_idx, meta in enumerate(meta_list):
                if meta is None:
                    continue
                a = float(attn[b_idx, k_idx].item())
                s = float(support[b_idx, k_idx].item())
                r = float(refute[b_idx, k_idx].item())
                support_rank.append({**meta, "attention": a, "stance_score": a * s})
                refute_rank.append({**meta, "attention": a, "stance_score": a * r})
            support_rank.sort(key=lambda x: x["stance_score"], reverse=True)
            refute_rank.sort(key=lambda x: x["stance_score"], reverse=True)
            results.append(
                {
                    "support": support_rank[:top_n],
                    "refute": refute_rank[:top_n],
                }
            )
        return results
