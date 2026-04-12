from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from liar_raw_cde.stage_d.graph_builder import NODE_TYPES, REL_TYPES


class RelationGraphLayer(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_size, hidden_size)
        self.rel_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_relations)])
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H]
        # adj: [B, R, N, N]
        messages = self.self_proj(h)
        for r, proj in enumerate(self.rel_projs):
            transformed = proj(h)  # [B, N, H]
            a = adj[:, r]  # [B, N, N]
            deg = a.sum(dim=-1, keepdim=True).clamp_min(1.0)
            msg_r = torch.bmm(a, transformed) / deg
            messages = messages + msg_r

        out = self.out_proj(F.gelu(messages))
        out = self.dropout(out)
        out = self.norm(h + out)
        out = out * node_mask.unsqueeze(-1)
        return out


class GraphVerifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_labels: int = 6,
        use_ordinal_head: bool = True,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=config)
        self.num_rel = len(REL_TYPES)
        self.num_node_types = len(NODE_TYPES)
        self.num_stances = 3
        self.use_ordinal_head = use_ordinal_head

        enc_hidden = config.hidden_size
        self.node_proj = nn.Linear(enc_hidden, hidden_size)
        self.node_type_emb = nn.Embedding(self.num_node_types, hidden_size)
        self.stance_emb = nn.Embedding(self.num_stances, hidden_size)
        self.scalar_proj = nn.Linear(2, hidden_size)

        self.layers = nn.ModuleList(
            [RelationGraphLayer(hidden_size, self.num_rel, dropout=dropout) for _ in range(num_layers)]
        )

        self.pool_query = nn.Parameter(torch.randn(hidden_size))
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_labels),
        )
        if self.use_ordinal_head:
            self.ordinal_head = nn.Linear(hidden_size * 2, num_labels - 1)

    def _pool_encoder_output(self, outputs: Any) -> torch.Tensor:
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def _encode_nodes(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        return self._pool_encoder_output(outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
        node_type_ids: torch.Tensor,
        stance_ids: torch.Tensor,
        scalar_feats: torch.Tensor,
        adj: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_nodes = node_mask.shape
        total_nodes = batch_size * max_nodes

        valid_counts = node_mask.sum(dim=1).tolist()
        flat_node_inputs = []
        flat_mask_inputs = []
        flat_tt_inputs = [] if token_type_ids is not None else None

        start = 0
        for count in valid_counts:
            count = int(count)
            flat_node_inputs.append(input_ids[start:start + count])
            flat_mask_inputs.append(attention_mask[start:start + count])
            if token_type_ids is not None:
                flat_tt_inputs.append(token_type_ids[start:start + count])
            start += count

        flat_input_ids = torch.cat(flat_node_inputs, dim=0)
        flat_attention_mask = torch.cat(flat_mask_inputs, dim=0)
        flat_token_type_ids = torch.cat(flat_tt_inputs, dim=0) if flat_tt_inputs is not None else None

        flat_encoded = self._encode_nodes(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
        )

        h = torch.zeros(batch_size, max_nodes, flat_encoded.size(-1), device=flat_encoded.device)
        ptr = 0
        for b_idx, count in enumerate(valid_counts):
            count = int(count)
            h[b_idx, :count] = flat_encoded[ptr:ptr + count]
            ptr += count

        h = self.node_proj(h)
        h = h + self.node_type_emb(node_type_ids) + self.stance_emb(stance_ids) + self.scalar_proj(scalar_feats)
        h = h * node_mask.unsqueeze(-1)

        for layer in self.layers:
            h = layer(h, adj, node_mask)

        # claim+subclaim pooling mask: node_type_ids in {0,1}
        pool_mask = node_mask & ((node_type_ids == 0) | (node_type_ids == 1))
        attn_logits = torch.einsum("bnh,h->bn", h, self.pool_query)
        attn_logits = attn_logits.masked_fill(~pool_mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)
        attn = torch.where(pool_mask, attn, torch.zeros_like(attn))
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        pooled_main = torch.sum(attn.unsqueeze(-1) * h, dim=1)

        # evidence pooling
        ev_mask = node_mask & ((node_type_ids == 2) | (node_type_ids == 3))
        ev_attn_logits = torch.einsum("bnh,h->bn", h, self.pool_query)
        ev_attn_logits = ev_attn_logits.masked_fill(~ev_mask, float("-inf"))
        ev_attn = F.softmax(ev_attn_logits, dim=-1)
        ev_attn = torch.where(ev_mask, ev_attn, torch.zeros_like(ev_attn))
        ev_attn = ev_attn / ev_attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled_ev = torch.sum(ev_attn.unsqueeze(-1) * h, dim=1)

        pooled = torch.cat([pooled_main, pooled_ev], dim=-1)
        pooled = self.dropout(pooled)

        class_logits = self.classifier(pooled)
        outputs = {
            "class_logits": class_logits,
            "node_attention": attn,
            "evidence_attention": ev_attn,
        }
        if self.use_ordinal_head:
            outputs["ordinal_logits"] = self.ordinal_head(pooled)
        return outputs
