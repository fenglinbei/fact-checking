from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from stage_d.graph_builder import REL_TYPES


class GraphBatchCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_rel = len(REL_TYPES)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        max_nodes = max(len(x["nodes"]) for x in batch)

        node_texts: list[str] = []
        node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        node_type_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        stance_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        scalar_feats = torch.zeros(batch_size, max_nodes, 2, dtype=torch.float)
        labels = torch.tensor([int(x["label_id"]) for x in batch], dtype=torch.long)

        adj = torch.zeros(batch_size, self.num_rel, max_nodes, max_nodes, dtype=torch.float)
        event_ids: list[str] = []
        claims: list[str] = []
        gold_explains: list[str] = []
        raw_nodes: list[list[dict[str, Any]]] = []

        for b_idx, item in enumerate(batch):
            event_ids.append(item["event_id"])
            claims.append(item["claim"])
            gold_explains.append(item.get("gold_explain", ""))
            raw_nodes.append(item["nodes"])

            for n_idx, node in enumerate(item["nodes"]):
                node_texts.append(node["text"])
                node_mask[b_idx, n_idx] = True
                node_type_ids[b_idx, n_idx] = int(node["node_type_id"])
                stance_ids[b_idx, n_idx] = int(node["stance_id"])
                scalar_feats[b_idx, n_idx, 0] = float(node.get("score", 0.0))
                scalar_feats[b_idx, n_idx, 1] = float(node.get("position", 0.0))

            for edge in item["edges"]:
                src = int(edge["src"])
                dst = int(edge["dst"])
                rel = int(edge["rel_id"])
                if src < max_nodes and dst < max_nodes:
                    adj[b_idx, rel, src, dst] = 1.0

        enc = self.tokenizer(
            node_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "node_mask": node_mask,
            "node_type_ids": node_type_ids,
            "stance_ids": stance_ids,
            "scalar_feats": scalar_feats,
            "adj": adj,
            "event_ids": event_ids,
            "claims": claims,
            "gold_explains": gold_explains,
            "raw_nodes": raw_nodes,
            "token_type_ids": enc.get("token_type_ids"),
        }
