from __future__ import annotations

from typing import Any

import torch

from liar_raw.utils.labels import id_to_label


def build_graph_prediction_records(
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    top_n_subclaims: int = 3,
    top_n_evidence: int = 4,
) -> list[dict[str, Any]]:
    pred_ids = outputs["class_logits"].argmax(dim=-1)
    node_attn = outputs["node_attention"]
    ev_attn = outputs["evidence_attention"]
    node_mask = batch["node_mask"]

    results: list[dict[str, Any]] = []
    for b_idx, event_id in enumerate(batch["event_ids"]):
        raw_nodes = batch["raw_nodes"][b_idx]
        node_mask_row = node_mask[b_idx]

        subclaims = []
        evidence = []

        for n_idx, node in enumerate(raw_nodes):
            if not bool(node_mask_row[n_idx].item()):
                continue
            record = dict(node)
            if node["node_type"] == "subclaim":
                record["graph_attention"] = float(node_attn[b_idx, n_idx].item())
                subclaims.append(record)
            elif node["node_type"] in {"support_evidence", "refute_evidence"}:
                record["graph_attention"] = float(ev_attn[b_idx, n_idx].item())
                evidence.append(record)

        subclaims = sorted(subclaims, key=lambda x: x["graph_attention"], reverse=True)[:top_n_subclaims]
        evidence = sorted(evidence, key=lambda x: x["graph_attention"], reverse=True)[:top_n_evidence]

        support_evidence = [x for x in evidence if x["node_type"] == "support_evidence"]
        refute_evidence = [x for x in evidence if x["node_type"] == "refute_evidence"]

        results.append(
            {
                "event_id": event_id,
                "claim": batch["claims"][b_idx],
                "gold_label": id_to_label(int(batch["labels"][b_idx].item())),
                "pred_label": id_to_label(int(pred_ids[b_idx].item())),
                "selected_subclaims": subclaims,
                "support_evidence": support_evidence,
                "refute_evidence": refute_evidence,
                "gold_explain": batch["gold_explains"][b_idx],
            }
        )

    return results
