from __future__ import annotations

from collections import defaultdict
from typing import Any

from utils.labels import label_to_id
from utils.text import clean_text, jaccard


REL_TYPES = [
    "claim_to_subclaim",
    "subclaim_to_support",
    "subclaim_to_refute",
    "same_report",
    "lexical_overlap",
]
REL_TO_ID = {name: i for i, name in enumerate(REL_TYPES)}

NODE_TYPES = ["claim", "subclaim", "support_evidence", "refute_evidence"]
NODE_TYPE_TO_ID = {name: i for i, name in enumerate(NODE_TYPES)}

STANCE_TYPES = {"neutral": 0, "support": 1, "refute": 2}


def _normalize_evidence_item(item: dict[str, Any], stance: str) -> dict[str, Any]:
    sentence = str(item.get("sentence") or item.get("sent") or item.get("text") or "").strip()
    score = float(
        item.get("importance")
        or item.get("latent_support_score")
        or item.get("latent_refute_score")
        or item.get("stage_a_score")
        or item.get("subclaim_similarity")
        or 0.0
    )
    return {
        "sentence": clean_text(sentence),
        "report_id": int(item.get("report_id", -1)) if str(item.get("report_id", "")).lstrip("-").isdigit() else -1,
        "position": int(item.get("position", -1)) if str(item.get("position", "")).lstrip("-").isdigit() else -1,
        "domain": str(item.get("domain") or item.get("link") or ""),
        "score": score,
        "stance": stance,
    }


def build_graph_item(
    raw_item: dict[str, Any],
    stage_b_pred_item: dict[str, Any],
    decomposition_info: dict[str, Any],
    add_same_report_edges: bool = True,
    add_lexical_overlap_edges: bool = True,
    lexical_overlap_jaccard_threshold: float = 0.18,
) -> dict[str, Any]:
    claim = clean_text(raw_item["claim"])
    gold_label = str(raw_item.get("label") or raw_item.get("gold_label") or stage_b_pred_item.get("gold_label") or stage_b_pred_item.get("label"))
    label_id = label_to_id(gold_label)
    gold_explain = clean_text(raw_item.get("explain", ""))

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # claim node
    nodes.append(
        {
            "node_id": 0,
            "node_type": "claim",
            "node_type_id": NODE_TYPE_TO_ID["claim"],
            "stance": "neutral",
            "stance_id": STANCE_TYPES["neutral"],
            "text": claim,
            "score": 1.0,
            "report_id": -1,
            "position": 0,
        }
    )

    next_node_id = 1
    evidence_nodes_by_key: dict[tuple[str, int, int, str], int] = {}

    assignments = decomposition_info["assignments"]
    subclaim_node_ids: list[int] = []

    for a in assignments:
        subclaim = clean_text(a["subclaim"])
        sub_id = next_node_id
        next_node_id += 1
        subclaim_node_ids.append(sub_id)

        nodes.append(
            {
                "node_id": sub_id,
                "node_type": "subclaim",
                "node_type_id": NODE_TYPE_TO_ID["subclaim"],
                "stance": "neutral",
                "stance_id": STANCE_TYPES["neutral"],
                "text": subclaim,
                "score": 1.0,
                "report_id": -1,
                "position": 0,
            }
        )
        edges.append({"src": 0, "dst": sub_id, "rel_type": "claim_to_subclaim", "rel_id": REL_TO_ID["claim_to_subclaim"]})

        for stance_name, rel_name, node_type in [
            ("support_evidence", "subclaim_to_support", "support_evidence"),
            ("refute_evidence", "subclaim_to_refute", "refute_evidence"),
        ]:
            stance_value = "support" if node_type == "support_evidence" else "refute"
            for item in a.get(stance_name, []):
                ev = _normalize_evidence_item(item, stance=stance_value)
                key = (ev["sentence"], ev["report_id"], ev["position"], stance_value)
                if key not in evidence_nodes_by_key:
                    ev_id = next_node_id
                    next_node_id += 1
                    evidence_nodes_by_key[key] = ev_id
                    nodes.append(
                        {
                            "node_id": ev_id,
                            "node_type": node_type,
                            "node_type_id": NODE_TYPE_TO_ID[node_type],
                            "stance": stance_value,
                            "stance_id": STANCE_TYPES[stance_value],
                            "text": ev["sentence"],
                            "score": ev["score"],
                            "report_id": ev["report_id"],
                            "position": ev["position"],
                            "domain": ev["domain"],
                        }
                    )
                ev_id = evidence_nodes_by_key[key]
                edges.append({"src": sub_id, "dst": ev_id, "rel_type": rel_name, "rel_id": REL_TO_ID[rel_name]})

    evidence_ids = [n["node_id"] for n in nodes if n["node_type"] in {"support_evidence", "refute_evidence"}]
    node_lookup = {n["node_id"]: n for n in nodes}

    if add_same_report_edges:
        by_report = defaultdict(list)
        for node_id in evidence_ids:
            rid = node_lookup[node_id].get("report_id", -1)
            if rid != -1:
                by_report[rid].append(node_id)
        for ids in by_report.values():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edges.append({"src": ids[i], "dst": ids[j], "rel_type": "same_report", "rel_id": REL_TO_ID["same_report"]})
                    edges.append({"src": ids[j], "dst": ids[i], "rel_type": "same_report", "rel_id": REL_TO_ID["same_report"]})

    if add_lexical_overlap_edges:
        for i in range(len(evidence_ids)):
            for j in range(i + 1, len(evidence_ids)):
                a = node_lookup[evidence_ids[i]]
                b = node_lookup[evidence_ids[j]]
                if jaccard(a["text"], b["text"]) >= lexical_overlap_jaccard_threshold:
                    edges.append({"src": a["node_id"], "dst": b["node_id"], "rel_type": "lexical_overlap", "rel_id": REL_TO_ID["lexical_overlap"]})
                    edges.append({"src": b["node_id"], "dst": a["node_id"], "rel_type": "lexical_overlap", "rel_id": REL_TO_ID["lexical_overlap"]})

    return {
        "event_id": raw_item["event_id"],
        "claim": claim,
        "label": gold_label,
        "label_id": label_id,
        "gold_explain": gold_explain,
        "decomposition": decomposition_info,
        "nodes": nodes,
        "edges": edges,
    }
