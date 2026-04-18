from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
try:
    from jtqdm import jtqdm
except ImportError:
    from tqdm.auto import tqdm as jtqdm

from liar_raw.stage_c.assign import SubclaimEvidenceAssigner
from liar_raw.stage_c.decompose import HFLocalClaimDecomposer, HeuristicClaimDecomposer
from liar_raw.stage_c.gating import ComplexityGate
from liar_raw.stage_d.graph_builder import build_graph_item
from liar_raw.utils.io import ensure_dir, load_yaml, read_json, read_jsonl, write_jsonl


def _build_one_split(
    raw_path: str,
    pred_path: str,
    out_path: str,
    cfg: dict,
    device: str | None = None,
):
    raw_items = read_json(raw_path)
    pred_items = read_jsonl(pred_path)
    raw_by_id = {str(x["event_id"]): x for x in raw_items}
    pred_by_id = {str(x["event_id"]): x for x in pred_items}

    gate = ComplexityGate(
        min_claim_tokens_to_split=cfg["decomposition"]["min_claim_tokens_to_split"]
    )

    method = cfg["decomposition"]["method"]
    if method == "hf_local":
        decomposer = HFLocalClaimDecomposer(
            model_name=cfg["decomposition"]["hf_model_name"],
            max_new_tokens=cfg["decomposition"]["max_new_tokens"],
            max_subclaims=cfg["decomposition"]["max_subclaims"],
            use_vllm=cfg["decomposition"].get("use_vllm", False),
            vllm_tensor_parallel_size=cfg["decomposition"].get("vllm_tensor_parallel_size", 1),
            vllm_gpu_memory_utilization=cfg["decomposition"].get("vllm_gpu_memory_utilization", 0.9),
        )
    else:
        decomposer = HeuristicClaimDecomposer(
            max_subclaims=cfg["decomposition"]["max_subclaims"],
            min_subclaim_tokens=cfg["decomposition"]["min_subclaim_tokens"],
        )

    assigner = SubclaimEvidenceAssigner(
        embedder_model=cfg["assignment"]["embedder_model"],
        top_k_support_per_subclaim=cfg["assignment"]["top_k_support_per_subclaim"],
        top_k_refute_per_subclaim=cfg["assignment"]["top_k_refute_per_subclaim"],
        device=device,
    )

    split_event_ids: list[str] = []
    split_claims: list[str] = []
    split_claims_map: dict[str, list[str]] = {}
    for event_id, raw_item in raw_by_id.items():
        pred = pred_by_id.get(event_id)
        if pred is None:
            continue
        claim = raw_item["claim"]
        gate_decision = gate.decide(claim)
        if gate_decision.should_split:
            split_event_ids.append(event_id)
            split_claims.append(claim)

    if split_claims and hasattr(decomposer, "decompose_many"):
        decomp_results = decomposer.decompose_many(split_claims)
        split_claims_map = {event_id: result.subclaims for event_id, result in zip(split_event_ids, decomp_results)}

    outputs = []
    for event_id, raw_item in jtqdm(
        raw_by_id.items(),
        total=len(raw_by_id),
        desc=f"Build graph inputs [{Path(out_path).stem}]",
    ):
        pred = pred_by_id.get(event_id)
        if pred is None:
            continue

        claim = raw_item["claim"]
        gate_decision = gate.decide(claim)
        if gate_decision.should_split:
            if event_id in split_claims_map:
                subclaims = split_claims_map[event_id]
                method_used = "hf_local"
            else:
                decomp = decomposer.decompose(claim)
                subclaims = decomp.subclaims
                method_used = decomp.method
        else:
            subclaims = [claim]
            method_used = "no_split"

        assignments = assigner.assign(
            claim=claim,
            subclaims=subclaims,
            support_evidence=pred.get("support_evidence", []),
            refute_evidence=pred.get("refute_evidence", []),
        )
        subclaim_claim_similarities = [
            float(a.get("subclaim_claim_similarity", 0.0))
            for a in assignments
            if isinstance(a.get("subclaim_claim_similarity"), (int, float))
        ]
        subclaim_claim_similarity_stats = {
            "count": len(subclaim_claim_similarities),
            "mean": float(sum(subclaim_claim_similarities) / max(1, len(subclaim_claim_similarities))),
            "min": float(min(subclaim_claim_similarities)) if subclaim_claim_similarities else 0.0,
            "max": float(max(subclaim_claim_similarities)) if subclaim_claim_similarities else 0.0,
        }

        decomposition_info = {
            "subclaims": subclaims,
            "method": method_used,
            "gate": {
                "should_split": gate_decision.should_split,
                "reason": gate_decision.reason,
                "features": gate_decision.features,
            },
            "subclaim_claim_similarity_stats": subclaim_claim_similarity_stats,
            "assignments": assignments,
        }

        graph_item = build_graph_item(
            raw_item=raw_item,
            stage_b_pred_item=pred,
            decomposition_info=decomposition_info,
            add_same_report_edges=cfg["graph"]["add_same_report_edges"],
            add_lexical_overlap_edges=cfg["graph"]["add_lexical_overlap_edges"],
            lexical_overlap_jaccard_threshold=cfg["graph"]["lexical_overlap_jaccard_threshold"],
        )
        outputs.append(graph_item)

    write_jsonl(outputs, out_path)
    print(f"Saved {len(outputs)} graph items to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(cfg["output"]["dir"])

    _build_one_split(cfg["data"]["raw_train_path"], cfg["data"]["stage_b_train_predictions"], str(Path(out_dir) / "train.graph.jsonl"), cfg, device=args.device)
    _build_one_split(cfg["data"]["raw_val_path"], cfg["data"]["stage_b_val_predictions"], str(Path(out_dir) / "val.graph.jsonl"), cfg, device=args.device)
    _build_one_split(cfg["data"]["raw_test_path"], cfg["data"]["stage_b_test_predictions"], str(Path(out_dir) / "test.graph.jsonl"), cfg, device=args.device)


if __name__ == "__main__":
    main()
