"""Selection-only eval for the Step3 set-aware listwise evidence selector."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fact_checking.selectors.listwise import (
    ListwiseSelector,
    ListwiseSelectorConfig,
    forward_listwise_examples,
)
from fact_checking.selectors.metrics import (
    build_order_control_trace,
    build_selection_trace,
    ranked_indices_from_candidate_pool,
    ranked_indices_from_hybrid,
    random_order_controls,
    reorder_predicted_set,
    summarize_ordered_selection,
)
from fact_checking.selectors.stage2_oracle import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_SELECTOR_TOP_K,
    EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT,
    load_stage2_oracle_examples,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate listwise selector against Stage2 oracle order.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--expected-chunk-mmr-fingerprint", default=EXPECTED_STAGE2_CHUNK_MMR_FINGERPRINT)
    p.add_argument("--allow-model-fingerprint-mismatch", action="store_true")
    p.add_argument("--max-candidates", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    p.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    p.add_argument("--filter-policy", default="all", choices=["all", "is_correct", "margin_positive", "high_margin"])
    p.add_argument("--min-margin", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_stage2_oracle_examples(
        args.oracle_results,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        max_candidates=args.max_candidates,
        top_k=args.top_k,
        filter_policy=args.filter_policy,
        min_margin=args.min_margin,
        sample_limit=args.sample_limit,
    )
    if not examples:
        raise ValueError("No evaluation examples after Stage2 audit/filtering.")

    selector = ListwiseSelector(
        ListwiseSelectorConfig(
            model_dir=args.model_dir,
            device=args.device,
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            strict_fingerprint=not bool(args.allow_model_fingerprint_mismatch),
            expected_chunk_mmr_fingerprint=args.expected_chunk_mmr_fingerprint,
        )
    )

    traces: list[dict] = []
    hybrid_traces: list[dict] = []
    candidate_order_traces: list[dict] = []
    same_set_hybrid_traces: list[dict] = []
    same_set_candidate_traces: list[dict] = []
    random_traces: list[dict] = []

    selector.model.eval()
    for batch in tqdm(
        _batches(examples, int(args.batch_size)),
        total=max(math.ceil(len(examples) / max(int(args.batch_size), 1)), 1),
        desc=f"listwise selector eval [{args.split}]",
        unit="batch",
        dynamic_ncols=True,
        disable=args.no_progress,
    ):
        with torch.inference_mode():
            grouped_scores = forward_listwise_examples(
                selector.model,
                selector.tokenizer,
                batch,
                device=selector.device,
                max_length=int(args.max_length),
                max_candidates=int(args.max_candidates),
            )
        for example, scores_tensor in zip(batch, grouped_scores):
            scores = scores_tensor.detach().float().cpu().numpy()
            trace = build_selection_trace(
                example,
                scores,
                selector_name="set_aware_listwise",
                top_k=int(args.top_k),
            )
            traces.append(trace)

            hybrid_traces.append(
                build_order_control_trace(
                    trace,
                    ranked_indices_from_hybrid(example, top_k=int(args.top_k)),
                    selector_name="hybrid_score_top5",
                    top_k=int(args.top_k),
                )
            )
            candidate_order_traces.append(
                build_order_control_trace(
                    trace,
                    ranked_indices_from_candidate_pool(example, top_k=int(args.top_k)),
                    selector_name="candidate_pool_order_top5",
                    top_k=int(args.top_k),
                )
            )
            predicted = [int(idx) for idx in trace["selector_ordered_indices"]]
            same_set_hybrid_traces.append(
                build_order_control_trace(
                    trace,
                    reorder_predicted_set(predicted, example=example, mode="hybrid_order"),
                    selector_name="same_set_hybrid_order",
                    top_k=int(args.top_k),
                )
            )
            same_set_candidate_traces.append(
                build_order_control_trace(
                    trace,
                    reorder_predicted_set(predicted, example=example, mode="candidate_pool_order"),
                    selector_name="same_set_candidate_pool_order",
                    top_k=int(args.top_k),
                )
            )
            random_traces.extend(
                random_order_controls(
                    predicted,
                    example=example,
                    seeds=[0, 1, 2, 3, 4],
                    top_k=int(args.top_k),
                )
            )

    selector_metrics = summarize_ordered_selection(traces)
    controls = {
        "hybrid_score_top5": summarize_ordered_selection(hybrid_traces),
        "candidate_pool_order_top5": summarize_ordered_selection(candidate_order_traces),
        "same_set_hybrid_order": summarize_ordered_selection(same_set_hybrid_traces),
        "same_set_candidate_pool_order": summarize_ordered_selection(same_set_candidate_traces),
        "same_set_random_order_mean": summarize_ordered_selection(random_traces),
    }
    metrics = {
        "model_dir": args.model_dir,
        "oracle_results": args.oracle_results,
        "split": args.split,
        "filter_policy": args.filter_policy,
        "chunk_mmr_fingerprint": args.expected_chunk_mmr_fingerprint,
        "n_claims": len(examples),
        "selector": selector_metrics,
        "controls": controls,
        "selector_metadata": selector.metadata,
    }
    write_json(out_dir / "selection_metrics.json", metrics)
    write_jsonl(out_dir / "selection_trace.jsonl", traces)
    write_jsonl(out_dir / "control_hybrid_trace.jsonl", hybrid_traces)
    write_jsonl(out_dir / "control_candidate_pool_trace.jsonl", candidate_order_traces)

    print(f"Wrote selection metrics: {out_dir / 'selection_metrics.json'}")
    print(
        "Listwise Recall@5={rec:.4f}, Jaccard@5={jac:.4f}, "
        "NDCG@5={ndcg:.4f}; Hybrid Jaccard@5={hjac:.4f}".format(
            rec=float(selector_metrics.get("recall@5", np.nan)),
            jac=float(selector_metrics.get("jaccard@5", np.nan)),
            ndcg=float(selector_metrics.get("oracle_rank_ndcg@5", np.nan)),
            hjac=float(controls["hybrid_score_top5"].get("jaccard@5", np.nan)),
        )
    )


def _batches(items: list, batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


if __name__ == "__main__":
    main()
