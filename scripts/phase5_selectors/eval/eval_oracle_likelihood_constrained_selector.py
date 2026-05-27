#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    COUNT_AMPLIFIED_SELECTOR,
    CountAmplifiedParams,
    select_count_amplified_topk,
    select_order_control,
)
from fact_checking.selectors.evidence_quality import retrieval_score, source_group
from fact_checking.selectors.oracle_likelihood_constrained_selector import (
    ORACLE_LIKELIHOOD_SELECTOR,
    PRIMARY_SELECTOR,
    SOURCE_DIVERSE_ORACLE_LIKELIHOOD_SELECTOR,
    STAGE2_ANCHOR1_ORACLE_LIKELIHOOD_SELECTOR,
    ConstrainedSelectionParams,
    build_oracle_likelihood_trace,
    oracle_likelihood_diagnostics,
    select_constrained_likelihood_topk,
    select_likelihood_topk,
    summarize_oracle_likelihood_traces,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_SCORED_FILE = "outputs/selectors/oracle_likelihood_constrained_selector/v0_3_val/candidate_oracle_likelihood_scores_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/oracle_likelihood_constrained_selector/v0_3_val/eval"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate oracle-likelihood constrained selector outputs.")
    p.add_argument("--scored-candidates", default=DEFAULT_SCORED_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--anchor-k", type=int, default=2)
    p.add_argument("--source-penalty", type=float, default=0.10)
    p.add_argument("--stance-region-penalty", type=float, default=0.04)
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.scored_candidates)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise ValueError("No scored candidate rows loaded.")

    traces: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        traces.extend(_control_traces(row, top_k=int(args.top_k)))
        traces.append(
            build_oracle_likelihood_trace(
                row,
                select_likelihood_topk(candidates, top_k=int(args.top_k), selector_name=ORACLE_LIKELIHOOD_SELECTOR),
                selector_name=ORACLE_LIKELIHOOD_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        source_diverse = select_constrained_likelihood_topk(
            candidates,
            params=ConstrainedSelectionParams(
                top_k=int(args.top_k),
                anchor_k=0,
                source_penalty=float(args.source_penalty),
                stance_region_penalty=0.0,
            ),
            selector_name=SOURCE_DIVERSE_ORACLE_LIKELIHOOD_SELECTOR,
        )
        traces.append(
            build_oracle_likelihood_trace(
                row,
                source_diverse,
                selector_name=SOURCE_DIVERSE_ORACLE_LIKELIHOOD_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        anchor1 = select_constrained_likelihood_topk(
            candidates,
            params=ConstrainedSelectionParams(
                top_k=int(args.top_k),
                anchor_k=1,
                source_penalty=0.0,
                stance_region_penalty=0.0,
            ),
            selector_name=STAGE2_ANCHOR1_ORACLE_LIKELIHOOD_SELECTOR,
        )
        traces.append(
            build_oracle_likelihood_trace(
                row,
                anchor1,
                selector_name=STAGE2_ANCHOR1_ORACLE_LIKELIHOOD_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        primary = select_constrained_likelihood_topk(
            candidates,
            params=ConstrainedSelectionParams(
                top_k=int(args.top_k),
                anchor_k=int(args.anchor_k),
                source_penalty=float(args.source_penalty),
                stance_region_penalty=float(args.stance_region_penalty),
            ),
            selector_name=PRIMARY_SELECTOR,
        )
        traces.append(
            build_oracle_likelihood_trace(
                row,
                primary,
                selector_name=PRIMARY_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        count_selected, slot_trace, count_payload = select_count_amplified_topk(
            candidates,
            params=CountAmplifiedParams(
                top_k=int(args.top_k),
                alpha=0.5,
                gamma_stance=0.8,
                rho=2.0,
                ambiguous_bucket_penalty=0.6,
                use_directness_scoring=True,
                adaptive_polar_quota=True,
                tau_polar_ready=0.8,
                max_forced_polar_slots=2,
                tau_c=0.50,
                tau_r=0.15,
                n_stance_buckets=int(row.get("n_stance_buckets") or _infer_n_buckets(candidates) or 7),
            ),
            selector_name=COUNT_AMPLIFIED_SELECTOR,
        )
        traces.append(
            build_oracle_likelihood_trace(
                row,
                count_selected,
                selector_name=COUNT_AMPLIFIED_SELECTOR,
                top_k=int(args.top_k),
            )
        )

    selector_metrics = summarize_oracle_likelihood_traces(traces)
    diagnostics = oracle_likelihood_diagnostics(rows, traces, primary_selector=PRIMARY_SELECTOR)
    decision = _decision(selector_metrics, diagnostics)
    diagnostics["decision"] = decision

    trace_path = out_dir / f"selection_trace_{args.split}.jsonl"
    write_jsonl(traces, trace_path)
    save_json(selector_metrics, out_dir / "selector_metrics.json")
    save_json(diagnostics, out_dir / "oracle_likelihood_diagnostics.json")
    save_json(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "scored_candidates": str(args.scored_candidates),
            "split": str(args.split),
            "top_k": int(args.top_k),
            "params": {
                "anchor_k": int(args.anchor_k),
                "source_penalty": float(args.source_penalty),
                "stance_region_penalty": float(args.stance_region_penalty),
            },
            "n_events": len(rows),
            "outputs": {
                "selection_trace": str(trace_path),
                "selector_metrics": str(out_dir / "selector_metrics.json"),
                "diagnostics": str(out_dir / "oracle_likelihood_diagnostics.json"),
                "analysis_summary": str(out_dir / "analysis_summary.md"),
            },
            "decision": decision,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        out_dir / "manifest.json",
    )
    _write_analysis(out_dir / "analysis_summary.md", selector_metrics, diagnostics, decision)
    primary_metrics = selector_metrics.get(PRIMARY_SELECTOR, {})
    print(f"Wrote oracle-likelihood eval under: {out_dir}")
    print(
        "Decision={decision}; primary_jaccard={jaccard:.4f}; primary_recall={recall:.4f}; auroc={auroc:.4f}".format(
            decision=decision["decision"],
            jaccard=float(primary_metrics.get("jaccard@5", 0.0)),
            recall=float(primary_metrics.get("recall@5", 0.0)),
            auroc=float((diagnostics.get("candidate_level") or {}).get("auroc", 0.0)),
        )
    )


def _control_traces(row: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    controls = [
        "original_pool_order_top5",
        "qd_union_pool_order_top5",
        "qd_union_source_score_top5",
        "completeness_only_top5",
    ]
    traces = [
        build_oracle_likelihood_trace(
            row,
            select_order_control(candidates, mode=name, top_k=top_k),
            selector_name=name,
            top_k=top_k,
        )
        for name in controls
    ]
    traces.append(
        build_oracle_likelihood_trace(
            row,
            _roundtable_qd_union_topk(candidates, top_k=top_k),
            selector_name="roundtable_qd_union_top5",
            top_k=top_k,
        )
    )
    return traces


def _roundtable_qd_union_topk(candidates: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    qd_candidates = [dict(candidate) for candidate in candidates if candidate.get("from_qd")]
    if not qd_candidates:
        qd_candidates = [dict(candidate) for candidate in candidates]
    try:
        from fact_checking.selectors.roundtable import (
            RoundtableParams,
            cluster_factions_for_pool,
            select_roundtable_topk,
        )

        prepared = []
        for idx, candidate in enumerate(qd_candidates):
            item = dict(candidate)
            item.setdefault("pool_position", int(item.get("union_pool_rank") or idx + 1) - 1)
            item.setdefault("roundtable_score", retrieval_score(item))
            item.setdefault("stance_to_claim", str(item.get("stance_bucket_derived") or "unclear"))
            item.setdefault("qd_question_ids", _question_ids(item))
            prepared.append(item)
        labeled, factions = cluster_factions_for_pool(
            prepared,
            sample=None,
            params=RoundtableParams(top_k=top_k),
        )
        return select_roundtable_topk(
            labeled,
            factions,
            top_k=top_k,
            selector_name="roundtable_qd_union_top5",
        )
    except Exception:
        rows = []
        seen_sources: set[str] = set()
        for candidate in sorted(qd_candidates, key=lambda row: (-retrieval_score(row), int(row.get("union_pool_rank") or 10**9))):
            item = dict(candidate)
            group = source_group(item)
            item["fallback_roundtable_score"] = retrieval_score(item) + (0.05 if group not in seen_sources else 0.0)
            rows.append(item)
            seen_sources.add(group)
        rows.sort(key=lambda row: (-float(row.get("fallback_roundtable_score") or 0.0), int(row.get("union_pool_rank") or 10**9)))
        selected = []
        seen_keys = set()
        for row in rows:
            key = str(row.get("candidate_key") or "")
            if key and key not in seen_keys:
                row["selector_name"] = "roundtable_qd_union_top5"
                row["selection_rank"] = len(selected) + 1
                selected.append(row)
                seen_keys.add(key)
            if len(selected) >= top_k:
                break
        return selected


def _question_ids(candidate: dict[str, Any]) -> list[str]:
    ids = []
    for route in candidate.get("qd_question_routes") or []:
        if isinstance(route, dict) and route.get("question_id"):
            ids.append(str(route["question_id"]))
    return sorted(set(ids))


def _decision(selector_metrics: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    primary = selector_metrics.get(PRIMARY_SELECTOR, {})
    original = selector_metrics.get("original_pool_order_top5", {})
    qd_order = selector_metrics.get("qd_union_pool_order_top5", {})
    qd_source = selector_metrics.get("qd_union_source_score_top5", {})
    likelihood = selector_metrics.get(ORACLE_LIKELIHOOD_SELECTOR, {})
    count = selector_metrics.get(COUNT_AMPLIFIED_SELECTOR, {})
    best_control_jaccard = max(float(original.get("jaccard@5", 0.0)), float(qd_source.get("jaccard@5", 0.0)))
    best_control_recall = max(float(original.get("recall@5", 0.0)), float(qd_source.get("recall@5", 0.0)))
    best_order_top1 = max(
        float(original.get("top1_match", 0.0)),
        float(qd_order.get("top1_match", 0.0)),
        float(qd_source.get("top1_match", 0.0)),
    )
    best_order_ndcg = max(
        float(original.get("oracle_rank_ndcg@5", 0.0)),
        float(qd_order.get("oracle_rank_ndcg@5", 0.0)),
        float(qd_source.get("oracle_rank_ndcg@5", 0.0)),
    )
    primary_jaccard = float(primary.get("jaccard@5", 0.0))
    primary_recall = float(primary.get("recall@5", 0.0))
    primary_top1 = float(primary.get("top1_match", 0.0))
    primary_ndcg = float(primary.get("oracle_rank_ndcg@5", 0.0))
    likelihood_jaccard = float(likelihood.get("jaccard@5", 0.0))
    count_jaccard = float(count.get("jaccard@5", 0.0))
    auroc = float((diagnostics.get("candidate_level") or {}).get("auroc", 0.0))
    go = (
        primary_jaccard >= best_control_jaccard + 0.005
        and primary_recall + 0.002 >= best_control_recall
        and primary_top1 >= best_order_top1
        and primary_ndcg >= best_order_ndcg
    )
    if go:
        decision = "go_oracle_likelihood_constrained_v0_3"
    elif auroc < 0.52:
        decision = "stop_weak_oracle_likelihood_signal_v0_3"
    elif likelihood_jaccard >= count_jaccard + 0.005:
        decision = "analysis_model_signal_positive_strategy_needs_tuning_v0_3"
    else:
        decision = "analysis_only_oracle_likelihood_constrained_v0_3"
    return {
        "decision": decision,
        "primary_jaccard@5": primary_jaccard,
        "primary_recall@5": primary_recall,
        "primary_top1_match": primary_top1,
        "primary_oracle_rank_ndcg@5": primary_ndcg,
        "best_control_jaccard@5": best_control_jaccard,
        "best_control_recall@5": best_control_recall,
        "best_order_top1_match": best_order_top1,
        "best_order_oracle_rank_ndcg@5": best_order_ndcg,
        "oracle_likelihood_top5_jaccard@5": likelihood_jaccard,
        "v0_2_count_jaccard@5": count_jaccard,
        "candidate_auroc": auroc,
        "passes_primary_gate": bool(go),
    }


def _write_analysis(
    path: Path,
    selector_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Oracle-Likelihood Constrained Selector v0.3",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- candidate_auroc: `{float(decision.get('candidate_auroc', 0.0)):.4f}`",
        f"- oracle_selected_score_lift: `{float(diagnostics.get('oracle_selected_score_lift', 0.0)):.4f}`",
        "",
        "## Selector Metrics",
        "",
        "| selector | recall@5 | jaccard@5 | top1 | ndcg@5 | pairwise@5 | learned_score@5 | anchor_count | fill_count | anchor_hit | fill_hit | source_entropy | stance_entropy | collapse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, metrics in sorted(selector_metrics.items()):
        lines.append(
            "| {selector} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | {ndcg:.4f} | {pairwise:.4f} | {score:.4f} | {anchors:.4f} | {fills:.4f} | {anchor_hit:.4f} | {fill_hit:.4f} | {source:.4f} | {stance:.4f} | {collapse:.4f} |".format(
                selector=selector,
                recall=float(metrics.get("recall@5", 0.0)),
                jaccard=float(metrics.get("jaccard@5", 0.0)),
                top1=float(metrics.get("top1_match", 0.0)),
                ndcg=float(metrics.get("oracle_rank_ndcg@5", 0.0)),
                pairwise=float(metrics.get("pairwise_order_acc@5", 0.0)),
                score=float(metrics.get("mean_oracle_likelihood_score@5", 0.0)),
                anchors=float(metrics.get("mean_anchor_count@5", 0.0)),
                fills=float(metrics.get("mean_learned_fill_count@5", 0.0)),
                anchor_hit=float(metrics.get("anchor_hit_rate@5", 0.0)),
                fill_hit=float(metrics.get("learned_fill_hit_rate@5", 0.0)),
                source=float(metrics.get("source_entropy@5", 0.0)),
                stance=float(metrics.get("stance_bucket_entropy@5", 0.0)),
                collapse=float(metrics.get("single_bucket_collapse_rate@5", 0.0)),
            )
        )
    candidate = diagnostics.get("candidate_level") or {}
    lines.extend(
        [
            "",
            "## Candidate Diagnostics",
            "",
            f"- auroc: `{float(candidate.get('auroc', 0.0)):.4f}`",
            f"- auprc: `{float(candidate.get('auprc', 0.0)):.4f}`",
            f"- brier: `{float(candidate.get('brier', 0.0)):.4f}`",
            f"- log_loss: `{float(candidate.get('log_loss', 0.0)):.4f}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _infer_n_buckets(candidates: Sequence[dict[str, Any]]) -> int:
    for candidate in candidates:
        probs = candidate.get("teacher_stance_probs")
        if isinstance(probs, dict) and probs:
            return len(probs)
    return 0


if __name__ == "__main__":
    main()
