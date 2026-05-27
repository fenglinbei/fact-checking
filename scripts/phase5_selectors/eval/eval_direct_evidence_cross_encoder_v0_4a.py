#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    COUNT_AMPLIFIED_SELECTOR,
    CountAmplifiedParams,
    select_count_amplified_topk,
    select_order_control,
)
from fact_checking.selectors.direct_evidence_cross_encoder import (
    DIRECT_CE_SOURCE_DIVERSE_SELECTOR,
    DIRECT_CE_TEXT_ONLY_SELECTOR,
    V03_REFERENCE_SELECTOR,
    build_direct_ce_trace,
    direct_ce_diagnostics,
    select_direct_ce_topk,
    select_source_diverse_direct_ce_topk,
    summarize_direct_ce_traces,
)
from fact_checking.selectors.evidence_quality import retrieval_score
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_SCORED_FILE = "outputs/selectors/direct_evidence_cross_encoder/v0_4a_val/direct_ce_scored_candidates_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/direct_evidence_cross_encoder/v0_4a_val/eval"
DEFAULT_V03_REFERENCE = (
    "outputs/selectors/oracle_likelihood_constrained_selector/v0_3_1_val/"
    "pointwise_all_features/candidate_oracle_likelihood_scores_val.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate v0.4a direct evidence CrossEncoder selector scores.")
    p.add_argument("--scored-candidates", default=DEFAULT_SCORED_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--source-penalty", type=float, default=0.05)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--v03-reference-scored-candidates", default=DEFAULT_V03_REFERENCE)
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
        raise ValueError(f"No scored rows loaded from {args.scored_candidates}")
    v03_by_event = _load_v03_reference(args.v03_reference_scored_candidates, rows)

    traces: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        for control_name in ["original_pool_order_top5", "qd_union_pool_order_top5", "qd_union_source_score_top5", "completeness_only_top5"]:
            traces.append(
                build_direct_ce_trace(
                    row,
                    select_order_control(candidates, mode=control_name, top_k=int(args.top_k)),
                    selector_name=control_name,
                    top_k=int(args.top_k),
                )
            )
        count_selected, _, _ = select_count_amplified_topk(
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
            build_direct_ce_trace(
                row,
                count_selected,
                selector_name=COUNT_AMPLIFIED_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        traces.append(
            build_direct_ce_trace(
                row,
                select_direct_ce_topk(candidates, top_k=int(args.top_k)),
                selector_name=DIRECT_CE_TEXT_ONLY_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        traces.append(
            build_direct_ce_trace(
                row,
                select_source_diverse_direct_ce_topk(
                    candidates,
                    top_k=int(args.top_k),
                    source_penalty=float(args.source_penalty),
                ),
                selector_name=DIRECT_CE_SOURCE_DIVERSE_SELECTOR,
                top_k=int(args.top_k),
            )
        )
        ref_row = v03_by_event.get(str(row.get("event_id") or ""))
        if ref_row:
            traces.append(
                build_direct_ce_trace(
                    row,
                    _select_v03_reference_topk(ref_row.get("candidates") or [], top_k=int(args.top_k)),
                    selector_name=V03_REFERENCE_SELECTOR,
                    top_k=int(args.top_k),
                )
            )

    selector_metrics = summarize_direct_ce_traces(traces)
    diagnostics = direct_ce_diagnostics(rows, traces, primary_selector=DIRECT_CE_TEXT_ONLY_SELECTOR)
    decision = _decision(selector_metrics, diagnostics)
    diagnostics["decision"] = decision

    trace_path = out_dir / f"selection_trace_{args.split}.jsonl"
    write_jsonl(traces, trace_path)
    save_json(selector_metrics, out_dir / "selector_metrics.json")
    save_json(diagnostics, out_dir / "direct_ce_diagnostics.json")
    save_json(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "scored_candidates": str(args.scored_candidates),
            "v03_reference_scored_candidates": str(args.v03_reference_scored_candidates),
            "split": str(args.split),
            "top_k": int(args.top_k),
            "params": {"source_penalty": float(args.source_penalty)},
            "n_events": len(rows),
            "outputs": {
                "selection_trace": str(trace_path),
                "selector_metrics": str(out_dir / "selector_metrics.json"),
                "diagnostics": str(out_dir / "direct_ce_diagnostics.json"),
                "analysis_summary": str(out_dir / "analysis_summary.md"),
            },
            "decision": decision,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        out_dir / "manifest.json",
    )
    _write_analysis(out_dir / "analysis_summary.md", selector_metrics, diagnostics, decision)
    direct = selector_metrics.get(DIRECT_CE_TEXT_ONLY_SELECTOR, {})
    print(f"Wrote direct CE eval under: {out_dir}")
    print(
        "Decision={decision}; direct_jaccard={jaccard:.4f}; direct_recall={recall:.4f}; auroc={auroc:.4f}; same_source_pairwise={same_source:.4f}".format(
            decision=str(decision.get("decision")),
            jaccard=float(direct.get("jaccard@5", 0.0)),
            recall=float(direct.get("recall@5", 0.0)),
            auroc=float((diagnostics.get("candidate_level") or {}).get("auroc", 0.0)),
            same_source=float((diagnostics.get("same_source_hard_negative_pairwise") or {}).get("pairwise_acc", 0.0)),
        )
    )


def _load_v03_reference(path: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ref_path = Path(path)
    if not ref_path.exists():
        return {}
    wanted = {str(row.get("event_id") or "") for row in rows}
    return {
        str(row.get("event_id") or ""): row
        for row in read_jsonl(ref_path)
        if str(row.get("event_id") or "") in wanted
    }


def _select_v03_reference_topk(candidates: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    rows.sort(
        key=lambda row: (
            -float(row.get("oracle_likelihood_score") or 0.0),
            -retrieval_score(row),
            str(row.get("candidate_key") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("candidate_key") or row.get("candidate_uid") or "")
        if not key or key in seen:
            continue
        item = dict(row)
        item["selector_name"] = V03_REFERENCE_SELECTOR
        item["selection_origin"] = "v0_3_1_oracle_likelihood_reference"
        item["selection_rank"] = len(selected) + 1
        selected.append(item)
        seen.add(key)
        if len(selected) >= int(top_k):
            break
    return selected


def _decision(selector_metrics: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    direct = selector_metrics.get(DIRECT_CE_TEXT_ONLY_SELECTOR, {})
    v03 = selector_metrics.get(V03_REFERENCE_SELECTOR, {})
    original = selector_metrics.get("original_pool_order_top5", {})
    qd_source = selector_metrics.get("qd_union_source_score_top5", {})
    auroc = float((diagnostics.get("candidate_level") or {}).get("auroc", 0.0))
    same_source_acc = float((diagnostics.get("same_source_hard_negative_pairwise") or {}).get("pairwise_acc", 0.0))
    direct_jaccard = float(direct.get("jaccard@5", 0.0))
    direct_recall = float(direct.get("recall@5", 0.0))
    v03_jaccard = float(v03.get("jaccard@5", 0.2445))
    best_control_jaccard = max(float(original.get("jaccard@5", 0.0)), float(qd_source.get("jaccard@5", 0.0)))
    passes = auroc > 0.56 and same_source_acc > 0.57 and direct_jaccard >= 0.250
    beats_v03 = direct_jaccard > v03_jaccard
    if passes and beats_v03:
        name = "go_direct_evidence_ce_v0_4a"
    elif auroc <= 0.52 and direct_jaccard <= best_control_jaccard:
        name = "stop_weak_text_only_direct_evidence_signal_v0_4a"
    else:
        name = "analysis_text_only_signal_needs_training_or_fusion_v0_4a"
    return {
        "decision": name,
        "direct_ce_jaccard@5": direct_jaccard,
        "direct_ce_recall@5": direct_recall,
        "v0_3_1_reference_jaccard@5": v03_jaccard,
        "best_control_jaccard@5": best_control_jaccard,
        "candidate_auroc": auroc,
        "same_source_hard_negative_pairwise_acc": same_source_acc,
        "passes_v0_4a_gate": bool(passes),
        "beats_v0_3_1_reference": bool(beats_v03),
    }


def _write_analysis(
    path: Path,
    selector_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Direct Evidence Cross-Encoder v0.4a",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- candidate_auroc: `{float(decision.get('candidate_auroc', 0.0)):.4f}`",
        f"- same_source_hard_negative_pairwise_acc: `{float(decision.get('same_source_hard_negative_pairwise_acc', 0.0)):.4f}`",
        f"- oracle_selected_score_lift: `{float(diagnostics.get('oracle_selected_score_lift', 0.0)):.4f}`",
        "",
        "## Selector Metrics",
        "",
        "| selector | recall@5 | jaccard@5 | top1 | ndcg@5 | pairwise@5 | direct_ce@5 | hit_rate@5 | source_entropy | stance_entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, metrics in sorted(selector_metrics.items()):
        lines.append(
            "| {selector} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | {ndcg:.4f} | {pairwise:.4f} | {score:.4f} | {hit:.4f} | {source:.4f} | {stance:.4f} |".format(
                selector=selector,
                recall=float(metrics.get("recall@5", 0.0)),
                jaccard=float(metrics.get("jaccard@5", 0.0)),
                top1=float(metrics.get("top1_match", 0.0)),
                ndcg=float(metrics.get("oracle_rank_ndcg@5", 0.0)),
                pairwise=float(metrics.get("pairwise_order_acc@5", 0.0)),
                score=float(metrics.get("mean_direct_ce_score@5", 0.0)),
                hit=float(metrics.get("direct_ce_hit_rate@5", 0.0)),
                source=float(metrics.get("source_entropy@5", 0.0)),
                stance=float(metrics.get("stance_bucket_entropy@5", 0.0)),
            )
        )
    candidate = diagnostics.get("candidate_level") or {}
    same_source = diagnostics.get("same_source_hard_negative_pairwise") or {}
    within = diagnostics.get("within_event_pairwise") or {}
    high_ret = diagnostics.get("high_retrieval_non_oracle_false_positive_rate") or {}
    lines.extend(
        [
            "",
            "## Candidate Diagnostics",
            "",
            f"- auroc: `{float(candidate.get('auroc', 0.0)):.4f}`",
            f"- auprc: `{float(candidate.get('auprc', 0.0)):.4f}`",
            f"- same_source_pairwise_acc: `{float(same_source.get('pairwise_acc', 0.0)):.4f}` over `{int(same_source.get('n_pairs', 0))}` pairs",
            f"- within_event_pairwise_acc: `{float(within.get('pairwise_acc', 0.0)):.4f}` over `{int(within.get('n_pairs', 0))}` pairs",
            f"- high_retrieval_non_oracle_rate_score_ge_0_5: `{float(high_ret.get('rate_score_ge_0_5', 0.0)):.4f}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _infer_n_buckets(candidates: list[dict[str, Any]]) -> int:
    for candidate in candidates:
        probs = candidate.get("teacher_stance_probs")
        if isinstance(probs, dict) and probs:
            return len(probs)
    return 0


if __name__ == "__main__":
    main()
