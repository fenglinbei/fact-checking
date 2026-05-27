#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    COUNT_AMPLIFIED_SELECTOR,
    LINEAR_STANCE_SELECTOR,
    CountAmplifiedParams,
    build_selector_trace,
    bucket_count_payload,
    oracle_observation_metrics,
    select_count_amplified_topk,
    select_order_control,
    summarize_selector_traces,
)
from fact_checking.selectors.evidence_quality import retrieval_score, source_group
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the count-amplified stance-bucket selector.")
    p.add_argument("--candidate-stance-buckets", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma-values", default="1.0,1.6,1.8")
    p.add_argument("--primary-gamma", type=float, default=1.6)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--ambiguous-bucket-penalty", type=float, default=1.0)
    p.add_argument("--use-directness-scoring", action="store_true")
    p.add_argument("--adaptive-polar-quota", action="store_true")
    p.add_argument("--tau-polar-ready", type=float, default=0.8)
    p.add_argument("--max-forced-polar-slots", type=int, default=0)
    p.add_argument("--tau-c", type=float, default=0.50)
    p.add_argument("--tau-r", type=float, default=0.15)
    p.add_argument("--min-bucket-membership", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.candidate_stance_buckets)
    if not rows:
        raise ValueError("No candidate stance bucket rows loaded.")

    traces: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for row in rows:
        traces.extend(_control_traces(row, top_k=int(args.top_k)))
        candidates = list(row.get("candidates") or [])
        n_buckets = int(row.get("n_stance_buckets") or _infer_n_buckets(candidates) or 3)
        for gamma in _parse_float_list(args.gamma_values):
            selector_name = _selector_name_for_gamma(gamma, primary_gamma=float(args.primary_gamma))
            params = CountAmplifiedParams(
                top_k=int(args.top_k),
                alpha=float(args.alpha),
                gamma_stance=float(gamma),
                rho=float(args.rho),
                ambiguous_bucket_penalty=float(args.ambiguous_bucket_penalty),
                use_directness_scoring=bool(args.use_directness_scoring),
                adaptive_polar_quota=bool(args.adaptive_polar_quota),
                tau_polar_ready=float(args.tau_polar_ready),
                max_forced_polar_slots=int(args.max_forced_polar_slots),
                tau_c=float(args.tau_c),
                tau_r=float(args.tau_r),
                min_bucket_membership=args.min_bucket_membership,
                n_stance_buckets=n_buckets,
            )
            selected, slot_trace, count_payload = select_count_amplified_topk(
                candidates,
                params=params,
                selector_name=selector_name,
            )
            trace = build_selector_trace(
                row,
                selected,
                selector_name=selector_name,
                top_k=int(args.top_k),
                slot_trace=slot_trace,
                count_payload=count_payload,
            )
            traces.append(trace)
            count_rows.append(
                {
                    "event_id": str(row.get("event_id") or ""),
                    "selector_name": selector_name,
                    "n_stance_buckets": n_buckets,
                    "count_payload": count_payload,
                    "selected_stance_buckets": [
                        str(item.get("selected_stance_bucket") or item.get("stance_bucket_derived") or "")
                        for item in selected
                    ],
                    "slot_trace": list(slot_trace),
                }
            )

    selector_metrics = summarize_selector_traces(traces)
    observation_metrics = oracle_observation_metrics(rows)
    count_metrics = _summarize_count_rows(count_rows)
    decision = _decision(selector_metrics, observation_metrics)

    trace_path = out_dir / f"stance_bucket_selection_trace_{args.split}.jsonl"
    write_jsonl(traces, trace_path)
    save_json(selector_metrics, out_dir / "selector_metrics.json")
    save_json(observation_metrics, out_dir / "oracle_observation_metrics.json")
    save_json(count_metrics, out_dir / "stance_bucket_count_metrics.json")
    save_json(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "candidate_stance_buckets": str(args.candidate_stance_buckets),
            "split": str(args.split),
            "top_k": int(args.top_k),
            "params": {
                "alpha": float(args.alpha),
                "gamma_values": _parse_float_list(args.gamma_values),
                "primary_gamma": float(args.primary_gamma),
                "rho": float(args.rho),
                "ambiguous_bucket_penalty": float(args.ambiguous_bucket_penalty),
                "use_directness_scoring": bool(args.use_directness_scoring),
                "adaptive_polar_quota": bool(args.adaptive_polar_quota),
                "tau_polar_ready": float(args.tau_polar_ready),
                "max_forced_polar_slots": int(args.max_forced_polar_slots),
                "tau_c": float(args.tau_c),
                "tau_r": float(args.tau_r),
                "min_bucket_membership": args.min_bucket_membership,
            },
            "n_events": len(rows),
            "outputs": {
                "selection_trace": str(trace_path),
                "selector_metrics": str(out_dir / "selector_metrics.json"),
                "oracle_observation_metrics": str(out_dir / "oracle_observation_metrics.json"),
                "stance_bucket_count_metrics": str(out_dir / "stance_bucket_count_metrics.json"),
                "analysis": str(out_dir / "analysis.md"),
            },
            "decision": decision,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        out_dir / "manifest.json",
    )
    _write_analysis(out_dir / "analysis.md", selector_metrics, observation_metrics, count_metrics, decision)
    print(f"Wrote count-amplified eval under: {out_dir}")
    primary = selector_metrics.get(COUNT_AMPLIFIED_SELECTOR, {})
    linear = selector_metrics.get(LINEAR_STANCE_SELECTOR, {})
    print(
        "Decision={decision}; count_jaccard={count:.4f}; linear_jaccard={linear:.4f}".format(
            decision=decision["decision"],
            count=float(primary.get("jaccard@5", 0.0)),
            linear=float(linear.get("jaccard@5", 0.0)),
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
        build_selector_trace(
            row,
            select_order_control(candidates, mode=name, top_k=top_k),
            selector_name=name,
            top_k=top_k,
        )
        for name in controls
    ]
    traces.append(
        build_selector_trace(
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


def _summarize_count_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("selector_name") or "")].append(dict(row))
    payload: dict[str, Any] = {}
    for selector_name, items in sorted(grouped.items()):
        bucket_names = _first_bucket_names(items)
        effective_by_bucket = {
            bucket: _mean(
                (item.get("count_payload") or {}).get("effective_counts", {}).get(bucket, 0.0)
                for item in items
            )
            for bucket in bucket_names
        }
        raw_by_bucket = {
            bucket: _mean(
                (item.get("count_payload") or {}).get("raw_effective_counts", {}).get(bucket, 0.0)
                for item in items
            )
            for bucket in bucket_names
        }
        quality_by_bucket = {
            bucket: _mean(
                (item.get("count_payload") or {}).get("bucket_quality", {}).get(bucket, 0.0)
                for item in items
            )
            for bucket in bucket_names
        }
        polar_ready_by_bucket = {
            bucket: _mean(
                (item.get("count_payload") or {}).get("polar_ready", {}).get(bucket, 0.0)
                for item in items
            )
            for bucket in bucket_names
        }
        collapse = []
        quota_trigger = []
        forced_hits = []
        for item in items:
            selected = [bucket for bucket in item.get("selected_stance_buckets") or [] if bucket]
            if selected:
                counts = Counter(selected)
                collapse.append(float(max(counts.values()) == len(selected)))
            forced_slots = [
                slot
                for slot in item.get("slot_trace") or []
                if bool(slot.get("forced_by_adaptive_quota"))
            ]
            quota_trigger.append(float(bool(forced_slots)))
            forced_hits.extend(1.0 if bool(slot.get("oracle_selected")) else 0.0 for slot in forced_slots)
        payload[selector_name] = {
            "n_events": len(items),
            "bucket_names": bucket_names,
            "mean_effective_counts": effective_by_bucket,
            "mean_raw_effective_counts": raw_by_bucket,
            "mean_bucket_quality": quality_by_bucket,
            "mean_polar_ready": polar_ready_by_bucket,
            "mean_dedup_effective_delta": {
                bucket: float(effective_by_bucket.get(bucket, 0.0) - raw_by_bucket.get(bucket, 0.0))
                for bucket in bucket_names
            },
            "single_bucket_collapse_rate": _mean(collapse),
            "adaptive_quota_trigger_rate": _mean(quota_trigger),
            "forced_polar_slot_hit_rate": _mean(forced_hits),
            "n_forced_polar_slots": int(len(forced_hits)),
        }
    return payload


def _decision(selector_metrics: dict[str, Any], observation_metrics: dict[str, Any]) -> dict[str, Any]:
    count = selector_metrics.get(COUNT_AMPLIFIED_SELECTOR, {})
    linear = selector_metrics.get(LINEAR_STANCE_SELECTOR, {})
    qd = selector_metrics.get("qd_union_pool_order_top5", {})
    roundtable = selector_metrics.get("roundtable_qd_union_top5", {})
    completeness_lift = float(observation_metrics.get("completeness_selected_lift", 0.0))
    alignment_lift = float(observation_metrics.get("oracle_vs_pool_stance_alignment_lift", 0.0))
    count_jaccard = float(count.get("jaccard@5", 0.0))
    count_top1 = float(count.get("top1_match", 0.0))
    beats_control = count_jaccard > max(float(qd.get("jaccard@5", 0.0)), float(roundtable.get("jaccard@5", 0.0)))
    beats_linear = count_jaccard > float(linear.get("jaccard@5", 0.0)) or count_top1 > float(linear.get("top1_match", 0.0))
    if completeness_lift > 0.0 and alignment_lift > 0.0 and beats_control and beats_linear:
        decision = "go_count_amplified_stance_bucket_v0"
    elif completeness_lift > 0.0 or alignment_lift > 0.0:
        decision = "analysis_only_count_amplified_stance_bucket_v0"
    else:
        decision = "stop_count_amplified_stance_bucket_v0"
    return {
        "decision": decision,
        "completeness_selected_lift": completeness_lift,
        "oracle_vs_pool_stance_alignment_lift": alignment_lift,
        "beats_qd_or_roundtable_jaccard": bool(beats_control),
        "beats_linear_control": bool(beats_linear),
        "count_jaccard@5": count_jaccard,
        "linear_jaccard@5": float(linear.get("jaccard@5", 0.0)),
        "qd_union_jaccard@5": float(qd.get("jaccard@5", 0.0)),
        "roundtable_qd_union_jaccard@5": float(roundtable.get("jaccard@5", 0.0)),
    }


def _write_analysis(
    path: Path,
    selector_metrics: dict[str, Any],
    observation_metrics: dict[str, Any],
    count_metrics: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Count-Amplified Stance-Bucket Selector v0",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- completeness_selected_lift: `{float(observation_metrics.get('completeness_selected_lift', 0.0)):.4f}`",
        f"- oracle_selected_directness_lift: `{float(observation_metrics.get('oracle_selected_directness_lift', 0.0)):.4f}`",
        f"- oracle_vs_pool_stance_alignment_lift: `{float(observation_metrics.get('oracle_vs_pool_stance_alignment_lift', 0.0)):.4f}`",
        "",
        "## Selector Metrics",
        "",
        "| selector | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 | mean_completeness@5 | direct@5 | specificity@5 | background@5 | direct_rate@5 | source_entropy@5 | stance_entropy@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, metrics in sorted(selector_metrics.items()):
        lines.append(
            "| {selector} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | {ndcg:.4f} | {pairwise:.4f} | {comp:.4f} | {direct:.4f} | {specificity:.4f} | {background:.4f} | {direct_rate:.4f} | {source:.4f} | {stance:.4f} |".format(
                selector=selector,
                recall=float(metrics.get("recall@5", 0.0)),
                jaccard=float(metrics.get("jaccard@5", 0.0)),
                top1=float(metrics.get("top1_match", 0.0)),
                ndcg=float(metrics.get("oracle_rank_ndcg@5", 0.0)),
                pairwise=float(metrics.get("pairwise_order_acc@5", 0.0)),
                comp=float(metrics.get("mean_semantic_completeness@5", 0.0)),
                direct=float(metrics.get("mean_direct_evidence@5", 0.0)),
                specificity=float(metrics.get("mean_claim_specificity@5", 0.0)),
                background=float(metrics.get("mean_background_only@5", 0.0)),
                direct_rate=float(metrics.get("direct_or_partial_rate@5", 0.0)),
                source=float(metrics.get("source_entropy@5", 0.0)),
                stance=float(metrics.get("stance_bucket_entropy@5", 0.0)),
            )
        )
    lines.extend(["", "## Count Metrics", ""])
    for selector, metrics in sorted(count_metrics.items()):
        lines.append(
            "- `{selector}`: single_bucket_collapse_rate=`{collapse:.4f}`".format(
                selector=selector,
                collapse=float(metrics.get("single_bucket_collapse_rate", 0.0)),
            )
        )
        lines.append(
            "  adaptive_quota_trigger_rate=`{trigger:.4f}`, forced_polar_slot_hit_rate=`{hit:.4f}`, n_forced_polar_slots=`{n}`".format(
                trigger=float(metrics.get("adaptive_quota_trigger_rate", 0.0)),
                hit=float(metrics.get("forced_polar_slot_hit_rate", 0.0)),
                n=int(metrics.get("n_forced_polar_slots", 0)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _selector_name_for_gamma(gamma: float, *, primary_gamma: float) -> str:
    if abs(float(gamma) - 1.0) < 1e-9:
        return LINEAR_STANCE_SELECTOR
    if abs(float(gamma) - float(primary_gamma)) < 1e-9:
        return COUNT_AMPLIFIED_SELECTOR
    return "count_amplified_stance_bucket_gamma_{:.1f}_top5".format(float(gamma)).replace(".", "_")


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in values:
            values.append(value)
    return values or [1.0, 1.6, 1.8]


def _infer_n_buckets(candidates: Sequence[dict[str, Any]]) -> int:
    for candidate in candidates:
        probs = candidate.get("teacher_stance_probs")
        if isinstance(probs, dict) and probs:
            return len(probs)
    return 0


def _first_bucket_names(items: Sequence[dict[str, Any]]) -> list[str]:
    for item in items:
        payload = item.get("count_payload") or {}
        names = payload.get("bucket_names") or []
        if names:
            return [str(name) for name in names]
    return []


def _mean(values: Sequence[float] | Any) -> float:
    vals = [float(value) for value in values if value is not None]
    return float(np.mean(vals)) if vals else 0.0


if __name__ == "__main__":
    main()
