#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from fact_checking.build.cache import load_pickle
from fact_checking.selectors.roundtable import (
    ORIGINAL_POOL,
    QD_UNION_POOL,
    RoundtableParams,
    add_conflicting_factions,
    apply_auxiliary_labels,
    build_auxiliary_index,
    build_pool_comparison,
    build_selection_trace,
    candidate_role_rows,
    cluster_factions_for_pool,
    decision_payload,
    normalize_original_candidates,
    normalize_qd_union_candidates,
    oracle_ordered_keys,
    pool_summary,
    select_pool_order,
    select_qd_source_score,
    select_roundtable_topk,
    summarize_faction_metrics,
    summarize_pool_deltas,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_QD_UNION = "outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl"
DEFAULT_TRAIN_QD_UNION = "outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl"
DEFAULT_CHUNK_CACHE_TMPL = "outputs/cache/chunk_mmr/432dfc970e75/{split}.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build roundtable evidence-map analysis over original Stage2 and QD union pools."
    )
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--qd-union-pool-jsonl", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--chunk-cache-path", default=None)
    p.add_argument("--stance-scores", nargs="*", default=None)
    p.add_argument("--aspect-alignment", nargs="*", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--similarity-threshold", type=float, default=0.72)
    p.add_argument("--min-factions", type=int, default=2)
    p.add_argument("--max-factions", type=int, default=6)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    split = str(args.split)
    oracle_results = args.oracle_results or _default_oracle_results(split)
    qd_union_pool = args.qd_union_pool_jsonl or _default_qd_union_pool(split)
    chunk_cache_path = args.chunk_cache_path or DEFAULT_CHUNK_CACHE_TMPL.format(split=split)
    output_dir = Path(args.output_dir or f"outputs/selectors/roundtable_evidence_map/qwen_qd_union_vs_original_v0_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = read_jsonl(oracle_results)
    if args.sample_limit is not None:
        oracle_rows = oracle_rows[: int(args.sample_limit)]
    if not oracle_rows:
        raise ValueError("No oracle rows loaded.")

    qd_rows = read_jsonl(qd_union_pool)
    qd_by_event = {str(row.get("event_id") or ""): row for row in qd_rows}
    chunk_samples = _load_chunk_samples(chunk_cache_path)
    stance_index = build_auxiliary_index(_read_jsonl_many(args.stance_scores or []))
    aspect_index = build_auxiliary_index(_read_jsonl_many(args.aspect_alignment or []))

    params = RoundtableParams(
        top_k=int(args.top_k),
        similarity_threshold=float(args.similarity_threshold),
        min_factions=int(args.min_factions),
        max_factions=int(args.max_factions),
    )

    pool_comparison_rows: list[dict[str, Any]] = []
    faction_map_rows: list[dict[str, Any]] = []
    candidate_label_rows: list[dict[str, Any]] = []
    selection_trace_rows: list[dict[str, Any]] = []
    missing_qd_events: list[str] = []

    iterator = tqdm(
        oracle_rows,
        desc="roundtable",
        unit="claim",
        dynamic_ncols=True,
        disable=bool(args.no_progress),
    )
    for oracle_row in iterator:
        event_id = str(oracle_row.get("event_id") or "")
        qd_row = qd_by_event.get(event_id)
        if qd_row is None:
            missing_qd_events.append(event_id)
            qd_row = {"event_id": event_id, "claim": oracle_row.get("claim"), "candidates": []}

        oracle_keys = oracle_ordered_keys(oracle_row)
        oracle_key_to_step = {key: step for step, key in enumerate(oracle_keys)}
        sample = chunk_samples.get(event_id)

        original_candidates = normalize_original_candidates(oracle_row)
        qd_candidates = normalize_qd_union_candidates(qd_row, oracle_key_to_step=oracle_key_to_step)
        original_candidates = apply_auxiliary_labels(
            original_candidates,
            stance_index=stance_index,
            aspect_index=aspect_index,
        )
        qd_candidates = apply_auxiliary_labels(
            qd_candidates,
            stance_index=stance_index,
            aspect_index=aspect_index,
        )

        original_candidates, original_factions = cluster_factions_for_pool(
            original_candidates,
            sample=sample,
            params=params,
        )
        qd_candidates, qd_factions = cluster_factions_for_pool(
            qd_candidates,
            sample=sample,
            params=params,
        )
        original_factions = add_conflicting_factions(original_factions)
        qd_factions = add_conflicting_factions(qd_factions)

        comparison = build_pool_comparison(
            event_id=event_id,
            oracle_keys=oracle_keys,
            original_candidates=original_candidates,
            qd_candidates=qd_candidates,
        )
        comparison["claim"] = str(oracle_row.get("claim") or "")
        comparison["gold_label"] = str(oracle_row.get("gold_label") or "")
        pool_comparison_rows.append(comparison)

        original_summary = pool_summary(original_candidates, original_factions)
        qd_summary = pool_summary(qd_candidates, qd_factions)
        faction_map_rows.append(
            {
                "event_id": event_id,
                "claim": str(oracle_row.get("claim") or ""),
                "gold_label": str(oracle_row.get("gold_label") or ""),
                "oracle_ordered_keys": oracle_keys,
                ORIGINAL_POOL: original_summary,
                QD_UNION_POOL: qd_summary,
                "pool_delta": {
                    key: comparison[key]
                    for key in (
                        "overlap_count",
                        "qd_only_count",
                        "original_only_count",
                        "oracle_selected_preserved_by_qd_union_count",
                        "oracle_selected_dropped_by_qd_union_count",
                    )
                },
            }
        )

        candidate_label_rows.extend(candidate_role_rows(original_candidates))
        candidate_label_rows.extend(candidate_role_rows(qd_candidates))

        selector_rows = _selection_rows(
            event_id=event_id,
            claim=str(oracle_row.get("claim") or ""),
            gold_label=str(oracle_row.get("gold_label") or ""),
            original_candidates=original_candidates,
            original_factions=original_factions,
            qd_candidates=qd_candidates,
            qd_factions=qd_factions,
            oracle_keys=oracle_keys,
            top_k=int(args.top_k),
        )
        selection_trace_rows.extend(selector_rows)

    pool_delta_metrics = summarize_pool_deltas(pool_comparison_rows)
    faction_metrics = summarize_faction_metrics(
        faction_rows=faction_map_rows,
        selection_traces=selection_trace_rows,
        pool_comparisons=pool_comparison_rows,
    )
    decision = decision_payload(
        faction_metrics=faction_metrics,
        pool_delta_metrics=pool_delta_metrics,
    )
    analysis_summary = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "n_events": len(oracle_rows),
        "n_missing_qd_events": len(missing_qd_events),
        "missing_qd_events_sample": missing_qd_events[:10],
        "oracle_results": str(oracle_results),
        "qd_union_pool_jsonl": str(qd_union_pool),
        "chunk_cache_path": str(chunk_cache_path),
        "stance_scores": [str(path) for path in (args.stance_scores or [])],
        "aspect_alignment": [str(path) for path in (args.aspect_alignment or [])],
        "pool_delta_metrics": pool_delta_metrics,
        "decision": decision,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }

    write_jsonl(pool_comparison_rows, output_dir / f"pool_comparison_{split}.jsonl")
    write_jsonl(faction_map_rows, output_dir / f"faction_map_{split}.jsonl")
    write_jsonl(candidate_label_rows, output_dir / f"candidate_role_labels_{split}.jsonl")
    write_jsonl(selection_trace_rows, output_dir / f"roundtable_selection_trace_{split}.jsonl")
    save_json(faction_metrics, output_dir / "faction_metrics.json")
    save_json(pool_delta_metrics, output_dir / "pool_delta_metrics.json")
    save_json(analysis_summary, output_dir / "analysis_summary.json")
    save_json(
        {
            "status": "completed",
            "created_at": analysis_summary["created_at"],
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "params": params.__dict__,
            "inputs": {
                "oracle_results": str(oracle_results),
                "qd_union_pool_jsonl": str(qd_union_pool),
                "chunk_cache_path": str(chunk_cache_path),
                "stance_scores": [str(path) for path in (args.stance_scores or [])],
                "aspect_alignment": [str(path) for path in (args.aspect_alignment or [])],
            },
            "outputs": {
                "pool_comparison": str(output_dir / f"pool_comparison_{split}.jsonl"),
                "faction_map": str(output_dir / f"faction_map_{split}.jsonl"),
                "candidate_role_labels": str(output_dir / f"candidate_role_labels_{split}.jsonl"),
                "selection_trace": str(output_dir / f"roundtable_selection_trace_{split}.jsonl"),
                "faction_metrics": str(output_dir / "faction_metrics.json"),
                "pool_delta_metrics": str(output_dir / "pool_delta_metrics.json"),
                "analysis_summary": str(output_dir / "analysis_summary.json"),
            },
            "elapsed_seconds": analysis_summary["elapsed_seconds"],
        },
        output_dir / "manifest.json",
    )
    _write_markdown(output_dir / "analysis.md", analysis_summary, faction_metrics)

    print(f"Wrote roundtable analysis under: {output_dir}")
    print(
        "Decision={decision}; qd_preserved={preserved:.4f}; qd_drop={drop:.4f}; "
        "roundtable_qd_jaccard={jaccard:.4f}".format(
            decision=decision["decision"],
            preserved=float(pool_delta_metrics.get("qd_union_preserved_oracle_selected_rate", 0.0)),
            drop=float(pool_delta_metrics.get("qd_union_dropped_oracle_selected_rate", 0.0)),
            jaccard=float(
                faction_metrics.get("selectors", {})
                .get("roundtable_qd_union_top5", {})
                .get("jaccard@5", 0.0)
            ),
        )
    )


def _selection_rows(
    *,
    event_id: str,
    claim: str,
    gold_label: str,
    original_candidates: list[dict[str, Any]],
    original_factions: list[dict[str, Any]],
    qd_candidates: list[dict[str, Any]],
    qd_factions: list[dict[str, Any]],
    oracle_keys: Sequence[str],
    top_k: int,
) -> list[dict[str, Any]]:
    original_oracle_factions = _oracle_faction_ids(original_candidates)
    qd_oracle_factions = _oracle_faction_ids(qd_candidates)
    selections = [
        (
            ORIGINAL_POOL,
            "original_pool_order_top5",
            select_pool_order(original_candidates, top_k=top_k, selector_name="original_pool_order_top5"),
            original_oracle_factions,
        ),
        (
            QD_UNION_POOL,
            "qd_union_pool_order_top5",
            select_pool_order(qd_candidates, top_k=top_k, selector_name="qd_union_pool_order_top5"),
            qd_oracle_factions,
        ),
        (
            QD_UNION_POOL,
            "qd_union_source_score_top5",
            select_qd_source_score(qd_candidates, top_k=top_k),
            qd_oracle_factions,
        ),
        (
            ORIGINAL_POOL,
            "roundtable_original_top5",
            select_roundtable_topk(
                original_candidates,
                original_factions,
                top_k=top_k,
                selector_name="roundtable_original_top5",
            ),
            original_oracle_factions,
        ),
        (
            QD_UNION_POOL,
            "roundtable_qd_union_top5",
            select_roundtable_topk(
                qd_candidates,
                qd_factions,
                top_k=top_k,
                selector_name="roundtable_qd_union_top5",
            ),
            qd_oracle_factions,
        ),
    ]
    return [
        build_selection_trace(
            event_id=event_id,
            claim=claim,
            gold_label=gold_label,
            pool_name=pool_name,
            selector_name=selector_name,
            selected=selected,
            oracle_keys=oracle_keys,
            oracle_faction_ids=oracle_faction_ids,
            top_k=top_k,
        )
        for pool_name, selector_name, selected, oracle_faction_ids in selections
    ]


def _oracle_faction_ids(candidates: Sequence[dict[str, Any]]) -> list[str]:
    values = [
        str(candidate.get("faction_id") or "")
        for candidate in candidates
        if bool(candidate.get("oracle_selected")) and candidate.get("faction_id")
    ]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _load_chunk_samples(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    samples = load_pickle(cache_path)
    return {str(sample.event_id): sample for sample in samples}


def _read_jsonl_many(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {path}")
        rows.extend(read_jsonl(path))
    return rows


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _default_qd_union_pool(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_QD_UNION
    return DEFAULT_VAL_QD_UNION


def _write_markdown(path: Path, summary: dict[str, Any], faction_metrics: dict[str, Any]) -> None:
    decision = summary.get("decision", {})
    pool_delta = summary.get("pool_delta_metrics", {})
    selectors = faction_metrics.get("selectors", {})
    lines = [
        "# Roundtable Evidence Map Analysis",
        "",
        f"- split: `{summary.get('split')}`",
        f"- n_events: `{summary.get('n_events')}`",
        f"- decision: `{decision.get('decision')}`",
        f"- qd_union_preserved_oracle_selected_rate: `{float(pool_delta.get('qd_union_preserved_oracle_selected_rate', 0.0)):.4f}`",
        f"- qd_union_dropped_oracle_selected_rate: `{float(pool_delta.get('qd_union_dropped_oracle_selected_rate', 0.0)):.4f}`",
        "",
        "## Selector Metrics",
        "",
        "| selector | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | source_domains@5 | oracle_faction_recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in sorted(selectors.items()):
        lines.append(
            "| {name} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | {ndcg:.4f} | {domains:.4f} | {faction:.4f} |".format(
                name=name,
                recall=float(metrics.get("recall@5", 0.0)),
                jaccard=float(metrics.get("jaccard@5", 0.0)),
                top1=float(metrics.get("top1_match", 0.0)),
                ndcg=float(metrics.get("oracle_rank_ndcg@5", 0.0)),
                domains=float(metrics.get("mean_source_domains_per_top5", 0.0)),
                faction=float(metrics.get("oracle_faction_recall@5", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Pool Structure",
            "",
            "| pool | mean_pool_size | mean_factions | collapse_rate | mean_source_domains | stance_entropy | question_coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pool_name, metrics in sorted(faction_metrics.get("pools", {}).items()):
        lines.append(
            "| {pool} | {size:.4f} | {factions:.4f} | {collapse:.4f} | {domains:.4f} | {stance:.4f} | {questions:.4f} |".format(
                pool=pool_name,
                size=float(metrics.get("mean_pool_size", 0.0)),
                factions=float(metrics.get("mean_factions_per_event", 0.0)),
                collapse=float(metrics.get("single_faction_collapse_rate", 0.0)),
                domains=float(metrics.get("mean_source_domains", 0.0)),
                stance=float(metrics.get("mean_stance_entropy", 0.0)),
                questions=float(metrics.get("mean_question_coverage", 0.0)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
