"""Build pointwise oracle-supervision rows from oracle evidence search output."""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from fact_checking.oracle_pointwise import (
    ANCHOR_WEIGHTS,
    DEFAULT_FEATURE_NAMES,
    LEGACY_POSITIVE_INJECTION_POOL_MODE,
    PIPELINE_POOL_MODE,
    TRUE_SIDE_LABELS,
    build_candidate_pool,
    build_pipeline_style_candidate_pool,
    finite_or_zero,
    load_build_config,
    load_chunk_samples_by_event,
    pool_to_pointwise_rows,
    read_jsonl,
    resolve_chunk_cache_path,
    summarize_filtering,
    supervision_policy_for_record,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pointwise rows for oracle evidence supervision.")
    p.add_argument("--oracle-results", required=True, help="oracle_results_<split>.jsonl")
    p.add_argument("--config", default="configs/experiment/b3_mmr_topk_sweep_1024.yaml")
    p.add_argument("--config-overrides", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--chunk-mmr-cache", default=None, help="Explicit Chunk-MMR cache pickle")
    p.add_argument("--chunk-mmr-cache-root", default="outputs/cache/chunk_mmr")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--filter-preset", default="v1a", choices=["v1a", "v1b", "all"])
    p.add_argument("--mostly-true-anchor-weight", type=float, default=ANCHOR_WEIGHTS["mostly-true"])
    p.add_argument("--true-anchor-weight", type=float, default=ANCHOR_WEIGHTS["true"])
    p.add_argument(
        "--max-true-side-anchors-per-label",
        type=int,
        default=0,
        help="Optional cap for V1b true-side anchors per label; 0 means no cap.",
    )
    p.add_argument(
        "--fixed-mmr-predictions",
        default=None,
        help="Optional fixed-MMR predictions JSONL. Correct true-side rows become conservative anchors.",
    )
    p.add_argument(
        "--fixed-mmr-candidates",
        default=None,
        help="Build JSONL aligned with --fixed-mmr-predictions sample_idx; supplies fixed-MMR selected texts.",
    )
    p.add_argument(
        "--pool-mode",
        default=PIPELINE_POOL_MODE,
        choices=[PIPELINE_POOL_MODE, LEGACY_POSITIVE_INJECTION_POOL_MODE],
    )
    p.add_argument("--fallback-pool-size", type=int, default=15)
    p.add_argument(
        "--expected-chunk-mmr-fingerprint",
        default=None,
        help="Optional hard expectation for the Chunk-MMR cache fingerprint.",
    )
    p.add_argument(
        "--allow-cache-fingerprint-mismatch",
        action="store_true",
        help="Permit an explicit --chunk-mmr-cache whose parent fingerprint differs from the config.",
    )
    p.add_argument(
        "--allow-missing-oracle-candidate-pool",
        action="store_true",
        help="Only for legacy diagnostics; pipeline-style data should use saved oracle candidate_pool.",
    )
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_cfg = load_build_config(
        args.config,
        config_overrides=args.config_overrides,
        model_base_path=args.model_base_path,
    )
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    alpha_dense = float(args.alpha_dense if args.alpha_dense is not None else retrieval_cfg.get("alpha_dense", 0.70))
    alpha_lexical = float(args.alpha_lexical if args.alpha_lexical is not None else retrieval_cfg.get("alpha_lexical", 0.20))
    alpha_bm25 = float(args.alpha_bm25 if args.alpha_bm25 is not None else retrieval_cfg.get("alpha_bm25", 0.10))

    cache_path, cache_resolution = resolve_chunk_cache_path(
        build_cfg,
        split=args.split,
        cache_root=args.chunk_mmr_cache_root,
        explicit_path=args.chunk_mmr_cache,
        expected_fingerprint=args.expected_chunk_mmr_fingerprint,
        allow_explicit_mismatch=args.allow_cache_fingerprint_mismatch,
    )
    chunk_mmr_fingerprint = str(
        cache_resolution.get("fingerprint")
        or cache_resolution.get("expected_fingerprint")
        or cache_path.parent.name
    )
    samples_by_event = load_chunk_samples_by_event(cache_path)
    oracle_records = read_jsonl(args.oracle_results)
    if args.sample_limit is not None:
        oracle_records = oracle_records[: args.sample_limit]
    fixed_anchor_texts = _load_fixed_anchor_texts(
        args.fixed_mmr_predictions,
        args.fixed_mmr_candidates,
    )
    fixed_correct_event_ids = set(fixed_anchor_texts)
    true_side_anchor_weights = {
        "mostly-true": float(args.mostly_true_anchor_weight),
        "true": float(args.true_anchor_weight),
    }

    rows = []
    kept_event_ids: set[str] = set()
    matched_counts: list[tuple[int, int]] = []
    kept_by_bucket: dict[str, int] = {}
    kept_weight_by_bucket: dict[str, float] = {}
    anchor_kept_by_label: dict[str, int] = {label: 0 for label in TRUE_SIDE_LABELS}
    skipped = {
        "missing_cache_sample": 0,
        "filter": 0,
        "anchor_cap": 0,
        "no_candidates": 0,
        "no_positive_match": 0,
    }

    iterator = tqdm(
        oracle_records,
        desc="build pointwise rows",
        unit="claim",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for rec in iterator:
        eid = str(rec.get("event_id", ""))
        sample = samples_by_event.get(eid)
        if sample is None:
            skipped["missing_cache_sample"] += 1
            continue
        policy = supervision_policy_for_record(
            rec,
            args.filter_preset,
            fixed_correct_event_ids=fixed_correct_event_ids,
            true_side_anchor_weights=true_side_anchor_weights,
        )
        if not policy["keep"]:
            skipped["filter"] += 1
            continue
        label = str(rec.get("gold_label", "")).lower()
        if (
            args.max_true_side_anchors_per_label > 0
            and label in TRUE_SIDE_LABELS
            and str(policy.get("bucket", "")).endswith("_anchor")
            and anchor_kept_by_label[label] >= args.max_true_side_anchors_per_label
        ):
            skipped["anchor_cap"] += 1
            continue

        rec_for_pool = dict(rec)
        if policy.get("anchor_source") == "fixed_mmr_correct" and eid in fixed_anchor_texts:
            if args.pool_mode == PIPELINE_POOL_MODE:
                raise ValueError(
                    "fixed_mmr_correct anchors are not compatible with pipeline-style oracle pools; "
                    "use the legacy positive-injection mode only for that diagnostic."
                )
            rec_for_pool["selected_texts"] = fixed_anchor_texts[eid]
            rec_for_pool["n_candidates"] = max(
                int(rec_for_pool.get("n_candidates") or 0),
                int(args.fallback_pool_size),
            )

        if args.pool_mode == PIPELINE_POOL_MODE:
            pool = build_pipeline_style_candidate_pool(
                sample,
                rec_for_pool,
                alpha_dense=alpha_dense,
                alpha_lexical=alpha_lexical,
                alpha_bm25=alpha_bm25,
                fallback_pool_size=args.fallback_pool_size,
                require_oracle_candidate_pool=not args.allow_missing_oracle_candidate_pool,
                expected_chunk_mmr_fingerprint=chunk_mmr_fingerprint,
            )
        else:
            pool = build_candidate_pool(
                sample,
                rec_for_pool,
                alpha_dense=alpha_dense,
                alpha_lexical=alpha_lexical,
                alpha_bm25=alpha_bm25,
                pool_mode=args.pool_mode,
                fallback_pool_size=args.fallback_pool_size,
            )
            pool.chunk_mmr_fingerprint = chunk_mmr_fingerprint
        if not pool.candidates:
            skipped["no_candidates"] += 1
            continue
        matched_counts.append((pool.matched_positive_count, pool.oracle_positive_count))
        if pool.matched_positive_count <= 0:
            skipped["no_positive_match"] += 1
            continue
        kept_event_ids.add(eid)
        bucket = str(policy["bucket"])
        weight = float(policy["supervision_weight"])
        kept_by_bucket[bucket] = kept_by_bucket.get(bucket, 0) + 1
        kept_weight_by_bucket[bucket] = kept_weight_by_bucket.get(bucket, 0.0) + weight
        if label in TRUE_SIDE_LABELS and bucket.endswith("_anchor"):
            anchor_kept_by_label[label] += 1
        rows.extend(
            pool_to_pointwise_rows(
                pool,
                rec_for_pool,
                bucket,
                supervision_weight=weight,
                anchor_source=str(policy.get("anchor_source", "")),
            )
        )

    train_path = output_dir / f"{args.split}_pointwise.jsonl"
    write_jsonl(train_path, rows)
    schema = {
        "feature_names": DEFAULT_FEATURE_NAMES,
        "label_name": "is_oracle_selected",
        "pool_mode": args.pool_mode,
        "candidate_pool_source": (
            "oracle_results_candidate_pool"
            if args.pool_mode == PIPELINE_POOL_MODE
            else "legacy_reconstructed_with_positive_injection"
        ),
        "chunk_mmr_fingerprint": chunk_mmr_fingerprint,
        "oracle_results": args.oracle_results,
        "chunk_mmr_cache": str(cache_path),
        "filter_preset": args.filter_preset,
        "supervision_weight_name": "supervision_weight",
        "anchor_weights": true_side_anchor_weights,
    }
    write_json(output_dir / "feature_schema.json", schema)

    report = summarize_filtering(
        oracle_records,
        kept_event_ids,
        matched_counts=matched_counts,
        pool_mode=args.pool_mode,
        cache_path=str(cache_path),
    )
    report.update({
        "cache_resolution": cache_resolution,
        "chunk_mmr_fingerprint": chunk_mmr_fingerprint,
        "split": args.split,
        "output_rows": len(rows),
        "skipped": skipped,
        "kept_by_bucket": kept_by_bucket,
        "kept_weight_by_bucket": {
            key: finite_or_zero(value)
            for key, value in kept_weight_by_bucket.items()
        },
        "true_side_anchor_kept_by_label": anchor_kept_by_label,
        "fixed_mmr_anchor_events": len(fixed_correct_event_ids),
        "feature_names": DEFAULT_FEATURE_NAMES,
        "score_weights": {
            "alpha_dense": finite_or_zero(alpha_dense),
            "alpha_lexical": finite_or_zero(alpha_lexical),
            "alpha_bm25": finite_or_zero(alpha_bm25),
        },
        "notes": [
            "Rows are candidate-level examples; event_id must be used for grouped splits.",
            "Default rows use pipeline-style pools: dedup -> hybrid top candidate_pool_size -> selector topK.",
            "pipeline_hybrid_topk requires the saved oracle candidate_pool and matching chunk_mmr_fingerprint.",
            "oracle_n_top_hybrid_with_positives is legacy only and should not be used as a selection-only gate.",
        ],
    })
    write_json(output_dir / "filter_report.json", report)

    print(f"Wrote {len(rows)} rows to {train_path}")
    print(f"Kept {len(kept_event_ids)} claims; positive match rate={report['positive_text_match']['rate']:.4f}")
    print(f"Filter report: {output_dir / 'filter_report.json'}")


def _load_fixed_anchor_texts(
    predictions_path: str | None,
    candidates_path: str | None,
) -> dict[str, list[str]]:
    if not predictions_path and not candidates_path:
        return {}
    if not predictions_path or not candidates_path:
        raise ValueError("--fixed-mmr-predictions and --fixed-mmr-candidates must be provided together.")

    predictions = read_jsonl(predictions_path)
    candidates = read_jsonl(candidates_path)
    anchors: dict[str, list[str]] = {}
    for pred in predictions:
        idx = int(pred.get("sample_idx", -1))
        if idx < 0 or idx >= len(candidates):
            continue
        gold_label = str(pred.get("gold_label", "")).lower()
        if gold_label not in TRUE_SIDE_LABELS:
            continue
        pred_label = str(pred.get("pred_label", "")).lower()
        pred_correct = pred_label == gold_label
        if not pred_correct and pred.get("pred_id") is not None and pred.get("gold_id") is not None:
            pred_correct = int(pred["pred_id"]) == int(pred["gold_id"])
        if not pred_correct:
            continue
        row = candidates[idx]
        texts = [
            str(cand.get("text", "")).strip()
            for cand in (row.get("candidates") or [])
            if isinstance(cand, dict) and str(cand.get("text", "")).strip()
        ]
        if texts:
            anchors[str(row.get("event_id", ""))] = texts
    return anchors


if __name__ == "__main__":
    main()
