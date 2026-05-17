"""Build pointwise oracle-supervision rows from oracle evidence search output."""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from fact_checking.oracle_pointwise import (
    DEFAULT_FEATURE_NAMES,
    build_candidate_pool,
    finite_or_zero,
    load_build_config,
    load_chunk_samples_by_event,
    oracle_filter_passes,
    pool_to_pointwise_rows,
    read_jsonl,
    resolve_chunk_cache_path,
    summarize_filtering,
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
    p.add_argument("--filter-preset", default="v1a", choices=["v1a", "all"])
    p.add_argument(
        "--pool-mode",
        default="oracle_n_top_hybrid_with_positives",
        choices=["oracle_n_top_hybrid_with_positives"],
    )
    p.add_argument("--fallback-pool-size", type=int, default=15)
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
    )
    samples_by_event = load_chunk_samples_by_event(cache_path)
    oracle_records = read_jsonl(args.oracle_results)
    if args.sample_limit is not None:
        oracle_records = oracle_records[: args.sample_limit]

    rows = []
    kept_event_ids: set[str] = set()
    matched_counts: list[tuple[int, int]] = []
    skipped = {
        "missing_cache_sample": 0,
        "filter": 0,
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
        if not oracle_filter_passes(rec, args.filter_preset):
            skipped["filter"] += 1
            continue

        pool = build_candidate_pool(
            sample,
            rec,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
            pool_mode=args.pool_mode,
            fallback_pool_size=args.fallback_pool_size,
        )
        if not pool.candidates:
            skipped["no_candidates"] += 1
            continue
        matched_counts.append((pool.matched_positive_count, pool.oracle_positive_count))
        if pool.matched_positive_count <= 0:
            skipped["no_positive_match"] += 1
            continue
        kept_event_ids.add(eid)
        rows.extend(pool_to_pointwise_rows(pool, rec, args.filter_preset))

    train_path = output_dir / f"{args.split}_pointwise.jsonl"
    write_jsonl(train_path, rows)
    schema = {
        "feature_names": DEFAULT_FEATURE_NAMES,
        "label_name": "is_oracle_selected",
        "pool_mode": args.pool_mode,
        "filter_preset": args.filter_preset,
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
        "split": args.split,
        "output_rows": len(rows),
        "skipped": skipped,
        "feature_names": DEFAULT_FEATURE_NAMES,
        "score_weights": {
            "alpha_dense": finite_or_zero(alpha_dense),
            "alpha_lexical": finite_or_zero(alpha_lexical),
            "alpha_bm25": finite_or_zero(alpha_bm25),
        },
        "notes": [
            "Rows are candidate-level examples; event_id must be used for grouped splits.",
            "If cache_resolution.mode is fallback or explicit to a non-original cache, candidate pools are reconstructed by text-matching oracle positives and filling negatives by hybrid rank.",
        ],
    })
    write_json(output_dir / "filter_report.json", report)

    print(f"Wrote {len(rows)} rows to {train_path}")
    print(f"Kept {len(kept_event_ids)} claims; positive match rate={report['positive_text_match']['rate']:.4f}")
    print(f"Filter report: {output_dir / 'filter_report.json'}")


if __name__ == "__main__":
    main()
