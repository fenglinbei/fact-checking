"""Selection-only evaluation for the pointwise oracle evidence selector."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from fact_checking.oracle_pointwise import (
    average_precision,
    bce_loss,
    build_candidate_pool,
    claim_selection_metrics,
    feature_matrix,
    labels_array,
    load_build_config,
    load_chunk_samples_by_event,
    pool_to_pointwise_rows,
    read_jsonl,
    resolve_chunk_cache_path,
    roc_auc,
    selected_evidence_rows,
    sigmoid,
    supervision_policy_for_record,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pointwise oracle selector against oracle sets.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--oracle-results", required=True)
    p.add_argument("--config", default="configs/experiment/b3_mmr_topk_sweep_1024.yaml")
    p.add_argument("--config-overrides", default=None)
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--chunk-mmr-cache", default=None)
    p.add_argument("--chunk-mmr-cache-root", default="outputs/cache/chunk_mmr")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--filter-preset", default="v1a", choices=["v1a", "v1b", "all"])
    p.add_argument("--pool-mode", default="oracle_n_top_hybrid_with_positives")
    p.add_argument("--fallback-pool-size", type=int, default=15)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = _load_model(Path(args.model_dir) / "model.npz")
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
    skipped = {
        "missing_cache_sample": 0,
        "filter": 0,
        "no_candidates": 0,
        "no_positive_match": 0,
    }
    for rec in tqdm(
        oracle_records,
        desc="build eval rows",
        unit="claim",
        dynamic_ncols=True,
        disable=args.no_progress,
    ):
        eid = str(rec.get("event_id", ""))
        sample = samples_by_event.get(eid)
        if sample is None:
            skipped["missing_cache_sample"] += 1
            continue
        policy = supervision_policy_for_record(rec, args.filter_preset)
        if not policy["keep"]:
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
        if pool.matched_positive_count <= 0:
            skipped["no_positive_match"] += 1
            continue
        rows.extend(
            pool_to_pointwise_rows(
                pool,
                rec,
                str(policy["bucket"]),
                supervision_weight=float(policy["supervision_weight"]),
                anchor_source=str(policy.get("anchor_source", "")),
            )
        )

    if not rows:
        raise ValueError("No evaluation rows were produced.")

    x_raw = feature_matrix(rows, model["feature_names"])
    x = (x_raw - model["feature_mean"]) / model["feature_std"]
    y = labels_array(rows)
    model_scores = sigmoid(x @ model["weights"] + model["bias"])
    hybrid_scores = np.array([float(row["features"].get("hybrid_score", 0.0)) for row in rows], dtype=np.float32)

    metrics = {
        "model_dir": args.model_dir,
        "oracle_results": args.oracle_results,
        "split": args.split,
        "filter_preset": args.filter_preset,
        "cache_path": str(cache_path),
        "cache_resolution": cache_resolution,
        "n_rows": len(rows),
        "n_claims": len({row["event_id"] for row in rows}),
        "skipped": skipped,
        "row_metrics": {
            "model": {
                "loss": bce_loss(y, model_scores),
                "auprc": average_precision(y, model_scores),
                "auroc": roc_auc(y, model_scores),
                "positive_rate": float(y.mean()) if y.size else 0.0,
            },
            "hybrid_score": {
                "auprc": average_precision(y, hybrid_scores),
                "auroc": roc_auc(y, hybrid_scores),
            },
        },
        "selection_metrics": {
            "model": claim_selection_metrics(rows, model_scores, top_k=args.top_k, score_name="model"),
            "hybrid_score": claim_selection_metrics(rows, hybrid_scores, top_k=args.top_k, score_name="hybrid_score"),
        },
    }
    write_json(output_dir / "selection_metrics.json", metrics)
    write_jsonl(output_dir / "selected_evidence.jsonl", selected_evidence_rows(rows, model_scores, top_k=args.top_k))

    pred_rows = []
    for row, score, hybrid in zip(rows, model_scores, hybrid_scores):
        item = dict(row)
        item["model_score"] = float(score)
        item["hybrid_score"] = float(hybrid)
        pred_rows.append(item)
    write_jsonl(output_dir / "candidate_scores.jsonl", pred_rows)

    model_sel = metrics["selection_metrics"]["model"]
    hybrid_sel = metrics["selection_metrics"]["hybrid_score"]
    print(f"Wrote selection metrics: {output_dir / 'selection_metrics.json'}")
    print(
        "Model Jaccard@{k}={mj:.4f}, Recall@{k}={mr:.4f}; "
        "hybrid Jaccard@{k}={hj:.4f}, Recall@{k}={hr:.4f}".format(
            k=args.top_k,
            mj=model_sel["jaccard_at_k"],
            mr=model_sel["recall_at_k"],
            hj=hybrid_sel["jaccard_at_k"],
            hr=hybrid_sel["recall_at_k"],
        )
    )


def _load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "weights": data["weights"].astype(np.float32),
        "bias": float(data["bias"][0]),
        "feature_mean": data["feature_mean"].astype(np.float32),
        "feature_std": data["feature_std"].astype(np.float32),
        "feature_names": [str(x) for x in data["feature_names"].tolist()],
    }


if __name__ == "__main__":
    main()
