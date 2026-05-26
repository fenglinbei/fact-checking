#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.selectors.question_decomp_reranker import (
    PairwiseRerankerParams,
    build_feature_rows,
    build_selected_rows,
    default_feature_names,
    evaluate_selected_rows,
    feature_importance,
    score_rows,
    source_composition,
    split_event_ids,
    train_pairwise_logistic,
)
from fact_checking.selectors.question_decomp_retrieval import _compute_prediction_metrics, oracle_selected_texts_by_event
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json, write_jsonl


DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a lightweight feature reranker over baseline+QD union pools.")
    p.add_argument("--union-pool-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--oracle-results", default=None)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    union_rows = read_jsonl(args.union_pool_jsonl)
    if args.sample_limit is not None:
        union_rows = union_rows[: int(args.sample_limit)]
    if not union_rows:
        raise ValueError("No union pool rows loaded.")
    oracle_results = args.oracle_results or _default_oracle_results(str(args.split))
    oracle_rows = read_jsonl(oracle_results)
    if args.sample_limit is not None:
        oracle_rows = oracle_rows[: int(args.sample_limit)]
    oracle_texts = oracle_selected_texts_by_event(oracle_rows)

    feature_names = default_feature_names()
    feature_rows = build_feature_rows(union_rows, oracle_texts=oracle_texts)
    train_events, val_events = split_event_ids(feature_rows, val_fraction=float(args.val_fraction), seed=int(args.seed))
    train_rows = [row for row in feature_rows if str(row["event_id"]) in train_events]
    val_rows = [row for row in feature_rows if str(row["event_id"]) in val_events]
    params = PairwiseRerankerParams(
        top_k=int(args.top_k),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
    )
    model = train_pairwise_logistic(train_rows, val_rows, feature_names, params=params)
    all_scores = score_rows(feature_rows, feature_names, model)
    train_scores = score_rows(train_rows, feature_names, model)
    val_scores = score_rows(val_rows, feature_names, model)

    predictions = []
    for row, score in zip(feature_rows, all_scores):
        item = dict(row)
        item["model_score"] = float(score)
        predictions.append(item)

    all_selected = _selection_bundle(feature_rows, all_scores, top_k=int(args.top_k))
    train_selected = _selection_bundle(train_rows, train_scores, top_k=int(args.top_k))
    val_selected = _selection_bundle(val_rows, val_scores, top_k=int(args.top_k))
    controls_all = _control_bundle(union_rows, top_k=int(args.top_k))
    controls_train = {key: _filter_event_rows(rows, train_events) for key, rows in controls_all.items()}
    controls_val = {key: _filter_event_rows(rows, val_events) for key, rows in controls_all.items()}

    metrics = {
        "model_type": "numpy_pairwise_logistic",
        "n_events": len({str(row.get("event_id")) for row in feature_rows}),
        "n_feature_rows": len(feature_rows),
        "n_train_events": len(train_events),
        "n_val_events": len(val_events),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "positive_rate": float(np.mean([int(row.get("label") or 0) for row in feature_rows])),
        "params": params.__dict__,
        "pairwise": {
            "n_train_pairs": model["n_train_pairs"],
            "n_val_pairs": model["n_val_pairs"],
            "train_pairwise_acc": model["train_pairwise_acc"],
            "val_pairwise_acc": model["val_pairwise_acc"],
        },
        "train": _metrics_bundle(train_selected, controls_train, oracle_texts),
        "val": _metrics_bundle(val_selected, controls_val, oracle_texts),
        "all": _metrics_bundle(all_selected, controls_all, oracle_texts),
        "history": model["history"],
    }

    write_jsonl(out_dir / f"feature_rows_{args.split}.jsonl", feature_rows)
    write_json(out_dir / "feature_schema.json", {"feature_names": feature_names})
    write_jsonl(out_dir / f"reranker_predictions_{args.split}.jsonl", predictions)
    for name, rows in all_selected.items():
        write_jsonl(out_dir / f"{name}_{args.split}.jsonl", rows)
    write_json(out_dir / "train_event_ids.json", {"event_ids": sorted(train_events)})
    write_json(out_dir / "val_event_ids.json", {"event_ids": sorted(val_events)})
    np.savez(
        out_dir / "reranker_model.npz",
        weights=np.asarray(model["weights"], dtype=np.float32),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(model["feature_std"], dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
        metadata_json=np.array(json.dumps({"model_type": "numpy_pairwise_logistic"}, ensure_ascii=False), dtype=object),
    )
    write_json(out_dir / "feature_importance.json", {"features": feature_importance(feature_names, model["weights"])})
    write_json(out_dir / "reranker_metrics.json", _json_safe(metrics))
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "union_pool_jsonl": str(args.union_pool_jsonl),
        "oracle_results": str(oracle_results),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "feature_names": feature_names,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(out_dir / "metadata.json", _json_safe(manifest))

    print(f"Wrote reranker outputs: {out_dir}")
    print(
        "Val learned recall={:.4f} jaccard={:.4f}; top2+learned recall={:.4f} jaccard={:.4f}; baseline recall={:.4f}".format(
            metrics["val"]["learned_top5"]["oracle_selected_recall@5"],
            metrics["val"]["learned_top5"]["jaccard@5"],
            metrics["val"]["baseline_top2_plus_learned"]["oracle_selected_recall@5"],
            metrics["val"]["baseline_top2_plus_learned"]["jaccard@5"],
            metrics["val"]["baseline_top5"]["oracle_selected_recall@5"],
        )
    )


def _selection_bundle(rows: list[dict[str, Any]], scores: np.ndarray, *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "learned_top5": build_selected_rows(rows, scores, top_k=top_k, mode="learned_top5"),
        "baseline_top1_plus_learned": build_selected_rows(rows, scores, top_k=top_k, mode="baseline_top1_plus_learned", baseline_anchor_k=1),
        "baseline_top2_plus_learned": build_selected_rows(rows, scores, top_k=top_k, mode="baseline_top2_plus_learned", baseline_anchor_k=2),
        "baseline_top3_plus_learned": build_selected_rows(rows, scores, top_k=top_k, mode="baseline_top3_plus_learned", baseline_anchor_k=3),
    }


def _control_bundle(union_rows: list[dict[str, Any]], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    baseline_rows: list[dict[str, Any]] = []
    qd_rows: list[dict[str, Any]] = []
    union_pool_rows: list[dict[str, Any]] = []
    for row in union_rows:
        candidates = list(row.get("candidates") or [])
        baseline = [dict(candidate) for candidate in candidates if candidate.get("from_baseline")]
        baseline.sort(key=lambda c: int(c.get("baseline_rank") or 10**9))
        qd = [dict(candidate) for candidate in candidates if candidate.get("from_qd")]
        qd.sort(key=lambda c: int(c.get("qd_pool_rank") or 10**9))
        baseline_rows.append({"event_id": row.get("event_id"), "claim": row.get("claim"), "candidates": baseline[:top_k]})
        qd_rows.append({"event_id": row.get("event_id"), "claim": row.get("claim"), "candidates": qd[:top_k]})
        union_pool_rows.append({"event_id": row.get("event_id"), "claim": row.get("claim"), "candidates": candidates})
    return {"baseline_top5": baseline_rows, "qd_rrf_top5": qd_rows, "union_pool": union_pool_rows}


def _metrics_bundle(
    selections: dict[str, list[dict[str, Any]]],
    controls: dict[str, list[dict[str, Any]]],
    oracle_texts: dict[str, set[str]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, rows in {**controls, **selections}.items():
        if name == "union_pool":
            metrics[name] = _compute_prediction_metrics(
                pool_rows=[{"event_id": row.get("event_id"), "pool": row.get("candidates") or [], "selected": row.get("candidates") or []} for row in rows],
                oracle_texts=oracle_texts,
                include_pool=True,
            )
        else:
            metrics[name] = evaluate_selected_rows(rows, oracle_texts=oracle_texts)
            metrics[name]["source_composition"] = source_composition(rows)
    return metrics


def _filter_event_rows(rows: list[dict[str, Any]], event_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("event_id")) in event_ids]


def _default_oracle_results(split: str) -> str:
    if split == "train":
        return DEFAULT_TRAIN_ORACLE_RESULTS
    return DEFAULT_VAL_ORACLE_RESULTS


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
