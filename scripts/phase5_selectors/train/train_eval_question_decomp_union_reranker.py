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
    pairwise_metrics_for_rows,
    save_pairwise_logistic_model,
    score_rows,
    source_composition,
    train_pairwise_logistic,
)
from fact_checking.selectors.question_decomp_retrieval import _compute_prediction_metrics, oracle_selected_texts_by_event
from fact_checking.selectors.stage2_oracle import read_jsonl, write_json, write_jsonl


DEFAULT_TRAIN_UNION_POOL = "outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl"
DEFAULT_VAL_UNION_POOL = "outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl"
DEFAULT_TRAIN_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
DEFAULT_VAL_ORACLE_RESULTS = "outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the lightweight union-pool reranker on train and apply/evaluate on val."
    )
    p.add_argument("--train-union-pool-jsonl", default=DEFAULT_TRAIN_UNION_POOL)
    p.add_argument("--val-union-pool-jsonl", default=DEFAULT_VAL_UNION_POOL)
    p.add_argument("--train-oracle-results", default=DEFAULT_TRAIN_ORACLE_RESULTS)
    p.add_argument("--val-oracle-results", default=DEFAULT_VAL_ORACLE_RESULTS)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-sample-limit", type=int, default=None)
    p.add_argument("--val-sample-limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument(
        "--use-val-for-early-stopping",
        action="store_true",
        help="Use val labels for early stopping. Default keeps val as a pure held-out evaluation set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_union_rows = _load_rows(args.train_union_pool_jsonl, sample_limit=args.train_sample_limit)
    val_union_rows = _load_rows(args.val_union_pool_jsonl, sample_limit=args.val_sample_limit)
    if not train_union_rows:
        raise ValueError("No train union rows loaded.")
    if not val_union_rows:
        raise ValueError("No val union rows loaded.")

    train_oracle_rows = _load_rows(args.train_oracle_results, sample_limit=args.train_sample_limit)
    val_oracle_rows = _load_rows(args.val_oracle_results, sample_limit=args.val_sample_limit)
    train_oracle_texts = oracle_selected_texts_by_event(train_oracle_rows)
    val_oracle_texts = oracle_selected_texts_by_event(val_oracle_rows)

    _assert_events_covered(train_union_rows, train_oracle_texts, split="train")
    _assert_events_covered(val_union_rows, val_oracle_texts, split="val")

    feature_names = default_feature_names()
    train_feature_rows = build_feature_rows(train_union_rows, oracle_texts=train_oracle_texts)
    val_feature_rows = build_feature_rows(val_union_rows, oracle_texts=val_oracle_texts)
    early_stop_rows = val_feature_rows if args.use_val_for_early_stopping else []

    params = PairwiseRerankerParams(
        top_k=int(args.top_k),
        val_fraction=0.0,
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
    )
    model = train_pairwise_logistic(train_feature_rows, early_stop_rows, feature_names, params=params)

    train_scores = score_rows(train_feature_rows, feature_names, model)
    val_scores = score_rows(val_feature_rows, feature_names, model)
    train_selected = _selection_bundle(train_feature_rows, train_scores, top_k=int(args.top_k))
    val_selected = _selection_bundle(val_feature_rows, val_scores, top_k=int(args.top_k))
    train_controls = _control_bundle(train_union_rows, top_k=int(args.top_k))
    val_controls = _control_bundle(val_union_rows, top_k=int(args.top_k))

    train_pairwise = pairwise_metrics_for_rows(train_feature_rows, feature_names, model)
    val_pairwise = pairwise_metrics_for_rows(val_feature_rows, feature_names, model)
    metrics = {
        "model_type": "numpy_pairwise_logistic",
        "training_mode": "train_on_train_eval_on_val",
        "early_stopping": "val" if args.use_val_for_early_stopping else "train",
        "n_train_events": len({str(row.get("event_id")) for row in train_feature_rows}),
        "n_val_events": len({str(row.get("event_id")) for row in val_feature_rows}),
        "n_train_feature_rows": len(train_feature_rows),
        "n_val_feature_rows": len(val_feature_rows),
        "train_positive_rate": _positive_rate(train_feature_rows),
        "val_positive_rate": _positive_rate(val_feature_rows),
        "params": params.__dict__,
        "pairwise": {
            "train": train_pairwise,
            "val": val_pairwise,
            "trainer_n_train_pairs": model["n_train_pairs"],
            "trainer_n_early_stop_pairs": model["n_val_pairs"],
            "trainer_train_pairwise_acc": model["train_pairwise_acc"],
            "trainer_early_stop_pairwise_acc": model["val_pairwise_acc"],
        },
        "train": _metrics_bundle(train_selected, train_controls, train_oracle_texts),
        "val": _metrics_bundle(val_selected, val_controls, val_oracle_texts),
        "history": model["history"],
    }

    write_jsonl(out_dir / "feature_rows_train.jsonl", _json_safe_rows(train_feature_rows))
    write_jsonl(out_dir / "feature_rows_val.jsonl", _json_safe_rows(val_feature_rows))
    write_json(out_dir / "feature_schema.json", {"feature_names": feature_names})
    write_jsonl(out_dir / "train_history.jsonl", _json_safe_rows(model["history"]))
    write_jsonl(out_dir / "reranker_predictions_train.jsonl", _prediction_rows(train_feature_rows, train_scores))
    write_jsonl(out_dir / "reranker_predictions_val.jsonl", _prediction_rows(val_feature_rows, val_scores))
    for name, rows in train_selected.items():
        write_jsonl(out_dir / f"{name}_train.jsonl", _json_safe_rows(rows))
    for name, rows in val_selected.items():
        write_jsonl(out_dir / f"{name}_val.jsonl", _json_safe_rows(rows))
    save_pairwise_logistic_model(
        out_dir / "reranker_model.npz",
        model=model,
        feature_names=feature_names,
        metadata={
            "model_type": "numpy_pairwise_logistic",
            "training_mode": "train_on_train_eval_on_val",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(out_dir / "feature_importance.json", {"features": feature_importance(feature_names, model["weights"])})
    write_json(out_dir / "reranker_metrics.json", _json_safe(metrics))
    write_json(
        out_dir / "metadata.json",
        _json_safe(
            {
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
                "train_union_pool_jsonl": str(args.train_union_pool_jsonl),
                "val_union_pool_jsonl": str(args.val_union_pool_jsonl),
                "train_oracle_results": str(args.train_oracle_results),
                "val_oracle_results": str(args.val_oracle_results),
                "output_dir": str(out_dir),
                "train_sample_limit": int(args.train_sample_limit) if args.train_sample_limit is not None else None,
                "val_sample_limit": int(args.val_sample_limit) if args.val_sample_limit is not None else None,
                "feature_names": feature_names,
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        ),
    )

    print(f"Wrote train/eval reranker outputs: {out_dir}")
    _print_summary(metrics)


def _load_rows(path: str, *, sample_limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if sample_limit is not None:
        rows = rows[: int(sample_limit)]
    return rows


def _assert_events_covered(rows: list[dict[str, Any]], oracle_texts: dict[str, set[str]], *, split: str) -> None:
    row_events = {str(row.get("event_id") or "") for row in rows}
    missing = sorted(event_id for event_id in row_events if event_id not in oracle_texts)
    if missing:
        raise ValueError(f"{split} union rows missing oracle labels for events: {missing[:5]}")


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
                pool_rows=[
                    {"event_id": row.get("event_id"), "pool": row.get("candidates") or [], "selected": row.get("candidates") or []}
                    for row in rows
                ],
                oracle_texts=oracle_texts,
                include_pool=True,
            )
        else:
            metrics[name] = evaluate_selected_rows(rows, oracle_texts=oracle_texts)
            metrics[name]["source_composition"] = source_composition(rows)
    return metrics


def _prediction_rows(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["model_score"] = float(score)
        predictions.append(item)
    return _json_safe_rows(predictions)


def _positive_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([int(row.get("label") or 0) for row in rows]))


def _print_summary(metrics: dict[str, Any]) -> None:
    val = metrics["val"]
    print(
        "Val learned recall={:.4f} jaccard={:.4f} top1={:.4f}".format(
            val["learned_top5"]["oracle_selected_recall@5"],
            val["learned_top5"]["jaccard@5"],
            val["learned_top5"]["top1_match"],
        )
    )
    print(
        "Val baseline_top2+learned recall={:.4f} jaccard={:.4f}; baseline_top5 recall={:.4f} jaccard={:.4f}".format(
            val["baseline_top2_plus_learned"]["oracle_selected_recall@5"],
            val["baseline_top2_plus_learned"]["jaccard@5"],
            val["baseline_top5"]["oracle_selected_recall@5"],
            val["baseline_top5"]["jaccard@5"],
        )
    )
    print(
        "Val pairwise_acc={:.4f} n_pairs={}".format(
            metrics["pairwise"]["val"]["pairwise_acc"],
            metrics["pairwise"]["val"]["n_pairs"],
        )
    )


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in rows]


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
