#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.selectors.oracle_likelihood_constrained_selector import (
    FEATURE_SET_ALL,
    FEATURE_SET_CHOICES,
    OBJECTIVE_CHOICES,
    OBJECTIVE_POINTWISE,
    LogisticParams,
    attach_scores_to_candidate_rows,
    build_feature_rows,
    candidate_level_metrics,
    cross_fit_score_rows,
    feature_importance_rows,
    feature_names_for_set,
    filter_feature_rows,
    save_logistic_model,
    train_heldout_score_rows,
)
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_BUCKET_FILE = "outputs/selectors/count_amplified_stance_bucket_selector/v0_2_val/candidate_stance_buckets_v02_n7_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/oracle_likelihood_constrained_selector/v0_3_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and score oracle-likelihood constrained selector features.")
    p.add_argument("--candidate-stance-buckets", default=DEFAULT_BUCKET_FILE)
    p.add_argument("--train-candidate-stance-buckets", default=None)
    p.add_argument("--eval-candidate-stance-buckets", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--cross-fit-folds", type=int, default=5)
    p.add_argument("--feature-set", default=FEATURE_SET_ALL, choices=FEATURE_SET_CHOICES)
    p.add_argument("--objective", default=OBJECTIVE_POINTWISE, choices=OBJECTIVE_CHOICES)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--dev-fraction", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = LogisticParams(
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        seed=int(args.seed),
        dev_fraction=float(args.dev_fraction),
    )

    if args.train_candidate_stance_buckets and args.eval_candidate_stance_buckets:
        mode = "heldout_train_eval"
        train_rows = _load_rows(args.train_candidate_stance_buckets, sample_limit=args.sample_limit)
        eval_rows = _load_rows(args.eval_candidate_stance_buckets, sample_limit=args.sample_limit)
        train_feature_rows_all, all_feature_names = build_feature_rows(train_rows)
        feature_names = feature_names_for_set(all_feature_names, str(args.feature_set))
        train_feature_rows = filter_feature_rows(train_feature_rows_all, feature_names)
        eval_feature_rows_all, _ = build_feature_rows(eval_rows, feature_names=all_feature_names)
        eval_feature_rows = filter_feature_rows(eval_feature_rows_all, feature_names)
        scored_feature_rows, model, train_metadata = train_heldout_score_rows(
            train_feature_rows,
            eval_feature_rows,
            feature_names,
            params=params,
            objective=str(args.objective),
        )
        save_logistic_model(
            out_dir / "oracle_likelihood_model.npz",
            model=model,
            metadata={
                "mode": mode,
                **train_metadata,
                "feature_set": str(args.feature_set),
                "objective": str(args.objective),
                "all_feature_names": all_feature_names,
                "feature_names": feature_names,
            },
        )
        models = [model]
        fold_records: list[dict[str, Any]] = [train_metadata]
        scored_rows = attach_scores_to_candidate_rows(eval_rows, scored_feature_rows)
        input_paths = {
            "train_candidate_stance_buckets": str(args.train_candidate_stance_buckets),
            "eval_candidate_stance_buckets": str(args.eval_candidate_stance_buckets),
        }
    else:
        mode = "event_level_cross_fit"
        rows = _load_rows(args.candidate_stance_buckets, sample_limit=args.sample_limit)
        feature_rows_all, all_feature_names = build_feature_rows(rows)
        feature_names = feature_names_for_set(all_feature_names, str(args.feature_set))
        feature_rows = filter_feature_rows(feature_rows_all, feature_names)
        scored_feature_rows, models, fold_records = cross_fit_score_rows(
            feature_rows,
            feature_names,
            folds=int(args.cross_fit_folds),
            params=params,
            objective=str(args.objective),
        )
        for model in models:
            fold = int(model.get("fold") or 0)
            save_logistic_model(
                out_dir / f"oracle_likelihood_model_fold{fold}.npz",
                model=model,
                metadata={
                    "mode": mode,
                    "fold": fold,
                    "n_train_events": len(model.get("train_event_ids") or []),
                    "n_dev_events": len(model.get("dev_event_ids") or []),
                    "n_heldout_events": len(model.get("heldout_event_ids") or []),
                    "feature_set": str(args.feature_set),
                    "objective": str(args.objective),
                    "all_feature_names": all_feature_names,
                    "feature_names": feature_names,
                },
            )
        scored_rows = attach_scores_to_candidate_rows(rows, scored_feature_rows)
        input_paths = {"candidate_stance_buckets": str(args.candidate_stance_buckets)}

    metrics = candidate_level_metrics(scored_feature_rows)
    feature_schema = {
        "feature_set": str(args.feature_set),
        "objective": str(args.objective),
        "all_feature_names": all_feature_names,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "forbidden_fields_note": "oracle labels, gold label, event id identity, raw text, candidate key, and candidate uid are metadata only, not model features.",
    }
    save_json(feature_schema, out_dir / "feature_schema.json")
    write_jsonl(scored_rows, out_dir / f"candidate_oracle_likelihood_scores_{args.split}.jsonl")
    save_json({"features": feature_importance_rows(models, feature_names)}, out_dir / "feature_importance.json")
    save_json({"candidate_level": metrics, "folds": fold_records}, out_dir / "calibration_metrics.json")
    save_json(
        _json_safe(
            {
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
                "mode": mode,
                "split": str(args.split),
                "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
                "cross_fit_folds": int(args.cross_fit_folds),
                "feature_set": str(args.feature_set),
                "objective": str(args.objective),
                "n_features": len(feature_names),
                "params": params.__dict__,
                "inputs": input_paths,
                "n_scored_events": len(scored_rows),
                "n_scored_candidates": len(scored_feature_rows),
                "outputs": {
                    "scored_candidates": str(out_dir / f"candidate_oracle_likelihood_scores_{args.split}.jsonl"),
                    "feature_schema": str(out_dir / "feature_schema.json"),
                    "feature_importance": str(out_dir / "feature_importance.json"),
                    "calibration_metrics": str(out_dir / "calibration_metrics.json"),
                },
                "candidate_level": metrics,
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        ),
        out_dir / "manifest.json",
    )
    print(f"Wrote oracle-likelihood scores under: {out_dir}")
    print(
        "AUROC={auroc:.4f} AUPRC={auprc:.4f} scored_events={events} scored_candidates={candidates}".format(
            auroc=float(metrics.get("auroc", 0.0)),
            auprc=float(metrics.get("auprc", 0.0)),
            events=len(scored_rows),
            candidates=len(scored_feature_rows),
        )
    )


def _load_rows(path: str, *, sample_limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if sample_limit is not None:
        rows = rows[: int(sample_limit)]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
