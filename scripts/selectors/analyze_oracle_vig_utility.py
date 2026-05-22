#!/usr/bin/env python3
"""Analyze verifier information-gain cache with interpretable feature groups."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.selectors.stage2_oracle import write_json


FEATURE_GROUPS: dict[str, list[str]] = {
    "retrieval": [
        "hybrid_rank",
        "hybrid_reciprocal_rank",
        "dense_score",
        "lexical_score",
        "bm25_score",
        "hybrid_score",
    ],
    "text_overlap": [
        "candidate_token_len",
        "candidate_char_len",
        "candidate_has_number",
        "claim_candidate_token_jaccard",
        "prefix_candidate_max_jaccard",
        "prefix_candidate_mean_jaccard",
        "prefix_token_len_total",
    ],
    "prefix_state": [
        "step",
        "prefix_size",
        "base_gold_logprob",
        "base_best_wrong_logprob",
        "base_margin",
        "base_pred_is_gold",
    ],
    "single_verifier": [
        "single_gold_logprob",
        "single_best_wrong_logprob",
        "single_margin",
        "single_pred_is_gold",
    ],
}

BASELINE_SCORES: dict[str, str] = {
    "minus_hybrid_rank": "minus_hybrid_rank",
    "hybrid_score": "hybrid_score",
    "single_margin": "single_margin",
    "single_gold_logprob": "single_gold_logprob",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run interpretable feature-group probes on oracle VIG rows."
    )
    p.add_argument("--vig-cache", nargs="+", required=True)
    p.add_argument("--final-counterfactuals", nargs="*", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--top-examples", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.vig_cache)
    if not rows:
        raise ValueError("No VIG rows found.")
    final_rows = _load_rows(args.final_counterfactuals or []) if args.final_counterfactuals else []
    rows = _augment_rows(rows)
    event_ids = sorted({str(row["event_id"]) for row in rows})
    train_events, test_events = _split_events(event_ids, test_fraction=float(args.test_fraction), seed=int(args.seed))
    train_rows = [row for row in rows if str(row["event_id"]) in train_events]
    test_rows = [row for row in rows if str(row["event_id"]) in test_events]
    if not train_rows or not test_rows:
        raise ValueError("Train/test split is empty; adjust --test-fraction or cache size.")

    feature_sets = _feature_sets()
    group_results = []
    models: dict[str, dict[str, Any]] = {}
    for name, features in feature_sets.items():
        result, model = _fit_and_eval_feature_set(
            name,
            features,
            train_rows=train_rows,
            test_rows=test_rows,
            ridge_alpha=float(args.ridge_alpha),
        )
        group_results.append(result)
        models[name] = model

    baseline_results = [
        _eval_score_baseline(name, column, test_rows)
        for name, column in BASELINE_SCORES.items()
    ]
    true_delta = _eval_score_baseline("true_delta_margin_oracle_probe", "delta_margin", test_rows)
    train_true_delta = _eval_score_baseline("true_delta_margin_oracle_probe_train", "delta_margin", train_rows)
    all_model = models.get("all")
    permutation = _permutation_importance(
        all_model,
        test_rows=test_rows,
        groups=FEATURE_GROUPS,
        seed=int(args.seed),
    ) if all_model else []
    coefficients = _coefficient_importance(all_model, limit=30) if all_model else []

    analysis = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "vig_cache": [str(path) for path in args.vig_cache],
        "final_counterfactuals": [str(path) for path in (args.final_counterfactuals or [])],
        "split": str(args.split),
        "n_rows": int(len(rows)),
        "n_final_counterfactual_rows": int(len(final_rows)),
        "n_events": int(len(event_ids)),
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_train_events": int(len(train_events)),
        "n_test_events": int(len(test_events)),
        "ridge_alpha": float(args.ridge_alpha),
        "test_fraction": float(args.test_fraction),
        "oracle_self_check": {
            "test": true_delta,
            "train": train_true_delta,
            "interpretation": (
                "If true delta_margin cannot rank the saved oracle target highly, "
                "the cache scorer likely differs from the oracle scorer or prompt."
            ),
        },
        "target_delta_summary": _target_delta_summary(rows),
        "step_summary": _step_summary(rows),
        "final_counterfactual_summary": _final_counterfactual_summary(final_rows),
        "feature_group_results": group_results,
        "score_baselines": baseline_results,
        "permutation_importance": permutation,
        "top_coefficients": coefficients,
        "decision": _decision_payload(true_delta, group_results, baseline_results),
        "examples": _example_payload(rows, limit=int(args.top_examples)),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    write_json(out_dir / "vig_utility_analysis.json", analysis)
    _write_markdown(out_dir / "analysis.md", analysis)
    print(f"Wrote VIG utility analysis: {out_dir / 'vig_utility_analysis.json'}")
    print(
        "Decision={decision}; true_delta_top1={top1:.4f}; all_group_r2={r2:.4f}".format(
            decision=analysis["decision"]["decision"],
            top1=float(true_delta.get("step_top1_match", math.nan)),
            r2=float(_find_result(group_results, "all").get("r2", math.nan)),
        )
    )


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        expanded = [Path(p) for p in sorted(glob.glob(raw_path))] if any(ch in raw_path for ch in "*?[]") else [Path(raw_path)]
        for path in expanded:
            if not path.exists():
                raise FileNotFoundError(f"Input JSONL not found: {path}")
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _augment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    single: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if int(row.get("step", -1)) == 0:
            key = (str(row.get("event_id")), int(row.get("candidate_idx", -1)))
            single[key] = {
                "single_gold_logprob": _as_float(row.get("after_gold_logprob")),
                "single_best_wrong_logprob": _as_float(row.get("after_best_wrong_logprob")),
                "single_margin": _as_float(row.get("after_margin")),
                "single_pred_is_gold": float(bool(row.get("after_pred_is_gold"))),
            }
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        key = (str(row.get("event_id")), int(row.get("candidate_idx", -1)))
        item.update(single.get(key, {}))
        rank = _as_float(item.get("hybrid_rank"))
        item["minus_hybrid_rank"] = -rank if not math.isnan(rank) else math.nan
        item["hybrid_reciprocal_rank"] = 1.0 / (rank + 1.0) if not math.isnan(rank) and rank >= 0 else math.nan
        item["candidate_has_number"] = float(bool(item.get("candidate_has_number")))
        item["base_pred_is_gold"] = float(bool(item.get("base_pred_is_gold")))
        item["after_pred_is_gold"] = float(bool(item.get("after_pred_is_gold")))
        out.append(item)
    return out


def _split_events(event_ids: list[str], *, test_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    scored = []
    for event_id in event_ids:
        digest = hashlib.sha1(f"{seed}:{event_id}".encode("utf-8")).hexdigest()
        scored.append((int(digest, 16), event_id))
    scored.sort()
    n_test = max(1, int(round(len(event_ids) * float(test_fraction))))
    n_test = min(max(n_test, 1), max(len(event_ids) - 1, 1))
    test = {event_id for _, event_id in scored[:n_test]}
    train = set(event_ids) - test
    return train, test


def _feature_sets() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for group, features in FEATURE_GROUPS.items():
        sets[f"group:{group}"] = list(features)
    retrieval_plus_single = FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["single_verifier"]
    sets["retrieval+single_verifier"] = retrieval_plus_single
    all_features: list[str] = []
    for features in FEATURE_GROUPS.values():
        all_features.extend(features)
    sets["all"] = all_features
    return sets


def _fit_and_eval_feature_set(
    name: str,
    features: list[str],
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    ridge_alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train = _matrix(train_rows, features)
    y_train = _target(train_rows, "delta_margin")
    x_test = _matrix(test_rows, features)
    y_test = _target(test_rows, "delta_margin")
    x_train, x_test, means, stds = _standardize(x_train, x_test)
    coef = _fit_ridge(x_train, y_train, alpha=float(ridge_alpha))
    pred_train = _predict_ridge(x_train, coef)
    pred_test = _predict_ridge(x_test, coef)
    result = {
        "name": name,
        "features": features,
        "n_features": int(len(features)),
        "train_r2": _r2(y_train, pred_train),
        "r2": _r2(y_test, pred_test),
        "rmse": _rmse(y_test, pred_test),
        "spearman": _spearman(y_test, pred_test),
        "target_auroc": _roc_auc_score(_target(test_rows, "target"), pred_test),
        "step_top1_match": _step_top1_match(test_rows, pred_test),
    }
    model = {
        "name": name,
        "features": features,
        "means": means,
        "stds": stds,
        "coef": coef,
        "test_base": result,
    }
    return result, model


def _eval_score_baseline(name: str, column: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = _target(rows, column)
    y_delta = _target(rows, "delta_margin")
    y_target = _target(rows, "target")
    return {
        "name": name,
        "score_column": column,
        "r2_against_delta": _r2(y_delta, scores),
        "spearman_against_delta": _spearman(y_delta, scores),
        "target_auroc": _roc_auc_score(y_target, scores),
        "step_top1_match": _step_top1_match(rows, scores),
    }


def _permutation_importance(
    model: dict[str, Any],
    *,
    test_rows: list[dict[str, Any]],
    groups: dict[str, list[str]],
    seed: int,
) -> list[dict[str, Any]]:
    if not model:
        return []
    features = list(model["features"])
    x = _matrix(test_rows, features)
    y = _target(test_rows, "delta_margin")
    x_std = (x - np.asarray(model["means"])) / np.asarray(model["stds"])
    base_pred = _predict_ridge(x_std, np.asarray(model["coef"]))
    base_r2 = _r2(y, base_pred)
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    feature_to_pos = {feature: idx for idx, feature in enumerate(features)}
    for group, group_features in groups.items():
        positions = [feature_to_pos[f] for f in group_features if f in feature_to_pos]
        if not positions:
            continue
        x_perm = np.array(x_std, copy=True)
        for pos in positions:
            x_perm[:, pos] = rng.permutation(x_perm[:, pos])
        pred = _predict_ridge(x_perm, np.asarray(model["coef"]))
        perm_r2 = _r2(y, pred)
        out.append(
            {
                "group": group,
                "base_r2": base_r2,
                "permuted_r2": perm_r2,
                "r2_drop": base_r2 - perm_r2,
            }
        )
    return sorted(out, key=lambda row: float(row["r2_drop"]), reverse=True)


def _coefficient_importance(model: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    if not model:
        return []
    features = list(model["features"])
    coef = np.asarray(model["coef"], dtype=np.float64)[1:]
    rows = [
        {
            "feature": feature,
            "coefficient": float(weight),
            "abs_coefficient": float(abs(weight)),
            "group": _feature_group(feature),
        }
        for feature, weight in zip(features, coef)
    ]
    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)[: int(limit)]


def _matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    mat = np.empty((len(rows), len(features)), dtype=np.float64)
    for r_idx, row in enumerate(rows):
        for c_idx, feature in enumerate(features):
            mat[r_idx, c_idx] = _as_float(row.get(feature))
    if mat.size:
        col_means = np.nanmean(mat, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(col_means, inds[1])
    return mat


def _target(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if key == "target":
        return np.asarray([1.0 if bool(row.get("target")) else 0.0 for row in rows], dtype=np.float64)
    return np.asarray([_as_float(row.get(key)) for row in rows], dtype=np.float64)


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(x_train, axis=0) if x_train.size else np.zeros((x_train.shape[1],), dtype=np.float64)
    stds = np.std(x_train, axis=0) if x_train.size else np.ones((x_train.shape[1],), dtype=np.float64)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (x_train - means) / stds, (x_test - means) / stds, means, stds


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y


def _predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ coef


def _target_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_deltas = [_as_float(row.get("delta_margin")) for row in rows if bool(row.get("target"))]
    non_target_deltas = [_as_float(row.get("delta_margin")) for row in rows if not bool(row.get("target"))]
    target_gold_deltas = [_as_float(row.get("delta_gold_logprob")) for row in rows if bool(row.get("target"))]
    non_target_gold_deltas = [_as_float(row.get("delta_gold_logprob")) for row in rows if not bool(row.get("target"))]
    target_wrong_deltas = [_as_float(row.get("delta_best_wrong_logprob")) for row in rows if bool(row.get("target"))]
    non_target_wrong_deltas = [_as_float(row.get("delta_best_wrong_logprob")) for row in rows if not bool(row.get("target"))]
    selected_harmful = [row for row in rows if bool(row.get("target")) and _as_float(row.get("delta_margin")) < 0.0]
    non_target_positive = [row for row in rows if not bool(row.get("target")) and _as_float(row.get("delta_margin")) > 0.0]
    return {
        "target_delta_margin": _numeric_summary(target_deltas),
        "non_target_delta_margin": _numeric_summary(non_target_deltas),
        "target_minus_non_target_mean": _safe_mean(target_deltas) - _safe_mean(non_target_deltas),
        "target_delta_gold_logprob": _numeric_summary(target_gold_deltas),
        "non_target_delta_gold_logprob": _numeric_summary(non_target_gold_deltas),
        "target_delta_best_wrong_logprob": _numeric_summary(target_wrong_deltas),
        "non_target_delta_best_wrong_logprob": _numeric_summary(non_target_wrong_deltas),
        "target_gold_delta_minus_non_target_mean": _safe_mean(target_gold_deltas) - _safe_mean(non_target_gold_deltas),
        "target_wrong_delta_minus_non_target_mean": _safe_mean(target_wrong_deltas) - _safe_mean(non_target_wrong_deltas),
        "target_harmful_rate": len(selected_harmful) / max(sum(1 for row in rows if bool(row.get("target"))), 1),
        "non_target_positive_rate": len(non_target_positive) / max(sum(1 for row in rows if not bool(row.get("target"))), 1),
    }


def _step_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row.get("step", -1))].append(row)
    out = []
    for step, step_rows in sorted(by_step.items()):
        out.append(
            {
                "step": int(step),
                "n_rows": int(len(step_rows)),
                "true_delta_target_auroc": _roc_auc_score(_target(step_rows, "target"), _target(step_rows, "delta_margin")),
                "true_delta_step_top1_match": _step_top1_match(step_rows, _target(step_rows, "delta_margin")),
                "target_delta_mean": _safe_mean([_as_float(row.get("delta_margin")) for row in step_rows if bool(row.get("target"))]),
                "non_target_delta_mean": _safe_mean([_as_float(row.get("delta_margin")) for row in step_rows if not bool(row.get("target"))]),
            }
        )
    return out


def _final_counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0}
    removals = [row for row in rows if row.get("counterfactual_type") == "remove_selected"]
    replacements = [row for row in rows if row.get("counterfactual_type") == "replace_selected"]
    removal_contrib = [_as_float(row.get("final_contribution_delta_margin")) for row in removals]
    harmful_removals = [row for row in removals if _as_float(row.get("final_contribution_delta_margin")) < 0.0]

    by_selected: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in replacements:
        key = (
            str(row.get("event_id")),
            int(row.get("selected_step", -1)),
            int(row.get("selected_idx", -1)),
        )
        by_selected[key].append(row)
    best_replacements: list[dict[str, Any]] = []
    for key, items in by_selected.items():
        best = max(items, key=lambda row: _as_float(row.get("replacement_delta_margin")))
        best_replacements.append(
            {
                "event_id": key[0],
                "selected_step": key[1],
                "selected_idx": key[2],
                "best_replacement_candidate_idx": int(best.get("replacement_candidate_idx", -1)),
                "best_replacement_delta_margin": _as_float(best.get("replacement_delta_margin")),
                "best_replacement_improves_final_margin": bool(best.get("replacement_improves_final_margin")),
                "selected_text_preview": best.get("selected_text_preview"),
                "replacement_text_preview": best.get("replacement_text_preview"),
            }
        )
    best_delta = [_as_float(row.get("best_replacement_delta_margin")) for row in best_replacements]
    replaceable = [row for row in best_replacements if _as_float(row.get("best_replacement_delta_margin")) > 0.0]

    removal_by_step: dict[int, list[float]] = defaultdict(list)
    for row in removals:
        removal_by_step[int(row.get("selected_step", -1))].append(_as_float(row.get("final_contribution_delta_margin")))
    replacement_by_step: dict[int, list[float]] = defaultdict(list)
    for row in best_replacements:
        replacement_by_step[int(row.get("selected_step", -1))].append(_as_float(row.get("best_replacement_delta_margin")))

    return {
        "n_rows": int(len(rows)),
        "n_remove_rows": int(len(removals)),
        "n_replace_rows": int(len(replacements)),
        "removal_contribution_delta_margin": _numeric_summary(removal_contrib),
        "selected_harmful_final_rate": float(len(harmful_removals) / max(len(removals), 1)),
        "best_replacement_delta_margin": _numeric_summary(best_delta),
        "selected_replaceable_rate": float(len(replaceable) / max(len(best_replacements), 1)),
        "replacement_row_improves_rate": float(
            sum(1 for row in replacements if bool(row.get("replacement_improves_final_margin"))) / max(len(replacements), 1)
        ),
        "removal_by_step": [
            {
                "step": int(step),
                "contribution_mean": _safe_mean(values),
                "harmful_rate": float(sum(1 for value in values if value < 0.0) / max(len(values), 1)),
            }
            for step, values in sorted(removal_by_step.items())
        ],
        "best_replacement_by_step": [
            {
                "step": int(step),
                "best_replacement_delta_mean": _safe_mean(values),
                "replaceable_rate": float(sum(1 for value in values if value > 0.0) / max(len(values), 1)),
            }
            for step, values in sorted(replacement_by_step.items())
        ],
        "examples": {
            "most_harmful_selected": [
                _final_example(row)
                for row in sorted(removals, key=lambda row: _as_float(row.get("final_contribution_delta_margin")))[:10]
            ],
            "best_replacements": [
                row for row in sorted(best_replacements, key=lambda row: _as_float(row.get("best_replacement_delta_margin")), reverse=True)[:10]
            ],
        },
    }


def _final_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "selected_step": row.get("selected_step"),
        "selected_idx": row.get("selected_idx"),
        "final_contribution_delta_margin": row.get("final_contribution_delta_margin"),
        "base_final_margin": row.get("base_final_margin"),
        "counterfactual_margin": row.get("counterfactual_margin"),
        "selected_text_preview": row.get("selected_text_preview"),
    }


def _decision_payload(
    true_delta: dict[str, Any],
    group_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> dict[str, Any]:
    all_result = _find_result(group_results, "all")
    hybrid = _find_result(baseline_results, "minus_hybrid_rank")
    cache_ok = float(true_delta.get("step_top1_match", 0.0)) >= 0.90
    interpretable_useful = (
        float(all_result.get("step_top1_match", 0.0)) >= float(hybrid.get("step_top1_match", 0.0)) + 0.03
        and float(all_result.get("target_auroc", 0.0)) >= 0.60
    )
    if not cache_ok:
        decision = "fix_vig_cache_or_prompt_mismatch"
    elif interpretable_useful:
        decision = "go_utility_feature_distillation"
    else:
        decision = "inspect_or_add_features_before_distillation"
    return {
        "decision": decision,
        "cache_matches_oracle_by_true_delta_top1": bool(cache_ok),
        "interpretable_features_useful": bool(interpretable_useful),
        "true_delta_step_top1_threshold": 0.90,
        "feature_top1_lift_threshold_pp": 3.0,
        "feature_target_auroc_threshold": 0.60,
        "true_delta_step_top1_match": true_delta.get("step_top1_match"),
        "all_feature_step_top1_match": all_result.get("step_top1_match"),
        "hybrid_rank_step_top1_match": hybrid.get("step_top1_match"),
        "all_feature_target_auroc": all_result.get("target_auroc"),
    }


def _example_payload(rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    targets = [row for row in rows if bool(row.get("target"))]
    high = sorted(targets, key=lambda row: _as_float(row.get("delta_margin")), reverse=True)[:limit]
    harmful = sorted(targets, key=lambda row: _as_float(row.get("delta_margin")))[:limit]
    not_rank1 = [row for row in targets if int(row.get("target_rank_by_delta_margin", 1)) != 1]
    not_rank1 = sorted(not_rank1, key=lambda row: int(row.get("target_rank_by_delta_margin", 999)))[:limit]
    return {
        "high_positive_target_delta": [_example_row(row) for row in high],
        "harmful_or_low_target_delta": [_example_row(row) for row in harmful],
        "target_not_rank1_by_true_delta": [_example_row(row) for row in not_rank1],
    }


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "step": row.get("step"),
        "candidate_idx": row.get("candidate_idx"),
        "delta_margin": row.get("delta_margin"),
        "target_rank_by_delta_margin": row.get("target_rank_by_delta_margin"),
        "base_margin": row.get("base_margin"),
        "after_margin": row.get("after_margin"),
        "hybrid_rank": row.get("hybrid_rank"),
        "single_margin": row.get("single_margin"),
        "candidate_text_preview": row.get("candidate_text_preview"),
    }


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    decision = analysis["decision"]
    all_result = _find_result(analysis["feature_group_results"], "all")
    true_delta = analysis["oracle_self_check"]["test"]
    lines = [
        "# Oracle VIG Utility Analysis",
        "",
        f"- decision: `{decision['decision']}`",
        f"- rows: {analysis['n_rows']}",
        f"- final_counterfactual_rows: {analysis['n_final_counterfactual_rows']}",
        f"- events: {analysis['n_events']}",
        f"- train/test events: {analysis['n_train_events']} / {analysis['n_test_events']}",
        "",
        "## Oracle Self Check",
        "",
        "| score | target AUROC | step top1 match |",
        "|---|---:|---:|",
        f"| true delta_margin | {float(true_delta['target_auroc']):.4f} | {float(true_delta['step_top1_match']):.4f} |",
        "",
        "## Delta Decomposition",
        "",
        "| metric | target mean | non-target mean | target - non-target |",
        "|---|---:|---:|---:|",
    ]
    target_summary = analysis.get("target_delta_summary") or {}
    for label, target_key, non_target_key, delta_key in [
        ("delta_margin", "target_delta_margin", "non_target_delta_margin", "target_minus_non_target_mean"),
        ("delta_gold_logprob", "target_delta_gold_logprob", "non_target_delta_gold_logprob", "target_gold_delta_minus_non_target_mean"),
        ("delta_best_wrong_logprob", "target_delta_best_wrong_logprob", "non_target_delta_best_wrong_logprob", "target_wrong_delta_minus_non_target_mean"),
    ]:
        target_stats = target_summary.get(target_key) or {}
        non_target_stats = target_summary.get(non_target_key) or {}
        lines.append(
            f"| {label} | {float(target_stats.get('mean', math.nan)):.4f} | "
            f"{float(non_target_stats.get('mean', math.nan)):.4f} | "
            f"{float(target_summary.get(delta_key, math.nan)):.4f} |"
        )
    lines.extend(
        [
            "",
        "## Feature Group Probe",
        "",
        "| feature set | R2 | RMSE | target AUROC | step top1 match |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["feature_group_results"]:
        lines.append(
            f"| {row['name']} | {float(row['r2']):.4f} | {float(row['rmse']):.4f} | "
            f"{float(row['target_auroc']):.4f} | {float(row['step_top1_match']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Score Baselines",
            "",
            "| score | target AUROC | step top1 match |",
            "|---|---:|---:|",
        ]
    )
    for row in analysis["score_baselines"]:
        lines.append(
            f"| {row['name']} | {float(row['target_auroc']):.4f} | {float(row['step_top1_match']):.4f} |"
        )
    final_summary = analysis.get("final_counterfactual_summary") or {}
    if int(final_summary.get("n_rows", 0) or 0):
        removal = final_summary.get("removal_contribution_delta_margin") or {}
        replacement = final_summary.get("best_replacement_delta_margin") or {}
        lines.extend(
            [
                "",
                "## Final-Set Counterfactuals",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| removal contribution mean | {float(removal.get('mean', math.nan)):.4f} |",
                f"| selected harmful final rate | {float(final_summary.get('selected_harmful_final_rate', math.nan)):.4f} |",
                f"| best replacement delta mean | {float(replacement.get('mean', math.nan)):.4f} |",
                f"| selected replaceable rate | {float(final_summary.get('selected_replaceable_rate', math.nan)):.4f} |",
                f"| replacement row improves rate | {float(final_summary.get('replacement_row_improves_rate', math.nan)):.4f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Group Importance",
            "",
            "| group | R2 drop when permuted |",
            "|---|---:|",
        ]
    )
    for row in analysis["permutation_importance"]:
        lines.append(f"| {row['group']} | {float(row['r2_drop']):.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If `true delta_margin` does not rank the saved oracle target highly, rerun the cache with the exact oracle verifier model, LoRA adapter, prompt config, and max length.",
            "If the all-feature probe beats retrieval baselines, the strongest feature groups are candidates for utility distillation.",
            f"Current all-feature R2 is `{float(all_result.get('r2', math.nan)):.4f}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_group(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "unknown"


def _find_result(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("name") == name:
            return row
    return {}


def _step_top1_match(rows: list[dict[str, Any]], scores: np.ndarray) -> float:
    grouped: dict[tuple[str, int], list[tuple[float, bool]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        key = (str(row.get("event_id")), int(row.get("step", -1)))
        if not math.isnan(float(score)):
            grouped[key].append((float(score), bool(row.get("target"))))
    hits = 0
    total = 0
    for items in grouped.values():
        if not items:
            continue
        best = max(items, key=lambda item: item[0])
        hits += int(best[1])
        total += 1
    return float(hits / total) if total else math.nan


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    valid = ~np.isnan(y) & ~np.isnan(pred)
    y = y[valid]
    pred = pred[valid]
    if y.size == 0:
        return math.nan
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return math.nan
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    valid = ~np.isnan(y) & ~np.isnan(pred)
    if not int(valid.sum()):
        return math.nan
    return float(np.sqrt(np.mean((y[valid] - pred[valid]) ** 2)))


def _spearman(y: np.ndarray, pred: np.ndarray) -> float:
    valid = ~np.isnan(y) & ~np.isnan(pred)
    if int(valid.sum()) < 2:
        return math.nan
    return float(np.corrcoef(_rankdata(y[valid]), _rankdata(pred[valid]))[0, 1])


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray([v for v in values if not math.isnan(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "max": math.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if not math.isnan(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else math.nan


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) and not math.isinf(number) else math.nan


if __name__ == "__main__":
    main()
