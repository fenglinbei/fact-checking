"""Step 4: Evaluate the trained λ predictor.

Reports MAE, RMSE, correlation, and distribution analysis of oracle vs predicted λ.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/evaluate_predictor.py \
        --model outputs/learned_lambda/predictor.pt \
        --feature-stats outputs/learned_lambda/feature_stats.json \
        --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
        --experiment b3_mmr_topk_sweep_1024
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
import torch
from tqdm.auto import tqdm

from fact_checking.build.candidates import _load_pickle
from fact_checking.learned_lambda.cache_utils import (
    load_experiment_build_cfg,
    pick_retrieval_value,
    resolve_chunk_mmr_cache_path,
)
from fact_checking.learned_lambda.embedding_features import build_matched_chunk_embedding_arrays
from fact_checking.learned_lambda.predictor import load_predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate λ predictor.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--feature-stats", type=str, required=True)
    p.add_argument("--oracle-lambdas", type=str, required=True)
    p.add_argument("--chunk-mmr-cache", type=str, default=None)
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--candidate-top-k",
        type=int,
        default=None,
        help="Number of hybrid-ranked chunk candidates to feed the predictor. Omit to use the full chunk pool.",
    )
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument(
        "--fixed-lambda-grid",
        type=str,
        default="auto",
        help="Comma-separated fixed λ values for baseline search, or 'auto'.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def _log(message: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)
    else:
        print(message, flush=True)


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    if len(x) < 2 or x.std() <= 1e-8 or y.std() <= 1e-8:
        return float("nan"), float("nan")
    if kind == "spearman":
        return scipy_stats.spearmanr(x, y)
    return scipy_stats.pearsonr(x, y)


def _oracle_margin(rec: dict) -> float | None:
    lp_by_lambda = rec.get("logprobs_by_lambda")
    if not isinstance(lp_by_lambda, dict) or len(lp_by_lambda) < 2:
        return None
    values = sorted((float(v) for v in lp_by_lambda.values()), reverse=True)
    return values[0] - values[1]


def _parse_fixed_grid(grid_arg: str, stats: dict, oracle_arr: np.ndarray) -> np.ndarray:
    if grid_arg.strip().lower() != "auto":
        values = [float(x.strip()) for x in grid_arg.split(",") if x.strip()]
        return np.array(sorted(set(values)), dtype=np.float32)
    if stats.get("lambda_grid"):
        return np.array(stats["lambda_grid"], dtype=np.float32)
    values = sorted(set(np.round(oracle_arr, 2).tolist()))
    if len(values) >= 2:
        return np.array(values, dtype=np.float32)
    return np.arange(0.0, 1.0 + 1e-8, 0.05, dtype=np.float32)


def _assign_bins(lambdas: np.ndarray, boundaries: list[tuple[float, float]]) -> np.ndarray:
    bins = np.zeros(len(lambdas), dtype=int)
    for i, (lo, hi) in enumerate(boundaries):
        if hi >= 1.0 - 1e-8:
            bins[(lambdas >= lo) & (lambdas <= hi)] = i
        else:
            bins[(lambdas >= lo) & (lambdas < hi)] = i
    return bins


def _print_classification_metrics(
    pred_arr: np.ndarray, oracle_arr: np.ndarray, lambda_grid: list[float] | None
) -> None:
    if not lambda_grid or len(lambda_grid) != 3:
        return

    boundaries = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    bin_names = ["diversity [0,.3]", "balanced (.3,.7)", "relevance [.7,1]"]
    bin_centers = [float(v) for v in lambda_grid]

    pred_bins = _assign_bins(pred_arr, boundaries)
    oracle_bins = _assign_bins(oracle_arr, boundaries)

    n_correct = int((oracle_bins == pred_bins).sum())
    acc = n_correct / len(oracle_arr)

    print(f"\n3-Bin Classification:")
    print(f"  Bin centers: {', '.join(f'{c:.2f}' for c in bin_centers)}")
    print(f"  Accuracy: {acc:.4f} (n={n_correct}/{len(oracle_arr)}, random=0.333)")

    print(f"\n  Confusion matrix (rows=oracle, cols=pred):")
    header = " " * 27 + "".join(f"{n:>8s}" for n in ["diversity", "balanced", "relevance"])
    print(f"  {header}")
    for i, name in enumerate(bin_names):
        row_counts = [
            int(((oracle_bins == i) & (pred_bins == j)).sum())
            for j in range(3)
        ]
        print(f"    {name:25s} " + "".join(f"{c:>8d}" for c in row_counts))

    print(f"\n  Per-bin metrics:")
    for i, name in enumerate(bin_names):
        tp = int(((oracle_bins == i) & (pred_bins == i)).sum())
        n_pred = max(int((pred_bins == i).sum()), 1)
        n_oracle = max(int((oracle_bins == i).sum()), 1)
        prec = tp / n_pred
        rec = tp / n_oracle
        acc_per_bin = tp / n_oracle  # per-class accuracy
        print(f"    {name:25s} n={n_oracle:5d}  precision={prec:.3f}  recall={rec:.3f}  per-class-acc={acc_per_bin:.3f}")


def _print_baselines(pred_arr: np.ndarray, oracle_arr: np.ndarray, fixed_grid: np.ndarray) -> None:
    mse = float(np.mean((pred_arr - oracle_arr) ** 2))
    mean_pred = float(oracle_arr.mean())
    mean_mse = float(np.mean((mean_pred - oracle_arr) ** 2))
    mean_mae = float(np.mean(np.abs(mean_pred - oracle_arr)))

    fixed_scores = []
    for lam in fixed_grid:
        lam_f = float(lam)
        fixed_scores.append((
            lam_f,
            float(np.mean((lam_f - oracle_arr) ** 2)),
            float(np.mean(np.abs(lam_f - oracle_arr))),
        ))
    best_lam, best_mse, best_mae = min(fixed_scores, key=lambda item: item[1])
    r2_vs_mean = 1.0 - mse / mean_mse if mean_mse > 1e-12 else float("nan")
    r2_vs_best_fixed = 1.0 - mse / best_mse if best_mse > 1e-12 else float("nan")

    print("\nBaselines:")
    print(f"  Mean oracle λ={mean_pred:.3f}: MAE={mean_mae:.4f}, RMSE={np.sqrt(mean_mse):.4f}, MSE={mean_mse:.5f}")
    print(f"  Best fixed λ={best_lam:.2f}: MAE={best_mae:.4f}, RMSE={np.sqrt(best_mse):.4f}, MSE={best_mse:.5f}")
    print(f"  Predictor R2 vs mean:       {r2_vs_mean:.4f}")
    print(f"  Predictor R2 vs best fixed: {r2_vs_best_fixed:.4f}")


def _print_error_by_oracle_bucket(pred_arr: np.ndarray, oracle_arr: np.ndarray) -> None:
    print("\nError by oracle λ:")
    rounded = np.round(oracle_arr, 2)
    for lam in sorted(set(rounded.tolist())):
        mask = rounded == lam
        bucket_pred = pred_arr[mask]
        bucket_oracle = oracle_arr[mask]
        mae = float(np.mean(np.abs(bucket_pred - bucket_oracle)))
        rmse = float(np.sqrt(np.mean((bucket_pred - bucket_oracle) ** 2)))
        print(
            f"  λ={lam:.2f}: n={int(mask.sum()):5d}  "
            f"pred_mean={bucket_pred.mean():.3f}  MAE={mae:.4f}  RMSE={rmse:.4f}"
        )


def _print_margin_analysis(margins: np.ndarray, pred_arr: np.ndarray, oracle_arr: np.ndarray) -> None:
    if len(margins) == 0:
        return
    print("\nOracle logprob margin:")
    print(f"  N:      {len(margins)}")
    print(f"  Mean:   {margins.mean():.4f}")
    print(f"  Median: {np.median(margins):.4f}")
    print(f"  Q25:    {np.quantile(margins, 0.25):.4f}")
    print(f"  Q75:    {np.quantile(margins, 0.75):.4f}")
    for threshold in (0.01, 0.05, 0.10):
        pct = 100.0 * float((margins < threshold).mean())
        print(f"  margin < {threshold:.2f}: {pct:5.1f}%")

    print("\nError by oracle margin:")
    bins = [0.0, 0.01, 0.05, 0.10, 0.25, float("inf")]
    labels = ["[0,.01)", "[.01,.05)", "[.05,.10)", "[.10,.25)", "[.25,inf)"]
    abs_err = np.abs(pred_arr - oracle_arr)
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask = (margins >= lo) & (margins < hi)
        if not np.any(mask):
            continue
        print(
            f"  {label}: n={int(mask.sum()):5d}  "
            f"MAE={abs_err[mask].mean():.4f}  oracle_std={oracle_arr[mask].std():.3f}"
        )

    # High-margin vs low-margin summary with fallback comparison
    for threshold in (0.05, 0.10):
        high_mask = margins >= threshold
        low_mask = ~high_mask
        if not np.any(high_mask) or not np.any(low_mask):
            continue
        print(f"\n  margin >= {threshold:.2f} (high, n={int(high_mask.sum())}):")
        print(f"    oracle_mean={oracle_arr[high_mask].mean():.3f}  oracle_std={oracle_arr[high_mask].std():.3f}")
        print(f"    MAE={abs_err[high_mask].mean():.4f}  "
              f"RMSE={float(np.sqrt((abs_err[high_mask] ** 2).mean())):.4f}"
              f"  Pearson_r={_safe_corr(oracle_arr[high_mask], pred_arr[high_mask], 'pearson')[0]:.4f}")
        print(f"  margin <  {threshold:.2f} (low,  n={int(low_mask.sum())}):")
        print(f"    oracle_mean={oracle_arr[low_mask].mean():.3f}  oracle_std={oracle_arr[low_mask].std():.3f}")
        print(f"    MAE={abs_err[low_mask].mean():.4f}  "
              f"RMSE={float(np.sqrt((abs_err[low_mask] ** 2).mean())):.4f}"
              f"  Pearson_r={_safe_corr(oracle_arr[low_mask], pred_arr[low_mask], 'pearson')[0]:.4f}")
        # Fallback comparison: how would default 0.7 do on low-margin?
        fallback_mae = float(np.mean(np.abs(0.7 - oracle_arr[low_mask])))
        fallback_rmse = float(np.sqrt(np.mean((0.7 - oracle_arr[low_mask]) ** 2)))
        print(f"    Fallback (λ=0.7) MAE={fallback_mae:.4f} RMSE={fallback_rmse:.4f}")
        break  # only show one threshold


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    build_cfg = load_experiment_build_cfg(args.experiment, args.config_overrides)
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    chunk_mmr_cache = (
        Path(args.chunk_mmr_cache)
        if args.chunk_mmr_cache
        else resolve_chunk_mmr_cache_path(
            build_cfg,
            split_name=args.split_name,
            cache_root=args.chunk_mmr_cache_root,
        )
    )
    if not chunk_mmr_cache.exists():
        raise FileNotFoundError(
            f"Chunk-MMR cache not found: {chunk_mmr_cache}. "
            "Run scripts/learned_lambda/run_generate_oracle_prompts.sh first, "
            "or pass --chunk-mmr-cache."
        )

    oracle_by_eid: dict[str, dict] = {}
    with open(args.oracle_lambdas) as f:
        for line in tqdm(
            f,
            desc="load oracle lambdas",
            unit="line",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            rec = json.loads(line.strip())
            rec["oracle_lambda"] = float(rec["oracle_lambda"])
            oracle_by_eid[rec["event_id"]] = rec
    _log(f"Loaded {len(oracle_by_eid)} oracle λ values", show_progress=show_progress)

    model, stats = load_predictor(
        args.model,
        args.feature_stats,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    _log(f"Loaded predictor: {args.model}", show_progress=show_progress)
    _log(
        f"Predictor metadata: model_type={stats.get('model_type', 'regression')} "
        f"input_dim={stats.get('input_dim')}",
        show_progress=show_progress,
    )

    if args.candidate_top_k is not None:
        candidate_top_k = int(args.candidate_top_k)
    else:
        candidate_top_k = None
    candidate_pool_label = "full" if candidate_top_k is None else f"top_{candidate_top_k}"
    retrieval_top_k = int(pick_retrieval_value(None, retrieval_cfg, "top_k", 16))
    alpha_dense = float(pick_retrieval_value(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70))
    alpha_lexical = float(pick_retrieval_value(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20))
    alpha_bm25 = float(pick_retrieval_value(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10))

    _log(f"Loaded build config from experiment={args.experiment}", show_progress=show_progress)
    _log(f"chunk_mmr_cache={chunk_mmr_cache}", show_progress=show_progress)
    _log(
        f"candidate_pool={candidate_pool_label}, prompt_top_k={retrieval_top_k}, alpha_dense={alpha_dense}, "
        f"alpha_lexical={alpha_lexical}, alpha_bm25={alpha_bm25}",
        show_progress=show_progress,
    )

    chunk_samples = _load_pickle(chunk_mmr_cache)
    _log(f"Loaded {len(chunk_samples)} ChunkMMR samples", show_progress=show_progress)
    arrays, oracle_arr, oracle_records, skipped = build_matched_chunk_embedding_arrays(
        tqdm(
            chunk_samples,
            desc="build embedding tensors",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        ),
        oracle_by_eid,
        candidate_top_k=candidate_top_k,
        alpha_dense=alpha_dense,
        alpha_lexical=alpha_lexical,
        alpha_bm25=alpha_bm25,
    )
    if skipped > 0:
        _log(f"Skipped {skipped} samples without oracle λ", show_progress=show_progress)
    candidate_capacity = int(arrays["candidate_emb"].shape[1])
    candidate_counts = arrays["candidate_counts"]
    _log(
        f"Samples: {len(oracle_arr)} (matched with oracle λ), "
        f"candidate_capacity={candidate_capacity}, "
        f"candidate_count_mean={candidate_counts.mean():.1f}, "
        f"candidate_count_max={candidate_counts.max()}",
        show_progress=show_progress,
    )

    with torch.no_grad():
        pred_arr = model(
            torch.from_numpy(arrays["claim_emb"]),
            torch.from_numpy(arrays["candidate_emb"]),
            torch.from_numpy(arrays["candidate_mask"]),
        ).numpy()
    margins_with_missing = [_oracle_margin(rec) for rec in oracle_records]
    has_margin = np.array([m is not None for m in margins_with_missing], dtype=bool)
    margins = np.array([m for m in margins_with_missing if m is not None], dtype=np.float32)

    # Regression metrics
    mse = float(np.mean((pred_arr - oracle_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - oracle_arr)))
    rmse = float(np.sqrt(mse))
    pearson_r, pearson_p = _safe_corr(oracle_arr, pred_arr, "pearson")
    spearman_r, spearman_p = _safe_corr(oracle_arr, pred_arr, "spearman")

    print(f"\n{'=' * 50}")
    print(f"Predictor Evaluation")
    print(f"{'=' * 50}")
    print(f"MAE:        {mae:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"MSE:        {mse:.5f}")
    print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

    fixed_grid = _parse_fixed_grid(args.fixed_lambda_grid, stats, oracle_arr)
    _print_baselines(pred_arr, oracle_arr, fixed_grid)

    # Oracle λ distribution
    print(f"\nOracle λ distribution:")
    print(f"  Mean: {oracle_arr.mean():.3f}")
    print(f"  Std:  {oracle_arr.std():.3f}")
    print(f"  Min:  {oracle_arr.min():.2f}")
    print(f"  Max:  {oracle_arr.max():.2f}")

    oracle_counts = Counter(np.round(oracle_arr, 2).tolist())
    for lam in sorted(oracle_counts.keys()):
        count = oracle_counts[lam]
        pct = 100.0 * count / len(oracle_arr)
        bar = "#" * int(pct / 2)
        print(f"  λ={lam:.2f}: {count:5d} ({pct:5.1f}%) {bar}")

    # Predicted λ distribution
    print(f"\nPredicted λ distribution:")
    print(f"  Mean: {pred_arr.mean():.3f}")
    print(f"  Std:  {pred_arr.std():.3f}")
    print(f"  Min:  {pred_arr.min():.3f}")
    print(f"  Max:  {pred_arr.max():.3f}")
    std_ratio = float(pred_arr.std() / oracle_arr.std()) if oracle_arr.std() > 1e-8 else float("nan")
    print(f"  Std ratio pred/oracle: {std_ratio:.3f}")

    bins = np.arange(0, 1.05 + 1e-8, 0.05)
    hist, _ = np.histogram(pred_arr, bins=bins)
    for i in range(len(bins) - 1):
        pct = 100.0 * hist[i] / len(pred_arr)
        bar = "#" * int(pct / 2)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:5d} ({pct:5.1f}%) {bar}")

    _print_error_by_oracle_bucket(pred_arr, oracle_arr)
    if len(margins) > 0:
        _print_margin_analysis(margins, pred_arr[has_margin], oracle_arr[has_margin])

    # Classification metrics for classifier models
    model_type = str(stats.get("model_type") or "regression").strip().lower()
    if model_type == "classifier":
        _print_classification_metrics(pred_arr, oracle_arr, stats.get("lambda_grid"))

    # Quality assessment
    print(f"\n{'=' * 50}")
    print("Assessment:")
    if oracle_arr.std() < 0.05:
        print("  WARNING: Oracle λ has very low variance (std < 0.05).")
        print("  Claim-adaptive λ may have limited value.")
    if mae > 0.2:
        print(f"  WARNING: MAE ({mae:.4f}) is high. Predictor may need more features or data.")
    if pearson_r < 0.2:
        print(f"  WARNING: Low correlation ({pearson_r:.4f}). Features may not capture oracle λ signal well.")
    if np.isfinite(std_ratio) and std_ratio < 0.5:
        print(f"  WARNING: Predicted λ variance is compressed (std ratio={std_ratio:.3f}).")
    if mae < 0.15 and pearson_r > 0.3:
        print("  Predictor quality looks acceptable for pipeline integration.")


if __name__ == "__main__":
    main()
