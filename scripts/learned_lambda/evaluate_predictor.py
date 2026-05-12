"""Step 4: Evaluate the trained λ predictor.

Reports MAE, RMSE, correlation, and distribution analysis of oracle vs predicted λ.

Usage:
    PYTHONPATH=src python scripts/learned_lambda/evaluate_predictor.py \
        --model outputs/learned_lambda/predictor.pt \
        --feature-stats outputs/learned_lambda/feature_stats.json \
        --oracle-lambdas outputs/learned_lambda/oracle_lambda_train.jsonl \
        --premmr-cache outputs/cache/pre_mmr/53a3588e485d/train.pkl
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
from fact_checking.learned_lambda.features import extract_features
from fact_checking.learned_lambda.predictor import load_predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate λ predictor.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--feature-stats", type=str, required=True)
    p.add_argument("--oracle-lambdas", type=str, required=True)
    p.add_argument("--premmr-cache", type=str, required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--alpha-dense", type=float, default=0.70)
    p.add_argument("--alpha-lexical", type=float, default=0.20)
    p.add_argument("--alpha-bm25", type=float, default=0.10)
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def _log(message: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)
    else:
        print(message, flush=True)


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

    oracle_by_eid: dict[str, float] = {}
    with open(args.oracle_lambdas) as f:
        for line in tqdm(
            f,
            desc="load oracle lambdas",
            unit="line",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            rec = json.loads(line.strip())
            oracle_by_eid[rec["event_id"]] = rec["oracle_lambda"]
    _log(f"Loaded {len(oracle_by_eid)} oracle λ values", show_progress=show_progress)

    pre_samples = _load_pickle(Path(args.premmr_cache))
    _log(f"Loaded {len(pre_samples)} PreMMR samples", show_progress=show_progress)
    matched = [
        p for p in tqdm(
            pre_samples,
            desc="match samples",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        if p.event_id in oracle_by_eid
    ]
    _log(f"Samples: {len(matched)} (matched with oracle λ)", show_progress=show_progress)
    if not matched:
        raise ValueError("No PreMMR samples matched oracle λ values by event_id.")

    model, stats = load_predictor(
        args.model,
        args.feature_stats,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    _log(f"Loaded predictor: {args.model}", show_progress=show_progress)

    features = np.stack([
        extract_features(p, args.alpha_dense, args.alpha_lexical, args.alpha_bm25)
        for p in tqdm(
            matched,
            desc="extract features",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        )
    ])
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    features_norm = (features - mean) / std

    with torch.no_grad():
        x = torch.from_numpy(features_norm)
        pred_arr = model(x).numpy()
    oracle_arr = np.array([oracle_by_eid[p.event_id] for p in matched])

    # Regression metrics
    mae = float(np.mean(np.abs(pred_arr - oracle_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - oracle_arr) ** 2)))
    pearson_r, pearson_p = scipy_stats.pearsonr(oracle_arr, pred_arr)
    spearman_r, spearman_p = scipy_stats.spearmanr(oracle_arr, pred_arr)

    print(f"\n{'=' * 50}")
    print(f"Predictor Evaluation")
    print(f"{'=' * 50}")
    print(f"MAE:        {mae:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

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

    bins = np.arange(0, 1.05 + 1e-8, 0.05)
    hist, _ = np.histogram(pred_arr, bins=bins)
    for i in range(len(bins) - 1):
        pct = 100.0 * hist[i] / len(pred_arr)
        bar = "#" * int(pct / 2)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:5d} ({pct:5.1f}%) {bar}")

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
    if mae < 0.15 and pearson_r > 0.3:
        print("  Predictor quality looks acceptable for pipeline integration.")


if __name__ == "__main__":
    main()
