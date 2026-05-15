"""Check whether state features actually differentiate winner vs loser λ choices.

Usage:
    PYTHONPATH=src python scripts/rl_mmr/diagnose_features.py \
        --pairs outputs/rl_mmr/dpo_stepwise/preference_pairs_v2/train_pairs.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.pairs, allow_pickle=True)
    wf = data["win_features"]    # [N, K, D]
    wl = data["win_lambdas"]     # [N, K]  lambda index (0-4)
    lf = data["lose_features"]   # [N, K, D]
    ll = data["lose_lambdas"]    # [N, K]
    gaps = data["utility_gaps"]

    N, K, D = wf.shape
    n_actions = 5
    print(f"N={N} K={K} D={D} actions={n_actions}")

    # ============================================================
    # Check 1: Do winners and losers use DIFFERENT lambdas?
    # ============================================================
    same_lambda = (wl == ll)  # [N, K]
    all_same = same_lambda.all(axis=1)  # winner and loser use identical λ at every step
    any_diff = ~all_same
    print(f"\n=== Check 1: Lambda choice differences ===")
    print(f"Pairs where winner & loser use SAME λ at all steps: {all_same.sum()} / {N}")
    print(f"Pairs with at least one step different: {any_diff.sum()} / {N}")
    if all_same.sum() > N * 0.8:
        print("  → MOST pairs have identical λ choices. DPO has nothing to distinguish!")
        print("  → Different utilities likely come from deterministic MMR behavior at same λ,")
        print("    not from different λ choices. This means state features are irrelevant.")

    # Per-step: what fraction of winners use each λ?
    print(f"\nWinner λ distribution by step:")
    for step in range(K):
        step_wl = wl[:, step]
        counts = np.bincount(step_wl, minlength=n_actions)
        pcts = 100.0 * counts / max(counts.sum(), 1)
        dist = "  ".join(f"λ{i}:{counts[i]}({pcts[i]:.0f}%)" for i in range(n_actions))
        print(f"  Step {step}: {dist}")

    print(f"\nLoser λ distribution by step:")
    for step in range(K):
        step_ll = ll[:, step]
        counts = np.bincount(step_ll, minlength=n_actions)
        pcts = 100.0 * counts / max(counts.sum(), 1)
        dist = "  ".join(f"λ{i}:{counts[i]}({pcts[i]:.0f}%)" for i in range(n_actions))
        print(f"  Step {step}: {dist}")

    # ============================================================
    # Check 2: Feature difference between winner and loser
    # ============================================================
    feat_diff = wf - lf  # [N, K, D]
    # When winner and loser use DIFFERENT λ at that step, are features different?
    for step in range(K):
        mask = wl[:, step] != ll[:, step]  # pairs where λ differs at this step
        if mask.sum() < 10:
            print(f"\nStep {step}: only {mask.sum()} pairs have different λ, skipping")
            continue
        diff = feat_diff[mask, step, :]  # [n_diff, D]
        abs_mean = np.abs(diff).mean(axis=0)
        print(f"\nStep {step} (n_diff={mask.sum()}): mean |feature_diff| per dimension:")
        for d in range(D):
            print(f"  dim {d:2d}: {abs_mean[d]:.4f}")

    # ============================================================
    # Check 3: Can we predict which λ is better from features alone?
    # ============================================================
    print(f"\n=== Check 3: Supervised λ prediction ===")
    # Simple: for each step independently, predict winner λ from winner features
    from sklearn.linear_model import LogisticRegression

    for step in range(K):
        X_train = wf[:, step, :]  # [N, D]
        y_train = wl[:, step]      # [N]
        # Train on 80%, test on 20%
        n_train = int(N * 0.8)
        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(X_train[:n_train], y_train[:n_train])
        train_acc = model.score(X_train[:n_train], y_train[:n_train])
        test_acc = model.score(X_train[n_train:], y_train[n_train:])
        baseline = max(np.bincount(y_train[:n_train], minlength=n_actions)) / n_train
        print(f"  Step {step}: train_acc={train_acc:.3f} test_acc={test_acc:.3f} baseline={baseline:.3f}")

    # ============================================================
    # Check 4: Feature correlation with utility gap
    # ============================================================
    print(f"\n=== Check 4: Feature-gap correlation ===")
    # Average feature values for winners
    # Does higher utility gap correlate with larger feature differences?
    for step in range(K):
        mask = wl[:, step] != ll[:, step]
        if mask.sum() < 50:
            continue
        diff_norm = np.linalg.norm(feat_diff[mask, step, :], axis=1)
        corr = np.corrcoef(diff_norm, gaps[mask])[0, 1]
        print(f"  Step {step}: corr(|feat_diff|, utility_gap) = {corr:.3f}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n=== Summary ===")
    if all_same.sum() > N * 0.8:
        print("PROBLEM: Most preference pairs have identical λ schedules for winner and loser.")
        print("The utility difference is NOT caused by different λ choices, but by randomness")
        print("in the MMR selection process or feature computation.")
        print("This is why DPO cannot learn — the λ choices don't actually differ.")
        print("")
        print("FIX: Modify trajectory generation so that different schedules truly produce")
        print("different λ sequences. Currently, schedules like [0.7,0.7,0.7,0.5,0.3] and")
        print("[0.7,0.7,0.7,0.7,0.7] may select the SAME items at early steps, leading to")
        print("identical or near-identical state features.")
    else:
        print("Lambda choices differ between winners and losers — feature quality is the issue.")


if __name__ == "__main__":
    main()
