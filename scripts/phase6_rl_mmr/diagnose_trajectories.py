"""Diagnose whether λ schedule changes actually affect utility and preference signal quality.

Usage:
    PYTHONPATH=src python scripts/phase6_rl_mmr/diagnose_trajectories.py \
        --trajectories outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train_scored.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose trajectory utility signal quality.")
    p.add_argument("--trajectories", type=str, required=True)
    p.add_argument("--pairs", type=str, default=None,
                   help="Optional preference pairs NPZ for gap diagnostics.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load trajectories
    trajs = []
    with Path(args.trajectories).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajs.append(json.loads(line))

    # Filter valid
    valid = [t for t in trajs if t.get("utility") is not None and float(t["utility"]) > -99.0]
    print(f"Total: {len(trajs)}  Valid: {len(valid)}")

    by_event: dict[str, list[dict]] = defaultdict(list)
    for t in valid:
        by_event[t["event_id"]].append(t)

    n_claims = len(by_event)
    print(f"Claims with valid trajectories: {n_claims}")

    # ============================================================
    # Check 1: Per-claim utility dispersion
    # ============================================================
    utility_ranges: list[float] = []
    utility_stds: list[float] = []
    fixed_07_utils: list[float] = []
    best_non_fixed_deltas: list[float] = []

    for eid, claim_trajs in by_event.items():
        utils = [float(t["utility"]) for t in claim_trajs if t.get("utility") is not None]
        if len(utils) < 3:
            continue

        utility_ranges.append(max(utils) - min(utils))
        utility_stds.append(float(np.std(utils)))

        # Utility of fixed λ=0.7 baseline
        fixed_trajs = [t for t in claim_trajs
                       if t.get("schedule_type") == "handcrafted"
                       and len(set(t.get("lambda_schedule", []))) == 1
                       and round(t["lambda_schedule"][0], 2) == 0.7]
        if fixed_trajs:
            fixed_u = float(max(t["utility"] for t in fixed_trajs))
            fixed_07_utils.append(fixed_u)
            # Best non-0.7 utility delta
            non_fixed = [float(t["utility"]) for t in claim_trajs
                         if not (len(set(t.get("lambda_schedule", []))) == 1
                                 and round(t["lambda_schedule"][0], 2) == 0.7)]
            if non_fixed:
                best_non_fixed_deltas.append(max(non_fixed) - fixed_u)

    utility_ranges = np.array(utility_ranges)
    utility_stds = np.array(utility_stds)
    fixed_07_utils = np.array(fixed_07_utils)
    best_non_fixed_deltas = np.array(best_non_fixed_deltas)

    print(f"\n=== Check 1: Per-claim utility dispersion ===")
    print(f"Claims with >= 3 trajectories: {len(utility_ranges)}")
    print(f"Utility range per claim: mean={utility_ranges.mean():.4f} median={np.median(utility_ranges):.4f}")
    print(f"   p10={np.percentile(utility_ranges, 10):.4f} p25={np.percentile(utility_ranges, 25):.4f}")
    print(f"   p75={np.percentile(utility_ranges, 75):.4f} p90={np.percentile(utility_ranges, 90):.4f}")
    print(f"Utility std per claim: mean={utility_stds.mean():.4f} median={np.median(utility_stds):.4f}")
    print(f"Claims with range < 0.01: {(utility_ranges < 0.01).sum()}")
    print(f"Claims with range < 0.05: {(utility_ranges < 0.05).sum()}")
    print(f"Claims with range < 0.10: {(utility_ranges < 0.10).sum()}")
    print(f"Best non-0.7 delta: mean={best_non_fixed_deltas.mean():.4f} median={np.median(best_non_fixed_deltas):.4f}")
    n_positive = (best_non_fixed_deltas > 0).sum()
    print(f"Claims where best non-0.7 > fixed: {n_positive}/{len(best_non_fixed_deltas)}")
    if len(best_non_fixed_deltas) > 0:
        print(f"  Of those, delta mean={best_non_fixed_deltas[best_non_fixed_deltas > 0].mean():.4f}")

    # ============================================================
    # Check 2: Oracle λ distribution — which λ wins most often?
    # ============================================================
    lambda_wins: dict[float, int] = defaultdict(int)
    lambda_utility_sum: dict[float, float] = defaultdict(float)
    schedule_type_wins: dict[str, int] = defaultdict(int)

    for eid, claim_trajs in by_event.items():
        # Best trajectory per claim
        best = max(claim_trajs, key=lambda t: float(t["utility"]))
        avg_lam = np.mean(best.get("lambda_schedule", [0.7]))
        # Round to nearest grid value
        nearest = min([0.1, 0.3, 0.5, 0.7, 0.9], key=lambda x: abs(x - avg_lam))
        lambda_wins[nearest] += 1
        lambda_utility_sum[nearest] += float(best["utility"])
        schedule_type_wins[best.get("schedule_type", "?")] += 1

    print(f"\n=== Check 2: Oracle (best) λ per claim ===")
    total_wins = sum(lambda_wins.values())
    for lam in sorted(lambda_wins.keys()):
        n = lambda_wins[lam]
        avg_u = lambda_utility_sum[lam] / max(n, 1)
        print(f"  λ≈{lam:.1f}: {n} wins ({100*n/max(total_wins,1):.1f}%) avg_utility={avg_u:.4f}")
    print(f"  Schedule type wins: {dict(schedule_type_wins)}")

    # ============================================================
    # Check 3: Does constant λ=0.7 dominate?
    # ============================================================
    fixed_best_count = 0
    for eid, claim_trajs in by_event.items():
        best = max(claim_trajs, key=lambda t: float(t["utility"]))
        sched = best.get("lambda_schedule", [])
        if len(set(sched)) == 1 and round(sched[0], 2) == 0.7:
            fixed_best_count += 1
    print(f"\nClaims where fixed λ=0.7 IS the best: {fixed_best_count}/{n_claims} ({100*fixed_best_count/max(n_claims,1):.1f}%)")

    # ============================================================
    # Check 4: Preference pair signal (if pairs available)
    # ============================================================
    if args.pairs:
        data = np.load(args.pairs, allow_pickle=True)
        gaps = data["utility_gaps"]
        gaps = gaps[gaps < 50]  # exclude extreme outliers
        print(f"\n=== Check 4: Preference pair signal ===")
        print(f"Pairs: {len(gaps)} (excluding gaps > 50)")
        print(f"Gap: mean={gaps.mean():.4f} median={np.median(gaps):.4f}")
        print(f"  p25={np.percentile(gaps, 25):.4f} p75={np.percentile(gaps, 75):.4f}")
        print(f"  p90={np.percentile(gaps, 90):.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n=== Summary ===")
    median_range = np.median(utility_ranges)
    if median_range < 0.05:
        signal = "VERY WEAK"
    elif median_range < 0.1:
        signal = "WEAK"
    elif median_range < 0.5:
        signal = "MODERATE"
    else:
        signal = "STRONG"

    print(f"Utility range median: {median_range:.4f} → signal quality: {signal}")
    print(f"Fixed λ=0.7 is best for {100*fixed_best_count/max(n_claims,1):.1f}% of claims")
    if signal in ("VERY WEAK", "WEAK"):
        print(f"\nRECOMMENDATION: The λ schedule has almost no impact on utility for most claims.")
        print(f"DPO cannot learn meaningful step-wise preferences under these conditions.")
        print(f"Consider:")
        print(f"  - Checking if utility computation (verifier logprob) is correct")
        print(f"  - Using a different reward definition (not just correct label logprob)")
        print(f"  - Simplifying to claim-level λ (which experiments 1-4 already showed is weak)")
        print(f"  - Moving to multi-weight MMR (experiment 6) which has stronger differentiation")


if __name__ == "__main__":
    main()
