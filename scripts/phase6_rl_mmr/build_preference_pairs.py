"""Build preference pairs from scored trajectory pool for DPO training.

Usage:
    PYTHONPATH=src python scripts/phase6_rl_mmr/build_preference_pairs.py \\
        --trajectories outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train_scored.jsonl \\
        --output-dir outputs/rl_mmr/dpo_stepwise/preference_pairs \\
        --delta 0.05 --max-pairs-per-claim 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fact_checking.rl_mmr.trajectory import LAMBDA_GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DPO preference pairs from trajectory pool.")
    p.add_argument("--trajectories", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--delta", type=float, default=0.05,
                   help="Minimum utility gap for a valid preference pair.")
    p.add_argument("--max-pairs-per-claim", type=int, default=10)
    p.add_argument("--lambda-grid", nargs="*", type=float, default=LAMBDA_GRID)
    p.add_argument("--split-name", type=str, default="train")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _lambda_to_index(lam: float, grid: list[float]) -> int:
    """Map a lambda value to the closest index in the grid."""
    grid_arr = np.array(grid, dtype=np.float32)
    return int(np.argmin(np.abs(grid_arr - float(lam))))


def main() -> None:
    args = parse_args()
    lambda_grid = [float(x) for x in args.lambda_grid]
    n_lambda = len(lambda_grid)

    # Load trajectories
    with Path(args.trajectories).open("r", encoding="utf-8") as f:
        all_trajs = [json.loads(line) for line in f if line.strip()]

    # Filter: only trajectories with utility and state_features
    # Also exclude sentinel utilities (<= -99) that come from missing oracle logprobs
    def _valid_utility(u):
        return u is not None and float(u) > -99.0

    valid_trajs = [
        t for t in all_trajs
        if _valid_utility(t.get("utility")) and t.get("state_features") is not None
    ]
    n_sentinel = sum(1 for t in all_trajs
                     if t.get("utility") is not None and float(t["utility"]) <= -99.0)
    print(f"Loaded {len(all_trajs)} trajectories, {len(valid_trajs)} with valid utility+features")
    if n_sentinel:
        print(f"  Filtered {n_sentinel} trajectories with sentinel utility (<= -99)")

    # Group by event_id
    by_event: dict[str, list[dict]] = defaultdict(list)
    for t in valid_trajs:
        by_event[t["event_id"]].append(t)

    n_claims = len(by_event)
    print(f"Claims with valid trajectories: {n_claims}")

    # Build preference pairs
    rng = np.random.default_rng(args.seed)

    win_features_list: list[np.ndarray] = []
    win_lambdas_list: list[np.ndarray] = []
    lose_features_list: list[np.ndarray] = []
    lose_lambdas_list: list[np.ndarray] = []
    utility_gaps: list[float] = []
    event_ids: list[str] = []
    pair_stats: list[dict] = []

    n_claims_with_pairs = 0
    n_skipped_low_gap = 0

    for eid, trajs in by_event.items():
        # Sort by utility descending
        trajs.sort(key=lambda t: float(t["utility"]), reverse=True)

        pairs_for_claim = 0
        n_trajs = len(trajs)

        # Generate candidate pairs sorted by utility gap
        candidate_pairs: list[tuple[int, int, float]] = []
        for i in range(n_trajs):
            for j in range(i + 1, n_trajs):
                gap = float(trajs[i]["utility"]) - float(trajs[j]["utility"])
                if gap >= args.delta:
                    candidate_pairs.append((i, j, gap))

        if not candidate_pairs:
            n_skipped_low_gap += 1
            continue

        # Sort by utility gap descending, prioritize evidence_set_diff
        candidate_pairs.sort(key=lambda x: (
            trajs[x[0]]["evidence_set_key"] != trajs[x[1]]["evidence_set_key"],
            x[2],
        ), reverse=True)

        # Limit pairs per claim
        selected = candidate_pairs[:args.max_pairs_per_claim]

        for i, j, gap in selected:
            tw = trajs[i]
            tl = trajs[j]

            # Extract state features and lambda indices
            wf = np.array(tw["state_features"], dtype=np.float32)  # [K, D]
            lf = np.array(tl["state_features"], dtype=np.float32)

            wl = np.array([_lambda_to_index(lam, lambda_grid) for lam in tw["lambda_schedule"]], dtype=np.int64)
            ll = np.array([_lambda_to_index(lam, lambda_grid) for lam in tl["lambda_schedule"]], dtype=np.int64)

            # Ensure same K
            K = wf.shape[0]
            if lf.shape[0] != K or len(wl) != K or len(ll) != K:
                continue

            win_features_list.append(wf)
            win_lambdas_list.append(wl)
            lose_features_list.append(lf)
            lose_lambdas_list.append(ll)
            utility_gaps.append(gap)
            event_ids.append(eid)

            pair_stats.append({
                "event_id": eid,
                "utility_win": float(tw["utility"]),
                "utility_lose": float(tl["utility"]),
                "utility_gap": gap,
                "evidence_set_diff": tw["evidence_set_key"] != tl["evidence_set_key"],
                "schedule_win": tw["lambda_schedule"],
                "schedule_lose": tl["lambda_schedule"],
            })

            pairs_for_claim += 1

        if pairs_for_claim > 0:
            n_claims_with_pairs += 1

    if not win_features_list:
        raise ValueError("No valid preference pairs generated. Check delta threshold and trajectory utilities.")

    # Stack into arrays
    win_features = np.stack(win_features_list).astype(np.float32)
    win_lambdas = np.stack(win_lambdas_list).astype(np.int64)
    lose_features = np.stack(lose_features_list).astype(np.float32)
    lose_lambdas = np.stack(lose_lambdas_list).astype(np.int64)
    utility_gaps_arr = np.array(utility_gaps, dtype=np.float32)

    print(f"\nPair statistics:")
    print(f"  Claims with pairs: {n_claims_with_pairs}/{n_claims}")
    print(f"  Skipped (low gap): {n_skipped_low_gap}")
    print(f"  Total pairs: {len(win_features)}")
    print(f"  Utility gap mean: {utility_gaps_arr.mean():.4f} std: {utility_gaps_arr.std():.4f}")
    print(f"  Evidence set diff ratio: {sum(1 for p in pair_stats if p['evidence_set_diff']) / len(pair_stats):.3f}")
    print(f"  Feature dim: {win_features.shape[2]} steps: {win_features.shape[1]}")

    # Save NPZ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{args.split_name}_pairs.npz"
    np.savez_compressed(
        npz_path,
        win_features=win_features,
        win_lambdas=win_lambdas,
        lose_features=lose_features,
        lose_lambdas=lose_lambdas,
        utility_gaps=utility_gaps_arr,
        event_ids=np.array(event_ids, dtype=object),
    )
    print(f"Saved pairs to {npz_path}")

    # Save statistics
    stats = {
        "n_claims_total": n_claims,
        "n_claims_with_pairs": n_claims_with_pairs,
        "n_claims_skipped_low_gap": n_skipped_low_gap,
        "n_pairs_total": len(win_features),
        "utility_gap_mean": float(utility_gaps_arr.mean()),
        "utility_gap_std": float(utility_gaps_arr.std()),
        "utility_gap_min": float(utility_gaps_arr.min()),
        "utility_gap_max": float(utility_gaps_arr.max()),
        "evidence_set_diff_fraction": float(
            sum(1 for p in pair_stats if p["evidence_set_diff"]) / max(len(pair_stats), 1)
        ),
        "delta": args.delta,
        "max_pairs_per_claim": args.max_pairs_per_claim,
        "feature_dim": int(win_features.shape[2]),
        "n_steps": int(win_features.shape[1]),
        "n_lambda": n_lambda,
        "lambda_grid": lambda_grid,
    }
    stats_path = output_dir / f"{args.split_name}_pair_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()
