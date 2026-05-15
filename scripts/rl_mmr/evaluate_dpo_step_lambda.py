"""Evaluate a trained DPO step-wise λ policy.

Two evaluation modes:
  A. Offline utility comparison (no training needed):
     Compares DPO-selected evidence sets vs fixed λ=0.7 using pre-computed utilities.

  B. Full pipeline guidance:
     Instructions for running the complete build → train → infer pipeline.

Usage (offline eval):
    PYTHONPATH=src python scripts/rl_mmr/evaluate_dpo_step_lambda.py \\
        --policy outputs/rl_mmr/dpo_stepwise/checkpoints \\
        --chunk-mmr-cache outputs/cache/chunk_mmr/<fp>/chunk_mmr_test.pkl \\
        --oracle-logprobs data/learned_lambda/oracle_lambda_test.jsonl \\
        --output-dir outputs/rl_mmr/dpo_stepwise/eval
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.candidates import (
    ChunkMMRSample,
    _load_pickle,
    compute_hybrid_scores,
)
from fact_checking.learned_lambda.cache_utils import (
    load_experiment_build_cfg,
    pick_retrieval_value,
    resolve_chunk_mmr_cache_path,
)
from fact_checking.rl_mmr.dpo_selector import load_dpo_step_policy, select_candidates_dpo_stepwise
from fact_checking.rl_mmr.trajectory import LAMBDA_GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DPO step-wise λ policy.")
    p.add_argument("--policy", type=str, required=True, help="Path to trained policy directory.")
    p.add_argument("--chunk-mmr-cache", type=str, default=None)
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--oracle-logprobs", type=str, default=None,
                   help="Path to oracle logprobs JSONL for utility comparison.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--lambda-grid", nargs="*", type=float, default=LAMBDA_GRID)
    p.add_argument("--inference-mode", type=str, default="argmax", choices=["argmax", "sample"])
    p.add_argument("--sample-temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def load_oracle_logprobs(path: str) -> dict[str, dict[str, float]]:
    """Load oracle logprobs. Returns {event_id: {"0.70": logprob, ...}}."""
    records: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = str(rec.get("event_id", ""))
            if not eid:
                continue
            lp_by_lam = rec.get("logprobs_by_lambda", {})
            records[eid] = {str(k): float(v) for k, v in lp_by_lam.items()}
    return records


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_grid = np.array([float(x) for x in args.lambda_grid], dtype=np.float32)

    # Resolve chunk cache
    build_cfg = load_experiment_build_cfg(args.experiment, args.config_overrides)
    retrieval_cfg = dict(build_cfg.get("retrieval", {}) or {})
    if args.chunk_mmr_cache:
        cache_path = Path(args.chunk_mmr_cache)
    else:
        cache_path = resolve_chunk_mmr_cache_path(
            build_cfg, split_name=args.split_name, cache_root=args.chunk_mmr_cache_root,
        )

    top_k = int(pick_retrieval_value(args.top_k, retrieval_cfg, "top_k", 5))
    alpha_dense = float(pick_retrieval_value(args.alpha_dense, retrieval_cfg, "alpha_dense", 0.70))
    alpha_lexical = float(pick_retrieval_value(args.alpha_lexical, retrieval_cfg, "alpha_lexical", 0.20))
    alpha_bm25 = float(pick_retrieval_value(args.alpha_bm25, retrieval_cfg, "alpha_bm25", 0.10))

    print(f"Policy: {args.policy}")
    print(f"Chunk-MMR cache: {cache_path}")
    print(f"top_k={top_k} alpha=({alpha_dense:.2f},{alpha_lexical:.2f},{alpha_bm25:.2f})")
    print(f"Inference mode: {args.inference_mode}")

    # Load policy
    policy, feature_stats = load_dpo_step_policy(args.policy)
    print(f"Loaded policy: input_dim={feature_stats['input_dim']} actions={feature_stats['n_actions']}")

    # Load chunk cache
    chunk_samples: list[ChunkMMRSample] = _load_pickle(cache_path)
    if args.sample_limit:
        chunk_samples = chunk_samples[:args.sample_limit]
    print(f"Loaded {len(chunk_samples)} chunk samples")

    # Load oracle logprobs for utility comparison
    oracle = None
    if args.oracle_logprobs:
        oracle = load_oracle_logprobs(args.oracle_logprobs)
        print(f"Loaded oracle logprobs for {len(oracle)} claims")

    # Run DPO step-wise selection
    results: list[dict[str, Any]] = []
    lambda_dist_by_step: dict[int, list[float]] = defaultdict(list)

    for sample in tqdm(
        chunk_samples, desc="DPO selection", unit="claim",
        dynamic_ncols=True, disable=not show_progress,
    ):
        row = select_candidates_dpo_stepwise(
            sample, policy, feature_stats, lambda_grid,
            top_k=top_k,
            alpha_dense=alpha_dense, alpha_lexical=alpha_lexical, alpha_bm25=alpha_bm25,
            inference_mode=args.inference_mode,
            sample_temperature=args.sample_temperature,
            random_seed=args.seed,
        )
        chosen_lambdas = row.pop("_dpo_chosen_lambdas", [])
        selected_ids = [
            int(i) for c in row.get("candidates", [])
            for i, orig_c in enumerate(sample.candidates)
            if c.get("text") == orig_c.get("text")
        ]
        results.append({
            "event_id": sample.event_id,
            "claim": sample.claim,
            "gold_label": sample.label,
            "chosen_lambdas": chosen_lambdas,
            "selected_ids": selected_ids,
            "n_candidates": len(sample.candidates),
        })
        for t, lam in enumerate(chosen_lambdas):
            lambda_dist_by_step[t].append(lam)

    # Statistics
    all_lambdas = [lam for r in results for lam in r["chosen_lambdas"]]
    lam_arr = np.array(all_lambdas, dtype=np.float32)

    print(f"\n=== DPO Step-wise λ Evaluation ({args.split_name}) ===")
    print(f"Samples: {len(results)}")
    print(f"λ overall: mean={lam_arr.mean():.3f} std={lam_arr.std():.3f}")

    # Per-step λ distribution
    print("\nPer-step λ distribution:")
    for t in sorted(lambda_dist_by_step.keys()):
        vals = np.array(lambda_dist_by_step[t], dtype=np.float32)
        unique, counts = np.unique(vals, return_counts=True)
        dist_str = " ".join(f"λ={u:.1f}:{c}" for u, c in zip(unique, counts))
        print(f"  Step {t}: mean={vals.mean():.3f} std={vals.std():.3f}  {dist_str}")

    # Evidence overlap with fixed λ=0.7 (use oracle or recompute)
    if oracle is not None:
        # Compute DPO utility from oracle
        dpo_utils = []
        fixed_utils = []
        n_better = 0
        n_worse = 0
        n_total_with_utility = 0

        for r in results:
            eid = r["event_id"]
            if eid not in oracle:
                continue
            # DPO utility: we need to compute it. For now, use the closest single-λ
            # oracle entry as a proxy (this is approximate)
            oro = oracle[eid]
            fixed_u = oro.get("0.70")
            if fixed_u is None:
                continue

            n_total_with_utility += 1
            fixed_utils.append(fixed_u)

            # For DPO, use mean of the chosen lambdas to look up oracle
            if r["chosen_lambdas"]:
                mean_lam = np.mean(r["chosen_lambdas"])
                closest_key = min(oro.keys(), key=lambda k: abs(float(k) - mean_lam))
                dpo_u = oro.get(closest_key, fixed_u)
            else:
                dpo_u = fixed_u
            dpo_utils.append(dpo_u)

            if dpo_u > fixed_u:
                n_better += 1
            elif dpo_u < fixed_u:
                n_worse += 1

        if fixed_utils:
            dpo_arr = np.array(dpo_utils, dtype=np.float32)
            fixed_arr = np.array(fixed_utils, dtype=np.float32)
            print(f"\nUtility comparison (approximate, via oracle):")
            print(f"  DPO mean utility: {dpo_arr.mean():.4f}")
            print(f"  Fixed λ=0.7 mean utility: {fixed_arr.mean():.4f}")
            print(f"  Δ: {dpo_arr.mean() - fixed_arr.mean():.4f}")
            print(f"  Better: {n_better}  Worse: {n_worse}  Same: {n_total_with_utility - n_better - n_worse}")

            # Bucket analysis
            print(f"\nBucket analysis:")
            # By candidate count
            for lo, hi in [(1, 4), (5, 6), (7, 15), (16, 30), (31, 999)]:
                bucket_results = [r for r in results if lo <= r["n_candidates"] <= hi]
                if not bucket_results:
                    continue
                bucket_eids = {r["event_id"] for r in bucket_results}
                bucket_dpo = []
                bucket_fixed = []
                for r in bucket_results:
                    eid = r["event_id"]
                    if eid not in oracle:
                        continue
                    oro = oracle[eid]
                    fu = oro.get("0.70")
                    if fu is None:
                        continue
                    bucket_fixed.append(fu)
                    if r["chosen_lambdas"]:
                        mean_lam = np.mean(r["chosen_lambdas"])
                        closest = min(oro.keys(), key=lambda k: abs(float(k) - mean_lam))
                        bucket_dpo.append(oro.get(closest, fu))
                    else:
                        bucket_dpo.append(fu)
                if bucket_dpo:
                    dpo_m = np.mean(bucket_dpo)
                    fixed_m = np.mean(bucket_fixed)
                    print(f"  candidates [{lo}-{hi}]: n={len(bucket_eids)} DPO={dpo_m:.4f} Fixed={fixed_m:.4f} Δ={dpo_m-fixed_m:.4f}")

    # Save results
    results_path = output_dir / f"dpo_selection_{args.split_name}.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved selection results to {results_path}")

    # Save summary
    summary = {
        "split": args.split_name,
        "n_samples": len(results),
        "policy_path": args.policy,
        "inference_mode": args.inference_mode,
        "lambda_overall_mean": float(lam_arr.mean()),
        "lambda_overall_std": float(lam_arr.std()),
        "lambda_by_step": {
            str(t): {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "counts": {f"{u:.1f}": int(c) for u, c in zip(*np.unique(vals, return_counts=True))},
            }
            for t, vals in sorted(lambda_dist_by_step.items())
        },
    }
    summary_path = output_dir / f"eval_summary_{args.split_name}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved eval summary to {summary_path}")

    print(f"\n=== Full Pipeline Evaluation ===")
    print(f"To run the complete build → train → infer pipeline with this policy:")
    print(f"")
    print(f"  PYTHONPATH=src python -m fact_checking.pipeline.run \\")
    print(f"    experiment=mmr_dpo_step_lambda pipeline.mode=full")
    print(f"")
    print(f"Or step-by-step:")
    print(f"  PYTHONPATH=src python -m fact_checking.pipeline.run \\")
    print(f"    experiment=mmr_dpo_step_lambda pipeline.mode=build")
    print(f"  PYTHONPATH=src python -m fact_checking.pipeline.run \\")
    print(f"    experiment=mmr_dpo_step_lambda pipeline.mode=train")
    print(f"  PYTHONPATH=src python -m fact_checking.pipeline.run \\")
    print(f"    experiment=mmr_dpo_step_lambda pipeline.mode=infer")


if __name__ == "__main__":
    main()
