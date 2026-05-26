"""Generate MMR trajectories for DPO step-wise λ training.

Reads Chunk-MMR cache, runs step-wise MMR with handcrafted and random λ
schedules, and outputs per-trajectory JSONL records.

Usage:
    PYTHONPATH=src python scripts/phase6_rl_mmr/generate_trajectories.py \\
        --chunk-mmr-cache outputs/cache/chunk_mmr/<fp>/chunk_mmr_train.pkl \\
        --output-dir outputs/rl_mmr/dpo_stepwise/trajectories \\
        --top-k 5 --n-random 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.build.candidates import _load_pickle, compute_hybrid_scores
from fact_checking.learned_lambda.cache_utils import (
    load_experiment_build_cfg,
    pick_retrieval_value,
    resolve_chunk_mmr_cache_path,
)
from fact_checking.rl_mmr.step_features import extract_episode_features
from fact_checking.rl_mmr.trajectory import (
    HANDCRAFTED_SCHEDULES,
    LAMBDA_GRID,
    Trajectory,
    generate_random_schedules,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MMR trajectories for DPO training.")
    p.add_argument("--chunk-mmr-cache", type=str, default=None,
                   help="Path to Chunk-MMR cache pickle. Auto-resolved from experiment if not given.")
    p.add_argument("--chunk-mmr-cache-root", type=str, default="outputs/cache/chunk_mmr")
    p.add_argument("--experiment", type=str, default="b3_mmr_topk_sweep_1024")
    p.add_argument("--config-overrides", nargs="*", default=[])
    p.add_argument("--split-name", type=str, default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--alpha-dense", type=float, default=None)
    p.add_argument("--alpha-lexical", type=float, default=None)
    p.add_argument("--alpha-bm25", type=float, default=None)
    p.add_argument("--n-random", type=int, default=30,
                   help="Number of random schedules per claim.")
    p.add_argument("--lambda-grid", nargs="*", type=float, default=LAMBDA_GRID)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--include-state-features", action="store_true", default=True,
                   help="Include per-step state feature vectors in output.")
    return p.parse_args()


def _trajectory_to_dict(traj: Trajectory, include_state_features: bool = True) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "event_id": traj.event_id,
        "claim": traj.claim,
        "gold_label": traj.gold_label,
        "schedule_type": traj.schedule_type,
        "lambda_schedule": traj.lambda_schedule,
        "selected_ids": traj.selected_ids,
        "evidence_set_key": traj.evidence_set_key,
        "steps": [
            {
                "step_idx": s.step_idx,
                "lambda_val": s.lambda_val,
                "selected_idx": s.selected_idx,
                "hybrid_score": s.hybrid_score,
                "max_sim_to_selected": s.max_sim_to_selected,
                "mmr_score": s.mmr_score,
            }
            for s in traj.steps
        ],
    }
    if traj.utility is not None:
        rec["utility"] = traj.utility
    if include_state_features and traj.state_features is not None:
        rec["state_features"] = [f.tolist() for f in traj.state_features]
    return rec


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress

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

    print(f"Chunk-MMR cache: {cache_path}")
    print(f"top_k={top_k} alpha=({alpha_dense:.2f},{alpha_lexical:.2f},{alpha_bm25:.2f})")
    print(f"n_random={args.n_random} seed={args.seed}")

    chunk_samples = load_pickle(cache_path)
    if args.sample_limit:
        chunk_samples = chunk_samples[:args.sample_limit]
    print(f"Loaded {len(chunk_samples)} ChunkMMR samples")

    random_schedules = generate_random_schedules(
        n_schedules=args.n_random, top_k=top_k,
        lambda_grid=args.lambda_grid, seed=args.seed,
    )
    all_schedules: list[tuple[list[float], str]] = []
    for s in HANDCRAFTED_SCHEDULES:
        all_schedules.append((s[:top_k], "handcrafted"))
    for s in random_schedules:
        all_schedules.append((s, "random"))
    print(f"Total schedules: {len(all_schedules)} ({len(HANDCRAFTED_SCHEDULES)} handcrafted + {len(random_schedules)} random)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trajectories_{args.split_name}.jsonl"

    n_trajectories = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for sample in tqdm(
            chunk_samples, desc="generate trajectories", unit="claim",
            dynamic_ncols=True, disable=not show_progress,
        ):
            scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
            hybrid_scores = scored["hybrid_scores"]
            chunk_emb = scored["chunk_emb"]

            for sched, sched_type in all_schedules:
                traj = Trajectory.from_chunk_sample(
                    sample, sched, sched_type,
                    alpha_dense=alpha_dense, alpha_lexical=alpha_lexical, alpha_bm25=alpha_bm25,
                )
                if not traj.steps:
                    continue

                if args.include_state_features:
                    step_records = [
                        {
                            "step_idx": s.step_idx,
                            "lambda_val": s.lambda_val,
                            "selected_idx": s.selected_idx,
                            "hybrid_score": s.hybrid_score,
                            "max_sim_to_selected": s.max_sim_to_selected,
                            "mmr_score": s.mmr_score,
                            "candidate_mask_before": None,
                            "mmr_scores_before": None,
                        }
                        for s in traj.steps
                    ]
                    # Re-run with proper records to get candidate_mask_before
                    from fact_checking.retrieval.mmr import maximal_marginal_relevance_stepwise
                    _, real_records = maximal_marginal_relevance_stepwise(
                        query_scores=hybrid_scores,
                        sentence_vectors=chunk_emb,
                        lambda_weights=sched[:min(top_k, int(scored["n"]))],
                    )
                    feats = extract_episode_features(
                        hybrid_scores, chunk_emb, real_records, top_k,
                        lambda_schedule=sched[:min(top_k, int(scored["n"]))],
                    )
                    traj.state_features = feats

                writer.write(json.dumps(_trajectory_to_dict(traj, args.include_state_features), ensure_ascii=False) + "\n")
                n_trajectories += 1

    print(f"Wrote {n_trajectories} trajectories to {output_path}")


if __name__ == "__main__":
    main()
