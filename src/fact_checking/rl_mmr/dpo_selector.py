"""Build-pipeline integration for DPO step-wise λ policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fact_checking.build.candidates import ChunkMMRSample, canonicalize_sentence, compute_hybrid_scores
from fact_checking.rl_mmr.dpo_policy import StepLambdaPolicy
from fact_checking.rl_mmr.step_features import (
    ALL_FEATURE_NAMES,
    extract_episode_features,
)
from fact_checking.retrieval.mmr import maximal_marginal_relevance_stepwise


def load_dpo_step_policy(model_path: str | Path) -> tuple[StepLambdaPolicy, dict[str, Any]]:
    """Load a trained DPO step-wise λ policy and its feature stats."""
    model_dir = Path(model_path)
    stats_path = model_dir / "feature_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"feature_stats.json not found in {model_dir}")
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    model_file = model_dir / "model_best.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    input_dim = int(stats["input_dim"])
    hidden_dims = stats.get("hidden_dims", [64, 32])
    dropout = float(stats.get("dropout", 0.1))
    n_actions = int(stats["n_actions"])

    policy = StepLambdaPolicy(
        input_dim=input_dim, hidden_dims=hidden_dims,
        dropout=dropout, n_actions=n_actions,
    )
    policy.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    policy.eval()
    return policy, stats


def normalize_features(features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    if features.shape[1] != mean.shape[0]:
        raise ValueError(f"Feature dimension mismatch: {features.shape[1]} vs {mean.shape[0]}")
    return ((features - mean) / std).astype(np.float32, copy=False)


def select_candidates_dpo_stepwise(
    sample: ChunkMMRSample,
    policy: StepLambdaPolicy,
    feature_stats: dict[str, Any],
    lambda_grid: np.ndarray,
    *,
    top_k: int = 5,
    alpha_dense: float = 0.70,
    alpha_lexical: float = 0.20,
    alpha_bm25: float = 0.10,
    inference_mode: str = "argmax",
    sample_temperature: float = 0.5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run step-wise DPO MMR selection for a single sample.

    Returns the same dict format as ``_select_candidates_from_chunk_sample``.
    """
    scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
    n = int(scored["n"])
    if n == 0:
        return {
            "event_id": sample.event_id, "claim": sample.claim,
            "label": sample.label, "explain": sample.explain, "candidates": [],
        }

    hybrid_scores = scored["hybrid_scores"]
    chunk_emb = scored["chunk_emb"]
    dense_scores = scored["dense_scores"]
    lexical_scores = scored["lexical_scores"]
    bm25_scores = scored["bm25_scores"]

    n_actions = len(lambda_grid)
    effective_k = min(int(top_k), n)

    # Run step-wise MMR with policy-guided λ selection
    selected_indices: list[int] = []
    candidate_mask = np.ones(n, dtype=bool)
    max_sim_to_selected = np.zeros(n, dtype=np.float32)
    similarity = chunk_emb @ chunk_emb.T
    chosen_lambdas: list[float] = []

    rng = np.random.default_rng(random_seed)

    for t in range(effective_k):
        # Build state features for current step
        step_records_temp = [{
            "step_idx": t,
            "selected_idx": 0,  # placeholder
            "candidate_mask_before": candidate_mask.copy(),
            "mmr_scores_before": None,
        }]
        # Compute MMR scores for all candidates
        if not selected_indices:
            mmr_scores = hybrid_scores.copy()
        else:
            lam_temp = lambda_grid[len(lambda_grid) // 2]  # dummy for feature extraction
            mmr_scores = lam_temp * hybrid_scores - (1.0 - lam_temp) * max_sim_to_selected
        mmr_scores[~candidate_mask] = -np.inf
        step_records_temp[0]["mmr_scores_before"] = mmr_scores.copy()

        feats = extract_episode_features(hybrid_scores, chunk_emb, step_records_temp, effective_k)
        state = normalize_features(feats[0].reshape(1, -1), feature_stats)

        # Get policy prediction
        policy.eval()
        with torch.no_grad():
            logits = policy(torch.from_numpy(state))
            probs = torch.softmax(logits, dim=-1).numpy().reshape(-1)

        mode = str(inference_mode).strip().lower()
        if mode == "argmax":
            action = int(np.argmax(probs))
        elif mode == "sample":
            temp = max(float(sample_temperature), 1e-6)
            scaled = np.log(np.clip(probs, 1e-8, 1.0)) / temp
            scaled -= scaled.max()
            sample_probs = np.exp(scaled) / np.exp(scaled).sum()
            action = int(rng.choice(n_actions, p=sample_probs))
        else:
            action = int(np.argmax(probs))

        lam_t = float(lambda_grid[action])
        chosen_lambdas.append(lam_t)

        # MMR selection with chosen λ
        if not selected_indices:
            mmr_scores = hybrid_scores.copy()
        else:
            mmr_scores = lam_t * hybrid_scores - (1.0 - lam_t) * max_sim_to_selected
        mmr_scores[~candidate_mask] = -np.inf
        best_idx = int(np.argmax(mmr_scores))

        selected_indices.append(best_idx)
        candidate_mask[best_idx] = False
        np.maximum(max_sim_to_selected, similarity[best_idx, :], out=max_sim_to_selected)

    # Build candidate output (same format as _select_candidates_from_chunk_sample)
    candidates: list[dict[str, Any]] = []
    for idx in selected_indices:
        candidate = dict(sample.candidates[int(idx)])
        candidate.update({
            "dense_score": float(dense_scores[idx]),
            "lexical_score": float(lexical_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
        })
        candidates.append(candidate)

    # Dedup by text (same logic as original)
    deduped: dict[str, dict[str, Any]] = {}
    for c in candidates:
        dedup_key = canonicalize_sentence(str(c.get("text", "")))
        old = deduped.get(dedup_key)
        if old is None or c["hybrid_score"] > old["hybrid_score"]:
            deduped[dedup_key] = c
    deduped_list = list(deduped.values())
    deduped_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
    deduped_list = deduped_list[:effective_k]

    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": deduped_list,
        "_dpo_chosen_lambdas": chosen_lambdas,
    }


def run_dpo_stepwise_selection(
    chunk_samples: list[ChunkMMRSample],
    *,
    learned_lambda_cfg: dict[str, Any],
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Run DPO step-wise selection for all samples.

    Returns (lambda_overrides, trace_rows, summary).
    lambda_overrides maps event_id → dummy lambda (for backward compat, always 0.7
    since the actual per-step λ is applied inside select_candidates_dpo_stepwise).
    """
    dpo_cfg = dict(learned_lambda_cfg.get("dpo_stepwise", {}) or {})
    model_path = str(dpo_cfg.get("model_path", "outputs/rl_mmr/dpo_stepwise/checkpoints"))
    lambda_grid = np.array(dpo_cfg.get("lambda_grid", [0.1, 0.3, 0.5, 0.7, 0.9]), dtype=np.float32)
    inference_mode = str(dpo_cfg.get("inference_mode", "argmax")).strip().lower()

    policy, feature_stats = load_dpo_step_policy(model_path)

    trace_rows: list[dict[str, Any]] = []
    chosen_lambdas_flat: list[float] = []

    for sample in chunk_samples:
        row = select_candidates_dpo_stepwise(
            sample, policy, feature_stats, lambda_grid,
            top_k=top_k, alpha_dense=alpha_dense, alpha_lexical=alpha_lexical, alpha_bm25=alpha_bm25,
            inference_mode=inference_mode, sample_temperature=float(dpo_cfg.get("sample_temperature", 0.5)),
        )
        chosen = row.pop("_dpo_chosen_lambdas", [])
        trace_rows.append({
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "chosen_lambdas": chosen,
            "n_candidates": len(sample.candidates),
            "inference_mode": inference_mode,
        })
        chosen_lambdas_flat.extend(chosen)

    # Return dummy lambda_overrides (the real δ values are per-step, applied above)
    lambda_overrides = {s.event_id: 0.70 for s in chunk_samples}

    vals = np.array(chosen_lambdas_flat, dtype=np.float32)
    summary = {
        "num_samples": len(chunk_samples),
        "model_path": model_path,
        "lambda_grid": [float(x) for x in lambda_grid.tolist()],
        "inference_mode": inference_mode,
        "chosen_lambda_mean": float(vals.mean()) if vals.size else 0.0,
        "chosen_lambda_std": float(vals.std()) if vals.size else 0.0,
        "config": dpo_cfg,
    }
    return lambda_overrides, trace_rows, summary


def dump_trace_rows(trace_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
