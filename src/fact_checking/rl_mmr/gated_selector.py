"""Build-pipeline entry for sensitivity-gated MMR.

Given a list of cached ``ChunkMMRSample`` (the same artifact the standard MMR
phase consumes), this module produces:

* a ``lambda_overrides`` dict mapping each ``event_id`` to its gated lambda
* a list of per-sample trace rows ready for JSONL dump

The hybrid relevance score is delegated to
``fact_checking.build.candidates.compute_hybrid_scores`` so the gating logic
sees exactly the same scores that the final MMR selection will see.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, compute_hybrid_scores
from fact_checking.rl_mmr.sensitivity import (
    GATE_TRIVIAL,
    gating_decision,
    sensitivity_features,
)


DEFAULT_LAMBDA_LOW = 0.30
DEFAULT_LAMBDA_BASE = 0.70
DEFAULT_LAMBDA_PROBE = 1.00
DEFAULT_THETA_S = 0.40
DEFAULT_THETA_R = 0.40
DEFAULT_GATING_MODE = "conservative"


def _resolve_sensitivity_cfg(learned_lambda_cfg: dict[str, Any]) -> dict[str, Any]:
    sens_cfg = dict(learned_lambda_cfg.get("sensitivity", {}) or {})
    floor_cfg = dict(sens_cfg.get("relevance_floor", {}) or {})
    return {
        "lambda_low": float(sens_cfg.get("lambda_low", DEFAULT_LAMBDA_LOW)),
        "lambda_base": float(sens_cfg.get("lambda_base", DEFAULT_LAMBDA_BASE)),
        "lambda_probe": float(sens_cfg.get("lambda_probe", DEFAULT_LAMBDA_PROBE)),
        "theta_s": float(sens_cfg.get("theta_s", DEFAULT_THETA_S)),
        "theta_r": float(sens_cfg.get("theta_r", DEFAULT_THETA_R)),
        "gating_mode": str(sens_cfg.get("gating_mode", DEFAULT_GATING_MODE)).strip().lower(),
        "top_k": int(sens_cfg.get("top_k", 5)),
        "pool_redundancy_topn": int(sens_cfg.get("pool_redundancy_topn", 32)),
        "min_n_candidates_for_gate": int(sens_cfg.get("min_n_candidates_for_gate", 2)),
        "relevance_floor": {
            "mode": str(floor_cfg.get("mode", "mean_delta")).strip().lower(),
            "epsilon": float(floor_cfg.get("epsilon", 0.05)),
            "p_floor": float(floor_cfg.get("p_floor", 0.50)),
        },
        "dump_trace": bool(sens_cfg.get("dump_trace", True)),
    }


def build_lambda_overrides_from_sensitivity(
    chunk_samples: list[ChunkMMRSample],
    *,
    learned_lambda_cfg: dict[str, Any],
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    top_k: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Compute per-sample gated lambda and a trace row for each sample.

    Args:
        chunk_samples: cached ``ChunkMMRSample`` list for a single split.
        learned_lambda_cfg: the ``retrieval.learned_lambda`` subtree.
        alpha_dense / alpha_lexical / alpha_bm25: same coefficients used by
            the production MMR phase.
        top_k: final evidence budget. If provided, overrides the value in
            ``learned_lambda.sensitivity.top_k`` (so the build pipeline can
            pass ``retrieval.top_k`` directly).

    Returns:
        ``(lambda_overrides, trace_rows, summary)``.
    """
    cfg = _resolve_sensitivity_cfg(learned_lambda_cfg)
    if top_k is not None:
        cfg["top_k"] = int(top_k)

    lambdas = (cfg["lambda_low"], cfg["lambda_base"], cfg["lambda_probe"])

    lambda_overrides: dict[str, float] = {}
    trace_rows: list[dict[str, Any]] = []
    gate_counts: dict[str, int] = {}
    chosen_values: list[float] = []
    sens_values: list[float] = []
    pool_red_values: list[float] = []

    for sample in chunk_samples:
        scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
        hybrid_scores: np.ndarray = scored["hybrid_scores"]
        chunk_emb: np.ndarray = scored["chunk_emb"]
        n = int(scored["n"])

        if n == 0:
            chosen = float(cfg["lambda_base"])
            gate = GATE_TRIVIAL
            extras: dict[str, Any] = {
                "theta_s": cfg["theta_s"],
                "theta_r": cfg["theta_r"],
                "lambda_low_cfg": cfg["lambda_low"],
                "lambda_base_cfg": cfg["lambda_base"],
                "gating_mode": cfg["gating_mode"],
                "relevance_floor_ok": None,
            }
            feats: dict[str, Any] = {
                "n_candidates": 0,
                "top_k": int(cfg["top_k"]),
                "lambda_low": cfg["lambda_low"],
                "lambda_base": cfg["lambda_base"],
                "lambda_probe": cfg["lambda_probe"],
                "sens_low_base": 0.0,
                "sens_base_probe": 0.0,
                "sens_low_probe": 0.0,
                "pool_redundancy": 0.0,
                "max_pool_redundancy": 0.0,
                "selected_redundancy_low": 0.0,
                "selected_redundancy_base": 0.0,
                "mean_rel_low": 0.0,
                "mean_rel_base": 0.0,
                "score_entropy": 0.0,
                "top1_top2_gap": 0.0,
                "top5_score_std": 0.0,
                "kendall_low_base": 1.0,
                "top1_change": 0,
                "overlap_size": 0,
                "S_low": [],
                "S_base": [],
                "S_probe": [],
            }
        else:
            feats = sensitivity_features(
                hybrid_scores,
                chunk_emb,
                top_k=cfg["top_k"],
                lambdas=lambdas,
                pool_redundancy_topn=cfg["pool_redundancy_topn"],
            )
            chosen, gate, extras = gating_decision(
                feats,
                hybrid_scores,
                gating_mode=cfg["gating_mode"],
                theta_s=cfg["theta_s"],
                theta_r=cfg["theta_r"],
                lambda_low=cfg["lambda_low"],
                lambda_base=cfg["lambda_base"],
                min_n_candidates_for_gate=cfg["min_n_candidates_for_gate"],
                relevance_floor_kwargs=cfg["relevance_floor"],
            )

        event_id = str(sample.event_id)
        lambda_overrides[event_id] = float(chosen)
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        chosen_values.append(float(chosen))
        sens_values.append(float(feats.get("sens_low_base", 0.0)))
        pool_red_values.append(float(feats.get("pool_redundancy", 0.0)))

        trace_rows.append(
            {
                "event_id": event_id,
                "claim": sample.claim,
                "label": sample.label,
                "n_candidates": feats["n_candidates"],
                "top_k": feats["top_k"],
                "S_low": feats["S_low"],
                "S_base": feats["S_base"],
                "S_probe": feats["S_probe"],
                "sens_low_base": feats["sens_low_base"],
                "sens_base_probe": feats["sens_base_probe"],
                "sens_low_probe": feats["sens_low_probe"],
                "pool_redundancy": feats["pool_redundancy"],
                "max_pool_redundancy": feats["max_pool_redundancy"],
                "selected_redundancy_low": feats["selected_redundancy_low"],
                "selected_redundancy_base": feats["selected_redundancy_base"],
                "mean_rel_low": feats["mean_rel_low"],
                "mean_rel_base": feats["mean_rel_base"],
                "score_entropy": feats["score_entropy"],
                "top1_top2_gap": feats["top1_top2_gap"],
                "top5_score_std": feats["top5_score_std"],
                "kendall_low_base": feats["kendall_low_base"],
                "top1_change": feats["top1_change"],
                "overlap_size": feats["overlap_size"],
                "gate": gate,
                "chosen_lambda": float(chosen),
                "relevance_floor_ok": extras.get("relevance_floor_ok"),
                "lambda_low_cfg": extras["lambda_low_cfg"],
                "lambda_base_cfg": extras["lambda_base_cfg"],
                "theta_s": extras["theta_s"],
                "theta_r": extras["theta_r"],
                "gating_mode": extras["gating_mode"],
            }
        )

    summary = {
        "num_samples": len(chunk_samples),
        "gate_counts": gate_counts,
        "chosen_lambda_mean": float(np.mean(chosen_values)) if chosen_values else 0.0,
        "chosen_lambda_std": float(np.std(chosen_values)) if chosen_values else 0.0,
        "sens_low_base_mean": float(np.mean(sens_values)) if sens_values else 0.0,
        "pool_redundancy_mean": float(np.mean(pool_red_values)) if pool_red_values else 0.0,
        "config": cfg,
    }
    return lambda_overrides, trace_rows, summary


def dump_trace_rows(trace_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for row in trace_rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
