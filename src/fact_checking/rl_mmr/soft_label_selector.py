"""Build-pipeline integration for soft-label lambda policies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fact_checking.build.candidates import ChunkMMRSample, compute_hybrid_scores
from fact_checking.rl_mmr.soft_label_features import (
    DEFAULT_LAMBDA_GRID,
    SOFT_LABEL_FEATURE_NAMES,
    extract_soft_label_features,
    feature_dict_to_vector,
)


class SoftLabelMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, n_classes)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, n_classes), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoftLabelRegressorEnsemble:
    """One-vs-rest regressors trained to predict a soft target distribution."""

    def __init__(self, estimators: list[Any], *, backend: str):
        self.estimators = estimators
        self.backend = backend

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for estimator in self.estimators:
            pred = np.asarray(estimator.predict(x), dtype=np.float32).reshape(-1)
            cols.append(pred)
        raw = np.stack(cols, axis=1)
        return _normalize_prob_rows(raw)


@dataclass
class SoftLabelPolicy:
    model: Any
    stats: dict[str, Any]
    model_type: str
    lambda_grid: np.ndarray
    model_dir: Path


def _normalize_prob_rows(values: np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float32)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.clip(probs, 1e-8, None)
    denom = probs.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (probs / denom).astype(np.float32, copy=False)


def _parse_lambda_grid_from_stats(stats: dict[str, Any]) -> np.ndarray:
    values = stats.get("lambda_grid") or DEFAULT_LAMBDA_GRID
    return np.array([float(x) for x in values], dtype=np.float32)


def load_soft_label_policy(model_path: str | Path) -> SoftLabelPolicy:
    model_dir = Path(model_path)
    stats_path = model_dir / "feature_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"feature_stats.json not found under soft-label model dir: {model_dir}")
    with stats_path.open("r", encoding="utf-8") as reader:
        stats = json.load(reader)

    model_type = str(stats.get("model_type", "lr")).strip().lower()
    lambda_grid = _parse_lambda_grid_from_stats(stats)
    if model_type in {"lr", "lightgbm", "gbdt"}:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load sklearn/lightgbm soft-label policies.") from exc
        model_file = model_dir / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Soft-label model file not found: {model_file}")
        model = joblib.load(model_file)
    elif model_type == "mlp":
        model_file = model_dir / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Soft-label MLP checkpoint not found: {model_file}")
        model = SoftLabelMLP(
            input_dim=int(stats["input_dim"]),
            hidden_dim=int(stats.get("hidden_dim", 128)),
            dropout=float(stats.get("dropout", 0.1)),
            n_classes=len(lambda_grid),
        )
        model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
        model.eval()
    else:
        raise ValueError(f"Unsupported soft-label model_type={model_type!r}")

    return SoftLabelPolicy(
        model=model,
        stats=stats,
        model_type=model_type,
        lambda_grid=lambda_grid,
        model_dir=model_dir,
    )


def normalize_features_for_policy(features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    if features.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: extracted {features.shape[1]}, policy expects {mean.shape[0]}."
        )
    std = np.where(std < 1e-8, 1.0, std)
    return ((features - mean) / std).astype(np.float32, copy=False)


def predict_policy_proba(policy: SoftLabelPolicy, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    n_classes = len(policy.lambda_grid)
    if policy.model_type == "mlp":
        with torch.no_grad():
            logits = policy.model(torch.from_numpy(x))
            probs = torch.softmax(logits, dim=-1).numpy()
        return _normalize_prob_rows(probs)

    if hasattr(policy.model, "predict_proba"):
        raw = np.asarray(policy.model.predict_proba(x), dtype=np.float32)
        classes = getattr(policy.model, "classes_", None)
        if classes is not None and raw.shape[1] != n_classes:
            full = np.zeros((raw.shape[0], n_classes), dtype=np.float32)
            for col_idx, cls in enumerate(classes):
                cls_idx = int(cls)
                if 0 <= cls_idx < n_classes:
                    full[:, cls_idx] = raw[:, col_idx]
            raw = full
        return _normalize_prob_rows(raw)

    raise TypeError(f"Loaded soft-label policy does not expose predict_proba(): {type(policy.model)}")


def select_lambdas_from_probs(
    probs: np.ndarray,
    lambda_grid: np.ndarray,
    *,
    inference_mode: str = "argmax",
    sample_temperature: float = 0.5,
    random_seed: int = 42,
) -> np.ndarray:
    mode = str(inference_mode).strip().lower()
    grid = np.asarray(lambda_grid, dtype=np.float32)
    p = _normalize_prob_rows(probs)
    if mode == "argmax":
        return grid[np.argmax(p, axis=1)].astype(np.float32, copy=False)
    if mode == "expected":
        return (p * grid[None, :]).sum(axis=1).astype(np.float32, copy=False)
    if mode == "sample":
        temp = max(float(sample_temperature), 1e-6)
        scaled = np.log(np.clip(p, 1e-8, 1.0)) / temp
        scaled -= scaled.max(axis=1, keepdims=True)
        sample_probs = _normalize_prob_rows(np.exp(scaled))
        rng = np.random.default_rng(int(random_seed))
        choices = [rng.choice(len(grid), p=sample_probs[i]) for i in range(sample_probs.shape[0])]
        return grid[np.array(choices, dtype=np.int64)].astype(np.float32, copy=False)
    raise ValueError("inference_mode must be one of: argmax, expected, sample")


def _resolve_soft_label_cfg(learned_lambda_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(learned_lambda_cfg.get("soft_label", {}) or {})
    if "model_path" not in cfg and learned_lambda_cfg.get("model_path"):
        cfg["model_path"] = learned_lambda_cfg["model_path"]
    return {
        "model_path": str(cfg.get("model_path", "outputs/rl_mmr/soft_label/lightgbm")),
        "lambda_grid": [float(x) for x in cfg.get("lambda_grid", DEFAULT_LAMBDA_GRID)],
        "inference_mode": str(cfg.get("inference_mode", "argmax")).strip().lower(),
        "sample_temperature": float(cfg.get("sample_temperature", 0.5)),
        "random_seed": int(cfg.get("random_seed", 42)),
        "top_k": int(cfg.get("top_k", 5)),
        "dump_trace": bool(cfg.get("dump_trace", True)),
    }


def build_lambda_overrides_from_soft_label(
    chunk_samples: list[ChunkMMRSample],
    *,
    learned_lambda_cfg: dict[str, Any],
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    top_k: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Predict per-sample MMR lambda overrides from a trained soft-label policy."""
    cfg = _resolve_soft_label_cfg(learned_lambda_cfg)
    if top_k is not None:
        cfg["top_k"] = int(top_k)
    policy = load_soft_label_policy(cfg["model_path"])

    raw_vectors: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    event_ids: list[str] = []
    for sample in chunk_samples:
        scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
        feats = extract_soft_label_features(
            sample,
            scored["hybrid_scores"],
            scored["chunk_emb"],
            lambda_grid=tuple(float(x) for x in policy.lambda_grid.tolist()),
            top_k=int(cfg["top_k"]),
        )
        raw_vectors.append(feature_dict_to_vector(feats, list(policy.stats.get("feature_names") or SOFT_LABEL_FEATURE_NAMES)))
        feature_rows.append(feats)
        event_ids.append(str(sample.event_id))

    if not raw_vectors:
        return {}, [], {"num_samples": 0, "config": cfg}

    raw_features = np.stack(raw_vectors).astype(np.float32, copy=False)
    features = normalize_features_for_policy(raw_features, policy.stats)
    probs = predict_policy_proba(policy, features)
    chosen = select_lambdas_from_probs(
        probs,
        policy.lambda_grid,
        inference_mode=cfg["inference_mode"],
        sample_temperature=cfg["sample_temperature"],
        random_seed=cfg["random_seed"],
    )

    lambda_overrides = {event_ids[i]: float(chosen[i]) for i in range(len(event_ids))}
    pred_entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    argmax_idx = np.argmax(probs, axis=1)
    argmax_counts = {
        f"{float(policy.lambda_grid[i]):.2f}": int((argmax_idx == i).sum())
        for i in range(len(policy.lambda_grid))
    }

    trace_rows: list[dict[str, Any]] = []
    for i, sample in enumerate(chunk_samples):
        probs_by_lambda = {
            f"{float(policy.lambda_grid[j]):.2f}": float(probs[i, j])
            for j in range(len(policy.lambda_grid))
        }
        feats = feature_rows[i]
        trace_rows.append({
            "event_id": event_ids[i],
            "claim": sample.claim,
            "label": sample.label,
            "n_candidates": int(float(feats.get("n_candidates", 0.0))),
            "pool_redundancy": float(feats.get("pool_redundancy", 0.0)),
            "sens_0p30_0p70": float(feats.get("sens_0p30_0p70", 0.0)),
            "prediction_entropy": float(pred_entropy[i]),
            "argmax_lambda": float(policy.lambda_grid[argmax_idx[i]]),
            "chosen_lambda": float(chosen[i]),
            "inference_mode": cfg["inference_mode"],
            "probs_by_lambda": probs_by_lambda,
        })

    chosen_values = np.array(list(lambda_overrides.values()), dtype=np.float32)
    summary = {
        "num_samples": len(event_ids),
        "model_path": cfg["model_path"],
        "model_type": policy.model_type,
        "lambda_grid": [float(x) for x in policy.lambda_grid.tolist()],
        "inference_mode": cfg["inference_mode"],
        "chosen_lambda_mean": float(chosen_values.mean()) if chosen_values.size else 0.0,
        "chosen_lambda_std": float(chosen_values.std()) if chosen_values.size else 0.0,
        "prediction_entropy_mean": float(pred_entropy.mean()) if pred_entropy.size else 0.0,
        "argmax_counts": argmax_counts,
        "config": cfg,
    }
    return lambda_overrides, trace_rows, summary


def dump_trace_rows(trace_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for row in trace_rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
