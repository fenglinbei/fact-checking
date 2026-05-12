from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fact_checking.build.candidates import PreMMRSample
from fact_checking.learned_lambda.features import (
    FEATURE_NAMES,
    extract_features_batch,
)


class LambdaPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class LambdaClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lambda_grid: list[float] | np.ndarray | None = None,
    ):
        super().__init__()
        grid = np.array(lambda_grid if lambda_grid is not None else np.linspace(0.0, 1.0, 11), dtype=np.float32)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(grid)),
        )
        self.register_buffer("lambda_grid", torch.from_numpy(grid))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.forward_logits(x), dim=-1)
        return torch.sum(probs * self.lambda_grid.to(device=x.device, dtype=probs.dtype), dim=-1)


def save_predictor(
    model: LambdaPredictor | LambdaClassifier,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    output_dir: str | Path,
    *,
    model_type: str = "regression",
    lambda_grid: list[float] | np.ndarray | None = None,
    hidden_dim: int = 64,
    dropout: float = 0.1,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "predictor.pt")
    stats = {
        "feature_names": FEATURE_NAMES,
        "mean": feature_mean.tolist(),
        "std": feature_std.tolist(),
        "input_dim": len(FEATURE_NAMES),
        "model_type": model_type,
        "lambda_grid": [float(x) for x in np.array(lambda_grid, dtype=float).tolist()] if lambda_grid is not None else None,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
    }
    with (output_dir / "feature_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)


def load_predictor(
    model_path: str | Path,
    stats_path: str | Path,
    hidden_dim: int | None = None,
    dropout: float | None = None,
) -> tuple[LambdaPredictor | LambdaClassifier, dict[str, Any]]:
    with Path(stats_path).open() as f:
        stats = json.load(f)
    input_dim = stats["input_dim"]
    resolved_hidden_dim = int(hidden_dim if hidden_dim is not None else stats.get("hidden_dim", 64))
    resolved_dropout = float(dropout if dropout is not None else stats.get("dropout", 0.1))
    model_type = str(stats.get("model_type") or "regression").strip().lower()
    if model_type == "classifier":
        model = LambdaClassifier(
            input_dim,
            hidden_dim=resolved_hidden_dim,
            dropout=resolved_dropout,
            lambda_grid=stats.get("lambda_grid"),
        )
    else:
        model = LambdaPredictor(input_dim, hidden_dim=resolved_hidden_dim, dropout=resolved_dropout)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, stats


def normalize_features_for_stats(features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    input_dim = int(stats["input_dim"])
    if features.shape[1] != input_dim:
        saved_names = list(stats.get("feature_names") or [])
        if features.shape[1] > input_dim and saved_names == FEATURE_NAMES[:input_dim]:
            features = features[:, :input_dim]
        else:
            raise ValueError(
                f"Feature dimension mismatch: extracted {features.shape[1]} features, "
                f"but predictor expects {input_dim}."
            )

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return (features - mean) / std


def predict_lambdas_for_samples(
    pre_samples: list[PreMMRSample],
    model: LambdaPredictor | LambdaClassifier,
    stats: dict[str, Any],
    retrieval_cfg: dict[str, Any],
) -> dict[str, float]:
    alpha_dense = float(retrieval_cfg.get("alpha_dense", 0.70))
    alpha_lexical = float(retrieval_cfg.get("alpha_lexical", 0.20))
    alpha_bm25 = float(retrieval_cfg.get("alpha_bm25", 0.10))

    features = extract_features_batch(pre_samples, alpha_dense, alpha_lexical, alpha_bm25)
    features_norm = normalize_features_for_stats(features, stats)

    with torch.no_grad():
        x = torch.from_numpy(features_norm)
        predicted = model(x).numpy()

    return {pre.event_id: float(predicted[i]) for i, pre in enumerate(pre_samples)}
