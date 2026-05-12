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


def save_predictor(
    model: LambdaPredictor,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "predictor.pt")
    stats = {
        "feature_names": FEATURE_NAMES,
        "mean": feature_mean.tolist(),
        "std": feature_std.tolist(),
        "input_dim": len(FEATURE_NAMES),
    }
    with (output_dir / "feature_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)


def load_predictor(
    model_path: str | Path,
    stats_path: str | Path,
    hidden_dim: int = 64,
    dropout: float = 0.1,
) -> tuple[LambdaPredictor, dict[str, Any]]:
    with Path(stats_path).open() as f:
        stats = json.load(f)
    input_dim = stats["input_dim"]
    model = LambdaPredictor(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, stats


def predict_lambdas_for_samples(
    pre_samples: list[PreMMRSample],
    model: LambdaPredictor,
    stats: dict[str, Any],
    retrieval_cfg: dict[str, Any],
) -> dict[str, float]:
    alpha_dense = float(retrieval_cfg.get("alpha_dense", 0.70))
    alpha_lexical = float(retrieval_cfg.get("alpha_lexical", 0.20))
    alpha_bm25 = float(retrieval_cfg.get("alpha_bm25", 0.10))

    features = extract_features_batch(pre_samples, alpha_dense, alpha_lexical, alpha_bm25)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    features_norm = (features - mean) / std

    with torch.no_grad():
        x = torch.from_numpy(features_norm)
        predicted = model(x).numpy()

    return {pre.event_id: float(predicted[i]) for i, pre in enumerate(pre_samples)}
