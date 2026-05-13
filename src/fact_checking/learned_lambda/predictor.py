from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fact_checking.build.candidates import ChunkMMRSample, PreMMRSample
from fact_checking.learned_lambda.embedding_features import (
    CHUNK_EMBEDDING_FEATURE_MODE,
    build_chunk_embedding_arrays,
)
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


class ChunkEmbeddingLambdaEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        encoder_dim: int = 256,
        dropout: float = 0.1,
        lambda_grid: list[float] | np.ndarray | None = None,
    ):
        super().__init__()
        self.claim_proj = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(
            nn.Linear(encoder_dim * 4, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(encoder_dim * 6, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 2, len(lambda_grid) if lambda_grid is not None else 1),
        )
        if lambda_grid is None:
            self.lambda_grid = None
        else:
            grid = np.array(lambda_grid, dtype=np.float32)
            self.register_buffer("lambda_grid", torch.from_numpy(grid))

    def _encode(
        self,
        claim_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = candidate_mask.to(dtype=torch.bool)
        claim_h = self.claim_proj(claim_emb)
        cand_h = self.candidate_proj(candidate_emb)

        mask_f = mask.unsqueeze(-1).to(dtype=cand_h.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = (cand_h * mask_f).sum(dim=1) / denom

        masked_cand = cand_h.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool = masked_cand.max(dim=1).values
        has_any = mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))

        claim_expanded = claim_h.unsqueeze(1).expand_as(cand_h)
        attn_input = torch.cat(
            [
                cand_h,
                claim_expanded,
                cand_h * claim_expanded,
                torch.abs(cand_h - claim_expanded),
            ],
            dim=-1,
        )
        attn_logits = self.attn(attn_input).squeeze(-1).masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(attn_logits, dim=1) * mask.to(dtype=claim_h.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        attn_pool = torch.sum(cand_h * attn_weights.unsqueeze(-1), dim=1)

        return torch.cat(
            [
                claim_h,
                attn_pool,
                mean_pool,
                max_pool,
                claim_h * attn_pool,
                torch.abs(claim_h - attn_pool),
            ],
            dim=-1,
        )

    def forward_logits(
        self,
        claim_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self._encode(claim_emb, candidate_emb, candidate_mask))

    def forward(
        self,
        claim_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward_logits(claim_emb, candidate_emb, candidate_mask)
        if self.lambda_grid is None:
            return torch.sigmoid(logits).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.lambda_grid.to(device=logits.device, dtype=probs.dtype), dim=-1)


def save_predictor(
    model: LambdaPredictor | LambdaClassifier | ChunkEmbeddingLambdaEncoder,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    output_dir: str | Path,
    *,
    model_type: str = "regression",
    lambda_grid: list[float] | np.ndarray | None = None,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    feature_mode: str = "handcrafted",
    feature_names: list[str] | None = None,
    input_dim: int | None = None,
    embedding_dim: int | None = None,
    candidate_top_k: int | None = None,
    retrieval_config: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "predictor.pt")
    names = FEATURE_NAMES if feature_names is None else feature_names
    stats = {
        "feature_mode": feature_mode,
        "feature_names": names,
        "mean": feature_mean.tolist() if feature_mean is not None else [],
        "std": feature_std.tolist() if feature_std is not None else [],
        "input_dim": int(input_dim if input_dim is not None else len(names)),
        "model_type": model_type,
        "lambda_grid": [float(x) for x in np.array(lambda_grid, dtype=float).tolist()] if lambda_grid is not None else None,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
        "candidate_top_k": int(candidate_top_k) if candidate_top_k is not None else None,
        "retrieval_config": retrieval_config or {},
    }
    with (output_dir / "feature_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)


def load_predictor(
    model_path: str | Path,
    stats_path: str | Path,
    hidden_dim: int | None = None,
    dropout: float | None = None,
) -> tuple[LambdaPredictor | LambdaClassifier | ChunkEmbeddingLambdaEncoder, dict[str, Any]]:
    with Path(stats_path).open() as f:
        stats = json.load(f)
    input_dim = stats["input_dim"]
    resolved_hidden_dim = int(hidden_dim if hidden_dim is not None else stats.get("hidden_dim", 64))
    resolved_dropout = float(dropout if dropout is not None else stats.get("dropout", 0.1))
    model_type = str(stats.get("model_type") or "regression").strip().lower()
    feature_mode = str(stats.get("feature_mode") or "handcrafted").strip().lower()
    if feature_mode == CHUNK_EMBEDDING_FEATURE_MODE:
        lambda_grid = stats.get("lambda_grid") if model_type == "classifier" else None
        model = ChunkEmbeddingLambdaEncoder(
            embedding_dim=int(stats["embedding_dim"]),
            encoder_dim=resolved_hidden_dim,
            dropout=resolved_dropout,
            lambda_grid=lambda_grid,
        )
    elif model_type == "classifier":
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
    feature_mode = str(stats.get("feature_mode") or "handcrafted").strip().lower()
    if feature_mode == CHUNK_EMBEDDING_FEATURE_MODE:
        return features

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
    samples: list[PreMMRSample] | list[ChunkMMRSample],
    model: LambdaPredictor | LambdaClassifier | ChunkEmbeddingLambdaEncoder,
    stats: dict[str, Any],
    retrieval_cfg: dict[str, Any],
) -> dict[str, float]:
    alpha_dense = float(retrieval_cfg.get("alpha_dense", 0.70))
    alpha_lexical = float(retrieval_cfg.get("alpha_lexical", 0.20))
    alpha_bm25 = float(retrieval_cfg.get("alpha_bm25", 0.10))

    feature_mode = str(stats.get("feature_mode") or "handcrafted").strip().lower()
    if feature_mode == CHUNK_EMBEDDING_FEATURE_MODE:
        candidate_top_k = int(stats.get("candidate_top_k") or retrieval_cfg.get("top_k", 16))
        arrays = build_chunk_embedding_arrays(
            samples,  # type: ignore[arg-type]
            candidate_top_k=candidate_top_k,
            alpha_dense=alpha_dense,
            alpha_lexical=alpha_lexical,
            alpha_bm25=alpha_bm25,
        )
        with torch.no_grad():
            predicted = model(
                torch.from_numpy(arrays["claim_emb"]),
                torch.from_numpy(arrays["candidate_emb"]),
                torch.from_numpy(arrays["candidate_mask"]),
            ).numpy()
        return {
            str(arrays["event_ids"][i]): float(predicted[i])
            for i in range(len(arrays["event_ids"]))
        }

    pre_samples = samples  # type: ignore[assignment]
    features = extract_features_batch(pre_samples, alpha_dense, alpha_lexical, alpha_bm25)
    features_norm = normalize_features_for_stats(features, stats)

    with torch.no_grad():
        x = torch.from_numpy(features_norm)
        predicted = model(x).numpy()

    return {pre.event_id: float(predicted[i]) for i, pre in enumerate(pre_samples)}
