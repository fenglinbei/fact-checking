"""Dataset construction for soft-label lambda policies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fact_checking.build.candidates import _load_pickle, compute_hybrid_scores
from fact_checking.rl_mmr.soft_label_features import (
    DEFAULT_LAMBDA_GRID,
    SOFT_LABEL_FEATURE_NAMES,
    extract_soft_label_features,
    feature_dict_to_vector,
)


def parse_lambda_grid(value: str | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    if isinstance(value, str):
        items = [float(x.strip()) for x in value.split(",") if x.strip()]
    else:
        items = [float(x) for x in value]
    if len(items) < 2:
        raise ValueError("lambda_grid must contain at least two values.")
    return np.array(sorted(dict.fromkeys(items)), dtype=np.float32)


def load_oracle_logprobs(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            event_id = str(rec.get("event_id", "")).strip()
            if not event_id:
                continue
            records[event_id] = rec
    return records


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    scaled = np.asarray(values, dtype=np.float32) / temp
    shifted = scaled - float(np.max(scaled))
    expv = np.exp(shifted)
    total = float(expv.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(scaled, 1.0 / len(scaled), dtype=np.float32)
    return (expv / total).astype(np.float32)


def _rounded_lookup(logprobs_by_lambda: dict[str, Any]) -> dict[float, float]:
    values: dict[float, float] = {}
    for key, val in logprobs_by_lambda.items():
        try:
            values[round(float(key), 6)] = float(val)
        except (TypeError, ValueError):
            continue
    return values


def utility_vector_from_record(record: dict[str, Any], lambda_grid: np.ndarray) -> np.ndarray:
    lp_by_lambda = record.get("logprobs_by_lambda")
    values = np.full(len(lambda_grid), -100.0, dtype=np.float32)
    if isinstance(lp_by_lambda, dict) and lp_by_lambda:
        lookup = _rounded_lookup(lp_by_lambda)
        for i, lam in enumerate(lambda_grid):
            rounded = round(float(lam), 6)
            if rounded in lookup:
                values[i] = lookup[rounded]
    if np.all(values <= -99.0):
        oracle_lambda = float(record.get("oracle_lambda", lambda_grid[len(lambda_grid) // 2]))
        idx = int(np.argmin(np.abs(lambda_grid - oracle_lambda)))
        values[idx] = 0.0
    return values


def soft_target_from_utility(utility: np.ndarray, temperature: float) -> np.ndarray:
    return _softmax(utility, temperature)


def sample_weight_from_utility(
    utility: np.ndarray,
    lambda_grid: np.ndarray,
    *,
    mode: str,
    base_lambda: float = 0.7,
) -> float:
    mode = str(mode).strip().lower()
    if mode == "none":
        return 1.0
    finite = utility[np.isfinite(utility)]
    if finite.size < 2:
        return 1.0
    sorted_values = np.sort(finite)[::-1]
    if mode == "gap":
        return max(0.0, float(sorted_values[0] - sorted_values[1]))
    if mode == "margin":
        base_idx = int(np.argmin(np.abs(lambda_grid - float(base_lambda))))
        return max(0.0, float(np.max(utility) - utility[base_idx]))
    raise ValueError("weight_mode must be one of: margin, gap, none")


def normalize_sample_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.size == 0:
        return weights
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, 0.0)
    mean = float(weights.mean())
    if mean <= 1e-8:
        return np.ones_like(weights, dtype=np.float32)
    return (weights / mean).astype(np.float32)


@dataclass
class SoftLabelDataset:
    """Tabular dataset built from oracle utility curves and Chunk-MMR cache."""

    features: np.ndarray
    soft_targets: np.ndarray
    sample_weights: np.ndarray
    event_ids: list[str]
    lambda_grid: np.ndarray
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    utilities: np.ndarray
    raw_features: np.ndarray | None = None
    raw_feature_dicts: list[dict[str, Any]] | None = None
    oracle_records: list[dict[str, Any]] | None = None

    @classmethod
    def from_oracle_and_cache(
        cls,
        oracle_jsonl: str | Path,
        chunk_cache_pkl: str | Path,
        lambda_grid: list[float] | tuple[float, ...] | np.ndarray = DEFAULT_LAMBDA_GRID,
        temperature: float = 1.0,
        weight_mode: str = "margin",
        *,
        top_k: int = 5,
        alpha_dense: float = 0.70,
        alpha_lexical: float = 0.20,
        alpha_bm25: float = 0.10,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        standardize: bool = True,
        sample_limit: int | None = None,
    ) -> "SoftLabelDataset":
        grid = parse_lambda_grid(lambda_grid)
        oracle_by_eid = load_oracle_logprobs(oracle_jsonl)
        chunk_samples = _load_pickle(Path(chunk_cache_pkl))

        raw_vectors: list[np.ndarray] = []
        raw_dicts: list[dict[str, Any]] = []
        soft_targets: list[np.ndarray] = []
        utilities: list[np.ndarray] = []
        sample_weights: list[float] = []
        event_ids: list[str] = []
        records: list[dict[str, Any]] = []

        for sample in chunk_samples:
            event_id = str(sample.event_id)
            rec = oracle_by_eid.get(event_id)
            if rec is None:
                continue
            scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
            feats = extract_soft_label_features(
                sample,
                scored["hybrid_scores"],
                scored["chunk_emb"],
                lambda_grid=tuple(float(x) for x in grid.tolist()),
                top_k=top_k,
            )
            utility = utility_vector_from_record(rec, grid)

            raw_vectors.append(feature_dict_to_vector(feats, SOFT_LABEL_FEATURE_NAMES))
            raw_dicts.append(dict(feats))
            soft_targets.append(soft_target_from_utility(utility, temperature))
            utilities.append(utility)
            sample_weights.append(sample_weight_from_utility(utility, grid, mode=weight_mode))
            event_ids.append(event_id)
            records.append(rec)
            if sample_limit is not None and len(event_ids) >= int(sample_limit):
                break

        if not raw_vectors:
            raise ValueError(
                f"No ChunkMMR samples in {chunk_cache_pkl} matched oracle records from {oracle_jsonl}."
            )

        raw_features = np.stack(raw_vectors).astype(np.float32, copy=False)
        if standardize:
            if feature_mean is None:
                mean = raw_features.mean(axis=0)
            else:
                mean = np.asarray(feature_mean, dtype=np.float32)
            if feature_std is None:
                std = raw_features.std(axis=0)
            else:
                std = np.asarray(feature_std, dtype=np.float32)
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            features = ((raw_features - mean) / std).astype(np.float32, copy=False)
        else:
            mean = np.zeros(raw_features.shape[1], dtype=np.float32)
            std = np.ones(raw_features.shape[1], dtype=np.float32)
            features = raw_features

        return cls(
            features=features,
            soft_targets=np.stack(soft_targets).astype(np.float32, copy=False),
            sample_weights=normalize_sample_weights(np.array(sample_weights, dtype=np.float32)),
            event_ids=event_ids,
            lambda_grid=grid,
            feature_names=list(SOFT_LABEL_FEATURE_NAMES),
            feature_mean=mean.astype(np.float32, copy=False),
            feature_std=std.astype(np.float32, copy=False),
            utilities=np.stack(utilities).astype(np.float32, copy=False),
            raw_features=raw_features,
            raw_feature_dicts=raw_dicts,
            oracle_records=records,
        )

    def save_npz(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            features=self.features,
            soft_targets=self.soft_targets,
            sample_weights=self.sample_weights,
            event_ids=np.array(self.event_ids, dtype=object),
            lambda_grid=self.lambda_grid,
            feature_names=np.array(self.feature_names, dtype=object),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            utilities=self.utilities,
            raw_features=self.raw_features if self.raw_features is not None else self.features,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SoftLabelDataset":
        data = np.load(Path(path), allow_pickle=True)
        return cls(
            features=data["features"].astype(np.float32, copy=False),
            soft_targets=data["soft_targets"].astype(np.float32, copy=False),
            sample_weights=data["sample_weights"].astype(np.float32, copy=False),
            event_ids=[str(x) for x in data["event_ids"].tolist()],
            lambda_grid=data["lambda_grid"].astype(np.float32, copy=False),
            feature_names=[str(x) for x in data["feature_names"].tolist()],
            feature_mean=data["feature_mean"].astype(np.float32, copy=False),
            feature_std=data["feature_std"].astype(np.float32, copy=False),
            utilities=data["utilities"].astype(np.float32, copy=False),
            raw_features=data.get("raw_features"),
        )
