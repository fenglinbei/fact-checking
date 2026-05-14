from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, compute_hybrid_scores
from fact_checking.rl_mmr.soft_label_dataset import SoftLabelDataset
from fact_checking.rl_mmr.soft_label_features import (
    SOFT_LABEL_FEATURE_NAMES,
    extract_soft_label_features,
)
from fact_checking.rl_mmr.soft_label_selector import select_lambdas_from_probs


def _sample(event_id: str = "e1") -> ChunkMMRSample:
    claim_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chunk_emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidates = [
        {"text": "The claim is directly supported.", "report_id": "r1", "sent_idx": i}
        for i in range(len(chunk_emb))
    ]
    return ChunkMMRSample(
        event_id=event_id,
        claim="The unemployment rate was higher in 2020.",
        label="true",
        explain="",
        candidates=candidates,
        chunk_emb=chunk_emb,
        claim_emb=claim_emb,
    )


def test_extract_soft_label_features_has_expected_numeric_vector() -> None:
    sample = _sample()
    scored = compute_hybrid_scores(sample, 0.7, 0.2, 0.1)
    features = extract_soft_label_features(
        sample,
        scored["hybrid_scores"],
        scored["chunk_emb"],
        lambda_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
        top_k=2,
    )

    assert set(SOFT_LABEL_FEATURE_NAMES).issubset(features.keys())
    assert features["n_candidates"] == 4.0
    assert 0.0 <= features["jaccard_0p30_0p70"] <= 1.0
    assert features["claim_word_count"] > 0


def test_soft_label_dataset_builds_targets_and_weights(tmp_path: Path) -> None:
    cache_path = tmp_path / "chunk.pkl"
    with cache_path.open("wb") as f:
        pickle.dump([_sample("e1")], f)

    oracle_path = tmp_path / "oracle.jsonl"
    record = {
        "event_id": "e1",
        "gold_label": "true",
        "oracle_lambda": 0.7,
        "best_logprob": -0.1,
        "logprobs_by_lambda": {
            "0.10": -2.0,
            "0.30": -1.0,
            "0.50": -0.5,
            "0.70": -0.1,
            "0.90": -0.3,
        },
    }
    oracle_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    ds = SoftLabelDataset.from_oracle_and_cache(
        oracle_path,
        cache_path,
        lambda_grid=[0.1, 0.3, 0.5, 0.7, 0.9],
        top_k=2,
        weight_mode="gap",
    )

    assert ds.features.shape == (1, len(SOFT_LABEL_FEATURE_NAMES))
    assert ds.soft_targets.shape == (1, 5)
    assert np.allclose(ds.soft_targets.sum(axis=1), 1.0)
    assert ds.sample_weights.shape == (1,)


def test_select_lambdas_from_probs_modes() -> None:
    probs = np.array([[0.1, 0.2, 0.6, 0.1, 0.0]], dtype=np.float32)
    grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)

    assert select_lambdas_from_probs(probs, grid, inference_mode="argmax")[0] == np.float32(0.5)
    expected = select_lambdas_from_probs(probs, grid, inference_mode="expected")[0]
    assert np.isclose(expected, 0.44)
    sampled = select_lambdas_from_probs(probs, grid, inference_mode="sample", random_seed=1)
    assert sampled.shape == (1,)
