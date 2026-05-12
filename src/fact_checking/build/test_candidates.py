from __future__ import annotations

from copy import deepcopy

from fact_checking.build.candidates import _premmr_config_fingerprint


def _base_build_cfg() -> dict:
    return {
        "data": {
            "train_path": "data/raw/LIAR-RAW/train.json",
            "val_path": "data/raw/LIAR-RAW/val.json",
            "test_path": "data/raw/LIAR-RAW/test.json",
        },
        "retrieval": {
            "embedder_model": "/data/models/bge-base-en-v1.5/",
            "device": "cuda",
            "max_length": 256,
            "batch_size": 64,
            "top_k": 16,
            "alpha_dense": 0.70,
            "alpha_lexical": 0.20,
            "alpha_bm25": 0.10,
            "mmr_lambda": 0.70,
            "precision": "bf16",
            "num_gpus": 4,
            "prefetch_size": 200,
            "cpu_workers": 4,
            "chunking": {
                "strategy": "sentence",
                "context_k": 1,
            },
        },
        "prompt": {
            "model_name_or_path": "/data/models/Qwen2.5-7B-Instruct",
            "auto_length": True,
            "max_length": 2048,
            "output_mode": "label_only",
            "label_format": "letter",
        },
    }


def test_premmr_fingerprint_ignores_mmr_selection_settings() -> None:
    base = _base_build_cfg()
    changed = deepcopy(base)
    changed["retrieval"]["top_k"] = 24
    changed["retrieval"]["mmr_lambda"] = 0.1

    assert _premmr_config_fingerprint(base) == _premmr_config_fingerprint(changed)


def test_premmr_fingerprint_keeps_embedding_settings() -> None:
    base = _base_build_cfg()
    changed = deepcopy(base)
    changed["retrieval"]["max_length"] = 512

    assert _premmr_config_fingerprint(base) != _premmr_config_fingerprint(changed)
