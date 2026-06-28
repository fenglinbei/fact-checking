#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNK_MMR_CACHE_VERSION = "chunk-text-embedding-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the chunk-MMR cache fingerprint for an experiment config.")
    parser.add_argument("--config", required=True, help="Experiment name or configs/experiment/*.yaml path.")
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(str(args.config))
    build_cfg = dict(cfg.get("build", {}) or {})
    if args.sample_limit is not None:
        data_cfg = dict(build_cfg.get("data", {}) or {})
        data_cfg["sample_limit"] = int(args.sample_limit)
        build_cfg["data"] = data_cfg
    print(_chunk_mmr_config_fingerprint(build_cfg))


def _load_config(config: str) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return _load_experiment_yaml(_experiment_name(config))

    config_path = Path(config)
    if not config_path.suffix:
        experiment_name = config
    else:
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        experiment_dir = PROJECT_ROOT / "configs" / "experiment"
        try:
            rel = config_path.resolve().relative_to(experiment_dir.resolve())
        except ValueError:
            loaded = OmegaConf.load(config_path)
            return dict(OmegaConf.to_container(loaded, resolve=True) or {})
        experiment_name = rel.with_suffix("").as_posix()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="pipeline/default", overrides=[f"experiment={experiment_name}"])
    return dict(OmegaConf.to_container(cfg, resolve=True) or {})


def _experiment_name(config: str) -> str:
    path = Path(config)
    if not path.suffix:
        return config
    return path.stem


def _load_experiment_yaml(experiment_name: str) -> dict[str, Any]:
    import yaml

    path = PROJECT_ROOT / "configs" / "experiment" / f"{experiment_name}.yaml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return {}
    data = dict(data)
    defaults = list(data.pop("defaults", []) or [])
    merged: dict[str, Any] = {}
    for item in defaults:
        if item == "_self_":
            continue
        if isinstance(item, str):
            if item.startswith("/"):
                continue
            merged = _deep_merge(merged, _load_experiment_yaml(item))
        elif isinstance(item, dict):
            exp = item.get("experiment")
            if exp:
                merged = _deep_merge(merged, _load_experiment_yaml(str(exp)))
    return _deep_merge(merged, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _chunk_mmr_config_fingerprint(cfg: dict[str, Any]) -> str:
    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    retrieval = {
        key: retrieval_cfg.get(key)
        for key in ("embedder_model", "device", "max_length", "precision")
        if key in retrieval_cfg
    }
    payload = {
        "version": CHUNK_MMR_CACHE_VERSION,
        "data": cfg.get("data", {}),
        "retrieval": retrieval,
        "chunking": retrieval_cfg.get("chunking", {}),
    }
    sentence_reader = _sentence_reader_fingerprint_payload(cfg)
    if sentence_reader is not None:
        payload["sentence_reader"] = sentence_reader
    return _fingerprint(payload)


def _sentence_reader_fingerprint_payload(cfg: dict[str, Any]) -> dict[str, Any] | None:
    data_cfg = dict(cfg.get("data", {}) or {})
    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    selection_method = str(retrieval_cfg.get("selection_method", "mmr")).strip().lower()
    source = str(data_cfg.get("sentence_source") or data_cfg.get("source") or "").strip().lower()
    if not source:
        source = "tokenized" if selection_method in {"raw_top_evidence", "raw_label_topk", "raw_evidence"} else "content"
    reader = {
        "sentence_source": source,
        "sentence_min_char_len": int(data_cfg.get("sentence_min_char_len", data_cfg.get("min_char_len", 10))),
    }
    if reader == {"sentence_source": "content", "sentence_min_char_len": 10}:
        return None
    return reader


def _fingerprint(payload: Any, *, length: int = 12, algorithm: str = "sha1") -> str:
    encoded = json.dumps(
        _stringify_mapping_keys(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.new(algorithm, encoded.encode("utf-8")).hexdigest()
    return digest[:length]


def _stringify_mapping_keys(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _stringify_mapping_keys(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_stringify_mapping_keys(value) for value in payload]
    if isinstance(payload, tuple):
        return [_stringify_mapping_keys(value) for value in payload]
    return payload


if __name__ == "__main__":
    main()
