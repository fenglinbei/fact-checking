from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fact_checking.build.candidates import _chunk_mmr_config_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_experiment_build_cfg(experiment: str, overrides: list[str] | None = None) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="pipeline/default",
            overrides=[f"experiment={experiment}", *(overrides or [])],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra config did not resolve to a dictionary.")
    build_cfg = resolved.get("build", {})
    if not isinstance(build_cfg, dict):
        raise TypeError("Resolved Hydra config does not contain a build dictionary.")
    return build_cfg


def resolve_chunk_mmr_cache_path(
    build_cfg: dict[str, Any],
    *,
    split_name: str,
    cache_root: str | Path,
) -> Path:
    fp = _chunk_mmr_config_fingerprint(build_cfg)
    return Path(cache_root) / fp / f"{split_name}.pkl"


def pick_retrieval_value(cli_value, retrieval_cfg: dict[str, Any], key: str, default):
    if cli_value is not None:
        return cli_value
    value = retrieval_cfg.get(key)
    if value is not None:
        return value
    return default
