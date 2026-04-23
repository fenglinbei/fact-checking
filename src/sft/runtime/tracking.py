from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Number
from typing import Any

DEFAULT_PROJECT = "fact-checking-stage-ab"
DEFAULT_RUN_NAME = "llm_baseline_sft"

_DISABLED_BACKENDS = {"", "none", "off", "false", "disabled", "null"}
_TRACKING_CONTROL_KEYS = {"enabled", "backend", "type", "log_with", "provider"}


@dataclass(frozen=True)
class TrackingSetup:
    backend: str | None
    project_name: str | None
    run_name: str | None
    log_with: Any | None
    init_kwargs: dict[str, dict[str, Any]]

    @property
    def enabled(self) -> bool:
        return self.backend is not None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "none", "null"}:
            return False
    return bool(value)


def _as_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _resolve_backend_config(cfg: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    tracking_cfg = cfg.get("tracking")
    if isinstance(tracking_cfg, dict):
        raw_backend = tracking_cfg.get("backend", tracking_cfg.get("type", tracking_cfg.get("log_with", "")))
        backend = str(raw_backend or "").strip().lower()
        if backend in _DISABLED_BACKENDS or not _as_bool(tracking_cfg.get("enabled"), default=True):
            return None, {}
        provider_cfg = dict(cfg.get(backend, {}) or {})
        provider_cfg.update(
            {
                key: value
                for key, value in tracking_cfg.items()
                if key not in _TRACKING_CONTROL_KEYS
            }
        )
        return backend, provider_cfg

    swanlab_cfg = dict(cfg.get("swanlab", {}) or {})
    if _as_bool(swanlab_cfg.get("enabled"), default=False):
        return "swanlab", swanlab_cfg

    wandb_cfg = dict(cfg.get("wandb", {}) or {})
    if _as_bool(wandb_cfg.get("enabled"), default=False):
        return "wandb", wandb_cfg

    return None, {}


def _build_wandb_setup(provider_cfg: dict[str, Any]) -> TrackingSetup:
    project_name = str(provider_cfg.get("project", DEFAULT_PROJECT))
    run_name = str(provider_cfg.get("run_name", provider_cfg.get("name", DEFAULT_RUN_NAME)))

    os.environ.setdefault("WANDB_PROJECT", project_name)
    if provider_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(provider_cfg["entity"])
    os.environ.setdefault("WANDB_LOG_MODEL", _as_env_value(provider_cfg.get("log_model", "false")))
    os.environ.setdefault("WANDB_WATCH", _as_env_value(provider_cfg.get("watch", "false")))

    wandb_kwargs = dict(provider_cfg.get("init_kwargs", {}) or {})
    wandb_kwargs.setdefault("name", run_name)
    for key in ("group", "job_type", "tags", "notes", "id", "resume", "mode", "dir"):
        if key in provider_cfg:
            wandb_kwargs.setdefault(key, provider_cfg[key])

    return TrackingSetup(
        backend="wandb",
        project_name=project_name,
        run_name=run_name,
        log_with="wandb",
        init_kwargs={"wandb": _clean_mapping(wandb_kwargs)},
    )


def _swanlab_log_with(project_name: str, swanlab_kwargs: dict[str, Any]) -> Any:
    try:
        from accelerate.tracking import SwanLabTracker  # noqa: F401

        return "swanlab"
    except (ImportError, AttributeError):
        pass

    try:
        from swanlab.integration.accelerate import SwanLabTracker
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab tracking requires `swanlab` plus an Accelerate version with SwanLab support. "
            "Install dependencies from requirements.txt or run `pip install swanlab accelerate>=1.8.0`."
        ) from exc

    try:
        return SwanLabTracker(project_name, **swanlab_kwargs)
    except TypeError:
        return SwanLabTracker(project_name)


def _build_swanlab_setup(provider_cfg: dict[str, Any]) -> TrackingSetup:
    project_name = str(provider_cfg.get("project", DEFAULT_PROJECT))
    run_name = str(
        provider_cfg.get(
            "experiment_name",
            provider_cfg.get("run_name", provider_cfg.get("name", DEFAULT_RUN_NAME)),
        )
    )

    swanlab_kwargs = dict(provider_cfg.get("init_kwargs", {}) or {})
    swanlab_kwargs.setdefault("experiment_name", run_name)
    for key in ("workspace", "mode", "logdir", "tags", "description"):
        if key in provider_cfg:
            swanlab_kwargs.setdefault(key, provider_cfg[key])
    swanlab_kwargs = _clean_mapping(swanlab_kwargs)

    return TrackingSetup(
        backend="swanlab",
        project_name=project_name,
        run_name=run_name,
        log_with=_swanlab_log_with(project_name, swanlab_kwargs),
        init_kwargs={"swanlab": swanlab_kwargs},
    )


def build_tracking_setup(cfg: dict[str, Any]) -> TrackingSetup:
    backend, provider_cfg = _resolve_backend_config(cfg)
    if backend is None:
        return TrackingSetup(
            backend=None,
            project_name=None,
            run_name=None,
            log_with=None,
            init_kwargs={},
        )
    if backend == "wandb":
        return _build_wandb_setup(provider_cfg)
    if backend == "swanlab":
        return _build_swanlab_setup(provider_cfg)
    raise ValueError(f"Unsupported tracking backend: {backend!r}. Use 'wandb', 'swanlab', or 'none'.")


def _is_swanlab_loggable(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Number) and not isinstance(value, complex):
        return True
    try:
        import torch

        return isinstance(value, torch.Tensor) and value.numel() == 1
    except Exception:
        return False


def log_metrics(
    accelerator: Any,
    values: dict[str, Any],
    *,
    step: int | None = None,
    backend: str | None = None,
) -> None:
    if backend == "swanlab":
        values = {key: value for key, value in values.items() if _is_swanlab_loggable(value)}
        if not values:
            return
    accelerator.log(values, step=step)
