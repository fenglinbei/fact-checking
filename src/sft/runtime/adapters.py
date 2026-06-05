from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _matches_module_prefix(name: str, prefix: str) -> bool:
    normalized = prefix.rstrip(".")
    return name == normalized or name.startswith(normalized + ".")


def lora_enabled(train_cfg: dict[str, Any]) -> bool:
    lora_cfg = train_cfg.get("lora", {})
    if isinstance(lora_cfg, dict):
        return bool(lora_cfg.get("enabled", False))
    return bool(lora_cfg)


def _module_list(
    value: object,
    *,
    default: list[str] | None = None,
    allow_none: bool = False,
    allow_single_string: bool = False,
) -> list[str] | str | None:
    if value is None:
        if allow_none:
            return None
        return list(default or [])

    if isinstance(value, str):
        normalized = value.strip()
        if allow_none and normalized.lower() in {"", "none", "null"}:
            return None
        if "," not in normalized:
            return normalized if allow_single_string else [normalized]
        return [part.strip() for part in normalized.split(",") if part.strip()]

    return [str(item) for item in value]  # type: ignore[operator]


def freeze_modules_by_prefix(
    model: AutoModelForCausalLM,
    train_cfg: dict[str, Any],
    *,
    logger: Any | None = None,
) -> AutoModelForCausalLM:
    prefixes = _module_list(train_cfg.get("freeze_module_prefixes"), allow_none=True)
    if not prefixes:
        return model
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    prefixes = [prefix.rstrip(".") for prefix in prefixes if prefix.strip()]
    if not prefixes:
        return model

    total_params = 0
    frozen_params = 0
    matched_prefixes: set[str] = set()
    for name, param in model.named_parameters():
        total_params += param.numel()
        for prefix in prefixes:
            if _matches_module_prefix(name, prefix):
                param.requires_grad_(False)
                frozen_params += param.numel()
                matched_prefixes.add(prefix)
                break

    if logger is not None:
        pct = 100.0 * frozen_params / max(total_params, 1)
        logger.info(
            "[INFO] Frozen parameters by prefix: %d / %d (%.4f%%), prefixes=%s",
            frozen_params,
            total_params,
            pct,
            prefixes,
        )
        unmatched = [prefix for prefix in prefixes if prefix not in matched_prefixes]
        if unmatched:
            logger.warning("[WARN] freeze_module_prefixes matched no parameters: %s", unmatched)

    return model


def apply_lora_if_enabled(
    model: AutoModelForCausalLM,
    train_cfg: dict[str, Any],
    *,
    gradient_checkpointing: bool,
    logger: Any | None = None,
) -> AutoModelForCausalLM:
    if not lora_enabled(train_cfg):
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "sft_train.lora.enabled=true requires the `peft` package. "
            "Install project dependencies again or run `pip install peft`."
        ) from exc

    lora_cfg = train_cfg.get("lora", {}) or {}
    if not isinstance(lora_cfg, dict):
        lora_cfg = {}

    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 32))),
        lora_dropout=float(lora_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05))),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=_module_list(
            lora_cfg.get("target_modules"),
            default=DEFAULT_LORA_TARGET_MODULES,
            allow_single_string=True,
        ),
        modules_to_save=_module_list(
            lora_cfg.get("modules_to_save"),
            allow_none=True,
        ),
    )
    model = get_peft_model(model, peft_config)

    if logger is not None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100.0 * trainable / max(total, 1)
        logger.info(
            "[INFO] LoRA enabled: trainable parameters=%d / %d (%.4f%%), r=%d, alpha=%d",
            trainable,
            total,
            pct,
            peft_config.r,
            peft_config.lora_alpha,
        )

    return model


def is_peft_model(model: object) -> bool:
    return hasattr(model, "peft_config")


def checkpoint_has_peft_adapter(output_path: Path) -> bool:
    if not (output_path / "adapter_config.json").exists():
        return False
    return (output_path / "adapter_model.safetensors").exists() or (output_path / "adapter_model.bin").exists()


def checkpoint_has_hf_artifacts(output_path: Path) -> bool:
    if checkpoint_has_peft_adapter(output_path):
        return True

    if not (output_path / "config.json").exists():
        return False

    weight_patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    ]
    return any(any(output_path.glob(pattern)) for pattern in weight_patterns)
