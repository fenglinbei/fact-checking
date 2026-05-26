"""Shared training infrastructure used by both generative and label-token trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft.runtime.adapters import apply_lora_if_enabled
from sft.runtime.device import enable_tf32_if_available
from sft.runtime.deps import flash_attn2_available, fla_fast_path_available
from sft.runtime.tracking import build_tracking_setup


def setup_accelerator_and_tracker(
    train_cfg: dict[str, Any], cfg: dict[str, Any]
) -> tuple[Accelerator, object]:
    """Initialize Accelerator with DeepSpeed config and tracker integration."""
    enable_tf32_if_available()
    tracking_setup = build_tracking_setup(cfg)
    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        mixed_precision=mixed_precision,
        log_with=tracking_setup.log_with,
    )
    if tracking_setup.enabled:
        accelerator.init_trackers(
            project_name=tracking_setup.project_name,
            config=cfg,
            init_kwargs=tracking_setup.init_kwargs,
        )
    return accelerator, tracking_setup


def setup_model_and_tokenizer(
    *,
    model_name_or_path: str,
    model_cfg: dict[str, Any],
    lora_cfg: dict[str, Any] | None,
    tokenized_cache_dir: str | None,
    accelerator: Accelerator,
    logger: Any,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load tokenizer and model with flash-attn, FLA, LoRA, and gradient checkpointing."""
    use_flash_attention_2 = bool(model_cfg.get("use_flash_attention_2", True))
    use_fused_linear_attention = bool(model_cfg.get("use_fused_linear_attention", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if use_flash_attention_2:
        available = flash_attn2_available()
        if available:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using flash_attention_2.")
        else:
            logger.info("flash_attn is not available. Continuing with default attention.")
    if use_fused_linear_attention:
        fla_ok = fla_fast_path_available()
        if fla_ok:
            logger.info("FusedLinearAttention (FLA) detected; enabling fused linear attention kernel.")
        else:
            logger.info("FusedLinearAttention requested but not installed/available. Skipping.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if model_cfg.get("bf16", True) else torch.float32,
        **model_kwargs,
    )

    if bool(model_cfg.get("gradient_checkpointing_enable", True)):
        model.gradient_checkpointing_enable({"use_reentrant": False})

    if lora_cfg:
        lora_cfg = dict(lora_cfg)
        lora_cfg.pop("target_modules", None)
        model = apply_lora_if_enabled(model, lora_cfg)

    if tokenized_cache_dir:
        cache_path = Path(tokenized_cache_dir)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        from transformers import set_seed
        set_seed(int(model_cfg.get("seed", 42)))

    return model, tokenizer
