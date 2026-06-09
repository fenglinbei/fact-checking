#!/usr/bin/env python
"""Prepare a resolved RAWFC v0.6c config for a backbone migration run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


BASE_EXPERIMENTS = {
    "lora": "v0_6c_rawfc3_rule_step_adaptive5_10_eval25",
    "fullft": "v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft",
}

PHI_LORA_TARGET_MODULES = [
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]

GEMMA4_LORA_TARGET_MODULES = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)

GEMMA4_TEXT_ONLY_LORA_TARGET_MODULES = (
    r"model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)

MISTRAL3_VLLM_EXTRA_ARGS = [
    "--tokenizer-mode",
    "mistral",
    "--config-format",
    "mistral",
    "--load-format",
    "mistral",
]

GEMMA4_TEXT_ONLY_FULLFT_FREEZE_PREFIXES = [
    "model.vision_tower",
    "model.audio_tower",
    "model.embed_vision",
    "model.embed_audio",
]

MULTIMODAL_TEXT_ONLY_FULLFT_DEEPSPEED_CONFIG = "configs/deepspeed_zero3.json"
LARGE_FULLFT_DEEPSPEED_CONFIG = "configs/deepspeed_zero3.json"
GEMMA4_FULLFT_DEEPSPEED_CONFIG = "configs/deepspeed_zero3_bsz1_ga8_ultralowpeak.json"
MISTRAL3_MULTIMODAL_FULLFT_DEEPSPEED_CONFIG = "configs/deepspeed_zero3_bsz1_ga8_lowpeak.json"
MISTRAL3_MULTIMODAL_FULLFT_FREEZE_PREFIXES = [
    "model.vision_tower",
    "model.multi_modal_projector",
]
MINISTRAL3_LORA_DEEPSPEED_CONFIG = "configs/deepspeed_zero2_bsz1_ga8.json"
LORA_PER_DEVICE_TRAIN_BATCH_SIZE = 8
LORA_PER_DEVICE_EVAL_BATCH_SIZE = 1
LORA_GRADIENT_ACCUMULATION_STEPS = 1
LORA_EARLY_STOPPING_PATIENCE = 16
LOW_MICRO_BATCH_LORA_BACKBONES = {"ministral3_8b"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping YAML at {path}, got {type(payload).__name__}")
    return payload


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_to_path(
    *,
    project_root: Path,
    current_path: Path,
    entry: Any,
    experiment_override: str | None,
) -> Path | None:
    if isinstance(entry, str):
        if entry == "_self_":
            return None
        if entry.startswith("/"):
            group_value = entry.lstrip("/")
            if ":" not in group_value:
                raise ValueError(f"Unsupported defaults entry {entry!r} in {current_path}")
            group, value = [part.strip() for part in group_value.split(":", 1)]
            return project_root / "configs" / group / f"{value}.yaml"
        return current_path.parent / f"{entry}.yaml"

    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"Unsupported defaults mapping {entry!r} in {current_path}")
        raw_group, raw_value = next(iter(entry.items()))
        group = str(raw_group).lstrip("/")
        value = str(raw_value)
        if group == "experiment" and experiment_override:
            value = experiment_override
        return project_root / "configs" / group / f"{value}.yaml"

    raise ValueError(f"Unsupported defaults entry {entry!r} in {current_path}")


def _load_with_defaults(
    *,
    project_root: Path,
    path: Path,
    experiment_override: str | None = None,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config defaults cycle detected: {cycle}")

    payload = _load_yaml(path)
    defaults = list(payload.get("defaults", []) or [])
    self_payload = {key: value for key, value in payload.items() if key != "defaults"}
    result: dict[str, Any] = {}
    saw_self = False

    for entry in defaults:
        if entry == "_self_":
            result = _merge_dicts(result, self_payload)
            saw_self = True
            continue
        default_path = _default_to_path(
            project_root=project_root,
            current_path=path,
            entry=entry,
            experiment_override=experiment_override,
        )
        if default_path is None:
            continue
        result = _merge_dicts(
            result,
            _load_with_defaults(
                project_root=project_root,
                path=default_path,
                experiment_override=None,
                stack=(*stack, path),
            ),
        )

    if not saw_self:
        result = _merge_dicts(result, self_payload)
    return result


def _compose_experiment(project_root: Path, experiment: str) -> dict[str, Any]:
    return _load_with_defaults(
        project_root=project_root,
        path=project_root / "configs" / "pipeline" / "default.yaml",
        experiment_override=experiment,
    )


def _set_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node: dict[str, Any] = payload
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _chat_template_for_backbone(backbone: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "mode": "tokenizer_default",
        "add_generation_prompt": True,
        "template_kwargs": {},
    }
    if backbone in {"qwen3_17b", "qwen3_8b"}:
        cfg["template_kwargs"] = {"enable_thinking": False}
        cfg["thinking_control"] = "hard_enable_thinking_false"
    elif backbone == "qwen3_4b_2507":
        cfg["thinking_control"] = "native_non_thinking"
    elif backbone in {"llama31_8b", "phi4_mini", "gemma4_e4b", "ministral3_8b"}:
        cfg["migration_note"] = "transfer_backbone_tokenizer_default"
    return cfg


def _apply_backbone_overrides(payload: dict[str, Any], backbone: str, model_variant: str) -> None:
    if backbone == "phi4_mini":
        _set_path(payload, "sft_train.lora.target_modules", PHI_LORA_TARGET_MODULES)
        _set_path(payload, "backbone_migration.lora_target_policy", "phi_fused_qkv_gate_up")
    elif backbone == "gemma4_e4b":
        if model_variant == "text_only":
            _set_path(payload, "sft_train.lora.target_modules", GEMMA4_TEXT_ONLY_LORA_TARGET_MODULES)
            _set_path(payload, "backbone_migration.lora_target_policy", "gemma4_text_only_regex")
            _set_path(payload, "backbone_migration.loader_note", "gemma4 text-only export loads through Gemma4ForCausalLM")
        else:
            _set_path(payload, "sft_train.lora.target_modules", GEMMA4_LORA_TARGET_MODULES)
            _set_path(payload, "backbone_migration.lora_target_policy", "gemma4_text_tower_regex")
            _set_path(payload, "backbone_migration.loader_note", "gemma4 AutoModelForCausalLM maps to Gemma4ForConditionalGeneration")
        _set_path(payload, "sft_train.use_flash_attention_2", False)
        _set_path(payload, "backbone_migration.attention_policy", "disable_flash_attention_2_head_dim_512")
    elif backbone == "ministral3_8b":
        _set_path(payload, "infer.server.extra_args", MISTRAL3_VLLM_EXTRA_ARGS)
        _set_path(payload, "backbone_migration.loader_note", "mistral3 requires conditional-generation fallback for HF label-token evaluation")
        _set_path(payload, "backbone_migration.quantization_policy", "finegrained_fp8_dequantize_for_hf_forward")
        _set_path(payload, "backbone_migration.vllm_extra_args_policy", "mistral_format")


def _deepspeed_policy_name(deepspeed_config: str) -> str:
    if "ultralowpeak" in deepspeed_config and "zero3" in deepspeed_config:
        return "zero3_no_cpu_offload_ultralowpeak"
    if "lowpeak" in deepspeed_config and "zero3" in deepspeed_config:
        return "zero3_no_cpu_offload_lowpeak"
    if "zero2_bsz1_ga8" in deepspeed_config:
        return "zero2_bsz1_ga8"
    if "zero2_bsz8_ga1" in deepspeed_config:
        return "zero2_bsz8_ga1"
    if "zero2" in deepspeed_config:
        return "zero2_custom"
    if "optimizer_offload" in deepspeed_config:
        return "zero3_cpu_offload_optimizer_only"
    if "cpu_offload" in deepspeed_config:
        return "zero3_cpu_offload_optimizer_and_param_low_peak"
    if "zero3" in deepspeed_config:
        return "zero3_no_cpu_offload"
    return "custom_deepspeed_config"


def _apply_ordinal_loss_overrides(payload: dict[str, Any], alpha: float) -> None:
    if alpha < 0.0:
        raise ValueError(f"--ordinal-loss-alpha must be non-negative, got {alpha}.")
    _set_path(payload, "sft_train.label_token_ce.ordinal_loss.enabled", True)
    _set_path(payload, "sft_train.label_token_ce.ordinal_loss.alpha", float(alpha))
    _set_path(payload, "sft_train.label_token_ce.ordinal_loss.normalize_distance", True)
    _set_path(payload, "backbone_migration.method_upgrade", "ordinal_aware_loss")
    _set_path(payload, "backbone_migration.ordinal_loss", "expected_absolute_rank_distance")
    _set_path(payload, "backbone_migration.ordinal_loss_alpha", float(alpha))
    _set_path(payload, "backbone_migration.ordinal_loss_normalize_distance", True)


def _prepare_config(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    base_experiment = str(args.base_experiment or BASE_EXPERIMENTS[args.finetune])
    payload = _compose_experiment(project_root, base_experiment)
    model_path = str(args.model_path)
    case_name = str(args.case_name)
    backbone = str(args.backbone)
    model_variant = str(args.model_variant)

    _set_path(payload, "model_name_or_path", model_path)
    _set_path(payload, "build.prompt.model_name_or_path", model_path)
    _set_path(payload, "build.prompt.chat_template", _chat_template_for_backbone(backbone))
    _set_path(payload, "train.model_name_or_path", model_path)
    _set_path(payload, "experiment.name", case_name)
    _set_path(payload, "baseline.variant", case_name)
    _set_path(payload, "swanlab.experiment_name", case_name)
    _set_path(payload, "backbone_migration.backbone", backbone)
    _set_path(payload, "backbone_migration.model_name_or_path", model_path)
    _set_path(payload, "backbone_migration.model_variant", model_variant)
    _apply_backbone_overrides(payload, backbone, model_variant)
    if args.ordinal_loss_alpha is not None:
        _apply_ordinal_loss_overrides(payload, float(args.ordinal_loss_alpha))

    if args.finetune == "lora":
        lora_train_batch_size = LORA_PER_DEVICE_TRAIN_BATCH_SIZE
        lora_grad_accum_steps = LORA_GRADIENT_ACCUMULATION_STEPS
        lora_batch_policy = "legacy_bsz8_eval_bsz1_ga1"
        lora_deepspeed_config = str(args.deepspeed_config or "")
        if backbone in LOW_MICRO_BATCH_LORA_BACKBONES:
            lora_train_batch_size = 1
            lora_grad_accum_steps = 8
            lora_batch_policy = f"{backbone}_legacy_effective_bsz32_bsz1_eval_bsz1_ga8"
            if not lora_deepspeed_config:
                lora_deepspeed_config = MINISTRAL3_LORA_DEEPSPEED_CONFIG
        if lora_deepspeed_config:
            _set_path(payload, "train.deepspeed_config", lora_deepspeed_config)
            _set_path(payload, "backbone_migration.deepspeed_policy", _deepspeed_policy_name(lora_deepspeed_config))
        _set_path(payload, "sft_train.per_device_train_batch_size", lora_train_batch_size)
        _set_path(payload, "sft_train.per_device_eval_batch_size", LORA_PER_DEVICE_EVAL_BATCH_SIZE)
        _set_path(payload, "sft_train.gradient_accumulation_steps", lora_grad_accum_steps)
        _set_path(payload, "sft_train.early_stopping_patience", LORA_EARLY_STOPPING_PATIENCE)
        _set_path(payload, "backbone_migration.lora_batch_policy", lora_batch_policy)
        _set_path(payload, "backbone_migration.lora_patience_policy", f"eval25_patience{LORA_EARLY_STOPPING_PATIENCE}")

    if args.finetune == "fullft":
        _set_path(payload, "sft_train.lora.enabled", False)
        _set_path(payload, "infer.merge_lora_cache.enabled", False)
        _set_path(payload, "sft_train.save_latest_state", True)
        _set_path(payload, "sft_train.resume_latest_state", True)
        _set_path(payload, "sft_train.cleanup_latest_state_on_complete", True)
        _set_path(payload, "backbone_migration.latest_state_policy", "accelerate_latest_state_for_interrupted_resume")
        if float(args.size_b) >= 7.0:
            _set_path(payload, "sft_train.per_device_train_batch_size", 1)
            _set_path(payload, "sft_train.per_device_eval_batch_size", 1)
            _set_path(payload, "sft_train.gradient_accumulation_steps", 8)
        fullft_deepspeed_config = str(args.deepspeed_config or "")
        if fullft_deepspeed_config:
            _set_path(payload, "train.deepspeed_config", fullft_deepspeed_config)
        if args.max_length is not None:
            _set_path(payload, "sft_train.max_length", int(args.max_length))
            _set_path(payload, "backbone_migration.max_length_policy", f"manual_max_length_{int(args.max_length)}")
        if backbone in {"gemma4_e4b", "ministral3_8b"}:
            effective_fullft_deepspeed_config = fullft_deepspeed_config or MULTIMODAL_TEXT_ONLY_FULLFT_DEEPSPEED_CONFIG
            if backbone == "gemma4_e4b" and not fullft_deepspeed_config:
                effective_fullft_deepspeed_config = GEMMA4_FULLFT_DEEPSPEED_CONFIG
            if backbone == "ministral3_8b" and model_variant != "text_only" and not fullft_deepspeed_config:
                effective_fullft_deepspeed_config = MISTRAL3_MULTIMODAL_FULLFT_DEEPSPEED_CONFIG
            _set_path(
                payload,
                "train.deepspeed_config",
                effective_fullft_deepspeed_config,
            )
            _set_path(payload, "sft_train.per_device_train_batch_size", 1)
            _set_path(payload, "sft_train.per_device_eval_batch_size", 1)
            _set_path(payload, "sft_train.gradient_accumulation_steps", 8)
            if backbone == "gemma4_e4b" and model_variant != "text_only":
                _set_path(payload, "sft_train.freeze_module_prefixes", GEMMA4_TEXT_ONLY_FULLFT_FREEZE_PREFIXES)
                _set_path(payload, "backbone_migration.freeze_policy", "gemma4_text_only_fullft_freeze_vision_audio")
            elif backbone == "ministral3_8b" and model_variant != "text_only":
                _set_path(payload, "sft_train.freeze_module_prefixes", MISTRAL3_MULTIMODAL_FULLFT_FREEZE_PREFIXES)
                _set_path(payload, "backbone_migration.freeze_policy", "mistral3_text_effective_fullft_freeze_vision_projector")
            _set_path(
                payload,
                "backbone_migration.fullft_batch_policy",
                f"{backbone}_{_deepspeed_policy_name(effective_fullft_deepspeed_config)}_bsz1_eval_bsz1_ga8",
            )
            _set_path(
                payload,
                "backbone_migration.deepspeed_policy",
                _deepspeed_policy_name(str(payload.get("train", {}).get("deepspeed_config", ""))),
            )
        elif float(args.size_b) >= 7.0:
            effective_fullft_deepspeed_config = fullft_deepspeed_config or LARGE_FULLFT_DEEPSPEED_CONFIG
            _set_path(payload, "train.deepspeed_config", effective_fullft_deepspeed_config)
            _set_path(
                payload,
                "backbone_migration.deepspeed_policy",
                _deepspeed_policy_name(effective_fullft_deepspeed_config),
            )
            _set_path(
                payload,
                "backbone_migration.fullft_batch_policy",
                f"{backbone}_{_deepspeed_policy_name(effective_fullft_deepspeed_config)}_bsz1_eval_bsz1_ga8",
            )

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--size-b", type=float, required=True)
    p.add_argument("--finetune", choices=sorted(BASE_EXPERIMENTS), required=True)
    p.add_argument("--case-name", required=True)
    p.add_argument("--base-experiment", default=None)
    p.add_argument("--model-variant", choices=["default", "text_only"], default="default")
    p.add_argument("--deepspeed-config", default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--ordinal-loss-alpha", type=float, default=None)
    p.add_argument(
        "--output-root",
        default="outputs/cache/backbone_migration/configs",
        help="Directory for generated resolved configs.",
    )
    p.add_argument("--print-path", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    payload = _prepare_config(args, project_root)
    output_root = Path(str(args.output_root))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_dir = output_root / str(args.finetune)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.case_name}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    if args.print_path:
        print(output_path)
    else:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
