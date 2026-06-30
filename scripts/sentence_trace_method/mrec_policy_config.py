#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit shell exports for an MREC policy YAML.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = _load_yaml(config_path)

    experiment = dict(cfg.get("experiment") or {})
    paths = dict(cfg.get("paths") or {})
    trace = dict(cfg.get("trace") or {})
    prompt_evidence = dict(cfg.get("prompt_evidence") or {})
    guard = dict(prompt_evidence.get("max_length_guard") or {})
    train = dict(cfg.get("train") or {})
    sft_train = dict(cfg.get("sft_train") or {})
    lora = dict(sft_train.get("lora") or {})
    label_token_ce = dict(sft_train.get("label_token_ce") or {})
    build = dict(cfg.get("build") or {})
    data = dict(build.get("data") or {})
    trace_weight_training = dict(trace.get("weight_training") or {})

    policy = str(prompt_evidence.get("policy") or "prefix_topk")
    min_count = _int_or_default(prompt_evidence.get("min_evidence_count"), 0)
    max_count = _int_or_default(prompt_evidence.get("max_evidence_count"), 0)
    evidence_token_budget = prompt_evidence.get("evidence_token_budget")
    max_length_guard = "off"
    if bool(guard.get("enabled", False)):
        max_length_guard = str(guard.get("on_violation") or "warn")

    exports = {
        "MREC_POLICY_CONFIG": str(config_path),
        "ATOM_ANCHOR_ROOT": _get(paths, "atom_anchor_root", "outputs/selectors/atom_anchor/liar_raw_abc_v0_1"),
        "SOURCE_FEATURE_ROOT": _get(paths, "source_feature_root", ""),
        "TRACE_ROOT": _get(trace, "output_root", ""),
        "TRACE_BUILD_MODE": _get(trace, "build_mode", "mrec"),
        "TRACE_SHUFFLE_SOURCE_ROOT": _get(trace, "shuffle_source_root", ""),
        "TRACE_SHUFFLE_SEED": str(_int_or_default(trace.get("shuffle_seed"), 0)),
        "WEIGHT_FILE": _get(trace, "weight_file", ""),
        "EXPECTED_SELECTOR_NAME": _get(trace, "selector_name", ""),
        "EXPECTED_ADAPTIVE_POLICY": _get(trace, "adaptive_policy", ""),
        "EXPECTED_SELECTION_POLICY": _get(trace, "selection_policy", ""),
        "SOURCE_SELECTOR_NAME": _get(trace, "source_selector_name", ""),
        "TRACE_CANDIDATE_TOP_N": str(_int_or_default(trace.get("candidate_top_n"), 0)),
        "TRACE_MAX_STEPS": str(_int_or_default(trace.get("max_steps"), 0)),
        "TRACE_MIN_STEPS": str(_int_or_default(trace.get("min_steps"), 0)),
        "TRACE_TOKEN_BUDGET": "" if trace.get("token_budget") is None else str(trace.get("token_budget")),
        "TRACE_TARGET_RESOLVED_RATE": str(trace.get("target_resolved_rate", 1.0)),
        "TRACE_STOP_THRESHOLD": str(trace.get("stop_threshold", -1000000000)),
        "TRACE_POST_TARGET_FILL_POLICY": _get(trace, "post_target_fill_policy", "contrast_only"),
        "TRACE_CONTINUE_AFTER_TARGET_FOR_CONTRAST": _bool_text(
            trace.get("continue_after_target_for_contrast", False)
        ),
        "MREC_SPLITS": _csv(trace.get("splits"), "train,val,test"),
        "MREC_AUTO_TRAIN_WEIGHTS": _bool_text(trace_weight_training.get("enabled", False)),
        "MREC_WEIGHT_OUTPUT_DIR": _get(
            trace_weight_training,
            "output_dir",
            str(Path(_get(trace, "weight_file", "weights.json")).parent),
        ),
        "MREC_WEIGHT_CANDIDATE_TOP_N": str(
            _int_or_default(trace_weight_training.get("candidate_top_n"), 20)
        ),
        "MREC_WEIGHT_ROLLOUT_STEPS": str(
            _int_or_default(trace_weight_training.get("rollout_steps"), 5)
        ),
        "MREC_WEIGHT_EPOCHS": str(_int_or_default(trace_weight_training.get("epochs"), 50)),
        "MREC_WEIGHT_LEARNING_RATE": str(trace_weight_training.get("learning_rate", 0.05)),
        "MREC_WEIGHT_SAMPLE_LIMIT": str(_int_or_default(trace_weight_training.get("sample_limit"), 0)),
        "MREC_WEIGHT_TRAIN_SAMPLE_LIMIT": str(
            _int_or_default(trace_weight_training.get("train_sample_limit"), 0)
        ),
        "MREC_WEIGHT_VAL_SAMPLE_LIMIT": str(
            _int_or_default(trace_weight_training.get("val_sample_limit"), 0)
        ),
        "CONFIG_PATH": str(config_path),
        "CASE_SUFFIX": _get(experiment, "case_suffix", ""),
        "BASE_CASE_NAME": _get(experiment, "base_case_name", _get(data, "dataset", "liar_raw") + "__ministral3_8b"),
        "OUTPUT_ROOT": _get(experiment, "output_root", "outputs/sentence_trace_method"),
        "LORA_SUFFIX": _get(experiment, "lora_suffix", "_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"),
        "RUN_LABEL": _get(experiment, "run_label", "atom-anchor-v0.2-fullpool-policy"),
        "RUN_HEADER_LABEL": _get(experiment, "run_header_label", "atom-anchor-v0.2-fullpool-policy-full"),
        "WRAPPER_SWANLAB_PROJECT": _get(
            cfg.get("swanlab") or {},
            "project",
            "fact-checking-sentence-trace-method-atom-anchor-v0-2",
        ),
        "QUALITY_AUDIT": _get(paths, "quality_audit", ""),
        "QUALITY_AUDIT_MODE": _get(paths, "quality_audit_mode", "full"),
        "DATASET": _get(data, "dataset", _get(cfg, "dataset", "liar_raw")),
        "LABEL_SCHEMA": _get(data, "label_schema", _get(cfg, "label_schema", "liar6")),
        "TRAIN_RAW": _get(data, "train_path", "data/raw/LIAR-RAW/train.json"),
        "VAL_RAW": _get(data, "val_path", "data/raw/LIAR-RAW/val.json"),
        "TEST_RAW": _get(data, "test_path", "data/raw/LIAR-RAW/test.json"),
        "TRACE_PROMPT_STYLE": _get(prompt_evidence, "trace_prompt_style", "mrec_min"),
        "EVIDENCE_TEXT_MODE": _get(prompt_evidence, "evidence_text_mode", "full"),
        "TRACE_TOP_K": str(max_count or 0),
        "PROMPT_EVIDENCE_POLICY": policy,
        "PROMPT_EVIDENCE_MIN_COUNT": str(min_count),
        "PROMPT_EVIDENCE_MAX_COUNT": str(max_count),
        "PROMPT_EVIDENCE_TOKEN_BUDGET": "" if evidence_token_budget is None else str(evidence_token_budget),
        "PROMPT_EVIDENCE_MAX_LENGTH_GUARD": max_length_guard,
        "MODEL_PATH": _get(train, "model_name_or_path", ""),
        "DEEPSPEED_CONFIG": _get(train, "deepspeed_config", ""),
        "NPROC_PER_NODE": str(_int_or_default(train.get("nproc_per_node"), 4)),
        "SFT_GRADIENT_ACCUMULATION_STEPS": str(_int_or_default(sft_train.get("gradient_accumulation_steps"), 4)),
        "SFT_LEARNING_RATE": str(sft_train.get("learning_rate", "2e-5")),
        "SFT_NUM_TRAIN_EPOCHS": str(sft_train.get("num_train_epochs", 12)),
        "SFT_EVAL_STEPS": str(_int_or_default(sft_train.get("eval_steps"), 100)),
        "SFT_SAVE_STEPS": str(_int_or_default(sft_train.get("save_steps"), sft_train.get("eval_steps") or 100)),
        "SFT_EARLY_STOPPING_PATIENCE": str(_int_or_default(sft_train.get("early_stopping_patience"), 12)),
        "SFT_EARLY_STOPPING_METRIC": _get(label_token_ce, "early_stopping_metric", "macro_f1"),
        "LORA_R": str(_int_or_default(lora.get("r"), 16)),
        "LORA_ALPHA": str(_int_or_default(lora.get("alpha"), 32)),
        "LORA_DROPOUT": str(lora.get("dropout", 0.1)),
        "LORA_BIAS": _get(lora, "bias", "none"),
        "CLASS_WEIGHTS": _class_weights(label_token_ce),
        "LIAR_CLASS_WEIGHTS": _class_weights(label_token_ce),
        "EVAL_SPLITS": _csv((cfg.get("eval") or {}).get("splits"), "val,test"),
        "RUN_TAU_EVAL": str((cfg.get("eval") or {}).get("run_tau_eval", "auto")),
        "TAUS": _csv((cfg.get("eval") or {}).get("taus"), "0.75"),
    }

    if not exports["SOURCE_FEATURE_ROOT"]:
        exports["SOURCE_FEATURE_ROOT"] = str(Path(exports["ATOM_ANCHOR_ROOT"]) / "04_evidence_map")
    if not exports["QUALITY_AUDIT"]:
        exports["QUALITY_AUDIT"] = str(Path(exports["ATOM_ANCHOR_ROOT"]) / "quality_audit_after_fix.json")

    for key, value in exports.items():
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    parent = payload.pop("extends", None)
    if not parent:
        return payload
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = _load_yaml(parent_path)
    return _deep_merge(base, payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _get(mapping: dict[str, Any], key: str, default: str) -> str:
    value = mapping.get(key)
    if value is None:
        return default
    return str(value)


def _int_or_default(value: Any, default: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _csv(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _class_weights(label_token_ce: dict[str, Any]) -> str:
    weights = label_token_ce.get("class_weights")
    if not isinstance(weights, dict):
        return "pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8"
    return ",".join(f"{key}={value}" for key, value in weights.items())


if __name__ == "__main__":
    raise SystemExit(main())
