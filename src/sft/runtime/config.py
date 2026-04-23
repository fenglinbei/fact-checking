from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def normalize_prompt_truncation_config(cfg: dict) -> dict:
    baseline_cfg = cfg.setdefault("baseline", {})
    trunc_cfg = baseline_cfg.setdefault("prompt_truncation", {})
    if "enabled" not in trunc_cfg:
        trunc_cfg["enabled"] = False
    if "strategy" not in trunc_cfg:
        trunc_cfg["strategy"] = "tail_evidence"
    if "min_evidence_to_keep" not in trunc_cfg:
        trunc_cfg["min_evidence_to_keep"] = 1
    return cfg


def apply_runtime_output_layout(cfg: dict) -> dict:
    baseline_cfg = cfg.setdefault("baseline", {})
    train_cfg = cfg.setdefault("sft_train", {})

    timestamp = os.environ.get("FC_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")
    variant = str(baseline_cfg.get("variant", "")).strip() or timestamp
    run_output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / f"{variant}_{timestamp}"

    cfg["output_dir"] = str(run_output_dir)
    if not train_cfg.get("tokenized_cache_dir"):
        train_cfg["tokenized_cache_dir"] = str(run_output_dir / "tokenized_cache")
    return cfg
