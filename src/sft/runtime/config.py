from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def apply_runtime_output_layout(cfg: dict) -> dict:
    baseline_cfg = cfg.setdefault("baseline", {})
    train_cfg = cfg.setdefault("sft_train", {})

    if bool(train_cfg.get("resolved_output_dir", False)):
        run_output_dir = Path(cfg.get("output_dir", "outputs/runs/train"))
        cfg["output_dir"] = str(run_output_dir)
        if not train_cfg.get("tokenized_cache_dir"):
            train_cfg["tokenized_cache_dir"] = str(run_output_dir / "tokenized_cache")
        return cfg

    timestamp = os.environ.get("FC_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")
    variant = str(cfg.get("experiment", {}).get("name") or baseline_cfg.get("variant", "")).strip() or timestamp
    run_output_dir = Path(cfg.get("output_dir", "outputs/runs/train")) / f"{variant}_{timestamp}"

    cfg["output_dir"] = str(run_output_dir)
    if not train_cfg.get("tokenized_cache_dir"):
        train_cfg["tokenized_cache_dir"] = str(run_output_dir / "tokenized_cache")
    return cfg
