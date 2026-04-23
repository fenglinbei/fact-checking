import datetime
import os

import numpy as np
from numpy.compat import Path
import torch

from accelerate import Accelerator
from fact_checking.data.constants import LABEL2ID
from fact_checking.utils.logging import init_logger

logger = init_logger(__name__)

def _normalize_gold_label(row: dict) -> str:
    gold_label = str(row.get("label", "")).strip().lower()
    if gold_label not in LABEL2ID:
        return ""
    return gold_label

def _normalize_prompt_truncation_config(cfg: dict) -> dict:
    baseline_cfg = cfg.setdefault("baseline", {})
    trunc_cfg = baseline_cfg.setdefault("prompt_truncation", {})
    if "enabled" not in trunc_cfg:
        trunc_cfg["enabled"] = False
    if "strategy" not in trunc_cfg:
        trunc_cfg["strategy"] = "tail_evidence"
    if "min_evidence_to_keep" not in trunc_cfg:
        trunc_cfg["min_evidence_to_keep"] = 1
    return cfg


def _print_prompt_summary(summary: dict[str, object]) -> None:
    before = summary["prompt_length_before_truncation"]
    after = summary["prompt_length_after_truncation"]
    trunc = summary["evidence_truncation"]
    logger.info(
        "[PROMPT_STATS] split=%s mode=%s strategy=%s pre_mean=%.2f pre_p95=%.0f pre_overflow=%d "
        "post_mean=%.2f post_p95=%.0f post_overflow=%d truncated=%s trunc_rate=%.4f",
        summary["split"],
        summary["output_mode"],
        summary["prompt_truncation_strategy"],
        before["mean"],
        before["p95"],
        int(before["overflow_count"]),
        after["mean"],
        after["p95"],
        int(after["overflow_count"]),
        trunc["truncated_count"],
        trunc["truncation_rate"],
    )

def _select_mini_val_rows(
    rows: list[dict],
    mini_val_size: int,
    mini_val_seed: int,
    accelerator: Accelerator,
) -> list[dict]:
    if mini_val_size <= 0 or mini_val_size >= len(rows):
        return rows

    rng = np.random.default_rng(mini_val_seed)
    indices = rng.choice(len(rows), size=mini_val_size, replace=False)
    mini_rows = [rows[int(i)] for i in indices.tolist()]
    if accelerator.is_main_process:
        logger.info(
            "[INFO] mini-val enabled: sampled %d / %d validation rows (seed=%d).",
            len(mini_rows),
            len(rows),
            mini_val_seed,
        )
    return mini_rows

def maybe_empty_cache(accelerator: Accelerator) -> None:
    if torch.cuda.is_available():
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

def _apply_runtime_output_layout(cfg: dict) -> dict:
    baseline_cfg = cfg.setdefault("baseline", {})
    train_cfg = cfg.setdefault("sft_train", {})

    timestamp = os.environ.get("FC_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")
    variant = str(baseline_cfg.get("variant", "")).strip() or timestamp
    run_output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / f"{variant}_{timestamp}"

    cfg["output_dir"] = str(run_output_dir)
    if not train_cfg.get("tokenized_cache_dir"):
        train_cfg["tokenized_cache_dir"] = str(run_output_dir / "tokenized_cache")
    return cfg

def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0