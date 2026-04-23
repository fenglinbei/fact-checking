from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from fact_checking.config import load_yaml
from fact_checking.utils.logging import init_logger
from sft import trainer
from sft.utils import _normalize_prompt_truncation_config, _apply_runtime_output_layout, _is_main_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin entry for SFT training loop.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mini-val-size", type=int, default=None)
    parser.add_argument("--mini-val-seed", type=int, default=None)
    parser.add_argument("--prompt-length-stats-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = init_logger(__name__)
    is_main_process = _is_main_process()

    cfg = load_yaml(args.config)
    cfg = _normalize_prompt_truncation_config(cfg)
    cfg = _apply_runtime_output_layout(cfg)

    if is_main_process:
        log_dir = Path(cfg["output_dir"]) / "logs"
        logger = init_logger(__name__, log_dir=log_dir, log_filename="train_llm_baseline_sft.log")

    baseline_cfg = cfg.get("baseline", {})
    train_cfg = cfg.get("sft_train", {})
    run_summary = {
        "config_path": args.config,
        "output_dir": cfg.get("output_dir"),
        "backbone_model": baseline_cfg.get("model_name_or_path"),
        "embedder_model": baseline_cfg.get("retrieval_model"),
        "device": train_cfg.get("device", "auto"),
        "cuda_available": torch.cuda.is_available(),
        "top_k": baseline_cfg.get("top_k"),
        "batch_size": train_cfg.get("per_device_train_batch_size"),
        "max_length": train_cfg.get("max_length"),
        "epochs": train_cfg.get("num_train_epochs"),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps"),
    }
    if is_main_process:
        logger.info("SFT run summary: %s", run_summary)

    forwarded = ["--config", args.config]
    if args.mini_val_size is not None:
        forwarded += ["--mini-val-size", str(args.mini_val_size)]
    if args.mini_val_seed is not None:
        forwarded += ["--mini-val-seed", str(args.mini_val_seed)]
    if args.prompt_length_stats_only:
        forwarded.append("--prompt-length-stats-only")

    sys.argv = ["train_loop", *forwarded]
    if is_main_process:
        logger.info("Forwarding args to training loop: %s", forwarded)
    trainer.main()


if __name__ == "__main__":
    main()
