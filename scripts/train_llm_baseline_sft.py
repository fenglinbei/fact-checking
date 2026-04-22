from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import yaml

from fact_checking.config import load_yaml
from fact_checking.utils.logging import init_logger
from sft import train_loop


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

    cfg = load_yaml(args.config)
    cfg = _normalize_prompt_truncation_config(cfg)
    cfg = _apply_runtime_output_layout(cfg)

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
    logger.info("SFT run summary: %s", run_summary)

    forwarded_config = _materialize_runtime_config(cfg)

    forwarded = ["--config", forwarded_config]
    if args.mini_val_size is not None:
        forwarded += ["--mini-val-size", str(args.mini_val_size)]
    if args.mini_val_seed is not None:
        forwarded += ["--mini-val-seed", str(args.mini_val_seed)]
    if args.prompt_length_stats_only:
        forwarded.append("--prompt-length-stats-only")

    sys.argv = ["train_loop", *forwarded]
    logger.info("Forwarding args to training loop: %s", forwarded)
    train_loop.main()


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


def _apply_runtime_output_layout(cfg: dict) -> dict:
    baseline_cfg = cfg.setdefault("baseline", {})
    train_cfg = cfg.setdefault("sft_train", {})

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    variant = str(baseline_cfg.get("variant", "")).strip() or timestamp
    run_output_dir = Path(cfg.get("output_dir", "outputs/liar-raw/llm_baseline")) / f"{variant}_{timestamp}"

    cfg["output_dir"] = str(run_output_dir)
    if not train_cfg.get("tokenized_cache_dir"):
        train_cfg["tokenized_cache_dir"] = str(run_output_dir / "tokenized_cache")
    return cfg


def _materialize_runtime_config(cfg: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="sft_runtime_",
        delete=False,
        dir=str(Path.cwd()),
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, allow_unicode=True, sort_keys=False)
    return tmp.name


if __name__ == "__main__":
    main()
