#!/usr/bin/env python3
"""Apply late-bound train config overrides for sentence-trace runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fact_checking.config import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deepspeed-config", default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--swanlab-project", default=None)
    parser.add_argument("--swanlab-experiment-name", default=None)
    parser.add_argument(
        "--class-weight",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT",
        help="Override sft_train.label_token_ce.class_weights. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    class_weights = _parse_class_weight_overrides(args.class_weight)
    changed = False

    if args.deepspeed_config:
        train_cfg = dict(cfg.get("train") or {})
        if train_cfg.get("deepspeed_config") != args.deepspeed_config:
            train_cfg["deepspeed_config"] = args.deepspeed_config
            cfg["train"] = train_cfg
            changed = True

    sft_train = dict(cfg.get("sft_train") or {})
    for key, value in _sft_scalar_overrides(args).items():
        if value is not None and sft_train.get(key) != value:
            sft_train[key] = value
            changed = True
    if class_weights:
        label_token_ce = dict(sft_train.get("label_token_ce") or {})
        existing_weights = dict(label_token_ce.get("class_weights") or {})
        for label, weight in class_weights.items():
            if float(existing_weights.get(label, 1.0)) != float(weight):
                existing_weights[label] = float(weight)
                changed = True
        label_token_ce["class_weights"] = existing_weights
        sft_train["label_token_ce"] = label_token_ce
    if changed:
        cfg["sft_train"] = sft_train

    if args.swanlab_project or args.swanlab_experiment_name:
        swanlab = dict(cfg.get("swanlab") or {})
        if args.swanlab_project and swanlab.get("project") != args.swanlab_project:
            swanlab["project"] = args.swanlab_project
            changed = True
        if args.swanlab_experiment_name and swanlab.get("experiment_name") != args.swanlab_experiment_name:
            swanlab["experiment_name"] = args.swanlab_experiment_name
            changed = True
        cfg["swanlab"] = swanlab

    if changed:
        save_yaml(cfg, config_path)
    print(config_path)
    return 0


def _sft_scalar_overrides(args: argparse.Namespace) -> dict[str, int | float | None]:
    return {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
    }


def _parse_class_weight_overrides(raw_items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for raw in raw_items or []:
        if "=" not in raw:
            raise ValueError(f"--class-weight must use LABEL=WEIGHT format, got: {raw}")
        label, value = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"--class-weight label cannot be empty: {raw}")
        overrides[label] = float(value)
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
