#!/usr/bin/env python3
"""Prepare a self-contained LoRA train config from a clean sentence-trace build."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from fact_checking.config import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, help="FullFT train.resolved.yaml produced by run_one.sh MODE=build.")
    parser.add_argument("--output-root", required=True, help="LoRA case output root, e.g. outputs/.../case_lora.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--swanlab-project", default="fact-checking-sentence-trace-method-lora")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--bias", default="none")
    parser.add_argument("--deepspeed-config", default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--logit-adjust-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--logit-adjust-tau", type=float, default=None)
    parser.add_argument("--coverage-label-token-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--coverage-label-token-loss-weight", type=float, default=None)
    parser.add_argument("--coverage-label-token-prefix", default=None)
    parser.add_argument(
        "--class-weight",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT",
        help="Override sft_train.label_token_ce.class_weights. May be passed multiple times.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-build", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_config = Path(args.source_config)
    output_root = Path(args.output_root)
    output_config = output_root / "train.resolved.yaml"
    class_weights = _parse_class_weight_overrides(args.class_weight)
    logit_adjust_enabled = _parse_optional_bool(args.logit_adjust_enabled)
    coverage_label_token_enabled = _parse_optional_bool(args.coverage_label_token_enabled)
    if output_config.exists() and not args.force:
        _sync_existing_config(
            output_config,
            deepspeed_config=args.deepspeed_config,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            early_stopping_patience=args.early_stopping_patience,
            logit_adjust_enabled=logit_adjust_enabled,
            logit_adjust_tau=args.logit_adjust_tau,
            coverage_label_token_enabled=coverage_label_token_enabled,
            coverage_label_token_loss_weight=args.coverage_label_token_loss_weight,
            coverage_label_token_prefix=args.coverage_label_token_prefix,
            class_weights=class_weights,
        )
        print(output_config)
        return 0

    cfg = load_yaml(source_config)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.copy_build:
        _copy_build_files(cfg, output_root=output_root)

    cfg["output_dir"] = str(output_root / "train")
    cfg["eval_output_dir"] = str(output_root / "eval")
    cfg["prompt_stats_output_dir"] = str(output_root / "prompt_stats")
    if args.deepspeed_config:
        train_cfg = dict(cfg.get("train") or {})
        train_cfg["deepspeed_config"] = str(args.deepspeed_config)
        cfg["train"] = train_cfg

    sft_train = dict(cfg.get("sft_train") or {})
    lora = dict(sft_train.get("lora") or {})
    lora.update(
        {
            "enabled": True,
            "r": int(args.r),
            "alpha": int(args.alpha),
            "dropout": float(args.dropout),
            "bias": str(args.bias),
        }
    )
    sft_train["lora"] = lora
    sft_train["resolved_output_dir"] = True
    sft_train.setdefault("save_latest_state", True)
    sft_train.setdefault("resume_latest_state", True)
    cfg["sft_train"] = sft_train
    _apply_sft_overrides(
        cfg,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        logit_adjust_enabled=logit_adjust_enabled,
        logit_adjust_tau=args.logit_adjust_tau,
        coverage_label_token_enabled=coverage_label_token_enabled,
        coverage_label_token_loss_weight=args.coverage_label_token_loss_weight,
        coverage_label_token_prefix=args.coverage_label_token_prefix,
        class_weights=class_weights,
    )

    swanlab = dict(cfg.get("swanlab") or {})
    swanlab["project"] = str(args.swanlab_project)
    swanlab["experiment_name"] = str(args.experiment_name)
    cfg["swanlab"] = swanlab

    save_yaml(cfg, output_config)
    print(output_config)
    return 0


def _copy_build_files(cfg: dict[str, Any], *, output_root: Path) -> None:
    data = dict(cfg.get("data") or {})
    build_dir = output_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    remapped: dict[str, str] = {}
    for split in ("train", "val", "test"):
        key = f"{split}_candidates"
        src = Path(str(data.get(key) or ""))
        if not src.exists():
            raise FileNotFoundError(f"Missing build file for {key}: {src}")
        dst = build_dir / f"build_{split}.jsonl"
        shutil.copy2(src, dst)
        remapped[key] = str(dst)
    data.update(remapped)
    cfg["data"] = data

    source_build_dir = Path(str(data["train_candidates"])).parent
    source_report = source_build_dir / "build_report.json"
    if source_report.exists():
        shutil.copy2(source_report, build_dir / "build_report.json")


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


def _parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.strip().lower() == "true"


def _apply_sft_overrides(
    cfg: dict[str, Any],
    *,
    gradient_accumulation_steps: int | None,
    num_train_epochs: float | None,
    eval_steps: int | None,
    save_steps: int | None,
    early_stopping_patience: int | None,
    logit_adjust_enabled: bool | None,
    logit_adjust_tau: float | None,
    coverage_label_token_enabled: bool | None,
    coverage_label_token_loss_weight: float | None,
    coverage_label_token_prefix: str | None,
    class_weights: dict[str, float],
) -> bool:
    sft_train = dict(cfg.get("sft_train") or {})
    changed = False

    scalar_overrides: dict[str, int | float | None] = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "early_stopping_patience": early_stopping_patience,
    }
    for key, value in scalar_overrides.items():
        if value is not None and sft_train.get(key) != value:
            sft_train[key] = value
            changed = True

    if logit_adjust_enabled is not None or logit_adjust_tau is not None:
        logit_adjust = dict(sft_train.get("logit_adjust") or {})
        if logit_adjust_enabled is not None and bool(logit_adjust.get("enabled", False)) != logit_adjust_enabled:
            logit_adjust["enabled"] = logit_adjust_enabled
            changed = True
        if logit_adjust_tau is not None and float(logit_adjust.get("tau", 1.0)) != float(logit_adjust_tau):
            logit_adjust["tau"] = float(logit_adjust_tau)
            changed = True
        sft_train["logit_adjust"] = logit_adjust

    if class_weights:
        label_token_ce = dict(sft_train.get("label_token_ce") or {})
        existing_weights = dict(label_token_ce.get("class_weights") or {})
        for label, weight in class_weights.items():
            if float(existing_weights.get(label, 1.0)) != float(weight):
                existing_weights[label] = float(weight)
                changed = True
        label_token_ce["class_weights"] = existing_weights
        sft_train["label_token_ce"] = label_token_ce

    if (
        coverage_label_token_enabled is not None
        or coverage_label_token_loss_weight is not None
        or coverage_label_token_prefix is not None
    ):
        coverage_label_token = dict(sft_train.get("coverage_label_token") or {})
        if (
            coverage_label_token_enabled is not None
            and bool(coverage_label_token.get("enabled", False)) != coverage_label_token_enabled
        ):
            coverage_label_token["enabled"] = coverage_label_token_enabled
            changed = True
        if (
            coverage_label_token_loss_weight is not None
            and float(coverage_label_token.get("loss_weight", 1.0)) != float(coverage_label_token_loss_weight)
        ):
            coverage_label_token["loss_weight"] = float(coverage_label_token_loss_weight)
            changed = True
        if coverage_label_token_prefix is not None and coverage_label_token.get("label_prefix") != coverage_label_token_prefix:
            coverage_label_token["label_prefix"] = str(coverage_label_token_prefix)
            changed = True
        sft_train["coverage_label_token"] = coverage_label_token

    if changed:
        cfg["sft_train"] = sft_train
    return changed


def _sync_existing_config(
    output_config: Path,
    *,
    deepspeed_config: str | None,
    gradient_accumulation_steps: int | None,
    num_train_epochs: float | None,
    eval_steps: int | None,
    save_steps: int | None,
    early_stopping_patience: int | None,
    logit_adjust_enabled: bool | None,
    logit_adjust_tau: float | None,
    coverage_label_token_enabled: bool | None,
    coverage_label_token_loss_weight: float | None,
    coverage_label_token_prefix: str | None,
    class_weights: dict[str, float],
) -> None:
    cfg = load_yaml(output_config)
    changed = False

    train_cfg = dict(cfg.get("train") or {})
    if deepspeed_config and train_cfg.get("deepspeed_config") != deepspeed_config:
        train_cfg["deepspeed_config"] = deepspeed_config
        cfg["train"] = train_cfg
        changed = True

    sft_train = dict(cfg.get("sft_train") or {})
    for key in ("save_latest_state", "resume_latest_state"):
        if key not in sft_train:
            sft_train[key] = True
            changed = True
    cfg["sft_train"] = sft_train

    changed = (
        _apply_sft_overrides(
            cfg,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            eval_steps=eval_steps,
            save_steps=save_steps,
            early_stopping_patience=early_stopping_patience,
            logit_adjust_enabled=logit_adjust_enabled,
            logit_adjust_tau=logit_adjust_tau,
            coverage_label_token_enabled=coverage_label_token_enabled,
            coverage_label_token_loss_weight=coverage_label_token_loss_weight,
            coverage_label_token_prefix=coverage_label_token_prefix,
            class_weights=class_weights,
        )
        or changed
    )
    if changed:
        save_yaml(cfg, output_config)


if __name__ == "__main__":
    raise SystemExit(main())
