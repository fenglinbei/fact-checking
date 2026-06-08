#!/usr/bin/env python
"""Patch a resolved label-token trainer config with method-upgrade parameters.

Usage:
    python patch_config_for_upgrade.py \\
        --config path/to/train.resolved.yaml \\
        --weight-decay 0.01 \\
        --alpha-warmup-ratio 0.3 \\
        --lr-scheduler cosine_with_restarts \\
        --lr-kwargs '{"num_cycles": 2}' \\
        --early-stopping-metric macro_f1_plus_true_side_plus_mae \\
        --mae-metric-weight 0.3

All arguments are optional — only specified overrides are applied.
The input config is modified in-place unless --output is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def _set_path(payload: dict[str, Any], dotted_path: str, value: object) -> None:
    keys = dotted_path.split(".")
    current: Any = payload
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, required=True, help="Path to resolved YAML config")
    p.add_argument("--output", type=str, default=None, help="Output path (default: overwrite input)")
    p.add_argument("--weight-decay", type=float, default=None, help="Set sft_train.weight_decay")
    p.add_argument("--alpha-warmup-ratio", type=float, default=None,
                   help="Set ordinal_loss.alpha_warmup_ratio (0.0-1.0)")
    p.add_argument("--lr-scheduler", type=str, default=None,
                   help="Set sft_train.lr_scheduler_type (e.g. cosine_with_restarts)")
    p.add_argument("--lr-kwargs", type=str, default=None,
                   help="JSON dict for sft_train.lr_scheduler_kwargs")
    p.add_argument("--early-stopping-metric", type=str, default=None,
                   help="Set label_token_ce.early_stopping_metric")
    p.add_argument("--mae-metric-weight", type=float, default=None,
                   help="Set label_token_ce.mae_metric_weight")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    import yaml
    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    patches: list[tuple[str, str, object]] = []

    if args.weight_decay is not None:
        patches.append(("sft_train.weight_decay", "weight_decay", float(args.weight_decay)))

    if args.alpha_warmup_ratio is not None:
        val = float(args.alpha_warmup_ratio)
        patches.append(("sft_train.label_token_ce.ordinal_loss.alpha_warmup_ratio",
                        "alpha_warmup_ratio", val))

    if args.lr_scheduler is not None:
        patches.append(("sft_train.lr_scheduler_type", "lr_scheduler_type",
                        str(args.lr_scheduler)))

    if args.lr_kwargs is not None:
        lr_kwargs = json.loads(args.lr_kwargs)
        patches.append(("sft_train.lr_scheduler_kwargs", "lr_scheduler_kwargs", lr_kwargs))

    if args.early_stopping_metric is not None:
        patches.append(("sft_train.label_token_ce.early_stopping_metric",
                        "early_stopping_metric", str(args.early_stopping_metric)))

    if args.mae_metric_weight is not None:
        patches.append(("sft_train.label_token_ce.mae_metric_weight",
                        "mae_metric_weight", float(args.mae_metric_weight)))

    if not patches:
        print("No patches specified — config unchanged.", file=sys.stderr)
        sys.exit(0)

    for dotted_path, name, value in patches:
        _set_path(payload, dotted_path, value)
        print(f"  [patch] {dotted_path} = {value!r}")

    output_path = Path(args.output) if args.output else config_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  [saved] {output_path}")


if __name__ == "__main__":
    main()
