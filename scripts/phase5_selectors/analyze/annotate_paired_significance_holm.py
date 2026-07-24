#!/usr/bin/env python3
"""Annotate paired-significance output with a preregistered Holm family."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--primary-comparison", required=True)
    parser.add_argument("--secondary-comparison", action="append", default=[])
    parser.add_argument("--diagnostic-comparison", action="append", default=[])
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def holm_adjust(p_values: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Return monotone Holm adjusted p-values keyed by comparison name."""

    ordered = sorted((str(name), float(value)) for name, value in p_values)
    ordered.sort(key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")
        running = max(running, (m - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def annotate(
    payload: dict[str, Any],
    *,
    primary: str,
    secondary: Sequence[str],
    diagnostic: Sequence[str],
    metric: str,
    alpha: float,
) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("input has no comparisons array")
    by_name = {
        str(row.get("name") or ""): row
        for row in comparisons
        if isinstance(row, dict)
    }
    requested = [primary, *secondary, *diagnostic]
    if len(set(requested)) != len(requested):
        raise ValueError("comparison families must be disjoint")
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"unknown comparisons: {missing}")

    p_values = []
    for name in secondary:
        try:
            value = by_name[name]["paired_randomization"][metric]["p_value_two_sided"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name} has no paired-randomization p-value for {metric}") from exc
        p_values.append((name, float(value)))
    adjusted = holm_adjust(p_values)
    payload["multiple_testing"] = {
        "primary_comparison": primary,
        "primary_inference": "unadjusted preregistered primary",
        "secondary_family": list(secondary),
        "secondary_method": "Holm step-down",
        "metric": metric,
        "alpha": float(alpha),
        "holm_adjusted_p_values": adjusted,
        "holm_reject": {name: value <= alpha for name, value in adjusted.items()},
        "diagnostic_comparisons_excluded_from_family": list(diagnostic),
    }
    return payload


def main() -> int:
    args = parse_args()
    path = Path(args.input_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotate(
        payload,
        primary=str(args.primary_comparison),
        secondary=list(args.secondary_comparison),
        diagnostic=list(args.diagnostic_comparison),
        metric=str(args.metric),
        alpha=float(args.alpha),
    )
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
