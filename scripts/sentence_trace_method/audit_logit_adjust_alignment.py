#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from fact_checking.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether train config, saved train priors, and eval metrics used aligned logit_adjust settings."
    )
    parser.add_argument("--case-root", required=True, help="Run case root containing train.resolved.yaml and eval/.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional resolved config path. Defaults to train.resolved.yaml under --case-root.",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_root = Path(args.case_root)
    report = build_report(case_root=case_root, config_path=Path(args.config) if args.config else None)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        _print_text(report)
    return 0


def build_report(*, case_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    resolved_config_path = _resolve_config_path(case_root=case_root, config_path=config_path)
    config = _config_logit_adjust(resolved_config_path)
    train_saved_path = case_root / "train" / "logit_adjust.json"
    train_saved = _saved_logit_adjust(train_saved_path)

    eval_rows: list[dict[str, Any]] = []
    for metrics_path in sorted((case_root / "eval").glob("*/*/*/metrics.json")):
        row = _eval_logit_adjust(metrics_path, case_root=case_root)
        row["matches_config"] = _settings_match(row, config)
        row["matches_train_saved"] = _settings_match(row, train_saved)
        eval_rows.append(row)

    mismatched_config = sum(1 for row in eval_rows if not row["matches_config"])
    mismatched_train_saved = sum(1 for row in eval_rows if not row["matches_train_saved"])
    return {
        "case_root": str(case_root),
        "config_path": str(resolved_config_path) if resolved_config_path else None,
        "config": config,
        "train_saved_path": str(train_saved_path),
        "train_saved": train_saved,
        "eval_metrics": eval_rows,
        "summary": {
            "eval_metrics_count": len(eval_rows),
            "mismatched_config_count": mismatched_config,
            "mismatched_train_saved_count": mismatched_train_saved,
        },
    }


def _resolve_config_path(*, case_root: Path, config_path: Path | None) -> Path | None:
    if config_path is not None:
        return config_path
    for candidate in (
        case_root / "train.resolved.yaml",
        case_root / "train" / "config.resolved.yaml",
        case_root / "configs" / "train.resolved.yaml",
    ):
        if candidate.exists():
            return candidate
    return case_root / "train.resolved.yaml"


def _config_logit_adjust(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {"exists": False, "enabled": False, "tau": 1.0}
    cfg = load_yaml(config_path) or {}
    sft_train = cfg.get("sft_train", {}) or {}
    block = sft_train.get("logit_adjust", {}) or {}
    return {
        "exists": True,
        "enabled": bool(block.get("enabled", False)),
        "tau": float(block.get("tau", 1.0)),
    }


def _saved_logit_adjust(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "enabled": False, "tau": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_settings(payload, exists=True)


def _eval_logit_adjust(metrics_path: Path, *, case_root: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    settings = _normalize_settings(metrics.get("logit_adjust"), exists="logit_adjust" in metrics)
    rel_parts = metrics_path.relative_to(case_root).parts
    row = {
        "path": str(metrics_path),
        "split": rel_parts[1] if len(rel_parts) > 1 else None,
        "checkpoint": rel_parts[2] if len(rel_parts) > 2 else None,
        "eval_name": rel_parts[3] if len(rel_parts) > 3 else None,
    }
    row.update(settings)
    return row


def _normalize_settings(payload: Any, *, exists: bool) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"exists": bool(exists), "enabled": False, "tau": None}
    enabled = bool(payload.get("enabled", False))
    tau_value = payload.get("tau", 1.0 if enabled else None)
    return {
        "exists": bool(exists),
        "enabled": enabled,
        "tau": float(tau_value) if tau_value is not None else None,
    }


def _settings_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if bool(left.get("enabled", False)) != bool(right.get("enabled", False)):
        return False
    if not bool(left.get("enabled", False)):
        return bool(left.get("exists", True)) == bool(right.get("exists", True))
    left_tau = left.get("tau")
    right_tau = right.get("tau")
    return left_tau is not None and right_tau is not None and abs(float(left_tau) - float(right_tau)) <= 1e-12


def _print_text(report: Mapping[str, Any]) -> None:
    print(f"case_root: {report['case_root']}")
    print(f"config: enabled={report['config']['enabled']} tau={report['config']['tau']}")
    print(f"train_saved: enabled={report['train_saved']['enabled']} tau={report['train_saved']['tau']}")
    print(
        "eval_metrics: "
        f"{report['summary']['eval_metrics_count']} "
        f"config_mismatch={report['summary']['mismatched_config_count']} "
        f"train_saved_mismatch={report['summary']['mismatched_train_saved_count']}"
    )
    for row in report["eval_metrics"]:
        print(
            f"- {row['split']}/{row['checkpoint']}/{row['eval_name']}: "
            f"enabled={row['enabled']} tau={row['tau']} "
            f"matches_config={row['matches_config']} "
            f"matches_train_saved={row['matches_train_saved']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
