#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.mrec_learned_marginal import (  # noqa: E402
    evaluate_learned_marginal_reward_weights,
    learned_marginal_weight_fingerprint,
    load_learned_marginal_weights,
    save_learned_marginal_weights,
    train_learned_marginal_reward_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MREC v0.2 learned marginal reward weights.")
    parser.add_argument("--train-reward-input", required=True, help="reward_records_train.jsonl")
    parser.add_argument("--val-reward-input", required=True, help="reward_records_val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-weight-file", default="", help="Optional proxy/reward weights used for initialization and prior regularization.")
    parser.add_argument("--sample-limit", type=int, default=0, help="Shared limit for train and val; 0 disables.")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--val-sample-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.2)
    parser.add_argument("--huber-weight", type=float, default=0.2)
    parser.add_argument("--prior-weight", type=float, default=0.02)
    parser.add_argument("--soft-tau", type=float, default=0.3)
    parser.add_argument("--pairwise-eps", type=float, default=1.0e-6)
    parser.add_argument("--max-pairs-per-group", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_limit = int(args.train_sample_limit or args.sample_limit or 0)
    val_limit = int(args.val_sample_limit or args.sample_limit or 0)
    train_rows = _read_jsonl(Path(args.train_reward_input), sample_limit=train_limit)
    val_rows = _read_jsonl(Path(args.val_reward_input), sample_limit=val_limit)
    prior_weights = (
        load_learned_marginal_weights(args.prior_weight_file, allow_default=False)
        if str(args.prior_weight_file or "")
        else None
    )

    weights, train_metrics = train_learned_marginal_reward_weights(
        train_rows,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        pairwise_weight=float(args.pairwise_weight),
        listwise_weight=float(args.listwise_weight),
        huber_weight=float(args.huber_weight),
        prior_weight=float(args.prior_weight),
        soft_tau=float(args.soft_tau),
        pairwise_eps=float(args.pairwise_eps),
        max_pairs_per_group=int(args.max_pairs_per_group),
        prior_weights=prior_weights,
    )
    val_metrics = evaluate_learned_marginal_reward_weights(val_rows, weights)
    weights_path = output_dir / "weights.json"
    save_learned_marginal_weights(weights_path, weights)
    fingerprint = learned_marginal_weight_fingerprint(weights)
    train_metrics = dict(train_metrics)
    train_metrics.update(evaluate_learned_marginal_reward_weights(train_rows, weights))
    train_metrics["weight_fingerprint"] = fingerprint
    val_metrics["weight_fingerprint"] = fingerprint

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector_name": "mrec_greedy_transition_v0_2_learned_marginal_reward",
        "selection_policy": "learned_marginal_reward",
        "train_reward_input": str(args.train_reward_input),
        "val_reward_input": str(args.val_reward_input),
        "output_dir": str(output_dir),
        "weights": str(weights_path),
        "weight_fingerprint": fingerprint,
        "prior_weight_file": str(args.prior_weight_file or ""),
        "params": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "pairwise_weight": float(args.pairwise_weight),
            "listwise_weight": float(args.listwise_weight),
            "huber_weight": float(args.huber_weight),
            "prior_weight": float(args.prior_weight),
            "soft_tau": float(args.soft_tau),
            "pairwise_eps": float(args.pairwise_eps),
            "max_pairs_per_group": int(args.max_pairs_per_group),
            "train_sample_limit": train_limit,
            "val_sample_limit": val_limit,
        },
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
    }

    _write_json(output_dir / "train_metrics.json", train_metrics)
    _write_json(output_dir / "val_metrics.json", val_metrics)
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote learned marginal reward weights to {weights_path}")
    print(f"Fingerprint: {fingerprint}")
    print(f"Train pairs: {train_metrics.get('pair_count', 0)}")
    print(f"Val pair accuracy: {val_metrics.get('pair_accuracy', 0.0):.4f}")
    print(f"Val step top1 match: {val_metrics.get('step_top1_match', 0.0):.4f}")
    return 0


def _read_jsonl(path: Path, *, sample_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if sample_limit > 0 and len(rows) >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows read from {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
