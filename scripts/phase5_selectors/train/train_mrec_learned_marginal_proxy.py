#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.mrec_learned_marginal import (  # noqa: E402
    SUPPORTED_PROXY_SUPERVISION_MODES,
    SUPERVISION_MODE_LEGACY_HYBRID,
    evaluate_learned_marginal_proxy_weights,
    learned_marginal_weight_fingerprint,
    save_learned_marginal_weights,
    train_learned_marginal_proxy_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MREC v0.2 learned marginal proxy weights.")
    parser.add_argument("--train-input", required=True, help="candidate_evidence_map_features_train.jsonl")
    parser.add_argument("--val-input", required=True, help="candidate_evidence_map_features_val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=0, help="Shared limit for train and val; 0 disables.")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--val-sample-limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--supervision-mode",
        default=SUPERVISION_MODE_LEGACY_HYBRID,
        choices=sorted(SUPPORTED_PROXY_SUPERVISION_MODES),
        help="legacy_hybrid reproduces oracle-then-structure preferences; structure_only reads structural fields only.",
    )
    parser.add_argument(
        "--map-ablation-mode",
        default="full",
        choices=["full", "no_map", "no_directness", "no_confidence", "no_relation"],
        help="Evidence-map ablation mode for learned-marginal selector features.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_limit = int(args.train_sample_limit or args.sample_limit or 0)
    val_limit = int(args.val_sample_limit or args.sample_limit or 0)
    train_rows = _read_jsonl(Path(args.train_input), sample_limit=train_limit)
    val_rows = _read_jsonl(Path(args.val_input), sample_limit=val_limit)

    weights, train_metrics = train_learned_marginal_proxy_weights(
        train_rows,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        candidate_top_n=int(args.candidate_top_n),
        rollout_steps=int(args.rollout_steps),
        map_ablation_mode=str(args.map_ablation_mode),
        supervision_mode=str(args.supervision_mode),
    )
    val_metrics = evaluate_learned_marginal_proxy_weights(
        val_rows,
        weights,
        candidate_top_n=int(args.candidate_top_n),
        rollout_steps=int(args.rollout_steps),
        map_ablation_mode=str(args.map_ablation_mode),
        supervision_mode=str(args.supervision_mode),
    )
    weights_path = output_dir / "weights.json"
    save_learned_marginal_weights(weights_path, weights)
    fingerprint = learned_marginal_weight_fingerprint(weights)
    train_metrics = dict(train_metrics)
    train_metrics["weight_fingerprint"] = fingerprint
    val_metrics["weight_fingerprint"] = fingerprint
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector_name": "mrec_greedy_transition_v0_2_learned_marginal_proxy",
        "selection_policy": "learned_marginal_proxy",
        "train_input": str(args.train_input),
        "val_input": str(args.val_input),
        "output_dir": str(output_dir),
        "weights": str(weights_path),
        "weight_fingerprint": fingerprint,
        "params": {
            "candidate_top_n": int(args.candidate_top_n),
            "rollout_steps": int(args.rollout_steps),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "map_ablation_mode": str(args.map_ablation_mode),
            "supervision_mode": str(args.supervision_mode),
            "train_sample_limit": train_limit,
            "val_sample_limit": val_limit,
        },
        "supervision": {
            "mode": str(args.supervision_mode),
            "train_fingerprint": train_metrics.get("supervision_fingerprint"),
            "val_fingerprint": val_metrics.get("supervision_fingerprint"),
            "train_oracle_read_row_count": int(train_metrics.get("oracle_read_row_count", 0)),
            "train_gold_label_read_count": int(train_metrics.get("gold_label_read_count", 0)),
            "train_teacher_read_count": int(train_metrics.get("teacher_read_count", 0)),
            "train_utility_read_count": int(train_metrics.get("utility_read_count", 0)),
            "train_reward_read_count": int(train_metrics.get("reward_read_count", 0)),
        },
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
    }

    _write_json(output_dir / "train_metrics.json", train_metrics)
    _write_json(output_dir / "val_metrics.json", val_metrics)
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote learned marginal weights to {weights_path}")
    print(f"Fingerprint: {fingerprint}")
    print(f"Supervision mode: {args.supervision_mode}")
    print(f"Supervision fingerprint: {train_metrics.get('supervision_fingerprint', '')}")
    print(f"Train pairs: {train_metrics.get('pair_count', 0)}")
    print(f"Val pair accuracy: {val_metrics.get('pair_accuracy', 0.0):.4f}")
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
