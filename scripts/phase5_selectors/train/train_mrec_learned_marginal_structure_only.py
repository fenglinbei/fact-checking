#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors import mrec_learned_marginal as mrec  # noqa: E402


TOP_LEVEL_SUPERVISION_FIELDS = {
    "gold_id",
    "gold_label",
    "label",
    "target_label",
}
RECURSIVE_SUPERVISION_FIELDS = {
    "gold_margin",
    "reward_delta",
    "teacher_margin",
    "teacher_score",
    "utility_margin",
    "verifier_utility",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CPU-only MREC v0.2 weights from structure-induced preferences."
    )
    parser.add_argument("--train-input", required=True, help="candidate_evidence_map_features_train.jsonl")
    parser.add_argument("--val-input", required=True, help="candidate_evidence_map_features_val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=0, help="Shared train/val limit; 0 uses all rows.")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--val-sample-limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--map-ablation-mode",
        default="full",
        choices=["full", "no_map", "no_directness", "no_confidence", "no_relation"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_dir = Path(args.output_dir)
    _require_empty_output_dir(output_dir)
    train_limit = int(args.train_sample_limit or args.sample_limit or 0)
    val_limit = int(args.val_sample_limit or args.sample_limit or 0)
    train_rows, train_input_audit = _read_jsonl_with_supervision_audit(
        Path(args.train_input), sample_limit=train_limit
    )
    val_rows, val_input_audit = _read_jsonl_with_supervision_audit(Path(args.val_input), sample_limit=val_limit)

    weights, train_metrics = mrec.train_learned_marginal_proxy_weights(
        train_rows,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        candidate_top_n=int(args.candidate_top_n),
        rollout_steps=int(args.rollout_steps),
        map_ablation_mode=str(args.map_ablation_mode),
        supervision_mode=mrec.SUPERVISION_MODE_STRUCTURE_ONLY,
    )
    val_metrics = mrec.evaluate_learned_marginal_proxy_weights(
        val_rows,
        weights,
        candidate_top_n=int(args.candidate_top_n),
        rollout_steps=int(args.rollout_steps),
        map_ablation_mode=str(args.map_ablation_mode),
        supervision_mode=mrec.SUPERVISION_MODE_STRUCTURE_ONLY,
    )
    _assert_structure_only_contract(train_metrics, split="train")
    _assert_structure_only_contract(val_metrics, split="val")

    weights_path = output_dir / "weights.json"
    mrec.save_learned_marginal_weights(weights_path, weights)
    fingerprint = mrec.learned_marginal_weight_fingerprint(weights)
    train_metrics = {
        **dict(train_metrics),
        "supervision_mode": "structure_only",
        "compute_device": "cpu",
        "input_supervision_audit": train_input_audit,
        "weight_fingerprint": fingerprint,
    }
    val_metrics = {
        **val_metrics,
        "supervision_mode": "structure_only",
        "compute_device": "cpu",
        "input_supervision_audit": val_input_audit,
        "weight_fingerprint": fingerprint,
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector_name": "mrec_greedy_transition_v0_2_learned_marginal_structure_only",
        "selection_policy": "learned_marginal_proxy",
        "runtime_selection_policy": "learned_marginal_proxy",
        "training_supervision": "structure_only",
        "compute_device": "cpu",
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
            "train_sample_limit": train_limit,
            "val_sample_limit": val_limit,
        },
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "supervision_contract": {
            "core_supervision_mode": mrec.SUPERVISION_MODE_STRUCTURE_ONLY,
            "source_supervision_fields_preserved_for_poison_audit": True,
            "oracle_read_row_count": 0,
            "gold_label_read_count": 0,
            "teacher_read_count": 0,
            "utility_read_count": 0,
            "reward_read_count": 0,
            "initialization": "equal_weight_neutral_v0_1",
            "fallback_supervision": None,
            "validation_target": "structure_winner_vs_rest",
        },
        "train_input_supervision_audit": train_input_audit,
        "val_input_supervision_audit": val_input_audit,
    }

    _write_json(output_dir / "train_metrics.json", train_metrics)
    _write_json(output_dir / "val_metrics.json", val_metrics)
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote structure-only learned marginal weights to {weights_path}")
    print(f"Fingerprint: {fingerprint}")
    print(f"Train pairs: {train_metrics.get('pair_count', 0)}")
    print(f"Val pair accuracy: {val_metrics.get('pair_accuracy', 0.0):.4f}")
    return 0


def _read_jsonl_with_supervision_audit(
    path: Path, *, sample_limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_fields: Counter[str] = Counter()
    rows_with_supervision_fields = 0
    rows_with_oracle_fields = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if sample_limit > 0 and len(rows) >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_observed: Counter[str] = Counter()
            _audit_supervision_fields(row, observed_fields=row_observed, is_row_root=True)
            if row_observed:
                rows_with_supervision_fields += 1
                observed_fields.update(row_observed)
            if any(name.startswith("oracle_") for name in row_observed):
                rows_with_oracle_fields += 1
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows read from {path}")
    return rows, {
        "row_count": len(rows),
        "source_size_bytes": path.stat().st_size,
        "rows_with_supervision_fields": rows_with_supervision_fields,
        "rows_with_oracle_fields": rows_with_oracle_fields,
        "observed_supervision_field_count": int(sum(observed_fields.values())),
        "observed_supervision_fields": dict(sorted(observed_fields.items())),
        "fields_preserved_for_core_mode_enforcement": True,
    }


def _audit_supervision_fields(
    value: Any,
    *,
    observed_fields: Counter[str],
    is_row_root: bool = False,
) -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            observed = (
                key.startswith("oracle_")
                or key in RECURSIVE_SUPERVISION_FIELDS
                or (is_row_root and key in TOP_LEVEL_SUPERVISION_FIELDS)
            )
            if observed:
                observed_fields[key] += 1
            _audit_supervision_fields(item, observed_fields=observed_fields)
    elif isinstance(value, list):
        for item in value:
            _audit_supervision_fields(item, observed_fields=observed_fields)


def _assert_structure_only_contract(metrics: Mapping[str, Any], *, split: str) -> None:
    if metrics.get("supervision_mode") != mrec.SUPERVISION_MODE_STRUCTURE_ONLY:
        raise RuntimeError(f"{split} did not use structure_only supervision")
    for key in (
        "oracle_read_row_count",
        "gold_label_read_count",
        "teacher_read_count",
        "utility_read_count",
        "reward_read_count",
    ):
        if int(metrics.get(key, -1)) != 0:
            raise RuntimeError(f"{split} structure_only contract violated: {key}={metrics.get(key)!r}")


def _require_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
