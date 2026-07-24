#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors import mrec_learned_marginal as mrec  # noqa: E402


SUPERVISION_MODE = "verifier_utility_only"
ROLLIN_POLICY = "verifier_utility_greedy"
REWARD_SOURCE = "verifier_utility_delta_margin"
ALLOWED_SPLITS = {"train", "val"}
FINGERPRINT_FIELDS = (
    "run_fingerprint",
    "teacher_fingerprint",
    "scoring_fingerprint",
    "candidate_pool_fingerprint",
)


@dataclass(frozen=True)
class UtilityStepGroup:
    event_id: str
    step: int
    candidate_indices: tuple[int, ...]
    selected_idx: int
    features: tuple[tuple[float, ...], ...]
    deltas: tuple[float, ...]


@dataclass(frozen=True)
class ValidatedUtilityData:
    split: str
    positive_features: tuple[tuple[float, ...], ...]
    negative_features: tuple[tuple[float, ...], ...]
    groups: tuple[UtilityStepGroup, ...]
    supervision_fingerprint: str
    audit: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CPU-only MREC weights from strict verifier-utility preferences."
    )
    parser.add_argument(
        "--train-reward-input",
        required=True,
        nargs="+",
        help="One or more sharded reward_records JSONL files for train.",
    )
    parser.add_argument(
        "--val-reward-input",
        required=True,
        nargs="+",
        help="One or more sharded reward_records JSONL files for val.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rollout-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_dir = Path(args.output_dir)
    _require_empty_output_dir(output_dir)
    if int(args.expected_rollout_steps) <= 0:
        raise ValueError("expected_rollout_steps must be positive")

    train_data = load_and_validate_utility_inputs(
        [Path(path) for path in args.train_reward_input],
        expected_split="train",
        expected_rollout_steps=int(args.expected_rollout_steps),
    )
    val_data = load_and_validate_utility_inputs(
        [Path(path) for path in args.val_reward_input],
        expected_split="val",
        expected_rollout_steps=int(args.expected_rollout_steps),
    )
    _validate_cross_split_contract(train_data, val_data)
    if not train_data.positive_features:
        raise ValueError("train utility supervision contains no winner-vs-rest pairs")
    if not val_data.positive_features:
        raise ValueError("val utility supervision contains no winner-vs-rest pairs")

    metadata = {
        "trained_from": SUPERVISION_MODE,
        "supervision_mode": SUPERVISION_MODE,
        "supervision_source": REWARD_SOURCE,
        "rollin_policy": ROLLIN_POLICY,
        "supervision_fingerprint": train_data.supervision_fingerprint,
        "initialized_from": "equal_weight_neutral_v0_1",
        "pairwise_objective": "winner_vs_rest_pairwise_logistic",
        "gold_label_read_count": 0,
        "teacher_structure_read_count": 0,
        "teacher_order_read_count": 0,
        "oracle_read_row_count": 0,
        "oracle_ordered_row_count": 0,
    }
    weights, optimization_metrics = mrec.fit_pairwise_marginal_scorer(
        train_data.positive_features,
        train_data.negative_features,
        initial_weights=mrec.initial_neutral_learned_marginal_weights(),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        metadata=metadata,
    )
    train_metrics = {
        **optimization_metrics,
        **evaluate_utility_pairs(train_data, weights),
        **_zero_forbidden_read_counts(),
    }
    val_metrics = {
        **evaluate_utility_pairs(val_data, weights),
        **_zero_forbidden_read_counts(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.json"
    mrec.save_learned_marginal_weights(weights_path, weights)
    weight_fingerprint = mrec.learned_marginal_weight_fingerprint(weights)
    train_metrics["weight_fingerprint"] = weight_fingerprint
    val_metrics["weight_fingerprint"] = weight_fingerprint
    combined_supervision_fingerprint = _stable_fingerprint(
        {
            "mode": SUPERVISION_MODE,
            "train": train_data.supervision_fingerprint,
            "val": val_data.supervision_fingerprint,
        }
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selector_name": "mrec_greedy_transition_v0_2_learned_marginal_utility_only",
        "selection_policy": "learned_marginal_proxy",
        "runtime_selection_policy": "learned_marginal_proxy",
        "training_supervision": SUPERVISION_MODE,
        "compute_device": "cpu",
        "train_reward_inputs": [str(path) for path in args.train_reward_input],
        "val_reward_inputs": [str(path) for path in args.val_reward_input],
        "output_dir": str(output_dir),
        "weights": str(weights_path),
        "weight_fingerprint": weight_fingerprint,
        "supervision_fingerprint": combined_supervision_fingerprint,
        "params": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "expected_rollout_steps": int(args.expected_rollout_steps),
            "initialization": "equal_weight_neutral_v0_1",
            "objective": "winner_vs_rest_pairwise_logistic",
            "listwise_weight": 0.0,
            "huber_weight": 0.0,
            "prior_weight": 0.0,
        },
        "supervision_contract": {
            "mode": SUPERVISION_MODE,
            "rollin_policy": ROLLIN_POLICY,
            "reward_source": REWARD_SOURCE,
            "fallback_supervision": None,
            **_zero_forbidden_read_counts(),
        },
        "train_input_audit": train_data.audit,
        "val_input_audit": val_data.audit,
    }
    _write_json(output_dir / "train_metrics.json", train_metrics)
    _write_json(output_dir / "val_metrics.json", val_metrics)
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote verifier-utility-only learned marginal weights to {weights_path}")
    print(f"Fingerprint: {weight_fingerprint}")
    print(f"Train pairs: {train_metrics['pair_count']}")
    print(f"Val pair accuracy: {val_metrics['pair_accuracy']:.4f}")
    return 0


def load_and_validate_utility_inputs(
    paths: Sequence[Path], *, expected_split: str, expected_rollout_steps: int = 5
) -> ValidatedUtilityData:
    if expected_split not in ALLOWED_SPLITS:
        raise ValueError(f"utility-only training accepts only train/val, got {expected_split!r}")
    if not paths:
        raise ValueError(f"no {expected_split} reward inputs were provided")
    if int(expected_rollout_steps) <= 0:
        raise ValueError("expected_rollout_steps must be positive")
    normalized_paths = [Path(path) for path in paths]
    if len({str(path.resolve()) for path in normalized_paths}) != len(normalized_paths):
        raise ValueError(f"duplicate {expected_split} reward input path")

    rows: list[dict[str, Any]] = []
    source_by_row: list[str] = []
    run_fingerprints_by_input: dict[str, str] = {}
    for path in normalized_paths:
        file_rows = _read_jsonl(path)
        run_fingerprints = {_required_fingerprint(row, "run_fingerprint", context=str(path)) for row in file_rows}
        if len(run_fingerprints) != 1:
            raise ValueError(f"{path} contains inconsistent run_fingerprint values")
        run_fingerprints_by_input[str(path)] = next(iter(run_fingerprints))
        rows.extend(file_rows)
        source_by_row.extend([str(path)] * len(file_rows))

    teacher_fingerprints = {
        _required_fingerprint(row, "teacher_fingerprint", context=f"{expected_split} row") for row in rows
    }
    scoring_fingerprints = {
        _required_fingerprint(row, "scoring_fingerprint", context=f"{expected_split} row") for row in rows
    }
    if len(teacher_fingerprints) != 1:
        raise ValueError(f"{expected_split} inputs contain inconsistent teacher_fingerprint values")
    if len(scoring_fingerprints) != 1:
        raise ValueError(f"{expected_split} inputs contain inconsistent scoring_fingerprint values")

    event_rows: dict[str, list[tuple[dict[str, Any], str]]] = {}
    event_source: dict[str, str] = {}
    seen_keys: set[tuple[str, int, int]] = set()
    for row_index, (row, source) in enumerate(zip(rows, source_by_row)):
        split = str(row.get("split") or "")
        if split == "test":
            raise ValueError("test reward rows are forbidden for verifier-utility-only training")
        if split != expected_split:
            raise ValueError(
                f"expected split={expected_split!r}, got split={split!r} at row {row_index} of {source}"
            )
        if str(row.get("rollin_policy") or "") != ROLLIN_POLICY:
            raise ValueError(f"invalid rollin_policy at row {row_index} of {source}")
        if str(row.get("reward_source") or "") != REWARD_SOURCE:
            raise ValueError(f"invalid reward_source at row {row_index} of {source}")
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            raise ValueError(f"missing event_id at row {row_index} of {source}")
        if event_id in event_source and event_source[event_id] != source:
            raise ValueError(f"event_id={event_id!r} is split across reward shards")
        event_source[event_id] = source
        step = _strict_nonnegative_int(row.get("step"), field="step", context=f"event_id={event_id}")
        candidate_idx = _strict_nonnegative_int(
            row.get("candidate_idx"), field="candidate_idx", context=f"event_id={event_id}, step={step}"
        )
        key = (event_id, step, candidate_idx)
        if key in seen_keys:
            raise ValueError(f"duplicate reward row for event_id={event_id!r}, step={step}, candidate_idx={candidate_idx}")
        seen_keys.add(key)
        for field in FINGERPRINT_FIELDS:
            _required_fingerprint(row, field, context=f"event_id={event_id}, step={step}, candidate_idx={candidate_idx}")
        _feature_vector(row, context=f"event_id={event_id}, step={step}, candidate_idx={candidate_idx}")
        _finite_number(row.get("delta_margin"), field="delta_margin", context=f"event_id={event_id}, step={step}")
        if type(row.get("utility_selected")) is not bool:
            raise ValueError(f"utility_selected must be boolean for event_id={event_id!r}, step={step}")
        event_rows.setdefault(event_id, []).append((row, source))

    positive_features: list[tuple[float, ...]] = []
    negative_features: list[tuple[float, ...]] = []
    groups: list[UtilityStepGroup] = []
    digest = hashlib.sha256()
    candidate_pool_fingerprints: dict[str, str] = {}
    run_fingerprints_by_event: dict[str, str] = {}
    for event_id in sorted(event_rows):
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row, _source in event_rows[event_id]:
            grouped.setdefault(int(row["step"]), []).append(row)
        steps = sorted(grouped)
        if not steps or steps[0] != 0:
            raise ValueError(f"event_id={event_id!r} must start at step 0; got {steps}")

        event_run = {str(row["run_fingerprint"]) for rows_at_step in grouped.values() for row in rows_at_step}
        event_pool = {
            str(row["candidate_pool_fingerprint"])
            for rows_at_step in grouped.values()
            for row in rows_at_step
        }
        if len(event_run) != 1:
            raise ValueError(f"event_id={event_id!r} has inconsistent run_fingerprint values")
        if len(event_pool) != 1:
            raise ValueError(f"event_id={event_id!r} has inconsistent candidate_pool_fingerprint values")
        run_fingerprints_by_event[event_id] = next(iter(event_run))
        candidate_pool_fingerprints[event_id] = next(iter(event_pool))

        first_indices = sorted(int(row["candidate_idx"]) for row in grouped[0])
        if first_indices != list(range(len(first_indices))):
            raise ValueError(
                f"event_id={event_id!r}, step=0 candidate_idx must be contiguous from zero; got {first_indices}"
            )
        expected_steps = list(range(min(int(expected_rollout_steps), len(first_indices))))
        if steps != expected_steps:
            raise ValueError(
                f"incomplete rollout step coverage for event_id={event_id!r}: "
                f"expected={expected_steps}, actual={steps}"
            )
        expected_remaining = set(first_indices)
        selected_history: list[int] = []
        for step in steps:
            rows_at_step = sorted(grouped[step], key=lambda row: int(row["candidate_idx"]))
            actual_remaining = {int(row["candidate_idx"]) for row in rows_at_step}
            if actual_remaining != expected_remaining:
                raise ValueError(
                    f"incomplete candidate coverage for event_id={event_id!r}, step={step}: "
                    f"expected={sorted(expected_remaining)}, actual={sorted(actual_remaining)}"
                )
            for row in rows_at_step:
                prefix = _strict_prefix(row.get("prefix_indices"), event_id=event_id, step=step)
                if prefix != selected_history:
                    raise ValueError(
                        f"wrong prefix for event_id={event_id!r}, step={step}: "
                        f"expected={selected_history}, actual={prefix}"
                    )

            selected_rows = [row for row in rows_at_step if row["utility_selected"]]
            if len(selected_rows) != 1:
                raise ValueError(
                    f"event_id={event_id!r}, step={step} must have exactly one utility_selected row"
                )
            ordered_rows = sorted(
                rows_at_step,
                key=lambda row: (-float(row["delta_margin"]), int(row["candidate_idx"])),
            )
            winner_idx = int(ordered_rows[0]["candidate_idx"])
            selected_idx = int(selected_rows[0]["candidate_idx"])
            if selected_idx != winner_idx:
                raise ValueError(
                    f"wrong utility_selected for event_id={event_id!r}, step={step}: "
                    f"expected argmax candidate_idx={winner_idx}, actual={selected_idx}"
                )

            candidate_indices = tuple(int(row["candidate_idx"]) for row in rows_at_step)
            features = tuple(_feature_vector(row, context=f"event_id={event_id}, step={step}") for row in rows_at_step)
            deltas = tuple(float(row["delta_margin"]) for row in rows_at_step)
            local_by_candidate = {candidate_idx: idx for idx, candidate_idx in enumerate(candidate_indices)}
            order = [local_by_candidate[int(row["candidate_idx"])] for row in ordered_rows]
            for positive_local, negative_local in mrec.build_winner_vs_rest_preferences(order):
                positive_features.append(features[positive_local])
                negative_features.append(features[negative_local])
            groups.append(
                UtilityStepGroup(
                    event_id=event_id,
                    step=step,
                    candidate_indices=candidate_indices,
                    selected_idx=selected_idx,
                    features=features,
                    deltas=deltas,
                )
            )
            digest.update(
                json.dumps(
                    {
                        "event_id": event_id,
                        "step": step,
                        "candidate_indices": candidate_indices,
                        "selected_idx": selected_idx,
                        "features": features,
                        "deltas": deltas,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            selected_history.append(selected_idx)
            expected_remaining.remove(selected_idx)

    supervision_fingerprint = digest.hexdigest()[:16]
    audit = {
        "split": expected_split,
        "input_paths": [str(path) for path in normalized_paths],
        "input_count": len(normalized_paths),
        "row_count": len(rows),
        "event_count": len(event_rows),
        "step_count": len(groups),
        "pair_count": len(positive_features),
        "expected_rollout_steps": int(expected_rollout_steps),
        "rollin_policy": ROLLIN_POLICY,
        "reward_source": REWARD_SOURCE,
        "run_fingerprints_by_input": run_fingerprints_by_input,
        "run_fingerprints_by_event": run_fingerprints_by_event,
        "teacher_fingerprint": next(iter(teacher_fingerprints)),
        "scoring_fingerprint": next(iter(scoring_fingerprints)),
        "candidate_pool_fingerprints_by_event": candidate_pool_fingerprints,
        "supervision_fingerprint": supervision_fingerprint,
        "coverage_complete_for_recorded_steps": True,
        "rollout_step_coverage_complete": True,
        **_zero_forbidden_read_counts(),
    }
    return ValidatedUtilityData(
        split=expected_split,
        positive_features=tuple(positive_features),
        negative_features=tuple(negative_features),
        groups=tuple(groups),
        supervision_fingerprint=supervision_fingerprint,
        audit=audit,
    )


def evaluate_utility_pairs(
    data: ValidatedUtilityData, weights: mrec.LearnedMarginalWeights
) -> dict[str, Any]:
    pair_correct = 0
    for positive, negative in zip(data.positive_features, data.negative_features):
        positive_score = mrec.score_marginal_features(dict(zip(mrec.FEATURE_NAMES, positive)), weights)
        negative_score = mrec.score_marginal_features(dict(zip(mrec.FEATURE_NAMES, negative)), weights)
        pair_correct += int(positive_score > negative_score)

    top1_correct = 0
    for group in data.groups:
        scores = [
            mrec.score_marginal_features(dict(zip(mrec.FEATURE_NAMES, features)), weights)
            for features in group.features
        ]
        predicted_local = min(
            range(len(scores)), key=lambda idx: (-scores[idx], group.candidate_indices[idx])
        )
        top1_correct += int(group.candidate_indices[predicted_local] == group.selected_idx)
    return {
        "supervision_mode": SUPERVISION_MODE,
        "evaluation_target": "verifier_utility_winner_vs_rest",
        "supervision_fingerprint": data.supervision_fingerprint,
        "row_count": int(data.audit["row_count"]),
        "event_count": int(data.audit["event_count"]),
        "preference_step_count": len(data.groups),
        "pair_count": len(data.positive_features),
        "scored_pair_count": len(data.positive_features),
        "pair_accuracy": float(pair_correct / len(data.positive_features)) if data.positive_features else 0.0,
        "step_top1_match": float(top1_correct / len(data.groups)) if data.groups else 0.0,
        "verifier_utility_read_row_count": int(data.audit["row_count"]),
    }


def _validate_cross_split_contract(
    train_data: ValidatedUtilityData, val_data: ValidatedUtilityData
) -> None:
    if train_data.split != "train" or val_data.split != "val":
        raise ValueError("utility-only inputs must be train and val")
    for field in ("teacher_fingerprint", "scoring_fingerprint"):
        if train_data.audit[field] != val_data.audit[field]:
            raise ValueError(f"train/val {field} mismatch")
    train_events = set(train_data.audit["run_fingerprints_by_event"])
    val_events = set(val_data.audit["run_fingerprints_by_event"])
    overlap = sorted(train_events & val_events)
    if overlap:
        raise ValueError(f"train/val event_id overlap: {overlap[:3]}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"reward row must be an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no reward rows read from {path}")
    return rows


def _required_fingerprint(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"missing {field} for {context}")
    return value


def _strict_nonnegative_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer for {context}")
    return int(value)


def _strict_prefix(value: Any, *, event_id: str, step: int) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"prefix_indices must be a list of non-negative integers for event_id={event_id!r}, step={step}")
    if len(value) != len(set(value)):
        raise ValueError(f"prefix_indices contains duplicates for event_id={event_id!r}, step={step}")
    return [int(item) for item in value]


def _finite_number(value: Any, *, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric for {context}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite for {context}")
    return numeric


def _feature_vector(row: Mapping[str, Any], *, context: str) -> tuple[float, ...]:
    features = row.get("mrec_features")
    if not isinstance(features, Mapping):
        raise ValueError(f"mrec_features must be an object for {context}")
    missing = [name for name in mrec.FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"mrec_features missing {missing} for {context}")
    return tuple(
        _finite_number(features[name], field=f"mrec_features.{name}", context=context)
        for name in mrec.FEATURE_NAMES
    )


def _zero_forbidden_read_counts() -> dict[str, int]:
    return {
        "gold_label_read_count": 0,
        "teacher_structure_read_count": 0,
        "teacher_order_read_count": 0,
        "oracle_read_row_count": 0,
        "oracle_ordered_row_count": 0,
        "structure_preference_step_count": 0,
    }


def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _require_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
