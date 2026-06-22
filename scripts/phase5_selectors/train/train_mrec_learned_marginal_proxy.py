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
    extract_marginal_features,
    hard_state_to_soft_state,
    learned_marginal_weight_fingerprint,
    save_learned_marginal_weights,
    score_marginal_features,
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
    )
    val_metrics = _evaluate_rows(
        val_rows,
        weights=weights,
        candidate_top_n=int(args.candidate_top_n),
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
            "train_sample_limit": train_limit,
            "val_sample_limit": val_limit,
        },
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
    }

    _write_json(output_dir / "train_metrics.json", train_metrics)
    _write_json(output_dir / "val_metrics.json", val_metrics)
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote learned marginal weights to {weights_path}")
    print(f"Fingerprint: {fingerprint}")
    print(f"Train pairs: {train_metrics.get('pair_count', 0)}")
    print(f"Val pair accuracy: {val_metrics.get('pair_accuracy', 0.0):.4f}")
    return 0


def _evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    weights: Any,
    candidate_top_n: int,
) -> dict[str, Any]:
    correct = 0
    total = 0
    scored_rows = 0
    for row in rows:
        candidates = _row_candidates(row, candidate_top_n=candidate_top_n)
        if len(candidates) < 2:
            continue
        oracle_keys = _oracle_keys(row, candidates)
        if not oracle_keys:
            continue
        soft_state = hard_state_to_soft_state(_initial_atom_states(row))
        pool_max_token_cost = max([_token_cost(candidate) for candidate in candidates] or [1])
        scores = [
            score_marginal_features(
                extract_marginal_features(
                    candidate,
                    selected_steps=[],
                    soft_state=soft_state,
                    token_budget=None,
                    pool_max_token_cost=pool_max_token_cost,
                ),
                weights,
            )
            for candidate in candidates
        ]
        best_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        best_key = _candidate_key(candidates[best_idx])
        correct += int(best_key == oracle_keys[0])
        total += 1
        scored_rows += 1
    return {
        "row_count": len(rows),
        "scored_row_count": int(scored_rows),
        "pair_accuracy": float(correct / total) if total else 0.0,
    }


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


def _row_candidates(row: Mapping[str, Any], *, candidate_top_n: int) -> list[dict[str, Any]]:
    raw_candidates = row.get("candidate_pool") or row.get("candidates") or row.get("selected_candidates") or []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_candidates):
        if int(candidate_top_n) > 0 and len(out) >= int(candidate_top_n):
            break
        if not isinstance(raw, Mapping):
            continue
        candidate = dict(raw)
        candidate.setdefault("selector_candidate_idx", idx)
        candidate.setdefault("candidate_idx", idx)
        candidate.setdefault("evidence_id", str(candidate.get("candidate_uid") or f"E{idx + 1:02d}"))
        out.append(candidate)
    return out


def _initial_atom_states(row: Mapping[str, Any]) -> dict[str, str]:
    raw_atoms = (row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or []
    states: dict[str, str] = {}
    for idx, atom in enumerate(raw_atoms, start=1):
        if not isinstance(atom, Mapping):
            continue
        atom_id = _compact(atom.get("atom_id") or atom.get("node_id") or f"A{idx}")
        if atom_id:
            states[atom_id] = "U"
    if not states:
        states["A1"] = "U"
    return states


def _oracle_keys(row: Mapping[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    keys = [str(key) for key in row.get("oracle_ordered_keys") or [] if str(key)]
    if keys:
        return keys
    out: list[str] = []
    for idx in row.get("oracle_ordered_indices") or []:
        try:
            out.append(_candidate_key(candidates[int(idx)]))
        except (IndexError, TypeError, ValueError):
            continue
    return [key for key in out if key]


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_key", "candidate_uid", "evidence_id"):
        value = _compact(candidate.get(key) or "")
        if value:
            return value
    return ""


def _token_cost(candidate: Mapping[str, Any]) -> int:
    for key in ("mrec_token_cost", "token_cost", "prompt_token_count", "evidence_token_count"):
        if candidate.get(key) is not None:
            return max(0, _int_or_default(candidate.get(key), 0))
    text = str(candidate.get("text") or candidate.get("evidence_text") or "")
    return max(1, len(text.split())) if text.strip() else 0


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


if __name__ == "__main__":
    raise SystemExit(main())
