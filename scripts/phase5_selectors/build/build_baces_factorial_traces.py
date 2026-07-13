#!/usr/bin/env python3
"""Build the frozen 6 selector x 3 controller BACES factorial traces.

The learned artifact supplies only its full-pool UID order and per-candidate
token costs.  Coverage is always recompiled from the feature artifact, and no
stored MREC stopping state is consulted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.baces_exact import solve_exact  # noqa: E402
from fact_checking.selectors.baces_objective import (  # noqa: E402
    BacesEvaluation,
    BacesProblem,
    compile_feature_problem,
    evaluate_display,
    padded_auc,
    utility,
)


SCHEMA_VERSION = "baces_factorial_trace_v0_1"
FACTORIAL_VERSION = "baces_selector_controller_factorial_v0_1"
SELECTOR_LEVELS = (
    "retrieval_source",
    "map_quality_static",
    "ordinal_coverage_greedy",
    "state_free_structural",
    "baces_exact",
    "learned_marginal",
)
CONTROLLER_LEVELS = (
    "fixed5",
    "ordinal_replay_minmax5_10",
    "matched_token_cap",
)
K_MIN = 5
K_MAX = 10
SELECTOR_CONTRACTS = {
    "retrieval_source": (
        "map_selector_s0_retrieval_top5 full-order key: retrieval_score desc, "
        "evidence_map_base_score desc, union_pool_rank asc, stable key asc"
    ),
    "map_quality_static": (
        "map_selector_s2_map_quality_top5 full-order key: map quality desc, "
        "retrieval desc, map base desc, union rank asc, stable key asc"
    ),
    "ordinal_coverage_greedy": (
        "prefix-conditioned ordinal marginal desc, cost asc, retrieval desc, UID asc"
    ),
    "state_free_structural": (
        "standalone ordinal utility desc, cost asc, retrieval desc, UID asc"
    ),
    "baces_exact": (
        "exact ordered-state DP core plus canonical BACES v0.3 zero-gain fill"
    ),
    "learned_marginal": (
        "historical learned marginal admissible order; unranked tail is not synthesized"
    ),
}
CONTROLLER_CONTRACTS = {
    "fixed5": "first min(5, available candidates)",
    "ordinal_replay_minmax5_10": (
        "first prefix t>=5 reaching the common exact Kmax=10 ordinal target, else 10"
    ),
    "matched_token_cap": (
        "hard ordered-prefix cap matched per event to reference build selected token cost"
    ),
}


class FactorialAlignmentError(ValueError):
    """Raised when the three source artifacts cannot be joined by stable UID."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--learned-trace", required=True)
    parser.add_argument("--reference-build", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args(argv)
    if int(args.sample_limit) < 0:
        parser.error("--sample-limit must be non-negative")
    return args


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    feature_rows = _read_jsonl(
        Path(args.features),
        limit=int(args.sample_limit) if int(args.sample_limit) > 0 else None,
    )
    if not feature_rows:
        raise SystemExit(f"No feature rows read from {args.features}")
    feature_event_ids = {_event_id(row, "feature row") for row in feature_rows}
    if len(feature_event_ids) != len(feature_rows):
        raise FactorialAlignmentError("duplicate event_id in feature rows")
    learned_by_event = _read_jsonl_index(
        Path(args.learned_trace),
        artifact="learned trace",
        keep_events=feature_event_ids,
    )
    build_by_event = _read_jsonl_index(
        Path(args.reference_build),
        artifact="reference build",
        keep_events=feature_event_ids,
    )

    summary_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        (selector, controller): []
        for selector in SELECTOR_LEVELS
        for controller in CONTROLLER_LEVELS
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_name = f"selection_trace_{args.split}.jsonl"
    trace_paths: dict[tuple[str, str], Path] = {}
    temp_trace_paths: dict[tuple[str, str], Path] = {}
    for selector in SELECTOR_LEVELS:
        for controller in CONTROLLER_LEVELS:
            cell_dir = output_dir / _cell_id(selector, controller)
            cell_dir.mkdir(parents=True, exist_ok=True)
            trace_paths[(selector, controller)] = cell_dir / trace_name
            temp_trace_paths[(selector, controller)] = (
                cell_dir / f".{trace_name}.tmp.{os.getpid()}"
            )

    traces_promoted = False
    try:
        with ExitStack() as stack:
            handles = {
                factors: stack.enter_context(path.open("w", encoding="utf-8"))
                for factors, path in temp_trace_paths.items()
            }
            for feature_row in feature_rows:
                event_id = _event_id(feature_row, "feature row")
                try:
                    learned_row = learned_by_event[event_id]
                    reference_row = build_by_event[event_id]
                except KeyError as exc:
                    artifact = "learned trace" if event_id not in learned_by_event else "reference build"
                    raise FactorialAlignmentError(
                        f"{artifact} is missing event_id {event_id!r}"
                    ) from exc
                event_cells = build_event_factorial_rows(
                    feature_row=feature_row,
                    learned_row=learned_row,
                    reference_row=reference_row,
                )
                for factors, row in event_cells.items():
                    handles[factors].write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    summary_rows[factors].append(
                        {
                            "selected_count": row["selected_count"],
                            "selected_token_cost": row["selected_token_cost"],
                            "selector_order_is_complete": row["selector_order_is_complete"],
                            "selector_unranked_count": len(row["selector_unranked_indices"]),
                            "baces_display": row["baces_display"],
                            "factorial_metadata": row["factorial_metadata"],
                        }
                    )
        for factors, temp_path in temp_trace_paths.items():
            temp_path.replace(trace_paths[factors])
        traces_promoted = True
    finally:
        if not traces_promoted:
            for temp_path in temp_trace_paths.values():
                temp_path.unlink(missing_ok=True)

    manifest_cells: list[dict[str, Any]] = []
    for selector in SELECTOR_LEVELS:
        for controller in CONTROLLER_LEVELS:
            cell_id = _cell_id(selector, controller)
            rows = summary_rows[(selector, controller)]
            cell_dir = output_dir / cell_id
            summary = summarize_cell(rows, selector=selector, controller=controller)
            _write_json(cell_dir / "summary.json", summary)
            manifest_cells.append(
                {
                    "cell_id": cell_id,
                    "selector_level": selector,
                    "controller_level": controller,
                    "relative_dir": cell_id,
                    "trace_file": f"{cell_id}/{trace_name}",
                    "summary_file": f"{cell_id}/summary.json",
                    "row_count": len(rows),
                    "ready": len(rows) == len(feature_rows),
                }
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "factorial_version": FACTORIAL_VERSION,
        "split": str(args.split),
        "sample_limit": int(args.sample_limit),
        "selector_levels": list(SELECTOR_LEVELS),
        "controller_levels": list(CONTROLLER_LEVELS),
        "selector_contracts": dict(SELECTOR_CONTRACTS),
        "controller_contracts": dict(CONTROLLER_CONTRACTS),
        "cell_count": len(manifest_cells),
        "event_count": len(feature_rows),
        "all_ready": all(cell["ready"] for cell in manifest_cells),
        "cells": manifest_cells,
        "inputs": {
            "features": str(args.features),
            "learned_trace": str(args.learned_trace),
            "reference_build": str(args.reference_build),
        },
        "verifier_build_contract": {
            "script": "scripts/phase5_selectors/build/build_trace_verifier_data.py",
            "selection_mode": "trace",
            "trace_prompt_style": "mrec_min",
            "prompt_evidence_policy": "selected_set",
            "prompt_evidence_min_count": 0,
            "prompt_evidence_max_count": K_MAX,
            "expected_chunk_mmr_fingerprint": "",
            "note": (
                "Each cell already contains its controller-final slate; the verifier "
                "builder must not run a second stopping rule."
            ),
        },
        "source_contract": {
            "coverage_and_pool": "features",
            "learned_order": "learned_trace.selector_ordered_indices_by_candidate_uid",
            "learned_partial_order_policy": (
                "historical_admissible_order_no_tail_synthesis_v1; controllers stop "
                "with selector_order_exhausted when the historical selector left "
                "candidates unranked"
            ),
            "cost": "learned_trace.candidate_pool.mrec_token_cost_by_candidate_uid",
            "matched_token_cap": "reference_build.prompt_evidence_selected_token_cost_by_event",
            "atom_weights": "unit",
            "candidate_identity": "candidate_uid",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote {len(manifest_cells)} ready factorial cells to {output_dir}")
    return 0


def build_event_factorial_rows(
    *,
    feature_row: Mapping[str, Any],
    learned_row: Mapping[str, Any],
    reference_row: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build all 18 rows for one event without file I/O."""

    event_id = _event_id(feature_row, "feature row")
    if _event_id(learned_row, "learned trace") != event_id:
        raise FactorialAlignmentError("learned trace event_id differs from feature event_id")
    if _event_id(reference_row, "reference build") != event_id:
        raise FactorialAlignmentError("reference build event_id differs from feature event_id")

    learned_pool = _candidate_rows(learned_row.get("candidate_pool"), "learned candidate_pool")
    learned_uids = [_candidate_uid(row, "learned candidate_pool") for row in learned_pool]
    _require_unique(learned_uids, "learned candidate_pool")
    cost_overrides = {
        uid: _candidate_cost(candidate, uid=uid)
        for uid, candidate in zip(learned_uids, learned_pool)
    }

    raw_feature_candidates = _candidate_rows(feature_row.get("candidates"), "feature candidates")
    feature_uids = [_candidate_uid(row, "feature candidates") for row in raw_feature_candidates]
    _require_unique(feature_uids, "feature candidates")
    if set(feature_uids) != set(learned_uids):
        raise FactorialAlignmentError(
            "learned fullpool UID set differs from feature candidate UID set"
        )
    learned_order_uids = _learned_order_uids(learned_row, learned_pool)
    if not set(learned_order_uids).issubset(feature_uids):
        raise FactorialAlignmentError(
            "learned selector order contains UIDs outside the shared candidate pool"
        )

    matched_cap = _nonnegative_int(
        reference_row.get("prompt_evidence_selected_token_cost"),
        f"reference token cap for {event_id}",
    )
    feature_for_compile = dict(feature_row)
    canonical_pool: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_feature_candidates):
        candidate = dict(raw)
        uid = feature_uids[idx]
        candidate["candidate_uid"] = uid
        candidate["candidate_idx"] = idx
        candidate["selector_candidate_idx"] = idx
        candidate["selector_pool_rank"] = idx
        candidate["retrieval_score"] = _retrieval_score(candidate)
        candidate["mrec_token_cost"] = int(cost_overrides[uid])
        candidate["token_cost"] = int(cost_overrides[uid])
        canonical_pool.append(candidate)
    feature_for_compile["candidates"] = canonical_pool

    atom_count = _atom_count(feature_for_compile)
    unit_weights = [1] * atom_count
    base_problem = compile_feature_problem(
        feature_for_compile,
        k_max=max(K_MAX, len(canonical_pool)),
        weights=unit_weights,
        cost_overrides=cost_overrides,
    )
    by_uid = {candidate.key: candidate for candidate in base_problem.candidates}
    if set(by_uid) != set(feature_uids):
        raise FactorialAlignmentError("compiled BACES stable keys are not candidate_uid")
    for idx, candidate in enumerate(canonical_pool):
        candidate["baces_q"] = list(by_uid[feature_uids[idx]].q)

    uid_to_idx = {uid: idx for idx, uid in enumerate(feature_uids)}
    learned_indices = [uid_to_idx[uid] for uid in learned_order_uids]
    target_problem = _problem_with_budget(base_problem, k_max=min(K_MAX, len(canonical_pool)))
    target_eval = solve_exact(target_problem)
    target_state = tuple(target_eval.state)
    pool_fingerprint = _pool_fingerprint(base_problem)

    common_orders = {
        selector: _selector_order_indices(
            selector,
            problem=base_problem,
            pool=canonical_pool,
            learned_indices=learned_indices,
        )
        for selector in SELECTOR_LEVELS
        if selector != "baces_exact"
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for selector in SELECTOR_LEVELS:
        for controller in CONTROLLER_LEVELS:
            if selector == "baces_exact":
                full_order, exact_core = _exact_order_for_controller(
                    problem=base_problem,
                    pool=canonical_pool,
                    controller=controller,
                    matched_cap=matched_cap,
                    target_eval=target_eval,
                )
            else:
                full_order = common_orders[selector]
                exact_core = None
            selected, stop_reason = _apply_controller(
                full_order,
                problem=base_problem,
                controller=controller,
                target_state=target_state,
                matched_cap=matched_cap,
            )
            display_problem = _problem_with_budget(
                base_problem, k_max=max(len(selected), 0), token_budget=None
            )
            display_eval = evaluate_display(
                display_problem,
                [feature_uids[idx] for idx in selected],
            )
            out[(selector, controller)] = _trace_row(
                feature_row=feature_row,
                pool=canonical_pool,
                problem=base_problem,
                selector=selector,
                controller=controller,
                full_order=full_order,
                selected=selected,
                stop_reason=stop_reason,
                matched_cap=matched_cap,
                target_state=target_state,
                target_eval=target_eval,
                display_eval=display_eval,
                exact_core=exact_core,
                pool_fingerprint=pool_fingerprint,
                learned_order_count=len(learned_indices),
            )
    return out


def _selector_order_indices(
    selector: str,
    *,
    problem: BacesProblem,
    pool: Sequence[Mapping[str, Any]],
    learned_indices: Sequence[int],
) -> list[int]:
    indices = list(range(len(pool)))
    by_key = {candidate.key: candidate for candidate in problem.candidates}
    if selector == "retrieval_source":
        return sorted(indices, key=lambda idx: _retrieval_sort_key(pool[idx]))
    if selector == "map_quality_static":
        return sorted(indices, key=lambda idx: _map_quality_sort_key(pool[idx]))
    if selector == "state_free_structural":
        return sorted(
            indices,
            key=lambda idx: (
                -utility(by_key[_candidate_uid(pool[idx], "pool")].q, problem.weights),
                *_cost_retrieval_key(pool[idx]),
            ),
        )
    if selector == "ordinal_coverage_greedy":
        state = (0,) * len(problem.weights)
        remaining = set(indices)
        ordered: list[int] = []
        while remaining:
            ranked: list[tuple[tuple[Any, ...], int, tuple[int, ...]]] = []
            before = utility(state, problem.weights)
            for idx in remaining:
                q = by_key[_candidate_uid(pool[idx], "pool")].q
                after_state = tuple(max(left, right) for left, right in zip(state, q))
                delta = utility(after_state, problem.weights) - before
                ranked.append(((-delta, *_cost_retrieval_key(pool[idx])), idx, after_state))
            _rank, chosen, state = min(ranked, key=lambda item: item[0])
            ordered.append(chosen)
            remaining.remove(chosen)
        return ordered
    if selector == "learned_marginal":
        return list(learned_indices)
    raise ValueError(f"unsupported selector level: {selector}")


def _exact_order_for_controller(
    *,
    problem: BacesProblem,
    pool: Sequence[Mapping[str, Any]],
    controller: str,
    matched_cap: int,
    target_eval: BacesEvaluation,
) -> tuple[list[int], BacesEvaluation]:
    if controller == "fixed5":
        exact_problem = _problem_with_budget(problem, k_max=min(K_MIN, len(pool)))
        exact_eval = solve_exact(exact_problem)
    elif controller == "ordinal_replay_minmax5_10":
        exact_eval = target_eval
    elif controller == "matched_token_cap":
        exact_problem = _problem_with_budget(
            problem,
            k_max=min(K_MAX, len(pool)),
            token_budget=matched_cap,
        )
        exact_eval = solve_exact(exact_problem)
    else:
        raise ValueError(f"unsupported controller level: {controller}")
    uid_to_idx = {
        _candidate_uid(candidate, "pool"): idx for idx, candidate in enumerate(pool)
    }
    core = [uid_to_idx[key] for key in exact_eval.keys]
    core_set = set(core)
    fill = sorted(
        (idx for idx in range(len(pool)) if idx not in core_set),
        key=lambda idx: _baces_zero_gain_fill_key(
            problem.candidates[idx], pool[idx], weights=problem.weights
        ),
    )
    return core + fill, exact_eval


def _apply_controller(
    full_order: Sequence[int],
    *,
    problem: BacesProblem,
    controller: str,
    target_state: tuple[int, ...],
    matched_cap: int,
) -> tuple[list[int], str]:
    available_count = len(full_order)
    pool_count = len(problem.candidates)

    def exhausted_reason(*, selected_count: int) -> str:
        controller_horizon = min(K_MAX, pool_count)
        if available_count < controller_horizon and selected_count >= available_count:
            return "selector_order_exhausted"
        return "pool_exhausted"

    if controller == "fixed5":
        selected = list(full_order[: min(K_MIN, len(full_order))])
        return (
            selected,
            "fixed5" if len(selected) == K_MIN else exhausted_reason(selected_count=len(selected)),
        )
    if controller == "matched_token_cap":
        by_key = {candidate.key: candidate for candidate in problem.candidates}
        selected: list[int] = []
        total = 0
        for idx in full_order[:K_MAX]:
            uid = problem.candidates[idx].key
            next_total = total + int(by_key[uid].cost)
            if next_total > matched_cap:
                return selected, "matched_token_cap"
            selected.append(idx)
            total = next_total
        return (
            selected,
            "max10"
            if len(selected) == K_MAX
            else exhausted_reason(selected_count=len(selected)),
        )
    if controller == "ordinal_replay_minmax5_10":
        state = (0,) * len(problem.weights)
        by_key = {candidate.key: candidate for candidate in problem.candidates}
        selected: list[int] = []
        for idx in full_order[:K_MAX]:
            candidate = by_key[problem.candidates[idx].key]
            state = tuple(max(left, right) for left, right in zip(state, candidate.q))
            selected.append(idx)
            if len(selected) >= K_MIN and state == target_state:
                return selected, "ordinal_target_reached"
        return (
            selected,
            "max10"
            if len(selected) == K_MAX
            else exhausted_reason(selected_count=len(selected)),
        )
    raise ValueError(f"unsupported controller level: {controller}")


def _trace_row(
    *,
    feature_row: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    problem: BacesProblem,
    selector: str,
    controller: str,
    full_order: Sequence[int],
    selected: Sequence[int],
    stop_reason: str,
    matched_cap: int,
    target_state: tuple[int, ...],
    target_eval: BacesEvaluation,
    display_eval: BacesEvaluation,
    exact_core: BacesEvaluation | None,
    pool_fingerprint: str,
    learned_order_count: int,
) -> dict[str, Any]:
    selected_candidates = [dict(pool[idx]) for idx in selected]
    selected_uids = [_candidate_uid(candidate, "selected candidate") for candidate in selected_candidates]
    metadata = {
        "factorial_version": FACTORIAL_VERSION,
        "selector_level": selector,
        "controller_level": controller,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "matched_token_cap": matched_cap if controller == "matched_token_cap" else None,
        "controller_stop_reason": stop_reason,
        "common_exact_kmax10_target_state": list(target_state),
        "common_exact_kmax10_target_utility": int(target_eval.utility),
        "weight_policy": "unit",
        "candidate_identity": "candidate_uid",
        "stored_target_resolved_used": False,
        "mrec_steps_semantics": "ordinal_cue_adapter_only_no_stance_state",
        "selector_contract": SELECTOR_CONTRACTS[selector],
        "controller_contract": CONTROLLER_CONTRACTS[controller],
    }
    steps = _display_steps(
        display_eval,
        selected,
        pool,
        problem,
        target_state=target_state,
    )
    mrec_steps = _mrec_cue_steps(
        display_steps=steps,
        pool=pool,
        problem=problem,
        claim_atoms=_claim_atoms(feature_row),
        target_state=target_state,
    )
    candidate_scores = []
    selected_rank = {idx: rank for rank, idx in enumerate(selected)}
    full_rank = {idx: rank for rank, idx in enumerate(full_order)}
    order_is_complete = len(full_order) == len(pool)
    unranked_indices = [idx for idx in range(len(pool)) if idx not in full_rank]
    metadata.update(
        {
            "selector_order_is_complete": order_is_complete,
            "selector_available_order_count": len(full_order),
            "selector_unranked_count": len(unranked_indices),
            "learned_source_order_count": learned_order_count,
            "learned_partial_order_policy": (
                "historical_admissible_order_no_tail_synthesis_v1"
            ),
        }
    )
    for idx, candidate in enumerate(pool):
        candidate_scores.append(
            {
                "candidate_idx": idx,
                "candidate_uid": _candidate_uid(candidate, "pool"),
                "selector_name": selector,
                "selector_full_rank": full_rank.get(idx),
                "selector_ranked": idx in full_rank,
                "selector_selected_step": selected_rank.get(idx, -1),
                "selector_score": float(len(selected) - selected_rank[idx]) if idx in selected_rank else 0.0,
                "retrieval_score": _safe_float(candidate.get("retrieval_score")),
                "evidence_map_quality_score": _safe_float(candidate.get("evidence_map_quality_score")),
                "evidence_map_base_score": _safe_float(candidate.get("evidence_map_base_score")),
            }
        )
    cell_id = _cell_id(selector, controller)
    row = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": SCHEMA_VERSION,
        "mrec_trace_version": SCHEMA_VERSION,
        "event_id": _event_id(feature_row, "feature row"),
        "claim": str(feature_row.get("claim") or ""),
        "gold_label": str(feature_row.get("gold_label") or feature_row.get("label") or ""),
        "selector_name": f"baces_factorial_{cell_id}",
        "mrec_selector_name": f"baces_factorial_{cell_id}",
        "selection_policy": selector,
        "adaptive_policy": controller,
        "factorial_cell_id": cell_id,
        "factor_selector": selector,
        "factor_controller": controller,
        "factorial_metadata": metadata,
        "factor_metadata": metadata,
        "candidate_pool": [dict(candidate) for candidate in pool],
        "candidate_pool_fingerprint": pool_fingerprint,
        "candidate_pool_metadata": {
            "schema_version": SCHEMA_VERSION,
            "candidate_identity": "candidate_uid",
            "candidate_pool_fingerprint": pool_fingerprint,
            "factorial_cell_id": cell_id,
        },
        "candidate_scores": candidate_scores,
        "selector_order_is_complete": order_is_complete,
        "selector_available_ordered_indices": list(full_order),
        "selector_available_ordered_candidate_uids": [
            _candidate_uid(pool[idx], "pool") for idx in full_order
        ],
        "selector_unranked_indices": unranked_indices,
        "selector_unranked_candidate_uids": [
            _candidate_uid(pool[idx], "pool") for idx in unranked_indices
        ],
        "selector_full_ordered_indices": list(full_order) if order_is_complete else None,
        "selector_full_ordered_candidate_uids": (
            [_candidate_uid(pool[idx], "pool") for idx in full_order]
            if order_is_complete
            else None
        ),
        "ordered_indices": list(selected),
        "ordered_candidate_uids": selected_uids,
        "ordered_candidates": selected_candidates,
        "selector_ordered_indices": list(selected),
        "display_ordered_indices": list(selected),
        "selected_indices": list(selected),
        "selected_candidates": selected_candidates,
        "selected_candidate_uids": selected_uids,
        "selected_keys": selected_uids,
        "selected_candidate_keys": [str(candidate.get("candidate_key") or "") for candidate in selected_candidates],
        "selected_evidence_ids": [str(candidate.get("evidence_id") or "") for candidate in selected_candidates],
        "selected_count": len(selected),
        "selected_token_cost": int(display_eval.token_cost),
        "claim_atoms": _claim_atoms(feature_row),
        "baces_display_steps": steps,
        "mrec_steps": mrec_steps,
        "baces_display": {
            "terminal_state": list(display_eval.state),
            "terminal_utility": int(display_eval.utility),
            "acquisition_time": int(display_eval.acquisition_time),
            "token_cost": int(display_eval.token_cost),
            "length": int(display_eval.length),
            "padded_auc_horizon10": int(padded_auc(display_eval, K_MAX)),
        },
        "params": {
            "min_steps": K_MIN,
            "max_steps": K_MAX,
            "token_budget": matched_cap if controller == "matched_token_cap" else None,
            "factor_selector": selector,
            "factor_controller": controller,
            "learned_source_order_count": learned_order_count,
        },
    }
    if "evidence_map" in feature_row:
        row["evidence_map"] = feature_row["evidence_map"]
    if exact_core is not None:
        row["baces_exact_core"] = {
            "keys": list(exact_core.keys),
            "terminal_state": list(exact_core.state),
            "terminal_utility": int(exact_core.utility),
            "acquisition_time": int(exact_core.acquisition_time),
            "length": int(exact_core.length),
            "token_cost": int(exact_core.token_cost),
            "objective": _json_objective(exact_core.objective),
        }
    return row


def _display_steps(
    evaluation: BacesEvaluation,
    selected: Sequence[int],
    pool: Sequence[Mapping[str, Any]],
    problem: BacesProblem,
    target_state: tuple[int, ...],
) -> list[dict[str, Any]]:
    atom_ids = list(problem.atom_ids)
    out: list[dict[str, Any]] = []
    for step, idx in zip(evaluation.steps, selected):
        candidate = pool[idx]
        out.append(
            {
                "step": int(step.position),
                "position": int(step.position),
                "candidate_idx": int(idx),
                "candidate_uid": str(step.key),
                "candidate_key": str(candidate.get("candidate_key") or ""),
                "evidence_id": str(candidate.get("evidence_id") or ""),
                "operation": "COVER" if step.delta > 0 else "ZERO_GAIN_FILL",
                "state_before": list(step.state_before),
                "state_after": list(step.state_after),
                "state_before_by_atom": dict(zip(atom_ids, step.state_before)),
                "state_after_by_atom": dict(zip(atom_ids, step.state_after)),
                "ordinal_marginal": int(step.delta),
                "delta": int(step.delta),
                "cumulative_utility": int(step.cumulative_utility),
                "token_cost": int(step.candidate_cost),
                "cumulative_token_cost": int(step.cumulative_cost),
                "acquisition_time_so_far": int(step.acquisition_time_so_far),
                "ordinal_target_resolved": tuple(step.state_after) == target_state,
            }
        )
    return out


def _mrec_cue_steps(
    *,
    display_steps: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    problem: BacesProblem,
    claim_atoms: Sequence[Mapping[str, Any]],
    target_state: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Project ordinal replay to the cue-only subset consumed by ``mrec_min``.

    Deliberately absent are ``operation`` and stance-valued state fields.  This
    compatibility view is prompt routing metadata, not an MREC state trace.
    """

    atoms = list(claim_atoms)
    atom_ids = list(problem.atom_ids)
    atom_by_id = {
        str(atom.get("atom_id") or atom.get("node_id") or atom_ids[idx]): atom
        for idx, atom in enumerate(atoms)
        if idx < len(atom_ids)
    }
    by_key = {candidate.key: candidate for candidate in problem.candidates}
    out: list[dict[str, Any]] = []
    for display_step in display_steps:
        idx = int(display_step["candidate_idx"])
        uid = str(display_step["candidate_uid"])
        candidate = pool[idx]
        q = by_key[uid].q
        before = tuple(int(value) for value in display_step["state_before"])
        after = tuple(int(value) for value in display_step["state_after"])
        upgraded = [pos for pos, (left, right) in enumerate(zip(before, after)) if right > left]
        positive = [pos for pos, level in enumerate(q) if level > 0]
        cue_pos = (upgraded or positive or [0])[0]
        atom_id = atom_ids[cue_pos]
        atom = atom_by_id.get(atom_id, atoms[cue_pos] if cue_pos < len(atoms) else {})
        cue_text = _compact(
            atom.get("text")
            or atom.get("proposition")
            or atom.get("query_rendering")
            or atom_id
        )
        covered_atom_ids = [atom_ids[pos] for pos, level in enumerate(q) if level > 0]
        reached = sum(
            int(current >= target)
            for current, target in zip(after, target_state)
        )
        out.append(
            {
                "step": int(display_step["step"]),
                "candidate_idx": idx,
                "selector_candidate_idx": idx,
                "candidate_uid": uid,
                "evidence_id": str(candidate.get("evidence_id") or ""),
                "atom_id": atom_id,
                "cue_text": cue_text,
                "cue_source": "claim_atom",
                "covered_atom_ids": covered_atom_ids,
                "token_cost": int(display_step["token_cost"]),
                "target_resolved": after == target_state,
                "resolved_atom_rate": float(reached / len(target_state)),
            }
        )
    return out


def summarize_cell(
    rows: Sequence[Mapping[str, Any]], *, selector: str, controller: str
) -> dict[str, Any]:
    selected_counts = [int(row.get("selected_count") or 0) for row in rows]
    token_costs = [int(row.get("selected_token_cost") or 0) for row in rows]
    utilities = [int((row.get("baces_display") or {}).get("terminal_utility") or 0) for row in rows]
    stop_reasons = Counter(
        str((row.get("factorial_metadata") or {}).get("controller_stop_reason") or "")
        for row in rows
    )
    incomplete_order_count = sum(
        1 for row in rows if not bool(row.get("selector_order_is_complete", True))
    )
    selector_unranked_counts = [
        int(row.get("selector_unranked_count") or 0) for row in rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "factorial_version": FACTORIAL_VERSION,
        "cell_id": _cell_id(selector, controller),
        "selector_level": selector,
        "controller_level": controller,
        "row_count": len(rows),
        "ready": bool(rows),
        "selected_count": _numeric_summary(selected_counts),
        "selected_token_cost": _numeric_summary(token_costs),
        "terminal_utility": _numeric_summary(utilities),
        "incomplete_selector_order_count": incomplete_order_count,
        "selector_unranked_count": _numeric_summary(selector_unranked_counts),
        "controller_stop_reasons": dict(sorted(stop_reasons.items())),
    }


def _problem_with_budget(
    problem: BacesProblem, *, k_max: int, token_budget: int | None = None
) -> BacesProblem:
    return BacesProblem(
        candidates=problem.candidates,
        weights=problem.weights,
        k_max=int(k_max),
        token_budget=token_budget,
        atom_ids=problem.atom_ids,
    )


def _learned_order_uids(
    row: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> list[str]:
    # The input contract is the learned selector's full-pool order.  A later
    # display projection, when present, is intentionally not the learned factor.
    raw = row.get("selector_ordered_indices")
    if raw is None:
        raw = row.get("selected_indices")
    if raw is None:
        raw = row.get("display_ordered_indices")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise FactorialAlignmentError("learned trace has no ordered-index array")
    indices = [_nonnegative_int(value, "learned ordered index") for value in raw]
    if len(set(indices)) != len(indices) or any(idx >= len(pool) for idx in indices):
        raise FactorialAlignmentError("learned ordered indices are duplicate or out of range")
    return [_candidate_uid(pool[idx], "learned candidate_pool") for idx in indices]


def _retrieval_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, int, str, str]:
    return (
        -_safe_float(candidate.get("retrieval_score")),
        -_safe_float(candidate.get("evidence_map_base_score")),
        _rank_value(candidate.get("union_pool_rank")),
        str(candidate.get("candidate_key") or candidate.get("candidate_uid") or ""),
        str(candidate.get("candidate_uid") or ""),
    )


def _map_quality_sort_key(
    candidate: Mapping[str, Any],
) -> tuple[float, float, float, int, str, str]:
    return (
        -_safe_float(candidate.get("evidence_map_quality_score")),
        -_safe_float(candidate.get("retrieval_score")),
        -_safe_float(candidate.get("evidence_map_base_score")),
        _rank_value(candidate.get("union_pool_rank")),
        str(candidate.get("candidate_key") or candidate.get("candidate_uid") or ""),
        str(candidate.get("candidate_uid") or ""),
    )


def _cost_retrieval_key(candidate: Mapping[str, Any]) -> tuple[int, float, str]:
    return (
        _nonnegative_int(candidate.get("mrec_token_cost"), "candidate cost"),
        -_safe_float(candidate.get("retrieval_score")),
        str(candidate.get("candidate_uid") or candidate.get("candidate_key") or ""),
    )


def _baces_zero_gain_fill_key(
    candidate: BacesCandidate,
    raw_candidate: Mapping[str, Any],
    *,
    weights: Sequence[int],
) -> tuple[int, int, int, float, str]:
    """Frozen BACES v0.3 fill key: cost, -direct, -partial, -retrieval, UID."""

    direct_weight = sum(
        int(weight) for weight, level in zip(weights, candidate.q) if int(level) == 2
    )
    partial_weight = sum(
        int(weight) for weight, level in zip(weights, candidate.q) if int(level) == 1
    )
    return (
        int(candidate.cost),
        -direct_weight,
        -partial_weight,
        -_safe_float(raw_candidate.get("retrieval_score")),
        candidate.key,
    )


def _candidate_cost(candidate: Mapping[str, Any], *, uid: str) -> int:
    for field in (
        "mrec_token_cost",
        "token_cost",
        "evidence_token_count",
        "prompt_token_count",
        "num_tokens",
    ):
        if candidate.get(field) is not None:
            return _nonnegative_int(candidate.get(field), f"{field} for {uid}")
    text = str(candidate.get("text") or candidate.get("evidence_text") or "")
    return max(1, len(text.split())) if text.strip() else 0


def _claim_atoms(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("claim_atoms")
    evidence_map = row.get("evidence_map")
    if raw is None and isinstance(evidence_map, Mapping):
        raw = evidence_map.get("claim_atoms")
    return [dict(atom) for atom in raw or [] if isinstance(atom, Mapping)]


def _atom_count(row: Mapping[str, Any]) -> int:
    count = len(_claim_atoms(row))
    if not 1 <= count <= 6:
        raise FactorialAlignmentError(f"BACES requires 1..6 claim atoms, got {count}")
    return count


def _pool_fingerprint(problem: BacesProblem) -> str:
    projection = [
        {"candidate_uid": candidate.key, "q": list(candidate.q), "cost": candidate.cost}
        for candidate in problem.candidates
    ]
    payload = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_objective(value: tuple[Any, ...]) -> list[Any]:
    return [*value[:-1], list(value[-1])]


def _cell_id(selector: str, controller: str) -> str:
    return f"{selector}__{controller}"


def _event_id(row: Mapping[str, Any], context: str) -> str:
    value = str(row.get("event_id") or "").strip()
    if not value:
        raise FactorialAlignmentError(f"{context} is missing event_id")
    return value


def _candidate_uid(row: Mapping[str, Any], context: str) -> str:
    value = str(row.get("candidate_uid") or "").strip()
    if not value:
        raise FactorialAlignmentError(f"{context} candidate is missing candidate_uid")
    return value


def _candidate_rows(value: Any, context: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FactorialAlignmentError(f"{context} must be an array")
    rows = list(value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise FactorialAlignmentError(f"{context} contains a non-object")
    return rows


def _require_unique(values: Sequence[str], context: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise FactorialAlignmentError(f"duplicate candidate_uid in {context}: {duplicates}")


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise FactorialAlignmentError(f"{context} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FactorialAlignmentError(f"{context} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise FactorialAlignmentError(f"{context} must be integral")
    if parsed < 0:
        raise FactorialAlignmentError(f"{context} must be non-negative")
    return parsed


def _rank_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _retrieval_score(candidate: Mapping[str, Any]) -> float:
    """Mirror evidence_quality.retrieval_score without importing torch-heavy build code."""

    rrf = _safe_float(candidate.get("qd_rrf_score") or candidate.get("rrf_score")) * 20.0
    values = (
        candidate.get("retrieval_score"),
        candidate.get("baseline_hybrid_score"),
        candidate.get("hybrid_score"),
        candidate.get("qd_max_question_hybrid"),
        candidate.get("max_question_hybrid"),
        rrf,
    )
    return float(min(1.0, max(0.0, max(_safe_float(value) for value in values))))


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "sum": None}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise FactorialAlignmentError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _read_jsonl_index(
    path: Path,
    *,
    artifact: str,
    keep_events: set[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise FactorialAlignmentError(f"{path}:{line_number} is not an object")
            event_id = _event_id(row, f"{artifact}:{line_number}")
            if event_id in seen:
                raise FactorialAlignmentError(f"duplicate event_id {event_id!r} in {artifact}")
            seen.add(event_id)
            if event_id in keep_events:
                out[event_id] = row
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
