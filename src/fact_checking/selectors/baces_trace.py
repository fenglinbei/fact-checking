"""Canonical construction and replay of BACES v0.3 sequence traces.

The trace adapter in this module deliberately has a narrow boundary:

* the exact solver sees only unit-weight claim atoms, ordinal candidate
  projections, integer costs, retrieval metadata, and the two budgets;
* the positive-gain exact solution is kept separate from the deterministic
  zero-gain rendering fill;
* ``solver_role`` is immutable under a same-set display reordering, while all
  display-prefix state is replayed from scratch.

No oracle, gold-label, verifier, learned-weight, or stance-state field from the
feature artifact is copied into the canonical trace.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .baces_exact import solve_exact
from .baces_objective import (
    BacesCandidate,
    BacesEvaluation,
    BacesProblem,
    compile_feature_problem,
    evaluate_core,
    evaluate_display,
    padded_auc,
    utility,
    validate_problem,
)


TRACE_SCHEMA_VERSION = "baces_sequence_trace_v0_3"
SOLVER_VERSION = "baces_exact_ordered_state_dp_v0_3"
CANDIDATE_PROJECTION_SCHEMA = "baces_solver_projection_v0_3"
DEDUP_POLICY_VERSION = "baces_dedup_v0_3"
SELECTION_POLICY = "baces_lexicographic_early_coverage_v0_3"
MAIN_DISPLAY_ORDER_POLICY = "baces_early_coverage"
MREC_COMPATIBILITY_VIEW_VERSION = "baces_minimal_mrec_adapter_v0_3"

_RETRIEVAL_SCORE_FIELDS = (
    "retrieval_score",
    "hybrid_score",
    "baseline_hybrid_score",
    "base_score",
)

_CANDIDATE_PROJECTION_FIELDS = (
    "candidate_pool_idx",
    "candidate_stable_key",
    "candidate_uid",
    "evidence_id",
    "ordinal_quality_vector",
    "pair_coverage_levels",
    "token_cost",
    "retrieval_score",
    "direct_coverage_weight",
    "partial_coverage_weight",
)

_BACES_STEP_FIELDS = (
    "step",
    "solver_role",
    "solver_core_position",
    "operation",
    "display_operation",
    "candidate_pool_idx",
    "candidate_stable_key",
    "evidence_id",
    "token_cost",
    "valid_coverage_atom_ids",
    "pair_coverage_levels",
    "display_coverage_levels_before",
    "display_coverage_levels_after",
    "display_upgraded_atom_ids",
    "display_marginal_coverage_units",
    "display_cumulative_coverage_units",
    "display_cumulative_normalized_coverage",
    "display_weighted_acquisition_time_so_far",
    "target_coverage_reached",
    "cue_atom_id",
    "cue_text",
    "cue_source",
)

_TRACE_FINGERPRINT_FIELDS = (
    "schema_version",
    "solver_version",
    "map_schema_version",
    "candidate_pool_projection_schema",
    "dedup_policy_version",
    "cost_tokenizer_id",
    "cost_tokenizer_revision",
    "selection_policy",
    "event_id",
    "claim_atoms",
    "atom_weights",
    "ordinal_levels",
    "partial_utility_lambda",
    "k_min",
    "k_max",
    "token_budget",
    "candidate_pool_fingerprint",
    "k_pool_raw",
    "k_pool_dedup",
    "k_core",
    "k_sel",
    "coverage_core_indices",
    "coverage_core_keys",
    "selected_indices",
    "selected_keys",
    "display_ordered_indices",
    "display_ordered_keys",
    "selected_set_fingerprint",
    "display_order_fingerprint",
    "display_order_policy",
    "pool_reachable_ordinal_units",
    "terminal_ordinal_state",
    "terminal_ordinal_coverage_units",
    "terminal_normalized_coverage",
    "terminal_reachable_normalized_coverage",
    "core_token_cost",
    "selected_token_cost",
    "solver_objective_tuple",
    "core_weighted_coverage_acquisition_time",
    "display_weighted_coverage_acquisition_time",
    "prefix_auc_horizon",
    "core_padded_prefix_auc",
    "display_padded_prefix_auc",
    "zero_gain_fill_count",
    "min_count_unreachable",
    "baces_steps",
)

_FORBIDDEN_STANCE_STEP_FIELDS = frozenset(
    {
        "state_before",
        "state_after",
        "atom_states_before",
        "atom_states_after",
        "conflicted_atom_ids",
        "reasoning_transition",
    }
)


def build_exact_trace(
    feature_row: Mapping[str, Any],
    k_min: int,
    k_max: int,
    token_budget: int | None = None,
    cost_overrides: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build one canonical exact-core plus soft-fill BACES trace.

    Candidate identity is the compact, non-empty ``candidate_uid``.  The
    solver-visible pool is sorted by that identity before indices are assigned,
    so feature-array permutation cannot change the canonical trace.
    """

    if not isinstance(feature_row, Mapping):
        raise TypeError("feature_row must be a mapping")
    k_min = _require_int(k_min, name="k_min", minimum=0)
    k_max = _require_int(k_max, name="k_max", minimum=0)
    if k_min > k_max:
        raise ValueError("k_min must be less than or equal to k_max")
    if token_budget is not None:
        token_budget = _require_int(token_budget, name="token_budget", minimum=0)

    event_id = _compact(feature_row.get("event_id"))
    if not event_id:
        raise ValueError("feature_row must contain a non-empty event_id")

    raw_candidates = feature_row.get("candidates")
    if raw_candidates is None:
        raw_candidates = []
    if not isinstance(raw_candidates, (list, tuple)):
        raise TypeError("feature_row['candidates'] must be a list or tuple")
    raw_by_uid: dict[str, Mapping[str, Any]] = {}
    for raw_index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"candidates[{raw_index}] must be a mapping")
        uid = _compact(candidate.get("candidate_uid"))
        if not uid:
            raise ValueError(
                f"candidates[{raw_index}] has no candidate_uid; "
                "BACES trace identity is candidate_uid"
            )
        if uid in raw_by_uid:
            raise ValueError(f"duplicate candidate_uid: {uid!r}")
        raw_by_uid[uid] = candidate

    compiled = compile_feature_problem(
        feature_row,
        k_max=k_max,
        token_budget=token_budget,
        weights=None,
        cost_overrides=cost_overrides,
    )
    if any(candidate.key != candidate.uid for candidate in compiled.candidates):
        raise ValueError("canonical BACES candidate key must equal candidate_uid")

    problem = BacesProblem(
        candidates=tuple(sorted(compiled.candidates, key=lambda candidate: candidate.key)),
        weights=compiled.weights,
        k_max=compiled.k_max,
        token_budget=compiled.token_budget,
        atom_ids=compiled.atom_ids,
    )
    validate_problem(problem)
    candidate_pool = _project_candidates(problem, raw_by_uid)
    retrieval_by_key = {
        str(row["candidate_stable_key"]): row.get("retrieval_score")
        for row in candidate_pool
    }

    core = solve_exact(problem)
    fill_keys = _soft_fill_keys(
        problem,
        core,
        k_min=k_min,
        retrieval_by_key=retrieval_by_key,
    )
    display_keys = core.keys + fill_keys
    display = evaluate_display(problem, display_keys)
    atom_texts = _claim_atom_texts(feature_row, problem.atom_ids)
    index_by_key = _index_by_key(candidate_pool)

    trace: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "map_schema_version": _map_schema_version(feature_row),
        "candidate_pool_projection_schema": CANDIDATE_PROJECTION_SCHEMA,
        "dedup_policy_version": DEDUP_POLICY_VERSION,
        "cost_tokenizer_id": _compact(feature_row.get("cost_tokenizer_id")),
        "cost_tokenizer_revision": _compact(feature_row.get("cost_tokenizer_revision")),
        "selection_policy": SELECTION_POLICY,
        "event_id": event_id,
        "claim_atoms": list(problem.atom_ids),
        "claim_atom_texts": atom_texts,
        "atom_weights": {
            atom_id: weight for atom_id, weight in zip(problem.atom_ids, problem.weights)
        },
        "ordinal_levels": {"invalid": 0, "partial": 1, "direct": 2},
        "partial_utility_lambda": 0.5,
        "k_min": k_min,
        "k_max": k_max,
        "token_budget": token_budget,
        "candidate_pool": candidate_pool,
        "k_pool_raw": len(raw_candidates),
        "k_pool_dedup": len(candidate_pool),
        "k_core": core.length,
        "k_sel": display.length,
        "coverage_core_indices": [index_by_key[key] for key in core.keys],
        "coverage_core_keys": list(core.keys),
        "selected_indices": [index_by_key[key] for key in display.keys],
        "selected_keys": list(display.keys),
        "display_ordered_indices": [index_by_key[key] for key in display.keys],
        "display_ordered_keys": list(display.keys),
        "selector_ordered_indices": [index_by_key[key] for key in display.keys],
        "display_order_policy": MAIN_DISPLAY_ORDER_POLICY,
        "zero_gain_fill_count": len(fill_keys),
        "min_count_unreachable": display.length < k_min,
        "mrec_compatibility_view_version": MREC_COMPATIBILITY_VIEW_VERSION,
    }
    trace.update(_metric_fields(problem, core=core, display=display))
    trace["candidate_pool_fingerprint"] = _candidate_pool_fingerprint(trace)
    trace["selected_set_fingerprint"] = _selected_set_fingerprint(
        event_id, display.keys
    )
    trace["display_order_fingerprint"] = _display_order_fingerprint(
        event_id, display.keys
    )
    trace["baces_steps"] = _build_steps(
        problem,
        candidate_pool,
        display,
        core_keys=core.keys,
        terminal_state=core.state,
        atom_texts=atom_texts,
    )
    trace["mrec_steps"] = _minimal_mrec_steps(trace["baces_steps"])
    trace["trace_fingerprint"] = _trace_fingerprint(trace)
    return trace


def replay_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Independently replay and validate a canonical BACES trace.

    The function never trusts stored prefix state.  It reconstructs the exact
    problem from the candidate projection, solves the core again, reconstructs
    the frozen fill set, and then replays the current display order by
    componentwise maximum.  Invalid artifacts are reported in ``errors`` rather
    than raising.  Recomputed fields are available both at the result top level
    and in ``derived`` for convenient callers.
    """

    if not isinstance(trace, Mapping):
        return _replay_result(["trace must be a mapping"], {})

    errors: list[str] = []
    _expect(errors, "schema_version", trace.get("schema_version"), TRACE_SCHEMA_VERSION)
    _expect(errors, "solver_version", trace.get("solver_version"), SOLVER_VERSION)
    _expect(
        errors,
        "candidate_pool_projection_schema",
        trace.get("candidate_pool_projection_schema"),
        CANDIDATE_PROJECTION_SCHEMA,
    )
    _expect(
        errors,
        "dedup_policy_version",
        trace.get("dedup_policy_version"),
        DEDUP_POLICY_VERSION,
    )
    _expect(errors, "selection_policy", trace.get("selection_policy"), SELECTION_POLICY)

    parsed = _problem_from_trace(trace, errors)
    if parsed is None:
        return _replay_result(errors, {})
    problem, candidate_pool, k_min = parsed
    event_id = _compact(trace.get("event_id"))
    index_by_key = _index_by_key(candidate_pool)
    retrieval_by_key = {
        str(row["candidate_stable_key"]): row.get("retrieval_score")
        for row in candidate_pool
    }

    try:
        core = solve_exact(problem)
        fill_keys = _soft_fill_keys(
            problem,
            core,
            k_min=k_min,
            retrieval_by_key=retrieval_by_key,
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"exact core replay failed: {exc}")
        return _replay_result(errors, {})

    expected_selected_set = set(core.keys + fill_keys)
    display_keys = _string_sequence(
        trace.get("display_ordered_keys"),
        name="display_ordered_keys",
        errors=errors,
    )
    if display_keys is None:
        return _replay_result(errors, {})
    if len(display_keys) != len(set(display_keys)):
        errors.append("display_ordered_keys must contain distinct candidate keys")
    unknown = [key for key in display_keys if key not in index_by_key]
    if unknown:
        errors.append(f"display_ordered_keys contains unknown keys: {unknown!r}")
        return _replay_result(errors, {})
    if set(display_keys) != expected_selected_set:
        errors.append(
            "selected set differs from exact core plus canonical ZERO_GAIN_FILL: "
            f"expected={sorted(expected_selected_set)!r}, got={sorted(set(display_keys))!r}"
        )
    if len(display_keys) > problem.k_max:
        errors.append(
            f"display length {len(display_keys)} exceeds k_max={problem.k_max}"
        )

    policy = _compact(trace.get("display_order_policy"))
    if not policy:
        errors.append("display_order_policy must be a non-empty string")
    if policy == MAIN_DISPLAY_ORDER_POLICY and display_keys != core.keys + fill_keys:
        errors.append(
            "baces_early_coverage display order must be exact core followed by canonical fill"
        )

    try:
        display = evaluate_display(problem, display_keys)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"display replay failed: {exc}")
        return _replay_result(errors, {})
    if display.state != core.state:
        errors.append(
            "display terminal ordinal state differs from exact core terminal state: "
            f"expected={core.state!r}, got={display.state!r}"
        )

    atom_texts = _atom_texts_from_trace(trace, problem.atom_ids, errors)
    expected_steps = _build_steps(
        problem,
        candidate_pool,
        display,
        core_keys=core.keys,
        terminal_state=core.state,
        atom_texts=atom_texts,
    )
    expected_mrec_steps = _minimal_mrec_steps(expected_steps)
    expected_indices = [index_by_key[key] for key in display.keys]
    expected_core_indices = [index_by_key[key] for key in core.keys]

    expected: dict[str, Any] = dict(trace)
    expected.update(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "solver_version": SOLVER_VERSION,
            "candidate_pool_projection_schema": CANDIDATE_PROJECTION_SCHEMA,
            "dedup_policy_version": DEDUP_POLICY_VERSION,
            "selection_policy": SELECTION_POLICY,
            "k_pool_dedup": len(candidate_pool),
            "k_core": core.length,
            "k_sel": display.length,
            "coverage_core_indices": expected_core_indices,
            "coverage_core_keys": list(core.keys),
            "selected_indices": expected_indices,
            "selected_keys": list(display.keys),
            "display_ordered_indices": expected_indices,
            "display_ordered_keys": list(display.keys),
            "selector_ordered_indices": expected_indices,
            "zero_gain_fill_count": len(fill_keys),
            "min_count_unreachable": display.length < k_min,
            "baces_steps": expected_steps,
            "mrec_compatibility_view_version": MREC_COMPATIBILITY_VIEW_VERSION,
            "mrec_steps": expected_mrec_steps,
        }
    )
    expected.update(_metric_fields(problem, core=core, display=display))
    expected["candidate_pool_fingerprint"] = _candidate_pool_fingerprint(expected)
    expected["selected_set_fingerprint"] = _selected_set_fingerprint(
        event_id, display.keys
    )
    expected["display_order_fingerprint"] = _display_order_fingerprint(
        event_id, display.keys
    )
    expected["trace_fingerprint"] = _trace_fingerprint(expected)

    for field in (
        "candidate_pool_fingerprint",
        "k_pool_dedup",
        "k_core",
        "k_sel",
        "coverage_core_indices",
        "coverage_core_keys",
        "selected_indices",
        "selected_keys",
        "display_ordered_indices",
        "display_ordered_keys",
        "selector_ordered_indices",
        "selected_set_fingerprint",
        "display_order_fingerprint",
        "pool_reachable_ordinal_units",
        "terminal_ordinal_state",
        "terminal_ordinal_coverage_units",
        "terminal_normalized_coverage",
        "terminal_reachable_normalized_coverage",
        "core_token_cost",
        "selected_token_cost",
        "solver_objective_tuple",
        "core_weighted_coverage_acquisition_time",
        "display_weighted_coverage_acquisition_time",
        "prefix_auc_horizon",
        "core_padded_prefix_auc",
        "display_padded_prefix_auc",
        "zero_gain_fill_count",
        "min_count_unreachable",
        "mrec_compatibility_view_version",
    ):
        _expect(errors, field, trace.get(field), expected[field])

    _validate_steps(trace.get("baces_steps"), expected_steps, errors)
    _validate_mrec_steps(trace.get("mrec_steps"), expected_mrec_steps, errors)
    _expect(
        errors,
        "trace_fingerprint (stored artifact)",
        trace.get("trace_fingerprint"),
        _trace_fingerprint(trace),
    )
    _expect(
        errors,
        "trace_fingerprint (canonical replay)",
        trace.get("trace_fingerprint"),
        expected["trace_fingerprint"],
    )

    derived = {
        field: expected[field]
        for field in (
            "candidate_pool_fingerprint",
            "coverage_core_indices",
            "coverage_core_keys",
            "selected_indices",
            "selected_keys",
            "display_ordered_indices",
            "display_ordered_keys",
            "selected_set_fingerprint",
            "display_order_fingerprint",
            "k_core",
            "k_sel",
            "pool_reachable_ordinal_units",
            "terminal_ordinal_state",
            "terminal_ordinal_coverage_units",
            "terminal_normalized_coverage",
            "terminal_reachable_normalized_coverage",
            "core_token_cost",
            "selected_token_cost",
            "solver_objective_tuple",
            "core_weighted_coverage_acquisition_time",
            "display_weighted_coverage_acquisition_time",
            "prefix_auc_horizon",
            "core_padded_prefix_auc",
            "display_padded_prefix_auc",
            "zero_gain_fill_count",
            "min_count_unreachable",
            "baces_steps",
            "mrec_steps",
            "trace_fingerprint",
        )
    }
    return _replay_result(errors, derived)


def reorder_selected_trace(
    trace: Mapping[str, Any],
    ordered_keys: Iterable[str],
    display_order_policy: str,
) -> dict[str, Any]:
    """Return a strict same-set trace with freshly replayed display state.

    The exact core sequence and each candidate's ``CORE``/``FILL`` role remain
    frozen.  Only the display permutation and its realized prefix trajectory
    are changed.
    """

    replay = replay_trace(trace)
    if not replay["ok"]:
        raise ValueError("cannot reorder an invalid BACES trace: " + "; ".join(replay["errors"]))
    try:
        ordered = tuple(ordered_keys)
    except TypeError as exc:
        raise TypeError("ordered_keys must be an iterable of candidate keys") from exc
    if any(not isinstance(key, str) or not key.strip() for key in ordered):
        raise TypeError("ordered_keys must contain non-empty strings")
    if len(ordered) != len(set(ordered)):
        raise ValueError("ordered_keys must contain distinct candidate keys")
    source_keys = tuple(str(key) for key in trace.get("display_ordered_keys") or [])
    if set(ordered) != set(source_keys) or len(ordered) != len(source_keys):
        raise ValueError("ordered_keys must be a permutation of the frozen selected set")
    policy = _compact(display_order_policy)
    if not policy:
        raise ValueError("display_order_policy must be a non-empty string")

    errors: list[str] = []
    parsed = _problem_from_trace(trace, errors)
    if parsed is None or errors:
        raise AssertionError("validated trace could not be reconstructed")
    problem, candidate_pool, k_min = parsed
    core_keys = tuple(str(key) for key in trace.get("coverage_core_keys") or [])
    core = evaluate_core(problem, core_keys)
    display = evaluate_display(problem, ordered)
    index_by_key = _index_by_key(candidate_pool)
    atom_texts = _atom_texts_from_trace(trace, problem.atom_ids, errors)
    if errors:
        raise AssertionError("validated trace has invalid claim_atom_texts")

    reordered = deepcopy(dict(trace))
    reordered["source_display_order_policy"] = str(trace.get("display_order_policy") or "")
    reordered["source_display_order_fingerprint"] = str(
        trace.get("display_order_fingerprint") or ""
    )
    reordered["same_set_control"] = True
    reordered["display_order_policy"] = policy
    indices = [index_by_key[key] for key in ordered]
    reordered["selected_indices"] = indices
    reordered["selected_keys"] = list(ordered)
    reordered["display_ordered_indices"] = indices
    reordered["display_ordered_keys"] = list(ordered)
    reordered["selector_ordered_indices"] = indices
    reordered["k_sel"] = len(ordered)
    reordered["min_count_unreachable"] = len(ordered) < k_min
    reordered.update(_metric_fields(problem, core=core, display=display))
    reordered["selected_set_fingerprint"] = _selected_set_fingerprint(
        str(reordered["event_id"]), ordered
    )
    reordered["display_order_fingerprint"] = _display_order_fingerprint(
        str(reordered["event_id"]), ordered
    )
    reordered["baces_steps"] = _build_steps(
        problem,
        candidate_pool,
        display,
        core_keys=core.keys,
        terminal_state=core.state,
        atom_texts=atom_texts,
    )
    reordered["mrec_steps"] = _minimal_mrec_steps(reordered["baces_steps"])
    reordered["trace_fingerprint"] = _trace_fingerprint(reordered)

    verification = replay_trace(reordered)
    if not verification["ok"]:
        raise AssertionError(
            "internally generated same-set trace failed replay: "
            + "; ".join(verification["errors"])
        )
    return reordered


def _project_candidates(
    problem: BacesProblem,
    raw_by_uid: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projections: list[dict[str, Any]] = []
    for pool_index, candidate in enumerate(problem.candidates):
        raw = raw_by_uid[candidate.key]
        direct_weight = sum(
            weight for weight, level in zip(problem.weights, candidate.q) if level == 2
        )
        partial_weight = sum(
            weight for weight, level in zip(problem.weights, candidate.q) if level == 1
        )
        pair_levels = {
            atom_id: level for atom_id, level in zip(problem.atom_ids, candidate.q)
        }
        projections.append(
            {
                "candidate_pool_idx": pool_index,
                "candidate_stable_key": candidate.key,
                "candidate_uid": candidate.key,
                "evidence_id": _compact(raw.get("evidence_id")) or candidate.key,
                "ordinal_quality_vector": list(candidate.q),
                "pair_coverage_levels": pair_levels,
                "token_cost": candidate.cost,
                "retrieval_score": _canonical_retrieval_score(raw),
                "direct_coverage_weight": direct_weight,
                "partial_coverage_weight": partial_weight,
            }
        )
    return projections


def _soft_fill_keys(
    problem: BacesProblem,
    core: BacesEvaluation,
    *,
    k_min: int,
    retrieval_by_key: Mapping[str, Any],
) -> tuple[str, ...]:
    needed = max(0, min(k_min, problem.k_max) - core.length)
    if needed == 0:
        return ()
    core_set = set(core.keys)

    def fill_rank(candidate: BacesCandidate) -> tuple[object, ...]:
        direct_weight = sum(
            weight for weight, level in zip(problem.weights, candidate.q) if level == 2
        )
        partial_weight = sum(
            weight for weight, level in zip(problem.weights, candidate.q) if level == 1
        )
        retrieval = _finite_float(retrieval_by_key.get(candidate.key))
        retrieval_rank = math.inf if retrieval is None else -retrieval
        return (
            candidate.cost,
            -direct_weight,
            -partial_weight,
            retrieval_rank,
            candidate.key,
        )

    eligible = sorted(
        (
            candidate
            for candidate in problem.candidates
            if candidate.key not in core_set
            and all(level <= terminal for level, terminal in zip(candidate.q, core.state))
        ),
        key=fill_rank,
    )
    remaining = (
        None
        if problem.token_budget is None
        else problem.token_budget - core.token_cost
    )
    fill: list[str] = []
    spent = 0
    for candidate in eligible:
        if len(fill) >= needed:
            break
        if remaining is not None and spent + candidate.cost > remaining:
            # Cost is the first ascending fill key, so no later candidate can
            # make this prefix feasible.
            break
        fill.append(candidate.key)
        spent += candidate.cost
    return tuple(fill)


def _metric_fields(
    problem: BacesProblem,
    *,
    core: BacesEvaluation,
    display: BacesEvaluation,
) -> dict[str, Any]:
    reachable_state = tuple(
        max((candidate.q[index] for candidate in problem.candidates), default=0)
        for index in range(len(problem.weights))
    )
    reachable_utility = utility(reachable_state, problem.weights)
    max_utility = 2 * sum(problem.weights)
    terminal_normalized = core.utility / max_utility
    reachable_normalized = (
        core.utility / reachable_utility if reachable_utility > 0 else 0.0
    )
    return {
        "pool_reachable_ordinal_units": reachable_utility,
        "terminal_ordinal_state": {
            atom_id: level for atom_id, level in zip(problem.atom_ids, core.state)
        },
        "terminal_ordinal_coverage_units": core.utility,
        "terminal_normalized_coverage": terminal_normalized,
        "terminal_reachable_normalized_coverage": reachable_normalized,
        "core_token_cost": core.token_cost,
        "selected_token_cost": display.token_cost,
        "solver_objective_tuple": [
            -core.utility,
            core.acquisition_time,
            core.length,
            core.token_cost,
            list(core.keys),
        ],
        "core_weighted_coverage_acquisition_time": core.acquisition_time,
        "display_weighted_coverage_acquisition_time": display.acquisition_time,
        "prefix_auc_horizon": problem.k_max,
        "core_padded_prefix_auc": padded_auc(core, problem.k_max),
        "display_padded_prefix_auc": padded_auc(display, problem.k_max),
    }


def _build_steps(
    problem: BacesProblem,
    candidate_pool: Sequence[Mapping[str, Any]],
    display: BacesEvaluation,
    *,
    core_keys: Sequence[str],
    terminal_state: Sequence[int],
    atom_texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    projection_by_key = {
        str(row["candidate_stable_key"]): row for row in candidate_pool
    }
    core_position = {key: index for index, key in enumerate(core_keys, start=1)}
    denominator = 2 * sum(problem.weights)
    steps: list[dict[str, Any]] = []
    for evaluation_step in display.steps:
        key = evaluation_step.key
        projection = projection_by_key[key]
        candidate = next(candidate for candidate in problem.candidates if candidate.key == key)
        role = "CORE" if key in core_position else "FILL"
        upgraded_atom_ids = [
            atom_id
            for atom_id, before, after in zip(
                problem.atom_ids,
                evaluation_step.state_before,
                evaluation_step.state_after,
            )
            if after > before
        ]
        valid_atom_ids = [
            atom_id
            for atom_id, level in zip(problem.atom_ids, candidate.q)
            if level > 0
        ]
        cue_atom_id = (
            upgraded_atom_ids[0]
            if upgraded_atom_ids
            else (valid_atom_ids[0] if valid_atom_ids else problem.atom_ids[0])
        )
        steps.append(
            {
                "step": evaluation_step.position,
                "solver_role": role,
                "solver_core_position": core_position.get(key),
                "operation": "COVER" if role == "CORE" else "ZERO_GAIN_FILL",
                "display_operation": (
                    "ORDINAL_UPGRADE"
                    if evaluation_step.delta > 0
                    else "DISPLAY_ZERO_GAIN"
                ),
                "candidate_pool_idx": int(projection["candidate_pool_idx"]),
                "candidate_stable_key": key,
                "evidence_id": str(projection["evidence_id"]),
                "token_cost": evaluation_step.candidate_cost,
                "valid_coverage_atom_ids": valid_atom_ids,
                "pair_coverage_levels": {
                    atom_id: level for atom_id, level in zip(problem.atom_ids, candidate.q)
                },
                "display_coverage_levels_before": {
                    atom_id: level
                    for atom_id, level in zip(
                        problem.atom_ids, evaluation_step.state_before
                    )
                },
                "display_coverage_levels_after": {
                    atom_id: level
                    for atom_id, level in zip(
                        problem.atom_ids, evaluation_step.state_after
                    )
                },
                "display_upgraded_atom_ids": upgraded_atom_ids,
                "display_marginal_coverage_units": evaluation_step.delta,
                "display_cumulative_coverage_units": evaluation_step.cumulative_utility,
                "display_cumulative_normalized_coverage": (
                    evaluation_step.cumulative_utility / denominator
                ),
                "display_weighted_acquisition_time_so_far": (
                    evaluation_step.acquisition_time_so_far
                ),
                "target_coverage_reached": (
                    tuple(evaluation_step.state_after) == tuple(terminal_state)
                ),
                "cue_atom_id": cue_atom_id,
                "cue_text": _compact(atom_texts.get(cue_atom_id)) or cue_atom_id,
                "cue_source": "claim_atom",
            }
        )
    return steps


def _minimal_mrec_steps(
    baces_steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Provide only the cue/index compatibility needed by legacy builders.

    In particular, this view never fabricates U/S/R/Q/C stance states or MREC
    transition operations.
    """

    output: list[dict[str, Any]] = []
    for step in baces_steps:
        after = step.get("display_coverage_levels_after")
        after_levels = dict(after) if isinstance(after, Mapping) else {}
        output.append(
            {
                "step": int(step["step"]),
                "candidate_idx": int(step["candidate_pool_idx"]),
                "selector_candidate_idx": int(step["candidate_pool_idx"]),
                "candidate_uid": str(step["candidate_stable_key"]),
                "evidence_id": str(step["evidence_id"]),
                "atom_id": str(step["cue_atom_id"]),
                "atom_text": str(step["cue_text"]),
                "cue_text": str(step["cue_text"]),
                "cue_source": "claim_atom",
                "covered_atom_ids": list(step["valid_coverage_atom_ids"]),
                "token_cost": int(step["token_cost"]),
                "target_resolved": bool(step["target_coverage_reached"]),
                "resolved_atom_rate": float(
                    step["display_cumulative_normalized_coverage"]
                ),
                "unresolved_atom_ids": sorted(
                    atom_id for atom_id, level in after_levels.items() if int(level) == 0
                ),
                "not_direct_atom_ids": sorted(
                    atom_id for atom_id, level in after_levels.items() if int(level) < 2
                ),
            }
        )
    return output


def _problem_from_trace(
    trace: Mapping[str, Any], errors: list[str]
) -> tuple[BacesProblem, list[dict[str, Any]], int] | None:
    atom_ids_raw = trace.get("claim_atoms")
    if not isinstance(atom_ids_raw, list) or not atom_ids_raw:
        errors.append("claim_atoms must be a non-empty list")
        return None
    atom_ids = tuple(_compact(atom_id) for atom_id in atom_ids_raw)
    if any(not atom_id for atom_id in atom_ids) or len(atom_ids) != len(set(atom_ids)):
        errors.append("claim_atoms must contain distinct non-empty strings")
        return None
    if len(atom_ids) > 6:
        errors.append("BACES v0.3 supports at most six claim atoms")
        return None

    raw_weights = trace.get("atom_weights")
    if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(atom_ids):
        errors.append("atom_weights keys must exactly match claim_atoms")
        return None
    weights: list[int] = []
    for atom_id in atom_ids:
        value = raw_weights.get(atom_id)
        if not _is_int(value) or int(value) < 1:
            errors.append(f"atom_weights[{atom_id!r}] must be a positive integer")
            return None
        weights.append(int(value))
    if any(weight != 1 for weight in weights):
        errors.append("canonical BACES main trace must use unit atom weights")

    k_min = trace.get("k_min")
    k_max = trace.get("k_max")
    token_budget = trace.get("token_budget")
    if not _is_int(k_min) or int(k_min) < 0:
        errors.append("k_min must be a non-negative integer")
        return None
    if not _is_int(k_max) or int(k_max) < 0:
        errors.append("k_max must be a non-negative integer")
        return None
    k_min = int(k_min)
    k_max = int(k_max)
    if k_min > k_max:
        errors.append("k_min must be less than or equal to k_max")
        return None
    if token_budget is not None and (
        not _is_int(token_budget) or int(token_budget) < 0
    ):
        errors.append("token_budget must be a non-negative integer or null")
        return None
    token_budget = None if token_budget is None else int(token_budget)

    raw_pool = trace.get("candidate_pool")
    if not isinstance(raw_pool, list):
        errors.append("candidate_pool must be a list")
        return None
    candidate_pool: list[dict[str, Any]] = []
    candidates: list[BacesCandidate] = []
    fatal = False
    for index, raw_projection in enumerate(raw_pool):
        if not isinstance(raw_projection, Mapping):
            errors.append(f"candidate_pool[{index}] must be a mapping")
            fatal = True
            continue
        projection = dict(raw_projection)
        key = _compact(projection.get("candidate_stable_key"))
        uid = _compact(projection.get("candidate_uid"))
        if not key or uid != key:
            errors.append(
                f"candidate_pool[{index}] candidate_uid must equal non-empty candidate_stable_key"
            )
            fatal = True
            continue
        if projection.get("candidate_pool_idx") != index:
            errors.append(
                f"candidate_pool[{index}].candidate_pool_idx mismatch: "
                f"got {projection.get('candidate_pool_idx')!r}"
            )
        levels = projection.get("pair_coverage_levels")
        vector = projection.get("ordinal_quality_vector")
        if not isinstance(levels, Mapping) or set(levels) != set(atom_ids):
            errors.append(
                f"candidate_pool[{index}].pair_coverage_levels keys must match claim_atoms"
            )
            fatal = True
            continue
        q: list[int] = []
        for atom_id in atom_ids:
            level = levels.get(atom_id)
            if not _is_int(level) or int(level) not in (0, 1, 2):
                errors.append(
                    f"candidate_pool[{index}] level for {atom_id!r} must be 0, 1, or 2"
                )
                fatal = True
                break
            q.append(int(level))
        if fatal and len(q) != len(atom_ids):
            continue
        if vector != q:
            errors.append(
                f"candidate_pool[{index}].ordinal_quality_vector mismatch: "
                f"expected={q!r}, got={vector!r}"
            )
        cost = projection.get("token_cost")
        if not _is_int(cost) or int(cost) < 0:
            errors.append(f"candidate_pool[{index}].token_cost must be non-negative integer")
            fatal = True
            continue
        retrieval = projection.get("retrieval_score")
        if retrieval is not None and _finite_float(retrieval) is None:
            errors.append(
                f"candidate_pool[{index}].retrieval_score must be finite numeric or null"
            )
            fatal = True
            continue
        expected_direct = sum(weight for weight, level in zip(weights, q) if level == 2)
        expected_partial = sum(weight for weight, level in zip(weights, q) if level == 1)
        _expect(
            errors,
            f"candidate_pool[{index}].direct_coverage_weight",
            projection.get("direct_coverage_weight"),
            expected_direct,
        )
        _expect(
            errors,
            f"candidate_pool[{index}].partial_coverage_weight",
            projection.get("partial_coverage_weight"),
            expected_partial,
        )
        candidates.append(
            BacesCandidate(
                key=key,
                q=tuple(q),
                cost=int(cost),
                uid=uid,
                display_key=_compact(projection.get("evidence_id")) or key,
            )
        )
        candidate_pool.append(projection)
    if fatal:
        return None
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        errors.append("candidate_pool candidate keys must be distinct")
        return None
    if keys != sorted(keys):
        errors.append("candidate_pool must be canonically sorted by candidate_stable_key")

    problem = BacesProblem(
        candidates=tuple(candidates),
        weights=tuple(weights),
        k_max=k_max,
        token_budget=token_budget,
        atom_ids=atom_ids,
    )
    try:
        validate_problem(problem)
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid projected BACES problem: {exc}")
        return None
    return problem, candidate_pool, k_min


def _validate_steps(
    raw_steps: Any,
    expected_steps: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    if not isinstance(raw_steps, list):
        errors.append("baces_steps must be a list")
        return
    if len(raw_steps) != len(expected_steps):
        errors.append(
            f"baces_steps length mismatch: expected={len(expected_steps)}, got={len(raw_steps)}"
        )
    for index, expected in enumerate(expected_steps):
        if index >= len(raw_steps):
            break
        actual = raw_steps[index]
        if not isinstance(actual, Mapping):
            errors.append(f"baces_steps[{index}] must be a mapping")
            continue
        forbidden = sorted(_FORBIDDEN_STANCE_STEP_FIELDS.intersection(actual))
        if forbidden:
            errors.append(
                f"baces_steps[{index}] contains forbidden stance fields: {forbidden!r}"
            )
        for field in _BACES_STEP_FIELDS:
            _expect(
                errors,
                f"baces_steps[{index}].{field}",
                actual.get(field),
                expected[field],
            )


def _validate_mrec_steps(
    raw_steps: Any,
    expected_steps: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    if not isinstance(raw_steps, list):
        errors.append("mrec_steps must be a list")
        return
    if len(raw_steps) != len(expected_steps):
        errors.append(
            f"mrec_steps length mismatch: expected={len(expected_steps)}, got={len(raw_steps)}"
        )
    for index, expected in enumerate(expected_steps):
        if index >= len(raw_steps):
            break
        actual = raw_steps[index]
        if not isinstance(actual, Mapping):
            errors.append(f"mrec_steps[{index}] must be a mapping")
            continue
        forbidden = sorted(_FORBIDDEN_STANCE_STEP_FIELDS.intersection(actual))
        if forbidden:
            errors.append(
                f"mrec_steps[{index}] contains forbidden stance fields: {forbidden!r}"
            )
        for field, value in expected.items():
            _expect(errors, f"mrec_steps[{index}].{field}", actual.get(field), value)


def _candidate_pool_fingerprint(trace: Mapping[str, Any]) -> str:
    pool = trace.get("candidate_pool")
    canonical_pool: list[dict[str, Any]] = []
    if isinstance(pool, list):
        for row in pool:
            if isinstance(row, Mapping):
                canonical_pool.append(
                    {field: row.get(field) for field in _CANDIDATE_PROJECTION_FIELDS}
                )
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "map_schema_version": str(trace.get("map_schema_version") or ""),
        "candidate_pool_projection_schema": CANDIDATE_PROJECTION_SCHEMA,
        "dedup_policy_version": DEDUP_POLICY_VERSION,
        "cost_tokenizer_id": str(trace.get("cost_tokenizer_id") or ""),
        "cost_tokenizer_revision": str(trace.get("cost_tokenizer_revision") or ""),
        "event_id": str(trace.get("event_id") or ""),
        "claim_atoms": trace.get("claim_atoms"),
        "atom_weights": trace.get("atom_weights"),
        "ordinal_levels": trace.get("ordinal_levels"),
        "partial_utility_lambda": trace.get("partial_utility_lambda"),
        "k_min": trace.get("k_min"),
        "k_max": trace.get("k_max"),
        "token_budget": trace.get("token_budget"),
        "candidate_pool": canonical_pool,
    }
    return _sha256(payload)


def _selected_set_fingerprint(event_id: str, keys: Iterable[str]) -> str:
    return _sha256({"event_id": event_id, "selected_keys": sorted(keys)})


def _display_order_fingerprint(event_id: str, keys: Iterable[str]) -> str:
    return _sha256({"event_id": event_id, "display_ordered_keys": list(keys)})


def _trace_fingerprint(trace: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for field in _TRACE_FINGERPRINT_FIELDS:
        value = trace.get(field)
        if field == "baces_steps" and isinstance(value, list):
            value = [
                {name: step.get(name) for name in _BACES_STEP_FIELDS}
                for step in value
                if isinstance(step, Mapping)
            ]
        payload[field] = value
    return _sha256(payload)


def _sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_retrieval_score(candidate: Mapping[str, Any]) -> float | None:
    for field in _RETRIEVAL_SCORE_FIELDS:
        if candidate.get(field) is None:
            continue
        score = _finite_float(candidate.get(field))
        if score is not None:
            return score
    return None


def _claim_atom_texts(
    feature_row: Mapping[str, Any], atom_ids: Sequence[str]
) -> dict[str, str]:
    evidence_map = feature_row.get("evidence_map")
    nested = evidence_map.get("claim_atoms") if isinstance(evidence_map, Mapping) else None
    raw_atoms = nested or feature_row.get("claim_atoms") or []
    by_id: dict[str, str] = {}
    if isinstance(raw_atoms, (list, tuple)):
        for index, atom in enumerate(raw_atoms):
            if not isinstance(atom, Mapping):
                continue
            atom_id = _compact(atom.get("atom_id") or atom.get("node_id"))
            if not atom_id and index < len(atom_ids):
                atom_id = atom_ids[index]
            text = _compact(
                atom.get("proposition")
                or atom.get("text")
                or atom.get("claim_atom")
                or atom_id
            )
            if atom_id:
                by_id[atom_id] = text or atom_id
    return {atom_id: by_id.get(atom_id, atom_id) for atom_id in atom_ids}


def _atom_texts_from_trace(
    trace: Mapping[str, Any], atom_ids: Sequence[str], errors: list[str]
) -> dict[str, str]:
    raw = trace.get("claim_atom_texts")
    if raw is None:
        return {atom_id: atom_id for atom_id in atom_ids}
    if not isinstance(raw, Mapping) or set(raw) != set(atom_ids):
        errors.append("claim_atom_texts keys must exactly match claim_atoms")
        return {atom_id: atom_id for atom_id in atom_ids}
    return {
        atom_id: _compact(raw.get(atom_id)) or atom_id
        for atom_id in atom_ids
    }


def _map_schema_version(feature_row: Mapping[str, Any]) -> str:
    evidence_map = feature_row.get("evidence_map")
    nested = (
        evidence_map.get("schema_version") if isinstance(evidence_map, Mapping) else None
    )
    return _compact(
        feature_row.get("map_schema_version")
        or feature_row.get("evidence_map_schema_version")
        or nested
    )


def _index_by_key(
    candidate_pool: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        str(row["candidate_stable_key"]): int(row["candidate_pool_idx"])
        for row in candidate_pool
    }


def _string_sequence(
    raw: Any, *, name: str, errors: list[str]
) -> tuple[str, ...] | None:
    if not isinstance(raw, list):
        errors.append(f"{name} must be a list")
        return None
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        errors.append(f"{name} must contain non-empty strings")
        return None
    return tuple(raw)


def _replay_result(errors: list[str], derived: Mapping[str, Any]) -> dict[str, Any]:
    deduplicated_errors = list(dict.fromkeys(errors))
    result: dict[str, Any] = {
        "ok": not deduplicated_errors,
        "valid": not deduplicated_errors,
        "errors": deduplicated_errors,
        "derived": dict(derived),
    }
    result.update(derived)
    return result


def _expect(
    errors: list[str], name: str, actual: Any, expected: Any
) -> None:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if not isinstance(actual, bool) and math.isfinite(float(actual)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            return
    if actual != expected:
        errors.append(f"{name} mismatch: expected={expected!r}, got={actual!r}")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_int(value: Any, *, name: str, minimum: int) -> int:
    if not _is_int(value) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "SOLVER_VERSION",
    "CANDIDATE_PROJECTION_SCHEMA",
    "build_exact_trace",
    "replay_trace",
    "reorder_selected_trace",
]
