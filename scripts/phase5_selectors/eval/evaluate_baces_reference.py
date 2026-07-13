#!/usr/bin/env python3
"""Audit selector artifacts against the exact BACES structural reference.

The evaluator is deliberately read-only.  It joins evidence-map features,
selector traces, and (optionally) verifier-build rows by ``event_id`` and
``candidate_uid``.  Array indices are never used as candidate identities.

When a build artifact is supplied, ``build.candidates`` is interpreted as the
post-stop, pre-prompt-truncation slate.  The verifier-visible slate is its
prefix of length ``build.evidence_count``; this matches the prompt builder's
tail-pop truncation contract.  An unbounded full-pool trace is first clipped to
the verifier-facing count cap before ``U_full`` is evaluated, so the five-stage
decomposition compares feasible prefixes under one shared budget.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fact_checking.selectors.baces_exact import solve_exact, solve_fixed_set_order
from fact_checking.selectors.baces_objective import (
    compile_feature_problem,
    evaluate_display,
    padded_auc,
)


AUDIT_SCHEMA_VERSION = "baces-reference-audit-v0.1"
SUMMARY_NUMERIC_FIELDS = (
    "U_ideal",
    "U_pool",
    "U_opt",
    "U_full",
    "U_pre",
    "U_final",
    "loss_pool",
    "loss_budget",
    "loss_selector",
    "loss_stop",
    "loss_realization",
    "loss_total",
    "normalized_loss_pool",
    "normalized_loss_budget",
    "normalized_loss_selector",
    "normalized_loss_stop",
    "normalized_loss_realization",
    "normalized_total_loss",
    "normalized_controllable_loss",
    "selector_order_regret",
    "pre_order_regret",
    "final_order_regret",
    "T_opt",
    "T_full",
    "T_pre",
    "T_final",
    "AUC_opt",
    "AUC_full",
    "AUC_pre",
    "AUC_final",
    "K_core",
    "K_full_raw",
    "K_full",
    "K_pre",
    "K_final",
    "tokens_core",
    "tokens_full",
    "tokens_pre",
    "tokens_final",
)


class AlignmentError(ValueError):
    """Raised when artifacts cannot be safely aligned by stable identity."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        required=True,
        help="Evidence-map feature JSONL used to compile the BACES instances.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        action="append",
        metavar="NAME=PATH",
        help="Named selector trace JSONL. Repeat to audit multiple policies.",
    )
    parser.add_argument(
        "--build",
        help=(
            "Optional verifier build JSONL. It should correspond to the supplied "
            "policy; incompatible policies are retained with alignment errors."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--k-max",
        type=int,
        help="Override trace params.max_steps for the exact count budget.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N feature rows (intended for smoke tests).",
    )
    args = parser.parse_args()
    if args.k_max is not None and args.k_max < 0:
        parser.error("--k-max must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    feature_path = Path(args.features)
    trace_specs = _parse_trace_specs(args.trace)
    trace_rows = {
        name: _read_jsonl_index(path, artifact=f"trace:{name}")
        for name, path in trace_specs.items()
    }
    build_path = Path(args.build) if args.build else None
    build_rows = (
        _read_jsonl_index(build_path, artifact="build") if build_path else None
    )

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    feature_event_ids: set[str] = set()
    feature_count = 0
    with feature_path.open(encoding="utf-8") as feature_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(feature_handle, start=1):
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and feature_count >= args.limit:
                break
            feature_row = json.loads(line)
            event_id = _event_id(feature_row, f"features:{line_number}")
            if event_id in feature_event_ids:
                raise AlignmentError(
                    f"duplicate event_id {event_id!r} in features at line {line_number}"
                )
            feature_event_ids.add(event_id)
            feature_count += 1

            for policy_name, rows_by_event in trace_rows.items():
                trace_row = rows_by_event.get(event_id)
                build_row = build_rows.get(event_id) if build_rows is not None else None
                if trace_row is None:
                    audit = _missing_trace_row(
                        event_id=event_id,
                        policy_name=policy_name,
                        feature_row=feature_row,
                        build_supplied=build_rows is not None,
                    )
                else:
                    try:
                        audit = _audit_one(
                            feature_row=feature_row,
                            trace_row=trace_row,
                            build_row=build_row,
                            build_supplied=build_rows is not None,
                            policy_name=policy_name,
                            trace_path=trace_specs[policy_name],
                            k_max_override=args.k_max,
                        )
                    except Exception as exc:  # retain the row and expose exact failure
                        audit = _error_row(
                            event_id=event_id,
                            policy_name=policy_name,
                            feature_row=feature_row,
                            error=exc,
                            build_supplied=build_rows is not None,
                        )
                audit_rows.append(audit)
                output_handle.write(
                    json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n"
                )

    summary = _build_summary(
        rows=audit_rows,
        feature_path=feature_path,
        trace_specs=trace_specs,
        build_path=build_path,
        output_path=output_path,
        k_max_override=args.k_max,
        feature_count=feature_count,
        feature_event_ids=feature_event_ids,
        trace_rows=trace_rows,
        build_rows=build_rows,
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status_counts = Counter(str(row.get("status") or "") for row in audit_rows)
    print(
        f"Wrote {len(audit_rows)} audit rows to {output_path}; "
        f"status={dict(sorted(status_counts.items()))}"
    )
    print(f"Wrote summary to {summary_path}")
    failing_statuses = {"error", "missing_trace", "alignment_error"}
    return 1 if any(status_counts.get(status, 0) for status in failing_statuses) else 0


def _audit_one(
    *,
    feature_row: Mapping[str, Any],
    trace_row: Mapping[str, Any],
    build_row: Mapping[str, Any] | None,
    build_supplied: bool,
    policy_name: str,
    trace_path: Path,
    k_max_override: int | None,
) -> dict[str, Any]:
    event_id = _event_id(feature_row, "feature row")
    if _event_id(trace_row, "trace row") != event_id:
        raise AlignmentError("trace event_id differs from feature event_id")

    raw_full_keys = _candidate_uids(
        _as_sequence(trace_row.get("selected_candidates")),
        context=f"trace selected_candidates for {event_id}",
    )
    _require_unique(raw_full_keys, f"trace selected_candidates for {event_id}")

    trace_pool = _as_sequence(trace_row.get("candidate_pool"))
    trace_pool_keys = _candidate_uids(
        trace_pool,
        context=f"trace candidate_pool for {event_id}",
    )
    _require_unique(trace_pool_keys, f"trace candidate_pool for {event_id}")
    trace_order_errors, trace_order_warnings = _validate_trace_order_contract(
        trace_row=trace_row,
        trace_pool=trace_pool,
        selected_uids=raw_full_keys,
        event_id=event_id,
    )
    cost_overrides = _mrec_cost_overrides(trace_pool)

    params = trace_row.get("params")
    params = params if isinstance(params, Mapping) else {}
    k_max, k_max_source = _resolve_k_max(
        override=k_max_override,
        params=params,
        feature_row=feature_row,
        build_row=build_row,
        full_keys=raw_full_keys,
    )
    token_budget, token_budget_source = _resolve_token_budget(
        params=params,
        build_row=build_row,
        event_id=event_id,
    )
    unit_weights = [1] * _feature_atom_count(feature_row)
    problem = compile_feature_problem(
        feature_row,
        k_max=k_max,
        token_budget=token_budget,
        weights=unit_weights,
        cost_overrides=cost_overrides,
    )
    count_bounded_full_keys = raw_full_keys[:k_max]
    full_keys = _token_feasible_prefix(
        count_bounded_full_keys,
        problem=problem,
        token_budget=token_budget,
    )

    problem_keys = [_problem_candidate_uid(candidate) for candidate in problem.candidates]
    _require_unique(problem_keys, f"compiled candidate pool for {event_id}")
    problem_key_set = set(problem_keys)
    trace_pool_set = set(trace_pool_keys)
    trace_pool_match = trace_pool_set == problem_key_set
    trace_pool_only_uids = sorted(trace_pool_set - problem_key_set)
    feature_pool_only_uids = sorted(problem_key_set - trace_pool_set)
    full_known = set(raw_full_keys).issubset(problem_key_set)
    if not full_known:
        unknown = sorted(set(raw_full_keys) - problem_key_set)
        raise AlignmentError(f"trace selected_candidates contain unknown UIDs: {unknown}")

    weights = _problem_weights(problem)
    m = len(weights)
    U_ideal = 2 * sum(weights)
    pool_state = _pool_state(problem)
    exact_eval = solve_exact(problem)
    full_eval = evaluate_display(problem, full_keys)
    full_order_opt = solve_fixed_set_order(problem, full_keys)

    full_feasible_count = len(full_keys) <= k_max
    full_feasible_tokens = (
        token_budget is None or int(full_eval.token_cost) <= token_budget
    )
    full_feasible = full_feasible_count and full_feasible_tokens

    alignment_errors: list[str] = []
    alignment_warnings: list[str] = []
    alignment_errors.extend(trace_order_errors)
    alignment_warnings.extend(trace_order_warnings)
    if not trace_pool_match:
        alignment_errors.append("trace_pool_uid_set_differs_from_feature_pool")
    if not full_feasible_count:
        alignment_errors.append("full_slate_exceeds_k_max")
    if not full_feasible_tokens:
        alignment_errors.append("full_slate_exceeds_token_budget")

    pre_keys: list[str] | None = None
    final_keys: list[str] | None = None
    pre_eval = None
    final_eval = None
    pre_order_opt = None
    final_order_opt = None
    prompt_tail_truncated: bool | None = None
    build_status: str
    build_evidence_count: int | None = None
    build_evidence_count_before: int | None = None
    build_selected_count: int | None = None
    build_evidence_text_truncated: bool | None = None
    if not build_supplied:
        build_status = "not_provided"
    elif build_row is None:
        build_status = "missing_event"
        alignment_errors.append("build_event_missing")
    else:
        if _event_id(build_row, "build row") != event_id:
            raise AlignmentError("build event_id differs from feature event_id")
        build_selector_trace = build_row.get("selector_trace")
        build_selector_trace = (
            build_selector_trace if isinstance(build_selector_trace, Mapping) else {}
        )
        build_source_path = str(build_selector_trace.get("source_path") or "").strip()
        if build_source_path and not _same_path(trace_path, Path(build_source_path)):
            alignment_errors.append("build_source_path_differs_from_trace_path")
        pre_candidates = _as_sequence(build_row.get("candidates"))
        pre_keys = _candidate_uids(
            pre_candidates,
            context=f"build candidates for {event_id}",
        )
        _require_unique(pre_keys, f"build candidates for {event_id}")
        if not set(pre_keys).issubset(problem_key_set):
            unknown = sorted(set(pre_keys) - problem_key_set)
            raise AlignmentError(f"build candidates contain unknown UIDs: {unknown}")

        build_selected_count = _optional_nonnegative_int(
            build_row.get("prompt_evidence_selected_count_before_prompt_truncation"),
            context=f"build selected count for {event_id}",
        )
        if build_selected_count is None:
            build_selected_count = len(pre_keys)
            alignment_warnings.append("build_selected_count_missing_used_candidates_length")
        if build_selected_count != len(pre_keys):
            alignment_errors.append("build_selected_count_differs_from_candidates_length")

        build_evidence_count_before = _optional_nonnegative_int(
            build_row.get("evidence_count_before"),
            context=f"build evidence_count_before for {event_id}",
        )
        if (
            build_evidence_count_before is not None
            and build_evidence_count_before != len(pre_keys)
        ):
            alignment_errors.append("build_evidence_count_before_differs_from_candidates_length")

        build_evidence_count = _optional_nonnegative_int(
            build_row.get("evidence_count"),
            context=f"build evidence_count for {event_id}",
        )
        if build_evidence_count is None:
            raise AlignmentError("build evidence_count is required to recover final slate")
        if build_evidence_count > len(pre_keys):
            raise AlignmentError(
                "build evidence_count exceeds pre-truncation candidates length"
            )
        final_keys = pre_keys[:build_evidence_count]
        prompt_tail_truncated = len(final_keys) < len(pre_keys)
        build_evidence_text_truncated = bool(
            build_row.get("evidence_text_truncated")
        )
        if build_evidence_text_truncated:
            alignment_warnings.append(
                "final_text_truncated_coverage_not_revalidated"
            )

        if "was_truncated" in build_row:
            expected_was_truncated = (
                build_evidence_count < len(pre_keys) or build_evidence_text_truncated
            )
            if bool(build_row.get("was_truncated")) != expected_was_truncated:
                alignment_errors.append("build_was_truncated_inconsistent_with_counts")

        if full_keys[: len(pre_keys)] != pre_keys:
            alignment_errors.append("build_pre_slate_is_not_full_slate_prefix")
            build_status = "policy_mismatch"
        elif "build_source_path_differs_from_trace_path" in alignment_errors:
            build_status = "policy_mismatch"
        else:
            build_status = "aligned"

        pre_eval = evaluate_display(problem, pre_keys)
        final_eval = evaluate_display(problem, final_keys)
        pre_order_opt = solve_fixed_set_order(problem, pre_keys)
        final_order_opt = solve_fixed_set_order(problem, final_keys)

    U_pool = sum(weight * level for weight, level in zip(weights, pool_state))
    U_opt = int(exact_eval.utility)
    U_full = int(full_eval.utility)
    U_pre = int(pre_eval.utility) if pre_eval is not None else None
    U_final = int(final_eval.utility) if final_eval is not None else None

    loss_pool = U_ideal - U_pool
    loss_budget = U_pool - U_opt
    loss_selector = U_opt - U_full
    if U_pre is None or U_final is None:
        loss_stop = None
        loss_realization = None
        loss_total = U_ideal - U_full
        decomposition_sum = loss_pool + loss_budget + loss_selector
        conservation_scope = "selector_output"
    else:
        loss_stop = U_full - U_pre
        loss_realization = U_pre - U_final
        loss_total = U_ideal - U_final
        decomposition_sum = (
            loss_pool
            + loss_budget
            + loss_selector
            + loss_stop
            + loss_realization
        )
        conservation_scope = "verifier_visible"
    conservation_residual = loss_total - decomposition_sum
    conservation_ok = conservation_residual == 0

    nonnegative_terms = [loss_pool, loss_budget, loss_selector]
    if loss_stop is not None:
        nonnegative_terms.append(loss_stop)
    if loss_realization is not None:
        nonnegative_terms.append(loss_realization)
    losses_nonnegative = all(value >= 0 for value in nonnegative_terms)
    if not losses_nonnegative:
        alignment_errors.append("regret_component_is_negative")
    if not conservation_ok:
        alignment_errors.append("regret_conservation_failed")

    selector_order_regret = int(full_eval.acquisition_time) - int(
        full_order_opt.acquisition_time
    )
    pre_order_regret = (
        int(pre_eval.acquisition_time) - int(pre_order_opt.acquisition_time)
        if pre_eval is not None and pre_order_opt is not None
        else None
    )
    final_order_regret = (
        int(final_eval.acquisition_time) - int(final_order_opt.acquisition_time)
        if final_eval is not None and final_order_opt is not None
        else None
    )
    if selector_order_regret < 0 or (
        pre_order_regret is not None and pre_order_regret < 0
    ) or (final_order_regret is not None and final_order_regret < 0):
        alignment_errors.append("same_set_order_regret_is_negative")

    if alignment_errors:
        alignment_status = "error"
        status = "alignment_error"
    elif alignment_warnings or not build_supplied or build_row is None:
        alignment_status = "warning"
        status = "selector_only" if not build_supplied else "partial"
    else:
        alignment_status = "ok"
        status = "ok"

    row = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_id": event_id,
        "policy_name": policy_name,
        "status": status,
        "alignment_status": alignment_status,
        "alignment_errors": alignment_errors,
        "alignment_warnings": alignment_warnings,
        "build_status": build_status,
        "m": m,
        "n_candidates": len(problem_keys),
        "k_max": k_max,
        "k_max_source": k_max_source,
        "token_budget": token_budget,
        "token_budget_source": token_budget_source,
        "weight_policy": "unit",
        "cost_source": "trace_candidate_pool.mrec_token_cost_then_feature_num_tokens",
        "cost_override_count": len(cost_overrides),
        "trace_pool_match": trace_pool_match,
        "trace_pool_only_uids": trace_pool_only_uids,
        "feature_pool_only_uids": feature_pool_only_uids,
        "full_feasible": full_feasible,
        "full_feasible_count": full_feasible_count,
        "full_feasible_tokens": full_feasible_tokens,
        "full_sequence_scope": "budgeted_prefix_of_raw_selector_order",
        "raw_full_exceeds_budget": len(raw_full_keys) > len(full_keys),
        "raw_full_exceeds_count_budget": len(raw_full_keys) > k_max,
        "count_bounded_full_exceeds_token_budget": len(count_bounded_full_keys)
        > len(full_keys),
        "pre_is_full_prefix": (
            full_keys[: len(pre_keys)] == pre_keys
            if pre_keys is not None
            else None
        ),
        "final_reconstruction_rule": (
            "pre_slate_tail_prefix_by_evidence_count" if final_keys is not None else None
        ),
        "prompt_tail_truncated": prompt_tail_truncated,
        "build_was_truncated": (
            bool(build_row.get("was_truncated")) if build_row is not None else None
        ),
        "build_evidence_text_truncated": build_evidence_text_truncated,
        "final_identity_status": (
            "tail_prefix_exact_text_truncated_unverified"
            if build_evidence_text_truncated
            else "tail_prefix_exact"
            if final_keys is not None
            else None
        ),
        "build_selected_count": build_selected_count,
        "build_evidence_count": build_evidence_count,
        "build_evidence_count_before": build_evidence_count_before,
        "U_ideal": U_ideal,
        "U_pool": U_pool,
        "U_opt": U_opt,
        "U_full": U_full,
        "U_pre": U_pre,
        "U_final": U_final,
        "loss_pool": loss_pool,
        "loss_budget": loss_budget,
        "loss_selector": loss_selector,
        "loss_stop": loss_stop,
        "loss_realization": loss_realization,
        "loss_total": loss_total,
        "decomposition_sum": decomposition_sum,
        "conservation_scope": conservation_scope,
        "conservation_residual": conservation_residual,
        "conservation_ok": conservation_ok,
        "losses_nonnegative": losses_nonnegative,
        "normalized_loss_pool": _safe_ratio(loss_pool, U_ideal),
        "normalized_loss_budget": _safe_ratio(loss_budget, U_ideal),
        "normalized_loss_selector": _safe_ratio(loss_selector, U_ideal),
        "normalized_loss_stop": _safe_ratio(loss_stop, U_ideal)
        if loss_stop is not None
        else None,
        "normalized_loss_realization": _safe_ratio(loss_realization, U_ideal)
        if loss_realization is not None
        else None,
        "normalized_total_loss": _safe_ratio(loss_total, U_ideal),
        "normalized_controllable_loss": _safe_ratio(
            (U_pool - (U_final if U_final is not None else U_full)), U_pool
        ),
        "T_opt": int(exact_eval.acquisition_time),
        "T_full": int(full_eval.acquisition_time),
        "T_pre": int(pre_eval.acquisition_time) if pre_eval is not None else None,
        "T_final": int(final_eval.acquisition_time) if final_eval is not None else None,
        "selector_same_set_T_opt": int(full_order_opt.acquisition_time),
        "pre_same_set_T_opt": (
            int(pre_order_opt.acquisition_time) if pre_order_opt is not None else None
        ),
        "final_same_set_T_opt": (
            int(final_order_opt.acquisition_time) if final_order_opt is not None else None
        ),
        "selector_order_regret": selector_order_regret,
        "pre_order_regret": pre_order_regret,
        "final_order_regret": final_order_regret,
        "AUC_opt": int(padded_auc(exact_eval, k_max)),
        "AUC_full": int(padded_auc(full_eval, k_max)),
        "AUC_pre": int(padded_auc(pre_eval, k_max)) if pre_eval is not None else None,
        "AUC_final": (
            int(padded_auc(final_eval, k_max)) if final_eval is not None else None
        ),
        "K_core": int(exact_eval.length),
        "K_full_raw": len(raw_full_keys),
        "K_full": len(full_keys),
        "K_pre": len(pre_keys) if pre_keys is not None else None,
        "K_final": len(final_keys) if final_keys is not None else None,
        "tokens_core": int(exact_eval.token_cost),
        "tokens_full": int(full_eval.token_cost),
        "tokens_pre": int(pre_eval.token_cost) if pre_eval is not None else None,
        "tokens_final": int(final_eval.token_cost) if final_eval is not None else None,
        "exact_keys": list(exact_eval.keys),
        "full_keys_raw": raw_full_keys,
        "full_keys": full_keys,
        "pre_keys": pre_keys,
        "final_keys": final_keys,
        "selector_same_set_optimal_keys": list(full_order_opt.keys),
        "pre_same_set_optimal_keys": (
            list(pre_order_opt.keys) if pre_order_opt is not None else None
        ),
        "final_same_set_optimal_keys": (
            list(final_order_opt.keys) if final_order_opt is not None else None
        ),
        "exact_state": list(exact_eval.state),
        "pool_state": list(pool_state),
        "full_state": list(full_eval.state),
        "pre_state": list(pre_eval.state) if pre_eval is not None else None,
        "final_state": list(final_eval.state) if final_eval is not None else None,
    }
    return row


def _resolve_k_max(
    *,
    override: int | None,
    params: Mapping[str, Any],
    feature_row: Mapping[str, Any],
    build_row: Mapping[str, Any] | None,
    full_keys: Sequence[str],
) -> tuple[int, str]:
    if override is not None:
        return override, "cli"
    build_max_count = None
    if build_row is not None:
        raw_build_max = build_row.get("prompt_evidence_max_count")
        if raw_build_max is not None:
            parsed_build_max = _nonnegative_int(
                raw_build_max, "build prompt_evidence_max_count"
            )
            if parsed_build_max > 0:
                build_max_count = parsed_build_max
    raw_max_steps = params.get("max_steps")
    if raw_max_steps is not None:
        parsed_max_steps = _nonnegative_int(raw_max_steps, "trace params.max_steps")
        if parsed_max_steps > 0:
            if build_max_count is not None:
                return min(parsed_max_steps, build_max_count), "trace_and_build_min"
            return parsed_max_steps, "trace_params"
        # Current full-pool traces encode an unbounded ordering with max_steps=0;
        # the verifier-facing count cap then comes from the prompt policy.
        if build_max_count is not None:
            return build_max_count, "build_prompt_max_count"
    elif build_max_count is not None:
        return build_max_count, "build_prompt_max_count"
    raw_candidate_top_n = feature_row.get("candidate_top_n")
    if raw_candidate_top_n is not None:
        parsed_top_n = _nonnegative_int(raw_candidate_top_n, "feature candidate_top_n")
        if parsed_top_n > 0:
            return parsed_top_n, "feature_candidate_top_n"
    return len(full_keys), "full_slate_length"


def _resolve_token_budget(
    *,
    params: Mapping[str, Any],
    build_row: Mapping[str, Any] | None,
    event_id: str,
) -> tuple[int | None, str]:
    trace_budget = _optional_nonnegative_int(
        params.get("token_budget"),
        context=f"trace token_budget for {event_id}",
    )
    build_budget = (
        _optional_nonnegative_int(
            build_row.get("prompt_evidence_token_budget"),
            context=f"build prompt_evidence_token_budget for {event_id}",
        )
        if build_row is not None
        else None
    )
    if trace_budget is not None and build_budget is not None:
        return min(trace_budget, build_budget), "trace_and_build_min"
    if trace_budget is not None:
        return trace_budget, "trace_params"
    if build_budget is not None:
        return build_budget, "build_prompt_token_budget"
    return None, "none"


def _token_feasible_prefix(
    keys: Sequence[str],
    *,
    problem: Any,
    token_budget: int | None,
) -> list[str]:
    if token_budget is None:
        return list(keys)
    candidate_by_key = {candidate.key: candidate for candidate in problem.candidates}
    selected: list[str] = []
    cumulative_cost = 0
    for key in keys:
        candidate = candidate_by_key.get(key)
        if candidate is None:
            raise AlignmentError(f"selector order contains unknown candidate UID {key!r}")
        next_cost = cumulative_cost + int(candidate.cost)
        if next_cost > token_budget:
            break
        selected.append(key)
        cumulative_cost = next_cost
    return selected


def _validate_trace_order_contract(
    *,
    trace_row: Mapping[str, Any],
    trace_pool: Sequence[Mapping[str, Any]],
    selected_uids: Sequence[str],
    event_id: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ordered_indices = _trace_indices(
        trace_row.get("selector_ordered_indices"),
        field="selector_ordered_indices",
        event_id=event_id,
        errors=errors,
        warnings=warnings,
    )
    if ordered_indices is not None:
        if len(set(ordered_indices)) != len(ordered_indices):
            errors.append("selector_ordered_indices_contains_duplicates")
        if any(index >= len(trace_pool) for index in ordered_indices):
            errors.append("selector_ordered_indices_out_of_range")
        elif len(set(ordered_indices)) == len(ordered_indices):
            indexed_uids = [
                _candidate_uid(trace_pool[index], f"trace candidate_pool for {event_id}")
                for index in ordered_indices
            ]
            if list(selected_uids) != indexed_uids:
                errors.append("selected_candidates_differs_from_selector_ordered_indices")

    selected_indices = _trace_indices(
        trace_row.get("selected_indices"),
        field="selected_indices",
        event_id=event_id,
        errors=errors,
        warnings=warnings,
    )
    if selected_indices is not None:
        if len(set(selected_indices)) != len(selected_indices):
            errors.append("selected_indices_contains_duplicates")
        if any(index >= len(trace_pool) for index in selected_indices):
            errors.append("selected_indices_out_of_range")
        elif ordered_indices is not None and selected_indices != ordered_indices:
            errors.append("selected_indices_differs_from_selector_ordered_indices")
    return errors, warnings


def _trace_indices(
    value: Any,
    *,
    field: str,
    event_id: str,
    errors: list[str],
    warnings: list[str],
) -> list[int] | None:
    if value is None:
        warnings.append(f"trace_{field}_missing")
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"trace_{field}_is_not_an_array")
        return None
    parsed: list[int] = []
    for position, item in enumerate(value):
        try:
            parsed.append(
                _nonnegative_int(item, f"trace {field}[{position}] for {event_id}")
            )
        except AlignmentError:
            errors.append(f"trace_{field}_contains_non_integer")
            return None
    return parsed


def _mrec_cost_overrides(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for candidate in candidates:
        uid = _candidate_uid(candidate, "trace candidate_pool")
        raw_cost = candidate.get("mrec_token_cost")
        if raw_cost is None:
            continue
        cost = _nonnegative_int(raw_cost, f"mrec_token_cost for {uid}")
        if uid in overrides and overrides[uid] != cost:
            raise AlignmentError(f"conflicting mrec_token_cost values for UID {uid}")
        overrides[uid] = cost
    return overrides


def _problem_candidate_uid(candidate: Any) -> str:
    uid = str(getattr(candidate, "uid", "") or "").strip()
    if not uid:
        raise AlignmentError("compiled BACES candidate is missing uid")
    return uid


def _problem_weights(problem: Any) -> list[int]:
    raw_weights = problem.weights
    if isinstance(raw_weights, Mapping):
        values = list(raw_weights.values())
    else:
        values = list(raw_weights)
    weights = [_nonnegative_int(value, "problem weight") for value in values]
    if len(weights) != len(problem.atom_ids):
        raise AlignmentError("compiled problem weight/atom dimensions differ")
    return weights


def _pool_state(problem: Any) -> tuple[int, ...]:
    """Return the unconstrained candidate-pool ceiling state.

    ``evaluate_display`` correctly enforces ``k_max`` and therefore cannot be
    used to replay all candidates when the retrieval pool is larger than the
    selector budget.  The pool ceiling is just the componentwise maximum of
    every candidate quality vector and is intentionally budget-free.
    """

    state = [0] * len(problem.weights)
    for candidate in problem.candidates:
        if len(candidate.q) != len(state):
            raise AlignmentError("compiled candidate quality dimension mismatch")
        state = [max(before, quality) for before, quality in zip(state, candidate.q)]
    return tuple(state)


def _candidate_uids(
    candidates: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> list[str]:
    return [_candidate_uid(candidate, context) for candidate in candidates]


def _candidate_uid(candidate: Mapping[str, Any], context: str) -> str:
    if not isinstance(candidate, Mapping):
        raise AlignmentError(f"{context} contains a non-object candidate")
    uid = str(candidate.get("candidate_uid") or "").strip()
    if not uid:
        raise AlignmentError(f"{context} candidate is missing candidate_uid")
    return uid


def _as_sequence(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AlignmentError("candidate collection must be a JSON array")
    rows = list(value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise AlignmentError("candidate collection contains a non-object entry")
    return rows


def _require_unique(keys: Sequence[str], context: str) -> None:
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise AlignmentError(f"duplicate candidate_uid in {context}: {duplicates}")


def _event_id(row: Mapping[str, Any], context: str) -> str:
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        raise AlignmentError(f"{context} is missing event_id")
    return event_id


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise AlignmentError(f"{context} must be an integer, got boolean")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise AlignmentError(f"{context} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise AlignmentError(f"{context} must be integral, got {value!r}")
    if numeric < 0:
        raise AlignmentError(f"{context} must be non-negative, got {numeric}")
    return numeric


def _optional_nonnegative_int(value: Any, context: str) -> int | None:
    if value is None or value == "":
        return None
    return _nonnegative_int(value, context)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _missing_trace_row(
    *,
    event_id: str,
    policy_name: str,
    feature_row: Mapping[str, Any],
    build_supplied: bool,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_id": event_id,
        "policy_name": policy_name,
        "status": "missing_trace",
        "alignment_status": "error",
        "alignment_errors": ["trace_event_missing"],
        "alignment_warnings": [],
        "build_status": "not_checked" if build_supplied else "not_provided",
        "m": _feature_atom_count(feature_row),
    }


def _error_row(
    *,
    event_id: str,
    policy_name: str,
    feature_row: Mapping[str, Any],
    error: Exception,
    build_supplied: bool,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_id": event_id,
        "policy_name": policy_name,
        "status": "error",
        "alignment_status": "error",
        "alignment_errors": [str(error)],
        "alignment_warnings": [],
        "build_status": "not_checked" if build_supplied else "not_provided",
        "m": _feature_atom_count(feature_row),
        "error_type": type(error).__name__,
    }


def _feature_atom_count(feature_row: Mapping[str, Any]) -> int:
    atoms = feature_row.get("claim_atoms")
    return len(atoms) if isinstance(atoms, Sequence) and not isinstance(atoms, str) else 0


def _parse_trace_specs(values: Iterable[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--trace must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"--trace must be NAME=PATH, got {value!r}")
        if name in specs:
            raise ValueError(f"duplicate --trace name {name!r}")
        specs[name] = Path(raw_path)
    return specs


def _read_jsonl_index(path: Path, *, artifact: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AlignmentError(f"{artifact}:{line_number} is not a JSON object")
            event_id = _event_id(row, f"{artifact}:{line_number}")
            if event_id in rows:
                raise AlignmentError(
                    f"duplicate event_id {event_id!r} in {artifact} at line {line_number}"
                )
            rows[event_id] = row
    return rows


def _build_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    feature_path: Path,
    trace_specs: Mapping[str, Path],
    build_path: Path | None,
    output_path: Path,
    k_max_override: int | None,
    feature_count: int,
    feature_event_ids: set[str],
    trace_rows: Mapping[str, Mapping[str, Any]],
    build_rows: Mapping[str, Any] | None,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy_name in trace_specs:
        policy_rows = [row for row in rows if row.get("policy_name") == policy_name]
        groups = {
            "all": policy_rows,
            "m=1": [row for row in policy_rows if row.get("m") == 1],
            "m>=2": [row for row in policy_rows if _numeric_at_least(row.get("m"), 2)],
            "m>=3": [row for row in policy_rows if _numeric_at_least(row.get("m"), 3)],
            "truncated": [
                row for row in policy_rows if row.get("prompt_tail_truncated") is True
            ],
        }
        policies[policy_name] = {
            "trace_path": str(trace_specs[policy_name]),
            "trace_rows": len(trace_rows[policy_name]),
            "trace_events_not_in_evaluated_features": len(
                set(trace_rows[policy_name]) - feature_event_ids
            ),
            "strata": {name: _summarize_group(group) for name, group in groups.items()},
        }

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "inputs": {
            "features": str(feature_path),
            "traces": {name: str(path) for name, path in trace_specs.items()},
            "build": str(build_path) if build_path else None,
            "k_max_override": k_max_override,
        },
        "output_jsonl": str(output_path),
        "feature_rows_evaluated": feature_count,
        "audit_rows": len(rows),
        "build_rows": len(build_rows) if build_rows is not None else None,
        "build_events_not_in_evaluated_features": (
            len(set(build_rows) - feature_event_ids) if build_rows is not None else None
        ),
        "policies": policies,
    }


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    alignment_counts = Counter(str(row.get("alignment_status") or "") for row in rows)
    valid_rows = [
        row
        for row in rows
        if row.get("status") in {"ok", "selector_only", "partial"}
        and row.get("build_evidence_text_truncated") is not True
    ]
    rows_by_scope: dict[str, list[Mapping[str, Any]]] = {}
    for row in valid_rows:
        scope = str(row.get("conservation_scope") or "unknown")
        rows_by_scope.setdefault(scope, []).append(row)
    numeric_by_scope = {
        scope: {
            field: _numeric_summary(row.get(field) for row in scope_rows)
            for field in SUMMARY_NUMERIC_FIELDS
        }
        for scope, scope_rows in sorted(rows_by_scope.items())
    }
    numeric = (
        next(iter(numeric_by_scope.values()))
        if len(numeric_by_scope) == 1
        else {
            field: _numeric_summary(())
            for field in SUMMARY_NUMERIC_FIELDS
        }
    )
    return {
        "n": len(rows),
        "valid_metric_n": len(valid_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "alignment_status_counts": dict(sorted(alignment_counts.items())),
        "conservation_scope_counts": {
            scope: len(scope_rows) for scope, scope_rows in sorted(rows_by_scope.items())
        },
        "decomposition_complete_n": sum(
            row.get("conservation_scope") == "verifier_visible" for row in rows
        ),
        "conservation_failure_n": sum(
            row.get("conservation_ok") is False for row in rows
        ),
        "nonnegative_failure_n": sum(
            row.get("losses_nonnegative") is False for row in rows
        ),
        "truncated_n": sum(row.get("prompt_tail_truncated") is True for row in rows),
        "truncated_rate": _safe_ratio(
            sum(row.get("prompt_tail_truncated") is True for row in rows), len(rows)
        ),
        "numeric": numeric,
        "numeric_by_conservation_scope": numeric_by_scope,
    }


def _numeric_summary(values: Iterable[Any]) -> dict[str, int | float | None]:
    clean = [float(value) for value in values if _is_finite_number(value)]
    if not clean:
        return {"n": 0, "mean": None, "min": None, "max": None, "sum": None}
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "min": min(clean),
        "max": max(clean),
        "sum": sum(clean),
    }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_at_least(value: Any, threshold: int) -> bool:
    return _is_finite_number(value) and float(value) >= threshold


if __name__ == "__main__":
    raise SystemExit(main())
