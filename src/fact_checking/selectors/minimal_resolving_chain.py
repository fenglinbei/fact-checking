from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from fact_checking.selectors.mrec_learned_marginal import (
    extract_marginal_features,
    hard_state_to_soft_state,
    learned_marginal_weight_fingerprint,
    load_learned_marginal_weights,
    score_marginal_features,
    update_soft_state_from_relation,
)
from fact_checking.selectors.mrec_schema import (
    MREC_TRACE_VERSION,
    build_initial_atom_states,
    build_mrec_step,
    mrec_steps_to_compat_chain_steps,
    summarize_mrec_trace,
)


TokenCostFn = Callable[[Mapping[str, Any]], int]

MREC_SELECTOR_NAME = "mrec_greedy_transition_v0_1"
MREC_SELECTOR_NAME_V0_2_LEARNED_MARGINAL_PROXY = "mrec_greedy_transition_v0_2_learned_marginal_proxy"
MREC_SELECTION_POLICY_TRANSITION_V0_1 = "transition_v0_1"
MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY = "learned_marginal_proxy"
FALLBACK_CUE = "Verify the main factual claim."

_RESOLVING_STATES = {"S", "R", "Q", "C"}
_BACKGROUND_RELATIONS = {"background", "irrelevant"}
_NON_DIRECTNESS = {"context", "none"}
_TRANSITION_PRIORITY = {
    "OPEN": 100.0,
    "CONTRAST": 80.0,
    "CORROBORATE": 35.0,
    "BRIDGE": 10.0,
    "FALLBACK": 0.0,
}
_POST_TARGET_FILL_POLICIES = {"none", "contrast_only", "contrast_then_support"}
_SELECTION_POLICIES = {
    MREC_SELECTION_POLICY_TRANSITION_V0_1,
    MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
}


@dataclass(frozen=True)
class MRECSelectorParams:
    candidate_top_n: int = 20
    max_steps: int = 10
    min_steps: int = 0
    token_budget: int | None = None
    target_resolved_rate: float = 0.80
    continue_after_target_for_contrast: bool = False
    post_target_fill_policy: str = "contrast_only"
    allow_fallback: bool = True
    cue_policy: str = "atom_proposition"
    selector_name: str = MREC_SELECTOR_NAME
    selection_policy: str = MREC_SELECTION_POLICY_TRANSITION_V0_1
    weight_file: str = ""
    stop_threshold: float = 0.0


def build_mrec_trace_row(
    row: Mapping[str, Any],
    *,
    params: MRECSelectorParams | None = None,
    token_cost_fn: TokenCostFn | None = None,
) -> dict[str, Any]:
    params = params or MRECSelectorParams()
    claim_atoms = _claim_atoms(row)
    atom_by_id = _atom_by_id(claim_atoms)
    atom_states = build_initial_atom_states(claim_atoms)
    candidates = _candidate_pool(row, candidate_top_n=params.candidate_top_n, token_cost_fn=token_cost_fn)
    if not candidates:
        raise ValueError("MREC input trace has no candidate_pool.")

    selection = _select_mrec_steps(
        candidates,
        claim_atoms=claim_atoms,
        atom_by_id=atom_by_id,
        initial_atom_states=atom_states,
        params=params,
    )
    steps = list(selection["steps"])
    compat_steps = mrec_steps_to_compat_chain_steps(steps)
    selected_indices = [int(step.get("selector_candidate_idx", idx)) for idx, step in enumerate(steps)]
    selected_ids = [str(step.get("evidence_id") or "") for step in steps]
    selected_candidates = [
        dict(candidates[idx])
        for idx in selected_indices
        if 0 <= int(idx) < len(candidates)
    ]

    trace = {
        "event_id": str(row.get("event_id") or ""),
        "claim": str(row.get("claim") or ""),
        "gold_label": str(row.get("gold_label") or ""),
        "mrec_trace_version": MREC_TRACE_VERSION,
        "mrec_selector_name": str(params.selector_name),
        "selector_name": str(params.selector_name),
        "graph_version": MREC_TRACE_VERSION,
        "candidate_pool": candidates,
        "selected_indices": selected_indices,
        "selector_ordered_indices": selected_indices,
        "selected_candidates": selected_candidates,
        "selected_evidence_ids": selected_ids,
        "selected_keys": [str(candidate.get("candidate_key") or "") for candidate in selected_candidates],
        "claim_atoms": claim_atoms,
        "atom_states_initial": atom_states,
        "atom_states_final": dict(selection["atom_states_final"]),
        "mrec_steps": steps,
        "mrec_diagnostics": {},
        "compat_chain_steps": compat_steps,
        "chain_steps": compat_steps,
        "params": {
            "candidate_top_n": int(params.candidate_top_n),
            "max_steps": int(params.max_steps),
            "min_steps": int(params.min_steps),
            "token_budget": params.token_budget,
            "target_resolved_rate": float(params.target_resolved_rate),
            "continue_after_target_for_contrast": bool(params.continue_after_target_for_contrast),
            "post_target_fill_policy": _normalize_post_target_fill_policy(params.post_target_fill_policy),
            "allow_fallback": bool(params.allow_fallback),
            "cue_policy": str(params.cue_policy),
            "selection_policy": _normalize_selection_policy(params.selection_policy),
            "weight_file": str(params.weight_file or ""),
            "weight_fingerprint": str(selection.get("weight_fingerprint") or ""),
            "stop_threshold": float(params.stop_threshold),
        },
    }
    diagnostics = _mrec_diagnostics(
        trace,
        stop_reason=str(selection["stop_reason"]),
        rejected=dict(selection["rejected"]),
        total_candidate_count=len(candidates),
    )
    trace["mrec_diagnostics"] = diagnostics
    return trace


def _select_mrec_steps(
    candidates: Sequence[Mapping[str, Any]],
    *,
    claim_atoms: list[dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    initial_atom_states: dict[str, str],
    params: MRECSelectorParams,
) -> dict[str, Any]:
    policy = _normalize_selection_policy(params.selection_policy)
    if policy == MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY:
        return _select_learned_marginal_steps(
            candidates,
            claim_atoms=claim_atoms,
            atom_by_id=atom_by_id,
            initial_atom_states=initial_atom_states,
            params=params,
        )
    return _select_transition_v0_1_steps(
        candidates,
        claim_atoms=claim_atoms,
        atom_by_id=atom_by_id,
        initial_atom_states=initial_atom_states,
        params=params,
    )


def _select_transition_v0_1_steps(
    candidates: Sequence[Mapping[str, Any]],
    *,
    claim_atoms: list[dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    initial_atom_states: dict[str, str],
    params: MRECSelectorParams,
) -> dict[str, Any]:
    atom_states = dict(initial_atom_states)
    selected_indices: set[int] = set()
    selected_duplicate_groups: set[str] = set()
    selected_texts: set[str] = set()
    steps: list[dict[str, Any]] = []
    total_token_cost = 0
    stop_reason = ""
    min_steps = max(int(params.min_steps), 0)
    fill_policy = _normalize_post_target_fill_policy(params.post_target_fill_policy)

    while len(steps) < max(int(params.max_steps), 0):
        target_met = _resolved_rate(atom_states) >= float(params.target_resolved_rate)
        if target_met:
            fill_required = len(steps) < min_steps
            if fill_policy == "none":
                stop_reason = "target_resolution_reached"
                break
            if fill_policy != "contrast_only" and not fill_required:
                stop_reason = "min_steps_satisfied" if min_steps > 0 else "target_resolution_reached"
                break
            if not params.continue_after_target_for_contrast and not fill_required:
                stop_reason = "target_resolution_reached"
                break
            allowed_post_target_operations = _post_target_allowed_operations(fill_policy)
        else:
            allowed_post_target_operations = None

        ranked: list[dict[str, Any]] = []
        skipped_by_budget = False
        for idx, candidate in enumerate(candidates):
            if idx in selected_indices:
                continue
            if _is_duplicate_candidate(candidate, selected_duplicate_groups, selected_texts):
                continue
            evaluation = _evaluate_candidate_transition(
                candidate,
                atom_states=atom_states,
                atom_by_id=atom_by_id,
                cue_policy=str(params.cue_policy),
            )
            if not evaluation:
                continue
            if allowed_post_target_operations is not None and evaluation["operation"] not in allowed_post_target_operations:
                continue
            token_cost = int(candidate.get("mrec_token_cost") or 0)
            if params.token_budget is not None and total_token_cost + token_cost > int(params.token_budget):
                skipped_by_budget = True
                continue
            evaluation["candidate_idx"] = idx
            evaluation["token_cost"] = token_cost
            ranked.append(evaluation)

        if not ranked:
            if skipped_by_budget:
                stop_reason = "token_budget_exhausted"
            elif target_met:
                stop_reason = "no_post_target_transition" if len(steps) < min_steps and fill_policy != "contrast_only" else "target_resolution_reached"
            elif steps:
                stop_reason = "no_valid_transition"
            else:
                stop_reason = "fallback_only" if params.allow_fallback else "no_valid_transition"
                if params.allow_fallback:
                    fallback = _fallback_step(candidates, claim_atoms=claim_atoms, token_budget=params.token_budget)
                    if fallback is not None:
                        steps.append(fallback["step"])
                        selected_indices.add(int(fallback["candidate_idx"]))
                        total_token_cost += int(fallback["token_cost"])
            break

        ranked.sort(key=_transition_sort_key)
        pick = ranked[0]
        idx = int(pick["candidate_idx"])
        candidate = candidates[idx]
        step_candidate = pick.get("step_candidate")
        if not isinstance(step_candidate, Mapping):
            step_candidate = candidate
        step = build_mrec_step(
            step=len(steps) + 1,
            candidate=step_candidate,
            atom_id=str(pick["atom_id"]),
            atom_text=str(pick["atom_text"]),
            state_before=str(pick["state_before"]),
            state_after=str(pick["state_after"]),
            operation=str(pick["operation"]),
            cue_text=str(pick["cue_text"]),
            cue_source=str(pick["cue_source"]),
            transition_reason=str(pick["transition_reason"]),
            token_cost=int(pick["token_cost"]),
        )
        step["post_target_fill"] = bool(target_met)
        steps.append(step)
        selected_indices.add(idx)
        total_token_cost += int(pick["token_cost"])
        if step.get("atom_id"):
            atom_states[str(step["atom_id"])] = str(step["state_after"])
        duplicate_group = _compact(candidate.get("duplicate_group") or "")
        if duplicate_group:
            selected_duplicate_groups.add(duplicate_group)
        selected_texts.add(_normalize_text(candidate.get("text") or candidate.get("evidence_text") or ""))

    if not stop_reason:
        stop_reason = "reached_max_steps" if len(steps) >= int(params.max_steps) else "no_valid_transition"

    rejected = _rejection_counts(candidates, selected_indices=selected_indices, selected_steps=steps)
    return {
        "steps": steps,
        "atom_states_final": atom_states,
        "stop_reason": stop_reason,
        "rejected": rejected,
    }


def _select_learned_marginal_steps(
    candidates: Sequence[Mapping[str, Any]],
    *,
    claim_atoms: list[dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    initial_atom_states: dict[str, str],
    params: MRECSelectorParams,
) -> dict[str, Any]:
    weights = load_learned_marginal_weights(params.weight_file, allow_default=True)
    weight_fingerprint = learned_marginal_weight_fingerprint(weights)
    atom_states = dict(initial_atom_states)
    soft_state = hard_state_to_soft_state(atom_states)
    selected_indices: set[int] = set()
    selected_duplicate_groups: set[str] = set()
    selected_texts: set[str] = set()
    steps: list[dict[str, Any]] = []
    total_token_cost = 0
    stop_reason = ""
    min_steps = max(int(params.min_steps), 0)
    max_steps = max(int(params.max_steps), 0)
    pool_max_token_cost = max([int(candidate.get("mrec_token_cost") or 0) for candidate in candidates] or [1])

    while len(steps) < max_steps:
        target_met = _resolved_rate(atom_states) >= float(params.target_resolved_rate)
        ranked: list[dict[str, Any]] = []
        skipped_by_budget = False
        for idx, candidate in enumerate(candidates):
            if idx in selected_indices:
                continue
            if _is_duplicate_candidate(candidate, selected_duplicate_groups, selected_texts):
                continue
            token_cost = int(candidate.get("mrec_token_cost") or 0)
            if params.token_budget is not None and total_token_cost + token_cost > int(params.token_budget):
                skipped_by_budget = True
                continue
            features = extract_marginal_features(
                candidate,
                selected_steps=steps,
                soft_state=soft_state,
                token_budget=params.token_budget,
                pool_max_token_cost=pool_max_token_cost,
            )
            score = score_marginal_features(features, weights)
            evaluation = _evaluate_candidate_transition(
                candidate,
                atom_states=atom_states,
                atom_by_id=atom_by_id,
                cue_policy=str(params.cue_policy),
            )
            if not evaluation:
                evaluation = _fallback_candidate_evaluation(
                    candidate,
                    claim_atoms=claim_atoms,
                    atom_states=atom_states,
                    cue_policy=str(params.cue_policy),
                )
            evaluation["candidate_idx"] = idx
            evaluation["token_cost"] = token_cost
            evaluation["utility_score"] = float(score)
            evaluation["utility_features"] = features
            ranked.append(evaluation)

        if not ranked:
            if skipped_by_budget:
                stop_reason = "token_budget_exhausted"
            elif target_met and len(steps) >= min_steps:
                stop_reason = "min_steps_satisfied" if min_steps > 0 else "target_resolution_reached"
            elif steps:
                stop_reason = "no_valid_transition"
            else:
                stop_reason = "fallback_only" if params.allow_fallback else "no_valid_transition"
                if params.allow_fallback:
                    fallback = _fallback_step(candidates, claim_atoms=claim_atoms, token_budget=params.token_budget)
                    if fallback is not None:
                        steps.append(fallback["step"])
                        selected_indices.add(int(fallback["candidate_idx"]))
                        total_token_cost += int(fallback["token_cost"])
            break

        ranked.sort(key=_learned_marginal_sort_key)
        pick = ranked[0]
        if len(steps) >= min_steps and float(pick.get("utility_score") or 0.0) <= float(params.stop_threshold):
            stop_reason = "utility_below_threshold"
            break

        idx = int(pick["candidate_idx"])
        candidate = candidates[idx]
        step_candidate = pick.get("step_candidate")
        if not isinstance(step_candidate, Mapping):
            step_candidate = candidate
        step = build_mrec_step(
            step=len(steps) + 1,
            candidate=step_candidate,
            atom_id=str(pick["atom_id"]),
            atom_text=str(pick["atom_text"]),
            state_before=str(pick["state_before"]),
            state_after=str(pick["state_after"]),
            operation=str(pick["operation"]),
            cue_text=str(pick["cue_text"]),
            cue_source=str(pick["cue_source"]),
            transition_reason=str(pick["transition_reason"]),
            token_cost=int(pick["token_cost"]),
        )
        step["post_target_fill"] = bool(target_met)
        step["utility_score"] = float(pick.get("utility_score") or 0.0)
        step["utility_features"] = dict(pick.get("utility_features") or {})
        step["selection_policy"] = MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY
        _copy_step_metadata(step, candidate)
        steps.append(step)
        selected_indices.add(idx)
        total_token_cost += int(pick["token_cost"])
        if step.get("atom_id"):
            atom_states[str(step["atom_id"])] = str(step["state_after"])
            soft_state = update_soft_state_from_relation(
                soft_state,
                atom_id=str(step["atom_id"]),
                relation=str(step.get("relation") or ""),
            )
        duplicate_group = _compact(candidate.get("duplicate_group") or "")
        if duplicate_group:
            selected_duplicate_groups.add(duplicate_group)
        selected_texts.add(_normalize_text(candidate.get("text") or candidate.get("evidence_text") or ""))

    if not stop_reason:
        stop_reason = "reached_max_steps" if len(steps) >= max_steps else "no_valid_transition"

    rejected = _rejection_counts(candidates, selected_indices=selected_indices, selected_steps=steps)
    return {
        "steps": steps,
        "atom_states_final": atom_states,
        "stop_reason": stop_reason,
        "rejected": rejected,
        "weight_fingerprint": weight_fingerprint,
    }


def _evaluate_candidate_transition(
    candidate: Mapping[str, Any],
    *,
    atom_states: dict[str, str],
    atom_by_id: dict[str, dict[str, Any]],
    cue_policy: str,
) -> dict[str, Any] | None:
    atom_pairs = _candidate_atom_transition_pairs(candidate, atom_states=atom_states)
    if not atom_pairs:
        return None

    rows = []
    for pair in atom_pairs:
        atom_id = str(pair.get("atom_id") or "")
        relation = str(pair.get("relation") or "")
        directness = str(pair.get("directness") or "").lower()
        relation_state = _state_for_relation(relation)
        before = atom_states.get(atom_id, "U")
        operation = _operation_for_transition(before, relation_state, directness)
        if operation == "FALLBACK":
            continue
        after = _state_after(before, relation_state, operation)
        atom = atom_by_id.get(atom_id, {})
        cue = _choose_cue(candidate, atom=atom, cue_policy=cue_policy)
        step_candidate = dict(candidate)
        step_candidate["covered_atom_ids"] = [atom_id]
        step_candidate["map_relation"] = relation
        step_candidate["map_directness"] = directness
        step_candidate["map_confidence"] = pair.get("confidence", candidate.get("map_confidence"))
        rows.append(
            {
                "operation": operation,
                "atom_id": atom_id,
                "atom_text": _compact(atom.get("proposition") or atom.get("text") or ""),
                "state_before": before,
                "state_after": after,
                "cue_text": cue["cue_text"],
                "cue_source": cue["cue_source"],
                "transition_reason": _transition_reason(operation, before, after, atom_id),
                "utility": _transition_utility(operation, atom, candidate),
                "relation_priority": _relation_priority(relation),
                "base_score": _float_or_default(candidate.get("base_score"), 0.0),
                "map_quality": _float_or_default(candidate.get("evidence_map_quality_score"), 0.0),
                "step_candidate": step_candidate,
            }
        )
    if not rows:
        return None
    return sorted(rows, key=_transition_row_sort_key)[0]


def _fallback_step(
    candidates: Sequence[Mapping[str, Any]],
    *,
    claim_atoms: list[dict[str, Any]],
    token_budget: int | None,
) -> dict[str, Any] | None:
    for idx, candidate in enumerate(candidates):
        token_cost = int(candidate.get("mrec_token_cost") or 0)
        if token_budget is not None and token_cost > int(token_budget):
            continue
        atom = _fallback_atom(candidate, claim_atoms=claim_atoms)
        atom_id = _compact(atom.get("atom_id") or "")
        atom_text = _atom_proposition_text(atom)
        step_candidate = dict(candidate)
        if atom_id and not step_candidate.get("covered_atom_ids"):
            step_candidate["covered_atom_ids"] = [atom_id]
        step = build_mrec_step(
            step=1,
            candidate=step_candidate,
            atom_id=atom_id,
            atom_text=atom_text,
            state_before="U",
            state_after="U",
            operation="FALLBACK",
            cue_text=atom_text or FALLBACK_CUE,
            cue_source="claim_atom" if atom_text else "fallback",
            transition_reason="no resolving atom transition was available",
            token_cost=token_cost,
        )
        return {"step": step, "candidate_idx": idx, "token_cost": token_cost}
    return None


def _fallback_atom(candidate: Mapping[str, Any], *, claim_atoms: list[dict[str, Any]]) -> dict[str, Any]:
    atom_by_id = {_compact(atom.get("atom_id") or ""): atom for atom in claim_atoms if _compact(atom.get("atom_id") or "")}
    for atom_id in candidate.get("covered_atom_ids") or []:
        atom = atom_by_id.get(_compact(atom_id))
        if atom is not None:
            return atom
    return claim_atoms[0] if claim_atoms else {}


def _fallback_candidate_evaluation(
    candidate: Mapping[str, Any],
    *,
    claim_atoms: list[dict[str, Any]],
    atom_states: dict[str, str],
    cue_policy: str,
) -> dict[str, Any]:
    atom = _fallback_atom(candidate, claim_atoms=claim_atoms)
    atom_id = _compact(atom.get("atom_id") or "")
    before = atom_states.get(atom_id, "U") if atom_id else "U"
    atom_text = _atom_proposition_text(atom)
    cue = _choose_cue(candidate, atom=atom, cue_policy=cue_policy)
    step_candidate = dict(candidate)
    if atom_id and not step_candidate.get("covered_atom_ids"):
        step_candidate["covered_atom_ids"] = [atom_id]
    step_candidate.setdefault("map_relation", _compact(candidate.get("map_relation") or candidate.get("relation") or "unknown"))
    step_candidate.setdefault("map_directness", _compact(candidate.get("map_directness") or candidate.get("directness") or "unknown"))
    return {
        "operation": "FALLBACK",
        "atom_id": atom_id,
        "atom_text": atom_text,
        "state_before": before,
        "state_after": before,
        "cue_text": cue["cue_text"],
        "cue_source": cue["cue_source"],
        "transition_reason": "selected by learned marginal utility without resolving transition",
        "utility": 0.0,
        "relation_priority": _relation_priority(str(step_candidate.get("map_relation") or "")),
        "base_score": _float_or_default(candidate.get("base_score"), 0.0),
        "map_quality": _float_or_default(candidate.get("evidence_map_quality_score"), 0.0),
        "step_candidate": step_candidate,
    }


def _copy_step_metadata(step: dict[str, Any], candidate: Mapping[str, Any]) -> None:
    for key in (
        "candidate_key",
        "candidate_uid",
        "duplicate_group",
        "source_group",
        "source_report",
        "report_id",
        "source_id",
        "hybrid_score",
        "baseline_hybrid_score",
        "base_score",
    ):
        if key in candidate:
            step[key] = candidate[key]


def _mrec_diagnostics(
    trace: Mapping[str, Any],
    *,
    stop_reason: str,
    rejected: dict[str, int],
    total_candidate_count: int,
) -> dict[str, Any]:
    summary = summarize_mrec_trace(trace)
    final_states = {str(key): str(value) for key, value in (trace.get("atom_states_final") or {}).items()}
    unresolved = sorted(atom_id for atom_id, state in final_states.items() if state == "U")
    conflicted = sorted(atom_id for atom_id, state in final_states.items() if state == "C")
    resolved_count = sum(1 for state in final_states.values() if state in _RESOLVING_STATES)
    total_atoms = max(len(final_states), 1)
    params = trace.get("params") or {}
    post_target_steps = [
        step for step in trace.get("mrec_steps") or []
        if isinstance(step, Mapping) and bool(step.get("post_target_fill"))
    ]
    post_target_operation_counts = Counter(str(step.get("operation") or "") for step in post_target_steps)
    summary.update(
        {
            "stop_reason": stop_reason,
            "total_candidate_count": int(total_candidate_count),
            "resolved_atom_rate": float(resolved_count / total_atoms),
            "unresolved_atom_ids": unresolved,
            "conflicted_atom_ids": conflicted,
            "duplicate_rejected_count": int(rejected.get("duplicate", 0)),
            "background_rejected_count": int(rejected.get("background", 0)),
            "no_transition_rejected_count": int(rejected.get("no_transition", 0)),
            "min_steps": int(params.get("min_steps") or 0),
            "post_target_fill_policy": _normalize_post_target_fill_policy(
                params.get("post_target_fill_policy") or "contrast_only"
            ),
            "selection_policy": _normalize_selection_policy(params.get("selection_policy") or MREC_SELECTION_POLICY_TRANSITION_V0_1),
            "weight_fingerprint": str(params.get("weight_fingerprint") or ""),
            "learned_marginal_score_summary": _utility_score_summary(trace.get("mrec_steps") or []),
            "post_target_fill_step_count": len(post_target_steps),
            "post_target_operation_counts": {
                key: value for key, value in dict(post_target_operation_counts).items() if key
            },
        }
    )
    return summary


def _candidate_pool(
    row: Mapping[str, Any],
    *,
    candidate_top_n: int,
    token_cost_fn: TokenCostFn | None,
) -> list[dict[str, Any]]:
    raw_candidates = (
        row.get("candidate_pool")
        or row.get("selected_candidates")
        or row.get("candidates")
        or []
    )
    out: list[dict[str, Any]] = []
    limit = int(candidate_top_n)
    for idx, raw in enumerate(raw_candidates):
        if limit > 0 and len(out) >= limit:
            break
        if not isinstance(raw, Mapping):
            continue
        candidate = dict(raw)
        candidate["selector_candidate_idx"] = idx
        candidate.setdefault("candidate_idx", idx)
        candidate.setdefault("evidence_id", str(candidate.get("candidate_uid") or f"E{idx + 1:02d}"))
        candidate["mrec_token_cost"] = _token_cost(candidate, token_cost_fn=token_cost_fn)
        out.append(candidate)
    return out


def _claim_atoms(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_atoms = (row.get("evidence_map") or {}).get("claim_atoms") or row.get("claim_atoms") or []
    atoms: list[dict[str, Any]] = []
    for idx, raw_atom in enumerate(raw_atoms, start=1):
        if not isinstance(raw_atom, Mapping):
            continue
        atom = dict(raw_atom)
        atom.setdefault("atom_id", str(atom.get("node_id") or f"A{idx}"))
        if not _compact(atom.get("text") or "") and _compact(atom.get("proposition") or ""):
            atom["text"] = _compact(atom.get("proposition") or "")
        atoms.append(atom)
    if not atoms:
        atoms.append({"atom_id": "A1", "text": "Full claim", "importance": 1.0})
    return atoms


def _atom_by_id(claim_atoms: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, atom in enumerate(claim_atoms, start=1):
        atom_id = _compact(atom.get("atom_id") or atom.get("node_id") or f"A{idx}")
        if atom_id:
            out[atom_id] = dict(atom)
    return out


def _operation_for_transition(state_before: str, relation_state: str, directness: str) -> str:
    if directness in _NON_DIRECTNESS and relation_state == "U":
        return "BRIDGE"
    if state_before == "U":
        return "OPEN" if relation_state in {"S", "R", "Q"} else "FALLBACK"
    if relation_state in {"S", "R"} and state_before in {"S", "R"} and relation_state != state_before:
        return "CONTRAST"
    if relation_state == "Q" and state_before in {"S", "R", "Q", "C"}:
        return "CONTRAST"
    if relation_state == state_before and state_before in {"S", "R", "Q"}:
        return "CORROBORATE"
    return "FALLBACK"


def _state_after(state_before: str, relation_state: str, operation: str) -> str:
    if operation in {"BRIDGE", "FALLBACK"}:
        return state_before
    if operation == "CONTRAST":
        return "Q" if relation_state == "Q" else "C"
    return relation_state if relation_state in {"S", "R", "Q"} else state_before


def _state_for_relation(value: Any) -> str:
    relation = _compact(value).lower()
    if relation in {"support", "supports", "supported_by", "entails", "consistent"}:
        return "S"
    if relation in {"refute", "refutes", "contradict", "contradicts", "counter", "conflict"}:
        return "R"
    if relation in {"qualify", "qualifies", "qualified", "condition", "hedge", "mixed", "partially_supports", "partial"}:
        return "Q"
    return "U"


def _candidate_atom_transition_pairs(
    candidate: Mapping[str, Any],
    *,
    atom_states: Mapping[str, str],
) -> list[dict[str, Any]]:
    pair_rows = candidate.get("candidate_atom_alignments") or []
    if isinstance(pair_rows, (list, tuple)):
        candidate_eid = _compact(candidate.get("evidence_id") or candidate.get("candidate_uid") or "")
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw in pair_rows:
            if not isinstance(raw, Mapping):
                continue
            evidence_id = _compact(raw.get("evidence_id") or "")
            if evidence_id and candidate_eid and evidence_id != candidate_eid:
                continue
            atom_id = _compact(raw.get("atom_id") or "")
            if atom_id not in atom_states:
                continue
            relation = _compact(raw.get("relation") or "irrelevant").lower()
            directness = _compact(raw.get("directness") or "none").lower()
            key = (atom_id, relation, directness)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "atom_id": atom_id,
                    "relation": relation,
                    "directness": directness,
                    "confidence": raw.get("confidence"),
                }
            )
        if out:
            return out

    covered_atoms = [atom_id for atom_id in _string_list(candidate.get("covered_atom_ids")) if atom_id in atom_states]
    relation = _compact(candidate.get("map_relation") or candidate.get("relation") or "").lower()
    directness = _compact(candidate.get("map_directness") or candidate.get("directness") or "").lower()
    return [
        {
            "atom_id": atom_id,
            "relation": relation,
            "directness": directness,
            "confidence": candidate.get("map_confidence"),
        }
        for atom_id in covered_atoms
    ]


def _choose_cue(candidate: Mapping[str, Any], *, atom: Mapping[str, Any], cue_policy: str) -> dict[str, str]:
    policy = _compact(cue_policy or "legacy_route_prefer").lower()
    if policy == "atom_proposition":
        atom_text = _atom_proposition_text(atom)
        if atom_text:
            return {"cue_text": atom_text, "cue_source": "claim_atom"}
        return {"cue_text": FALLBACK_CUE, "cue_source": "fallback"}
    if policy == "atom_query":
        atom_query = _best_atom_query(candidate, atom_id=_compact(atom.get("atom_id") or ""))
        if atom_query:
            return {"cue_text": atom_query, "cue_source": "atom_query"}
        atom_text = _atom_proposition_text(atom)
        if atom_text:
            return {"cue_text": atom_text, "cue_source": "claim_atom"}
        return {"cue_text": FALLBACK_CUE, "cue_source": "fallback"}

    for route in candidate.get("qd_question_routes") or candidate.get("question_routes") or []:
        if not isinstance(route, Mapping):
            continue
        question = _compact(route.get("question") or "")
        if question and question.lower() not in {"is this claim true?", "is the claim true?"}:
            return {"cue_text": question, "cue_source": "qd_question"}
    atom_query = _best_atom_query(candidate, atom_id=_compact(atom.get("atom_id") or ""))
    if atom_query:
        return {"cue_text": atom_query, "cue_source": "atom_query"}
    atom_text = _atom_proposition_text(atom)
    if atom_text:
        return {"cue_text": atom_text, "cue_source": "claim_atom"}
    return {"cue_text": FALLBACK_CUE, "cue_source": "fallback"}


def _atom_proposition_text(atom: Mapping[str, Any]) -> str:
    return _compact(atom.get("proposition") or atom.get("text") or "")


def _best_atom_query(candidate: Mapping[str, Any], *, atom_id: str) -> str:
    routes = []
    for route in candidate.get("atom_routes") or []:
        if not isinstance(route, Mapping):
            continue
        if atom_id and _compact(route.get("atom_id") or "") != atom_id:
            continue
        query = _compact(route.get("query_rendering") or "")
        if not query:
            continue
        routes.append((int(route.get("rank") or 10**9), query))
    if not routes:
        return ""
    routes.sort(key=lambda row: row[0])
    return routes[0][1]


def _transition_reason(operation: str, before: str, after: str, atom_id: str) -> str:
    if operation == "OPEN":
        return f"{atom_id} changes from unresolved to {after}"
    if operation == "CONTRAST":
        return f"{atom_id} changes from {before} to {after} after conflicting or qualifying evidence"
    if operation == "CORROBORATE":
        return f"{atom_id} receives independent corroborating evidence"
    if operation == "BRIDGE":
        return f"{atom_id} receives context evidence"
    return "no resolving state transition"


def _transition_utility(operation: str, atom: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    importance = _float_or_default(atom.get("importance"), 1.0)
    map_quality = _float_or_default(candidate.get("evidence_map_quality_score"), 0.0)
    base_score = _float_or_default(candidate.get("base_score"), 0.0)
    return _TRANSITION_PRIORITY.get(operation, 0.0) + importance * 10.0 + map_quality + base_score * 0.1


def _transition_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -_float_or_default(row.get("utility"), 0.0),
        -_float_or_default(row.get("relation_priority"), 0.0),
        -_float_or_default(row.get("map_quality"), 0.0),
        -_float_or_default(row.get("base_score"), 0.0),
        str(row.get("atom_id") or ""),
    )


def _relation_priority(relation: str) -> float:
    return {
        "refute": 4.0,
        "qualify": 3.0,
        "mixed": 3.0,
        "support": 2.0,
        "insufficient": 1.0,
    }.get(str(relation), 0.0)


def _transition_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -_float_or_default(row.get("utility"), 0.0),
        int(row.get("token_cost", 0)),
        str((row.get("candidate_idx"))),
    )


def _learned_marginal_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    features = row.get("utility_features") or {}
    if not isinstance(features, Mapping):
        features = {}
    return (
        -_float_or_default(row.get("utility_score"), 0.0),
        -_float_or_default(features.get("resolution_delta"), 0.0),
        -_float_or_default(features.get("entropy_reduction"), 0.0),
        -_float_or_default(features.get("stance_tension"), 0.0),
        -_float_or_default(features.get("new_atom_coverage"), 0.0),
        -_float_or_default(features.get("new_relation_for_atom"), 0.0),
        -_float_or_default(features.get("map_confidence"), 0.0),
        int(row.get("token_cost", 0)),
        int(row.get("candidate_idx", 0)),
    )


def _utility_score_summary(steps: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = [
        _float_or_default(step.get("utility_score"), 0.0)
        for step in steps
        if isinstance(step, Mapping) and step.get("utility_score") is not None
    ]
    if not values:
        return {"count": 0.0}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _normalize_selection_policy(value: Any) -> str:
    policy = _compact(value or MREC_SELECTION_POLICY_TRANSITION_V0_1).lower()
    if policy not in _SELECTION_POLICIES:
        raise ValueError(f"unsupported selection_policy: {value!r}")
    return policy


def _normalize_post_target_fill_policy(value: Any) -> str:
    policy = _compact(value or "contrast_only").lower()
    if policy not in _POST_TARGET_FILL_POLICIES:
        raise ValueError(f"unsupported post_target_fill_policy: {value!r}")
    return policy


def _post_target_allowed_operations(policy: str) -> set[str]:
    normalized = _normalize_post_target_fill_policy(policy)
    if normalized == "contrast_only":
        return {"CONTRAST"}
    if normalized == "contrast_then_support":
        return {"CONTRAST", "CORROBORATE", "BRIDGE"}
    return set()


def _resolved_rate(atom_states: Mapping[str, str]) -> float:
    if not atom_states:
        return 0.0
    resolved = sum(1 for state in atom_states.values() if state in _RESOLVING_STATES)
    return float(resolved / len(atom_states))


def _is_duplicate_candidate(candidate: Mapping[str, Any], selected_duplicate_groups: set[str], selected_texts: set[str]) -> bool:
    duplicate_group = _compact(candidate.get("duplicate_group") or "")
    if duplicate_group and duplicate_group in selected_duplicate_groups:
        return True
    text = _normalize_text(candidate.get("text") or candidate.get("evidence_text") or "")
    return bool(text and text in selected_texts)


def _rejection_counts(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_indices: set[int],
    selected_steps: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    selected_duplicate_groups = {
        _compact(candidates[int(step.get("selector_candidate_idx", -1))].get("duplicate_group") or "")
        for step in selected_steps
        if 0 <= int(step.get("selector_candidate_idx", -1)) < len(candidates)
    }
    selected_texts = {
        _normalize_text(candidates[int(step.get("selector_candidate_idx", -1))].get("text") or candidates[int(step.get("selector_candidate_idx", -1))].get("evidence_text") or "")
        for step in selected_steps
        if 0 <= int(step.get("selector_candidate_idx", -1)) < len(candidates)
    }
    counts: Counter[str] = Counter()
    for idx, candidate in enumerate(candidates):
        if idx in selected_indices:
            continue
        if _is_duplicate_candidate(candidate, selected_duplicate_groups, selected_texts):
            counts["duplicate"] += 1
            continue
        if _is_background_candidate(candidate):
            counts["background"] += 1
            continue
        counts["no_transition"] += 1
    return dict(counts)


def _is_background_candidate(candidate: Mapping[str, Any]) -> bool:
    relation = _compact(candidate.get("map_relation") or candidate.get("relation") or "").lower()
    directness = _compact(candidate.get("map_directness") or candidate.get("directness") or "").lower()
    return (relation in _BACKGROUND_RELATIONS or directness in _NON_DIRECTNESS) and not _string_list(candidate.get("covered_atom_ids"))


def _token_cost(candidate: Mapping[str, Any], *, token_cost_fn: TokenCostFn | None) -> int:
    if token_cost_fn is not None:
        return max(0, int(token_cost_fn(candidate)))
    for key in ("token_cost", "mrec_token_cost", "prompt_token_count", "evidence_token_count"):
        if candidate.get(key) is not None:
            return max(0, _int_or_default(candidate.get(key), 0))
    text = str(candidate.get("text") or candidate.get("evidence_text") or "")
    return max(1, len(text.split())) if text.strip() else 0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return []


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_text(value: Any) -> str:
    return _compact(value).lower()


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
