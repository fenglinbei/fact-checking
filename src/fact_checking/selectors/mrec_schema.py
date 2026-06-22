from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypedDict


MREC_TRACE_VERSION = "mrec_trace_v0_1"

AtomState = Literal["U", "S", "R", "Q", "C"]
MRECOperation = Literal["OPEN", "CORROBORATE", "CONTRAST", "BRIDGE", "FALLBACK"]

VALID_ATOM_STATES = {"U", "S", "R", "Q", "C"}
VALID_OPERATIONS = {"OPEN", "CORROBORATE", "CONTRAST", "BRIDGE", "FALLBACK"}
VALID_CUE_SOURCES = {"qd_question", "claim_atom", "atom_query", "fallback"}


class MRECStep(TypedDict):
    step: int
    operation: str
    atom_id: str
    atom_text: str
    state_before: str
    state_after: str
    cue_text: str
    cue_source: str
    evidence_id: str
    candidate_idx: int
    selector_candidate_idx: int
    evidence_text: str
    covered_atom_ids: list[str]
    relation: str
    directness: str
    map_confidence: float | None
    evidence_map_quality_score: float | None
    token_cost: int | None
    transition_reason: str


@dataclass(frozen=True)
class MRECTraceKeys:
    version: str = "mrec_trace_version"
    selector_name: str = "mrec_selector_name"
    steps: str = "mrec_steps"
    diagnostics: str = "mrec_diagnostics"
    compat_steps: str = "compat_chain_steps"


def normalize_atom_state(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "u": "U",
        "unresolved": "U",
        "s": "S",
        "support": "S",
        "supported": "S",
        "r": "R",
        "refute": "R",
        "refuted": "R",
        "q": "Q",
        "qualify": "Q",
        "qualified": "Q",
        "partial": "Q",
        "partially_resolved": "Q",
        "partially resolved": "Q",
        "c": "C",
        "conflict": "C",
        "conflicted": "C",
    }
    if text in aliases:
        return aliases[text]
    upper = str(value or "").strip().upper()
    if upper in VALID_ATOM_STATES:
        return upper
    raise ValueError(f"invalid atom state: {value!r}")


def build_initial_atom_states(claim_atoms: list[Mapping[str, Any]]) -> dict[str, str]:
    states: dict[str, str] = {}
    for idx, atom in enumerate(claim_atoms, start=1):
        atom_id = _compact(atom.get("atom_id") or atom.get("node_id") or f"A{idx}")
        if atom_id:
            states[atom_id] = "U"
    return states


def build_mrec_step(
    *,
    step: int,
    candidate: Mapping[str, Any],
    atom_id: str = "",
    atom_text: str = "",
    state_before: str = "U",
    state_after: str | None = None,
    operation: str | None = None,
    cue_text: str = "",
    cue_source: str = "claim_atom",
    transition_reason: str = "",
    token_cost: int | None = None,
) -> MRECStep:
    before = normalize_atom_state(state_before)
    relation = _normalize_relation(candidate.get("map_relation") or candidate.get("relation") or "")
    directness = _compact(candidate.get("map_directness") or candidate.get("directness") or "unknown") or "unknown"
    covered_atom_ids = _string_list(candidate.get("covered_atom_ids"))
    resolved_atom_id = _compact(atom_id or (covered_atom_ids[0] if covered_atom_ids else ""))
    op = _normalize_operation(operation) if operation else _infer_operation(before, relation, directness, resolved_atom_id)
    after = normalize_atom_state(state_after) if state_after is not None else _infer_state_after(before, relation, op)
    source = _normalize_cue_source(cue_source)
    evidence_text = str(candidate.get("text") or candidate.get("evidence_text") or "").strip()

    return {
        "step": int(step),
        "operation": op,
        "atom_id": resolved_atom_id,
        "atom_text": _compact(atom_text),
        "state_before": before,
        "state_after": after,
        "cue_text": _compact(cue_text),
        "cue_source": source,
        "evidence_id": _compact(candidate.get("evidence_id") or candidate.get("candidate_uid") or ""),
        "candidate_idx": _int_or_default(candidate.get("candidate_idx"), int(step) - 1),
        "selector_candidate_idx": _int_or_default(candidate.get("selector_candidate_idx"), _int_or_default(candidate.get("candidate_idx"), int(step) - 1)),
        "evidence_text": evidence_text,
        "covered_atom_ids": covered_atom_ids,
        "relation": relation,
        "directness": directness,
        "map_confidence": _float_or_none(candidate.get("map_confidence")),
        "evidence_map_quality_score": _float_or_none(candidate.get("evidence_map_quality_score")),
        "token_cost": None if token_cost is None else int(token_cost),
        "transition_reason": _compact(transition_reason),
    }


def mrec_steps_to_compat_chain_steps(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compat: list[dict[str, Any]] = []
    for raw_step in steps:
        operation = str(raw_step.get("operation") or "").lower()
        compat.append(
            {
                "step": int(raw_step.get("step", len(compat) + 1)),
                "atom_id": _compact(raw_step.get("atom_id") or ""),
                "atom_text": _compact(raw_step.get("atom_text") or ""),
                "cue_text": _compact(raw_step.get("cue_text") or ""),
                "cue_source": _compact(raw_step.get("cue_source") or "fallback") or "fallback",
                "candidate_idx": _int_or_default(raw_step.get("candidate_idx"), len(compat)),
                "selector_candidate_idx": _int_or_default(raw_step.get("selector_candidate_idx"), _int_or_default(raw_step.get("candidate_idx"), len(compat))),
                "evidence_id": _compact(raw_step.get("evidence_id") or ""),
                "evidence_text": str(raw_step.get("evidence_text") or ""),
                "role": operation,
                "relation": _compact(raw_step.get("relation") or "") or "unknown",
                "directness": _compact(raw_step.get("directness") or "") or "unknown",
                "map_confidence": _float_or_none(raw_step.get("map_confidence")),
                "evidence_map_quality_score": _float_or_none(raw_step.get("evidence_map_quality_score")),
                "covered_atom_ids": _string_list(raw_step.get("covered_atom_ids")),
                "covered_by_previous_step": bool(raw_step.get("covered_by_previous_step", False)),
                "anchor_step": _int_or_default(raw_step.get("anchor_step"), 0),
            }
        )
    return compat


def summarize_mrec_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    steps = [step for step in trace.get("mrec_steps") or [] if isinstance(step, Mapping)]
    operation_counts = Counter(str(step.get("operation") or "") for step in steps)
    state_after_counts = Counter(str(step.get("state_after") or "") for step in steps)
    cue_source_counts = Counter(str(step.get("cue_source") or "") for step in steps)
    covered_atom_ids = {
        atom_id
        for step in steps
        for atom_id in _string_list(step.get("covered_atom_ids"))
    }
    token_costs = [int(step["token_cost"]) for step in steps if step.get("token_cost") is not None]
    return {
        "mrec_trace_version": str(trace.get("mrec_trace_version") or MREC_TRACE_VERSION),
        "mrec_selector_name": str(trace.get("mrec_selector_name") or ""),
        "step_count": len(steps),
        "operation_counts": {key: value for key, value in dict(operation_counts).items() if key},
        "state_after_counts": {key: value for key, value in dict(state_after_counts).items() if key},
        "cue_source_counts": {key: value for key, value in dict(cue_source_counts).items() if key},
        "covered_atom_ids": sorted(covered_atom_ids),
        "fallback_step_count": int(operation_counts.get("FALLBACK", 0)),
        "contrast_step_count": int(operation_counts.get("CONTRAST", 0)),
        "total_token_cost": sum(token_costs) if token_costs else 0,
        "mean_token_cost": (sum(token_costs) / len(token_costs)) if token_costs else 0.0,
    }


def validate_mrec_trace(trace: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _compact(trace.get("event_id") or ""):
        errors.append("missing event_id")
    if str(trace.get("mrec_trace_version") or "") != MREC_TRACE_VERSION:
        errors.append(f"mrec_trace_version must be {MREC_TRACE_VERSION!r}")

    raw_steps = trace.get("mrec_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append("mrec_steps must be a non-empty list")
        return errors

    step_numbers: list[int] = []
    for idx, step in enumerate(raw_steps, start=1):
        if not isinstance(step, Mapping):
            errors.append(f"step {idx} must be a mapping")
            continue
        step_number = _int_or_default(step.get("step"), -1)
        step_numbers.append(step_number)
        operation = str(step.get("operation") or "")
        if operation not in VALID_OPERATIONS:
            errors.append(f"step {step_number} has invalid operation={operation!r}")
        for field in ("state_before", "state_after"):
            value = step.get(field)
            if str(value or "").strip().upper() not in VALID_ATOM_STATES:
                errors.append(f"step {step_number} has invalid {field}={value!r}")
        cue_source = str(step.get("cue_source") or "")
        if cue_source and cue_source not in VALID_CUE_SOURCES:
            errors.append(f"step {step_number} has invalid cue_source={cue_source!r}")
        token_cost = step.get("token_cost")
        if token_cost is not None and _int_or_default(token_cost, -1) < 0:
            errors.append(f"step {step_number} has invalid token_cost={token_cost!r}")

    expected = list(range(1, len(raw_steps) + 1))
    if step_numbers != expected:
        errors.append("mrec_steps step numbers must be consecutive starting at 1")
    return errors


def _infer_operation(state_before: str, relation: str, directness: str, atom_id: str) -> str:
    if not atom_id:
        return "FALLBACK"
    if directness in {"context", "none"} and relation in {"background", "irrelevant", "unknown"}:
        return "BRIDGE"
    relation_state = _state_for_relation(relation)
    if state_before == "U":
        return "OPEN" if relation_state != "U" else "FALLBACK"
    if relation_state in {"S", "R"} and state_before in {"S", "R"} and relation_state != state_before:
        return "CONTRAST"
    if relation_state == "Q" and state_before in {"S", "R", "Q"}:
        return "CONTRAST"
    if relation_state == state_before and state_before in {"S", "R", "Q"}:
        return "CORROBORATE"
    return "FALLBACK"


def _infer_state_after(state_before: str, relation: str, operation: str) -> str:
    relation_state = _state_for_relation(relation)
    if operation == "FALLBACK" or operation == "BRIDGE":
        return state_before
    if operation == "CONTRAST":
        return "Q" if relation_state == "Q" else "C"
    if relation_state != "U":
        return relation_state
    return state_before


def _state_for_relation(relation: str) -> str:
    if relation == "support":
        return "S"
    if relation == "refute":
        return "R"
    if relation in {"qualify", "mixed"}:
        return "Q"
    return "U"


def _normalize_operation(value: Any) -> str:
    operation = str(value or "").strip().upper().replace("-", "_")
    if operation in VALID_OPERATIONS:
        return operation
    raise ValueError(f"invalid MREC operation: {value!r}")


def _normalize_cue_source(value: Any) -> str:
    cue_source = _compact(value or "fallback")
    if cue_source in VALID_CUE_SOURCES:
        return cue_source
    return "fallback"


def _normalize_relation(value: Any) -> str:
    relation = _compact(value).lower()
    if relation in {"support", "supports", "supported_by", "entails", "consistent"}:
        return "support"
    if relation in {"refute", "refutes", "contradict", "contradicts", "counter", "conflict"}:
        return "refute"
    if relation in {"qualify", "qualifies", "qualified", "condition", "context", "hedge"}:
        return "qualify"
    if relation in {"mixed", "partially_supports", "partial"}:
        return "mixed"
    if relation in {"background", "irrelevant", "unknown", "none", ""}:
        return relation or "unknown"
    return relation


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return []


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
