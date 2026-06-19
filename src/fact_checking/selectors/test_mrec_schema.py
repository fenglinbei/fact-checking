from __future__ import annotations

from fact_checking.selectors.mrec_schema import (
    MREC_TRACE_VERSION,
    build_initial_atom_states,
    build_mrec_step,
    mrec_steps_to_compat_chain_steps,
    normalize_atom_state,
    summarize_mrec_trace,
    validate_mrec_trace,
)


def test_initial_atom_states_default_to_unresolved() -> None:
    states = build_initial_atom_states(
        [
            {"atom_id": "A1", "text": "The bill passed."},
            {"node_id": "A2", "text": "It passed in 2024."},
        ]
    )

    assert states == {"A1": "U", "A2": "U"}
    assert normalize_atom_state("unresolved") == "U"


def test_open_step_infers_supported_refuted_and_qualified_states() -> None:
    support = build_mrec_step(
        step=1,
        candidate=_candidate("E1", relation="support", atoms=["A1"]),
        atom_id="A1",
        atom_text="The bill passed.",
        state_before="U",
        operation="OPEN",
        cue_text="The bill passed.",
        transition_reason="first direct support",
    )
    refute = build_mrec_step(
        step=2,
        candidate=_candidate("E2", relation="refute", atoms=["A2"]),
        atom_id="A2",
        atom_text="It passed in 2024.",
        state_before="U",
        operation="OPEN",
        cue_text="It passed in 2024.",
        transition_reason="first direct refutation",
    )
    qualify = build_mrec_step(
        step=3,
        candidate=_candidate("E3", relation="mixed", atoms=["A3"]),
        atom_id="A3",
        atom_text="The claim is qualified.",
        state_before="U",
        operation="OPEN",
        cue_text="The claim is qualified.",
        transition_reason="first qualifier",
    )

    assert support["state_after"] == "S"
    assert refute["state_after"] == "R"
    assert qualify["state_after"] == "Q"


def test_opposite_relation_on_resolved_atom_infers_contrast() -> None:
    step = build_mrec_step(
        step=2,
        candidate=_candidate("E2", relation="refute", atoms=["A1"]),
        atom_id="A1",
        atom_text="The bill passed.",
        state_before="S",
        cue_text="The bill passed.",
        transition_reason="conflicting evidence",
    )

    assert step["operation"] == "CONTRAST"
    assert step["state_after"] == "C"


def test_fallback_step_without_covered_atoms_is_valid() -> None:
    step = build_mrec_step(
        step=1,
        candidate=_candidate("E1", relation="background", atoms=[]),
        state_before="U",
        operation="FALLBACK",
        cue_text="Verify the main factual claim.",
        cue_source="fallback",
        transition_reason="fallback evidence with no atom coverage",
    )
    trace = _trace([step], claim_atoms=[{"atom_id": "A1", "text": "The bill passed."}])

    assert step["operation"] == "FALLBACK"
    assert step["covered_atom_ids"] == []
    assert validate_mrec_trace(trace) == []


def test_compat_chain_steps_project_fields_consumed_by_qec_builder() -> None:
    mrec_step = build_mrec_step(
        step=1,
        candidate=_candidate("E1", relation="support", directness="direct", atoms=["A1"]),
        atom_id="A1",
        atom_text="The bill passed.",
        state_before="U",
        operation="OPEN",
        cue_text="The bill passed.",
        cue_source="claim_atom",
        transition_reason="first direct support",
    )

    compat = mrec_steps_to_compat_chain_steps([mrec_step])

    assert compat == [
        {
            "step": 1,
            "atom_id": "A1",
            "atom_text": "The bill passed.",
            "cue_text": "The bill passed.",
            "cue_source": "claim_atom",
            "candidate_idx": 0,
            "selector_candidate_idx": 0,
            "evidence_id": "E1",
            "evidence_text": "Evidence E1",
            "role": "open",
            "relation": "support",
            "directness": "direct",
            "map_confidence": 0.8,
            "evidence_map_quality_score": 0.7,
            "covered_atom_ids": ["A1"],
            "covered_by_previous_step": False,
            "anchor_step": 0,
        }
    ]


def test_summarize_mrec_trace_reports_state_and_operation_diagnostics() -> None:
    steps = [
        build_mrec_step(
            step=1,
            candidate=_candidate("E1", relation="support", atoms=["A1"]),
            atom_id="A1",
            atom_text="The bill passed.",
            state_before="U",
            operation="OPEN",
            cue_text="The bill passed.",
            transition_reason="first direct support",
            token_cost=12,
        ),
        build_mrec_step(
            step=2,
            candidate=_candidate("E2", relation="refute", atoms=["A1"]),
            atom_id="A1",
            atom_text="The bill passed.",
            state_before="S",
            cue_text="The bill passed.",
            transition_reason="conflicting evidence",
            token_cost=9,
        ),
    ]
    trace = _trace(steps, claim_atoms=[{"atom_id": "A1", "text": "The bill passed."}])

    summary = summarize_mrec_trace(trace)

    assert summary["mrec_trace_version"] == MREC_TRACE_VERSION
    assert summary["step_count"] == 2
    assert summary["operation_counts"] == {"OPEN": 1, "CONTRAST": 1}
    assert summary["state_after_counts"] == {"S": 1, "C": 1}
    assert summary["total_token_cost"] == 21
    assert summary["fallback_step_count"] == 0


def test_validate_mrec_trace_reports_required_shape_errors() -> None:
    invalid = {
        "mrec_trace_version": MREC_TRACE_VERSION,
        "mrec_steps": [
            {"step": 2, "state_before": "BAD", "state_after": "S", "operation": "OPEN"},
        ],
    }

    errors = validate_mrec_trace(invalid)

    assert "missing event_id" in errors
    assert "mrec_steps step numbers must be consecutive starting at 1" in errors
    assert "step 2 has invalid state_before='BAD'" in errors


def test_validate_mrec_trace_reports_empty_steps() -> None:
    errors = validate_mrec_trace({"event_id": "100", "mrec_trace_version": MREC_TRACE_VERSION, "mrec_steps": []})

    assert errors == ["mrec_steps must be a non-empty list"]


def _candidate(
    evidence_id: str,
    *,
    relation: str = "support",
    directness: str = "direct",
    atoms: list[str] | None = None,
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "candidate_idx": 0,
        "selector_candidate_idx": 0,
        "text": f"Evidence {evidence_id}",
        "covered_atom_ids": list(atoms or []),
        "map_relation": relation,
        "map_directness": directness,
        "map_confidence": 0.8,
        "evidence_map_quality_score": 0.7,
    }


def _trace(steps: list[dict[str, object]], *, claim_atoms: list[dict[str, object]]) -> dict[str, object]:
    return {
        "event_id": "100",
        "claim": "The bill passed.",
        "gold_label": "true",
        "mrec_trace_version": MREC_TRACE_VERSION,
        "mrec_selector_name": "mrec_schema_test",
        "candidate_pool": [],
        "selected_indices": [],
        "selected_candidates": [],
        "claim_atoms": claim_atoms,
        "atom_states_initial": build_initial_atom_states(claim_atoms),
        "atom_states_final": {str(step.get("atom_id")): str(step.get("state_after")) for step in steps if step.get("atom_id")},
        "mrec_steps": steps,
        "mrec_diagnostics": {},
        "compat_chain_steps": mrec_steps_to_compat_chain_steps(steps),
    }
