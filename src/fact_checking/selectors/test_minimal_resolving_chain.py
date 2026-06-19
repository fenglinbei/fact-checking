from __future__ import annotations

import pytest

from fact_checking.selectors.minimal_resolving_chain import (
    MRECSelectorParams,
    build_mrec_trace_row,
)
from fact_checking.selectors.mrec_schema import validate_mrec_trace


def test_build_mrec_trace_row_opens_unresolved_atoms_and_skips_noise() -> None:
    row = _row(
        claim_atoms=[
            {"atom_id": "A1", "text": "The bill passed.", "importance": 1.0},
            {"atom_id": "A2", "text": "It passed in 2024.", "importance": 1.0},
        ],
        candidates=[
            _candidate("E-bg", relation="background", atoms=[], text="General legislative context."),
            _candidate("E1", relation="support", atoms=["A1"], text="The bill passed the House.", duplicate_group="D1"),
            _candidate("E1-dup", relation="support", atoms=["A1"], text="The bill passed the House.", duplicate_group="D1"),
            _candidate("E2", relation="refute", atoms=["A2"], text="The vote happened in 2023, not 2024."),
        ],
    )

    trace = build_mrec_trace_row(row, params=MRECSelectorParams(max_steps=5, target_resolved_rate=1.0))

    assert validate_mrec_trace(trace) == []
    assert [step["evidence_id"] for step in trace["mrec_steps"]] == ["E1", "E2"]
    assert [step["operation"] for step in trace["mrec_steps"]] == ["OPEN", "OPEN"]
    assert trace["atom_states_final"] == {"A1": "S", "A2": "R"}
    assert trace["mrec_diagnostics"]["duplicate_rejected_count"] == 1
    assert trace["mrec_diagnostics"]["background_rejected_count"] == 1
    assert trace["mrec_diagnostics"]["stop_reason"] == "target_resolution_reached"
    assert trace["selected_evidence_ids"] == ["E1", "E2"]
    assert trace["chain_steps"] == trace["compat_chain_steps"]


def test_build_mrec_trace_row_can_continue_for_contrast_after_resolution() -> None:
    row = _row(
        claim_atoms=[{"atom_id": "A1", "text": "The bill passed.", "importance": 1.0}],
        candidates=[
            _candidate("E1", relation="support", atoms=["A1"], text="The bill passed the House."),
            _candidate("E2", relation="refute", atoms=["A1"], text="The bill failed in the Senate."),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(
            max_steps=3,
            target_resolved_rate=1.0,
            continue_after_target_for_contrast=True,
        ),
    )

    assert [step["operation"] for step in trace["mrec_steps"]] == ["OPEN", "CONTRAST"]
    assert [step["state_before"] for step in trace["mrec_steps"]] == ["U", "S"]
    assert [step["state_after"] for step in trace["mrec_steps"]] == ["S", "C"]
    assert trace["atom_states_final"] == {"A1": "C"}
    assert trace["mrec_diagnostics"]["conflicted_atom_ids"] == ["A1"]


def test_build_mrec_trace_row_respects_token_budget() -> None:
    row = _row(
        claim_atoms=[
            {"atom_id": "A1", "text": "The bill passed.", "importance": 1.0},
            {"atom_id": "A2", "text": "It passed in 2024.", "importance": 1.0},
        ],
        candidates=[
            _candidate("E1", relation="support", atoms=["A1"], text="one two three four"),
            _candidate("E2", relation="support", atoms=["A2"], text="five six seven eight"),
        ],
    )

    trace = build_mrec_trace_row(row, params=MRECSelectorParams(max_steps=5, token_budget=5, target_resolved_rate=1.0))

    assert [step["evidence_id"] for step in trace["mrec_steps"]] == ["E1"]
    assert trace["mrec_steps"][0]["token_cost"] == 4
    assert trace["mrec_diagnostics"]["stop_reason"] == "token_budget_exhausted"
    assert trace["mrec_diagnostics"]["unresolved_atom_ids"] == ["A2"]


def test_build_mrec_trace_row_uses_single_fallback_when_no_resolving_evidence_exists() -> None:
    row = _row(
        claim_atoms=[{"atom_id": "A1", "text": "The bill passed.", "importance": 1.0}],
        candidates=[
            _candidate("E-bg", relation="background", atoms=[], text="General legislative context."),
            _candidate("E-none", relation="unknown", directness="none", atoms=[], text="No clear evidence."),
        ],
    )

    trace = build_mrec_trace_row(row, params=MRECSelectorParams(max_steps=3, allow_fallback=True))

    assert validate_mrec_trace(trace) == []
    assert len(trace["mrec_steps"]) == 1
    assert trace["mrec_steps"][0]["operation"] == "FALLBACK"
    assert trace["mrec_diagnostics"]["stop_reason"] == "fallback_only"
    assert trace["atom_states_final"] == {"A1": "U"}


def test_build_mrec_trace_row_rejects_empty_candidate_pool() -> None:
    with pytest.raises(ValueError, match="MREC input trace has no candidate_pool"):
        build_mrec_trace_row(
            _row(
                claim_atoms=[{"atom_id": "A1", "text": "The bill passed.", "importance": 1.0}],
                candidates=[],
            )
        )


def _row(*, claim_atoms: list[dict[str, object]], candidates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "event_id": "case-1",
        "claim": "The bill passed in 2024.",
        "gold_label": "false",
        "claim_atoms": claim_atoms,
        "candidate_pool": candidates,
    }


def _candidate(
    evidence_id: str,
    *,
    relation: str,
    atoms: list[str],
    text: str,
    directness: str = "direct",
    duplicate_group: str = "",
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "candidate_idx": 0,
        "selector_candidate_idx": 0,
        "text": text,
        "covered_atom_ids": atoms,
        "map_relation": relation,
        "map_directness": directness,
        "map_confidence": 0.8,
        "evidence_map_quality_score": 0.7,
        "duplicate_group": duplicate_group,
        "base_score": 0.5,
    }
