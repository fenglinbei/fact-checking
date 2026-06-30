from __future__ import annotations

import pytest

from fact_checking.selectors.mrec_learned_marginal import (
    LearnedMarginalWeights,
    REWARD_WEIGHT_SCHEMA_VERSION,
    save_learned_marginal_weights,
)
from fact_checking.selectors.minimal_resolving_chain import (
    MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY,
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


def test_build_mrec_trace_row_can_fill_post_target_support_steps_until_min_steps() -> None:
    row = _row(
        claim_atoms=[{"atom_id": "A1", "text": "The bill passed.", "importance": 1.0}],
        candidates=[
            _candidate("E1", relation="support", atoms=["A1"], text="The bill passed the House."),
            _candidate("E2", relation="support", atoms=["A1"], text="The bill also passed the Senate."),
            _candidate(
                "E3",
                relation="unknown",
                directness="context",
                atoms=["A1"],
                text="The bill was discussed during the legislative session.",
            ),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(
            max_steps=5,
            min_steps=3,
            target_resolved_rate=1.0,
            continue_after_target_for_contrast=True,
            post_target_fill_policy="contrast_then_support",
        ),
    )

    assert [step["evidence_id"] for step in trace["mrec_steps"]] == ["E1", "E2", "E3"]
    assert [step["operation"] for step in trace["mrec_steps"]] == ["OPEN", "CORROBORATE", "BRIDGE"]
    assert [step.get("post_target_fill") for step in trace["mrec_steps"]] == [False, True, True]
    assert trace["mrec_diagnostics"]["stop_reason"] == "min_steps_satisfied"
    assert trace["mrec_diagnostics"]["post_target_fill_policy"] == "contrast_then_support"
    assert trace["mrec_diagnostics"]["min_steps"] == 3
    assert trace["mrec_diagnostics"]["post_target_fill_step_count"] == 2
    assert trace["mrec_diagnostics"]["post_target_operation_counts"] == {"BRIDGE": 1, "CORROBORATE": 1}


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


def test_build_mrec_trace_row_treats_zero_max_steps_as_full_candidate_pool() -> None:
    row = _row(
        claim_atoms=[
            {"atom_id": "A1", "text": "The bill passed.", "importance": 1.0},
            {"atom_id": "A2", "text": "It passed in 2024.", "importance": 1.0},
            {"atom_id": "A3", "text": "The vote was bipartisan.", "importance": 1.0},
        ],
        candidates=[
            _candidate("E1", relation="support", atoms=["A1"], text="The bill passed."),
            _candidate("E2", relation="support", atoms=["A2"], text="It passed in 2024."),
            _candidate("E3", relation="support", atoms=["A3"], text="The vote was bipartisan."),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(candidate_top_n=0, max_steps=0, target_resolved_rate=1.0),
    )

    assert [step["evidence_id"] for step in trace["mrec_steps"]] == ["E1", "E2", "E3"]
    assert trace["mrec_diagnostics"]["total_candidate_count"] == 3
    assert trace["mrec_diagnostics"]["stop_reason"] == "reached_max_steps"
    assert trace["mrec_steps"][0]["trace_state"]["selected_count"] == 1
    assert trace["mrec_steps"][-1]["trace_state"]["target_resolved"] is True
    assert trace["mrec_steps"][-1]["trace_state"]["unresolved_atom_ids"] == []


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
    assert trace["mrec_steps"][0]["cue_source"] == "claim_atom"
    assert trace["mrec_steps"][0]["cue_text"] == "The bill passed."
    assert trace["mrec_diagnostics"]["stop_reason"] == "fallback_only"
    assert trace["atom_states_final"] == {"A1": "U"}


def test_build_mrec_trace_row_atom_proposition_cue_ignores_qd_routes() -> None:
    row = _row(
        claim_atoms=[{"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}],
        candidates=[
            {
                **_candidate("E1", relation="support", atoms=["A1"], text="The city council approved the project."),
                "qd_question_routes": [{"question": "Did the old question win?", "rank": 1}],
                "atom_routes": [{"atom_id": "A1", "query_rendering": "What did the city approve?", "rank": 1}],
            }
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(max_steps=3, cue_policy="atom_proposition"),
    )

    assert trace["mrec_steps"][0]["cue_source"] == "claim_atom"
    assert trace["mrec_steps"][0]["cue_text"] == "The city approved the project."
    assert "old question" not in trace["mrec_steps"][0]["cue_text"]


def test_build_mrec_trace_row_uses_pair_level_alignment_for_atom_transition() -> None:
    row = _row(
        claim_atoms=[
            {"atom_id": "A1", "text": "The bill passed.", "importance": 1.0},
            {"atom_id": "A2", "text": "The bill passed in 2024.", "importance": 1.0},
        ],
        candidates=[
            {
                **_candidate("E1", relation="support", atoms=["A1", "A2"], text="The bill passed, but in 2023."),
                "candidate_atom_alignments": [
                    {
                        "evidence_id": "E1",
                        "atom_id": "A1",
                        "relation": "support",
                        "directness": "direct",
                        "confidence": 0.9,
                    },
                    {
                        "evidence_id": "E1",
                        "atom_id": "A2",
                        "relation": "refute",
                        "directness": "direct",
                        "confidence": 0.9,
                    },
                ],
            }
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(max_steps=1, target_resolved_rate=1.0, cue_policy="atom_proposition"),
    )

    assert trace["mrec_steps"][0]["atom_id"] == "A2"
    assert trace["mrec_steps"][0]["relation"] == "refute"
    assert trace["atom_states_final"] == {"A1": "U", "A2": "R"}


def test_learned_marginal_proxy_fills_single_atom_to_min_steps_with_diverse_evidence() -> None:
    row = _row(
        claim_atoms=[{"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}],
        candidates=[
            _candidate("E-support-1", relation="support", atoms=["A1"], text="The city approved the project.", source_group="report-a"),
            _candidate("E-refute", relation="refute", atoms=["A1"], text="The project was rejected in committee.", source_group="report-b"),
            _candidate("E-qualify", relation="qualify", atoms=["A1"], text="The approval applied only to the first phase.", source_group="report-c"),
            _candidate("E-support-2", relation="support", atoms=["A1"], text="Minutes show the project received approval.", source_group="report-d"),
            _candidate("E-support-3", relation="support", atoms=["A1"], text="A separate report confirms the approval.", source_group="report-e"),
            _candidate("E-noise", relation="irrelevant", atoms=[], text="The city held a budget hearing.", source_group="report-f"),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(
            selection_policy="learned_marginal_proxy",
            selector_name="mrec_greedy_transition_v0_2_learned_marginal_proxy",
            max_steps=10,
            min_steps=5,
            target_resolved_rate=1.0,
        ),
    )

    evidence_ids = [step["evidence_id"] for step in trace["mrec_steps"]]
    assert len(trace["mrec_steps"]) >= 5
    assert "E-refute" in evidence_ids
    assert "E-qualify" in evidence_ids
    assert len({step.get("source_group") for step in trace["mrec_steps"] if step.get("source_group")}) >= 5
    assert all("utility_score" in step for step in trace["mrec_steps"])
    assert trace["mrec_diagnostics"]["selection_policy"] == "learned_marginal_proxy"


def test_map_quality_greedy_orders_by_map_quality_without_learned_weights() -> None:
    row = _row(
        claim_atoms=[
            {"atom_id": "A1", "text": "The bill passed.", "importance": 1.0},
            {"atom_id": "A2", "text": "The vote was bipartisan.", "importance": 1.0},
            {"atom_id": "A3", "text": "It passed in 2024.", "importance": 1.0},
        ],
        candidates=[
            _candidate(
                "E-high-hybrid-low-quality",
                relation="support",
                atoms=["A1"],
                text="A high retrieval score item has weak map quality.",
                evidence_map_quality_score=0.10,
                base_score=30.0,
                hybrid_score=99.0,
            ),
            _candidate(
                "E-map-quality-score",
                relation="support",
                atoms=["A2"],
                text="A fallback map_quality_score item should rank first.",
                evidence_map_quality_score=None,
                map_quality_score=0.95,
                base_score=0.1,
                hybrid_score=0.1,
            ),
            _candidate(
                "E-evidence-map-quality",
                relation="support",
                atoms=["A3"],
                text="An evidence_map_quality_score item should rank second.",
                evidence_map_quality_score=0.80,
                base_score=0.2,
                hybrid_score=0.2,
            ),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(
            selection_policy=MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY,
            selector_name="mrec_greedy_transition_v0_2_map_quality_greedy",
            max_steps=3,
            target_resolved_rate=1.1,
        ),
    )

    assert [step["evidence_id"] for step in trace["mrec_steps"]] == [
        "E-map-quality-score",
        "E-evidence-map-quality",
        "E-high-hybrid-low-quality",
    ]
    assert [step["map_quality_greedy_score"] for step in trace["mrec_steps"]] == [0.95, 0.80, 0.10]
    assert all(step["selection_policy"] == MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY for step in trace["mrec_steps"])
    assert trace["mrec_diagnostics"]["selection_policy"] == MREC_SELECTION_POLICY_MAP_QUALITY_GREEDY
    assert "utility_score" not in trace["mrec_steps"][0]


def test_learned_marginal_proxy_continues_to_unresolved_atom_after_target_rate() -> None:
    row = _row(
        claim_atoms=[
            {"atom_id": "A1", "text": "The bill passed.", "importance": 1.0},
            {"atom_id": "A2", "text": "The bill passed in 2024.", "importance": 1.0},
        ],
        candidates=[
            _candidate("E1", relation="support", atoms=["A1"], text="The bill passed."),
            _candidate("E2", relation="refute", atoms=["A2"], text="The bill passed in 2023 instead."),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(
            selection_policy="learned_marginal_proxy",
            max_steps=5,
            min_steps=2,
            target_resolved_rate=0.5,
        ),
    )

    assert [step["evidence_id"] for step in trace["mrec_steps"]] == ["E1", "E2"]
    assert trace["atom_states_final"] == {"A1": "S", "A2": "R"}
    assert trace["mrec_diagnostics"]["stop_reason"] == "min_steps_satisfied"


def test_learned_marginal_reward_stops_after_min_steps_when_reward_is_non_positive(tmp_path) -> None:
    weights_path = tmp_path / "reward_weights.json"
    save_learned_marginal_weights(
        weights_path,
        LearnedMarginalWeights(
            feature_weights={},
            cost_weight=0.0,
            schema_version=REWARD_WEIGHT_SCHEMA_VERSION,
            bias=-1.0,
        ),
    )
    row = _row(
        claim_atoms=[{"atom_id": "A1", "text": "The bill passed.", "importance": 1.0}],
        candidates=[
            _candidate("E1", relation="support", atoms=["A1"], text="The bill passed."),
            _candidate("E2", relation="support", atoms=["A1"], text="A second report says the bill passed."),
        ],
    )

    trace = build_mrec_trace_row(
        row,
        params=MRECSelectorParams(
            selection_policy="learned_marginal_reward",
            selector_name="mrec_greedy_transition_v0_2_learned_marginal_reward",
            weight_file=str(weights_path),
            max_steps=5,
            min_steps=1,
            stop_threshold=0.0,
        ),
    )

    assert [step["evidence_id"] for step in trace["mrec_steps"]] == ["E1"]
    assert trace["mrec_steps"][0]["selection_policy"] == "learned_marginal_reward"
    assert trace["mrec_steps"][0]["utility_score"] < 0.0
    assert trace["mrec_diagnostics"]["selection_policy"] == "learned_marginal_reward"
    assert trace["mrec_diagnostics"]["stop_reason"] == "utility_below_threshold"


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
    source_group: str = "report-a",
    evidence_map_quality_score: float | None = 0.7,
    map_quality_score: float | None = None,
    base_score: float = 0.5,
    hybrid_score: float | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "evidence_id": evidence_id,
        "candidate_idx": 0,
        "selector_candidate_idx": 0,
        "text": text,
        "covered_atom_ids": atoms,
        "map_relation": relation,
        "map_directness": directness,
        "map_confidence": 0.8,
        "duplicate_group": duplicate_group,
        "source_group": source_group,
        "base_score": base_score,
    }
    if evidence_map_quality_score is not None:
        candidate["evidence_map_quality_score"] = evidence_map_quality_score
    if map_quality_score is not None:
        candidate["map_quality_score"] = map_quality_score
    if hybrid_score is not None:
        candidate["hybrid_score"] = hybrid_score
    return candidate
