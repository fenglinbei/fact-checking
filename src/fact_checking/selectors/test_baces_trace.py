from __future__ import annotations

from copy import deepcopy

import pytest

from fact_checking.selectors.baces_trace import (
    CANDIDATE_PROJECTION_SCHEMA,
    SOLVER_VERSION,
    TRACE_SCHEMA_VERSION,
    build_exact_trace,
    reorder_selected_trace,
    replay_trace,
)


def test_exact_trace_appends_canonical_zero_gain_fill_and_compatibility_view() -> None:
    row = _row(
        _candidate("Z-core", (2, 2), cost=0, hybrid_score=0.01),
        _candidate("R-zero", (0, 0), cost=1, hybrid_score=0.99),
        _candidate("P-partial", (1, 0), cost=1, retrieval_score=0.20),
        _candidate("D-direct", (2, 0), cost=1, hybrid_score=0.10),
        _candidate("N-costly", (0, 0), cost=2, hybrid_score=1.00),
    )

    trace = build_exact_trace(row, k_min=4, k_max=5)

    assert trace["schema_version"] == TRACE_SCHEMA_VERSION
    assert trace["solver_version"] == SOLVER_VERSION
    assert trace["candidate_pool_projection_schema"] == CANDIDATE_PROJECTION_SCHEMA
    assert trace["coverage_core_keys"] == ["Z-core"]
    # The frozen fill key is (cost, -D, -P, -retrieval, stable key).
    assert trace["selected_keys"] == [
        "Z-core",
        "D-direct",
        "P-partial",
        "R-zero",
    ]
    assert trace["k_core"] == 1
    assert trace["k_sel"] == 4
    assert trace["zero_gain_fill_count"] == 3
    assert trace["min_count_unreachable"] is False
    assert [step["solver_role"] for step in trace["baces_steps"]] == [
        "CORE",
        "FILL",
        "FILL",
        "FILL",
    ]
    assert [step["operation"] for step in trace["baces_steps"]] == [
        "COVER",
        "ZERO_GAIN_FILL",
        "ZERO_GAIN_FILL",
        "ZERO_GAIN_FILL",
    ]
    assert [step["display_marginal_coverage_units"] for step in trace["baces_steps"]] == [
        4,
        0,
        0,
        0,
    ]
    assert trace["selector_ordered_indices"] == trace["display_ordered_indices"]
    assert trace["selected_indices"] == trace["display_ordered_indices"]
    assert all(
        not {"state_before", "state_after", "atom_states_before", "atom_states_after"}
        .intersection(step)
        for step in trace["mrec_steps"]
    )
    by_key = {
        candidate["candidate_stable_key"]: candidate
        for candidate in trace["candidate_pool"]
    }
    assert by_key["D-direct"]["retrieval_score"] == 0.10
    assert by_key["P-partial"]["retrieval_score"] == 0.20
    assert replay_trace(trace)["errors"] == []


def test_soft_floor_underfills_instead_of_violating_token_budget() -> None:
    row = _row(
        _candidate("core", (2,), cost=4),
        _candidate("fill-1", (0,), cost=1, hybrid_score=0.1),
        _candidate("fill-2", (0,), cost=2, hybrid_score=0.9),
        atoms=("A1",),
    )

    trace = build_exact_trace(row, k_min=4, k_max=4, token_budget=6)

    assert trace["coverage_core_keys"] == ["core"]
    assert trace["selected_keys"] == ["core", "fill-1"]
    assert trace["selected_token_cost"] == 5
    assert trace["k_sel"] == 2
    assert trace["min_count_unreachable"] is True
    assert trace["selected_token_cost"] <= trace["token_budget"]
    assert replay_trace(trace)["ok"] is True


def test_raw_oracle_poison_is_excluded_and_replay_detects_step_tampering() -> None:
    row = _row(
        _candidate("A", (2,), cost=1),
        _candidate("B", (1,), cost=1),
        atoms=("A1",),
    )
    row.update(
        {
            "oracle_ordered_keys": ["A"],
            "gold_label": "true",
            "verifier_score": 0.99,
            "learned_weights": {"resolution_delta": 1000},
        }
    )
    row["candidates"][0].update(
        {
            "oracle_selected": True,
            "verifier_reward": 999,
            "learned_marginal_score": 999,
        }
    )
    poisoned = deepcopy(row)
    poisoned["oracle_ordered_keys"] = ["B", "A"]
    poisoned["gold_label"] = "false"
    poisoned["verifier_score"] = -123
    poisoned["learned_weights"] = {"resolution_delta": -1000}
    poisoned["candidates"][0]["oracle_selected"] = False
    poisoned["candidates"][0]["verifier_reward"] = -999
    poisoned["candidates"][0]["learned_marginal_score"] = -999

    clean_trace = build_exact_trace(row, k_min=2, k_max=2)
    poisoned_trace = build_exact_trace(poisoned, k_min=2, k_max=2)

    assert poisoned_trace == clean_trace
    assert not {
        "oracle_ordered_keys",
        "gold_label",
        "verifier_score",
        "learned_weights",
    }.intersection(clean_trace)
    assert all(
        not {"oracle_selected", "verifier_reward", "learned_marginal_score"}
        .intersection(candidate)
        for candidate in clean_trace["candidate_pool"]
    )

    tampered = deepcopy(clean_trace)
    tampered["baces_steps"][0]["display_coverage_levels_after"]["A1"] = 0
    tampered["baces_steps"][0]["display_operation"] = "DISPLAY_ZERO_GAIN"
    result = replay_trace(tampered)

    assert result["ok"] is False
    assert any("display_coverage_levels_after" in error for error in result["errors"])
    assert any("display_operation" in error for error in result["errors"])
    assert any("trace_fingerprint" in error for error in result["errors"])


def test_same_set_reorder_freezes_solver_roles_but_replays_display_state() -> None:
    row = _row(
        _candidate("A-core", (2, 1), cost=1),
        _candidate("B-fill", (2, 0), cost=1),
        _candidate("C-core", (0, 2), cost=1),
    )
    original = build_exact_trace(row, k_min=3, k_max=3)
    assert original["coverage_core_keys"] == ["A-core", "C-core"]
    assert original["selected_keys"] == ["A-core", "C-core", "B-fill"]
    assert original["display_weighted_coverage_acquisition_time"] == 5

    reordered = reorder_selected_trace(
        original,
        ["B-fill", "A-core", "C-core"],
        display_order_policy="same_set_fill_first_fixture",
    )

    assert set(reordered["selected_keys"]) == set(original["selected_keys"])
    assert reordered["selected_set_fingerprint"] == original["selected_set_fingerprint"]
    assert reordered["display_order_fingerprint"] != original["display_order_fingerprint"]
    assert reordered["coverage_core_keys"] == original["coverage_core_keys"]
    assert reordered["solver_objective_tuple"] == original["solver_objective_tuple"]
    assert [step["solver_role"] for step in reordered["baces_steps"]] == [
        "FILL",
        "CORE",
        "CORE",
    ]
    assert [step["operation"] for step in reordered["baces_steps"]] == [
        "ZERO_GAIN_FILL",
        "COVER",
        "COVER",
    ]
    assert [step["display_operation"] for step in reordered["baces_steps"]] == [
        "ORDINAL_UPGRADE",
        "ORDINAL_UPGRADE",
        "ORDINAL_UPGRADE",
    ]
    assert [
        step["display_marginal_coverage_units"] for step in reordered["baces_steps"]
    ] == [2, 1, 1]
    assert reordered["baces_steps"][0]["display_coverage_levels_before"] == {
        "A1": 0,
        "A2": 0,
    }
    assert reordered["baces_steps"][0]["display_coverage_levels_after"] == {
        "A1": 2,
        "A2": 0,
    }
    assert reordered["display_weighted_coverage_acquisition_time"] == 7
    assert reordered["display_padded_prefix_auc"] == 9
    assert [step["target_coverage_reached"] for step in reordered["baces_steps"]] == [
        False,
        False,
        True,
    ]
    assert replay_trace(reordered)["ok"] is True


def test_trace_is_deterministic_under_feature_candidate_permutation() -> None:
    candidates = [
        _candidate("C", (0, 2), cost=2, hybrid_score=0.7),
        _candidate("A", (2, 0), cost=2, hybrid_score=0.9),
        _candidate("B", (1, 1), cost=1, hybrid_score=0.8),
    ]
    forward = _row(*candidates)
    reverse = _row(*reversed(candidates))

    first = build_exact_trace(forward, k_min=3, k_max=3)
    second = build_exact_trace(reverse, k_min=3, k_max=3)

    assert first == second
    assert [
        candidate["candidate_stable_key"] for candidate in first["candidate_pool"]
    ] == ["A", "B", "C"]
    assert first["trace_fingerprint"] == second["trace_fingerprint"]


def test_reorder_rejects_a_changed_set_or_invalid_source_trace() -> None:
    trace = build_exact_trace(
        _row(
            _candidate("A", (2,), cost=1),
            _candidate("B", (0,), cost=1),
            atoms=("A1",),
        ),
        k_min=2,
        k_max=2,
    )

    with pytest.raises(ValueError, match="permutation"):
        reorder_selected_trace(trace, ["A"], "dropped_one")

    poisoned = deepcopy(trace)
    poisoned["selected_indices"] = [999, 998]
    with pytest.raises(ValueError, match="invalid BACES trace"):
        reorder_selected_trace(poisoned, trace["selected_keys"], "cannot_use_poison")


def _row(
    *candidates: dict,
    atoms: tuple[str, ...] = ("A1", "A2"),
) -> dict:
    return {
        "event_id": "fixture-event",
        "claim_atoms": [
            {"atom_id": atom_id, "proposition": f"Proposition for {atom_id}"}
            for atom_id in atoms
        ],
        "candidates": [deepcopy(candidate) for candidate in candidates],
    }


def _candidate(
    uid: str,
    levels: tuple[int, ...],
    *,
    cost: int,
    retrieval_score: float | None = None,
    hybrid_score: float | None = None,
) -> dict:
    alignments: list[dict] = []
    evidence_id = f"evidence-{uid}"
    for atom_index, level in enumerate(levels, start=1):
        if level == 0:
            continue
        alignments.append(
            {
                "evidence_id": evidence_id,
                "atom_id": f"A{atom_index}",
                "relation": "support",
                "directness": "direct" if level == 2 else "partial",
                "confidence": 1.0,
                "key_spans": [f"span-{uid}-{atom_index}"],
            }
        )
    row = {
        "candidate_uid": uid,
        "candidate_key": f"readable-{uid}",
        "evidence_id": evidence_id,
        "num_tokens": cost,
        "candidate_atom_alignments": alignments,
    }
    if retrieval_score is not None:
        row["retrieval_score"] = retrieval_score
    if hybrid_score is not None:
        row["hybrid_score"] = hybrid_score
    return row
