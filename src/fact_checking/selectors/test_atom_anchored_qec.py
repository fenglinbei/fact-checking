from __future__ import annotations

from fact_checking.selectors.atom_anchored_qec import (
    AtomAnchoredQECParams,
    build_atom_anchored_qec_trace_row,
    summarize_atom_anchored_qec_traces,
)


def test_view_reorders_same_selected_set_by_atom_order() -> None:
    row = _trace_row(selected_indices=[1, 0, 2])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="keep_all_reorder"),
    )

    assert trace["selector_ordered_indices"] == [0, 1, 2]
    assert sorted(trace["selector_ordered_indices"]) == [0, 1, 2]
    assert trace["selector_name"] == "aa_qec_view_keep_all_qd_prefer_selected_min5_10"
    assert trace["graph_version"] == "atom_anchored_qec_v1"
    assert trace["candidate_pool_metadata"]["adaptive_policy"] == "aa_qec_view"
    assert [step["role"] for step in trace["chain_steps"]] == ["primary", "primary", "fallback"]
    assert [step["cue_text"] for step in trace["chain_steps"]] == [
        "Did atom one happen?",
        "Did atom two happen?",
        "Verify the main factual claim.",
    ]


def test_view_shuffled_order_is_seeded_and_preserves_selected_set() -> None:
    row = _trace_row(selected_indices=[0, 1, 2, 3])

    first = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="shuffled", random_seed=7),
    )
    second = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="shuffled", random_seed=7),
    )

    assert first["selector_ordered_indices"] == second["selector_ordered_indices"]
    assert sorted(first["selector_ordered_indices"]) == [0, 1, 2, 3]
    assert len(first["chain_steps"]) == len(first["selector_ordered_indices"])


def test_summarize_atom_anchored_qec_traces_reports_chain_diagnostics() -> None:
    trace = build_atom_anchored_qec_trace_row(
        _trace_row(selected_indices=[1, 0, 2]),
        params=AtomAnchoredQECParams(selection_policy="keep_all_reorder"),
    )

    assert trace["chain_diagnostics"]["chain_steps"] == 3
    assert trace["chain_diagnostics"]["primary_step_count"] == 2
    assert trace["chain_diagnostics"]["fallback_step_count"] == 1

    summary = summarize_atom_anchored_qec_traces([trace])

    assert summary["n_rows"] == 1
    assert summary["selector_names"] == {"aa_qec_view_keep_all_qd_prefer_selected_min5_10": 1}
    assert summary["chain_steps"]["mean"] == 3.0
    assert summary["role_counts"] == {"primary": 2, "fallback": 1}
    assert summary["cue_source_counts"] == {"qd_question": 2, "fallback": 1}


def test_constrained_primary_only_selects_best_primary_per_atom_from_selected_set() -> None:
    row = _stage2_trace_row(selected_indices=[0, 1, 2, 3, 4, 5])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(
            selection_policy="primary_only",
            source_selector_name="v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10",
        ),
    )

    assert trace["selector_name"] == "aa_qec_constrained_atom_facts_abc_primary_only_qd_prefer_selected_max10"
    assert trace["candidate_pool_metadata"]["adaptive_policy"] == "aa_qec_constrained_atom_facts_abc"
    assert trace["selector_ordered_indices"] == [1, 3]
    assert [step["role"] for step in trace["chain_steps"]] == ["primary", "primary"]
    assert trace["chain_diagnostics"]["primary_step_count"] == 2
    assert set(trace["selector_ordered_indices"]).issubset(set(row["selected_indices"]))


def test_constrained_primary_secondary_adds_counter_and_qualifier_steps() -> None:
    row = _stage2_trace_row(selected_indices=[0, 1, 2, 3, 4, 5])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="primary_secondary"),
    )

    assert trace["selector_ordered_indices"] == [1, 2, 3, 4]
    assert [step["role"] for step in trace["chain_steps"]] == [
        "primary",
        "secondary",
        "primary",
        "secondary",
    ]
    assert [step["relation"] for step in trace["chain_steps"]] == ["support", "refute", "support", "qualify"]
    assert trace["chain_diagnostics"]["secondary_step_count"] == 2


def test_constrained_primary_secondary_fallback_min5_fills_from_selected_order() -> None:
    row = _stage2_trace_row(selected_indices=[0, 1, 2, 3, 4, 5])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="primary_secondary_fallback_min5"),
    )

    assert trace["selector_ordered_indices"] == [1, 2, 3, 4, 0]
    assert [step["role"] for step in trace["chain_steps"]] == [
        "primary",
        "secondary",
        "primary",
        "secondary",
        "fallback",
    ]
    assert trace["chain_diagnostics"]["fallback_step_count"] == 1
    assert trace["chain_diagnostics"]["fallback_fill_rate"] == 0.2
    assert set(trace["selector_ordered_indices"]).issubset(set(row["selected_indices"]))


def test_constrained_primary_fallback_no_secondary_uses_fallback_roles_for_min5() -> None:
    row = _stage2_trace_row(selected_indices=[0, 1, 2, 3, 4, 5])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(selection_policy="primary_fallback_min5_no_secondary"),
    )

    assert trace["selector_ordered_indices"] == [1, 3, 0, 2, 4]
    assert [step["role"] for step in trace["chain_steps"]] == [
        "primary",
        "primary",
        "fallback",
        "fallback",
        "fallback",
    ]
    assert trace["chain_diagnostics"]["secondary_step_count"] == 0
    assert trace["chain_diagnostics"]["fallback_step_count"] == 3


def test_constrained_primary_modes_keep_one_fallback_when_no_selected_candidate_covers_atoms() -> None:
    row = _stage2_trace_row(selected_indices=[0, 1, 2, 3, 4, 5])
    for candidate in row["candidate_pool"]:
        candidate["covered_atom_ids"] = []
        candidate["map_relation"] = "irrelevant"
        candidate["map_directness"] = "none"
        candidate["map_confidence"] = 0.0
        candidate["evidence_map_quality_score"] = 0.0

    for policy in ("primary_only", "primary_secondary"):
        trace = build_atom_anchored_qec_trace_row(
            row,
            params=AtomAnchoredQECParams(selection_policy=policy),
        )

        assert trace["selector_ordered_indices"] == [0]
        assert [step["role"] for step in trace["chain_steps"]] == ["fallback"]
        assert trace["chain_diagnostics"]["primary_step_count"] == 0
        assert trace["chain_diagnostics"]["fallback_step_count"] == 1
        assert set(trace["selector_ordered_indices"]).issubset(set(row["selected_indices"]))


def test_full_top20_scope_can_select_candidates_outside_source_selected_set() -> None:
    row = _stage2_trace_row(selected_indices=[0, 5])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(
            candidate_scope="top20",
            selection_policy="primary_fallback_min5_no_secondary",
            source_selector_name="v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10",
        ),
    )

    assert trace["selector_name"] == (
        "aa_qec_full_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_top20_min5_10"
    )
    assert trace["candidate_pool_metadata"]["adaptive_policy"] == "aa_qec_full_atom_facts_abc"
    assert trace["selector_ordered_indices"] == [1, 3, 6, 0, 2]
    assert 6 in trace["selector_ordered_indices"]
    assert not set(trace["selector_ordered_indices"]).issubset(set(row["selected_indices"]))
    assert [step["role"] for step in trace["chain_steps"]] == [
        "primary",
        "primary",
        "primary",
        "fallback",
        "fallback",
    ]
    assert trace["chain_diagnostics"]["candidate_scope"] == "top20"
    assert trace["chain_diagnostics"]["source_selected_count"] == 2
    assert trace["chain_diagnostics"]["secondary_step_count"] == 0


def test_full_top20_primary_secondary_fallback_min5_uses_secondary_and_caps_max10() -> None:
    row = _stage2_trace_row(selected_indices=[0, 5])

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(
            candidate_scope="top20",
            selection_policy="primary_secondary_fallback_min5",
        ),
    )

    assert trace["selector_name"] == (
        "aa_qec_full_atom_facts_abc_primary_secondary_fallback_qd_prefer_top20_min5_10"
    )
    assert trace["selector_ordered_indices"] == [1, 2, 3, 4, 6]
    assert [step["role"] for step in trace["chain_steps"]] == [
        "primary",
        "secondary",
        "primary",
        "secondary",
        "primary",
    ]
    assert trace["chain_diagnostics"]["secondary_step_count"] == 2
    assert trace["chain_diagnostics"]["fallback_step_count"] == 0

    long_row = _stage2_trace_row(selected_indices=[0, 5])
    _add_secondary_pairs(long_row, atom_start=4, n_atoms=8)
    capped_trace = build_atom_anchored_qec_trace_row(
        long_row,
        params=AtomAnchoredQECParams(
            candidate_scope="top20",
            selection_policy="primary_secondary_fallback_min5",
        ),
    )

    assert len(capped_trace["selector_ordered_indices"]) == 10


def test_full_top20_primary_secondary_dynamic_has_no_min_fill_or_max_cap() -> None:
    row = _stage2_trace_row(selected_indices=[0, 5])
    _add_secondary_pairs(row, atom_start=4, n_atoms=8)

    trace = build_atom_anchored_qec_trace_row(
        row,
        params=AtomAnchoredQECParams(
            candidate_scope="top20",
            selection_policy="primary_secondary",
            min_chain_steps=0,
            max_chain_steps=0,
        ),
    )

    assert trace["selector_name"] == "aa_qec_full_atom_facts_abc_primary_secondary_dynamic_qd_prefer_top20"
    assert len(trace["selector_ordered_indices"]) > 10
    assert trace["chain_diagnostics"]["fallback_step_count"] == 0
    assert trace["chain_diagnostics"]["source_selected_count"] == 2
    assert not set(trace["selector_ordered_indices"]).issubset(set(row["selected_indices"]))


def _trace_row(*, selected_indices: list[int]) -> dict:
    return {
        "event_id": "event-1",
        "claim": "Original claim.",
        "gold_label": "true",
        "selector_name": "v0_7_budgeted_marginal_chain_adaptive5_10",
        "graph_version": "evidence_chain_graph_v0_7",
        "fingerprint": "432dfc970e75",
        "candidate_pool_metadata": {
            "chunk_mmr_fingerprint": "432dfc970e75",
            "selector_name": "v0_7_budgeted_marginal_chain_adaptive5_10",
        },
        "candidate_pool": [
            _candidate(0, "E01", atoms=["A1"], question="Did atom one happen?", rank=2),
            _candidate(1, "E02", atoms=["A2"], question="Did atom two happen?", rank=1),
            _candidate(2, "E03", atoms=[]),
            _candidate(3, "E04", atoms=["A1"], question="Did atom one also happen?", rank=3),
        ],
        "candidate_scores": [
            {"candidate_idx": 0, "hybrid_score": 0.90},
            {"candidate_idx": 1, "hybrid_score": 0.80},
            {"candidate_idx": 2, "hybrid_score": 0.70},
            {"candidate_idx": 3, "hybrid_score": 0.60},
        ],
        "selector_ordered_indices": selected_indices,
        "selected_indices": selected_indices,
        "oracle_ordered_indices": [0, 1],
        "claim_atoms": [
            {"atom_id": "A1", "text": "Atom one", "importance": 1.0},
            {"atom_id": "A2", "text": "Atom two", "importance": 1.0},
        ],
    }


def _stage2_trace_row(*, selected_indices: list[int]) -> dict:
    row = _trace_row(selected_indices=selected_indices)
    row["selector_name"] = "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10"
    row["fingerprint"] = "d4cbf7c18126"
    row["candidate_pool_metadata"] = {
        "chunk_mmr_fingerprint": "d4cbf7c18126",
        "selector_name": "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10",
        "graph_version": "evidence_chain_graph_v0_7",
        "adaptive_policy": "budgeted_marginal_v0_7",
    }
    row["candidate_pool"] = [
        _candidate(
            0,
            "E01",
            atoms=["A1"],
            question="Did atom one happen?",
            relation="support",
            directness="direct",
            confidence=0.6,
            quality=0.50,
            base_score=0.80,
        ),
        _candidate(
            1,
            "E02",
            atoms=["A1"],
            question="Did atom one happen?",
            relation="support",
            directness="direct",
            confidence=0.95,
            quality=0.95,
            base_score=0.90,
        ),
        _candidate(
            2,
            "E03",
            atoms=["A1"],
            question="Could atom one be contradicted?",
            relation="refute",
            directness="partial",
            confidence=0.80,
            quality=0.70,
            base_score=0.70,
        ),
        _candidate(
            3,
            "E04",
            atoms=["A2"],
            question="Did atom two happen?",
            relation="support",
            directness="direct",
            confidence=0.90,
            quality=0.90,
            base_score=0.85,
        ),
        _candidate(
            4,
            "E05",
            atoms=["A2"],
            question="What qualifies atom two?",
            relation="qualify",
            directness="partial",
            confidence=0.75,
            quality=0.68,
            base_score=0.65,
        ),
        _candidate(5, "E06", atoms=[]),
        _candidate(
            6,
            "E07",
            atoms=["A3"],
            question="Unselected evidence should stay out?",
            relation="support",
            directness="direct",
            confidence=0.99,
            quality=0.99,
            base_score=0.99,
        ),
    ]
    row["claim_atoms"] = [
        {"atom_id": "A1", "text": "Atom one", "importance": 1.0},
        {"atom_id": "A2", "text": "Atom two", "importance": 1.0},
        {"atom_id": "A3", "text": "Atom three", "importance": 1.0},
    ]
    row["candidate_scores"] = [
        {"candidate_idx": candidate["candidate_idx"], "hybrid_score": candidate["hybrid_score"]}
        for candidate in row["candidate_pool"]
    ]
    return row


def _add_secondary_pairs(row: dict, *, atom_start: int, n_atoms: int) -> None:
    start_idx = len(row["candidate_pool"])
    for offset in range(n_atoms):
        atom_num = atom_start + offset
        atom_id = f"A{atom_num}"
        row["claim_atoms"].append({"atom_id": atom_id, "text": f"Atom {atom_num}", "importance": 1.0})
        primary_idx = start_idx + offset * 2
        secondary_idx = primary_idx + 1
        row["candidate_pool"].append(
            _candidate(
                primary_idx,
                f"E{primary_idx + 1:02d}",
                atoms=[atom_id],
                question=f"Did atom {atom_num} happen?",
                relation="support",
                directness="direct",
                confidence=0.90,
                quality=0.90,
                base_score=0.80,
            )
        )
        row["candidate_pool"].append(
            _candidate(
                secondary_idx,
                f"E{secondary_idx + 1:02d}",
                atoms=[atom_id],
                question=f"Could atom {atom_num} be contradicted?",
                relation="refute",
                directness="partial",
                confidence=0.80,
                quality=0.70,
                base_score=0.70,
            )
        )
    row["candidate_scores"] = [
        {"candidate_idx": candidate["candidate_idx"], "hybrid_score": candidate["hybrid_score"]}
        for candidate in row["candidate_pool"]
    ]


def _candidate(
    candidate_idx: int,
    evidence_id: str,
    *,
    atoms: list[str],
    question: str = "",
    rank: int = 1,
    relation: str | None = None,
    directness: str | None = None,
    confidence: float = 0.8,
    quality: float = 0.7,
    base_score: float = 0.6,
) -> dict:
    routes = []
    if question:
        routes.append(
            {
                "question_id": f"q{candidate_idx}",
                "question": question,
                "rank": rank,
                "hybrid_score": 0.5,
            }
        )
    return {
        "candidate_idx": candidate_idx,
        "candidate_uid": f"uid-{evidence_id}",
        "candidate_key": f"{evidence_id} key",
        "evidence_id": evidence_id,
        "text": f"{evidence_id} evidence text.",
        "covered_atom_ids": atoms,
        "map_relation": relation or ("support" if atoms else "background"),
        "map_directness": directness or ("direct" if atoms else "context"),
        "map_confidence": confidence,
        "evidence_map_quality_score": quality,
        "evidence_map_base_score": base_score,
        "hybrid_score": base_score,
        "qd_question_routes": routes,
        "from_qd": bool(routes),
        "qd_max_question_hybrid": 0.5 if routes else 0.0,
        "duplicate_group": "",
        "source_group": "report:1",
    }
