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

    summary = summarize_atom_anchored_qec_traces([trace])

    assert summary["n_rows"] == 1
    assert summary["selector_names"] == {"aa_qec_view_keep_all_qd_prefer_selected_min5_10": 1}
    assert summary["chain_steps"]["mean"] == 3.0
    assert summary["role_counts"] == {"primary": 2, "fallback": 1}
    assert summary["cue_source_counts"] == {"qd_question": 2, "fallback": 1}


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


def _candidate(
    candidate_idx: int,
    evidence_id: str,
    *,
    atoms: list[str],
    question: str = "",
    rank: int = 1,
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
        "map_relation": "support" if atoms else "background",
        "map_directness": "direct" if atoms else "context",
        "map_confidence": 0.8,
        "evidence_map_quality_score": 0.7,
        "evidence_map_base_score": 0.6,
        "qd_question_routes": routes,
        "from_qd": bool(routes),
        "qd_max_question_hybrid": 0.5 if routes else 0.0,
        "duplicate_group": "",
        "source_group": "report:1",
    }
