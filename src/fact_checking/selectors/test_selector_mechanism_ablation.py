from __future__ import annotations

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample
from fact_checking.selectors.atom_retrieval_union import AtomUnionSelectionParams, select_atom_union_rules
from fact_checking.selectors.selector_mechanism_ablation import (
    SELECTOR_MECH_S0_NO_EVIDENCE,
    SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5,
    SELECTOR_MECH_S2_CLAIM_POOL_HYBRID_TOP5,
    SELECTOR_MECH_S3_CLAIM_POOL_HYBRID_MMR_TOP5,
    SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
    SelectorMechanismParams,
    build_claim_candidate_pool_row,
    build_selector_mechanism_trace_row,
)


def test_claim_pool_random_top5_is_seeded_and_unique() -> None:
    claim_pool = build_claim_candidate_pool_row(_sample(20), params=SelectorMechanismParams(claim_pool_top_n=20))

    first = build_selector_mechanism_trace_row(
        claim_pool_row=claim_pool,
        union_row=None,
        selector_name=SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5,
        params=SelectorMechanismParams(top_k=5, random_seed=13),
        chunk_mmr_fingerprint="fp",
    )
    second = build_selector_mechanism_trace_row(
        claim_pool_row=claim_pool,
        union_row=None,
        selector_name=SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5,
        params=SelectorMechanismParams(top_k=5, random_seed=13),
        chunk_mmr_fingerprint="fp",
    )

    assert first["selected_indices"] == second["selected_indices"]
    assert len(first["selected_indices"]) == 5
    assert len(set(first["selected_indices"])) == 5
    assert first["selected_indices"] != [0, 1, 2, 3, 4]


def test_claim_pool_hybrid_top5_uses_hybrid_descending_order() -> None:
    claim_pool = build_claim_candidate_pool_row(_sample(8), params=SelectorMechanismParams(claim_pool_top_n=8))

    trace = build_selector_mechanism_trace_row(
        claim_pool_row=claim_pool,
        union_row=None,
        selector_name=SELECTOR_MECH_S2_CLAIM_POOL_HYBRID_TOP5,
        params=SelectorMechanismParams(top_k=5),
        chunk_mmr_fingerprint="fp",
    )

    selected_scores = [
        trace["candidate_pool"][idx]["hybrid_score"]
        for idx in trace["selected_indices"]
    ]
    assert selected_scores == sorted(selected_scores, reverse=True)
    assert trace["selected_indices"] == [0, 1, 2, 3, 4]


def test_claim_pool_hybrid_mmr_selects_top5_from_same_pool() -> None:
    claim_pool = build_claim_candidate_pool_row(_sample(10), params=SelectorMechanismParams(claim_pool_top_n=10))

    trace = build_selector_mechanism_trace_row(
        claim_pool_row=claim_pool,
        union_row=None,
        selector_name=SELECTOR_MECH_S3_CLAIM_POOL_HYBRID_MMR_TOP5,
        params=SelectorMechanismParams(top_k=5, merge_mmr_lambda=0.70),
        chunk_mmr_fingerprint="fp",
    )

    assert len(trace["selected_indices"]) == 5
    assert len(set(trace["selected_indices"])) == 5
    assert all(0 <= idx < len(claim_pool["candidates"]) for idx in trace["selected_indices"])
    assert all("selector_selected_step" in trace["candidate_scores"][idx] for idx in trace["selected_indices"])


def test_atom_union_source_score_matches_existing_union_rule() -> None:
    union_row = {
        "event_id": "event0",
        "claim": "claim",
        "label": "true",
        "gold_label": "true",
        "claim_atoms": [{"atom_id": "A1", "text": "atom"}],
        "candidates": [
            {
                "text": "baseline",
                "canonical_text": "baseline",
                "from_baseline": True,
                "from_atom_route": False,
                "baseline_rank": 1,
                "atom_pool_rank": None,
                "atom_rrf_score": 0.0,
                "atom_route_hit_count": 0,
                "atom_max_route_hybrid": 0.0,
                "union_pool_rank": 1,
                "chunk_sent_indices": [0],
            },
            {
                "text": "atom",
                "canonical_text": "atom",
                "from_baseline": False,
                "from_atom_route": True,
                "baseline_rank": None,
                "atom_pool_rank": 1,
                "atom_rrf_score": 0.20,
                "atom_route_hit_count": 2,
                "atom_max_route_hybrid": 0.9,
                "union_pool_rank": 2,
                "chunk_sent_indices": [1],
            },
        ],
    }
    params = SelectorMechanismParams(top_k=2)

    trace = build_selector_mechanism_trace_row(
        claim_pool_row=None,
        union_row=union_row,
        selector_name=SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
        params=params,
        chunk_mmr_fingerprint="fp",
    )
    expected = select_atom_union_rules(
        union_row,
        params=AtomUnionSelectionParams(selector_top_k=2),
    )["atom_union_source_score_top5"]

    assert [c["text"] for c in trace["selected_candidates"]] == [c["text"] for c in expected]
    assert trace["selected_indices"] == [1, 0]


def test_no_evidence_trace_has_empty_pool_and_selected_indices() -> None:
    trace = build_selector_mechanism_trace_row(
        claim_pool_row={"event_id": "event0", "claim": "claim", "label": "true", "gold_label": "true", "candidates": []},
        union_row=None,
        selector_name=SELECTOR_MECH_S0_NO_EVIDENCE,
        params=SelectorMechanismParams(top_k=5),
        chunk_mmr_fingerprint="fp",
    )

    assert trace["candidate_pool"] == []
    assert trace["selected_indices"] == []
    assert trace["selector_ordered_indices"] == []


def _sample(n: int) -> ChunkMMRSample:
    candidates = [
        {
            "text": f"Evidence {idx} claim token",
            "chunk_sent_indices": [idx],
            "report_id": idx,
            "sent_idx": idx,
        }
        for idx in range(n)
    ]
    vectors = np.asarray([[float(n - idx), float(idx % 3)] for idx in range(n)], dtype=np.float32)
    return ChunkMMRSample(
        event_id="event0",
        claim="claim token",
        label="true",
        explain="",
        candidates=candidates,
        chunk_emb=vectors,
        claim_emb=np.asarray([1.0, 0.0], dtype=np.float32),
    )
