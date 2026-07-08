from __future__ import annotations

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample
from fact_checking.selectors.atom_retrieval_union import (
    AtomUnionSelectionParams,
    rank_atom_union_source_score_candidates,
    select_atom_union_rules,
)
from fact_checking.selectors.selector_mechanism_ablation import (
    SELECTOR_MECH_S0_NO_EVIDENCE,
    SELECTOR_MECH_S1_CLAIM_POOL_RANDOM_TOP5,
    SELECTOR_MECH_S2_CLAIM_POOL_HYBRID_TOP5,
    SELECTOR_MECH_S3_CLAIM_POOL_HYBRID_MMR_TOP5,
    SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED,
    SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
    SELECTOR_MECH_S4B_ATOM_ROUTE_ONLY_SOURCE_SCORE_TOP5,
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
    union_row = _union_row()
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


def test_atom_union_source_score_ordered_keeps_full_union_order_and_matches_top5_prefix() -> None:
    union_row = _union_row(
        extra_candidates=[
            {
                "text": "weak atom",
                "canonical_text": "weak atom",
                "from_baseline": False,
                "from_atom_route": True,
                "baseline_rank": None,
                "atom_pool_rank": 2,
                "atom_rrf_score": 0.01,
                "atom_route_hit_count": 1,
                "atom_max_route_hybrid": 0.1,
                "union_pool_rank": 3,
                "chunk_sent_indices": [2],
            },
            {
                "text": "second baseline",
                "canonical_text": "second baseline",
                "from_baseline": True,
                "from_atom_route": False,
                "baseline_rank": 2,
                "atom_pool_rank": None,
                "atom_rrf_score": 0.0,
                "atom_route_hit_count": 0,
                "atom_max_route_hybrid": 0.0,
                "union_pool_rank": 4,
                "chunk_sent_indices": [3],
            },
            {
                "text": "third baseline",
                "canonical_text": "third baseline",
                "from_baseline": True,
                "from_atom_route": False,
                "baseline_rank": 3,
                "atom_pool_rank": None,
                "atom_rrf_score": 0.0,
                "atom_route_hit_count": 0,
                "atom_max_route_hybrid": 0.0,
                "union_pool_rank": 5,
                "chunk_sent_indices": [4],
            },
            {
                "text": "tiny atom",
                "canonical_text": "tiny atom",
                "from_baseline": False,
                "from_atom_route": True,
                "baseline_rank": None,
                "atom_pool_rank": 3,
                "atom_rrf_score": 0.005,
                "atom_route_hit_count": 0,
                "atom_max_route_hybrid": 0.0,
                "union_pool_rank": 6,
                "chunk_sent_indices": [5],
            },
        ]
    )

    full_order_trace = build_selector_mechanism_trace_row(
        claim_pool_row=None,
        union_row=union_row,
        selector_name=SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_ORDERED,
        params=SelectorMechanismParams(top_k=5),
        chunk_mmr_fingerprint="fp",
    )
    top5_trace = build_selector_mechanism_trace_row(
        claim_pool_row=None,
        union_row=union_row,
        selector_name=SELECTOR_MECH_S4_ATOM_UNION_SOURCE_SCORE_TOP5,
        params=SelectorMechanismParams(top_k=5),
        chunk_mmr_fingerprint="fp",
    )
    ranked = rank_atom_union_source_score_candidates(
        union_row["candidates"],
        params=AtomUnionSelectionParams(selector_top_k=len(union_row["candidates"])),
    )

    assert len(full_order_trace["selector_ordered_indices"]) == len(union_row["candidates"])
    assert full_order_trace["selector_ordered_indices"][:5] == top5_trace["selector_ordered_indices"]
    assert [c["text"] for c in full_order_trace["selected_candidates"]] == [c["text"] for c in ranked]
    assert full_order_trace["adaptive_policy"] == "source_score_ordered"
    assert full_order_trace["candidate_pool_metadata"]["adaptive_policy"] == "source_score_ordered"


def test_atom_route_only_drops_baseline_only_candidates() -> None:
    """S4b keeps only atom-route candidates (including dual-source ones) and
    drops candidates that come solely from the baseline claim pool."""
    union_row = _union_row(
        extra_candidates=[
            {
                "text": "second baseline",
                "canonical_text": "second baseline",
                "from_baseline": True,
                "from_atom_route": False,
                "baseline_rank": 2,
                "atom_pool_rank": None,
                "atom_rrf_score": 0.0,
                "atom_route_hit_count": 0,
                "atom_max_route_hybrid": 0.0,
                "union_pool_rank": 3,
                "chunk_sent_indices": [3],
            },
            {
                "text": "dual source",
                "canonical_text": "dual source",
                "from_baseline": True,
                "from_atom_route": True,
                "baseline_rank": 3,
                "atom_pool_rank": 2,
                "atom_rrf_score": 0.10,
                "atom_route_hit_count": 1,
                "atom_max_route_hybrid": 0.5,
                "union_pool_rank": 4,
                "chunk_sent_indices": [4],
            },
        ]
    )
    # union_row now has: baseline-only, atom-only, baseline-only, dual-source
    # S4b should drop the two baseline-only candidates, keeping atom-only + dual.
    trace = build_selector_mechanism_trace_row(
        claim_pool_row=None,
        union_row=union_row,
        selector_name=SELECTOR_MECH_S4B_ATOM_ROUTE_ONLY_SOURCE_SCORE_TOP5,
        params=SelectorMechanismParams(top_k=5),
        chunk_mmr_fingerprint="fp",
    )

    pool_texts = [c["text"] for c in trace["candidate_pool"]]
    assert "baseline" not in pool_texts, "pure baseline candidate should be dropped"
    assert "second baseline" not in pool_texts, "pure baseline candidate should be dropped"
    assert "atom" in pool_texts, "atom-only candidate should be kept"
    assert "dual source" in pool_texts, "dual-source candidate should be kept (it is an atom-route hit)"

    # The source-score on the filtered pool should not include any baseline bonus,
    # so ordering is driven purely by atom components.
    for candidate in trace["candidate_pool"]:
        assert candidate.get("from_atom_route") is True


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


def _union_row(extra_candidates: list[dict[str, object]] | None = None) -> dict[str, object]:
    candidates = [
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
    ]
    candidates.extend(extra_candidates or [])
    return {
        "event_id": "event0",
        "claim": "claim",
        "label": "true",
        "gold_label": "true",
        "claim_atoms": [{"atom_id": "A1", "text": "atom"}],
        "candidates": candidates,
    }


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
