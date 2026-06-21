from __future__ import annotations

from fact_checking.selectors.map_selector_ablation import (
    MAP_SELECTOR_GRAPH_VERSION,
    MAP_SELECTOR_S0_RETRIEVAL_TOP5,
    MAP_SELECTOR_S1_MMR_POOL_TOP5,
    MAP_SELECTOR_S2_MAP_QUALITY_TOP5,
    MapSelectorAblationParams,
    build_map_selector_ablation_trace,
)


MAP_SELECTOR_S3_WEIGHTED_SET_COVER_TOP5 = "map_selector_s3_weighted_set_cover_top5"
MAP_SELECTOR_S4_MINIMAL_EVIDENCE_GROUP_TOP5 = "map_selector_s4_minimal_evidence_group_top5"
MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5 = "map_selector_s5_fixed_budget_marginal_greedy_top5"


def test_s0_retrieval_prefers_high_retrieval_even_when_map_quality_is_low() -> None:
    row = _row(
        [
            _candidate("low-map-high-retrieval", hybrid=0.95, map_quality=0.05, base=0.60, union_rank=2),
            _candidate("high-map-low-retrieval", hybrid=0.20, map_quality=0.99, base=0.90, union_rank=1),
        ]
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S0_RETRIEVAL_TOP5, top_k=1),
    )

    assert trace["selector_name"] == MAP_SELECTOR_S0_RETRIEVAL_TOP5
    assert trace["selected_keys"] == ["low-map-high-retrieval"]
    assert trace["candidate_scores"][0]["retrieval_score"] == 0.95


def test_s1_reuses_existing_mmr_rank_without_map_quality_resort() -> None:
    row = _row(
        [
            _candidate("mmr-second-high-map", hybrid=0.95, map_quality=1.00, base=0.90, mmr_rank=2, union_rank=2),
            _candidate("mmr-first-low-map", hybrid=0.20, map_quality=0.01, base=0.10, mmr_rank=1, union_rank=5),
        ]
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S1_MMR_POOL_TOP5, top_k=1),
    )

    assert trace["selector_name"] == MAP_SELECTOR_S1_MMR_POOL_TOP5
    assert trace["selected_keys"] == ["mmr-first-low-map"]
    assert trace["candidate_scores"][1]["mmr_rank"] == 1


def test_s2_map_quality_prefers_high_map_quality_over_retrieval() -> None:
    row = _row(
        [
            _candidate("retrieval-only", hybrid=0.99, map_quality=0.10, base=0.99, union_rank=1),
            _candidate("map-quality", hybrid=0.10, map_quality=0.80, base=0.10, union_rank=2),
        ]
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S2_MAP_QUALITY_TOP5, top_k=1),
    )

    assert trace["selector_name"] == MAP_SELECTOR_S2_MAP_QUALITY_TOP5
    assert trace["selected_keys"] == ["map-quality"]
    assert trace["candidate_scores"][1]["evidence_map_quality_score"] == 0.80


def test_trace_schema_is_stage_sources_compatible_and_uses_pool_indices() -> None:
    row = _row(
        [
            _candidate("c0", hybrid=0.90, map_quality=0.20, base=0.90, union_rank=1),
            _candidate("c1", hybrid=0.80, map_quality=0.70, base=0.80, union_rank=2),
            _candidate("c2", hybrid=0.70, map_quality=0.60, base=0.70, union_rank=3),
        ],
        oracle_ordered_keys=["c2", "c0"],
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(
            selector_name=MAP_SELECTOR_S2_MAP_QUALITY_TOP5,
            top_k=2,
            chunk_mmr_fingerprint="fp123",
        ),
    )

    assert trace["graph_version"] == MAP_SELECTOR_GRAPH_VERSION
    assert trace["fingerprint"] == "fp123"
    assert trace["candidate_pool_metadata"]["chunk_mmr_fingerprint"] == "fp123"
    assert [item["candidate_idx"] for item in trace["candidate_pool"]] == [0, 1, 2]
    assert [item["candidate_idx"] for item in trace["candidate_scores"]] == [0, 1, 2]
    assert trace["selector_ordered_indices"] == [1, 2]
    assert trace["selected_indices"] == [1, 2]
    assert trace["oracle_ordered_indices"] == [2, 0]
    assert all(candidate["chunk_sent_indices"] for candidate in trace["candidate_pool"])
    assert len(trace["selected_candidates"]) == 2


def test_s3_weighted_set_cover_prefers_new_high_weight_atom_over_map_quality() -> None:
    row = _row(
        [
            _candidate(
                "quality-a1",
                hybrid=0.95,
                map_quality=0.99,
                base=0.95,
                union_rank=1,
                covered_atom_ids=["A1"],
            ),
            _candidate(
                "covers-high-weight-a2",
                hybrid=0.20,
                map_quality=0.10,
                base=0.20,
                union_rank=2,
                covered_atom_ids=["A2"],
            ),
            _candidate(
                "duplicate-a1",
                hybrid=0.90,
                map_quality=0.80,
                base=0.90,
                union_rank=3,
                covered_atom_ids=["A1"],
            ),
        ],
        claim_atoms=[
            {"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0},
            {"atom_id": "A2", "text": "The approval was official.", "importance": 5.0},
        ],
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S3_WEIGHTED_SET_COVER_TOP5, top_k=2),
    )

    assert trace["selected_keys"] == ["covers-high-weight-a2", "quality-a1"]
    assert trace["selection_steps"][0]["weighted_new_atom_gain"] == 5.0
    assert trace["selection_steps"][0]["covered_new_atom_ids"] == ["A2"]


def test_s3_weighted_set_cover_fills_remaining_budget_by_map_quality_order() -> None:
    row = _row(
        [
            _candidate("covers-a1", hybrid=0.30, map_quality=0.20, base=0.30, union_rank=3, covered_atom_ids=["A1"]),
            _candidate("filler-high-quality", hybrid=0.20, map_quality=0.95, base=0.20, union_rank=1, covered_atom_ids=[]),
            _candidate("filler-mid-quality", hybrid=0.90, map_quality=0.80, base=0.90, union_rank=2, covered_atom_ids=[]),
        ],
        claim_atoms=[{"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}],
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S3_WEIGHTED_SET_COVER_TOP5, top_k=3),
    )

    assert trace["selected_keys"] == ["covers-a1", "filler-high-quality", "filler-mid-quality"]
    assert trace["fixed_budget_fill_indices"] == [1, 2]


def test_s4_minimal_evidence_group_records_core_and_fixed_budget_fill() -> None:
    row = _row(
        [
            _candidate("covers-a1", hybrid=0.30, map_quality=0.40, base=0.30, union_rank=3, covered_atom_ids=["A1"]),
            _candidate("covers-a2", hybrid=0.20, map_quality=0.30, base=0.20, union_rank=4, covered_atom_ids=["A2"]),
            _candidate("filler-high-quality", hybrid=0.10, map_quality=0.95, base=0.10, union_rank=1, covered_atom_ids=[]),
            _candidate("filler-mid-quality", hybrid=0.90, map_quality=0.80, base=0.90, union_rank=2, covered_atom_ids=[]),
        ],
        claim_atoms=[
            {"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0},
            {"atom_id": "A2", "text": "The approval was official.", "importance": 1.0},
        ],
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S4_MINIMAL_EVIDENCE_GROUP_TOP5, top_k=4),
    )

    assert trace["selected_keys"] == ["covers-a1", "covers-a2", "filler-high-quality", "filler-mid-quality"]
    assert trace["minimal_group_indices"] == [0, 1]
    assert trace["minimal_group_size"] == 2
    assert trace["fixed_budget_fill_indices"] == [2, 3]
    assert len(trace["selected_indices"]) == 4


def test_s5_fixed_budget_marginal_greedy_selects_exact_budget_and_keeps_steps() -> None:
    row = _row(
        [
            _candidate(f"c{i}", hybrid=1.0 - i * 0.05, map_quality=0.8 - i * 0.05, base=0.7 - i * 0.03, union_rank=i + 1)
            for i in range(6)
        ],
        claim_atoms=[
            {"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0},
            {"atom_id": "A2", "text": "The approval was official.", "importance": 1.0},
        ],
    )

    trace = build_map_selector_ablation_trace(
        row,
        params=MapSelectorAblationParams(selector_name=MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5, top_k=5),
    )

    assert trace["selector_name"] == MAP_SELECTOR_S5_FIXED_BUDGET_MARGINAL_GREEDY_TOP5
    assert trace["adaptive_policy"] == "fixed_budget_marginal_greedy"
    assert len(trace["selected_indices"]) == 5
    assert len(trace["selection_steps"]) == 5
    assert all("marginal_gain" in step for step in trace["selection_steps"])


def _row(
    candidates: list[dict[str, object]],
    *,
    oracle_ordered_keys: list[str] | None = None,
    claim_atoms: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    claim_atoms = claim_atoms or [
        {"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}
    ]
    return {
        "event_id": "evt-1",
        "claim": "The city approved the project.",
        "gold_label": "true",
        "oracle_ordered_keys": oracle_ordered_keys or [],
        "evidence_map": {"claim_atoms": claim_atoms},
        "candidates": candidates,
    }


def _candidate(
    key: str,
    *,
    hybrid: float,
    map_quality: float,
    base: float,
    union_rank: int,
    mmr_rank: int | None = None,
    covered_atom_ids: list[str] | None = None,
) -> dict[str, object]:
    candidate = {
        "candidate_key": key,
        "candidate_uid": key,
        "text": f"Evidence sentence for {key}.",
        "hybrid_score": hybrid,
        "evidence_map_quality_score": map_quality,
        "evidence_map_base_score": base,
        "union_pool_rank": union_rank,
        "source_group": "report:1",
        "source_domain": "example.org",
        "chunk_sent_indices": [0],
        "covered_atom_ids": covered_atom_ids if covered_atom_ids is not None else ["A1"],
        "map_relation": "support",
        "map_directness": "direct",
        "map_confidence": 0.9,
    }
    if mmr_rank is not None:
        candidate["mmr_rank"] = mmr_rank
    return candidate
