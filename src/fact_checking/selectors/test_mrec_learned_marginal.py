from __future__ import annotations

from fact_checking.selectors.mrec_learned_marginal import (
    extract_marginal_features,
    initial_learned_marginal_weights,
    rank_candidates_by_proxy,
    score_marginal_features,
    train_learned_marginal_proxy_weights,
)


def test_extract_marginal_features_prefers_direct_unresolved_resolution() -> None:
    soft_state = {"A1": {"U": 1.0}}
    direct = _candidate("E-direct", relation="support", atoms=["A1"], directness="direct", retrieval=0.2)
    irrelevant = _candidate("E-noise", relation="irrelevant", atoms=["A1"], directness="none", retrieval=0.99)

    direct_features = extract_marginal_features(
        direct,
        selected_steps=[],
        soft_state=soft_state,
        token_budget=None,
        pool_max_token_cost=10,
    )
    noise_features = extract_marginal_features(
        irrelevant,
        selected_steps=[],
        soft_state=soft_state,
        token_budget=None,
        pool_max_token_cost=10,
    )

    assert direct_features["resolution_delta"] > noise_features["resolution_delta"]
    assert direct_features["entropy_reduction"] > noise_features["entropy_reduction"]


def test_extract_marginal_features_tracks_text_and_source_novelty() -> None:
    soft_state = {"A1": {"S": 1.0}}
    selected = [
        {
            "evidence_text": "The city council approved the project.",
            "duplicate_group": "dup-1",
            "source_group": "report-a",
            "relation": "support",
            "atom_id": "A1",
        }
    ]

    repeated = _candidate(
        "E-repeat",
        relation="support",
        atoms=["A1"],
        text="The city council approved the project.",
        duplicate_group="dup-1",
        source_group="report-a",
    )
    fresh_source = _candidate(
        "E-fresh",
        relation="support",
        atoms=["A1"],
        text="A separate report says the project was approved.",
        duplicate_group="dup-2",
        source_group="report-b",
    )

    repeated_features = extract_marginal_features(
        repeated,
        selected_steps=selected,
        soft_state=soft_state,
        token_budget=None,
        pool_max_token_cost=10,
    )
    fresh_features = extract_marginal_features(
        fresh_source,
        selected_steps=selected,
        soft_state=soft_state,
        token_budget=None,
        pool_max_token_cost=10,
    )

    assert repeated_features["text_novelty"] == 0.0
    assert repeated_features["source_novelty"] == 0.0
    assert fresh_features["text_novelty"] == 1.0
    assert fresh_features["source_novelty"] == 1.0


def test_rank_candidates_by_proxy_uses_oracle_then_lexicographic_signal() -> None:
    soft_state = {"A1": {"U": 1.0}}
    direct = _candidate("E-direct", key="doc:direct", relation="support", atoms=["A1"], retrieval=0.1)
    oracle = _candidate("E-oracle", key="doc:oracle", relation="irrelevant", atoms=[], retrieval=0.0)
    high_retrieval_noise = _candidate("E-noise", key="doc:noise", relation="irrelevant", atoms=[], retrieval=0.99)

    oracle_order = rank_candidates_by_proxy(
        [direct, oracle, high_retrieval_noise],
        selected_steps=[],
        soft_state=soft_state,
        oracle_ordered_keys=["doc:oracle"],
    )
    fallback_order = rank_candidates_by_proxy(
        [high_retrieval_noise, direct],
        selected_steps=[],
        soft_state=soft_state,
        oracle_ordered_keys=[],
    )

    assert oracle_order[0] == 1
    assert fallback_order[0] == 1


def test_train_learned_marginal_proxy_weights_scores_oracle_above_noise() -> None:
    rows = [
        {
            "event_id": "case-1",
            "claim_atoms": [{"atom_id": "A1", "text": "The city approved the project."}],
            "oracle_ordered_keys": ["doc:direct"],
            "candidate_pool": [
                _candidate("E-noise", key="doc:noise", relation="irrelevant", atoms=[], retrieval=0.99),
                _candidate("E-direct", key="doc:direct", relation="support", atoms=["A1"], retrieval=0.1),
            ],
        }
    ]

    weights, metrics = train_learned_marginal_proxy_weights(rows, epochs=20, learning_rate=0.2)
    soft_state = {"A1": {"U": 1.0}}
    noise_features = extract_marginal_features(
        rows[0]["candidate_pool"][0],
        selected_steps=[],
        soft_state=soft_state,
        token_budget=None,
        pool_max_token_cost=10,
    )
    direct_features = extract_marginal_features(
        rows[0]["candidate_pool"][1],
        selected_steps=[],
        soft_state=soft_state,
        token_budget=None,
        pool_max_token_cost=10,
    )

    assert metrics["pair_count"] > 0
    assert score_marginal_features(direct_features, weights) > score_marginal_features(noise_features, weights)


def test_initial_learned_marginal_weights_are_interpretable_positive_weights() -> None:
    weights = initial_learned_marginal_weights()

    assert weights.feature_weights["resolution_delta"] > weights.feature_weights["retrieval_score"]
    assert weights.cost_weight > 0.0


def _candidate(
    evidence_id: str,
    *,
    relation: str,
    atoms: list[str],
    key: str | None = None,
    text: str | None = None,
    directness: str = "direct",
    retrieval: float = 0.5,
    duplicate_group: str = "",
    source_group: str = "report-a",
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "candidate_uid": evidence_id,
        "candidate_key": key or evidence_id,
        "text": text or f"{evidence_id} evidence text.",
        "covered_atom_ids": atoms,
        "map_relation": relation,
        "map_directness": directness,
        "map_confidence": 0.8,
        "evidence_map_quality_score": 0.7,
        "hybrid_score": retrieval,
        "duplicate_group": duplicate_group,
        "source_group": source_group,
        "mrec_token_cost": 5,
    }
