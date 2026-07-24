from __future__ import annotations

from copy import deepcopy

from fact_checking.selectors.mrec_learned_marginal import (
    LearnedMarginalWeights,
    REWARD_WEIGHT_SCHEMA_VERSION,
    SUPERVISION_MODE_LEGACY_HYBRID,
    SUPERVISION_MODE_STRUCTURE_ONLY,
    build_winner_vs_rest_preferences,
    evaluate_learned_marginal_reward_weights,
    evaluate_learned_marginal_proxy_weights,
    extract_marginal_features,
    initial_learned_marginal_weights,
    initial_neutral_learned_marginal_weights,
    learned_marginal_weight_fingerprint,
    rank_candidates_by_proxy,
    score_marginal_features,
    train_learned_marginal_proxy_weights,
    train_learned_marginal_reward_weights,
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


def test_structure_only_is_invariant_to_oracle_gold_teacher_and_utility_poison() -> None:
    clean = [
        {
            "event_id": "case-structure",
            "claim_atoms": [{"atom_id": "A1", "text": "The city approved the project."}],
            "candidate_pool": [
                _candidate("E-noise", key="doc:noise", relation="irrelevant", atoms=[], retrieval=0.99),
                _candidate("E-direct", key="doc:direct", relation="support", atoms=["A1"], retrieval=0.1),
                _candidate("E-partial", key="doc:partial", relation="support", atoms=["A1"], directness="partial"),
            ],
        }
    ]
    poisoned = deepcopy(clean)
    poisoned[0].update(
        {
            "gold_label": "POISON",
            "label": "POISON",
            "oracle_ordered_keys": ["doc:noise"],
            "oracle_ordered_indices": [0],
            "teacher_margin": 999.0,
            "utility_scores": {"doc:noise": 999.0},
            "verifier_reward": 999.0,
        }
    )
    for candidate in poisoned[0]["candidate_pool"]:
        candidate.update(
            {
                "oracle_selected": candidate["candidate_key"] == "doc:noise",
                "gold_label": "POISON",
                "teacher_score": 999.0,
                "utility": 999.0,
                "delta_margin": 999.0,
                "verifier_reward": 999.0,
            }
        )

    clean_weights, clean_metrics = train_learned_marginal_proxy_weights(
        clean,
        epochs=5,
        learning_rate=0.1,
        rollout_steps=2,
        supervision_mode=SUPERVISION_MODE_STRUCTURE_ONLY,
    )
    poison_weights, poison_metrics = train_learned_marginal_proxy_weights(
        poisoned,
        epochs=5,
        learning_rate=0.1,
        rollout_steps=2,
        supervision_mode=SUPERVISION_MODE_STRUCTURE_ONLY,
    )

    assert clean_metrics == poison_metrics
    assert clean_metrics["supervision_fingerprint"] == poison_metrics["supervision_fingerprint"]
    assert clean_metrics["oracle_read_row_count"] == 0
    assert clean_metrics["gold_label_read_count"] == 0
    assert clean_metrics["teacher_read_count"] == 0
    assert clean_metrics["utility_read_count"] == 0
    assert clean_metrics["reward_read_count"] == 0
    assert clean_metrics["oracle_preference_step_count"] == 0
    assert clean_metrics["structure_preference_step_count"] > 0
    assert clean_weights.to_json_dict() == poison_weights.to_json_dict()
    assert learned_marginal_weight_fingerprint(clean_weights) == learned_marginal_weight_fingerprint(poison_weights)


def test_structure_only_uses_neutral_initialization_and_has_nonempty_validation_metrics() -> None:
    neutral = initial_neutral_learned_marginal_weights()
    assert set(neutral.feature_weights.values()) == {1.0}
    assert neutral.cost_weight == 1.0

    rows = [
        {
            "claim_atoms": [{"atom_id": "A1"}],
            "candidate_pool": [
                _candidate("E-noise", relation="irrelevant", atoms=[], retrieval=0.9),
                _candidate("E-direct", relation="support", atoms=["A1"], retrieval=0.1),
            ],
        }
    ]
    weights, _ = train_learned_marginal_proxy_weights(
        rows,
        epochs=3,
        supervision_mode=SUPERVISION_MODE_STRUCTURE_ONLY,
    )
    metrics = evaluate_learned_marginal_proxy_weights(
        rows,
        weights,
        supervision_mode=SUPERVISION_MODE_STRUCTURE_ONLY,
    )

    assert metrics["scored_row_count"] == 1
    assert metrics["scored_pair_count"] == 1
    assert metrics["pair_accuracy"] == 1.0
    assert metrics["oracle_read_row_count"] == 0


def test_default_proxy_training_matches_explicit_legacy_hybrid_mode() -> None:
    rows = [
        {
            "claim_atoms": [{"atom_id": "A1"}],
            "oracle_ordered_keys": ["doc:noise"],
            "candidate_pool": [
                _candidate("E-direct", key="doc:direct", relation="support", atoms=["A1"]),
                _candidate("E-noise", key="doc:noise", relation="irrelevant", atoms=[]),
            ],
        }
    ]

    default_weights, default_metrics = train_learned_marginal_proxy_weights(rows, epochs=3)
    legacy_weights, legacy_metrics = train_learned_marginal_proxy_weights(
        rows,
        epochs=3,
        supervision_mode=SUPERVISION_MODE_LEGACY_HYBRID,
    )

    assert default_weights.to_json_dict() == legacy_weights.to_json_dict()
    assert default_metrics == legacy_metrics
    assert default_metrics["oracle_preference_step_count"] > 0


def test_winner_vs_rest_preferences_are_supervision_agnostic() -> None:
    assert build_winner_vs_rest_preferences([2, 0, 1]) == [(2, 0), (2, 1)]
    assert build_winner_vs_rest_preferences([]) == []


def test_initial_learned_marginal_weights_are_interpretable_positive_weights() -> None:
    weights = initial_learned_marginal_weights()

    assert weights.feature_weights["resolution_delta"] > weights.feature_weights["retrieval_score"]
    assert weights.cost_weight > 0.0


def test_reward_weights_include_signed_bias_in_score() -> None:
    weights = LearnedMarginalWeights(
        feature_weights={},
        cost_weight=0.0,
        schema_version=REWARD_WEIGHT_SCHEMA_VERSION,
        bias=-0.75,
    )

    assert score_marginal_features({}, weights) == -0.75


def test_train_learned_marginal_reward_weights_scores_positive_delta_above_negative() -> None:
    rows = [
        {
            "event_id": "case-1",
            "step": 0,
            "candidate_idx": 0,
            "delta_margin": -0.4,
            "mrec_features": {
                "resolution_delta": 0.0,
                "entropy_reduction": 0.0,
                "new_atom_coverage": 0.0,
                "retrieval_score": 0.9,
                "cost_ratio": 0.1,
            },
        },
        {
            "event_id": "case-1",
            "step": 0,
            "candidate_idx": 1,
            "delta_margin": 0.8,
            "mrec_features": {
                "resolution_delta": 1.0,
                "entropy_reduction": 1.0,
                "new_atom_coverage": 1.0,
                "retrieval_score": 0.1,
                "cost_ratio": 0.1,
            },
        },
    ]

    weights, metrics = train_learned_marginal_reward_weights(rows, epochs=30, learning_rate=0.1, prior_weight=0.0)
    eval_metrics = evaluate_learned_marginal_reward_weights(rows, weights)

    assert metrics["pair_count"] > 0
    assert eval_metrics["pair_accuracy"] == 1.0
    assert eval_metrics["step_top1_match"] == 1.0
    assert score_marginal_features(rows[1]["mrec_features"], weights) > score_marginal_features(rows[0]["mrec_features"], weights)


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
