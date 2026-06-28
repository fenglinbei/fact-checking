from __future__ import annotations

import argparse

from fact_checking.selectors.mrec_learned_marginal import (
    LearnedMarginalWeights,
    REWARD_WEIGHT_SCHEMA_VERSION,
    initial_learned_marginal_weights,
    save_learned_marginal_weights,
)
from fact_checking.selectors.minimal_resolving_chain import MRECSelectorParams
from fact_checking.selectors.mrec_schema import MREC_TRACE_VERSION
from scripts.phase5_selectors.build.build_mrec_traces import _build_trace, _manifest


def test_build_mrec_trace_preserves_source_metadata_and_compat_projection() -> None:
    row = {
        "event_id": "evt-1",
        "claim": "The city approved the project.",
        "gold_label": "true",
        "selector_name": "source_selector",
        "fingerprint": "abc123",
        "candidate_pool_metadata": {"chunk_mmr_fingerprint": "abc123"},
        "claim_atoms": [{"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}],
        "candidate_pool": [
            {
                "candidate_idx": 0,
                "evidence_id": "E1",
                "candidate_uid": "E1",
                "candidate_key": "doc:0",
                "text": "The city council approved the project in a public vote.",
                "covered_atom_ids": ["A1"],
                "map_relation": "support",
                "map_directness": "direct",
                "map_confidence": 0.9,
                "evidence_map_quality_score": 0.8,
                "chunk_sent_indices": [0],
                "qd_question_routes": [{"question": "Did the city approve the project?", "rank": 1}],
            }
        ],
    }

    trace = _build_trace(
        row,
        params=MRECSelectorParams(candidate_top_n=20, max_steps=5, cue_policy="legacy_route_prefer", selector_name="mrec_test"),
        source_selector_name="fallback_source",
    )

    assert trace["selector_name"] == "mrec_test"
    assert trace["mrec_selector_name"] == "mrec_test"
    assert trace["graph_version"] == MREC_TRACE_VERSION
    assert trace["fingerprint"] == "abc123"
    assert trace["candidate_pool_metadata"]["source_selector_name"] == "source_selector"
    assert trace["mrec_steps"][0]["operation"] == "OPEN"
    assert trace["mrec_steps"][0]["cue_text"] == "Did the city approve the project?"
    assert trace["compat_chain_steps"][0]["cue_text"] == "Did the city approve the project?"
    assert trace["mrec_diagnostics"]["resolved_atom_rate"] == 1.0


def test_build_mrec_trace_manifest_records_learned_policy_and_weight_fingerprint(tmp_path) -> None:
    weights_path = tmp_path / "weights.json"
    save_learned_marginal_weights(weights_path, initial_learned_marginal_weights())
    row = {
        "event_id": "evt-1",
        "claim": "The city approved the project.",
        "gold_label": "true",
        "selector_name": "source_selector",
        "claim_atoms": [{"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}],
        "candidate_pool": [
            {
                "candidate_idx": 0,
                "evidence_id": "E1",
                "candidate_uid": "E1",
                "candidate_key": "doc:0",
                "text": "The city council approved the project in a public vote.",
                "covered_atom_ids": ["A1"],
                "map_relation": "support",
                "map_directness": "direct",
                "map_confidence": 0.9,
                "evidence_map_quality_score": 0.8,
            }
        ],
    }
    params = MRECSelectorParams(
        selection_policy="learned_marginal_proxy",
        weight_file=str(weights_path),
        selector_name="mrec_greedy_transition_v0_2_learned_marginal_proxy",
        candidate_top_n=20,
        min_steps=1,
        max_steps=5,
    )

    trace = _build_trace(row, params=params, source_selector_name="fallback_source")
    manifest = _manifest(
        args=argparse.Namespace(
            output_dir=str(tmp_path),
            split="val",
            sample_limit=0,
            source_selector_name="fallback_source",
        ),
        input_path=tmp_path / "input.jsonl",
        params=params,
        n_input_rows=1,
        n_trace_rows=1,
    )

    assert trace["adaptive_policy"] == "learned_marginal_proxy_v0_2"
    assert trace["mrec_diagnostics"]["selection_policy"] == "learned_marginal_proxy"
    assert trace["mrec_diagnostics"]["weight_fingerprint"]
    assert manifest["adaptive_policy"] == "learned_marginal_proxy_v0_2"
    assert manifest["selector_name"] == "mrec_greedy_transition_v0_2_learned_marginal_proxy"
    assert manifest["params"]["selection_policy"] == "learned_marginal_proxy"
    assert manifest["weight_fingerprint"] == trace["mrec_diagnostics"]["weight_fingerprint"]


def test_build_mrec_trace_manifest_records_reward_policy_and_weight_fingerprint(tmp_path) -> None:
    weights_path = tmp_path / "reward_weights.json"
    save_learned_marginal_weights(
        weights_path,
        LearnedMarginalWeights(
            feature_weights={"resolution_delta": 1.0},
            cost_weight=0.0,
            schema_version=REWARD_WEIGHT_SCHEMA_VERSION,
            bias=0.1,
        ),
    )
    row = {
        "event_id": "evt-1",
        "claim": "The city approved the project.",
        "gold_label": "true",
        "selector_name": "source_selector",
        "claim_atoms": [{"atom_id": "A1", "text": "The city approved the project.", "importance": 1.0}],
        "candidate_pool": [
            {
                "candidate_idx": 0,
                "evidence_id": "E1",
                "candidate_uid": "E1",
                "candidate_key": "doc:0",
                "text": "The city council approved the project in a public vote.",
                "covered_atom_ids": ["A1"],
                "map_relation": "support",
                "map_directness": "direct",
                "map_confidence": 0.9,
                "evidence_map_quality_score": 0.8,
            }
        ],
    }
    params = MRECSelectorParams(
        selection_policy="learned_marginal_reward",
        weight_file=str(weights_path),
        selector_name="mrec_greedy_transition_v0_2_learned_marginal_reward",
        candidate_top_n=20,
        min_steps=1,
        max_steps=5,
    )

    trace = _build_trace(row, params=params, source_selector_name="fallback_source")
    manifest = _manifest(
        args=argparse.Namespace(
            output_dir=str(tmp_path),
            split="val",
            sample_limit=0,
            source_selector_name="fallback_source",
        ),
        input_path=tmp_path / "input.jsonl",
        params=params,
        n_input_rows=1,
        n_trace_rows=1,
    )

    assert trace["adaptive_policy"] == "learned_marginal_reward_v0_2"
    assert trace["mrec_diagnostics"]["selection_policy"] == "learned_marginal_reward"
    assert trace["mrec_diagnostics"]["weight_fingerprint"]
    assert manifest["adaptive_policy"] == "learned_marginal_reward_v0_2"
    assert manifest["selector_name"] == "mrec_greedy_transition_v0_2_learned_marginal_reward"
    assert manifest["params"]["selection_policy"] == "learned_marginal_reward"
    assert manifest["weight_fingerprint"] == trace["mrec_diagnostics"]["weight_fingerprint"]
