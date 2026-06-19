from __future__ import annotations

from fact_checking.selectors.minimal_resolving_chain import MRECSelectorParams
from fact_checking.selectors.mrec_schema import MREC_TRACE_VERSION
from scripts.phase5_selectors.build.build_mrec_traces import _build_trace


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
        params=MRECSelectorParams(candidate_top_n=20, max_steps=5, selector_name="mrec_test"),
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
