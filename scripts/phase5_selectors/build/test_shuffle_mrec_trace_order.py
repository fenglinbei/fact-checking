from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.phase5_selectors.build.shuffle_mrec_trace_order import main


def test_shuffle_mrec_trace_order_preserves_selected_set_and_reorders_chain(tmp_path: Path) -> None:
    source_row = _source_trace_row()
    input_path = tmp_path / "selection_trace_val.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(input_path, [source_row])

    exit_code = main(
        Namespace(
            input=str(input_path),
            output_dir=str(output_dir),
            split="val",
            sample_limit=0,
            selector_name="selector_mech_s6_learned_marginal_proxy_trace_shuffle",
            adaptive_policy="learned_marginal_proxy_trace_shuffle_v0_2",
            source_selector_name="mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool",
            seed=0,
        )
    )

    assert exit_code == 0
    rows = _read_jsonl(output_dir / "selection_trace_val.jsonl")
    assert len(rows) == 1
    row = rows[0]

    original_ids = source_row["selected_evidence_ids"]
    shuffled_ids = row["selected_evidence_ids"]
    assert sorted(shuffled_ids) == sorted(original_ids)
    assert shuffled_ids != original_ids
    assert row["candidate_pool"] == source_row["candidate_pool"]
    assert [row["candidate_pool"][idx]["evidence_id"] for idx in row["selected_indices"]] == shuffled_ids
    assert [candidate["evidence_id"] for candidate in row["selected_candidates"]] == shuffled_ids
    assert [step["evidence_id"] for step in row["mrec_steps"]] == shuffled_ids
    assert [step["step"] for step in row["mrec_steps"]] == [1, 2, 3, 4]
    assert [step["evidence_id"] for step in row["compat_chain_steps"]] == shuffled_ids
    assert [step["evidence_id"] for step in row["chain_steps"]] == shuffled_ids

    assert row["selector_name"] == "selector_mech_s6_learned_marginal_proxy_trace_shuffle"
    assert row["graph_version"] == "selector_mechanism_ablation_v0"
    assert row["mrec_trace_version"] == "mrec_trace_v0_1"
    assert row["candidate_pool_metadata"]["shuffle_seed"] == 0
    assert row["candidate_pool_metadata"]["shuffle_preserves_selected_set"] is True
    assert row["candidate_pool_metadata"]["shuffle_source_selector_name"] == (
        "mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool"
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selector_name"] == "selector_mech_s6_learned_marginal_proxy_trace_shuffle"
    assert manifest["shuffle_seed"] == 0
    assert manifest["n_trace_rows"] == 1


def _source_trace_row() -> dict[str, object]:
    candidates = [
        {
            "candidate_idx": idx,
            "selector_candidate_idx": idx,
            "candidate_key": f"doc:{idx}",
            "evidence_id": f"E{idx}",
            "candidate_uid": f"E{idx}",
            "text": f"Evidence {idx}",
            "covered_atom_ids": ["A1"],
            "map_relation": "support",
            "map_directness": "direct",
            "evidence_map_quality_score": 1.0 - idx * 0.1,
        }
        for idx in range(4)
    ]
    steps = [
        {
            "step": idx + 1,
            "operation": "OPEN",
            "atom_id": "A1",
            "atom_text": "The city approved the project.",
            "state_before": "U",
            "state_after": "S",
            "cue_text": "The city approved the project.",
            "cue_source": "claim_atom",
            "evidence_id": f"E{idx}",
            "candidate_idx": idx,
            "selector_candidate_idx": idx,
            "evidence_text": f"Evidence {idx}",
            "covered_atom_ids": ["A1"],
            "relation": "support",
            "directness": "direct",
            "map_confidence": 0.8,
            "evidence_map_quality_score": 1.0 - idx * 0.1,
            "token_cost": 2,
            "transition_reason": "test transition",
            "trace_state": {"selected_count": idx + 1},
        }
        for idx in range(4)
    ]
    return {
        "event_id": "evt-shuffle",
        "claim": "The city approved the project.",
        "gold_label": "true",
        "mrec_trace_version": "mrec_trace_v0_1",
        "mrec_selector_name": "mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool",
        "selector_name": "mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool",
        "graph_version": "mrec_trace_v0_1",
        "adaptive_policy": "learned_marginal_proxy_v0_2",
        "source_selector_name": "v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10",
        "candidate_pool_metadata": {"graph_version": "mrec_trace_v0_1"},
        "claim_atoms": [{"atom_id": "A1", "text": "The city approved the project."}],
        "candidate_pool": candidates,
        "selected_indices": [0, 1, 2, 3],
        "selector_ordered_indices": [0, 1, 2, 3],
        "selected_candidates": candidates,
        "selected_evidence_ids": ["E0", "E1", "E2", "E3"],
        "selected_keys": ["doc:0", "doc:1", "doc:2", "doc:3"],
        "mrec_steps": steps,
        "compat_chain_steps": [dict(step) for step in steps],
        "chain_steps": [dict(step) for step in steps],
        "mrec_diagnostics": {"selection_policy": "learned_marginal_proxy", "step_count": 4},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
