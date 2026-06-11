from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("stage_sources.py")
SPEC = importlib.util.spec_from_file_location("stage_sources", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stage_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_sources)


def test_stage_split_keeps_custom_selector_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target" / "selection_trace_val.jsonl"
    source_path.write_text(json.dumps(_trace_row()) + "\n", encoding="utf-8")

    manifest = stage_sources.stage_split(
        dataset="liar_raw",
        split="val",
        source_path=source_path,
        target_path=target_path,
        sample_limit=0,
        force=False,
        selector_name="v0_7_budgeted_marginal_chain_adaptive3_10",
        graph_version="evidence_chain_graph_v0_7",
        adaptive_policy="budgeted_marginal_v0_7",
        expected_fingerprint="fp",
        forbidden_fingerprints=set(),
    )

    row = json.loads(target_path.read_text(encoding="utf-8").strip())
    assert row["selector_name"] == "v0_7_budgeted_marginal_chain_adaptive3_10"
    assert row["graph_version"] == "evidence_chain_graph_v0_7"
    assert row["adaptive_policy"] == "budgeted_marginal_v0_7"
    assert row["candidate_pool_metadata"]["selector_name"] == "v0_7_budgeted_marginal_chain_adaptive3_10"
    assert manifest["selector_name"] == "v0_7_budgeted_marginal_chain_adaptive3_10"


def _trace_row() -> dict:
    return {
        "event_id": "case-1",
        "selector_name": "old",
        "graph_version": "old_graph",
        "adaptive_policy": "old_policy",
        "fingerprint": "fp",
        "candidate_pool_metadata": {"chunk_mmr_fingerprint": "fp"},
        "candidate_pool": [{"chunk_sent_indices": [0], "text": "Evidence."}],
        "selector_ordered_indices": [0],
        "oracle_ordered_indices": [0],
    }
