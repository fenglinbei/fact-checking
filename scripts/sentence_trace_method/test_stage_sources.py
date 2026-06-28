from __future__ import annotations

import importlib.util
import json
import sys
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


def test_stage_split_rejects_multi_sentence_candidates_by_default(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target" / "selection_trace_val.jsonl"
    source_path.write_text(json.dumps(_trace_row(chunk_sent_indices=[0, 1])) + "\n", encoding="utf-8")

    try:
        stage_sources.stage_split(
            dataset="rawfc",
            split="val",
            source_path=source_path,
            target_path=target_path,
            sample_limit=0,
            force=False,
            selector_name="v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10",
            graph_version="evidence_chain_graph_v0_7",
            adaptive_policy="budgeted_marginal_v0_7",
            expected_fingerprint="fp",
            forbidden_fingerprints=set(),
        )
    except ValueError as exc:
        assert "non-sentence candidate" in str(exc)
    else:
        raise AssertionError("Expected multi-sentence candidates to fail without opt-in.")


def test_stage_split_allows_multi_sentence_candidates_when_requested(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target" / "selection_trace_val.jsonl"
    source_path.write_text(json.dumps(_trace_row(chunk_sent_indices=[0, 1])) + "\n", encoding="utf-8")

    manifest = stage_sources.stage_split(
        dataset="rawfc",
        split="val",
        source_path=source_path,
        target_path=target_path,
        sample_limit=0,
        force=False,
        selector_name="v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10",
        graph_version="evidence_chain_graph_v0_7",
        adaptive_policy="budgeted_marginal_v0_7",
        expected_fingerprint="fp",
        forbidden_fingerprints=set(),
        allow_multi_sentence_candidates=True,
    )

    audit = manifest["sentence_chunk_audit"]
    assert audit["allow_multi_sentence_candidates"] is True
    assert audit["multi_sentence_candidates"] == 1
    assert audit["checked_candidates"] == 1


def test_stage_split_allows_empty_candidate_pool_when_requested(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target" / "selection_trace_val.jsonl"
    row = _trace_row()
    row["candidate_pool"] = []
    row["selector_ordered_indices"] = []
    row["selected_indices"] = []
    source_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = stage_sources.stage_split(
        dataset="liar_raw",
        split="val",
        source_path=source_path,
        target_path=target_path,
        sample_limit=0,
        force=False,
        selector_name="selector_mech_s0_no_evidence",
        graph_version="selector_mechanism_ablation_v0",
        adaptive_policy="fixed_top5",
        expected_fingerprint="fp",
        forbidden_fingerprints=set(),
        allow_empty_candidate_pool=True,
    )

    staged = json.loads(target_path.read_text(encoding="utf-8").strip())
    audit = manifest["sentence_chunk_audit"]
    assert staged["candidate_pool"] == []
    assert audit["empty_candidate_pool_rows"] == 1
    assert audit["allow_empty_candidate_pool"] is True


def test_main_summary_uses_custom_selector_name(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_split = source_root / "val"
    source_split.mkdir(parents=True)
    (source_split / "selection_trace_val.jsonl").write_text(json.dumps(_trace_row()) + "\n", encoding="utf-8")
    output_root = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_sources.py",
            "--dataset",
            "rawfc",
            "--output-root",
            str(output_root),
            "--source-root",
            str(source_root),
            "--selector-name",
            "v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10",
            "--graph-version",
            "evidence_chain_graph_v0_7",
            "--adaptive-policy",
            "budgeted_marginal_v0_7",
            "--expected-fingerprint",
            "fp",
            "--splits",
            "val",
            "--print-json",
        ],
    )

    assert stage_sources.main() == 0
    summary_path = output_root / "_sources" / "rawfc" / "v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10" / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_set"] == "v0_7_atom_facts_budgeted_marginal_chain_adaptive5_10"


def _trace_row(chunk_sent_indices: list[int] | None = None) -> dict:
    if chunk_sent_indices is None:
        chunk_sent_indices = [0]
    return {
        "event_id": "case-1",
        "selector_name": "old",
        "graph_version": "old_graph",
        "adaptive_policy": "old_policy",
        "fingerprint": "fp",
        "candidate_pool_metadata": {"chunk_mmr_fingerprint": "fp"},
        "candidate_pool": [{"chunk_sent_indices": chunk_sent_indices, "text": "Evidence."}],
        "selector_ordered_indices": [0],
        "oracle_ordered_indices": [0],
    }
