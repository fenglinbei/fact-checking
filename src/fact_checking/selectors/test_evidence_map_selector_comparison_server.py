from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.phase5_selectors.visualize.render_evidence_map_selector_comparison_html import (
    default_candidate_features_path,
    default_coverage_diff_path,
    default_left_trace_path,
    default_raw_data_path,
    default_right_trace_path,
)
from scripts.phase5_selectors.visualize.serve_evidence_map_selector_comparison import (
    ComparisonStore,
    is_authorized,
    strip_base_path,
)


class EvidenceMapSelectorComparisonServerTest(unittest.TestCase):
    def test_store_loads_cases_and_searches_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")

                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)
                results = store.search_cases("city budget", split="val", limit=5)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["event_id"], "case.json")
        self.assertEqual(results[0]["split"], "val")
        self.assertEqual(results[0]["gold_label"], "mostly-true")
        self.assertEqual(results[0]["coverage_label"], "uncovered")

    def test_store_search_skips_cases_missing_right_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_jsonl(default_candidate_features_path("train"), [_row()])
                _write_jsonl(default_left_trace_path("train"), [_left_trace()])

                store = ComparisonStore.load(Path(tmp), splits=["train"], max_candidates=20)
                results = store.search_cases("city budget", split="train", limit=5)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(results, [])

    def test_store_renders_existing_comparison_html_without_live_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")

                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)
                html = store.render_case(split="val", event_id="case", left_label="left", right_label="right")
            finally:
                os.chdir(old_cwd)

        self.assertIn("Evidence map selector comparison: case.json", html)
        self.assertIn("The city budget increased.", html)
        self.assertIn("Coverage Diff", html)

    def test_token_authorization_accepts_header_or_query(self) -> None:
        self.assertTrue(is_authorized("", {}, {}))
        self.assertTrue(is_authorized("secret", {"token": ["secret"]}, {}))
        self.assertTrue(is_authorized("secret", {}, {"X-Access-Token": "secret"}))
        self.assertFalse(is_authorized("secret", {}, {}))
        self.assertFalse(is_authorized("secret", {"token": ["wrong"]}, {"X-Access-Token": "wrong"}))

    def test_base_path_stripping(self) -> None:
        self.assertEqual(strip_base_path("/evidence-map/api/cases", "/evidence-map"), "/api/cases")
        self.assertEqual(strip_base_path("/evidence-map/", "/evidence-map"), "/")
        self.assertEqual(strip_base_path("/api/cases", ""), "/api/cases")


def _write_fixture_split(split: str) -> None:
    _write_jsonl(default_candidate_features_path(split), [_row()])
    _write_jsonl(default_left_trace_path(split), [_left_trace()])
    _write_jsonl(default_right_trace_path(split), [_right_trace()])
    _write_json(default_raw_data_path(split), {"case": _raw_row()})
    _write_jsonl(default_coverage_diff_path(split), [_coverage_diff()])
    _write_json(
        Path("outputs/analysis/map/v0.7") / f"{split}_evidence_map_compare_case_left_vs_right.zh.json",
        {"translations": {"claim": "城市预算增加。"}},
    )


def _write_json(path: str | Path, payload: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row() -> dict:
    return {
        "event_id": "case.json",
        "claim": "The city budget increased.",
        "gold_label": "mostly-true",
        "evidence_map": {
            "claim_atoms": [
                {"atom_id": "A1", "text": "The city has a budget.", "type": "entity", "importance": 1.0},
                {"atom_id": "A2", "text": "The budget increased.", "type": "predicate", "importance": 1.0},
            ]
        },
        "candidates": [
            _candidate("E01", atoms=["A1"], relation="support", directness="direct", text="The city budget exists."),
            _candidate("E02", atoms=["A2"], relation="support", directness="partial", text="The budget rose last year."),
        ],
    }


def _candidate(
    evidence_id: str,
    *,
    atoms: list[str],
    relation: str,
    directness: str,
    text: str,
) -> dict:
    return {
        "candidate_uid": f"uid-{evidence_id}",
        "candidate_key": text,
        "evidence_id": evidence_id,
        "text": text,
        "covered_atom_ids": atoms,
        "map_relation": relation,
        "map_directness": directness,
        "map_evidence_role": "claim_specific",
        "key_spans": [text.split(".")[0]],
        "source_group": "report:1",
        "evidence_map_quality_score": 0.9,
        "evidence_map_base_score": 0.7,
        "retrieval_score": 0.5,
    }


def _raw_row() -> dict:
    return {
        "event_id": "case.json",
        "label": "mostly-true",
        "explain": "The raw explanation describes the budget claim.",
    }


def _coverage_diff() -> dict:
    return {
        "event_id": "case.json",
        "label": "mostly-true",
        "coverage_label": "uncovered",
        "coverage_score": 0.604218,
        "weak_score": 0.624952,
        "decision_source": "rule",
        "critical_missing": ["budget:increase"],
        "top_evidence_preview": [
            {
                "rank": 1,
                "report_id": "report-1",
                "sent_idx": 3,
                "text": "Top source evidence mentions the budget.",
                "bm25": 8.5,
            }
        ],
    }


def _left_trace() -> dict:
    return {
        "event_id": "case.json",
        "selector_name": "left",
        "selected_evidence_ids": ["E01"],
        "precision@5": 0.5,
        "recall@5": 0.5,
        "jaccard@5": 0.33,
        "weighted_atom_coverage@5": 0.5,
        "missing_atom_rate@5": 0.5,
        "adaptive_stop_reason": "done",
        "selection_steps": [
            {
                "step": 1,
                "evidence_id": "E01",
                "rule": "anchor_core",
                "covered_new_atom_ids": ["A1"],
                "directness": "direct",
                "relation": "support",
            }
        ],
    }


def _right_trace() -> dict:
    return {
        "event_id": "case.json",
        "selector_name": "right",
        "selected_evidence_ids": ["E02"],
        "precision@5": 0.5,
        "recall@5": 0.5,
        "jaccard@5": 0.33,
        "weighted_atom_coverage@5": 0.5,
        "missing_atom_rate@5": 0.5,
        "adaptive_stop_reason": "done",
        "objective_final_score": 0.42,
        "objective_final_components": {
            "coverage": 0.5,
            "node_quality": 0.08,
            "pair_utility": 0.14,
            "background_penalty": -0.16,
        },
        "selection_steps": [
            {
                "step": 1,
                "evidence_id": "E02",
                "rule": "budgeted_marginal_gain",
                "marginal_gain": 0.31,
                "coverage_after_step": 0.5,
                "covered_new_atom_ids": ["A2"],
                "directness": "partial",
                "relation": "support",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
