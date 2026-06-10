from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.phase5_selectors.visualize.render_evidence_map_selector_comparison_html import (
    DEFAULT_RIGHT_TRACE,
    find_trace_row,
    load_raw_row,
    missing_trace_message,
    render_html,
)


class EvidenceMapSelectorComparisonHtmlTest(unittest.TestCase):
    def test_renderer_marks_common_left_only_and_right_only_candidates(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn('data-status="common"', html)
        self.assertIn('data-status="left-only"', html)
        self.assertIn('data-status="right-only"', html)
        self.assertIn("E02", html)
        self.assertIn("E03", html)

    def test_renderer_uses_original_evidence_map_graph_strategy(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Evidence Map Graphs", html)
        self.assertIn("graph-legend", html)
        self.assertIn("claim atoms", html)
        self.assertIn("evidence candidates", html)
        self.assertIn("TOP 1", html)

    def test_renderer_outputs_v07_marginal_gain_and_objective_fields(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("marginal gain", html)
        self.assertIn("coverage after step", html)
        self.assertIn("objective score", html)
        self.assertIn("pair utility", html)
        self.assertIn("background penalty", html)
        self.assertIn("0.24", html)

    def test_renderer_outputs_gold_explanation_from_raw_row(self) -> None:
        html = render_html(
            _row(),
            raw_row={"event_id": "case.json", "label": "mostly-true", "explain": "This is the original gold explanation."},
            left_trace=_left_trace(),
            right_trace=_right_trace(),
            args=_args(),
            translations={},
        )

        self.assertIn("Gold Explanation", html)
        self.assertIn("This is the original gold explanation.", html)
        self.assertIn("raw_label=mostly-true", html)

    def test_renderer_outputs_gold_explanation_from_feature_row_when_raw_missing(self) -> None:
        row = _row()
        row["gold_explain"] = "Feature row explanation."

        html = render_html(row, left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Feature row explanation.", html)

    def test_renderer_tolerates_v06c_trace_without_objective_fields(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_left_trace(), args=_args(), translations={})

        self.assertIn("No objective components on this trace.", html)
        self.assertIn("v0_6c_rule_step_adaptive5_10", html)

    def test_find_trace_row_reports_event_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_trace_val.jsonl"
            path.write_text(json.dumps({"event_id": "other.json", "selector_name": "selector"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No left trace row matched event_id"):
                find_trace_row(str(path), event_id="case.json", role="left")

    def test_find_trace_row_reports_selector_mismatch_when_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_trace_val.jsonl"
            path.write_text(json.dumps({"event_id": "case.json", "selector_name": "actual"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "selector_name='wanted'"):
                find_trace_row(str(path), event_id="case.json", role="left", expected_selector_name="wanted")

    def test_load_raw_row_matches_liar_raw_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            path.write_text(
                json.dumps(
                    [
                        {"event_id": "other.json", "explain": "Other"},
                        {"event_id": "case.json", "explain": "Matched explanation"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            raw_row = load_raw_row(str(path), event_id="case.json")

        self.assertIsNotNone(raw_row)
        self.assertEqual(raw_row["explain"], "Matched explanation")

    def test_missing_v07_trace_message_includes_generation_command(self) -> None:
        message = missing_trace_message(DEFAULT_RIGHT_TRACE, role="right")

        self.assertIn("Missing right trace file", message)
        self.assertIn("run_evidence_chain_graph_v0_7.sh", message)
        self.assertIn("liar_raw_dense_v0_6b_val", message)


def _args() -> Namespace:
    return Namespace(
        candidate_features="features.jsonl",
        left_trace="left.jsonl",
        right_trace="right.jsonl",
        raw_data="raw.json",
        left_label="v0.6c RuleStep",
        right_label="v0.7 BudgetedMarginal",
        max_candidates=20,
    )


def _row() -> dict:
    return {
        "event_id": "case.json",
        "claim": "The city budget increased.",
        "gold_label": "mostly-true",
        "evidence_map": {
            "claim_atoms": [
                {"atom_id": "A1", "text": "The city has a budget.", "type": "entity", "importance": 1.0},
                {"atom_id": "A2", "text": "The budget increased.", "type": "predicate", "importance": 1.0},
                {"atom_id": "A3", "text": "The increase was recent.", "type": "temporal", "importance": 0.7},
            ]
        },
        "candidates": [
            _candidate("E01", atoms=["A1"], relation="support", directness="direct", text="The city budget exists."),
            _candidate("E02", atoms=["A2"], relation="support", directness="partial", text="The budget rose last year."),
            _candidate("E03", atoms=["A3"], relation="qualify", directness="direct", text="The latest increase was smaller than planned."),
            _candidate("E04", atoms=[], relation="background", directness="context", text="The article describes local politics."),
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


def _left_trace() -> dict:
    return {
        "event_id": "case.json",
        "selector_name": "v0_6c_rule_step_adaptive5_10",
        "selected_evidence_ids": ["E01", "E02"],
        "precision@5": 0.5,
        "recall@5": 0.5,
        "jaccard@5": 0.33,
        "weighted_atom_coverage@5": 0.66,
        "missing_atom_rate@5": 0.33,
        "adaptive_stop_reason": "reached_min_top_k_no_rule_candidate",
        "selection_steps": [
            {
                "step": 1,
                "evidence_id": "E01",
                "rule": "anchor_core",
                "covered_new_atom_ids": ["A1"],
                "anchor_evidence_ids": [],
                "fallback_used": False,
                "directness": "direct",
                "relation": "support",
            },
            {
                "step": 2,
                "evidence_id": "E02",
                "rule": "P1_new_atom_core",
                "covered_new_atom_ids": ["A2"],
                "anchor_evidence_ids": ["E01"],
                "fallback_used": False,
                "directness": "partial",
                "relation": "support",
            },
        ],
    }


def _right_trace() -> dict:
    return {
        "event_id": "case.json",
        "selector_name": "v0_7_budgeted_marginal_chain_adaptive3_10",
        "selected_evidence_ids": ["E01", "E03"],
        "precision@5": 0.5,
        "recall@5": 0.5,
        "jaccard@5": 0.33,
        "weighted_atom_coverage@5": 0.70,
        "missing_atom_rate@5": 0.30,
        "adaptive_stop_reason": "sufficient_low_marginal_gain",
        "objective_final_score": 0.42,
        "objective_final_components": {
            "coverage": 0.70,
            "node_quality": 0.08,
            "pair_utility": 0.14,
            "background_penalty": -0.16,
            "length_penalty": -0.12,
        },
        "selection_steps": [
            {
                "step": 1,
                "evidence_id": "E01",
                "rule": "budgeted_marginal_gain",
                "marginal_gain": 0.31,
                "component_deltas": {"coverage": 0.4, "node_quality": 0.04, "length_penalty": -0.06},
                "coverage_after_step": 0.4,
                "covered_new_atom_ids": ["A1"],
                "anchor_evidence_ids": [],
                "directness": "direct",
                "relation": "support",
            },
            {
                "step": 2,
                "evidence_id": "E03",
                "rule": "budgeted_marginal_gain",
                "marginal_gain": 0.24,
                "component_deltas": {"coverage": 0.3, "pair_utility": 0.14, "background_penalty": -0.16},
                "coverage_after_step": 0.7,
                "covered_new_atom_ids": ["A3"],
                "anchor_evidence_ids": ["E01"],
                "directness": "direct",
                "relation": "qualify",
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
