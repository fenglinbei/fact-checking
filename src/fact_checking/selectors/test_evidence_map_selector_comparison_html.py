from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.phase5_selectors.visualize.render_evidence_map_selector_comparison_html import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RIGHT_TRACE,
    default_candidate_features_path,
    default_coverage_diff_path,
    default_left_chain_graph_path,
    default_left_trace_path,
    default_raw_data_path,
    default_right_chain_graph_path,
    default_right_trace_path,
    find_trace_row,
    load_coverage_diff_row,
    load_raw_row,
    missing_trace_message,
    parse_args,
    render_html,
    resolve_inputs,
)


class EvidenceMapSelectorComparisonHtmlTest(unittest.TestCase):
    def test_parse_args_defaults_to_split_scan_analysis_output_and_translation(self) -> None:
        with patch("sys.argv", ["render", "--event-id", "case.json"]):
            args = parse_args()

        self.assertEqual(args.candidate_features, "")
        self.assertEqual(args.left_trace, "")
        self.assertEqual(args.right_trace, "")
        self.assertEqual(args.raw_data, "")
        self.assertEqual(args.coverage_diff, "")
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertTrue(args.translate_zh)

    def test_default_paths_use_sentence_qd_union_mainline(self) -> None:
        self.assertEqual(
            default_candidate_features_path("val"),
            "outputs/selectors/evidence_map_selector/v0_7_atom_facts_val/candidate_evidence_map_features_val.jsonl",
        )
        self.assertEqual(
            default_left_trace_path("val"),
            "outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_val/selection_trace_val.jsonl",
        )
        self.assertEqual(
            default_right_trace_path("val"),
            "outputs/selectors/evidence_chain_graph/v0_7_atom_facts_budgeted_marginal_adaptive5_10_val/selection_trace_val.jsonl",
        )
        self.assertEqual(
            default_left_chain_graph_path("val"),
            "outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_val/chain_graph_val.jsonl",
        )
        self.assertEqual(
            default_right_chain_graph_path("val"),
            "outputs/selectors/evidence_chain_graph/v0_7_atom_facts_budgeted_marginal_adaptive5_10_val/chain_graph_val.jsonl",
        )
        self.assertNotIn("liar_raw_dense", default_candidate_features_path("val"))
        self.assertNotIn("liar_raw_dense", default_left_trace_path("val"))
        self.assertNotIn("liar_raw_dense", default_right_trace_path("val"))
        self.assertNotIn("liar_raw_dense", default_left_chain_graph_path("val"))
        self.assertNotIn("liar_raw_dense", default_right_chain_graph_path("val"))

    def test_resolve_inputs_scans_train_val_test_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_jsonl(default_candidate_features_path("val"), [_row()])
                _write_jsonl(default_left_trace_path("val"), [_left_trace()])
                _write_jsonl(default_right_trace_path("val"), [_right_trace()])
                _write_json(default_raw_data_path("val"), [{"event_id": "case.json", "explain": "Raw explanation"}])
                _write_jsonl(default_coverage_diff_path("val"), [_coverage_diff()])

                args = _args()
                args.candidate_features = ""
                args.left_trace = ""
                args.right_trace = ""
                args.raw_data = ""
                args.coverage_diff = ""
                args.splits = "train,val,test"
                args.output_dir = DEFAULT_OUTPUT_DIR

                resolved = resolve_inputs(args)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resolved.split, "val")
        self.assertEqual(args.candidate_features, default_candidate_features_path("val"))
        self.assertEqual(args.left_trace, default_left_trace_path("val"))
        self.assertEqual(args.right_trace, default_right_trace_path("val"))
        self.assertEqual(args.raw_data, default_raw_data_path("val"))
        self.assertEqual(args.coverage_diff, default_coverage_diff_path("val"))
        self.assertEqual(resolved.raw_row["explain"], "Raw explanation")
        self.assertEqual(resolved.coverage_diff["coverage_label"], "uncovered")

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

    def test_renderer_uses_chain_graph_edges_when_available(self) -> None:
        html = render_html(
            _row(),
            left_trace=_left_trace(),
            right_trace=_right_trace(),
            left_graph_row=_chain_graph_row(),
            right_graph_row=_chain_graph_row(),
            args=_args(),
            translations={},
        )

        self.assertIn("selected evidence chain", html)
        self.assertIn("other candidates", html)
        self.assertIn('id="chainGraphSvg-left"', html)
        self.assertIn("draggable", html)
        self.assertIn('data-edge-type="tension"', html)
        self.assertIn("Evidence-Evidence Edges", html)
        self.assertIn('data-from-evidence-id="E01"', html)
        self.assertIn('data-to-evidence-id="E03"', html)
        self.assertIn("shared atom with conflicting relation", html)

    def test_renderer_shows_one_switchable_evidence_map_graph(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Evidence Map Graphs", html)
        self.assertIn("data-graph-switcher", html)
        self.assertIn('data-graph-option="left"', html)
        self.assertIn('data-graph-option="right"', html)
        self.assertIn('data-graph-panel="left"', html)
        self.assertIn('data-graph-panel="right"', html)
        self.assertIn("graph-panel-hidden", html)
        self.assertNotIn("map-graph-grid", html)

    def test_renderer_includes_graph_controls_and_overflow_safe_text_classes(self) -> None:
        row = _row()
        row["claim"] = "A very long claim " + ("with repeated text " * 30)
        row["candidates"][0]["text"] = "A very long evidence passage " + ("with repeated supporting context " * 30)

        html = render_html(row, left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("data-graph-relation-filter", html)
        self.assertIn("data-graph-selected-only", html)
        self.assertIn("data-graph-fit-toggle", html)
        self.assertIn("text-wrap-safe", html)
        self.assertIn("metric-value", html)
        self.assertIn("path-value", html)

    def test_renderer_uses_inner_translation_button_for_live_translation_progress(self) -> None:
        args = _args()
        args.web_translation_enabled = True
        args.web_base_path = "/evidence-map"
        args.web_token = "secret"

        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=args, translations={})

        self.assertIn("data-translation-toggle", html)
        self.assertIn("window.EVIDENCE_TRANSLATION_REQUEST", html)
        self.assertIn("/evidence-map/api/translate", html)
        self.assertIn("尚未翻译，点击后调用 API。", html)
        self.assertIn("DEEPSEEK_API_KEY", html)
        self.assertIn("正在翻译，请稍候", html)
        self.assertNotIn("data-translation-toggle disabled", html)

    def test_renderer_marks_partial_translation_cache_as_needing_refresh(self) -> None:
        args = _args()
        args.web_translation_enabled = True
        args.web_base_path = "/evidence-map"
        args.web_token = "secret"

        html = render_html(
            _row(),
            left_trace=_left_trace(),
            right_trace=_right_trace(),
            left_graph_row=_chain_graph_row(),
            right_graph_row=_chain_graph_row(),
            args=args,
            translations={"claim": "城市预算增加。"},
        )

        self.assertIn("window.EVIDENCE_TRANSLATION_MISSING_COUNT", html)
        self.assertIn("missing, click to update", html)
        self.assertIn("missingTranslationCount > 0", html)
        self.assertIn("/evidence-map/api/translate", html)

    def test_graph_nodes_are_clickable_with_relationship_detail_panel(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("data-graph-detail", html)
        self.assertIn("data-graph-detail-title", html)
        self.assertIn("data-graph-detail-body", html)
        self.assertIn('tabindex="0"', html)
        self.assertIn('role="button"', html)
        self.assertIn('data-graph-evidence-id="E03"', html)
        self.assertIn('data-covered-atoms="A3"', html)
        self.assertIn("Evidence Relationships", html)
        self.assertIn('data-graph-relationship="1"', html)
        self.assertIn('data-from-evidence-id="E03"', html)
        self.assertIn('data-to-evidence-id="E01"', html)
        self.assertIn("pair utility", html)

    def test_renderer_outputs_chain_process_replay_controls(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Evidence Chain Process", html)
        self.assertIn("data-chain-process", html)
        self.assertIn('data-chain-side="right"', html)
        self.assertIn('data-chain-step="1"', html)
        self.assertIn('data-chain-step="2"', html)
        self.assertIn('data-step-evidence-id="E03"', html)
        self.assertIn('data-step-anchor-evidence-ids="E01"', html)
        self.assertIn("Step 2", html)
        self.assertIn("sufficient_low_marginal_gain", html)
        self.assertIn("data-chain-play", html)
        self.assertIn("data-chain-reset", html)
        self.assertIn("data-process-summary", html)

    def test_renderer_outputs_chain_coverage_heatmap_and_decision_waterfall(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Coverage Accumulation", html)
        self.assertIn("data-coverage-heatmap", html)
        self.assertIn('data-heatmap-atom="A3"', html)
        self.assertIn('data-heatmap-step="2"', html)
        self.assertIn('data-heatmap-state="new"', html)
        self.assertIn("Decision Waterfall", html)
        self.assertIn("data-decision-waterfall", html)
        self.assertIn('data-delta-component="pair_utility"', html)
        self.assertIn('data-delta-direction="positive"', html)
        self.assertIn('data-delta-component="background_penalty"', html)
        self.assertIn('data-delta-direction="negative"', html)

    def test_chain_process_script_links_steps_to_graph_nodes(self) -> None:
        html = render_html(_row(), left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("activateChainStep", html)
        self.assertIn("data-current-chain-step", html)
        self.assertIn("graph-process-active", html)
        self.assertIn("graph-process-complete", html)
        self.assertIn("graph-process-dim", html)
        self.assertIn("graph-step-atom-active", html)

    def test_graph_evidence_candidate_nodes_use_adaptive_wrapped_cards(self) -> None:
        row = _row()
        row["candidates"][0]["text"] = "A long evidence candidate sentence " + ("with enough content to require wrapping " * 8)

        html = render_html(row, left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("<foreignObject", html)
        self.assertIn("graph-evidence-text", html)
        self.assertIn('data-graph-node="evidence"', html)
        self.assertIn('height="80"', html)

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

    def test_renderer_uses_raw_label_when_feature_gold_label_is_empty(self) -> None:
        row = _row()
        row["gold_label"] = ""

        html = render_html(
            row,
            raw_row={"event_id": "case.json", "label": "mostly-true", "explain": "This is the original gold explanation."},
            left_trace=_left_trace(),
            right_trace=_right_trace(),
            args=_args(),
            translations={},
        )

        self.assertIn("gold_label=mostly-true", html)

    def test_renderer_outputs_gold_explanation_from_feature_row_when_raw_missing(self) -> None:
        row = _row()
        row["gold_explain"] = "Feature row explanation."

        html = render_html(row, left_trace=_left_trace(), right_trace=_right_trace(), args=_args(), translations={})

        self.assertIn("Feature row explanation.", html)

    def test_renderer_outputs_coverage_diff_case_audit(self) -> None:
        html = render_html(
            _row(),
            coverage_diff=_coverage_diff(),
            left_trace=_left_trace(),
            right_trace=_right_trace(),
            args=_args(),
            translations={},
        )

        self.assertIn("Coverage Diff", html)
        self.assertIn("coverage_label", html)
        self.assertIn("uncovered", html)
        self.assertIn("critical_missing", html)
        self.assertIn("year:1979", html)
        self.assertIn("in_covered_weak", html)
        self.assertIn("Top source evidence mentions 1979.", html)

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

    def test_load_coverage_diff_row_matches_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case_coverage_diff_val.jsonl"
            path.write_text(
                json.dumps({"event_id": "other.json", "coverage_label": "covered"}) + "\n"
                + json.dumps({"event_id": "case.json", "coverage_label": "weak_covered"}) + "\n",
                encoding="utf-8",
            )

            coverage_diff = load_coverage_diff_row(str(path), event_id="case")

        self.assertIsNotNone(coverage_diff)
        self.assertEqual(coverage_diff["coverage_label"], "weak_covered")

    def test_missing_v07_trace_message_includes_generation_command(self) -> None:
        message = missing_trace_message(DEFAULT_RIGHT_TRACE, role="right")

        self.assertIn("Missing right trace file", message)
        self.assertIn("run_evidence_chain_graph_v0_7_atom_facts.sh", message)
        self.assertIn("v0_7_atom_facts_val", message)
        self.assertNotIn("liar_raw_dense", message)


def _args() -> Namespace:
    return Namespace(
        candidate_features="features.jsonl",
        left_trace="left.jsonl",
        right_trace="right.jsonl",
        raw_data="raw.json",
        coverage_diff="coverage_diff.jsonl",
        splits="train,val,test",
        output_dir=DEFAULT_OUTPUT_DIR,
        left_label="v0.6c RuleStep",
        right_label="v0.7 AtomFacts BudgetedMarginal",
        max_candidates=20,
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


def _coverage_diff() -> dict:
    return {
        "event_id": "case.json",
        "label": "mostly-true",
        "coverage_label": "uncovered",
        "coverage_score": 0.604218,
        "weak_score": 0.624952,
        "decision_source": "rule",
        "critical_missing": ["year:1979"],
        "in_all": True,
        "in_covered": False,
        "in_covered_weak": False,
        "source_sidecar": "source_coverage_val.jsonl",
        "top_evidence_preview": [
            {
                "rank": 1,
                "report_id": "report-1",
                "sent_idx": 3,
                "text": "Top source evidence mentions 1979.",
                "bm25": 8.5,
                "lexical_coverage": 0.66,
                "embedding_score": 0.78,
                "hybrid_score": 0.72,
                "anchor_hits": {"years": ["1979"], "numbers": []},
            }
        ],
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
        "selector_name": "v0_7_budgeted_marginal_chain_adaptive5_10",
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


def _chain_graph_row() -> dict:
    return {
        "event_id": "case.json",
        "claim": "The city budget increased.",
        "gold_label": "mostly-true",
        "selector_name": "chain",
        "graph_version": "evidence_chain_graph_test",
        "claim_node": {"node_id": "C0", "type": "claim", "text": "The city budget increased."},
        "atom_nodes": [
            {"node_id": "A1", "atom_id": "A1", "text": "The city has a budget.", "importance": 1.0, "atom_type": "entity"},
            {"node_id": "A3", "atom_id": "A3", "text": "The increase was recent.", "importance": 0.7, "atom_type": "temporal"},
        ],
        "evidence_nodes": [
            {
                "node_id": "E01",
                "evidence_id": "E01",
                "text": "The city budget exists.",
                "relation": "support",
                "directness": "direct",
                "covered_atom_ids": ["A1"],
                "source_group": "report:1",
                "oracle_selected": False,
            },
            {
                "node_id": "E03",
                "evidence_id": "E03",
                "text": "The latest increase was smaller than planned.",
                "relation": "qualify",
                "directness": "direct",
                "covered_atom_ids": ["A3"],
                "source_group": "report:2",
                "oracle_selected": False,
            },
        ],
        "edges": [
            {"edge_type": "claim_has_atom", "source": "C0", "target": "A1", "weight": 1.0},
            {"edge_type": "claim_has_atom", "source": "C0", "target": "A3", "weight": 1.0},
            {"edge_type": "evidence_covers_atom", "source": "E01", "target": "A1", "weight": 0.8, "atom_ids": ["A1"], "relation": "support"},
            {"edge_type": "evidence_covers_atom", "source": "E03", "target": "A3", "weight": 0.7, "atom_ids": ["A3"], "relation": "qualify"},
            {
                "edge_type": "tension",
                "source": "E01",
                "target": "E03",
                "weight": 0.85,
                "atom_ids": ["A3"],
                "reason": "shared atom with conflicting relation",
            },
        ],
        "selected_evidence_ids": ["E01", "E03"],
        "selected_chain_id": "CH01",
        "chains": [
            {
                "chain_id": "CH01",
                "chain_score": 0.8,
                "weighted_atom_coverage": 1.0,
                "direct_or_partial_rate": 1.0,
                "positive_pair_edge_density": 1.0,
                "evidence_ids": ["E01", "E03"],
                "covered_atom_ids": ["A1", "A3"],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
