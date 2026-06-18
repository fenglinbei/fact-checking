from __future__ import annotations

import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from fact_checking.selectors.evidence_chain_graph import (
    BUDGETED_MARGINAL_SELECTOR,
    CHAIN_SELECTOR,
    RULE_STEP_CHAIN_SELECTOR,
    SUFFICIENCY_CONTRADICTION_SELECTOR,
    BudgetedMarginalChainParams,
    EvidenceChainParams,
    RuleStepEvidenceChainParams,
    SufficiencyContradictionEvidenceChainParams,
    build_budgeted_marginal_chain_graph_row,
    build_evidence_chain_graph_row,
    build_rule_step_evidence_chain_graph_row,
    build_sufficiency_contradiction_evidence_chain_graph_row,
    budgeted_marginal_chain_selector_name,
    rule_step_chain_selector_name,
    summarize_budgeted_marginal_chain_graph_rows,
)
from scripts.phase5_selectors.visualize.render_evidence_chain_graph_html import (
    load_or_build_translations,
    render_html,
    visual_width,
    wrap_text_for_svg,
)
from scripts.phase5_selectors.build import build_evidence_chain_graph_v0_7 as graph_cli


class EvidenceChainGraphTest(unittest.TestCase):
    def test_builds_typed_edges_from_map_features(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", sent_idx=1, base=0.95, duplicate="G1"),
                _candidate("E02", atoms=["A1"], relation="refute", directness="partial", source="report:2", sent_idx=8, base=0.82),
                _candidate("E03", atoms=["A2"], relation="support", directness="direct", source="report:3", sent_idx=1, base=0.78),
                _candidate("E04", atoms=["A1"], relation="background", directness="context", source="report:1", sent_idx=3, base=0.62),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:4", sent_idx=1, base=0.58),
                _candidate("E06", atoms=["A2"], relation="support", directness="direct", source="report:5", sent_idx=1, base=0.54, duplicate="G1"),
            ]
        )

        graph = build_evidence_chain_graph_row(row, params=EvidenceChainParams(top_k=3, beam_size=6))

        edge_types = {edge["edge_type"] for edge in graph["edges"]}
        self.assertIn("claim_has_atom", edge_types)
        self.assertIn("evidence_covers_atom", edge_types)
        self.assertIn("duplicate", edge_types)
        self.assertIn("same_source_context", edge_types)
        self.assertIn("complements", edge_types)
        self.assertIn("corroborates", edge_types)
        self.assertIn("tension", edge_types)
        self.assertIn("bridge_context", edge_types)

    def test_chain_selection_does_not_depend_on_gold_or_oracle_fields(self) -> None:
        candidates = [
            _candidate("E01", atoms=["A1"], base=0.85, oracle=True),
            _candidate("E02", atoms=["A2"], base=0.82),
            _candidate("E03", atoms=["A3"], base=0.79),
            _candidate("E04", atoms=["A1"], relation="background", directness="context", base=0.40),
        ]
        row = _row(candidates, gold_label="true")
        altered = _row([{**candidate, "oracle_selected": not bool(candidate.get("oracle_selected")), "oracle_step": 99} for candidate in candidates], gold_label="false")

        first = build_evidence_chain_graph_row(row, params=EvidenceChainParams(top_k=3))
        second = build_evidence_chain_graph_row(altered, params=EvidenceChainParams(top_k=3))

        self.assertEqual(first["selected_evidence_ids"], second["selected_evidence_ids"])
        self.assertEqual(first["selection_trace"]["selector_name"], CHAIN_SELECTOR)

    def test_duplicate_evidence_is_penalized_in_top_chain(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], base=0.95, duplicate="G1"),
                _candidate("E02", atoms=["A1"], base=0.93, duplicate="G1"),
                _candidate("E03", atoms=["A2"], base=0.70, duplicate="G2"),
                _candidate("E04", atoms=["A3"], base=0.68, duplicate="G3"),
            ]
        )

        graph = build_evidence_chain_graph_row(row, params=EvidenceChainParams(top_k=3, beam_size=8))

        selected_duplicates = [
            node["duplicate_group"]
            for node in graph["evidence_nodes"]
            if node["node_id"] in set(graph["selected_evidence_ids"]) and node.get("duplicate_group")
        ]
        self.assertEqual(len(selected_duplicates), len(set(selected_duplicates)))

    def test_post_order_preserves_selected_set_and_puts_claim_atoms_first(self) -> None:
        row = _row(
            [
                _candidate("E02", atoms=["A2"], base=0.96, source="report:2", sent_idx=8),
                _candidate("E01", atoms=["A1"], base=0.92, source="report:1", sent_idx=1),
                _candidate("E04", atoms=[], relation="background", directness="context", source="report:1", sent_idx=3, base=0.99),
            ]
        )

        graph = build_evidence_chain_graph_row(row, params=EvidenceChainParams(top_k=3, beam_size=6))
        chain = graph["chains"][0]

        self.assertEqual(set(chain["evidence_ids"]), set(chain["search_order_evidence_ids"]))
        self.assertEqual(set(graph["selected_evidence_ids"]), {"E01", "E02", "E04"})
        self.assertLess(graph["selected_evidence_ids"].index("E01"), graph["selected_evidence_ids"].index("E02"))
        self.assertLess(graph["selected_evidence_ids"].index("E02"), graph["selected_evidence_ids"].index("E04"))

    def test_context_follows_bridge_anchor_and_unanchored_context_goes_late(self) -> None:
        bridged = _row(
            [
                _candidate("E01", atoms=["A1"], base=0.90, source="report:1", sent_idx=1),
                _candidate("E02", atoms=["A1"], relation="background", directness="context", source="report:1", sent_idx=2, base=0.89),
                _candidate("E03", atoms=[], relation="irrelevant", directness="none", source="report:9", sent_idx=9, base=0.88),
            ]
        )
        unanchored = _row(
            [
                _candidate("E01", atoms=["A1"], base=0.90, source="report:1", sent_idx=1),
                _candidate("E02", atoms=[], relation="background", directness="context", source="report:8", sent_idx=2, base=0.89),
                _candidate("E03", atoms=["A2"], base=0.88, source="report:9", sent_idx=9),
            ]
        )

        bridged_graph = build_evidence_chain_graph_row(bridged, params=EvidenceChainParams(top_k=3, beam_size=6))
        unanchored_graph = build_evidence_chain_graph_row(unanchored, params=EvidenceChainParams(top_k=3, beam_size=6))

        self.assertEqual(bridged_graph["selected_evidence_ids"][:2], ["E01", "E02"])
        self.assertEqual(unanchored_graph["selected_evidence_ids"][-1], "E02")

    def test_pipeline_trace_uses_pool_coordinates_and_fingerprint(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], base=0.95, oracle=True),
                _candidate("E02", atoms=["A2"], base=0.90),
                _candidate("E03", atoms=["A3"], base=0.85, oracle=True),
                _candidate("E04", atoms=[], relation="background", directness="context", base=0.30),
            ]
        )

        graph = build_evidence_chain_graph_row(row, params=EvidenceChainParams(top_k=3, beam_size=6))
        trace = graph["selection_trace"]
        pool = trace["candidate_pool"]

        for key in ("candidate_pool", "candidate_scores", "selector_ordered_indices", "oracle_ordered_indices", "fingerprint"):
            self.assertIn(key, trace)
        self.assertEqual(trace["fingerprint"], "432dfc970e75")
        self.assertEqual([item["candidate_idx"] for item in pool], list(range(len(pool))))
        self.assertEqual([item["candidate_idx"] for item in trace["candidate_scores"]], list(range(len(pool))))
        self.assertTrue(all(0 <= idx < len(pool) for idx in trace["selector_ordered_indices"]))
        self.assertTrue(all(0 <= idx < len(pool) for idx in trace["oracle_ordered_indices"]))
        selected_from_pool = [pool[idx]["evidence_id"] for idx in trace["selector_ordered_indices"]]
        self.assertEqual(selected_from_pool, graph["selected_evidence_ids"])

    def test_rule_step_anchor_prefers_core_evidence_over_background(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=[], relation="background", directness="context", base=0.99),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", base=0.70),
                _candidate("E03", atoms=[], relation="irrelevant", directness="none", base=0.98),
                _candidate("E04", atoms=["A2"], relation="support", directness="partial", base=0.60),
                _candidate("E05", atoms=[], relation="background", directness="context", base=0.50),
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))

        self.assertEqual(graph["selector_name"], RULE_STEP_CHAIN_SELECTOR)
        self.assertEqual(graph["selected_evidence_ids"][0], "E02")
        self.assertEqual(graph["selection_steps"][0]["rule"], "anchor_core")

    def test_rule_step_selector_name_tracks_budget(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", base=0.90),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", base=0.80),
                _candidate("E03", atoms=[], relation="background", directness="context", base=0.70),
                _candidate("E04", atoms=[], relation="irrelevant", directness="none", base=0.60),
                _candidate("E05", atoms=[], relation="irrelevant", directness="none", base=0.50),
                _candidate("E06", atoms=[], relation="irrelevant", directness="none", base=0.40),
                _candidate("E07", atoms=[], relation="irrelevant", directness="none", base=0.30),
                _candidate("E08", atoms=[], relation="irrelevant", directness="none", base=0.20),
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=8))

        self.assertEqual(rule_step_chain_selector_name(5, 8), "v0_6c_rule_step_adaptive5_8")
        self.assertEqual(graph["selector_name"], "v0_6c_rule_step_adaptive5_8")
        self.assertEqual(graph["selection_trace"]["selector_name"], "v0_6c_rule_step_adaptive5_8")

    def test_rule_step_p1_beats_p2_and_uses_claim_atom_order(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", base=0.99, source="report:1"),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", base=0.98, source="report:2"),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", base=0.80, source="report:3"),
                _candidate("E04", atoms=["A3"], relation="support", directness="partial", base=0.70, source="report:4"),
                _candidate("E05", atoms=[], relation="irrelevant", directness="none", base=0.10, source="report:5"),
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))

        self.assertEqual(graph["selected_evidence_ids"][:3], ["E01", "E02", "E04"])
        self.assertEqual([step["rule"] for step in graph["selection_steps"][:3]], ["anchor_core", "P1_new_atom_core", "P1_new_atom_core"])

    def test_rule_step_p2_requires_strong_edge_and_core_candidate(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.90),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.80),
                _candidate("E03", atoms=[], relation="background", directness="context", source="report:3", base=0.99),
                _candidate("E04", atoms=[], relation="irrelevant", directness="none", source="report:4", base=0.98),
                _candidate("E05", atoms=[], relation="background", directness="context", source="report:5", base=0.97),
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))

        self.assertEqual(graph["selected_evidence_ids"][1], "E02")
        self.assertEqual(graph["selection_steps"][1]["rule"], "P2_strong_edge_core")

    def test_rule_step_p3_requires_core_anchor(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", sent_idx=1, base=0.90),
                _candidate("E02", atoms=["A1"], relation="background", directness="context", source="report:1", sent_idx=2, base=0.80),
                _candidate("E03", atoms=[], relation="background", directness="context", source="report:9", sent_idx=9, base=0.99),
                _candidate("E04", atoms=[], relation="irrelevant", directness="none", source="report:1", sent_idx=3, base=0.98),
                _candidate("E05", atoms=[], relation="background", directness="context", source="report:8", sent_idx=8, base=0.10),
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))

        self.assertEqual(graph["selected_evidence_ids"][1], "E02")
        self.assertEqual(graph["selection_steps"][1]["rule"], "P3_bridge_context")
        self.assertNotEqual(graph["selected_evidence_ids"][1], "E03")

    def test_rule_step_fallback_fills_to_five_then_stops_without_rules(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", base=0.90),
                _candidate("E02", atoms=[], relation="irrelevant", directness="none", base=0.80),
                _candidate("E03", atoms=[], relation="irrelevant", directness="none", base=0.70),
                _candidate("E04", atoms=[], relation="background", directness="context", base=0.60),
                _candidate("E05", atoms=[], relation="irrelevant", directness="none", base=0.50),
                _candidate("E06", atoms=[], relation="irrelevant", directness="none", base=0.40),
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))

        self.assertEqual(len(graph["selected_evidence_ids"]), 5)
        self.assertEqual(graph["adaptive_stop_reason"], "reached_min_top_k_no_rule_candidate")
        self.assertTrue(any(step["fallback_used"] for step in graph["selection_steps"]))

    def test_rule_step_max_top_k_cap_and_trace_coordinates(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                *[
                    _candidate(f"E{i:02d}", atoms=["A1"], relation="support", directness="direct", source=f"report:{i}", base=0.99 - i * 0.01)
                    for i in range(2, 13)
                ],
            ]
        )

        graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))
        trace = graph["selection_trace"]
        pool = trace["candidate_pool"]

        self.assertEqual(len(graph["selected_evidence_ids"]), 10)
        self.assertEqual(trace["adaptive_evidence_count"], 10)
        self.assertEqual([pool[idx]["evidence_id"] for idx in trace["selector_ordered_indices"]], graph["selected_evidence_ids"])
        self.assertTrue(all(0 <= idx < len(pool) for idx in trace["selector_ordered_indices"]))
        self.assertIn("jaccard@5", trace)
        self.assertIn("jaccard@10", trace)
        self.assertNotIn("post_order", graph["chains"][0])

    def test_sufficiency_v0_6d_preserves_v0_6c_prefix_before_min_top_k(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A2"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:5", base=0.95),
                _candidate("E06", atoms=["A1"], relation="refute", directness="direct", source="report:6", base=0.94),
            ]
        )

        old_graph = build_rule_step_evidence_chain_graph_row(row, params=RuleStepEvidenceChainParams(min_top_k=5, max_top_k=10))
        new_graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )

        self.assertEqual(new_graph["selector_name"], SUFFICIENCY_CONTRADICTION_SELECTOR)
        self.assertEqual(new_graph["selected_evidence_ids"][:5], old_graph["selected_evidence_ids"][:5])

    def test_sufficiency_v0_6d_stops_at_five_without_contradiction(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A2"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:5", base=0.95),
                _candidate("E06", atoms=["A2"], relation="support", directness="direct", source="report:6", base=0.94),
                _candidate("E07", atoms=["A1"], relation="background", directness="context", source="report:1", sent_idx=2, base=0.93),
            ]
        )

        graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )

        self.assertEqual(len(graph["selected_evidence_ids"]), 5)
        self.assertEqual(graph["adaptive_stop_reason"], "sufficient_no_contradiction_candidate")
        self.assertTrue(graph["sufficiency_state"]["is_sufficient"])

    def test_sufficiency_v0_6d_continues_for_tension_counter_evidence(self) -> None:
        row = _row_with_atoms(
            [{"atom_id": "A1", "text": "Atom 1", "type": "other", "importance": 1.0}],
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A1"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:5", base=0.95),
                _candidate("E06", atoms=["A1"], relation="refute", directness="direct", source="report:6", base=0.94),
            ],
        )

        graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )
        trace = graph["selection_trace"]

        self.assertEqual(graph["selected_evidence_ids"][5], "E06")
        self.assertEqual(graph["selection_steps"][5]["rule"], "P2_tension_counter_core")
        self.assertTrue(graph["selection_steps"][5]["contradiction_continuation"])
        self.assertEqual(graph["contradiction_aware_additions"][0]["evidence_id"], "E06")
        self.assertIn("sufficiency_state", trace)
        self.assertIn("contradiction_aware_additions", trace)

    def test_sufficiency_v0_6d_prioritizes_uncovered_important_atom_after_min(self) -> None:
        row = _row_with_atoms(
            [{"atom_id": f"A{i}", "text": f"Atom {i}", "type": "other", "importance": 1.0} for i in range(1, 7)],
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A3"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A4"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A5"], relation="support", directness="direct", source="report:5", base=0.95),
                _candidate("E06", atoms=["A1"], relation="refute", directness="direct", source="report:6", base=0.94),
                _candidate("E07", atoms=["A6"], relation="support", directness="direct", source="report:7", base=0.50),
            ],
        )

        graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )

        self.assertEqual(graph["selected_evidence_ids"][5], "E07")
        self.assertEqual(graph["selection_steps"][5]["rule"], "P1_new_important_atom_core")
        self.assertEqual(graph["selection_steps"][5]["covered_new_atom_ids"], ["A6"])

    def test_sufficiency_v0_6d_does_not_use_fallback_after_min_top_k(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", base=0.90),
                _candidate("E02", atoms=[], relation="irrelevant", directness="none", base=0.80),
                _candidate("E03", atoms=[], relation="irrelevant", directness="none", base=0.70),
                _candidate("E04", atoms=[], relation="background", directness="context", base=0.60),
                _candidate("E05", atoms=[], relation="irrelevant", directness="none", base=0.50),
                _candidate("E06", atoms=[], relation="irrelevant", directness="none", base=0.40),
            ]
        )

        graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )

        self.assertEqual(len(graph["selected_evidence_ids"]), 5)
        self.assertEqual(graph["adaptive_stop_reason"], "insufficient_no_coverable_atom_candidate")
        self.assertTrue(any(step["fallback_used"] for step in graph["selection_steps"][:5]))
        self.assertFalse(any(step["fallback_used"] for step in graph["selection_steps"][5:]))

    def test_sufficiency_v0_6d_skips_duplicate_tension_and_caps_at_ten(self) -> None:
        one_atom = [{"atom_id": "A1", "text": "Atom 1", "type": "other", "importance": 1.0}]
        duplicate_row = _row_with_atoms(
            one_atom,
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99, duplicate="G1"),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A1"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:5", base=0.95),
                _candidate("E06", atoms=["A1"], relation="refute", directness="direct", source="report:6", base=0.94, duplicate="G1"),
            ],
        )
        capped_row = _row_with_atoms(
            one_atom,
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A1"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:5", base=0.95),
                *[
                    _candidate(f"E{i:02d}", atoms=["A1"], relation="refute", directness="direct", source=f"report:{i}", base=0.95 - i * 0.01)
                    for i in range(6, 13)
                ],
            ],
        )

        duplicate_graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            duplicate_row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )
        capped_graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            capped_row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )

        self.assertNotIn("E06", duplicate_graph["selected_evidence_ids"])
        self.assertEqual(duplicate_graph["adaptive_stop_reason"], "sufficient_no_contradiction_candidate")
        self.assertEqual(len(capped_graph["selected_evidence_ids"]), 10)
        self.assertEqual(capped_graph["adaptive_stop_reason"], "reached_max_top_k")

    def test_sufficiency_v0_6d_trace_coordinates_and_fields(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A2"], relation="support", directness="direct", source="report:4", base=0.96),
                _candidate("E05", atoms=["A1"], relation="support", directness="direct", source="report:5", base=0.95),
                _candidate("E06", atoms=["A1"], relation="refute", directness="direct", source="report:6", base=0.94),
            ]
        )

        graph = build_sufficiency_contradiction_evidence_chain_graph_row(
            row,
            params=SufficiencyContradictionEvidenceChainParams(min_top_k=5, max_top_k=10),
        )
        trace = graph["selection_trace"]
        pool = trace["candidate_pool"]

        self.assertEqual(trace["selector_name"], SUFFICIENCY_CONTRADICTION_SELECTOR)
        self.assertEqual(trace["adaptive_policy"], "sufficiency_contradiction_v0_6d")
        self.assertIn("sufficiency_state", graph)
        self.assertIn("sufficiency_state", trace)
        self.assertIn("sufficiency_important_atom_threshold", trace)
        self.assertIn("sufficiency_weighted_coverage_threshold", trace)
        self.assertEqual([pool[idx]["evidence_id"] for idx in trace["selector_ordered_indices"]], graph["selected_evidence_ids"])
        self.assertEqual([step["evidence_id"] for step in graph["selection_steps"]], graph["selected_evidence_ids"])
        self.assertTrue(all(0 <= idx < len(pool) for idx in trace["selector_ordered_indices"]))

    def test_budgeted_marginal_new_atom_coverage_beats_high_prior_duplicate(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99, duplicate="G1"),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.98, duplicate="G1"),
                _candidate("E03", atoms=["A2"], relation="support", directness="direct", source="report:3", base=0.30),
                _candidate("E04", atoms=["A3"], relation="support", directness="direct", source="report:4", base=0.29),
            ]
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams(min_top_k=3, max_top_k=3))

        self.assertEqual(graph["selected_evidence_ids"], ["E01", "E03", "E04"])
        self.assertNotIn("E02", graph["selected_evidence_ids"])
        self.assertGreater(graph["selection_steps"][1]["component_deltas"]["coverage"], 0.0)

    def test_budgeted_marginal_stops_after_min_when_sufficient_and_low_gain(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A3"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A1"], relation="support", directness="direct", source="report:4", base=0.96, duplicate="G1"),
            ]
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams(min_top_k=3, max_top_k=10))

        self.assertEqual(len(graph["selected_evidence_ids"]), 3)
        self.assertEqual(graph["adaptive_stop_reason"], "sufficient_low_marginal_gain")
        self.assertGreaterEqual(graph["objective_final_components"]["coverage"], 0.80)

    def test_budgeted_marginal_continues_after_min_when_coverage_is_insufficient(self) -> None:
        row = _row_with_atoms(
            [{"atom_id": f"A{i}", "text": f"Atom {i}", "type": "other", "importance": 1.0} for i in range(1, 6)],
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A3"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A4"], relation="support", directness="direct", source="report:4", base=0.15),
                _candidate("E05", atoms=["A5"], relation="support", directness="direct", source="report:5", base=0.14),
            ],
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams(min_top_k=3, max_top_k=10))

        self.assertIn("E04", graph["selected_evidence_ids"][3:])
        self.assertGreater(graph["selection_steps"][3]["component_deltas"]["coverage"], 0.0)
        self.assertGreaterEqual(graph["objective_final_components"]["coverage"], 0.80)

    def test_budgeted_marginal_duplicate_penalty_is_soft_but_avoids_duplicate_when_possible(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99, duplicate="G1"),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.98, duplicate="G1"),
                _candidate("E03", atoms=["A2"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=["A3"], relation="support", directness="direct", source="report:4", base=0.96),
            ]
        )
        duplicate_only = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99, duplicate="G1"),
                _candidate("E02", atoms=["A1"], relation="support", directness="direct", source="report:2", base=0.98, duplicate="G1"),
                _candidate("E03", atoms=["A1"], relation="support", directness="direct", source="report:3", base=0.97, duplicate="G1"),
            ]
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams(min_top_k=3, max_top_k=3))
        duplicate_graph = build_budgeted_marginal_chain_graph_row(duplicate_only, params=BudgetedMarginalChainParams(min_top_k=3, max_top_k=3))

        self.assertEqual(graph["selected_evidence_ids"], ["E01", "E03", "E04"])
        self.assertEqual(duplicate_graph["selected_evidence_ids"], ["E01", "E02", "E03"])
        self.assertLess(duplicate_graph["objective_final_components"]["redundancy_penalty"], 0.0)

    def test_budgeted_marginal_conditional_tension_requires_core_shared_atom_pair(self) -> None:
        core_tension = _row_with_atoms(
            [{"atom_id": "A1", "text": "Atom 1", "type": "other", "importance": 1.0}],
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A1"], relation="refute", directness="direct", source="report:2", base=0.98),
            ],
        )
        context_tension = _row_with_atoms(
            [{"atom_id": "A1", "text": "Atom 1", "type": "other", "importance": 1.0}],
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A1"], relation="refute", directness="context", source="report:2", base=0.98),
            ],
        )

        core_graph = build_budgeted_marginal_chain_graph_row(core_tension, params=BudgetedMarginalChainParams(min_top_k=2, max_top_k=2))
        context_graph = build_budgeted_marginal_chain_graph_row(context_tension, params=BudgetedMarginalChainParams(min_top_k=2, max_top_k=2))

        self.assertGreater(core_graph["objective_final_components"]["conditional_tension_gain"], 0.0)
        self.assertEqual(context_graph["objective_final_components"]["conditional_tension_gain"], 0.0)

    def test_budgeted_marginal_bridge_context_can_beat_unanchored_background(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", sent_idx=1, base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", sent_idx=1, base=0.98),
                _candidate("E03", atoms=["A1"], relation="background", directness="context", source="report:1", sent_idx=2, base=0.20),
                _candidate("E04", atoms=[], relation="background", directness="context", source="report:9", sent_idx=9, base=0.90),
            ]
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams(min_top_k=3, max_top_k=3))

        self.assertEqual(graph["selected_evidence_ids"][2], "E03")
        self.assertGreater(graph["selection_steps"][2]["component_deltas"]["bridge_context_gain"], 0.0)

    def test_budgeted_marginal_trace_coordinates_and_objective_fields(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99, oracle=True),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A3"], relation="support", directness="direct", source="report:3", base=0.97, oracle=True),
                _candidate("E04", atoms=[], relation="irrelevant", directness="none", source="report:4", base=0.96),
            ]
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams())
        trace = graph["selection_trace"]
        pool = trace["candidate_pool"]

        self.assertEqual(budgeted_marginal_chain_selector_name(3, 10), BUDGETED_MARGINAL_SELECTOR)
        self.assertEqual(trace["selector_name"], BUDGETED_MARGINAL_SELECTOR)
        self.assertEqual(trace["adaptive_policy"], "budgeted_marginal_v0_7")
        self.assertIn("objective_weights", trace)
        self.assertIn("objective_final_components", trace)
        self.assertEqual([pool[idx]["evidence_id"] for idx in trace["selector_ordered_indices"]], graph["selected_evidence_ids"])
        self.assertEqual([step["evidence_id"] for step in graph["selection_steps"]], graph["selected_evidence_ids"])
        self.assertTrue(all("marginal_gain" in step and "component_deltas" in step for step in graph["selection_steps"]))
        self.assertTrue(all(0 <= idx < len(pool) for idx in trace["selector_ordered_indices"]))

    def test_budgeted_marginal_cli_accepts_tight_objective_weight_overrides(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "build_evidence_chain_graph_v0_7.py",
                "--objective-background-or-irrelevant",
                "0.24",
                "--objective-length",
                "0.08",
            ],
        ):
            args = graph_cli.parse_args()

        params = graph_cli._budgeted_params_from_args(args)

        self.assertEqual(params.objective_weights.background_or_irrelevant, 0.24)
        self.assertEqual(params.objective_weights.length, 0.08)

    def test_budgeted_marginal_summary_uses_row_selector_name(self) -> None:
        row = _row(
            [
                _candidate("E01", atoms=["A1"], relation="support", directness="direct", source="report:1", base=0.99),
                _candidate("E02", atoms=["A2"], relation="support", directness="direct", source="report:2", base=0.98),
                _candidate("E03", atoms=["A3"], relation="support", directness="direct", source="report:3", base=0.97),
                _candidate("E04", atoms=[], relation="irrelevant", directness="none", source="report:4", base=0.96),
                _candidate("E05", atoms=[], relation="irrelevant", directness="none", source="report:5", base=0.95),
            ]
        )

        graph = build_budgeted_marginal_chain_graph_row(row, params=BudgetedMarginalChainParams(min_top_k=5, max_top_k=10))
        summary = summarize_budgeted_marginal_chain_graph_rows([graph])

        self.assertEqual(graph["selector_name"], "v0_7_budgeted_marginal_chain_adaptive5_10")
        self.assertEqual(summary["selector_name"], graph["selector_name"])

    def test_oracle_zero_step_is_preserved_and_rendered_as_one_based_badge(self) -> None:
        graph = build_evidence_chain_graph_row(_row([_candidate("E01", atoms=["A1"], base=0.9, oracle=True)]), params=EvidenceChainParams(top_k=1))
        args = Namespace(chain_graph="chain_graph.jsonl", max_candidates=20, max_chains=8)
        oracle_node = next(node for node in graph["evidence_nodes"] if node["node_id"] == "E01")

        self.assertEqual(oracle_node["oracle_step"], 0)

        oracle_node["oracle_step"] = -1
        oracle_node["candidate"]["oracle_step"] = -1
        html = render_html(graph, args=args, translations={})

        self.assertIn("ORACLE 1", html)

    def test_renderer_outputs_svg_chain_and_tables(self) -> None:
        graph = build_evidence_chain_graph_row(_row([_candidate("E01", atoms=["A1"], base=0.9), _candidate("E02", atoms=["A2"], base=0.8)]), params=EvidenceChainParams(top_k=2))
        args = Namespace(chain_graph="chain_graph.jsonl", max_candidates=20, max_chains=8)

        html = render_html(graph, args=args, translations={})

        self.assertIn("<svg", html)
        self.assertIn("Selected Chain", html)
        self.assertIn("Candidates", html)
        self.assertIn("Edges", html)
        self.assertIn("CH01", html)

    def test_renderer_embeds_zh_translation_in_svg_graph(self) -> None:
        graph = build_evidence_chain_graph_row(_row([_candidate("E01", atoms=["A1"], base=0.9)]), params=EvidenceChainParams(top_k=1))
        args = Namespace(chain_graph="chain_graph.jsonl", max_candidates=20, max_chains=8)
        long_evidence_zh = "证据一说明相关内容，并且需要在卡片中完整显示，不应该因为中文模式而被过早截断。"

        html = render_html(
            graph,
            args=args,
            translations={
                "claim": "预算增加了。",
                "atom:A1:text": "预算增加。",
                "evidence:E01:text": long_evidence_zh,
            },
        )

        self.assertIn("预算增加。", html)
        self.assertIn("证据一说明相关内容", html)
        self.assertIn("模式而被过早截断。", html)
        self.assertIn('data-w="520.0"', html)
        self.assertIn("id=\"chainGraphSvg\"", html)

    def test_svg_wrap_breaks_long_chinese_token_after_prefix(self) -> None:
        lines = wrap_text_for_svg(
            "A1 · 参议员凯哈根在二零一四年错过了参议院军事委员会一半的听证会",
            18,
            max_lines=10,
        )

        self.assertGreater(len(lines), 2)
        self.assertTrue(all(visual_width(line) <= 18 for line in lines if line))

    def test_translate_zh_requires_api_key_when_cache_missing(self) -> None:
        graph = build_evidence_chain_graph_row(_row([_candidate("E01", atoms=["A1"], base=0.9)]), params=EvidenceChainParams(top_k=1))
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                translate_zh=True,
                force_translate=False,
                translation_cache="",
                translation_api_key_env="MISSING_DEEPSEEK_API_KEY_FOR_TEST",
                translation_model="deepseek-v4-flash",
                translation_base_url="https://api.deepseek.com",
                translation_batch_chars=7000,
                translation_max_tokens=4096,
                translation_thinking_type="disabled",
                translation_max_retries=0,
                translation_retry_base_sleep=0.0,
                translation_timeout=1.0,
                max_candidates=20,
                max_chains=8,
            )
            os.environ.pop("MISSING_DEEPSEEK_API_KEY_FOR_TEST", None)

            with self.assertRaises(RuntimeError):
                load_or_build_translations(graph, args=args, output_path=Path(tmp) / "case.html")


def _row(candidates: list[dict], *, gold_label: str = "true") -> dict:
    return {
        "event_id": "case-1.json",
        "claim": "The budget increased and the tax rate fell.",
        "gold_label": gold_label,
        "oracle_ordered_keys": ["E01 text", "E03 text"],
        "evidence_map": {
            "claim_atoms": [
                {"atom_id": "A1", "text": "The budget increased.", "type": "quantity", "importance": 1.0},
                {"atom_id": "A2", "text": "The tax rate fell.", "type": "quantity", "importance": 1.0},
                {"atom_id": "A3", "text": "The change happened this year.", "type": "date", "importance": 0.5},
            ]
        },
        "candidates": candidates,
    }


def _row_with_atoms(atoms: list[dict], candidates: list[dict], *, gold_label: str = "true") -> dict:
    row = _row(candidates, gold_label=gold_label)
    row["evidence_map"] = {"claim_atoms": atoms}
    return row


def _candidate(
    evidence_id: str,
    *,
    atoms: list[str],
    relation: str = "support",
    directness: str = "direct",
    source: str = "report:1",
    sent_idx: int = 1,
    base: float = 0.8,
    duplicate: str = "",
    oracle: bool = False,
    confidence: float = 1.0,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "candidate_uid": f"uid-{evidence_id}",
        "candidate_key": f"{evidence_id} text",
        "text": f"{evidence_id} text says something relevant.",
        "covered_atom_ids": atoms,
        "map_relation": relation,
        "map_directness": directness,
        "map_evidence_role": "primary_support" if relation == "support" else "qualifying_context",
        "key_spans": [f"{evidence_id} span"],
        "duplicate_group": duplicate,
        "source_group": source,
        "report_id": source.split(":", 1)[-1],
        "sent_idx": sent_idx,
        "evidence_map_base_score": base,
        "fusion_refit_score": base,
        "evidence_map_quality_score": 1.0 if relation != "background" else 0.3,
        "map_confidence": confidence,
        "union_pool_rank": int(evidence_id[-2:]),
        "oracle_selected": oracle,
        "oracle_step": 0 if oracle else -1,
    }


if __name__ == "__main__":
    unittest.main()
