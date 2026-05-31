from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from fact_checking.selectors.evidence_chain_graph import (
    CHAIN_SELECTOR,
    EvidenceChainParams,
    build_evidence_chain_graph_row,
)
from scripts.phase5_selectors.visualize.render_evidence_chain_graph_html import (
    load_or_build_translations,
    render_html,
    visual_width,
    wrap_text_for_svg,
)


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
        "union_pool_rank": int(evidence_id[-2:]),
        "oracle_selected": oracle,
        "oracle_step": 0 if oracle else -1,
    }


if __name__ == "__main__":
    unittest.main()
