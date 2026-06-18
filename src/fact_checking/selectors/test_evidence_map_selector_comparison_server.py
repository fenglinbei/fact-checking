from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.phase5_selectors.visualize.render_evidence_map_selector_comparison_html import (
    default_candidate_features_path,
    default_coverage_diff_path,
    default_left_chain_graph_path,
    default_left_trace_path,
    default_raw_data_path,
    default_right_chain_graph_path,
    default_right_trace_path,
)
from scripts.phase5_selectors.visualize.serve_evidence_map_selector_comparison import (
    ComparisonStore,
    index_html,
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

                store = ComparisonStore.load(Path(tmp), sources=["rawfc", "liar_raw"], splits=["val"], max_candidates=20)
                results = store.search_cases("city budget", split="val", limit=5)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["event_id"], "case.json")
        self.assertEqual(results[0]["split"], "val")
        self.assertEqual(results[0]["gold_label"], "mostly-true")
        self.assertEqual(results[0]["coverage_label"], "uncovered")

    def test_store_loads_source_profiles_and_filters_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val", source="rawfc", event_id="rawfc-case", claim="RAWFC source claim.")
                _write_fixture_split("val", source="liar_raw", event_id="liar-case.json", claim="LIAR-RAW source claim.")

                store = ComparisonStore.load(Path(tmp), sources=["rawfc", "liar_raw"], splits=["val"], max_candidates=20)
                rawfc_results = store.search_cases("source claim", source="rawfc", split="val", limit=5)
                liar_results = store.search_cases("source claim", source="liar_raw", split="val", limit=5)
                liar_html = store.render_case(
                    source="liar_raw",
                    split="val",
                    event_id="liar-case.json",
                    left_label="left",
                    right_label="right",
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual([item["event_id"] for item in rawfc_results], ["rawfc-case"])
        self.assertEqual(rawfc_results[0]["source"], "rawfc")
        self.assertEqual([item["event_id"] for item in liar_results], ["liar-case.json"])
        self.assertEqual(liar_results[0]["source"], "liar_raw")
        self.assertIn("data/raw/LIAR-RAW/val.json", liar_html)
        self.assertIn("LIAR-RAW source claim.", liar_html)

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

                store = ComparisonStore.load(Path(tmp), sources=["rawfc", "liar_raw"], splits=["val"], max_candidates=20)
                html = store.render_case(split="val", event_id="case", left_label="left", right_label="right")
            finally:
                os.chdir(old_cwd)

        self.assertIn("Evidence map selector comparison: case.json", html)
        self.assertIn("The city budget increased.", html)
        self.assertIn("Coverage Diff", html)

    def test_store_renders_chain_graph_edges_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val", include_chain_graph=True)

                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)
                html = store.render_case(split="val", event_id="case", left_label="left", right_label="right")
            finally:
                os.chdir(old_cwd)

        self.assertIn("selected evidence chain", html)
        self.assertIn('data-edge-type="tension"', html)
        self.assertIn("Evidence-Evidence Edges", html)

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

    def test_index_html_has_collapsible_sidebar_without_duplicate_translation_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")
                _write_fixture_split("val", source="liar_raw", event_id="liar-case.json", claim="LIAR-RAW source claim.")
                store = ComparisonStore.load(Path(tmp), sources=["rawfc", "liar_raw"], splits=["val"], max_candidates=20)
                html = index_html(store, base_path="/evidence-map", query={"token": ["secret"]}, translation_enabled=False)
            finally:
                os.chdir(old_cwd)

        self.assertIn("data-sidebar-toggle", html)
        self.assertIn("data-sidebar-state", html)
        self.assertIn('id="source"', html)
        self.assertIn('value="rawfc"', html)
        self.assertIn('value="liar_raw"', html)
        self.assertIn("source: source.value", html)
        self.assertIn("source: item.source", html)
        self.assertNotIn('id="translate"', html)
        self.assertNotIn("translateStatus", html)
        self.assertIn("localStorage", html)

    def test_translate_case_requires_live_translation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")
                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)

                with self.assertRaisesRegex(PermissionError, "Live translation is disabled"):
                    store.translate_case(split="val", event_id="case", left_label="left", right_label="right", enabled=False)
            finally:
                os.chdir(old_cwd)

    def test_translate_case_returns_cached_translation_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val")
                _write_json(
                    Path("outputs/analysis/map/v0.7") / "val_evidence_map_compare_case_left_vs_right.zh.json",
                    {"translations": _complete_cached_translations()},
                )
                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)

                result = store.translate_case(split="val", event_id="case", left_label="left", right_label="right", enabled=True)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["translation_count"], len(_complete_cached_translations()))
        self.assertTrue(str(result["cache_path"]).endswith("val_evidence_map_compare_case_left_vs_right.zh.json"))

    def test_translate_case_fills_missing_chain_graph_translation_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_fixture_split("val", include_chain_graph=True)
                store = ComparisonStore.load(Path(tmp), splits=["val"], max_candidates=20)

                def fake_translate(items, *, args, api_key):
                    return {key: f"ZH {value}" for key, value in items.items()}, {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    }

                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), patch(
                    "render_evidence_map_claim_html.translate_items_zh",
                    side_effect=fake_translate,
                ):
                    result = store.translate_case(
                        split="val",
                        event_id="case",
                        left_label="left",
                        right_label="right",
                        enabled=True,
                    )
                cache_path = Path(result["cache_path"])
                translations = json.loads(cache_path.read_text(encoding="utf-8"))["translations"]
            finally:
                os.chdir(old_cwd)

        self.assertIn("atom:A1:text", translations)
        self.assertIn("evidence:E01:text", translations)
        self.assertIn("evidence:E02:text", translations)
        self.assertGreater(result["translation_count"], 1)

    def test_web_launcher_enables_live_translation_by_default(self) -> None:
        launcher = Path("scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('ENABLE_LIVE_TRANSLATION="${ENABLE_LIVE_TRANSLATION:-1}"', launcher)
        self.assertIn('args+=(--enable-live-translation)', launcher)

    def test_web_launcher_loads_project_env_without_overriding_shell_values(self) -> None:
        launcher_source = Path("scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher_path = root / "scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh"
            launcher_path.parent.mkdir(parents=True)
            launcher_path.write_text(launcher_source, encoding="utf-8")
            launcher_path.chmod(0o755)
            (root / ".env").write_text(
                "DEEPSEEK_API_KEY=fixture-secret\nEVIDENCE_MAP_TOKEN=token-from-env-file\n",
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'DEEPSEEK_API_KEY=%s\\n' \"${DEEPSEEK_API_KEY:+SET}\"\n"
                "printf 'ARGS=%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = {
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                "EVIDENCE_MAP_TOKEN": "token-from-shell",
            }
            result = subprocess.run(
                ["bash", str(launcher_path)],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("DEEPSEEK_API_KEY=SET", result.stdout)
        self.assertIn("--token token-from-shell", result.stdout)

    def test_web_launcher_honors_python_bin_override(self) -> None:
        launcher_source = Path("scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher_path = root / "scripts/phase5_selectors/run/run_evidence_map_selector_comparison_web.sh"
            launcher_path.parent.mkdir(parents=True)
            launcher_path.write_text(launcher_source, encoding="utf-8")
            launcher_path.chmod(0o755)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "custom-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'PYTHON_BIN_USED=%s\\n' \"$0\"\n"
                "printf 'ARGS=%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            path_python = fake_bin / "python"
            path_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'PATH_PYTHON_USED=%s\\n' \"$0\"\n"
                "printf 'ARGS=%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            path_python.chmod(0o755)

            result = subprocess.run(
                ["bash", str(launcher_path)],
                cwd=root,
                env={
                    "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                    "EVIDENCE_MAP_TOKEN": "token-from-shell",
                    "PYTHON_BIN": str(fake_python),
                },
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn(f"PYTHON_BIN_USED={fake_python}", result.stdout)
        self.assertIn("--token token-from-shell", result.stdout)


def _write_fixture_split(
    split: str,
    *,
    source: str = "rawfc",
    event_id: str = "case.json",
    claim: str = "The city budget increased.",
    include_chain_graph: bool = False,
) -> None:
    row = _row()
    row["event_id"] = event_id
    row["claim"] = claim
    left_trace = _left_trace()
    left_trace["event_id"] = event_id
    right_trace = _right_trace()
    right_trace["event_id"] = event_id
    raw_row = _raw_row()
    raw_row["event_id"] = event_id
    raw_row["explain"] = f"The raw explanation describes {claim}"
    coverage_diff = _coverage_diff()
    coverage_diff["event_id"] = event_id

    _write_jsonl(default_candidate_features_path(split, source=source), [row])
    _write_jsonl(default_left_trace_path(split, source=source), [left_trace])
    _write_jsonl(default_right_trace_path(split, source=source), [right_trace])
    if include_chain_graph:
        left_graph = _chain_graph_row()
        left_graph["event_id"] = event_id
        right_graph = _chain_graph_row()
        right_graph["event_id"] = event_id
        _write_jsonl(default_left_chain_graph_path(split, source=source), [left_graph])
        _write_jsonl(default_right_chain_graph_path(split, source=source), [right_graph])
    _write_json(default_raw_data_path(split, source=source), {"case": raw_row})
    _write_jsonl(default_coverage_diff_path(split, source=source), [coverage_diff])
    _write_json(
        Path("outputs/analysis/map/v0.7") / f"{split}_evidence_map_compare_case_left_vs_right.zh.json",
        {"translations": {"claim": "城市预算增加。"}},
    )


def _complete_cached_translations() -> dict[str, str]:
    return {
        "claim": "城市预算增加。",
        "atom:A1:text": "城市有预算。",
        "atom:A2:text": "预算增加了。",
        "candidate:candidate_uid:uid-E01:title": "城市预算存在。",
        "candidate:candidate_uid:uid-E01:text": "城市预算存在。",
        "candidate:candidate_uid:uid-E01:span:0": "城市预算存在",
        "candidate:candidate_uid:uid-E02:title": "预算去年增长。",
        "candidate:candidate_uid:uid-E02:text": "预算去年增长。",
        "candidate:candidate_uid:uid-E02:span:0": "预算去年增长",
        "gold_explain": "原始解释描述了预算声明。",
        "coverage_preview:1:text": "顶部来源证据提到了预算。",
    }


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
            {"node_id": "A2", "atom_id": "A2", "text": "The budget increased.", "importance": 1.0, "atom_type": "predicate"},
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
                "node_id": "E02",
                "evidence_id": "E02",
                "text": "The budget rose last year.",
                "relation": "qualify",
                "directness": "partial",
                "covered_atom_ids": ["A2"],
                "source_group": "report:2",
                "oracle_selected": False,
            },
        ],
        "edges": [
            {"edge_type": "claim_has_atom", "source": "C0", "target": "A1", "weight": 1.0},
            {"edge_type": "claim_has_atom", "source": "C0", "target": "A2", "weight": 1.0},
            {"edge_type": "evidence_covers_atom", "source": "E01", "target": "A1", "weight": 0.8, "atom_ids": ["A1"], "relation": "support"},
            {"edge_type": "evidence_covers_atom", "source": "E02", "target": "A2", "weight": 0.7, "atom_ids": ["A2"], "relation": "qualify"},
            {
                "edge_type": "tension",
                "source": "E01",
                "target": "E02",
                "weight": 0.85,
                "atom_ids": ["A2"],
                "reason": "shared atom with conflicting relation",
            },
        ],
        "selected_evidence_ids": ["E01", "E02"],
        "selected_chain_id": "CH01",
        "chains": [
            {
                "chain_id": "CH01",
                "chain_score": 0.8,
                "weighted_atom_coverage": 1.0,
                "direct_or_partial_rate": 1.0,
                "positive_pair_edge_density": 1.0,
                "evidence_ids": ["E01", "E02"],
                "covered_atom_ids": ["A1", "A2"],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
