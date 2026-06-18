from __future__ import annotations

import unittest
import sys
from unittest.mock import patch

from fact_checking.selectors.evidence_map_selector import (
    ATOM_FACTS_ABC_PROMPT_VERSION,
    ATOM_FACTS_PROMPT_VERSION,
    COMPACT_PROMPT_VERSION,
    EVIDENCE_MAP_BASE_ONLY_SELECTOR,
    EVIDENCE_MAP_SELECTOR,
    EvidenceMapParams,
    attach_event_base_scores,
    atom_quality_diagnostics,
    audit_teacher_prompt,
    build_all_evidence_map_traces,
    build_teacher_messages,
    evidence_items_fingerprint,
    prepare_evidence_map_candidate_rows,
    select_evidence_map_topk,
    summarize_atom_quality_rows,
    validate_evidence_map_payload,
)
from scripts.phase5_selectors.build.annotate_evidence_maps_deepseek import _build_jobs
from scripts.phase5_selectors.build.build_evidence_map_verifier_data import _render_until_fit
from scripts.phase5_selectors.eval import eval_evidence_map_selector_v0_5a as eval_map_cli


class EvidenceMapSelectorTest(unittest.TestCase):
    def test_teacher_prompt_excludes_forbidden_metadata(self) -> None:
        row = prepare_evidence_map_candidate_rows([_event()], candidate_top_n=2)[0]

        system_prompt, user_prompt = build_teacher_messages(row)

        prompt = system_prompt + "\n" + user_prompt
        self.assertIn("E01", prompt)
        self.assertIn("Evidence 1 states", prompt)
        self.assertNotIn("event-1", prompt)
        self.assertNotIn("uid-1", prompt)
        self.assertNotIn("oracle_selected", prompt)
        audit_teacher_prompt(row, system_prompt=system_prompt, user_prompt=user_prompt)

    def test_compact_v0_6b_prompt_truncates_evidence_and_excludes_forbidden_fields(self) -> None:
        event = _event()
        event["candidates"][0]["candidate_uid"] = "secret-uid-1"
        event["candidates"][0]["candidate_key"] = "secret-key-1"
        event["candidates"][0]["text"] = "HEAD " + ("middle " * 80) + "TAIL_SENTINEL"
        row = prepare_evidence_map_candidate_rows([event], candidate_top_n=1)[0]

        system_prompt, user_prompt = build_teacher_messages(
            row,
            prompt_version=COMPACT_PROMPT_VERSION,
            max_evidence_chars=120,
        )

        prompt = system_prompt + "\n" + user_prompt
        self.assertIn("E01:", prompt)
        self.assertIn("HEAD", prompt)
        self.assertIn("TAIL_SENTINEL", prompt)
        self.assertNotIn("secret-uid-1", prompt)
        self.assertNotIn("secret-key-1", prompt)
        for forbidden in ("candidate_uid", "candidate_key", "gold_label", "oracle_selected", "oracle", "scores"):
            self.assertNotIn(forbidden, prompt)
        audit_teacher_prompt(row, system_prompt=system_prompt, user_prompt=user_prompt)

    def test_atom_facts_prompt_requires_complete_proposition_atoms_and_excludes_metadata(self) -> None:
        event = _event()
        event["claim"] = "Says Sen. Kay Hagan has missed half of the Senate Armed Services Committee hearings in 2014."
        event["candidates"][0]["candidate_uid"] = "secret-uid-1"
        event["candidates"][0]["candidate_key"] = "secret-key-1"
        row = prepare_evidence_map_candidate_rows([event], candidate_top_n=1)[0]

        system_prompt, user_prompt = build_teacher_messages(
            row,
            prompt_version=ATOM_FACTS_PROMPT_VERSION,
            max_evidence_chars=120,
        )

        prompt = system_prompt + "\n" + user_prompt
        self.assertIn("complete proposition", prompt)
        self.assertIn("subject, predicate, object", prompt)
        self.assertIn("Do not create standalone entity, date, or quantity atoms", prompt)
        self.assertIn("usually one atom", prompt)
        self.assertNotIn("Keep atoms small", prompt)
        self.assertNotIn("secret-uid-1", prompt)
        self.assertNotIn("secret-key-1", prompt)
        audit_teacher_prompt(row, system_prompt=system_prompt, user_prompt=user_prompt)

    def test_atom_facts_abc_prompt_reuses_atom_facts_schema_and_keys_by_evidence_fingerprint(self) -> None:
        event = _event()
        row = prepare_evidence_map_candidate_rows([event], candidate_top_n=1)[0]
        changed_event = _event()
        changed_event["candidates"][0]["text"] = "Evidence 1 states a different ABC chunk."
        changed_row = prepare_evidence_map_candidate_rows([changed_event], candidate_top_n=1)[0]

        system_prompt, user_prompt = build_teacher_messages(
            row,
            prompt_version=ATOM_FACTS_ABC_PROMPT_VERSION,
            max_evidence_chars=120,
        )
        prompt = system_prompt + "\n" + user_prompt

        self.assertIn("complete proposition", prompt)
        self.assertIn("subject, predicate, object", prompt)
        self.assertEqual(evidence_items_fingerprint(row["evidence_items"]), row["evidence_items_fingerprint"])
        self.assertNotEqual(row["evidence_items_fingerprint"], changed_row["evidence_items_fingerprint"])

        job = _build_jobs([row], model="deepseek-v4-flash", prompt_version=ATOM_FACTS_ABC_PROMPT_VERSION)[0]
        changed_job = _build_jobs([changed_row], model="deepseek-v4-flash", prompt_version=ATOM_FACTS_ABC_PROMPT_VERSION)[0]
        self.assertNotEqual(job.annotation_key, changed_job.annotation_key)

    def test_schema_validation_clamps_and_fills_alignments(self) -> None:
        payload = {
            "claim_atoms": [{"atom_id": "A1", "text": "Budget increased", "type": "quantity", "importance": 9}],
            "candidate_alignments": [
                {
                    "evidence_id": "E01",
                    "covered_atom_ids": ["A1", "A9"],
                    "relation": "bad",
                    "directness": "direct",
                    "evidence_role": "bad",
                    "key_spans": ["Budget increased by 10 percent"],
                    "duplicate_group": "G1",
                    "confidence": 2.0,
                }
            ],
        }

        valid = validate_evidence_map_payload(payload, valid_evidence_ids=["E01", "E02"])

        self.assertEqual(valid["claim_atoms"][0]["importance"], 1.0)
        first = {row["evidence_id"]: row for row in valid["candidate_alignments"]}["E01"]
        self.assertEqual(first["covered_atom_ids"], ["A1"])
        self.assertEqual(first["relation"], "irrelevant")
        self.assertEqual(first["confidence"], 1.0)
        second = {row["evidence_id"]: row for row in valid["candidate_alignments"]}["E02"]
        self.assertEqual(second["directness"], "none")

    def test_atom_quality_diagnostics_flags_10004_style_fragments(self) -> None:
        diagnostics = atom_quality_diagnostics(
            [
                {"atom_id": "A1", "text": "Sen. Kay Hagan", "type": "entity", "importance": 1.0},
                {"atom_id": "A2", "text": "has missed half", "type": "quantity", "importance": 1.0},
                {"atom_id": "A3", "text": "of the Senate Armed Services Committee's hearings", "type": "entity", "importance": 1.0},
                {"atom_id": "A4", "text": "in 2014", "type": "date", "importance": 0.8},
            ]
        )

        self.assertEqual(diagnostics["atom_count"], 4)
        self.assertEqual(diagnostics["fragment_atom_count"], 4)
        self.assertEqual(diagnostics["fragment_atom_ids"], ["A1", "A2", "A3", "A4"])
        self.assertIn("standalone_entity_or_modifier", diagnostics["issues_by_atom"]["A1"])
        self.assertIn("preposition_start", diagnostics["issues_by_atom"]["A3"])
        self.assertIn("preposition_start", diagnostics["issues_by_atom"]["A4"])

    def test_atom_quality_diagnostics_accepts_complete_proposition_atom(self) -> None:
        diagnostics = atom_quality_diagnostics(
            [
                {
                    "atom_id": "A1",
                    "text": "Sen. Kay Hagan missed half of the Senate Armed Services Committee hearings in 2014.",
                    "type": "quantity",
                    "importance": 1.0,
                }
            ]
        )

        self.assertEqual(diagnostics["atom_count"], 1)
        self.assertEqual(diagnostics["fragment_atom_count"], 0)
        self.assertEqual(diagnostics["fragment_atom_ids"], [])

    def test_summarize_atom_quality_rows_counts_fragment_cases(self) -> None:
        summary = summarize_atom_quality_rows(
            [
                {
                    "event_id": "10004.json",
                    "evidence_map": {
                        "claim_atoms": [
                            {"atom_id": "A1", "text": "Sen. Kay Hagan", "type": "entity"},
                            {"atom_id": "A2", "text": "has missed half", "type": "quantity"},
                        ]
                    },
                },
                {
                    "event_id": "ok.json",
                    "evidence_map": {
                        "claim_atoms": [
                            {
                                "atom_id": "A1",
                                "text": "Sen. Kay Hagan missed half of the Senate Armed Services Committee hearings in 2014.",
                                "type": "quantity",
                            }
                        ]
                    },
                },
            ]
        )

        self.assertEqual(summary["n_rows"], 2)
        self.assertEqual(summary["rows_with_fragment_atoms"], 1)
        self.assertEqual(summary["fragment_atom_count"], 2)
        self.assertEqual(summary["examples"][0]["event_id"], "10004.json")

    def test_evidence_id_mapping_round_trips_to_candidate_rows(self) -> None:
        row = prepare_evidence_map_candidate_rows([_event()], candidate_top_n=2)[0]

        mapping = {item["evidence_id"]: item["candidate_uid"] for item in row["evidence_items"]}

        self.assertEqual(mapping, {"E01": "uid-1", "E02": "uid-2"})
        self.assertEqual(row["candidates"][0]["evidence_id"], "E01")

    def test_qd_union_candidate_pool_uses_union_rank_prior_without_fusion_features(self) -> None:
        row = prepare_evidence_map_candidate_rows(
            [
                {
                    "event_id": "event-q",
                    "claim": "The budget increased.",
                    "candidates": [
                        {"text": "Rank two text.", "canonical_text": "rank two", "union_pool_rank": 2, "report_id": 2},
                        {"text": "Rank one text.", "canonical_text": "rank one", "union_pool_rank": 1, "report_id": 1},
                    ],
                }
            ],
            candidate_top_n=2,
            candidate_source="qd_union",
        )[0]
        attach_event_base_scores([row])

        self.assertEqual(row["evidence_map_candidate_source"], "qd_union")
        self.assertEqual([candidate["candidate_key"] for candidate in row["candidates"]], ["rank one", "rank two"])
        self.assertGreater(row["candidates"][0]["evidence_map_base_score"], row["candidates"][1]["evidence_map_base_score"])
        self.assertNotIn("fusion_refit_score", row["candidates"][0])
        self.assertNotIn("direct_ce_score", row["candidates"][0])
        self.assertNotIn("oracle_likelihood_score", row["candidates"][0])

    def test_greedy_rewards_new_atoms_and_penalizes_background_duplicates(self) -> None:
        candidates = [
            _candidate("a", uid="a", fusion=0.4, atoms=["A1"], directness="direct", relation="support", duplicate="G1"),
            _candidate("b", uid="b", fusion=0.9, atoms=[], directness="none", relation="background", duplicate="G2"),
            _candidate("c", uid="c", fusion=0.3, atoms=["A2"], directness="partial", relation="support", duplicate="G1"),
        ]
        attach_event_base_scores([{"candidates": candidates}])

        selected, _ = select_evidence_map_topk(candidates, params=EvidenceMapParams(top_k=2, base_weight=0.2))

        self.assertEqual([row["candidate_key"] for row in selected], ["a", "c"])

    def test_build_all_traces_accepts_custom_primary_selector_params(self) -> None:
        row = {
            **_event(),
            "candidates": [
                _candidate("background", uid="background", fusion=0.95, atoms=["A1"], directness="none", relation="background", union_rank=1),
                _candidate("direct", uid="direct", fusion=0.10, atoms=["A1"], directness="direct", relation="support", union_rank=2),
            ],
        }
        attach_event_base_scores([row])

        traces = build_all_evidence_map_traces(
            [row],
            top_k=1,
            params=EvidenceMapParams(
                top_k=1,
                base_weight=0.0,
                atom_coverage_weight=0.0,
                directness_weight=0.30,
                polar_relation_weight=0.0,
                duplicate_penalty=0.0,
                source_penalty=0.0,
                background_penalty=0.30,
            ),
        )
        primary = [trace for trace in traces if trace["selector_name"] == EVIDENCE_MAP_SELECTOR][0]

        self.assertEqual(primary["selected_keys"], ["direct"])

    def test_eval_evidence_map_cli_accepts_tight_selector_weight_overrides(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "eval_evidence_map_selector_v0_5a.py",
                "--candidate-features",
                "features.jsonl",
                "--selector-directness-weight",
                "0.30",
                "--selector-background-penalty",
                "0.30",
            ],
        ):
            args = eval_map_cli.parse_args()

        params = eval_map_cli._evidence_map_params_from_args(args)

        self.assertEqual(params.directness_weight, 0.30)
        self.assertEqual(params.background_penalty, 0.30)

    def test_base_only_order_is_deterministic(self) -> None:
        row = {
            **_event(),
            "candidates": [
                _candidate("low", uid="low", fusion=0.1, union_rank=1),
                _candidate("high", uid="high", fusion=0.9, union_rank=2),
            ],
        }
        attach_event_base_scores([row])

        traces = build_all_evidence_map_traces([row], top_k=2)
        base = [trace for trace in traces if trace["selector_name"] == EVIDENCE_MAP_BASE_ONLY_SELECTOR][0]

        self.assertEqual(base["selected_keys"], ["high", "low"])

    def test_map_verifier_prompt_fits_fake_budget(self) -> None:
        tokenizer = FakeTokenizer()
        prompt, token_count, evidence_count = _render_until_fit(
            claim="The budget increased.",
            atoms=[{"atom_id": "A1", "text": "Budget increased"}],
            selected=[
                _candidate("a", uid="a", text=" ".join(["long"] * 100), fusion=0.9),
                _candidate("b", uid="b", text=" ".join(["long"] * 100), fusion=0.8),
            ],
            tokenizer=tokenizer,
            system_msg="system",
            output_mode="label_only",
            label_format="name",
            budget=220,
            max_evidence_chars=500,
            max_span_chars=100,
        )

        self.assertLessEqual(token_count, 220)
        self.assertGreaterEqual(evidence_count, 0)
        self.assertIn("Claim Atoms", prompt)

    def test_trace_uses_text_key_ordered_metrics(self) -> None:
        row = _event()
        row["oracle_ordered_keys"] = ["Evidence 1 states"]
        for candidate in row["candidates"]:
            candidate["covered_atom_ids"] = ["A1"]
            candidate["covered_atom_weight"] = 1.0
            candidate["atom_coverage_score"] = 1.0
            candidate["map_relation"] = "support"
            candidate["map_directness"] = "direct"
            candidate["map_evidence_role"] = "primary_support"
            candidate["key_spans"] = ["span"]
            candidate["duplicate_group"] = candidate["candidate_uid"]
            candidate["directness_score"] = 1.0
            candidate["polar_relation_score"] = 1.0
            candidate["background_penalty_score"] = 0.0
            candidate["evidence_map_quality_score"] = 1.0
        attach_event_base_scores([row])

        traces = build_all_evidence_map_traces([row], top_k=1)
        primary = [trace for trace in traces if trace["selector_name"] == EVIDENCE_MAP_SELECTOR][0]

        self.assertEqual(primary["recall@5"], 1.0)


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0

    def __call__(self, text: str, truncation: bool = False, add_special_tokens: bool = False, **_: object) -> dict:
        return {"input_ids": list(range(len(str(text).split())))}

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "\n".join(str(message["content"]) for message in messages) + ("\nAssistant:" if add_generation_prompt else "")


def _event() -> dict:
    return {
        "event_id": "event-1",
        "claim": "The budget increased by 10 percent.",
        "gold_label": "true",
        "oracle_ordered_keys": ["Evidence 1 states"],
        "candidates": [
            _candidate("Evidence 1 states", uid="uid-1", fusion=0.9, oracle=True, union_rank=1),
            _candidate("Evidence 2 states", uid="uid-2", fusion=0.6, union_rank=2),
        ],
    }


def _candidate(
    key: str,
    *,
    uid: str,
    fusion: float,
    atoms: list[str] | None = None,
    directness: str = "direct",
    relation: str = "support",
    duplicate: str = "G1",
    text: str | None = None,
    oracle: bool = False,
    union_rank: int = 1,
) -> dict:
    return {
        "candidate_uid": uid,
        "candidate_key": key,
        "text": text or f"{key} text.",
        "fusion_refit_score": fusion,
        "oracle_likelihood_score": fusion,
        "direct_ce_score": fusion,
        "union_pool_rank": union_rank,
        "source_group": f"report:{union_rank}",
        "from_baseline": True,
        "from_qd": True,
        "baseline_rank": union_rank,
        "qd_pool_rank": union_rank,
        "retrieval_score": fusion,
        "semantic_completeness_score": 0.8,
        "direct_evidence_score": 0.8,
        "claim_specificity_score": 0.8,
        "background_only_score": 0.0 if relation != "background" else 1.0,
        "evidence_role": "direct_support_claim",
        "teacher_stance_probs": {"strong_support_claim_bucket": 1.0},
        "stance_bucket_derived": "strong_support_claim_bucket",
        "oracle_selected": oracle,
        "oracle_step": 0 if oracle else -1,
        "covered_atom_ids": atoms or ["A1"],
        "covered_atom_weight": float(len(atoms or ["A1"])),
        "atom_coverage_score": 1.0,
        "map_relation": relation,
        "map_directness": directness,
        "map_evidence_role": "primary_support",
        "key_spans": ["span"],
        "duplicate_group": duplicate,
        "directness_score": 1.0 if directness == "direct" else 0.0,
        "polar_relation_score": 1.0 if relation == "support" else 0.0,
        "background_penalty_score": 1.0 if relation == "background" else 0.0,
        "evidence_map_quality_score": 1.0 if relation != "background" else 0.0,
    }


if __name__ == "__main__":
    unittest.main()
