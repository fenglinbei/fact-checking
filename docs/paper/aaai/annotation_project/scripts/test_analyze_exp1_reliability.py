from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp1_reliability import (
    DEFAULT_DB,
    DEFAULT_TASK_UNIVERSE,
    ProjectSpec,
    agreement_metrics,
    analyze,
    build_blind_adjudication_tasks,
    collapse_claim_annotations,
    completeness_metrics,
    load_authoritative_universe,
    pair_atom_records,
    parse_annotation_result,
    parse_draft_result,
    semantic_atom_key,
    strict_claim_pass_metrics,
    validate_distinct_project_pair,
)


def _result(**overrides: str) -> str:
    values = {
        "claim_complexity": "simple",
        "completeness_missed": "0",
        "faithfulness": "yes",
        "atomicity": "yes",
    }
    values.update(overrides)
    entries = [
        {
            "from_name": field,
            "type": "choices",
            "value": {"choices": [choice]},
        }
        for field, choice in reversed(list(values.items()))
    ]
    entries.append({"from_name": "notes", "type": "textarea", "value": {"text": ["note"]}})
    return json.dumps(entries)


def _record(
    dataset: str,
    event_id: str,
    atom_id: str,
    *,
    task_id: int,
    completeness: str = "0",
    faithfulness: str = "yes",
    atomicity: str = "yes",
) -> dict:
    return {
        "dataset": dataset,
        "event_id": event_id,
        "atom_id": atom_id,
        "claim": f"claim-{dataset}-{event_id}",
        "proposition": f"prop-{atom_id}",
        "atom_type": "other",
        "all_atoms_text": "panorama",
        "task_id": task_id,
        "inner_id": task_id,
        "annotation_id": task_id + 100,
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
        "labels": {
            "claim_complexity": "simple",
            "completeness_missed": completeness,
            "faithfulness": faithfulness,
            "atomicity": atomicity,
            "notes": [],
        },
    }


class ParseResultTest(unittest.TestCase):
    def test_parses_controls_by_name_not_position(self) -> None:
        parsed = parse_annotation_result(_result(atomicity="no"))
        self.assertEqual(parsed["atomicity"], "no")
        self.assertEqual(parsed["completeness_missed"], "0")
        self.assertEqual(parsed["notes"], ["note"])

    def test_rejects_missing_required_control(self) -> None:
        payload = json.loads(_result())
        payload = [entry for entry in payload if entry.get("from_name") != "faithfulness"]
        with self.assertRaisesRegex(ValueError, "Missing required controls"):
            parse_annotation_result(payload)

    def test_preserves_three_plus_as_a_category(self) -> None:
        parsed = parse_annotation_result(_result(completeness_missed="3+"))
        self.assertEqual(parsed["completeness_missed"], "3+")

    def test_partial_draft_is_parsed_without_inventing_missing_controls(self) -> None:
        payload = json.loads(_result(atomicity="no"))
        payload = [
            entry
            for entry in payload
            if entry.get("from_name") in {"atomicity", "notes"}
        ]
        parsed, missing = parse_draft_result(payload)
        self.assertEqual(parsed["atomicity"], "no")
        self.assertEqual(
            missing,
            ["claim_complexity", "completeness_missed", "faithfulness"],
        )


class SemanticKeyTest(unittest.TestCase):
    def test_dataset_is_part_of_atom_key(self) -> None:
        left = semantic_atom_key({"dataset": "liar_raw", "event_id": "1", "atom_id": "A1"})
        right = semantic_atom_key({"dataset": "rawfc", "event_id": "1", "atom_id": "A1"})
        self.assertNotEqual(left, right)

    def test_rejects_flatten_fallback_atom(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid semantic atom key"):
            semantic_atom_key({"dataset": "rawfc", "event_id": "1", "atom_id": "-"})

    def test_rejects_null_and_whitespace_semantic_parts(self) -> None:
        for bad in (None, "", " A1", "A1 "):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Invalid semantic atom key"):
                semantic_atom_key({"dataset": "rawfc", "event_id": "1", "atom_id": bad})


class PairAndCollapseTest(unittest.TestCase):
    def test_pairs_cloned_tasks_by_semantic_key_not_database_id(self) -> None:
        key = ("liar_raw", "1.json", "A1")
        a_record = _record(*key, task_id=10)
        b_record = _record(*key, task_id=999)
        pairs = pair_atom_records({"records": {key: a_record}}, {"records": {key: b_record}})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["a"]["task_id"], 10)
        self.assertEqual(pairs[0]["b"]["task_id"], 999)

    def test_collapses_repeated_claim_fields_once(self) -> None:
        pairs = []
        for atom_id in ("A1", "A2"):
            key = ("rawfc", "7", atom_id)
            a = _record(*key, task_id=len(pairs) + 1)
            b = _record(*key, task_id=len(pairs) + 101)
            pairs.append(
                {
                    "dataset": key[0],
                    "event_id": key[1],
                    "atom_id": key[2],
                    "claim": a["claim"],
                    "proposition": a["proposition"],
                    "atom_type": "other",
                    "a": a,
                    "b": b,
                }
            )
        claims, issues = collapse_claim_annotations(pairs, "A", "B")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["atom_count"], 2)
        self.assertEqual(claims[0]["a"]["completeness_missed"], "0")
        self.assertEqual(issues, [])

    def test_claim_conflict_is_explicit_not_majority_voted(self) -> None:
        pairs = []
        for index, completeness in enumerate(("0", "1"), start=1):
            key = ("rawfc", "8", f"A{index}")
            a = _record(*key, task_id=index, completeness=completeness)
            b = _record(*key, task_id=index + 100, completeness="0")
            pairs.append(
                {
                    "dataset": key[0],
                    "event_id": key[1],
                    "atom_id": key[2],
                    "claim": a["claim"],
                    "proposition": a["proposition"],
                    "atom_type": "other",
                    "a": a,
                    "b": b,
                }
            )
        claims, issues = collapse_claim_annotations(pairs, "A", "B")
        self.assertIsNone(claims[0]["a"]["completeness_missed"])
        self.assertEqual(claims[0]["b"]["completeness_missed"], "0")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["values_by_atom"], {"A1": "0", "A2": "1"})


class AgreementMetricTest(unittest.TestCase):
    def test_reports_prevalence_robust_ac1_and_bounds(self) -> None:
        result = agreement_metrics(
            ["yes", "yes", "yes", "no"],
            ["yes", "yes", "no", "no"],
            ("yes", "no"),
            positive_label="yes",
        )
        self.assertEqual(result["exact_agreement_count"], 3)
        self.assertAlmostEqual(result["exact_agreement"], 0.75)
        self.assertAlmostEqual(result["pre_adjudication_positive_lower_bound"], 0.5)
        self.assertAlmostEqual(result["pre_adjudication_positive_upper_bound"], 0.75)
        self.assertAlmostEqual(result["gwet_ac1"], 0.5294117647058824)
        self.assertEqual(result["minority_label"], "no")
        self.assertAlmostEqual(result["minority_agreement"], 2 / 3)

    def test_completeness_keeps_binary_coverage_separate_from_ordinal(self) -> None:
        rows = []
        for index, (a_value, b_value) in enumerate(
            (("0", "0"), ("0", "1"), ("2", "3+"), ("3+", "3+")),
            start=1,
        ):
            rows.append(
                {
                    "dataset": "liar_raw",
                    "event_id": str(index),
                    "a": {"completeness_missed": a_value},
                    "b": {"completeness_missed": b_value},
                }
            )
        result, clean = completeness_metrics(rows, 0, 42)
        self.assertEqual(len(clean), 4)
        self.assertEqual(result["coverage_binary"]["positive_label"], "complete")
        self.assertEqual(result["coverage_binary"]["minority_label"], "complete")
        self.assertIn("linear_weighted_kappa", result["ordinal"])
        self.assertNotEqual(
            result["coverage_binary"]["gwet_ac1"],
            result["ordinal"]["gwet_ac1"],
        )


class StrictClaimMetricTest(unittest.TestCase):
    def test_componentwise_upper_bound_recovers_complementary_rater_failures(self) -> None:
        key = ("liar_raw", "1065.json", "A1")
        a = _record(*key, task_id=1, completeness="1", atomicity="yes")
        b = _record(*key, task_id=2, completeness="0", atomicity="no")
        pair = {
            "dataset": key[0],
            "event_id": key[1],
            "atom_id": key[2],
            "claim": a["claim"],
            "proposition": a["proposition"],
            "atom_type": "other",
            "a": a,
            "b": b,
        }
        claims, issues = collapse_claim_annotations([pair], "A", "B")
        self.assertEqual(issues, [])
        result, rows = strict_claim_pass_metrics([pair], claims, 0, 42)
        self.assertEqual(rows[0]["a_value"], "fail")
        self.assertEqual(rows[0]["b_value"], "fail")
        self.assertFalse(rows[0]["componentwise_confirmed_pass"])
        self.assertTrue(rows[0]["componentwise_possible_pass"])
        self.assertEqual(result["pre_adjudication_positive_lower_bound"], 0.0)
        self.assertEqual(result["pre_adjudication_positive_upper_bound"], 1.0)


class BlindAdjudicationTest(unittest.TestCase):
    def test_blind_task_does_not_expose_rater_labels(self) -> None:
        key = ("liar_raw", "1.json", "A1")
        a = _record(*key, task_id=1)
        b = _record(*key, task_id=2, atomicity="no")
        pair = {
            "dataset": key[0],
            "event_id": key[1],
            "atom_id": key[2],
            "claim": a["claim"],
            "proposition": a["proposition"],
            "atom_type": "other",
            "a": a,
            "b": b,
        }
        queue = [
            {
                "unit": "atom",
                "dataset": key[0],
                "event_id": key[1],
                "atom_id": key[2],
                "disagreements": {
                    "atomicity": {"annotator_a": "yes", "annotator_b": "no"}
                },
            }
        ]
        blind = build_blind_adjudication_tasks(queue, [pair], [])
        serialized = json.dumps(blind)
        self.assertNotIn("annotator_a", serialized)
        self.assertNotIn("annotator_b", serialized)
        self.assertNotIn("disagreements", serialized)
        self.assertEqual(blind[0]["fields_to_adjudicate"], ["atomicity"])

    def test_blind_id_is_stable_when_queue_order_changes(self) -> None:
        pairs = []
        queue = []
        for index, atom_id in enumerate(("A1", "A2"), start=1):
            key = ("liar_raw", "1.json", atom_id)
            a = _record(*key, task_id=index)
            b = _record(*key, task_id=index + 10, atomicity="no")
            pairs.append(
                {
                    "dataset": key[0],
                    "event_id": key[1],
                    "atom_id": key[2],
                    "claim": a["claim"],
                    "proposition": a["proposition"],
                    "atom_type": "other",
                    "a": a,
                    "b": b,
                }
            )
            queue.append(
                {
                    "unit": "atom",
                    "dataset": key[0],
                    "event_id": key[1],
                    "atom_id": key[2],
                    "disagreements": {"atomicity": {"annotator_a": "yes", "annotator_b": "no"}},
                }
            )
        forward = build_blind_adjudication_tasks(queue, pairs, [])
        reverse = build_blind_adjudication_tasks(list(reversed(queue)), pairs, [])
        forward_ids = {task["atom_id"]: task["adjudication_id"] for task in forward}
        reverse_ids = {task["atom_id"]: task["adjudication_id"] for task in reverse}
        self.assertEqual(forward_ids, reverse_ids)

    def test_internal_claim_conflict_uses_its_actual_field(self) -> None:
        key = ("rawfc", "7", "A1")
        a = _record(*key, task_id=1)
        b = _record(*key, task_id=2)
        pair = {
            "dataset": key[0],
            "event_id": key[1],
            "atom_id": key[2],
            "claim": a["claim"],
            "proposition": a["proposition"],
            "atom_type": "other",
            "a": a,
            "b": b,
        }
        claims, _ = collapse_claim_annotations([pair], "A", "B")
        queue = [
            {
                "issue_type": "within_annotator_claim_conflict",
                "dataset": key[0],
                "event_id": key[1],
                "field": "claim_complexity",
            }
        ]
        blind = build_blind_adjudication_tasks(queue, [pair], claims)
        self.assertEqual(blind[0]["fields_to_adjudicate"], ["claim_complexity"])

    def test_cross_rater_claim_task_excludes_auxiliary_complexity(self) -> None:
        key = ("rawfc", "9", "A1")
        a = _record(*key, task_id=1)
        b = _record(*key, task_id=2)
        pair = {
            "dataset": key[0],
            "event_id": key[1],
            "atom_id": key[2],
            "claim": a["claim"],
            "proposition": a["proposition"],
            "atom_type": "other",
            "a": a,
            "b": b,
        }
        claims, _ = collapse_claim_annotations([pair], "A", "B")
        queue = [
            {
                "unit": "claim",
                "dataset": key[0],
                "event_id": key[1],
                "disagreements": {
                    "claim_complexity": {"annotator_a": "simple", "annotator_b": "compound"},
                    "completeness_missed": {"annotator_a": "0", "annotator_b": "1"},
                },
            }
        ]
        blind = build_blind_adjudication_tasks(queue, [pair], claims)
        self.assertEqual(blind[0]["fields_to_adjudicate"], ["completeness_missed"])


class InputContractTest(unittest.TestCase):
    def test_authoritative_universe_has_expected_shape(self) -> None:
        universe = load_authoritative_universe(DEFAULT_TASK_UNIVERSE)
        self.assertEqual(universe["atom_count"], 257)
        self.assertEqual(universe["claim_count"], 200)
        self.assertEqual(universe["claims_by_dataset"], {"liar_raw": 100, "rawfc": 100})

    def test_rejects_same_project_or_same_annotator(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct projects"):
            validate_distinct_project_pair(
                {"project_id": 14, "annotator_email": "a@example.com"},
                {"project_id": 14, "annotator_email": "b@example.com"},
            )
        with self.assertRaisesRegex(ValueError, "distinct annotators"):
            validate_distinct_project_pair(
                {"project_id": 14, "annotator_email": "a@example.com"},
                {"project_id": 15, "annotator_email": "a@example.com"},
            )

    def test_end_to_end_publishes_complete_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "snapshot"
            metrics = analyze(
                DEFAULT_DB,
                DEFAULT_TASK_UNIVERSE,
                output,
                ProjectSpec(14, "Yulin", "1849812973@qq.com"),
                ProjectSpec(15, "Zhiqiang", "3180643570@qq.com"),
                0,
                42,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["analysis_input_sha256"], metrics["snapshot"]["analysis_input_sha256"])
            self.assertEqual(set(manifest["expected_artifacts"]), set(manifest["artifacts"]))
            for name, metadata in manifest["artifacts"].items():
                self.assertTrue((output / name).is_file())
                self.assertEqual(len(metadata["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
