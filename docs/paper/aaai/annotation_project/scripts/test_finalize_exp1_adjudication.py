from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_exp1_adjudication import (  # noqa: E402
    DEFAULT_DB,
    finalize,
    parse_adjudication_result,
)


class ParseAdjudicationResultTest(unittest.TestCase):
    def test_dynamic_decisions_follow_question_order_not_field_array_order(self) -> None:
        data = {
            "unit": "atom",
            "fields_to_adjudicate": ["atomicity", "faithfulness"],
            "questions": [
                {"field": "faithfulness"},
                {"field": "atomicity"},
            ],
        }
        result = [
            {"from_name": "decision_0", "value": {"choices": ["no"]}},
            {"from_name": "decision_1", "value": {"choices": ["yes"]}},
            {"from_name": "review_complete", "value": {"choices": ["confirmed"]}},
        ]
        decisions, notes = parse_adjudication_result(data, result)
        self.assertEqual(decisions, {"faithfulness": "no", "atomicity": "yes"})
        self.assertEqual(notes, [])

    def test_review_confirmation_is_required(self) -> None:
        data = {
            "unit": "claim",
            "fields_to_adjudicate": ["completeness_missed"],
        }
        result = [
            {"from_name": "completeness_missed", "value": {"choices": ["0"]}},
        ]
        with self.assertRaisesRegex(ValueError, "review_complete"):
            parse_adjudication_result(data, result)


class FinalizeAdjudicationIntegrationTest(unittest.TestCase):
    def test_live_snapshot_produces_expected_final_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "final"
            metrics = finalize(DEFAULT_DB, output, bootstrap_reps=5000, seed=42)

            self.assertEqual(metrics["analysis_status"], "final_adjudicated_gold")
            self.assertTrue(metrics["snapshot"]["current_formal_source_matches_frozen"])
            self.assertEqual(metrics["snapshot"]["sqlite_quick_check"], "ok")
            self.assertEqual(metrics["adjudication_audit"]["tasks_completed"], 47)
            self.assertEqual(metrics["adjudication_audit"]["field_decisions_completed"], 50)
            self.assertEqual(metrics["adjudication_audit"]["anomalies"], [])
            self.assertEqual(metrics["adjudication_audit"]["prior_pilot_exposure"]["count"], 2)

            final = metrics["final_gold"]
            self.assertEqual((final["faithfulness"]["pass_count"], final["faithfulness"]["n"]), (255, 257))
            self.assertEqual((final["atomicity"]["pass_count"], final["atomicity"]["n"]), (246, 257))
            self.assertEqual((final["complete_coverage"]["pass_count"], final["complete_coverage"]["n"]), (198, 200))
            self.assertEqual(
                (
                    final["strict_all_criteria_pass"]["pass_count"],
                    final["strict_all_criteria_pass"]["n"],
                ),
                (187, 200),
            )
            self.assertEqual(
                final["completeness_missed_distribution"],
                {"0": 198, "1": 2, "2": 0, "3+": 0},
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(set(manifest["expected_artifacts"]), set(manifest["artifacts"]))
            self.assertEqual(len((output / "adjudication_annotations.jsonl").read_text().splitlines()), 47)
            self.assertEqual(len((output / "gold_atom_annotations.jsonl").read_text().splitlines()), 257)
            self.assertEqual(len((output / "gold_claim_annotations.jsonl").read_text().splitlines()), 200)
            insert = (output / "paper_insert_v0.4.2.md").read_text(encoding="utf-8")
            self.assertIn("Claim Atomization Reliability Study (Exp1)", insert)
            self.assertIn("Evidence Map Annotation Reliability Study (Exp2; Placeholder)", insert)
            self.assertIn("高度符合独立人工质量判断", insert)
            self.assertIn("final gold 支持“模型生成的 atoms 基本符合人工质量判断”", insert)
            self.assertIn("不进一步证明 Evidence Map 标注可靠", insert)


if __name__ == "__main__":
    unittest.main()
