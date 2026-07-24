from __future__ import annotations

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_exp1_adjudication import (  # noqa: E402
    DEFAULT_ATOM_CONFIG,
    DEFAULT_BLIND_QUEUE,
    DEFAULT_COMPLETENESS_CONFIG,
    DEFAULT_METRICS,
    DEFAULT_SOURCE_MANIFEST,
    DEFAULT_UNIVERSE,
    FORBIDDEN_KEYS,
    TRANSLATION_FALLBACK,
    load_json,
    load_jsonl,
    load_universe,
    prepare,
    prepare_tasks,
)


class PrepareAdjudicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = load_json(DEFAULT_METRICS)
        cls.atom_tasks, cls.claim_tasks = prepare_tasks(
            load_jsonl(DEFAULT_BLIND_QUEUE),
            load_universe(DEFAULT_UNIVERSE),
            cls.metrics["methodology"]["gold_resolution_protocol_version"],
            cls.metrics["snapshot"]["analysis_input_sha256"],
        )

    def test_expected_counts_and_field_combinations(self) -> None:
        self.assertEqual(len(self.atom_tasks), 37)
        self.assertEqual(len(self.claim_tasks), 10)
        self.assertEqual(
            Counter(tuple(row["fields_to_adjudicate"]) for row in self.atom_tasks),
            Counter({("atomicity",): 26, ("faithfulness",): 8, ("atomicity", "faithfulness"): 3}),
        )
        self.assertEqual(
            {tuple(row["fields_to_adjudicate"]) for row in self.claim_tasks},
            {("completeness_missed",)},
        )

    def test_dynamic_questions_match_only_requested_fields(self) -> None:
        for row in self.atom_tasks:
            question_fields = [question["field"] for question in row["questions"]]
            self.assertEqual(
                question_fields,
                [field for field in ("faithfulness", "atomicity") if field in row["fields_to_adjudicate"]],
            )
            self.assertEqual(len(question_fields), len(row["fields_to_adjudicate"]))

    def test_no_rater_labels_leak_into_prepared_tasks(self) -> None:
        serialized = json.dumps(self.atom_tasks + self.claim_tasks, ensure_ascii=False)
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(f'"{key}"', serialized)

    def test_missing_translation_uses_explicit_fallback(self) -> None:
        row = next(
            task
            for task in self.atom_tasks
            if task["dataset"] == "liar_raw"
            and task["event_id"] == "8386.json"
            and task["atom_id"] == "A1"
        )
        self.assertEqual(row["claim_zh"], TRANSLATION_FALLBACK)

    def test_configs_are_well_formed_xml(self) -> None:
        ET.fromstring(DEFAULT_ATOM_CONFIG.read_text(encoding="utf-8"))
        ET.fromstring(DEFAULT_COMPLETENESS_CONFIG.read_text(encoding="utf-8"))

    def test_prepare_publishes_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "prepared"
            manifest = prepare(
                DEFAULT_BLIND_QUEUE,
                DEFAULT_SOURCE_MANIFEST,
                DEFAULT_METRICS,
                DEFAULT_UNIVERSE,
                DEFAULT_ATOM_CONFIG,
                DEFAULT_COMPLETENESS_CONFIG,
                output,
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["counts"], {"atom": 37, "completeness": 10, "total": 47})
            self.assertEqual(set(manifest["artifacts"]), {"atom_tasks.jsonl", "completeness_tasks.jsonl"})


if __name__ == "__main__":
    unittest.main()
