from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fact_checking.selectors.question_decomp import (
    QuestionGenerationSettings,
    QuestionInputExample,
    generate_or_load_questions,
    question_cache_key,
    question_config_fingerprint,
    read_question_cache,
)


def _json_question(text: str = "What evidence verifies the claim?") -> str:
    return json.dumps(
        {
            "complexity": "simple",
            "questions": [
                {
                    "id": "q1",
                    "question": text,
                    "focus": "overall",
                    "priority": 1,
                }
            ],
        }
    )


class _FakeClient:
    def __init__(self, outputs: list[str] | None = None, exc: BaseException | None = None) -> None:
        self.outputs = list(outputs or [])
        self.exc = exc
        self.calls = 0

    def generate(self, **_: object) -> str:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.outputs:
            raise AssertionError("Fake client had no remaining outputs.")
        return self.outputs.pop(0)


class QuestionDecompCacheTest(unittest.TestCase):
    def test_fingerprint_changes_only_with_generation_config(self) -> None:
        base = QuestionGenerationSettings(model="qwen", max_tokens=384, temperature=0.0)
        same = QuestionGenerationSettings(model="qwen", max_tokens=384, temperature=0.0)
        changed_model = QuestionGenerationSettings(model="deepseek", max_tokens=384, temperature=0.0)
        changed_temperature = QuestionGenerationSettings(model="qwen", max_tokens=384, temperature=0.2)
        changed_tokens = QuestionGenerationSettings(model="qwen", max_tokens=512, temperature=0.0)

        self.assertEqual(question_config_fingerprint(base), question_config_fingerprint(same))
        self.assertNotEqual(question_config_fingerprint(base), question_config_fingerprint(changed_model))
        self.assertNotEqual(question_config_fingerprint(base), question_config_fingerprint(changed_temperature))
        self.assertNotEqual(question_config_fingerprint(base), question_config_fingerprint(changed_tokens))

    def test_resume_expansion_generates_only_new_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            examples = self._examples(12)
            settings = QuestionGenerationSettings(model="qwen")
            first_client = _FakeClient([_json_question(f"What verifies claim {idx}?") for idx in range(8)])
            first = generate_or_load_questions(
                examples=examples[:8],
                split="train",
                output_dir=tmp_path / "run8",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: first_client,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(first.manifest["n_loaded_from_cache"], 0)
            self.assertEqual(first.manifest["n_api_generated"], 8)
            self.assertEqual(first_client.calls, 8)

            second_client = _FakeClient([_json_question(f"What verifies claim {idx}?") for idx in range(8, 12)])
            second = generate_or_load_questions(
                examples=examples,
                split="train",
                output_dir=tmp_path / "run12",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: second_client,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(second.manifest["n_loaded_from_cache"], 8)
            self.assertEqual(second.manifest["n_api_generated"], 4)
            self.assertEqual(second_client.calls, 4)
            self.assertEqual(len(second.rows), 12)

    def test_cache_complete_does_not_initialize_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            examples = self._examples(2)
            settings = QuestionGenerationSettings(model="qwen")
            first_client = _FakeClient([_json_question("What verifies A?"), _json_question("What verifies B?")])
            generate_or_load_questions(
                examples=examples,
                split="val",
                output_dir=tmp_path / "first",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: first_client,
                retry_initial_delay=0.0,
                no_progress=True,
            )

            def _unexpected_factory() -> _FakeClient:
                raise AssertionError("API client should not be initialized on full cache hit.")

            second = generate_or_load_questions(
                examples=examples,
                split="val",
                output_dir=tmp_path / "second",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=_unexpected_factory,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(second.manifest["n_loaded_from_cache"], 2)
            self.assertEqual(second.manifest["n_api_generated"], 0)

    def test_parse_failure_fallback_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            examples = self._examples(1)
            settings = QuestionGenerationSettings(model="qwen")
            client = _FakeClient(["not valid json"])
            first = generate_or_load_questions(
                examples=examples,
                split="val",
                output_dir=tmp_path / "first",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: client,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(first.manifest["parse_failures"], 1)
            self.assertEqual(first.rows[0]["parse_status"], "parse_failed")
            self.assertEqual(first.rows[0]["question_source"], "fallback_parse_failed")

            second = generate_or_load_questions(
                examples=examples,
                split="val",
                output_dir=tmp_path / "second",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: _FakeClient(exc=AssertionError("should not call API")),
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(second.manifest["n_loaded_from_cache"], 1)
            self.assertEqual(second.rows[0]["question_source"], "fallback_parse_failed")

    def test_api_exception_does_not_write_completed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            examples = self._examples(1)
            settings = QuestionGenerationSettings(model="qwen")
            with self.assertRaises(RuntimeError):
                generate_or_load_questions(
                    examples=examples,
                    split="val",
                    output_dir=tmp_path / "failed",
                    question_cache_dir=tmp_path / "cache",
                    settings=settings,
                    client_factory=lambda: _FakeClient(exc=RuntimeError("api down")),
                    retry_initial_delay=0.0,
                    no_progress=True,
                )
            cache_files = list((tmp_path / "cache").glob("question_cache_val_*.jsonl"))
            if cache_files:
                self.assertEqual(cache_files[0].read_text(encoding="utf-8"), "")

            client = _FakeClient([_json_question("What verifies retry?")])
            retry = generate_or_load_questions(
                examples=examples,
                split="val",
                output_dir=tmp_path / "retry",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: client,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(retry.manifest["n_loaded_from_cache"], 0)
            self.assertEqual(retry.manifest["n_api_generated"], 1)

    def test_duplicate_cache_rows_use_last_valid_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.jsonl"
            fp = "manual-cache"
            claim_hash = "claimhash"
            first = self._cache_row("event1", claim_hash, fp, "What verifies old?")
            second = self._cache_row("event1", claim_hash, fp, "What verifies new?")
            cache_path.write_text(
                "{bad json\n"
                + json.dumps(first)
                + "\n"
                + json.dumps(second)
                + "\n",
                encoding="utf-8",
            )
            index = read_question_cache(cache_path, cache_fingerprint=fp)
            key = question_cache_key("event1", claim_hash, fp)
            self.assertEqual(index.invalid_lines, 1)
            self.assertEqual(index.duplicate_rows, 1)
            self.assertEqual(index.rows_by_key[key]["questions"][0]["question"], "What verifies new?")

    def test_current_run_output_order_follows_current_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            examples = self._examples(2)
            settings = QuestionGenerationSettings(model="qwen")
            client = _FakeClient([_json_question("What verifies first?"), _json_question("What verifies second?")])
            generate_or_load_questions(
                examples=examples,
                split="train",
                output_dir=tmp_path / "first",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: client,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            reversed_examples = list(reversed(examples))
            second = generate_or_load_questions(
                examples=reversed_examples,
                split="train",
                output_dir=tmp_path / "second",
                question_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: _FakeClient(exc=AssertionError("should not call API")),
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual([row["event_id"] for row in second.rows], ["event1", "event0"])
            output_rows = [
                json.loads(line)
                for line in (tmp_path / "second" / "questions_train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["event_id"] for row in output_rows], ["event1", "event0"])

    @staticmethod
    def _examples(n: int) -> list[QuestionInputExample]:
        return [
            QuestionInputExample(event_id=f"event{idx}", claim=f"Claim {idx}", gold_label="true")
            for idx in range(n)
        ]

    @staticmethod
    def _cache_row(event_id: str, claim_hash: str, fp: str, question: str) -> dict[str, object]:
        return {
            "event_id": event_id,
            "claim": "Claim",
            "claim_sha256": claim_hash,
            "question_config_fingerprint": fp,
            "questions": [{"id": "q1", "question": question, "focus": "overall", "priority": 1}],
        }


if __name__ == "__main__":
    unittest.main()
