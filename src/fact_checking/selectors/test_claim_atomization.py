from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fact_checking.selectors.claim_atomization import (
    AtomGenerationSettings,
    AtomInputExample,
    fallback_atoms_for_claim,
    generate_or_load_claim_atoms,
    parse_claim_atoms_from_generation,
)


def _json_atoms() -> str:
    return json.dumps(
        {
            "complexity": "simple",
            "claim_atoms": [
                {
                    "atom_id": "custom",
                    "proposition": "Adam Schiff voted to allocate $100 billion in foreign aid.",
                    "importance": 2,
                    "type": "quantity",
                    "keywords": ["Adam Schiff", "$100 billion", "foreign aid"],
                    "query_rendering": "What amount of foreign aid did Adam Schiff vote to allocate",
                    "ignored_extra": "drop me",
                }
            ],
        }
    )


class _FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.last_response_metadata = {"finish_reason": "stop", "mock": True}

    def generate(self, **_: object) -> str:
        self.calls += 1
        if not self.outputs:
            raise AssertionError("Fake client had no remaining outputs.")
        return self.outputs.pop(0)


class ClaimAtomizationTest(unittest.TestCase):
    def test_parse_simplified_atoms_normalizes_ids_text_and_query(self) -> None:
        atoms, status, error, complexity = parse_claim_atoms_from_generation(_json_atoms(), claim="fallback")

        self.assertEqual(status, "ok")
        self.assertIsNone(error)
        self.assertEqual(complexity, "simple")
        self.assertEqual(
            atoms,
            [
                {
                    "atom_id": "A1",
                    "proposition": "Adam Schiff voted to allocate $100 billion in foreign aid.",
                    "text": "Adam Schiff voted to allocate $100 billion in foreign aid.",
                    "importance": 1.0,
                    "type": "quantity",
                    "keywords": ["Adam Schiff", "$100 billion", "foreign aid"],
                    "query_rendering": "What amount of foreign aid did Adam Schiff vote to allocate?",
                }
            ],
        )

    def test_parse_failure_falls_back_to_single_full_claim_atom(self) -> None:
        atoms, status, error, _ = parse_claim_atoms_from_generation("not json", claim="The claim text.")

        self.assertEqual(status, "parse_failed")
        self.assertIn("JSON", str(error))
        self.assertEqual(atoms, fallback_atoms_for_claim("The claim text."))
        self.assertEqual(atoms[0]["text"], "The claim text.")

    def test_cache_resume_uses_existing_atom_rows_without_calling_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = AtomGenerationSettings(model="deepseek-v4-flash")
            examples = [AtomInputExample(event_id="evt-1", claim="The claim text.", gold_label="true")]
            first_client = _FakeClient([_json_atoms()])
            first = generate_or_load_claim_atoms(
                examples=examples,
                split="val",
                output_dir=tmp_path / "first",
                atom_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=lambda: first_client,
                retry_initial_delay=0.0,
                no_progress=True,
            )
            self.assertEqual(first.manifest["n_api_generated"], 1)
            self.assertEqual(first_client.calls, 1)

            def _unexpected_factory() -> _FakeClient:
                raise AssertionError("API client should not be initialized on full cache hit.")

            second = generate_or_load_claim_atoms(
                examples=examples,
                split="val",
                output_dir=tmp_path / "second",
                atom_cache_dir=tmp_path / "cache",
                settings=settings,
                client_factory=_unexpected_factory,
                retry_initial_delay=0.0,
                no_progress=True,
            )

            self.assertEqual(second.manifest["n_loaded_from_cache"], 1)
            self.assertEqual(second.manifest["n_api_generated"], 0)
            self.assertEqual(second.rows[0]["claim_atoms"][0]["atom_id"], "A1")


if __name__ == "__main__":
    unittest.main()
