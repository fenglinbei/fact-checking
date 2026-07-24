from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from translate_exp3_trace_alignment import main  # noqa: E402


class TranslationProvenanceTest(unittest.TestCase):
    def test_complete_cache_rewrite_preserves_row_provenance(self) -> None:
        text_en = "A source sentence."
        key = hashlib.sha256(text_en.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "translation_inventory.jsonl"
            cache = root / "translation_cache.jsonl"
            requests.write_text(
                json.dumps(
                    {
                        "translation_key": f"sha256:{key}",
                        "text_en": text_en,
                        "field_types": ["evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cache.write_text(
                json.dumps(
                    {
                        "translation_key": f"sha256:{key}",
                        "text_en": text_en,
                        "text_zh": "一条来源句子。",
                        "field_types": ["evidence"],
                        "translation_model": "manual-qc-after-local-hf",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--backend",
                        "local-hf",
                        "--requests",
                        str(requests),
                        "--cache",
                        str(cache),
                    ]
                )

            self.assertEqual(result, 0)
            row = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(
                row["translation_model"], "manual-qc-after-local-hf"
            )
            self.assertEqual(row["text_zh"], "一条来源句子。")


if __name__ == "__main__":
    unittest.main()
