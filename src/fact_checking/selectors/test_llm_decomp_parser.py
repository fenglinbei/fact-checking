from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "selectors"
    / "generate_llm_claim_decomp_aspects.py"
)
_SPEC = importlib.util.spec_from_file_location("generate_llm_claim_decomp_aspects", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class LLMDecompParserTest(unittest.TestCase):
    def test_parse_json_subclaims(self) -> None:
        subclaims, status, error = _MODULE.parse_subclaims_from_generation(
            '{"sub_claims":["The state passed a tax cut.","The tax cut reduced school funding."]}'
        )
        self.assertEqual(status, "ok")
        self.assertIsNone(error)
        self.assertEqual(len(subclaims), 2)
        self.assertEqual(subclaims[0], "The state passed a tax cut.")

    def test_parse_numbered_lines_fallback(self) -> None:
        subclaims, status, error = _MODULE.parse_subclaims_from_generation(
            "1. Officials claimed that unemployment fell in 2012.\n"
            "2. The unemployment decline happened after the policy change."
        )
        self.assertEqual(status, "fallback_numbered_lines")
        self.assertIsNone(error)
        self.assertEqual(len(subclaims), 2)
        self.assertTrue(subclaims[1].startswith("The unemployment decline"))


if __name__ == "__main__":
    unittest.main()
