from __future__ import annotations

import unittest

from fact_checking.selectors.aspects import (
    LLM_DECOMP_PLUS_VERSION,
    build_claim_aspect_bundle_from_texts,
    extract_claim_aspects,
)


class ClaimAspectExtractionTest(unittest.TestCase):
    def test_parallel_policy_actions_are_atomic_and_retrievable(self) -> None:
        bundle = extract_claim_aspects(
            "Rick Perry has advocated abandoning Social Security, "
            "scuttling Medicaid and ending the federal income tax.",
            event_id="policy.json",
        )
        texts = [aspect.text for aspect in bundle.aspects]
        self.assertIn("Rick Perry advocated abandoning Social Security", texts)
        self.assertIn("Rick Perry advocated scuttling Medicaid", texts)
        self.assertIn("Rick Perry advocated ending the federal income tax", texts)
        for aspect in bundle.aspects:
            self.assertTrue(aspect.is_atomic)
            self.assertTrue(aspect.is_decontextualized)
            self.assertGreaterEqual(aspect.retrievability_score, 2)

    def test_otherwise_clause_gets_previous_context(self) -> None:
        bundle = extract_claim_aspects(
            "Says when armed civilians stop mass shootings with guns, "
            "an average of 2.5 people die; otherwise, an average of 18 people die.",
            event_id="shooting.json",
        )
        texts = [aspect.text for aspect in bundle.aspects]
        self.assertTrue(any("2.5 people die" in text for text in texts))
        self.assertTrue(any("otherwise, an average of 18 people die" in text for text in texts))
        self.assertTrue(any("armed civilians stop mass shootings" in text for text in texts))

    def test_instead_clause_is_decontextualized_when_safe(self) -> None:
        bundle = extract_claim_aspects(
            "When Obama was sworn into office, he DID NOT use the Holy Bible, "
            "but instead the Kuran.",
            event_id="bible.json",
        )
        texts = [aspect.text for aspect in bundle.aspects]
        self.assertTrue(any("Obama DID NOT use the Holy Bible" in text for text in texts))
        self.assertTrue(any("Obama used the Kuran instead of the Holy Bible" in text for text in texts))

    def test_llm_decomp_texts_use_same_bundle_schema(self) -> None:
        bundle = build_claim_aspect_bundle_from_texts(
            "The U.S. sells new jets abroad while using boneyards for fighter parts.",
            [
                "The U.S. sells new jets to other countries.",
                "The U.S. obtains parts for fighter jets from aircraft boneyards.",
                "The U.S. sells new jets to other countries.",
            ],
            event_id="jets.json",
        )
        self.assertEqual(bundle.extraction_version, LLM_DECOMP_PLUS_VERSION)
        self.assertEqual(len(bundle.aspects), 2)
        self.assertEqual(len(bundle.dropped_aspects), 1)
        self.assertEqual(bundle.aspects[0].source, "llm_decomp_plus")
        self.assertEqual(bundle.dropped_aspects[0].drop_reason, "duplicate_local_aspect")


if __name__ == "__main__":
    unittest.main()
