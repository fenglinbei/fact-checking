from __future__ import annotations

import unittest

from fact_checking.selectors.metrics import ordered_selection_metrics
from fact_checking.selectors.stage2_oracle import audit_stage2_oracle_record


class SelectorMetricsTest(unittest.TestCase):
    def test_ordered_selection_metrics_distinguish_set_and_order(self) -> None:
        metrics = ordered_selection_metrics([4, 2, 7, 1, 11], [2, 4, 7, 1, 11], top_k=5)

        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["jaccard@5"], 1.0)
        self.assertEqual(metrics["top1_match"], 0.0)
        self.assertEqual(metrics["prefix_match@5"], 0.0)
        self.assertGreater(metrics["pairwise_order_acc@5"], 0.0)
        self.assertLess(metrics["pairwise_order_acc@5"], 1.0)
        self.assertEqual(metrics["overlap_pair_count"], 10)

    def test_stage2_oracle_audit_fails_fast_on_fingerprint_mismatch(self) -> None:
        record = {
            "event_id": "x.json",
            "claim": "claim",
            "gold_label": "false",
            "candidate_pool": [{"text": "a"}, {"text": "b"}],
            "selected_indices": [0],
            "candidate_pool_metadata": {"chunk_mmr_fingerprint": "wrong"},
            "search_objective": "margin",
        }

        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            audit_stage2_oracle_record(record, expected_fingerprint="432dfc970e75")


if __name__ == "__main__":
    unittest.main()
