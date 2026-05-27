from __future__ import annotations

import math
import unittest

from fact_checking.selectors.count_amplified_stance_bucket_selector import (
    CountAmplifiedParams,
    build_union_analysis_row,
    bucket_count_payload,
    select_count_amplified_topk,
    source_dedup_weights,
    text_ordered_selection_metrics,
)
from fact_checking.selectors.evidence_quality import enrich_quality_fields, quality_gate, question_route_weight
from fact_checking.selectors.stance_buckets import stance_score_to_probs, validate_teacher_payload


class CountAmplifiedStanceBucketSelectorTest(unittest.TestCase):
    def test_stance_score_to_probs_is_normalized_and_centered(self) -> None:
        probs = stance_score_to_probs(1.0, n_stance_buckets=3, tau=2.0)

        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertGreater(probs["oppose_claim_bucket"], probs["ambiguous_claim_bucket"])
        self.assertGreater(probs["ambiguous_claim_bucket"], probs["support_claim_bucket"])

    def test_teacher_payload_clamps_values(self) -> None:
        annotation = validate_teacher_payload({"stance_score": 0, "semantic_completeness": 12})

        self.assertEqual(annotation.stance_score, 1.0)
        self.assertEqual(annotation.semantic_completeness, 10.0)
        self.assertTrue(annotation.stance_score_clamped)
        self.assertTrue(annotation.semantic_completeness_clamped)

    def test_quality_gate_and_original_route_weight(self) -> None:
        candidate = enrich_quality_fields(
            {
                "text": "The report says Americans are working longer hours than before.",
                "from_baseline": True,
                "from_qd": False,
                "baseline_hybrid_score": 0.8,
            },
            claim="Americans are working longer hours.",
        )

        self.assertTrue(quality_gate(candidate, tau_c=0.2, tau_r=0.1))
        self.assertAlmostEqual(question_route_weight(candidate), 0.75)

    def test_union_merge_uses_chunk_key_before_text_fallback(self) -> None:
        oracle_row = {
            "event_id": "e0",
            "claim": "claim",
            "gold_label": "false",
            "selected_indices": [0],
            "candidate_pool": [
                {"candidate_idx": 0, "source_index": 7, "text": "Shared evidence text."},
                {"candidate_idx": 1, "source_index": 8, "text": "Original evidence."},
            ],
            "candidate_scores": [
                {"candidate_idx": 0, "hybrid_rank": 0, "hybrid_score": 0.9},
                {"candidate_idx": 1, "hybrid_rank": 1, "hybrid_score": 0.8},
            ],
        }
        qd_row = {
            "event_id": "e0",
            "claim": "claim",
            "candidates": [
                {
                    "original_candidate_idx": 7,
                    "text": "Different surface form but same chunk.",
                    "qd_pool_rank": 1,
                    "qd_question_routes": [{"question_id": "q1", "rank": 1}],
                },
                {"text": "QD only evidence.", "qd_pool_rank": 2},
            ],
        }

        row = build_union_analysis_row(oracle_row, qd_row)
        shared = [candidate for candidate in row["candidates"] if candidate["source_index"] == 7][0]

        self.assertEqual(len(row["candidates"]), 3)
        self.assertTrue(shared["from_baseline"])
        self.assertTrue(shared["from_qd"])
        self.assertEqual(shared["qd_pool_rank"], 1)

    def test_source_dedup_weight_downweights_same_source_bucket(self) -> None:
        params = CountAmplifiedParams(tau_c=0.0, tau_r=0.0)
        candidates = [
            _candidate("a", "report:1", {"oppose_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}),
            _candidate("b", "report:1", {"oppose_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}),
            _candidate("c", "report:2", {"support_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "oppose_claim_bucket": 0.0}),
        ]

        weights = source_dedup_weights(
            candidates,
            bucket_names=["oppose_claim_bucket", "ambiguous_claim_bucket", "support_claim_bucket"],
            params=params,
        )

        self.assertAlmostEqual(weights[("a", "oppose_claim_bucket")], 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(weights[("b", "oppose_claim_bucket")], 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(weights[("c", "support_claim_bucket")], 1.0)

    def test_gamma_amplifies_larger_bucket_more_than_linear(self) -> None:
        candidates = [
            _candidate("o1", "report:1", {"oppose_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}),
            _candidate("o2", "report:2", {"oppose_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}),
            _candidate("s1", "report:3", {"support_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "oppose_claim_bucket": 0.0}),
        ]
        linear = bucket_count_payload(
            candidates,
            bucket_names=["oppose_claim_bucket", "ambiguous_claim_bucket", "support_claim_bucket"],
            params=CountAmplifiedParams(alpha=0.0, gamma_stance=1.0, tau_c=0.0, tau_r=0.0),
        )
        amplified = bucket_count_payload(
            candidates,
            bucket_names=["oppose_claim_bucket", "ambiguous_claim_bucket", "support_claim_bucket"],
            params=CountAmplifiedParams(alpha=0.0, gamma_stance=1.8, tau_c=0.0, tau_r=0.0),
        )

        linear_ratio = linear["bucket_mass"]["oppose_claim_bucket"] / linear["bucket_mass"]["support_claim_bucket"]
        amplified_ratio = amplified["bucket_mass"]["oppose_claim_bucket"] / amplified["bucket_mass"]["support_claim_bucket"]
        self.assertGreater(amplified_ratio, linear_ratio)

    def test_selected_count_penalty_moves_second_slot_on_tie(self) -> None:
        candidates = [
            _candidate("a", "report:1", {"oppose_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}),
            _candidate("b", "report:2", {"support_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "oppose_claim_bucket": 0.0}),
        ]

        selected, slot_trace, _ = select_count_amplified_topk(
            candidates,
            params=CountAmplifiedParams(top_k=2, alpha=0.0, gamma_stance=1.0, tau_c=0.0, tau_r=0.0),
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(slot_trace[0]["chosen_stance_bucket"], "oppose_claim_bucket")
        self.assertEqual(slot_trace[1]["chosen_stance_bucket"], "support_claim_bucket")

    def test_text_ordered_metrics_preserve_set_and_order_distinction(self) -> None:
        metrics = text_ordered_selection_metrics(
            ["a", "b"],
            [{"candidate_key": "b"}, {"candidate_key": "a"}],
            top_k=2,
        )

        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["jaccard@5"], 1.0)
        self.assertEqual(metrics["top1_match"], 0.0)
        self.assertEqual(metrics["pairwise_order_acc@5"], 0.0)


def _candidate(key: str, source: str, probs: dict[str, float]) -> dict:
    return {
        "candidate_uid": key,
        "candidate_key": key,
        "text": f"{key} says a complete factual sentence.",
        "semantic_completeness_score": 1.0,
        "relevance_gate_score": 1.0,
        "retrieval_score": 1.0,
        "teacher_stance_probs": probs,
        "stance_bucket_derived": max(probs, key=probs.get),
        "source_group": source,
        "question_route_weight": 1.0,
        "question_coverage_score": 1.0,
        "union_pool_rank": 1,
    }


if __name__ == "__main__":
    unittest.main()
