from __future__ import annotations

import unittest

from fact_checking.selectors.question_decomp_union import (
    UnionSelectionParams,
    build_union_pool_row,
    compute_union_metrics,
    select_union_rules,
)


class QuestionDecompUnionTest(unittest.TestCase):
    def test_union_pool_deduplicates_and_tracks_sources(self) -> None:
        row = build_union_pool_row(
            baseline_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {"text": "Shared evidence", "selection_rank": 1, "hybrid_score": 0.8},
                    {"text": "Baseline only", "selection_rank": 2, "hybrid_score": 0.7},
                ],
            },
            qd_pool_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {
                        "text": "Shared evidence",
                        "merge_rank": 1,
                        "rrf_score": 0.05,
                        "question_hit_count": 2,
                        "max_question_hybrid": 1.0,
                    },
                    {
                        "text": "QD only",
                        "merge_rank": 2,
                        "rrf_score": 0.04,
                        "question_hit_count": 1,
                        "max_question_hybrid": 0.9,
                    },
                ],
            },
        )
        self.assertEqual(len(row["candidates"]), 3)
        shared = row["candidates"][0]
        self.assertTrue(shared["from_baseline"])
        self.assertTrue(shared["from_qd"])
        self.assertEqual(shared["union_source"], "baseline+qd")

    def test_selection_rules_are_deterministic(self) -> None:
        union = build_union_pool_row(
            baseline_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {"text": "B1", "selection_rank": 1, "hybrid_score": 0.9},
                    {"text": "B2", "selection_rank": 2, "hybrid_score": 0.8},
                ],
            },
            qd_pool_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {"text": "Q1", "merge_rank": 1, "rrf_score": 0.08, "question_hit_count": 3, "max_question_hybrid": 1.0},
                    {"text": "Q2", "merge_rank": 2, "rrf_score": 0.07, "question_hit_count": 2, "max_question_hybrid": 0.9},
                ],
            },
        )
        selected = select_union_rules(union, params=UnionSelectionParams(selector_top_k=3))
        self.assertEqual([c["text"] for c in selected["union_baseline_first_top5"]], ["B1", "B2", "Q1"])
        self.assertEqual([c["text"] for c in selected["union_interleave_top5"]], ["B1", "Q1", "B2"])
        self.assertEqual(len(selected["union_source_score_top5"]), 3)
        self.assertTrue(all("selection_rank" in c for c in selected["union_source_score_top5"]))

    def test_union_metrics_include_pool_and_rules(self) -> None:
        union = {
            "event_id": "event0",
            "candidates": [{"text": "Target"}, {"text": "Other"}],
        }
        rules = {
            "union_baseline_first_top5": [{"event_id": "event0", "candidates": [{"text": "Other"}]}],
            "union_interleave_top5": [{"event_id": "event0", "candidates": [{"text": "Target"}]}],
            "union_source_score_top5": [{"event_id": "event0", "candidates": [{"text": "Target"}]}],
        }
        metrics = compute_union_metrics(
            union_rows=[union],
            rule_rows=rules,
            oracle_texts={"event0": {"target"}},
        )
        self.assertEqual(metrics["union_pool"]["oracle_pool_recall@15"], 1.0)
        self.assertEqual(metrics["union_baseline_first_top5"]["oracle_selected_recall@5"], 0.0)
        self.assertEqual(metrics["union_interleave_top5"]["oracle_selected_recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
