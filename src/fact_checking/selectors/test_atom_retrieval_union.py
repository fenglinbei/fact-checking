from __future__ import annotations

import unittest

import numpy as np

from fact_checking.selectors.atom_retrieval_union import (
    AtomUnionSelectionParams,
    build_atom_union_pool_row,
    select_atom_union_rules,
)


class AtomRetrievalUnionTest(unittest.TestCase):
    def test_atom_union_pool_deduplicates_and_tracks_sources(self) -> None:
        row = build_atom_union_pool_row(
            baseline_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {"text": "Shared evidence", "selection_rank": 1, "hybrid_score": 0.8},
                    {"text": "Baseline only", "selection_rank": 2, "hybrid_score": 0.7},
                ],
            },
            atom_pool_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {
                        "text": "Shared evidence",
                        "merge_rank": 1,
                        "atom_rrf_score": 0.05,
                        "atom_route_hit_count": 2,
                        "atom_max_route_hybrid": 1.0,
                        "matched_atom_ids": ["A1", "A2"],
                    },
                    {
                        "text": "Atom only",
                        "merge_rank": 2,
                        "atom_rrf_score": 0.04,
                        "atom_route_hit_count": 1,
                        "atom_max_route_hybrid": 0.9,
                        "matched_atom_ids": ["A2"],
                    },
                ],
            },
        )

        self.assertEqual(len(row["candidates"]), 3)
        shared = row["candidates"][0]
        self.assertTrue(shared["from_baseline"])
        self.assertTrue(shared["from_atom_route"])
        self.assertEqual(shared["union_source"], "baseline+atom")
        self.assertEqual(shared["matched_atom_ids"], ["A1", "A2"])
        self.assertNotIn("from_qd", shared)

    def test_atom_union_source_score_prefers_atom_route_signal(self) -> None:
        union = build_atom_union_pool_row(
            baseline_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [{"text": "B1", "selection_rank": 1, "hybrid_score": 0.9}],
            },
            atom_pool_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {"text": "A1", "merge_rank": 1, "atom_rrf_score": 0.10, "atom_route_hit_count": 3, "atom_max_route_hybrid": 1.0},
                    {"text": "A2", "merge_rank": 2, "atom_rrf_score": 0.01, "atom_route_hit_count": 1, "atom_max_route_hybrid": 0.2},
                ],
            },
        )

        selected = select_atom_union_rules(union, params=AtomUnionSelectionParams(selector_top_k=2))

        self.assertEqual([c["text"] for c in selected["atom_union_interleave_top5"]], ["B1", "A1"])
        self.assertEqual(selected["atom_union_source_score_top5"][0]["text"], "A1")
        self.assertIn("atom_union_source_score", selected["atom_union_source_score_top5"][0])

    def test_atom_union_final_mmr_has_fixed_size_and_final_ranks(self) -> None:
        row = build_atom_union_pool_row(
            baseline_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {"text": "B1", "selection_rank": 1, "hybrid_score": 0.90},
                    {"text": "B2", "selection_rank": 2, "hybrid_score": 0.80},
                    {"text": "B3", "selection_rank": 3, "hybrid_score": 0.70},
                ],
            },
            atom_pool_row={
                "event_id": "event0",
                "claim": "claim",
                "candidates": [
                    {
                        "text": "A1",
                        "merge_rank": 1,
                        "atom_rrf_score": 0.05,
                        "atom_route_hit_count": 1,
                        "atom_max_route_hybrid": 0.95,
                    }
                ],
            },
            candidate_vectors={
                "b1": np.asarray([1.0, 0.0], dtype=np.float32),
                "b2": np.asarray([0.99, 0.01], dtype=np.float32),
                "b3": np.asarray([0.0, 1.0], dtype=np.float32),
                "a1": np.asarray([-1.0, 0.0], dtype=np.float32),
            },
            final_pool_size=3,
            params=AtomUnionSelectionParams(selector_top_k=3, union_mmr_lambda=0.70),
        )

        self.assertEqual(row["union_pool_size_before_mmr"], 4)
        self.assertTrue(row["union_mmr_applied"])
        self.assertEqual(len(row["candidates"]), 3)
        self.assertEqual([candidate["union_pool_rank"] for candidate in row["candidates"]], [1, 2, 3])
        self.assertEqual([candidate["atom_union_mmr_rank"] for candidate in row["candidates"]], [1, 2, 3])
        self.assertIn("A1", [candidate["text"] for candidate in row["candidates"]])


if __name__ == "__main__":
    unittest.main()
