from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from fact_checking.selectors.roundtable import (
    RoundtableParams,
    build_pool_comparison,
    cluster_factions_for_pool,
    normalize_original_candidates,
    normalize_qd_union_candidates,
    oracle_ordered_keys,
    select_roundtable_topk,
)


class RoundtableTest(unittest.TestCase):
    def test_canonical_matching_across_original_and_qd_union(self) -> None:
        oracle_row = {
            "event_id": "e0",
            "claim": "claim",
            "gold_label": "true",
            "selected_indices": [0, 1],
            "candidate_pool": [
                {"candidate_idx": 0, "source_index": 0, "text": "Shared Evidence"},
                {"candidate_idx": 1, "source_index": 1, "text": "Original Only"},
            ],
            "candidate_scores": [
                {"candidate_idx": 0, "hybrid_score": 1.0},
                {"candidate_idx": 1, "hybrid_score": 0.5},
            ],
        }
        qd_row = {
            "event_id": "e0",
            "candidates": [
                {"text": " shared   evidence ", "original_candidate_idx": 0},
                {"text": "QD Only", "original_candidate_idx": 2},
            ],
        }

        oracle_keys = oracle_ordered_keys(oracle_row)
        original = normalize_original_candidates(oracle_row)
        qd = normalize_qd_union_candidates(
            qd_row,
            oracle_key_to_step={key: step for step, key in enumerate(oracle_keys)},
        )
        comparison = build_pool_comparison(
            event_id="e0",
            oracle_keys=oracle_keys,
            original_candidates=original,
            qd_candidates=qd,
        )

        self.assertEqual(comparison["overlap_count"], 1)
        self.assertEqual(comparison["qd_only_count"], 1)
        self.assertEqual(comparison["original_only_count"], 1)
        self.assertEqual(comparison["oracle_selected_preserved_by_qd_union_count"], 1)
        self.assertEqual(comparison["oracle_selected_dropped_by_qd_union_count"], 1)

    def test_cluster_factions_is_deterministic_on_tiny_embeddings(self) -> None:
        candidates = [
            _candidate("a", 0, score=0.9),
            _candidate("b", 1, score=0.8),
            _candidate("c", 2, score=0.7),
        ]
        sample = SimpleNamespace(
            chunk_emb=np.asarray(
                [
                    [1.0, 0.0],
                    [0.98, 0.02],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )

        labeled, factions = cluster_factions_for_pool(
            candidates,
            sample=sample,
            params=RoundtableParams(similarity_threshold=0.90, min_factions=2, max_factions=3),
        )

        self.assertEqual(len(factions), 2)
        by_key = {row["candidate_key"]: row["faction_id"] for row in labeled}
        self.assertEqual(by_key["a"], by_key["b"])
        self.assertNotEqual(by_key["a"], by_key["c"])

    def test_roundtable_selector_covers_factions_before_filling(self) -> None:
        candidates = [
            {**_candidate("f1-high", 0, score=1.0), "faction_id": "F1"},
            {**_candidate("f1-low", 1, score=0.9), "faction_id": "F1"},
            {**_candidate("f2", 2, score=0.2), "faction_id": "F2"},
        ]
        factions = [
            {"faction_id": "F1", "strength_score": 1.0},
            {"faction_id": "F2", "strength_score": 0.2},
        ]

        selected = select_roundtable_topk(
            candidates,
            factions,
            top_k=2,
            selector_name="roundtable_test",
        )

        self.assertEqual([row["candidate_key"] for row in selected], ["f1-high", "f2"])
        self.assertEqual([row["selection_rank"] for row in selected], [1, 2])

    def test_oracle_selected_flags_for_qd_union_use_canonical_text(self) -> None:
        qd = normalize_qd_union_candidates(
            {
                "event_id": "e0",
                "candidates": [
                    {"text": "Target Evidence", "original_candidate_idx": 0},
                    {"text": "Other Evidence", "original_candidate_idx": 1},
                ],
            },
            oracle_key_to_step={"target evidence": 0},
        )

        self.assertTrue(qd[0]["oracle_selected"])
        self.assertEqual(qd[0]["oracle_step"], 0)
        self.assertFalse(qd[1]["oracle_selected"])


def _candidate(key: str, embedding_index: int, *, score: float) -> dict:
    return {
        "event_id": "e0",
        "pool_name": "pool",
        "pool_position": embedding_index,
        "candidate_key": key,
        "text": key,
        "embedding_index": embedding_index,
        "source_domain": f"source{embedding_index}.test",
        "roundtable_score": score,
        "hybrid_score": score,
        "stance_to_claim": "unclear",
        "qd_question_ids": [],
        "qd_question_focuses": [],
        "oracle_selected": False,
    }


if __name__ == "__main__":
    unittest.main()
