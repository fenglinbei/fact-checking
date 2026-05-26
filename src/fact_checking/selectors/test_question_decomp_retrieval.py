from __future__ import annotations

import unittest

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample
from fact_checking.selectors.question_decomp_retrieval import (
    RetrievalParams,
    align_questions_and_chunks,
    build_baseline_claim_mmr_row,
    build_question_decomp_retrieval_row,
    compute_retrieval_metrics,
    merge_question_routes,
    oracle_selected_texts_by_event,
    score_question_routes,
    select_final_candidates_with_mmr,
)


class QuestionDecompRetrievalTest(unittest.TestCase):
    def test_align_missing_event_raises(self) -> None:
        with self.assertRaises(ValueError):
            align_questions_and_chunks(
                [{"event_id": "missing", "questions": []}],
                [self._sample("event0")],
            )

    def test_single_question_scoring_ranks_target_first(self) -> None:
        sample = self._sample(
            "event0",
            texts=["target evidence text", "unrelated text"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        routes = score_question_routes(
            sample,
            question={"id": "q1", "question": "target", "focus": "overall"},
            question_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            params=RetrievalParams(per_question_keep=2, alpha_dense=1.0, alpha_lexical=0.0, alpha_bm25=0.0),
        )
        self.assertEqual(routes[0]["candidate_idx"], 0)
        self.assertGreater(routes[0]["hybrid_score"], routes[1]["hybrid_score"])

    def test_rrf_deduplicates_text_and_preserves_routes(self) -> None:
        sample = self._sample(
            "event0",
            texts=["same evidence", "same evidence", "other evidence"],
            embeddings=[[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]],
        )
        pool = merge_question_routes(
            sample,
            [
                {"candidate_idx": 0, "question_id": "q1", "rank": 1, "hybrid_score": 0.8, "focus": "overall"},
                {"candidate_idx": 1, "question_id": "q2", "rank": 1, "hybrid_score": 0.9, "focus": "entity"},
            ],
            params=RetrievalParams(),
        )
        self.assertEqual(len(pool), 1)
        self.assertEqual(len(pool[0]["question_routes"]), 2)
        self.assertEqual(pool[0]["original_candidate_idx"], 1)

    def test_q1_weight_beats_other_question_at_same_rank(self) -> None:
        sample = self._sample("event0", texts=["q1 evidence", "q2 evidence"])
        pool = merge_question_routes(
            sample,
            [
                {"candidate_idx": 0, "question_id": "q1", "rank": 1, "hybrid_score": 0.1, "focus": "overall"},
                {"candidate_idx": 1, "question_id": "q2", "rank": 1, "hybrid_score": 1.0, "focus": "entity"},
            ],
            params=RetrievalParams(q1_weight=1.2, other_question_weight=1.0),
        )
        self.assertEqual(pool[0]["text"], "q1 evidence")

    def test_final_mmr_outputs_top_k_with_selection_rank(self) -> None:
        sample = self._sample("event0", texts=["a", "b", "c"], embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        pool = [
            {"text": "a", "original_candidate_idx": 0, "rrf_score": 0.9},
            {"text": "b", "original_candidate_idx": 1, "rrf_score": 0.8},
            {"text": "c", "original_candidate_idx": 2, "rrf_score": 0.7},
        ]
        selected = select_final_candidates_with_mmr(sample, pool, params=RetrievalParams(selector_top_k=2))
        self.assertEqual(len(selected), 2)
        self.assertEqual([row["selection_rank"] for row in selected], [1, 2])

    def test_oracle_matching_uses_text_not_candidate_index(self) -> None:
        oracle = oracle_selected_texts_by_event(
            [
                {
                    "event_id": "event0",
                    "candidate_pool": [{"text": "wrong"}, {"text": "Target Evidence"}],
                    "selected_indices": [1],
                }
            ]
        )
        metrics = compute_retrieval_metrics(
            question_rows=[{"event_id": "event0", "questions": [{"id": "q1"}]}],
            qd_rows=[
                {
                    "event_id": "event0",
                    "merged_candidate_pool": [{"text": "Target Evidence"}],
                    "selected_evidence": [{"text": "Target Evidence", "question_routes": [{"focus": "overall"}]}],
                }
            ],
            baseline_rows=[{"event_id": "event0", "candidates": [{"text": "wrong"}]}],
            oracle_texts=oracle,
        )
        self.assertEqual(metrics["question_decomp"]["oracle_selected_recall@5"], 1.0)
        self.assertEqual(metrics["baseline_claim_mmr"]["oracle_selected_recall@5"], 0.0)
        self.assertIn("delta", metrics)

    def test_end_to_end_row_and_baseline_metrics_are_present(self) -> None:
        sample = self._sample(
            "event0",
            texts=["target evidence", "other evidence"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        question_row = {
            "event_id": "event0",
            "claim": "target claim",
            "questions": [{"id": "q1", "question": "target evidence?", "focus": "overall"}],
        }
        qd = build_question_decomp_retrieval_row(
            question_row,
            sample,
            question_embeddings={("event0", "q1"): np.asarray([1.0, 0.0], dtype=np.float32)},
            params=RetrievalParams(alpha_dense=1.0, alpha_lexical=0.0, alpha_bm25=0.0),
        )
        baseline = build_baseline_claim_mmr_row(
            sample,
            params=RetrievalParams(alpha_dense=1.0, alpha_lexical=0.0, alpha_bm25=0.0),
        )
        metrics = compute_retrieval_metrics(
            question_rows=[question_row],
            qd_rows=[qd],
            baseline_rows=[baseline],
            oracle_texts={"event0": {"target evidence"}},
        )
        self.assertIn("question_decomp", metrics)
        self.assertIn("baseline_claim_mmr", metrics)

    @staticmethod
    def _sample(
        event_id: str,
        texts: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> ChunkMMRSample:
        texts = texts or ["target evidence", "other evidence"]
        if embeddings is None:
            embeddings = [[1.0, 0.0], [0.0, 1.0]][: len(texts)]
        return ChunkMMRSample(
            event_id=event_id,
            claim="target claim",
            label="true",
            explain="",
            candidates=[{"text": text} for text in texts],
            chunk_emb=np.asarray(embeddings, dtype=np.float32),
            claim_emb=np.asarray([1.0, 0.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
