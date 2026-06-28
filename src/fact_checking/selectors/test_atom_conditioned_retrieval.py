from __future__ import annotations

import unittest

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample
from fact_checking.selectors import atom_conditioned_retrieval as acr
from fact_checking.selectors.atom_conditioned_retrieval import (
    AtomRetrievalParams,
    align_atoms_and_chunks,
    build_atom_conditioned_retrieval_row,
    merge_atom_routes,
    score_atom_routes,
)


class AtomConditionedRetrievalTest(unittest.TestCase):
    def test_align_missing_event_raises(self) -> None:
        with self.assertRaises(ValueError):
            align_atoms_and_chunks(
                [{"event_id": "missing", "claim_atoms": []}],
                [self._sample("event0")],
            )

    def test_atom_route_scoring_uses_query_rendering_and_atom_ids(self) -> None:
        sample = self._sample(
            "event0",
            texts=["foreign aid vote", "unrelated text"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        routes = score_atom_routes(
            sample,
            atom={
                "atom_id": "A1",
                "proposition": "Adam Schiff voted to allocate foreign aid.",
                "query_rendering": "What foreign aid did Adam Schiff vote to allocate?",
            },
            atom_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            params=AtomRetrievalParams(per_atom_keep=2, alpha_dense=1.0, alpha_lexical=0.0, alpha_bm25=0.0),
        )

        self.assertEqual(routes[0]["candidate_idx"], 0)
        self.assertEqual(routes[0]["atom_id"], "A1")
        self.assertIn("query_rendering", routes[0])
        self.assertNotIn("question_id", routes[0])

    def test_rrf_merge_deduplicates_and_emits_atom_fields_only(self) -> None:
        sample = self._sample(
            "event0",
            texts=["same evidence", "same evidence"],
            embeddings=[[1.0, 0.0], [0.9, 0.0]],
        )
        pool = merge_atom_routes(
            sample,
            [
                {"candidate_idx": 0, "atom_id": "A1", "rank": 1, "hybrid_score": 0.8},
                {"candidate_idx": 1, "atom_id": "A2", "rank": 1, "hybrid_score": 0.9},
            ],
            params=AtomRetrievalParams(),
        )

        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["matched_atom_ids"], ["A1", "A2"])
        self.assertEqual(pool[0]["atom_route_hit_count"], 2)
        self.assertIn("atom_routes", pool[0])
        self.assertNotIn("qd_question_routes", pool[0])
        self.assertNotIn("question_routes", pool[0])

    def test_end_to_end_row_contains_atom_pool_and_selected_evidence(self) -> None:
        sample = self._sample(
            "event0",
            texts=["target evidence", "other evidence"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        atom_row = {
            "event_id": "event0",
            "claim": "target claim",
            "claim_atoms": [
                {
                    "atom_id": "A1",
                    "proposition": "target claim",
                    "query_rendering": "target evidence?",
                    "keywords": ["target"],
                }
            ],
        }
        row = build_atom_conditioned_retrieval_row(
            atom_row,
            sample,
            atom_embeddings={("event0", "A1"): np.asarray([1.0, 0.0], dtype=np.float32)},
            params=AtomRetrievalParams(alpha_dense=1.0, alpha_lexical=0.0, alpha_bm25=0.0),
        )

        self.assertEqual(row["claim_atoms"][0]["atom_id"], "A1")
        self.assertEqual(row["selected_evidence"][0]["text"], "target evidence")
        self.assertEqual(row["merged_candidate_pool"][0]["atom_pool_rank"], 1)

    def test_baseline_top_k_can_exceed_atom_selector_top_k(self) -> None:
        texts = [f"target evidence {idx}" for idx in range(8)]
        sample = self._sample(
            "event0",
            texts=texts,
            embeddings=[[1.0, float(idx) / 10.0] for idx in range(8)],
        )
        params = AtomRetrievalParams(
            per_atom_keep=8,
            merged_pool_size=8,
            selector_top_k=3,
            baseline_top_k=8,
            alpha_dense=1.0,
            alpha_lexical=0.0,
            alpha_bm25=0.0,
        )
        atom_row = {
            "event_id": "event0",
            "claim": "target claim",
            "claim_atoms": [
                {
                    "atom_id": "A1",
                    "proposition": "target claim",
                    "query_rendering": "target evidence?",
                }
            ],
        }

        baseline = acr.build_atom_baseline_claim_mmr_row(sample, params=params)
        atom_row_out = build_atom_conditioned_retrieval_row(
            atom_row,
            sample,
            atom_embeddings={("event0", "A1"): np.asarray([1.0, 0.0], dtype=np.float32)},
            params=params,
        )

        self.assertEqual(len(baseline["candidates"]), 8)
        self.assertEqual(len(atom_row_out["selected_evidence"]), 3)

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
