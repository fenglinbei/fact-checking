from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fact_checking.selectors.question_decomp_reranker import (
    PairwiseRerankerParams,
    build_feature_rows,
    build_selected_rows,
    default_feature_names,
    load_pairwise_logistic_model,
    pairwise_metrics_for_rows,
    save_pairwise_logistic_model,
    score_rows,
    split_event_ids,
    train_pairwise_logistic,
)


class QuestionDecompRerankerTest(unittest.TestCase):
    def test_feature_rows_include_report_metadata(self) -> None:
        rows = build_feature_rows([self._union_row()], oracle_texts={"e0": {"target evidence"}})
        target = next(row for row in rows if row["label"] == 1)
        features = target["features"]
        self.assertGreater(features["same_report_union_candidate_count"], 1.0)
        self.assertEqual(features["same_report_has_baseline"], 1.0)
        self.assertEqual(features["domain_is_news_like"], 1.0)
        self.assertIn("qd_focus_count_overall", features)

    def test_split_event_ids_is_deterministic(self) -> None:
        rows = [{"event_id": f"e{i}", "features": {}, "label": 0} for i in range(10)]
        first = split_event_ids(rows, val_fraction=0.2, seed=7)
        second = split_event_ids(rows, val_fraction=0.2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 2)

    def test_pairwise_reranker_prefers_positive_feature(self) -> None:
        rows = []
        for event_id in ("e0", "e1", "e2"):
            rows.append({"event_id": event_id, "label": 1, "features": {"signal": 1.0}, "text": "pos"})
            rows.append({"event_id": event_id, "label": 0, "features": {"signal": 0.0}, "text": "neg"})
        feature_names = ["signal"]
        model = train_pairwise_logistic(
            rows,
            rows,
            feature_names,
            params=PairwiseRerankerParams(epochs=80, eval_every=10, patience=30, lr=0.2),
        )
        scores = score_rows(rows, feature_names, model)
        self.assertGreater(float(np.mean(scores[::2])), float(np.mean(scores[1::2])))
        metrics = pairwise_metrics_for_rows(rows, feature_names, model)
        self.assertEqual(metrics["n_pairs"], 3)
        self.assertGreater(metrics["pairwise_acc"], 0.9)

    def test_pairwise_model_roundtrip(self) -> None:
        rows = [
            {"event_id": "e0", "label": 1, "features": {"signal": 1.0}, "text": "pos"},
            {"event_id": "e0", "label": 0, "features": {"signal": 0.0}, "text": "neg"},
        ]
        feature_names = ["signal"]
        model = train_pairwise_logistic(
            rows,
            [],
            feature_names,
            params=PairwiseRerankerParams(epochs=20, eval_every=5, patience=10, lr=0.2),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.npz"
            save_pairwise_logistic_model(str(path), model=model, feature_names=feature_names, metadata={"name": "test"})
            loaded = load_pairwise_logistic_model(str(path))
        np.testing.assert_allclose(loaded["weights"], model["weights"])
        self.assertEqual(loaded["feature_names"], feature_names)
        self.assertEqual(loaded["metadata"]["name"], "test")

    def test_anchor_selection_keeps_baseline_first(self) -> None:
        rows = [
            {"event_id": "e0", "claim": "c", "candidate_key": "b1", "text": "b1", "from_baseline": True, "baseline_rank": 1, "qd_pool_rank": None, "union_source": "baseline"},
            {"event_id": "e0", "claim": "c", "candidate_key": "q1", "text": "q1", "from_baseline": False, "baseline_rank": None, "qd_pool_rank": 1, "union_source": "qd"},
        ]
        selected = build_selected_rows(rows, [0.0, 10.0], top_k=2, mode="anchor", baseline_anchor_k=1)
        self.assertEqual(selected[0]["candidates"][0]["text"], "b1")
        self.assertEqual(selected[0]["candidates"][1]["text"], "q1")

    @staticmethod
    def _union_row() -> dict:
        return {
            "event_id": "e0",
            "claim": "target claim",
            "candidates": [
                {
                    "text": "target evidence",
                    "canonical_text": "target evidence",
                    "from_baseline": True,
                    "from_qd": True,
                    "baseline_rank": 1,
                    "baseline_hybrid_score": 1.0,
                    "qd_pool_rank": 1,
                    "qd_rrf_score": 0.05,
                    "qd_question_hit_count": 2,
                    "qd_max_question_hybrid": 1.0,
                    "qd_question_routes": [{"question_id": "q1", "focus": "overall", "rank": 1}],
                    "union_pool_rank": 1,
                    "report_id": "r1",
                    "sent_idx": 0,
                    "chunk_sent_indices": [0],
                    "source_report": {"domain": "https://cnn.com"},
                },
                {
                    "text": "other evidence",
                    "canonical_text": "other evidence",
                    "from_baseline": False,
                    "from_qd": True,
                    "baseline_rank": None,
                    "qd_pool_rank": 2,
                    "qd_rrf_score": 0.04,
                    "qd_question_hit_count": 1,
                    "qd_max_question_hybrid": 0.8,
                    "qd_question_routes": [{"question_id": "q2", "focus": "entity", "rank": 2}],
                    "union_pool_rank": 2,
                    "report_id": "r1",
                    "sent_idx": 1,
                    "chunk_sent_indices": [1],
                    "source_report": {"domain": "https://cnn.com"},
                },
            ],
        }


if __name__ == "__main__":
    unittest.main()
