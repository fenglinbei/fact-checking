from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.listwise import (
    FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR,
    FEATURE_ABLATION_NO_RANK_PRIOR,
    NUMERIC_FEATURE_NAMES,
    build_numeric_features,
    dropped_numeric_feature_names,
    listwise_selector_loss,
)


class ListwiseSelectorTest(unittest.TestCase):
    def test_listwise_loss_prefers_oracle_order(self) -> None:
        good_scores = [torch.tensor([0.1, 2.0, 0.3, 1.5, -0.2], dtype=torch.float32)]
        bad_scores = [torch.tensor([0.1, 1.5, 0.3, 2.0, -0.2], dtype=torch.float32)]
        selected = [[1, 3]]

        good_loss, good_parts = listwise_selector_loss(good_scores, selected)
        bad_loss, bad_parts = listwise_selector_loss(bad_scores, selected)

        self.assertLess(float(good_loss), float(bad_loss))
        self.assertGreater(good_parts["n_list_steps"], 0.0)
        self.assertGreater(good_parts["n_order_pairs"], 0.0)
        self.assertGreater(bad_parts["order_loss"], good_parts["order_loss"])

    def test_listwise_loss_backward_without_inplace_mask_error(self) -> None:
        scores = [torch.tensor([0.1, 2.0, 0.3, 1.5, -0.2], dtype=torch.float32, requires_grad=True)]

        loss, parts = listwise_selector_loss(scores, [[1, 3, 2]])
        loss.backward()

        self.assertGreater(parts["n_list_steps"], 0.0)
        self.assertIsNotNone(scores[0].grad)
        self.assertEqual(tuple(scores[0].grad.shape), tuple(scores[0].shape))

    def test_numeric_features_are_stable_length(self) -> None:
        features = build_numeric_features(
            "Claim says 70 percent in 2020.",
            {
                "text": "The article says 70 percent in 2020 and compares prior years.",
                "sent_idx": 4,
                "source_index": 12,
            },
            {
                "candidate_idx": 3,
                "hybrid_rank": 2,
                "dense_score": 0.7,
                "lexical_score": 0.25,
                "bm25_score": 2.0,
                "hybrid_score": 0.9,
            },
            idx=3,
            max_candidates=15,
        )

        self.assertEqual(len(features), 11)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features))
        self.assertGreater(features[-1], 0.0)

    def test_rank_prior_ablation_keeps_feature_shape(self) -> None:
        full = build_numeric_features(
            "Claim says 70 percent in 2020.",
            {
                "text": "The article says 70 percent in 2020 and compares prior years.",
                "sent_idx": 4,
                "source_index": 12,
            },
            {
                "candidate_idx": 3,
                "hybrid_rank": 2,
                "dense_score": 0.7,
                "lexical_score": 0.25,
                "bm25_score": 2.0,
                "hybrid_score": 0.9,
            },
            idx=3,
            max_candidates=15,
        )
        no_rank = build_numeric_features(
            "Claim says 70 percent in 2020.",
            {
                "text": "The article says 70 percent in 2020 and compares prior years.",
                "sent_idx": 4,
                "source_index": 12,
            },
            {
                "candidate_idx": 3,
                "hybrid_rank": 2,
                "dense_score": 0.7,
                "lexical_score": 0.25,
                "bm25_score": 2.0,
                "hybrid_score": 0.9,
            },
            idx=3,
            max_candidates=15,
            feature_ablation=FEATURE_ABLATION_NO_RANK_PRIOR,
        )
        hybrid_only = build_numeric_features(
            "Claim says 70 percent in 2020.",
            {
                "text": "The article says 70 percent in 2020 and compares prior years.",
                "sent_idx": 4,
                "source_index": 12,
            },
            {
                "candidate_idx": 3,
                "hybrid_rank": 2,
                "dense_score": 0.7,
                "lexical_score": 0.25,
                "bm25_score": 2.0,
                "hybrid_score": 0.9,
            },
            idx=3,
            max_candidates=15,
            feature_ablation=FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR,
        )

        self.assertEqual(len(full), len(no_rank))
        self.assertEqual(len(full), len(hybrid_only))
        name_to_idx = {name: idx for idx, name in enumerate(NUMERIC_FEATURE_NAMES)}
        self.assertEqual(no_rank[name_to_idx["hybrid_score"]], full[name_to_idx["hybrid_score"]])
        self.assertEqual(no_rank[name_to_idx["hybrid_rank_norm"]], 0.0)
        self.assertEqual(no_rank[name_to_idx["candidate_idx_norm"]], 0.0)
        self.assertEqual(hybrid_only[name_to_idx["dense_score"]], 0.0)
        self.assertEqual(hybrid_only[name_to_idx["lexical_score"]], 0.0)
        self.assertEqual(hybrid_only[name_to_idx["bm25_log_norm"]], 0.0)
        self.assertEqual(
            dropped_numeric_feature_names(FEATURE_ABLATION_HYBRID_SCORE_ONLY_PRIOR),
            [
                "dense_score",
                "lexical_score",
                "bm25_log_norm",
                "hybrid_rank_norm",
                "candidate_idx_norm",
            ],
        )


if __name__ == "__main__":
    unittest.main()
