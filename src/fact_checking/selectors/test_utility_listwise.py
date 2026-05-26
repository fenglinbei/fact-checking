from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.stage2_oracle import Stage2OracleExample
from fact_checking.selectors.utility_listwise import (
    build_utility_listwise_examples,
    permute_utility_listwise_example,
    utility_listwise_loss,
    utility_positive_indices,
    utility_soft_targets,
)


class UtilityListwiseTest(unittest.TestCase):
    def test_groups_step0_vig_rows_by_event(self) -> None:
        example = _example()
        rows = _vig_rows(example, [0.4, -0.1, 0.2])
        rows.append({"event_id": example.event_id, "step": 1, "candidate_idx": 0, "delta_margin": 99.0})

        groups = build_utility_listwise_examples(rows, [example], split="train")

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].event_id, example.event_id)
        self.assertEqual(groups[0].claim, example.claim)
        self.assertEqual(groups[0].delta_margins, [0.4, -0.1, 0.2])
        self.assertEqual([row["candidate_idx"] for row in groups[0].candidate_scores], [0, 1, 2])
        self.assertEqual(groups[0].as_candidate_group().candidates[0]["text"], "alpha evidence")

    def test_utility_targets_from_delta_margin(self) -> None:
        positives = utility_positive_indices([0.1, 0.04, -0.2, 0.08], positive_best_margin=0.05)
        soft = utility_soft_targets([0.1, 0.04, -0.2, 0.08], tau=0.3)

        self.assertEqual(positives, [0, 1, 3])
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=6)
        self.assertGreater(float(soft[0]), float(soft[2]))

    def test_all_negative_delta_still_has_positive(self) -> None:
        positives = utility_positive_indices([-0.5, -0.4, -0.8], positive_best_margin=0.05)

        self.assertEqual(positives, [1])

    def test_pairwise_loss_decreases_for_correct_order(self) -> None:
        deltas = [[1.0, 0.0, -1.0]]
        bad_scores = [torch.tensor([0.0, 1.0, 2.0], requires_grad=True)]
        good_scores = [torch.tensor([2.0, 1.0, 0.0], requires_grad=True)]

        bad_loss, bad_parts = utility_listwise_loss(
            bad_scores,
            deltas,
            pairwise_weight=1.0,
            soft_ce_weight=0.0,
            bce_weight=0.0,
        )
        good_loss, good_parts = utility_listwise_loss(
            good_scores,
            deltas,
            pairwise_weight=1.0,
            soft_ce_weight=0.0,
            bce_weight=0.0,
        )

        self.assertGreater(float(bad_loss.detach()), float(good_loss.detach()))
        self.assertEqual(bad_parts["pairwise_accuracy"], 0.0)
        self.assertEqual(good_parts["pairwise_accuracy"], 1.0)
        good_loss.backward()
        self.assertIsNotNone(good_scores[0].grad)

    def test_candidate_order_shuffle_keeps_utility_labels_aligned(self) -> None:
        example = _example()
        group = build_utility_listwise_examples(
            _vig_rows(example, [0.4, -0.1, 0.2]),
            [example],
            split="train",
        )[0]

        shuffled = permute_utility_listwise_example(group, [2, 0, 1])

        self.assertEqual(
            [row["text"] for row in shuffled.candidates],
            ["gamma evidence", "alpha evidence", "beta evidence"],
        )
        self.assertEqual(shuffled.delta_margins, [0.2, 0.4, -0.1])
        self.assertEqual(shuffled.positive_indices, [0, 1])
        self.assertEqual(shuffled.oracle_selected_indices, [0, 1])
        self.assertEqual([row["candidate_idx"] for row in shuffled.candidate_scores], [0, 1, 2])
        self.assertEqual([row["original_candidate_idx"] for row in shuffled.candidate_scores], [2, 0, 1])
        self.assertEqual(shuffled.oracle_example.selected_indices, [0, 1])


def _example() -> Stage2OracleExample:
    candidates = [
        {"candidate_idx": 0, "candidate_uid": "c0", "text": "alpha evidence", "sent_idx": 0, "source_index": 0},
        {"candidate_idx": 1, "candidate_uid": "c1", "text": "beta evidence", "sent_idx": 1, "source_index": 1},
        {"candidate_idx": 2, "candidate_uid": "c2", "text": "gamma evidence", "sent_idx": 2, "source_index": 2},
    ]
    candidate_scores = [
        {"candidate_idx": 0, "hybrid_rank": 0, "hybrid_score": 1.0, "dense_score": 0.8, "lexical_score": 0.3},
        {"candidate_idx": 1, "hybrid_rank": 1, "hybrid_score": 0.8, "dense_score": 0.7, "lexical_score": 0.2},
        {"candidate_idx": 2, "hybrid_rank": 2, "hybrid_score": 0.6, "dense_score": 0.6, "lexical_score": 0.1},
    ]
    return Stage2OracleExample(
        event_id="event.json",
        claim="claim text",
        gold_label="true",
        candidates=candidates,
        candidate_scores=candidate_scores,
        selected_indices=[2, 0],
        fingerprint="432dfc970e75",
        margin=1.0,
        is_correct=True,
        raw={},
    )


def _vig_rows(example: Stage2OracleExample, deltas: list[float]) -> list[dict[str, object]]:
    return [
        {
            "event_id": example.event_id,
            "split": "train",
            "step": 0,
            "candidate_idx": idx,
            "delta_margin": delta,
            "hybrid_score": example.candidate_scores[idx]["hybrid_score"],
            "dense_score": example.candidate_scores[idx]["dense_score"],
            "lexical_score": example.candidate_scores[idx]["lexical_score"],
        }
        for idx, delta in enumerate(deltas)
    ]


if __name__ == "__main__":
    unittest.main()
