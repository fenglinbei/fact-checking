from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.sequential import (
    DeepInteractionPointerHead,
    mask_step_logits,
    normalize_semantic_feature_profile,
    normalize_shallow_feature_profile,
    normalize_targeted_feature_profile,
    sequential_teacher_forcing_loss,
)


class SequentialSelectorTest(unittest.TestCase):
    def test_teacher_forcing_loss_prefers_oracle_actions(self) -> None:
        good = torch.tensor(
            [
                [
                    [0.0, 3.0, 0.2, 1.0],
                    [0.0, -1.0e4, 0.5, 2.5],
                ]
            ],
            dtype=torch.float32,
        )
        bad = torch.tensor(
            [
                [
                    [0.0, 0.3, 0.2, 3.0],
                    [0.0, -1.0e4, 2.5, 0.5],
                ]
            ],
            dtype=torch.float32,
        )

        good_loss, good_parts = sequential_teacher_forcing_loss(good, [[1, 3]], top_k=2)
        bad_loss, bad_parts = sequential_teacher_forcing_loss(bad, [[1, 3]], top_k=2)

        self.assertLess(float(good_loss), float(bad_loss))
        self.assertEqual(good_parts["n_steps"], 2.0)
        self.assertGreater(bad_parts["sequence_ce_loss"], good_parts["sequence_ce_loss"])

    def test_mask_step_logits_blocks_selected_and_padding(self) -> None:
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        candidate_mask = torch.tensor([[True, True, True, False]])
        selected_mask = torch.tensor([[False, True, False, False]])

        masked = mask_step_logits(logits, candidate_mask, selected_mask)

        self.assertGreater(float(masked[0, 0]), -100.0)
        self.assertLess(float(masked[0, 1]), -1000.0)
        self.assertGreater(float(masked[0, 2]), -100.0)
        self.assertLess(float(masked[0, 3]), -1000.0)

    def test_pointer_head_greedy_decode_has_no_duplicates(self) -> None:
        torch.manual_seed(7)
        head = DeepInteractionPointerHead(hidden_size=8, dropout=0.0)
        context = torch.randn(2, 4, 8)
        candidate_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )

        predictions = head.greedy_decode(context, candidate_mask, top_k=3)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0].ordered_indices), 3)
        self.assertEqual(len(set(predictions[0].ordered_indices)), 3)
        self.assertEqual(len(predictions[1].ordered_indices), 2)
        self.assertEqual(len(set(predictions[1].ordered_indices)), 2)
        self.assertTrue(all(idx in {0, 1} for idx in predictions[1].ordered_indices))

    def test_feature_profile_normalizers_lock_first_version(self) -> None:
        self.assertEqual(normalize_semantic_feature_profile("deep"), "deep")
        self.assertEqual(normalize_targeted_feature_profile("none"), "none")
        self.assertEqual(normalize_shallow_feature_profile("none"), "off")
        with self.assertRaises(ValueError):
            normalize_semantic_feature_profile("shallow_control")
        with self.assertRaises(ValueError):
            normalize_targeted_feature_profile("aspect")
        with self.assertRaises(ValueError):
            normalize_shallow_feature_profile("hybrid_only")


if __name__ == "__main__":
    unittest.main()
