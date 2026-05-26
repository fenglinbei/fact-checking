from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.cross_encoder import selector_logits, split_flat_scores


class CrossEncoderUtilitiesTest(unittest.TestCase):
    def test_selector_logits_single_label(self) -> None:
        logits = selector_logits(torch.tensor([[1.5], [-0.5]]))
        self.assertEqual(logits.tolist(), [1.5, -0.5])

    def test_selector_logits_two_label_margin(self) -> None:
        logits = selector_logits(torch.tensor([[1.0, 3.0], [2.0, -1.0]]))
        self.assertEqual(logits.tolist(), [2.0, -3.0])

    def test_split_flat_scores(self) -> None:
        groups = split_flat_scores(torch.tensor([1.0, 2.0, 3.0, 4.0]), [1, 3])
        self.assertEqual([group.tolist() for group in groups], [[1.0], [2.0, 3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
