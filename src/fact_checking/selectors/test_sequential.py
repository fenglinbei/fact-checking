from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.sequential import (
    CLAIM_FEATURE_MODE_CLAIM_ONLY,
    CLAIM_FEATURE_MODE_OFF,
    CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
    CLAIM_START_MODE_LEARNED,
    DeepInteractionPointerHead,
    PROJECTION_MODE_LINEAR,
    PROJECTION_MODE_MLP_RESIDUAL,
    infer_claim_feature_mode_from_model_config,
    infer_claim_start_mode_from_model_config,
    infer_projection_mode_from_model_config,
    mask_step_logits,
    normalize_claim_feature_mode,
    normalize_claim_start_mode,
    normalize_projection_mode,
    normalize_semantic_feature_profile,
    normalize_shallow_feature_profile,
    normalize_targeted_feature_profile,
    remaining_selected_bce_targets,
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

    def test_mask_bce_labels_all_remaining_selected_candidates(self) -> None:
        candidate_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)
        labels, valid = remaining_selected_bce_targets(
            [[1, 3]],
            candidate_mask,
            steps=2,
            max_candidates=4,
            top_k=2,
        )

        self.assertTrue(bool(valid[0, 0, 0]))
        self.assertTrue(bool(valid[0, 0, 1]))
        self.assertTrue(bool(valid[0, 0, 3]))
        self.assertEqual(float(labels[0, 0, 1]), 1.0)
        self.assertEqual(float(labels[0, 0, 3]), 1.0)
        self.assertFalse(bool(valid[0, 1, 1]))
        self.assertEqual(float(labels[0, 1, 3]), 1.0)

    def test_mask_bce_weight_adds_to_sequence_ce(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 3.0, 0.2, 1.0],
                    [0.0, -1.0e4, 0.5, 2.5],
                ]
            ],
            dtype=torch.float32,
        )
        candidate_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

        loss, parts = sequential_teacher_forcing_loss(
            logits,
            [[1, 3]],
            candidate_mask=candidate_mask,
            top_k=2,
            seq_loss_weight=1.0,
            mask_loss_weight=0.2,
        )

        expected = parts["sequence_ce_loss"] + 0.2 * parts["mask_loss"]
        self.assertGreater(parts["mask_loss"], 0.0)
        self.assertAlmostEqual(float(loss), expected, places=5)

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

    def test_pointer_head_claim_feature_requires_claim_embedding(self) -> None:
        torch.manual_seed(11)
        head = DeepInteractionPointerHead(
            hidden_size=8,
            dropout=0.0,
            claim_feature_mode=CLAIM_FEATURE_MODE_CLAIM_ONLY,
        )
        context = torch.randn(2, 4, 8)
        candidate_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )
        selected_mask = torch.zeros_like(candidate_mask)
        with self.assertRaises(ValueError):
            head.score_step(context, candidate_mask, selected_mask)

        claim_embedding = torch.randn(2, 8)
        logits = head.score_step(
            context,
            candidate_mask,
            selected_mask,
            claim_embedding=claim_embedding,
        )

        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertLess(float(logits[1, 2].detach()), -1000.0)

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

    def test_architecture_mode_normalizers_accept_legacy_aliases(self) -> None:
        self.assertEqual(normalize_projection_mode(None), PROJECTION_MODE_LINEAR)
        self.assertEqual(normalize_projection_mode("deep"), PROJECTION_MODE_LINEAR)
        self.assertEqual(normalize_projection_mode("proj2"), PROJECTION_MODE_MLP_RESIDUAL)
        self.assertEqual(normalize_projection_mode("mlp-residual"), PROJECTION_MODE_MLP_RESIDUAL)
        self.assertEqual(normalize_claim_start_mode(None), CLAIM_START_MODE_LEARNED)
        self.assertEqual(normalize_claim_start_mode("none"), CLAIM_START_MODE_LEARNED)
        self.assertEqual(
            normalize_claim_start_mode("candidate-pool-mean"),
            CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
        )
        self.assertEqual(normalize_claim_feature_mode(None), CLAIM_FEATURE_MODE_OFF)
        self.assertEqual(normalize_claim_feature_mode("h-claim"), CLAIM_FEATURE_MODE_CLAIM_ONLY)
        self.assertEqual(normalize_claim_feature_mode("claim_text"), CLAIM_FEATURE_MODE_CLAIM_ONLY)
        with self.assertRaises(ValueError):
            normalize_projection_mode("wide")
        with self.assertRaises(ValueError):
            normalize_claim_start_mode("claim_only")
        with self.assertRaises(ValueError):
            normalize_claim_feature_mode("candidate_pool_mean")

    def test_architecture_mode_inference_keeps_old_checkpoints_compatible(self) -> None:
        self.assertEqual(infer_projection_mode_from_model_config({}), PROJECTION_MODE_LINEAR)
        self.assertEqual(
            infer_projection_mode_from_model_config({"proj_residual": True}),
            PROJECTION_MODE_MLP_RESIDUAL,
        )
        self.assertEqual(
            infer_projection_mode_from_model_config({"proj_num_layers": 2}),
            PROJECTION_MODE_MLP_RESIDUAL,
        )
        self.assertEqual(
            infer_projection_mode_from_model_config(
                {},
                selector_state={
                    "item_projection": {
                        "0.weight": torch.zeros(512, 768),
                        "3.weight": torch.zeros(256, 512),
                    },
                    "proj_residual": {"weight": torch.zeros(256, 768)},
                },
            ),
            PROJECTION_MODE_MLP_RESIDUAL,
        )
        self.assertEqual(infer_claim_start_mode_from_model_config({}), CLAIM_START_MODE_LEARNED)
        self.assertEqual(
            infer_claim_start_mode_from_model_config({"claim_start": "candidate_pool_mean"}),
            CLAIM_START_MODE_CANDIDATE_POOL_MEAN,
        )
        self.assertEqual(
            infer_claim_start_mode_from_model_config({"claim_start_mode": "learned"}),
            CLAIM_START_MODE_LEARNED,
        )
        self.assertEqual(infer_claim_feature_mode_from_model_config({}), CLAIM_FEATURE_MODE_OFF)
        self.assertEqual(
            infer_claim_feature_mode_from_model_config({"claim_feature_mode": "claim_only"}),
            CLAIM_FEATURE_MODE_CLAIM_ONLY,
        )
        self.assertEqual(
            infer_claim_feature_mode_from_model_config(
                {},
                selector_state={
                    "pointer_head": {
                        "scorer.0.weight": torch.zeros(256, 49),
                    },
                },
                hidden_size=8,
            ),
            CLAIM_FEATURE_MODE_CLAIM_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
