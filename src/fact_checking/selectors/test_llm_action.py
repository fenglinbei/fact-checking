from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.llm_action import (
    action_completion,
    action_token,
    build_bad_prefix_action_samples,
    build_action_samples,
    build_action_prompt,
    build_vig_index,
    choice_action_labels,
    order_candidate_indices,
    parse_action,
    prompt_action_token_boundary,
    rebuild_action_sample_with_order,
    score_action_choices,
    softmax_deltas,
    utility_target_from_choices,
)
from fact_checking.selectors.llm_action_eval import (
    AGGREGATION_BORDA,
    AGGREGATION_MEAN_ZSCORE,
    DECODE_STRATEGY_PERMUTATION,
    aggregate_candidate_choice_scores,
    evaluate_llm_action_selection,
    rollout_llm_action_example,
)
from fact_checking.selectors.stage2_oracle import Stage2OracleExample
from scripts.phase5_selectors.train.train_llm_action_selector import METRIC_SUMS_SIZE, _batch_loss, _parts_to_sums


class LLMActionSelectorTest(unittest.TestCase):
    def test_action_token_and_parse_use_single_letter_ids(self) -> None:
        self.assertEqual(action_token(4), "E")
        self.assertEqual(action_token(14), "O")
        self.assertEqual(action_completion("E"), " E")
        self.assertEqual(parse_action("D"), 3)
        self.assertEqual(parse_action("choose M now"), 12)
        self.assertIsNone(parse_action("candidate 3"))
        self.assertIsNone(parse_action("E04"))
        with self.assertRaises(ValueError):
            action_token(15)

    def test_softmax_deltas_prefers_larger_delta(self) -> None:
        probs = softmax_deltas([0.0, 1.0, -1.0], tau=0.5)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[1], probs[0])
        self.assertGreater(probs[0], probs[2])

    def test_build_vig_index_keys_by_event_step_candidate(self) -> None:
        rows = [
            {"event_id": "a", "step": 0, "candidate_idx": 2, "delta_margin": 1.0},
            {"event_id": "a", "step": 1, "candidate_idx": 0, "delta_margin": 0.5},
        ]
        index = build_vig_index(rows)
        self.assertEqual(index["a"][0][2]["delta_margin"], 1.0)
        self.assertEqual(index["a"][1][0]["delta_margin"], 0.5)

    def test_build_action_samples_uses_vig_choices_without_prompt_label_leakage(self) -> None:
        example = _example()
        vig_rows = []
        for step, selected in enumerate(example.selected_indices):
            prefix = set(example.selected_indices[:step])
            for idx in range(len(example.candidates)):
                if idx in prefix:
                    continue
                vig_rows.append(
                    {
                        "event_id": example.event_id,
                        "step": step,
                        "candidate_idx": idx,
                        "delta_margin": 2.0 if idx == selected else -1.0,
                        "after_margin": 2.0 if idx == selected else -1.0,
                    }
                )

        samples, manifest = build_action_samples(
            [example],
            vig_rows=vig_rows,
            split="train",
            top_k=2,
            max_candidate_chars=80,
            include_retrieval_scores=True,
            strict=True,
        )

        self.assertEqual(manifest["n_samples"], 2)
        self.assertEqual(samples[0]["target_action"], " C")
        self.assertEqual(samples[0]["target_action_label"], "C")
        self.assertEqual(samples[1]["target_action"], " A")
        self.assertIn(2, samples[0]["remaining_indices"])
        self.assertNotIn(2, samples[1]["remaining_indices"])
        self.assertEqual(samples[0]["oracle_selected_indices"], [2, 0])
        self.assertEqual(samples[0]["remaining_oracle_indices"], [2, 0])
        self.assertEqual(samples[1]["remaining_oracle_indices"], [0])
        self.assertIn("- C:", samples[0]["prompt"])
        self.assertFalse(samples[0]["prompt"].endswith(" "))
        self.assertNotIn(example.gold_label, samples[0]["prompt"])
        self.assertEqual(samples[1]["prefix_indices"], [2])
        self.assertIn("candidate_text_by_idx", samples[0])
        self.assertIn("candidate_score_by_idx", samples[0])
        self.assertTrue(samples[0]["has_hard_target"])
        self.assertEqual(samples[0]["prefix_source"], "oracle")
        self.assertEqual(samples[0]["target_mode"], "oracle")
        self.assertEqual(samples[0]["oracle_next_idx"], 2)

    def test_utility_target_prefers_best_delta_with_hybrid_tiebreak(self) -> None:
        target = utility_target_from_choices(
            [
                {"candidate_idx": 0, "delta_margin": -0.1, "hybrid_rank": 3},
                {"candidate_idx": 1, "delta_margin": 0.5, "hybrid_rank": 2},
                {"candidate_idx": 2, "delta_margin": 0.5, "hybrid_rank": 1},
                {"candidate_idx": 3, "delta_margin": 0.0, "hybrid_rank": 0},
            ]
        )

        self.assertEqual(target["target_idx"], 2)
        self.assertEqual(target["positive_candidate_indices"], [1, 2])
        self.assertEqual(target["negative_candidate_indices"], [0, 3])

    def test_utility_target_has_positive_when_all_deltas_are_negative(self) -> None:
        target = utility_target_from_choices(
            [
                {"candidate_idx": 0, "delta_margin": -0.5, "hybrid_rank": 0},
                {"candidate_idx": 1, "delta_margin": -0.4, "hybrid_rank": 1},
                {"candidate_idx": 2, "delta_margin": -0.55, "hybrid_rank": 2},
            ]
        )

        self.assertEqual(target["target_idx"], 1)
        self.assertEqual(target["positive_candidate_indices"], [1])
        self.assertEqual(target["negative_candidate_indices"], [0, 2])

    def test_build_action_samples_utility_target_uses_delta_margin(self) -> None:
        example = _example()
        vig_rows = [
            {"event_id": example.event_id, "step": 0, "candidate_idx": 0, "delta_margin": 0.3, "after_margin": 0.3},
            {"event_id": example.event_id, "step": 0, "candidate_idx": 1, "delta_margin": -0.2, "after_margin": -0.2},
            {"event_id": example.event_id, "step": 0, "candidate_idx": 2, "delta_margin": -0.1, "after_margin": -0.1},
        ]
        samples, manifest = build_action_samples(
            [example],
            vig_rows=vig_rows,
            split="train",
            top_k=1,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            strict=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            target_mode="utility",
        )

        self.assertEqual(manifest["target_mode"], "utility")
        self.assertEqual(samples[0]["oracle_next_idx"], 2)
        self.assertEqual(samples[0]["target_idx"], 0)
        self.assertEqual(samples[0]["target_action"], " A")
        self.assertEqual(samples[0]["positive_candidate_indices"], [0])
        self.assertEqual(samples[0]["negative_candidate_indices"], [1, 2])
        self.assertEqual(samples[0]["remaining_oracle_indices"], [2])

    def test_local_choice_labels_are_position_local(self) -> None:
        first = choice_action_labels([2, 0, 1], action_label_mode="local_choice")
        second = choice_action_labels([0, 1, 2], action_label_mode="local_choice")
        self.assertEqual(first[2], "A")
        self.assertEqual(second[2], "C")
        self.assertEqual(choice_action_labels([2, 0, 1], action_label_mode="global_index")[2], "C")

    def test_random_candidate_order_is_seeded_and_reproducible(self) -> None:
        base = [0, 1, 2, 3, 4]
        first = order_candidate_indices(base, mode="random", seed=7, event_id="evt", step=1)
        second = order_candidate_indices(base, mode="random", seed=7, event_id="evt", step=1)
        other = order_candidate_indices(base, mode="random", seed=8, event_id="evt", step=1)
        self.assertEqual(first, second)
        self.assertCountEqual(first, base)
        self.assertNotEqual(first, other)

    def test_build_action_samples_local_choice_maps_target_action(self) -> None:
        example = _example()
        vig_rows = _vig_rows_for_example(example)
        samples, manifest = build_action_samples(
            [example],
            vig_rows=vig_rows,
            split="train",
            top_k=1,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            strict=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
        )

        self.assertEqual(manifest["action_label_mode"], "local_choice")
        self.assertEqual(samples[0]["target_idx"], 2)
        self.assertEqual(samples[0]["target_action"], " C")
        self.assertEqual(samples[0]["target_action_label"], "C")
        self.assertEqual(samples[0]["choices"][2]["candidate_idx"], 2)
        self.assertEqual(samples[0]["choices"][2]["action"], " C")
        self.assertEqual(samples[0]["choices"][2]["action_label"], "C")

    def test_rebuild_action_sample_dynamic_order_is_epoch_seeded(self) -> None:
        example = _example_many()
        samples, _manifest = build_action_samples(
            [example],
            vig_rows=_vig_rows_for_example(example),
            split="train",
            top_k=1,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            strict=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
        )

        first = rebuild_action_sample_with_order(
            samples[0],
            action_label_mode="local_choice",
            candidate_order_mode="random",
            candidate_order_seed=11,
            epoch=0,
            row_index=0,
        )
        second = rebuild_action_sample_with_order(
            samples[0],
            action_label_mode="local_choice",
            candidate_order_mode="random",
            candidate_order_seed=11,
            epoch=0,
            row_index=0,
        )
        next_epoch = rebuild_action_sample_with_order(
            samples[0],
            action_label_mode="local_choice",
            candidate_order_mode="random",
            candidate_order_seed=11,
            epoch=1,
            row_index=0,
        )

        self.assertEqual(first["remaining_indices"], second["remaining_indices"])
        self.assertNotEqual(first["candidate_order_seed"], next_epoch["candidate_order_seed"])
        self.assertCountEqual(first["remaining_indices"], samples[0]["remaining_indices"])
        target_idx = int(first["target_idx"])
        target_choice = [choice for choice in first["choices"] if int(choice["candidate_idx"]) == target_idx][0]
        self.assertEqual(first["target_action"], target_choice["action"])

    def test_rebuild_action_sample_preserves_utility_sets(self) -> None:
        example = _example_many()
        rows = _vig_rows_for_example(example)
        for row in rows:
            if int(row["step"]) == 0 and int(row["candidate_idx"]) == 3:
                row["delta_margin"] = 4.0
                row["after_margin"] = 4.0
        samples, _manifest = build_action_samples(
            [example],
            vig_rows=rows,
            split="train",
            top_k=1,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            strict=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            target_mode="utility",
        )

        rebuilt = rebuild_action_sample_with_order(
            samples[0],
            action_label_mode="local_choice",
            candidate_order_mode="random",
            candidate_order_seed=11,
        )

        self.assertEqual(rebuilt["target_idx"], 3)
        self.assertIn(3, rebuilt["positive_candidate_indices"])
        self.assertNotIn(3, rebuilt["negative_candidate_indices"])
        target_choice = [choice for choice in rebuilt["choices"] if int(choice["candidate_idx"]) == 3][0]
        self.assertEqual(rebuilt["target_action"], target_choice["action"])

    def test_rebuild_action_sample_requires_structured_candidate_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_text_by_idx"):
            rebuild_action_sample_with_order(
                {"event_id": "old", "step": 0, "remaining_indices": [0, 1], "choices": []},
                action_label_mode="local_choice",
                candidate_order_mode="random",
                candidate_order_seed=11,
            )

    def test_build_bad_prefix_samples_track_remaining_oracle(self) -> None:
        example = _example()
        samples, manifest = build_bad_prefix_action_samples(
            [example],
            split="train",
            top_k=2,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            bad_prefix_sources="hybrid",
            bad_prefix_max_replacements=1,
        )

        self.assertEqual(manifest["n_samples"], 1)
        sample = samples[0]
        self.assertFalse(sample["has_hard_target"])
        self.assertEqual(sample["prefix_source"], "hybrid")
        self.assertTrue(sample["remaining_oracle_indices"])
        self.assertTrue(set(sample["prefix_indices"]).isdisjoint(set(sample["remaining_indices"])))
        self.assertTrue(set(sample["prefix_indices"]).isdisjoint(set(sample["remaining_oracle_indices"])))

    def test_bad_prefix_batch_loss_skips_hard_ce_and_backprops_pairwise(self) -> None:
        example = _example()
        samples, _manifest = build_bad_prefix_action_samples(
            [example],
            split="train",
            top_k=2,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            bad_prefix_sources="hybrid",
            bad_prefix_max_replacements=1,
        )
        model = _TrainableChoiceModel(vocab_size=8)
        loss, parts = _batch_loss(
            model,
            _FakeTokenizer(),
            [samples[0]],
            device=torch.device("cpu"),
            max_length=16,
            choice_batch_size=8,
            score_mode="action_token",
            soft_tau=0.2,
            hard_loss_weight=1.0,
            soft_loss_weight=0.0,
            set_loss_weight=1.0,
            set_loss_type="multi_positive_ce",
            pairwise_loss_weight=1.0,
            bad_prefix_hard_loss_weight=0.0,
        )

        self.assertEqual(parts["n_hard_samples"], 0.0)
        self.assertEqual(parts["n_bad_prefix_samples"], 1.0)
        self.assertGreater(parts["n_positive_samples"], 0.0)
        loss.backward()
        self.assertIsNotNone(model.bias.grad)

    def test_batch_loss_uses_utility_positive_indices_for_pairwise(self) -> None:
        sample = {
            "prompt": "prompt",
            "choices": [
                {"candidate_idx": 0, "action": " A", "delta_margin": -1.0},
                {"candidate_idx": 1, "action": " B", "delta_margin": 1.0},
                {"candidate_idx": 2, "action": " C", "delta_margin": -0.5},
            ],
            "target_idx": 1,
            "has_hard_target": True,
            "remaining_oracle_indices": [0],
            "positive_candidate_indices": [1],
            "negative_candidate_indices": [0, 2],
        }
        model = _TrainableChoiceModel(vocab_size=8)
        with torch.no_grad():
            model.bias[2] = 0.0
            model.bias[3] = 3.0
            model.bias[5] = 1.0

        loss, parts = _batch_loss(
            model,
            _FakeTokenizer(),
            [sample],
            device=torch.device("cpu"),
            max_length=16,
            choice_batch_size=8,
            score_mode="action_token",
            soft_tau=0.2,
            hard_loss_weight=0.0,
            soft_loss_weight=0.0,
            set_loss_weight=0.0,
            set_loss_type="multi_positive_ce",
            pairwise_loss_weight=1.0,
            bad_prefix_hard_loss_weight=0.0,
        )

        self.assertEqual(parts["positive_hit@1"], 1.0)
        self.assertEqual(parts["oracle_remaining_hit@1"], 0.0)
        self.assertEqual(parts["remaining_oracle_hit@1"], 0.0)
        self.assertGreater(float(loss.detach()), 0.0)

    def test_metric_sums_vector_uses_declared_size(self) -> None:
        sums = _parts_to_sums(
            {
                "loss": 1.0,
                "hard_loss": 0.5,
                "soft_loss": 0.2,
                "set_loss": 0.1,
                "pairwise_loss": 0.3,
                "target_accuracy": 1.0,
                "positive_hit@1": 1.0,
                "oracle_remaining_hit@1": 0.0,
                "positive_prob": 0.8,
                "n_samples": 2.0,
                "n_hard_samples": 2.0,
                "n_soft_samples": 2.0,
                "n_positive_samples": 2.0,
                "n_oracle_remaining_samples": 1.0,
                "n_bad_prefix_samples": 0.0,
            },
            device=torch.device("cpu"),
        )

        self.assertEqual(sums.numel(), METRIC_SUMS_SIZE)

    def test_prompt_action_token_boundary_matches_completion_action(self) -> None:
        prompt = build_action_prompt(
            _example(),
            prefix_indices=[],
            remaining_indices=[0, 1, 2],
            max_candidate_chars=80,
            include_retrieval_scores=False,
        )
        self.assertTrue(prompt.endswith("Next evidence id:"))
        self.assertFalse(prompt.endswith(" "))
        boundary = prompt_action_token_boundary(_BoundaryTokenizer(), prompt, " A")
        self.assertTrue(boundary["matches"])
        bad_boundary = prompt_action_token_boundary(_BoundaryTokenizer(), prompt + " ", "A")
        self.assertFalse(bad_boundary["matches"])

    def test_score_action_choices_uses_constrained_action_likelihood(self) -> None:
        model = _FakeChoiceModel(vocab_size=8, preferred_action_id=3)
        tokenizer = _FakeTokenizer()
        sample = {
            "prompt": "prompt",
            "choices": [
                {"candidate_idx": 0, "action": " A"},
                {"candidate_idx": 1, "action": " B"},
            ],
        }

        scored = score_action_choices(
            model,
            tokenizer,
            [sample],
            device=torch.device("cpu"),
            max_length=8,
            choice_batch_size=8,
        )

        self.assertEqual(scored.candidate_indices, [[0, 1]])
        self.assertGreater(float(scored.scores[0][1]), float(scored.scores[0][0]))

    def test_action_token_score_matches_first_continuation_token(self) -> None:
        model = _FakeChoiceModel(vocab_size=8, preferred_action_id=3)
        tokenizer = _FakeTokenizer()
        sample = {
            "prompt": "prompt",
            "choices": [
                {"candidate_idx": 0, "action": " A"},
                {"candidate_idx": 1, "action": " B"},
            ],
        }

        action_token_scored = score_action_choices(
            model,
            tokenizer,
            [sample],
            device=torch.device("cpu"),
            max_length=8,
            score_mode="action_token",
        )
        continuation_scored = score_action_choices(
            model,
            tokenizer,
            [sample],
            device=torch.device("cpu"),
            max_length=8,
            choice_batch_size=8,
            score_mode="continuation",
            include_eos=False,
        )

        torch.testing.assert_close(
            action_token_scored.scores[0] - action_token_scored.scores[0][0],
            continuation_scored.scores[0] - continuation_scored.scores[0][0],
        )

    def test_action_token_score_requires_single_token_actions(self) -> None:
        model = _FakeChoiceModel(vocab_size=8, preferred_action_id=3)
        sample = {
            "prompt": "prompt",
            "choices": [{"candidate_idx": 0, "action": " A"}],
        }

        with self.assertRaisesRegex(ValueError, "SCORE_MODE=continuation"):
            score_action_choices(
                model,
                _MultiTokenActionTokenizer(),
                [sample],
                device=torch.device("cpu"),
                max_length=8,
                score_mode="action_token",
            )

    def test_selection_eval_summarizes_rollout_metrics(self) -> None:
        result = evaluate_llm_action_selection(
            _FakeChoiceModel(vocab_size=8, preferred_action_id=3),
            _FakeTokenizer(),
            [_example()],
            device=torch.device("cpu"),
            split="val",
            top_k=2,
            max_length=16,
            score_mode="action_token",
            choice_batch_size=8,
            max_candidate_chars=80,
            include_retrieval_scores=True,
            disable_progress=True,
        )

        metrics = result["metrics"]
        self.assertEqual(metrics["n_claims"], 1)
        self.assertEqual(metrics["estimated_forward_steps"], 2)
        self.assertEqual(result["traces"][0]["selector_ordered_indices"], [1, 0])
        self.assertIn("jaccard@5", metrics["selector"])

    def test_selection_eval_local_choice_restores_candidate_idx(self) -> None:
        result = evaluate_llm_action_selection(
            _FakeChoiceModel(vocab_size=8, preferred_action_id=5),
            _FakeTokenizer(),
            [_example()],
            device=torch.device("cpu"),
            split="val",
            top_k=1,
            max_length=16,
            score_mode="action_token",
            choice_batch_size=8,
            max_candidate_chars=80,
            include_retrieval_scores=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            disable_progress=True,
        )

        self.assertEqual(result["traces"][0]["selector_ordered_indices"], [2])
        self.assertEqual(result["traces"][0]["per_step_action_scores"][0]["selected_action"], "C")

    def test_raw_decode_matches_existing_rollout_default(self) -> None:
        kwargs = dict(
            device=torch.device("cpu"),
            top_k=2,
            max_length=16,
            score_mode="action_token",
            choice_batch_size=8,
            max_candidate_chars=80,
            include_retrieval_scores=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
        )

        raw = rollout_llm_action_example(
            _FakeChoiceModel(vocab_size=8, preferred_action_id=3),
            _FakeTokenizer(),
            _example(),
            **kwargs,
        )
        explicit = rollout_llm_action_example(
            _FakeChoiceModel(vocab_size=8, preferred_action_id=3),
            _FakeTokenizer(),
            _example(),
            decode_strategy="raw",
            **kwargs,
        )

        self.assertEqual(raw["ordered_indices"], explicit["ordered_indices"])
        self.assertEqual(raw["per_step_action_scores"], explicit["per_step_action_scores"])

    def test_permutation_decode_maps_local_labels_back_to_candidate_idx(self) -> None:
        result = evaluate_llm_action_selection(
            _FakeChoiceModel(vocab_size=8, preferred_action_id=3),
            _FakeTokenizer(),
            [_example()],
            device=torch.device("cpu"),
            split="val",
            top_k=1,
            max_length=16,
            score_mode="action_token",
            choice_batch_size=8,
            max_candidate_chars=80,
            include_retrieval_scores=True,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            decode_strategy=DECODE_STRATEGY_PERMUTATION,
            num_permutations=3,
            permutation_seed=5,
            disable_progress=True,
        )

        step = result["traces"][0]["per_step_action_scores"][0]
        self.assertEqual(step["num_permutations"], 3)
        self.assertTrue(any(len(row["labels_seen"]) > 1 for row in step["choice_scores"]))
        self.assertCountEqual([row["candidate_idx"] for row in step["choice_scores"]], [0, 1, 2])

    def test_aggregate_candidate_choice_scores_applies_calibrated_scores(self) -> None:
        rows = aggregate_candidate_choice_scores(
            [
                {
                    "permutation_index": 0,
                    "candidate_idx": 0,
                    "action": "A",
                    "raw_score": 5.0,
                    "calibrated_score": 1.0,
                    "selection_score": 1.0,
                },
                {
                    "permutation_index": 0,
                    "candidate_idx": 1,
                    "action": "B",
                    "raw_score": 3.0,
                    "calibrated_score": 2.0,
                    "selection_score": 2.0,
                },
            ],
            aggregation="mean_score",
        )

        by_idx = {row["candidate_idx"]: row for row in rows}
        self.assertEqual(by_idx[0]["mean_calibrated_score"], 1.0)
        self.assertEqual(by_idx[1]["mean_calibrated_score"], 2.0)
        self.assertGreater(by_idx[1]["aggregate_score"], by_idx[0]["aggregate_score"])

    def test_calibrated_decode_subtracts_content_free_label_bias(self) -> None:
        result = rollout_llm_action_example(
            _PromptAwareChoiceModel(vocab_size=8),
            _PromptAwareTokenizer(),
            _example(),
            device=torch.device("cpu"),
            top_k=1,
            max_length=16,
            score_mode="action_token",
            choice_batch_size=8,
            max_candidate_chars=80,
            include_retrieval_scores=False,
            action_label_mode="local_choice",
            candidate_order_mode="candidate_pool",
            decode_strategy="calibrated",
            calibration_mode="content_free_width",
            calibration_alpha=1.0,
            aggregation="mean_score",
        )

        self.assertEqual(result["ordered_indices"], [1])
        by_idx = {
            row["candidate_idx"]: row
            for row in result["per_step_action_scores"][0]["choice_scores"]
        }
        self.assertEqual(by_idx[0]["mean_raw_score"], 3.0)
        self.assertEqual(by_idx[0]["mean_calibrated_score"], 1.0)
        self.assertEqual(by_idx[1]["mean_calibrated_score"], 2.0)

    def test_mean_zscore_single_permutation_preserves_order_and_borda_is_deterministic(self) -> None:
        records = [
            {"permutation_index": 0, "candidate_idx": 0, "action": "A", "raw_score": 1.0, "calibrated_score": 1.0, "selection_score": 1.0},
            {"permutation_index": 0, "candidate_idx": 1, "action": "B", "raw_score": 3.0, "calibrated_score": 3.0, "selection_score": 3.0},
            {"permutation_index": 0, "candidate_idx": 2, "action": "C", "raw_score": 2.0, "calibrated_score": 2.0, "selection_score": 2.0},
        ]

        z_rows = aggregate_candidate_choice_scores(records, aggregation=AGGREGATION_MEAN_ZSCORE)
        borda_first = aggregate_candidate_choice_scores(records, aggregation=AGGREGATION_BORDA)
        borda_second = aggregate_candidate_choice_scores(list(reversed(records)), aggregation=AGGREGATION_BORDA)

        self.assertEqual(max(z_rows, key=lambda row: row["aggregate_score"])["candidate_idx"], 1)
        self.assertEqual(borda_first, borda_second)


def _example() -> Stage2OracleExample:
    candidates = [
        {"candidate_idx": 0, "candidate_uid": "c0", "text": "Evidence zero text."},
        {"candidate_idx": 1, "candidate_uid": "c1", "text": "Evidence one text."},
        {"candidate_idx": 2, "candidate_uid": "c2", "text": "Evidence two text."},
    ]
    candidate_scores = [
        {"candidate_idx": 0, "hybrid_rank": 0, "hybrid_score": 0.9},
        {"candidate_idx": 1, "hybrid_rank": 1, "hybrid_score": 0.8},
        {"candidate_idx": 2, "hybrid_rank": 2, "hybrid_score": 0.7},
    ]
    return Stage2OracleExample(
        event_id="evt",
        claim="A test claim",
        gold_label="true",
        candidates=candidates,
        candidate_scores=candidate_scores,
        selected_indices=[2, 0],
        fingerprint="432dfc970e75",
        margin=1.0,
        is_correct=True,
        raw={},
    )


def _example_many() -> Stage2OracleExample:
    candidates = [
        {"candidate_idx": idx, "candidate_uid": f"c{idx}", "text": f"Evidence {idx} text."}
        for idx in range(5)
    ]
    candidate_scores = [
        {"candidate_idx": idx, "hybrid_rank": idx, "hybrid_score": 1.0 - idx * 0.1}
        for idx in range(5)
    ]
    return Stage2OracleExample(
        event_id="evt-many",
        claim="A test claim with more candidates",
        gold_label="true",
        candidates=candidates,
        candidate_scores=candidate_scores,
        selected_indices=[4, 1],
        fingerprint="432dfc970e75",
        margin=1.0,
        is_correct=True,
        raw={},
    )


def _vig_rows_for_example(example: Stage2OracleExample) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step, selected in enumerate(example.selected_indices):
        prefix = set(example.selected_indices[:step])
        for idx in range(len(example.candidates)):
            if idx in prefix:
                continue
            rows.append(
                {
                    "event_id": example.event_id,
                    "step": step,
                    "candidate_idx": idx,
                    "delta_margin": 2.0 if idx == selected else -1.0,
                    "after_margin": 2.0 if idx == selected else -1.0,
                }
            )
    return rows


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if text in {"A", " A"}:
            return {"input_ids": [2]}
        if text in {"B", " B"}:
            return {"input_ids": [3]}
        if text in {"C", " C"}:
            return {"input_ids": [5]}
        return {"input_ids": [4, 4]}


class _MultiTokenActionTokenizer(_FakeTokenizer):
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if text in {"A", " A"}:
            return {"input_ids": [2, 5]}
        return super().__call__(text, **_)


class _BoundaryTokenizer(_FakeTokenizer):
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if text == " A":
            return {"input_ids": [2]}
        if text.endswith("Next evidence id:"):
            return {"input_ids": [7]}
        if text.endswith("Next evidence id: A"):
            return {"input_ids": [7, 2]}
        if text.endswith("Next evidence id: "):
            return {"input_ids": [7, 8]}
        if text.endswith("Next evidence id: A") or text.endswith("Next evidence id:  A"):
            return {"input_ids": [7, 8, 9]}
        return super().__call__(text, **_)


class _FakeChoiceModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int, preferred_action_id: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.preferred_action_id = int(preferred_action_id)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> object:
        del attention_mask, use_cache
        logits = torch.zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32)
        logits[:, 1, self.preferred_action_id] = 8.0
        if input_ids.shape[1] > 2:
            logits[:, 2, 1] = 8.0
        return type("FakeOutput", (), {"logits": logits})()


class _PromptAwareTokenizer(_FakeTokenizer):
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if "[claim omitted]" in text:
            return {"input_ids": [6, 6]}
        return super().__call__(text, **_)


class _PromptAwareChoiceModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> object:
        del attention_mask, use_cache
        logits = torch.zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32)
        for row in range(input_ids.shape[0]):
            if int(input_ids[row, 0].item()) == 6:
                logits[row, -1, 2] = 2.0
            else:
                logits[row, -1, 2] = 3.0
                logits[row, -1, 3] = 2.0
        return type("FakeOutput", (), {"logits": logits})()


class _TrainableChoiceModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> object:
        del attention_mask, use_cache
        logits = self.bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        return type("FakeOutput", (), {"logits": logits})()


if __name__ == "__main__":
    unittest.main()
