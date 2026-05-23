from __future__ import annotations

import unittest

import torch

from fact_checking.selectors.llm_action import (
    action_token,
    build_action_samples,
    build_vig_index,
    parse_action,
    score_action_choices,
    softmax_deltas,
)
from fact_checking.selectors.stage2_oracle import Stage2OracleExample


class LLMActionSelectorTest(unittest.TestCase):
    def test_action_token_and_parse_are_strict_two_digit_ids(self) -> None:
        self.assertEqual(action_token(4), "E04")
        self.assertEqual(action_token(14), "E14")
        self.assertEqual(parse_action("E03"), 3)
        self.assertEqual(parse_action("choose E12 now"), 12)
        self.assertIsNone(parse_action("candidate 3"))
        with self.assertRaises(ValueError):
            action_token(100)

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
        self.assertEqual(samples[0]["target_action"], "E02")
        self.assertEqual(samples[1]["target_action"], "E00")
        self.assertIn(2, samples[0]["remaining_indices"])
        self.assertNotIn(2, samples[1]["remaining_indices"])
        self.assertIn("E02", samples[0]["prompt"])
        self.assertNotIn(example.gold_label, samples[0]["prompt"])
        self.assertEqual(samples[1]["prefix_indices"], [2])

    def test_score_action_choices_uses_constrained_action_likelihood(self) -> None:
        model = _FakeChoiceModel(vocab_size=8, preferred_action_id=3)
        tokenizer = _FakeTokenizer()
        sample = {
            "prompt": "prompt",
            "choices": [
                {"candidate_idx": 0, "action": "E00"},
                {"candidate_idx": 1, "action": "E01"},
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
                {"candidate_idx": 0, "action": "E00"},
                {"candidate_idx": 1, "action": "E01"},
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
            "choices": [{"candidate_idx": 0, "action": "E00"}],
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


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if text == "E00":
            return {"input_ids": [2]}
        if text == "E01":
            return {"input_ids": [3]}
        return {"input_ids": [4, 4]}


class _MultiTokenActionTokenizer(_FakeTokenizer):
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if text == "E00":
            return {"input_ids": [2, 5]}
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


if __name__ == "__main__":
    unittest.main()
