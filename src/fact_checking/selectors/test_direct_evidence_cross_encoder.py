from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fact_checking.selectors.direct_evidence_cross_encoder import (
    QWEN_RERANKER_ST_REQUIRED_FILES,
    PROMPT_MODE_DEFAULT_QUERY,
    PROMPT_MODE_DIRECT_EVIDENCE_CUSTOM,
    DirectEvidenceCrossEncoderScorer,
    attach_direct_ce_scores,
    assert_score_sanity,
    average_precision_score,
    build_direct_ce_trace,
    build_text_only_pair,
    merge_scored_event_rows,
    score_rows_with_scorer,
    score_sanity_summary,
    select_direct_ce_topk,
    select_event_shard,
    select_source_diverse_direct_ce_topk,
    validate_local_qwen_reranker_snapshot,
)


class DirectEvidenceCrossEncoderTest(unittest.TestCase):
    def test_prompt_builder_excludes_forbidden_metadata(self) -> None:
        row = _event("e1")
        row["gold_label"] = "FORBIDDEN_SENTINEL"
        candidate = row["candidates"][0]
        candidate["candidate_key"] = "FORBIDDEN_SENTINEL"
        candidate["from_baseline"] = "FORBIDDEN_SENTINEL"

        pair = build_text_only_pair(row, candidate)

        self.assertIn("Claim one", pair.query)
        self.assertIn("directly says", pair.passage)
        self.assertNotIn("FORBIDDEN_SENTINEL", pair.query)
        self.assertNotIn("FORBIDDEN_SENTINEL", pair.passage)
        self.assertEqual(pair.source_fields, ("claim", "text"))

    def test_cross_encoder_backend_raises_clear_error(self) -> None:
        class FailingCrossEncoder:
            def __init__(self, *args, **kwargs) -> None:
                raise TypeError("not compatible")

        with self.assertRaisesRegex(RuntimeError, "no fallback backend"):
            DirectEvidenceCrossEncoderScorer(cross_encoder_cls=FailingCrossEncoder)

    def test_local_qwen_snapshot_requires_sentence_transformers_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="Qwen3-Reranker-8B-") as tmp:
            path = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Sentence Transformers v5.4 integration files"):
                validate_local_qwen_reranker_snapshot(str(path))
            for rel in QWEN_RERANKER_ST_REQUIRED_FILES:
                target = path / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}", encoding="utf-8")

            validate_local_qwen_reranker_snapshot(str(path))

    def test_cross_encoder_prompt_modes_and_raw_scores(self) -> None:
        captured: list[dict] = []
        case = self

        class FakeCrossEncoder:
            def __init__(self, *args, **kwargs) -> None:
                captured.append(dict(kwargs))

            def predict(self, payload, *, batch_size: int, show_progress_bar: bool):
                case.assertEqual(batch_size, 2)
                case.assertFalse(show_progress_bar)
                return [5.0, -5.0]

        scorer = DirectEvidenceCrossEncoderScorer(
            cross_encoder_cls=FakeCrossEncoder,
            prompt_mode=PROMPT_MODE_DIRECT_EVIDENCE_CUSTOM,
        )
        scores = scorer.score_pairs(
            [
                build_text_only_pair(_event("e1"), _event("e1")["candidates"][0]),
                build_text_only_pair(_event("e1"), _event("e1")["candidates"][2]),
            ],
            batch_size=2,
            show_progress_bar=False,
        )
        self.assertIn("prompts", captured[0])
        self.assertEqual(captured[0]["default_prompt_name"], "direct_evidence")
        self.assertEqual(scores.raw_scores, [5.0, -5.0])
        self.assertGreater(scores.scores[0], scores.scores[1])

        DirectEvidenceCrossEncoderScorer(
            cross_encoder_cls=FakeCrossEncoder,
            prompt_mode=PROMPT_MODE_DEFAULT_QUERY,
        )
        self.assertNotIn("prompts", captured[1])
        self.assertNotIn("default_prompt_name", captured[1])

    def test_input_id_dtype_repair_hook_casts_float_indices(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception:
            self.skipTest("torch is not installed")

        class FakeQwenForward(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen_dtype = None

            def forward(self, input_ids=None):
                self.seen_dtype = input_ids.dtype
                return input_ids

        class FakeCrossEncoder(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.qwen = FakeQwenForward()

            def predict(self, payload, *, batch_size: int, show_progress_bar: bool):
                self.qwen(input_ids=torch.tensor([[1.0, 2.0]], dtype=torch.float32))
                return [1.0]

        scorer = DirectEvidenceCrossEncoderScorer(cross_encoder_cls=FakeCrossEncoder)
        scorer.score_pairs(
            [build_text_only_pair(_event("e1"), _event("e1")["candidates"][0])],
            batch_size=1,
            show_progress_bar=False,
        )

        self.assertGreaterEqual(scorer.input_id_dtype_repair_hook_count, 1)
        self.assertEqual(scorer.model.qwen.seen_dtype, torch.long)

    def test_input_id_hook_rejects_empty_sequences(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception:
            self.skipTest("torch is not installed")

        class FakeQwenForward(nn.Module):
            def forward(self, input_ids=None):
                return input_ids

        class FakeCrossEncoder(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.qwen = FakeQwenForward()

            def predict(self, payload, *, batch_size: int, show_progress_bar: bool):
                self.qwen(input_ids=torch.empty((2, 0), dtype=torch.long))
                return [1.0, 1.0]

        scorer = DirectEvidenceCrossEncoderScorer(cross_encoder_cls=FakeCrossEncoder)
        with self.assertRaisesRegex(RuntimeError, "became empty"):
            scorer.score_pairs(
                [
                    build_text_only_pair(_event("e1"), _event("e1")["candidates"][0]),
                    build_text_only_pair(_event("e1"), _event("e1")["candidates"][1]),
                ],
                batch_size=2,
                show_progress_bar=False,
            )

    def test_shard_split_and_merge_cover_events_once(self) -> None:
        rows = [_event(f"e{idx}") for idx in range(7)]
        shards = [
            select_event_shard(rows, num_shards=3, shard_index=idx)
            for idx in range(3)
        ]

        merged_ids = [row["event_id"] for shard in shards for row in shard]

        self.assertEqual(sorted(merged_ids), [f"e{idx}" for idx in range(7)])
        merged = merge_scored_event_rows(rows, [row for shard in shards for row in shard])
        self.assertEqual([row["event_id"] for row in merged], [f"e{idx}" for idx in range(7)])

    def test_mock_scorer_attaches_scores(self) -> None:
        rows = [_event("e1")]

        scored = score_rows_with_scorer(
            rows,
            None,
            batch_size=2,
            model_name="mock",
            mock_scores=True,
        )

        candidates = scored[0]["candidates"]
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all("direct_ce_score" in candidate for candidate in candidates))
        self.assertTrue(all("direct_ce_raw_score" in candidate for candidate in candidates))
        self.assertTrue(all(candidate["direct_ce_prompt_mode"] == "direct_evidence_custom" for candidate in candidates))
        self.assertTrue(all(candidate["direct_ce_score_source"] == "mock_lexical_overlap" for candidate in candidates))

    def test_text_key_metrics_trace(self) -> None:
        row = _event("e1")
        selected = [
            _candidate("b", "Evidence b", score=0.7, source="report:2", oracle=True),
            _candidate("a", "Evidence a", score=0.8, source="report:1", oracle=True),
        ]

        trace = build_direct_ce_trace(row, selected, selector_name="test", top_k=2)

        self.assertEqual(trace["recall@5"], 1.0)
        self.assertEqual(trace["jaccard@5"], 1.0)
        self.assertEqual(trace["top1_match"], 0.0)
        self.assertEqual(trace["pairwise_order_acc@5"], 0.0)

    def test_source_diverse_selector_applies_penalty_without_rank_prior(self) -> None:
        candidates = [
            _candidate("a", "A", score=0.90, source="report:1"),
            _candidate("b", "B", score=0.86, source="report:1"),
            _candidate("c", "C", score=0.84, source="report:2"),
        ]

        selected = select_source_diverse_direct_ce_topk(candidates, top_k=2, source_penalty=0.05)

        self.assertEqual([row["candidate_key"] for row in selected], ["a", "c"])
        self.assertEqual(selected[1]["same_source_selected_count"], 0)

    def test_direct_ce_topk_tie_breaks_by_candidate_key(self) -> None:
        candidates = [
            _candidate("b", "B", score=0.5, source="report:2"),
            _candidate("a", "A", score=0.5, source="report:1"),
        ]

        selected = select_direct_ce_topk(candidates, top_k=2)

        self.assertEqual([row["candidate_key"] for row in selected], ["a", "b"])

    def test_attach_scores_requires_every_candidate(self) -> None:
        rows = [_event("e1")]
        scores = {("e1", "e1-a", "a"): 0.9}

        with self.assertRaises(KeyError):
            attach_direct_ce_scores(rows, scores, model_name="mock")

    def test_average_precision_tie_groups_use_step_precision(self) -> None:
        self.assertAlmostEqual(
            average_precision_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]),
            0.5,
        )

    def test_score_sanity_detects_collapsed_event_scores(self) -> None:
        rows = [_event("e1"), _event("e2")]
        for row in rows:
            for candidate in row["candidates"]:
                candidate["direct_ce_score"] = 0.5
                candidate["direct_ce_raw_score"] = 0.5

        summary = score_sanity_summary(rows, min_score_std=1e-4, min_unique_scores=3, max_event_all_tie_rate=0.5)

        self.assertFalse(summary["passes_score_sanity_gate"])
        self.assertEqual(summary["n_event_all_tie"], 2)
        with self.assertRaisesRegex(RuntimeError, "score sanity check failed"):
            assert_score_sanity(summary)


def _event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "claim": "Claim one says the budget increased.",
        "oracle_ordered_keys": ["a", "b"],
        "gold_label": "false",
        "candidates": [
            _candidate("a", "This evidence directly says the budget increased.", score=0.9, source="report:1", oracle=True, uid=f"{event_id}-a"),
            _candidate("b", "This evidence directly says the budget did not increase.", score=0.8, source="report:2", oracle=True, uid=f"{event_id}-b"),
            _candidate("c", "This background evidence discusses budgets generally.", score=0.1, source="report:1", oracle=False, uid=f"{event_id}-c"),
        ],
    }


def _candidate(
    key: str,
    text: str,
    *,
    score: float,
    source: str,
    oracle: bool = False,
    uid: str | None = None,
) -> dict:
    return {
        "candidate_uid": uid or key,
        "candidate_key": key,
        "text": text,
        "source_group": source,
        "direct_ce_score": float(score),
        "semantic_completeness_score": 0.8,
        "direct_evidence_score": 0.7,
        "teacher_stance_probs": {
            "oppose_claim_bucket": 0.0,
            "ambiguous_claim_bucket": 1.0,
            "support_claim_bucket": 0.0,
        },
        "stance_bucket_derived": "ambiguous_claim_bucket",
        "oracle_selected": bool(oracle),
        "oracle_step": 0 if oracle else -1,
    }


if __name__ == "__main__":
    unittest.main()
