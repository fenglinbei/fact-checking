from __future__ import annotations

import unittest

from fact_checking.selectors.direct_evidence_cross_encoder import (
    DirectEvidenceCrossEncoderScorer,
    attach_direct_ce_scores,
    build_direct_ce_trace,
    build_text_only_pair,
    merge_scored_event_rows,
    score_rows_with_scorer,
    select_direct_ce_topk,
    select_event_shard,
    select_source_diverse_direct_ce_topk,
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
