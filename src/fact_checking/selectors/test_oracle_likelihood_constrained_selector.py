from __future__ import annotations

import unittest

from fact_checking.selectors.oracle_likelihood_constrained_selector import (
    FORBIDDEN_FEATURE_FIELDS,
    ConstrainedSelectionParams,
    LogisticParams,
    build_feature_rows,
    build_oracle_likelihood_trace,
    cross_fit_score_rows,
    feature_names_for_set,
    filter_feature_rows,
    labels_array,
    pairwise_accuracy,
    roc_auc_score,
    score_feature_rows,
    select_constrained_likelihood_topk,
    split_event_ids_kfold,
    train_logistic,
    train_pairwise_logistic,
)


class OracleLikelihoodConstrainedSelectorTest(unittest.TestCase):
    def test_feature_extraction_excludes_forbidden_fields(self) -> None:
        rows = [_event("e1")]

        feature_rows, feature_names = build_feature_rows(rows)

        self.assertTrue(feature_rows)
        self.assertFalse(FORBIDDEN_FEATURE_FIELDS & set(feature_names))
        self.assertIn("oracle_selected", feature_rows[0])
        self.assertNotIn("oracle_selected", feature_rows[0]["features"])
        self.assertNotIn("candidate_key", feature_rows[0]["features"])

    def test_feature_set_ablation_filters_expected_groups(self) -> None:
        _, feature_names = build_feature_rows([_event("e1")])

        minus_provenance = feature_names_for_set(feature_names, "all_minus_provenance")
        directness_stance = feature_names_for_set(feature_names, "teacher_directness_stance_only")
        retrieval_quality = feature_names_for_set(feature_names, "retrieval_quality_only")

        self.assertNotIn("from_baseline", minus_provenance)
        self.assertNotIn("qd_only", minus_provenance)
        self.assertIn("direct_evidence_score", minus_provenance)
        self.assertIn("stance_prob_oppose_claim_bucket", directness_stance)
        self.assertIn("semantic_completeness_score", directness_stance)
        self.assertNotIn("retrieval_score", directness_stance)
        self.assertEqual(
            retrieval_quality,
            [
                "retrieval_score",
                "semantic_completeness_score",
                "claim_lexical_f1",
                "question_route_weight",
                "question_coverage_score",
            ],
        )

    def test_event_level_kfold_has_no_overlap(self) -> None:
        rows = []
        for idx in range(6):
            rows.extend(build_feature_rows([_event(f"e{idx}")])[0])

        folds = split_event_ids_kfold(rows, folds=3, seed=7)

        covered = set().union(*folds)
        self.assertEqual(covered, {f"e{idx}" for idx in range(6)})
        for left_idx, left in enumerate(folds):
            for right_idx, right in enumerate(folds):
                if left_idx != right_idx:
                    self.assertFalse(left & right)

    def test_logistic_trainer_learns_synthetic_signal(self) -> None:
        rows = []
        for idx in range(40):
            x = idx / 39.0
            rows.append({"event_id": f"e{idx}", "label": int(x > 0.5), "features": {"x": x}})
        model = train_logistic(
            rows[:30],
            rows[30:],
            ["x"],
            params=LogisticParams(epochs=200, lr=0.2, l2=0.0, patience=50, eval_every=10, seed=3),
        )

        scores, _ = score_feature_rows(rows, ["x"], model)
        self.assertGreater(roc_auc_score(labels_array(rows), scores), 0.95)
        self.assertGreater(float(model["weights"][0]), 0.0)

    def test_pairwise_trainer_learns_within_event_order(self) -> None:
        rows = []
        for event_idx in range(20):
            rows.append({"event_id": f"e{event_idx}", "label": 1, "features": {"x": 0.9}})
            rows.append({"event_id": f"e{event_idx}", "label": 0, "features": {"x": 0.1}})
        model = train_pairwise_logistic(
            rows[:30],
            rows[30:],
            ["x"],
            params=LogisticParams(epochs=120, lr=0.2, l2=0.0, patience=40, eval_every=10, seed=5),
        )

        scores, _ = score_feature_rows(rows, ["x"], model)
        self.assertGreater(pairwise_accuracy(rows, scores), 0.95)
        self.assertGreater(float(model["weights"][0]), 0.0)

    def test_cross_fit_scores_only_heldout_events(self) -> None:
        candidate_rows = [_event(f"e{idx}") for idx in range(8)]
        feature_rows, feature_names = build_feature_rows(candidate_rows)

        scored, models, _ = cross_fit_score_rows(
            feature_rows,
            feature_names,
            folds=4,
            params=LogisticParams(epochs=60, lr=0.05, patience=20, eval_every=10, seed=11),
        )

        self.assertEqual(len(scored), len(feature_rows))
        scored_events_by_fold = {}
        for row in scored:
            scored_events_by_fold.setdefault(int(row["oracle_likelihood_fold"]), set()).add(row["event_id"])
        for model in models:
            heldout = set(model["heldout_event_ids"])
            train = set(model["train_event_ids"])
            dev = set(model["dev_event_ids"])
            self.assertFalse(heldout & train)
            self.assertFalse(heldout & dev)
            self.assertEqual(scored_events_by_fold[int(model["fold"])], heldout)

    def test_cross_fit_pairwise_scores_only_heldout_events(self) -> None:
        candidate_rows = [_event(f"e{idx}") for idx in range(8)]
        feature_rows, feature_names = build_feature_rows(candidate_rows)
        selected_feature_names = feature_names_for_set(feature_names, "teacher_directness_stance_only")
        feature_rows = filter_feature_rows(feature_rows, selected_feature_names)

        scored, models, _ = cross_fit_score_rows(
            feature_rows,
            selected_feature_names,
            folds=4,
            objective="pairwise",
            params=LogisticParams(epochs=60, lr=0.05, patience=20, eval_every=10, seed=11),
        )

        self.assertEqual(len(scored), len(feature_rows))
        scored_events = {str(row["event_id"]) for row in scored}
        heldout_events = set()
        for model in models:
            self.assertEqual(model["objective"], "pairwise")
            heldout_events.update(model["heldout_event_ids"])
            self.assertFalse(set(model["heldout_event_ids"]) & set(model["train_event_ids"]))
        self.assertEqual(scored_events, heldout_events)

    def test_constrained_selector_keeps_anchor_and_penalizes_repeated_source(self) -> None:
        candidates = [
            _candidate("anchor", source="report:1", score=0.10, baseline_rank=1, bucket="ambiguous_claim_bucket"),
            _candidate("same_source", source="report:1", score=0.90, baseline_rank=None, bucket="support_claim_bucket"),
            _candidate("other_source", source="report:2", score=0.84, baseline_rank=None, bucket="oppose_claim_bucket"),
        ]

        selected = select_constrained_likelihood_topk(
            candidates,
            params=ConstrainedSelectionParams(top_k=2, anchor_k=1, source_penalty=0.10, stance_region_penalty=0.0),
        )

        self.assertEqual([row["candidate_key"] for row in selected], ["anchor", "other_source"])
        self.assertEqual(selected[0]["selection_origin"], "stage2_anchor")
        self.assertEqual(selected[1]["selection_origin"], "learned_fill")

    def test_trace_preserves_text_key_metrics(self) -> None:
        row = {
            "event_id": "e1",
            "claim": "claim",
            "oracle_ordered_keys": ["b", "a"],
        }
        selected = [
            _candidate("a", source="report:1", score=0.9, baseline_rank=None),
            _candidate("b", source="report:2", score=0.8, baseline_rank=None),
        ]

        trace = build_oracle_likelihood_trace(row, selected, selector_name="test", top_k=2)

        self.assertEqual(trace["recall@5"], 1.0)
        self.assertEqual(trace["jaccard@5"], 1.0)
        self.assertEqual(trace["top1_match"], 0.0)
        self.assertEqual(trace["pairwise_order_acc@5"], 0.0)


def _event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "claim": "The claim says the budget increased.",
        "gold_label": "false",
        "oracle_ordered_keys": [f"{event_id}-a"],
        "stance_bucket_names": ["oppose_claim_bucket", "ambiguous_claim_bucket", "support_claim_bucket"],
        "candidates": [
            _candidate(f"{event_id}-a", source="report:1", score=0.8, baseline_rank=1, bucket="oppose_claim_bucket", oracle=True),
            _candidate(f"{event_id}-b", source="report:2", score=0.2, baseline_rank=2, bucket="ambiguous_claim_bucket", oracle=False),
        ],
    }


def _candidate(
    key: str,
    *,
    source: str,
    score: float,
    baseline_rank: int | None,
    bucket: str = "support_claim_bucket",
    oracle: bool = False,
) -> dict:
    probs = {"oppose_claim_bucket": 0.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}
    probs[bucket] = 1.0
    return {
        "candidate_uid": key,
        "candidate_key": key,
        "text": f"{key} evidence text.",
        "from_baseline": baseline_rank is not None,
        "from_qd": baseline_rank is None,
        "baseline_rank": baseline_rank,
        "qd_pool_rank": 1 if baseline_rank is None else None,
        "union_pool_rank": 1 if baseline_rank == 1 else 2,
        "retrieval_score": score,
        "baseline_hybrid_score": score,
        "qd_rrf_score": score / 20.0,
        "qd_question_hit_count": 1,
        "qd_max_question_hybrid": score,
        "semantic_completeness_score": 0.8,
        "claim_lexical_f1": 0.5,
        "direct_evidence_score": score,
        "claim_specificity_score": score,
        "key_fact_overlap_score": score,
        "background_only_score": 1.0 - score,
        "claim_directness_score": score * score * score,
        "role_evidence_score": score,
        "teacher_stance_probs": probs,
        "stance_bucket_derived": bucket,
        "stance_expected_score": 5.0,
        "stance_entropy": 0.0,
        "question_route_weight": 0.75,
        "question_coverage_score": 0.33,
        "source_group": source,
        "oracle_selected": oracle,
        "oracle_step": 0 if oracle else -1,
        "oracle_likelihood_score": score,
    }


if __name__ == "__main__":
    unittest.main()
