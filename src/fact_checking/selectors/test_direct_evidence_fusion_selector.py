from __future__ import annotations

import unittest

from fact_checking.selectors.direct_evidence_fusion_selector import (
    DIRECT_CE_FUSION_FEATURES,
    FUSION_RANK_PREFIX,
    FUSION_Z_PREFIX,
    LogisticParams,
    build_all_fusion_traces,
    build_fusion_feature_rows,
    lambda_tag,
    merge_oracle_and_direct_ce_rows,
    run_refit_fusion,
    select_fusion_topk,
    validate_lambda_zero_reproduces_baseline,
)
from fact_checking.selectors.oracle_likelihood_constrained_selector import (
    FORBIDDEN_FEATURE_FIELDS,
    ORACLE_LIKELIHOOD_SELECTOR,
)


class DirectEvidenceFusionSelectorTest(unittest.TestCase):
    def test_merge_alignment_success_and_failures(self) -> None:
        oracle = [_event("e1")]
        direct = [_direct_row(_event("e1"))]

        merged = merge_oracle_and_direct_ce_rows(oracle, direct)

        self.assertEqual(len(merged), 1)
        self.assertIn("direct_ce_raw_score", merged[0]["candidates"][0])

        missing = [_direct_row(_event("e1"))]
        missing[0]["candidates"] = missing[0]["candidates"][:1]
        with self.assertRaisesRegex(ValueError, "Candidate set mismatch"):
            merge_oracle_and_direct_ce_rows(oracle, missing)

        duplicate = [_direct_row(_event("e1"))]
        duplicate[0]["candidates"].append(dict(duplicate[0]["candidates"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            merge_oracle_and_direct_ce_rows(oracle, duplicate)

    def test_lambda_zero_reproduces_v03_top5(self) -> None:
        rows = [merge_oracle_and_direct_ce_rows([_event("e1")], [_direct_row(_event("e1"))])[0]]
        for candidate in rows[0]["candidates"]:
            candidate["fusion_refit_score"] = 0.0

        traces = build_all_fusion_traces(rows, top_k=2, lambdas=[0.0])

        validate_lambda_zero_reproduces_baseline(traces)
        by_selector = {trace["selector_name"]: trace for trace in traces}
        self.assertEqual(
            by_selector[ORACLE_LIKELIHOOD_SELECTOR]["selected_keys"],
            by_selector[f"{FUSION_Z_PREFIX}_0_top5"]["selected_keys"],
        )
        self.assertEqual(
            by_selector[ORACLE_LIKELIHOOD_SELECTOR]["selected_keys"],
            by_selector[f"{FUSION_RANK_PREFIX}_0_top5"]["selected_keys"],
        )

    def test_event_z_and_rank_features_are_event_local(self) -> None:
        rows = merge_oracle_and_direct_ce_rows(
            [_event("e1"), _event("e2", direct_scores=(0.9, 0.8, 0.7))],
            [_direct_row(_event("e1")), _direct_row(_event("e2", direct_scores=(0.9, 0.8, 0.7)))],
        )

        for row in rows:
            z_values = [candidate["direct_ce_event_z"] for candidate in row["candidates"]]
            self.assertAlmostEqual(sum(z_values), 0.0, places=6)
            best = max(row["candidates"], key=lambda candidate: candidate["direct_ce_score"])
            self.assertEqual(best["direct_ce_rank_recip"], 1.0)

    def test_refit_fusion_has_no_event_leakage_and_features_are_safe(self) -> None:
        rows = merge_oracle_and_direct_ce_rows(
            [_event(f"e{idx}") for idx in range(8)],
            [_direct_row(_event(f"e{idx}")) for idx in range(8)],
        )
        feature_rows, feature_names = build_fusion_feature_rows(rows)

        self.assertTrue(set(DIRECT_CE_FUSION_FEATURES) <= set(feature_names))
        self.assertFalse(FORBIDDEN_FEATURE_FIELDS & set(feature_names))
        self.assertNotIn("oracle_selected", feature_rows[0]["features"])

        _, models, _, _, _ = run_refit_fusion(
            rows,
            folds=4,
            params=LogisticParams(epochs=40, lr=0.05, patience=20, eval_every=10, seed=17),
        )

        for model in models:
            heldout = set(model["heldout_event_ids"])
            self.assertFalse(heldout & set(model["train_event_ids"]))
            self.assertFalse(heldout & set(model["dev_event_ids"]))

    def test_fusion_tie_breaker_is_stable(self) -> None:
        candidates = [
            _candidate("c", oracle_score=0.5, direct_score=0.8, union_rank=3),
            _candidate("b", oracle_score=0.6, direct_score=0.1, union_rank=2),
            _candidate("a", oracle_score=0.6, direct_score=0.1, union_rank=1),
        ]
        for candidate in candidates:
            candidate["same_score"] = 1.0

        selected = select_fusion_topk(
            candidates,
            score_field="same_score",
            top_k=3,
            selector_name="test",
            origin="test",
            lambda_value=0.5,
        )

        self.assertEqual([row["candidate_key"] for row in selected], ["a", "b", "c"])


def _event(event_id: str, direct_scores: tuple[float, float, float] = (0.2, 0.8, 0.5)) -> dict:
    return {
        "event_id": event_id,
        "claim": "The claim says the budget increased.",
        "gold_label": "false",
        "oracle_ordered_keys": [f"{event_id}-a"],
        "stance_bucket_names": ["oppose_claim_bucket", "ambiguous_claim_bucket", "support_claim_bucket"],
        "candidates": [
            _candidate(f"{event_id}-a", oracle_score=0.9, direct_score=direct_scores[0], union_rank=1, oracle=True),
            _candidate(f"{event_id}-b", oracle_score=0.6, direct_score=direct_scores[1], union_rank=2, oracle=False),
            _candidate(f"{event_id}-c", oracle_score=0.3, direct_score=direct_scores[2], union_rank=3, oracle=False),
        ],
    }


def _direct_row(row: dict) -> dict:
    item = dict(row)
    candidates = []
    for candidate in row["candidates"]:
        c = dict(candidate)
        c["direct_ce_raw_score"] = c["direct_ce_score"] * 10.0 - 5.0
        c["direct_ce_model"] = "mock"
        c["direct_ce_prompt_version"] = "test"
        c["direct_ce_prompt_mode"] = "default_query"
        c["direct_ce_score_normalization"] = "identity"
        c["direct_ce_score_source"] = "mock"
        candidates.append(c)
    item["candidates"] = candidates
    return item


def _candidate(
    key: str,
    *,
    oracle_score: float,
    direct_score: float,
    union_rank: int,
    oracle: bool = False,
) -> dict:
    probs = {"oppose_claim_bucket": 1.0, "ambiguous_claim_bucket": 0.0, "support_claim_bucket": 0.0}
    return {
        "candidate_uid": key,
        "candidate_key": key,
        "text": f"{key} evidence text.",
        "from_baseline": union_rank <= 2,
        "from_qd": union_rank >= 2,
        "baseline_rank": union_rank if union_rank <= 2 else None,
        "qd_pool_rank": union_rank if union_rank >= 2 else None,
        "union_pool_rank": union_rank,
        "retrieval_score": oracle_score,
        "baseline_hybrid_score": oracle_score,
        "qd_rrf_score": oracle_score / 20.0,
        "qd_question_hit_count": 1,
        "qd_max_question_hybrid": oracle_score,
        "semantic_completeness_score": 0.8,
        "claim_lexical_f1": 0.5,
        "direct_evidence_score": direct_score,
        "claim_specificity_score": direct_score,
        "key_fact_overlap_score": direct_score,
        "background_only_score": 1.0 - direct_score,
        "claim_directness_score": direct_score,
        "role_evidence_score": direct_score,
        "teacher_stance_probs": probs,
        "stance_bucket_derived": "oppose_claim_bucket",
        "stance_expected_score": 1.0,
        "stance_entropy": 0.0,
        "question_route_weight": 0.75,
        "question_coverage_score": 0.33,
        "source_group": "report:1",
        "oracle_selected": oracle,
        "oracle_step": 0 if oracle else -1,
        "oracle_likelihood_score": oracle_score,
        "oracle_likelihood_logit": oracle_score * 4.0 - 2.0,
        "oracle_likelihood_fold": 0,
        "direct_ce_score": direct_score,
        "direct_ce_raw_score": direct_score * 10.0 - 5.0,
    }


if __name__ == "__main__":
    unittest.main()
