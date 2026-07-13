from __future__ import annotations

from scripts.phase13_scifact.export_scifact_submission import (
    _evaluate,
    _prompt_candidates,
    _submission_row,
)


def test_submission_row_groups_scifact_sentences_and_uses_prediction_label() -> None:
    trace = {
        "event_id": "52",
        "selected_candidates": [
            {"scifact_doc_id": "11", "scifact_sentence_ids": [1, 11], "map_relation": "refute"},
            {"scifact_doc_id": "11", "scifact_sentence_ids": [13], "map_relation": "support"},
        ],
    }

    row = _submission_row(trace, pred_label="support", max_sentences_per_doc=3)

    assert row["id"] == 52
    assert row["evidence"] == {"11": {"label": "SUPPORT", "sentences": [1, 11, 13]}}


def test_submission_row_emits_empty_evidence_for_nei_prediction() -> None:
    trace = {
        "event_id": "53",
        "selected_candidates": [
            {"scifact_doc_id": "11", "scifact_sentence_ids": [1], "map_relation": "support"},
        ],
    }

    row = _submission_row(trace, pred_label="nei", max_sentences_per_doc=3)

    assert row["id"] == 53
    assert row["evidence"] == {}


def test_submission_row_uses_relation_aligned_prompt_top1() -> None:
    trace = {
        "event_id": "54",
        "selected_candidates": [
            {"scifact_doc_id": "wrong-fullpool", "scifact_sentence_ids": [0], "map_relation": "support"},
        ],
    }
    prompt_candidates = [
        {"scifact_doc_id": "11", "scifact_sentence_ids": [1], "map_relation": "refute"},
        {"scifact_doc_id": "12", "scifact_sentence_ids": [3, 4], "map_relation": "support"},
        {"scifact_doc_id": "13", "scifact_sentence_ids": [5], "map_relation": "support"},
    ]

    row = _submission_row(
        trace,
        pred_label="support",
        max_sentences_per_doc=3,
        prompt_candidates=prompt_candidates,
    )

    assert row["evidence"] == {"12": {"label": "SUPPORT", "sentences": [3, 4]}}
    assert row["_evidence_source"] == "prompt_relation_aligned_top1"


def test_submission_row_falls_back_to_prompt_top1_without_aligned_relation() -> None:
    row = _submission_row(
        {"event_id": "55"},
        pred_label="contradict",
        max_sentences_per_doc=3,
        prompt_candidates=[
            {"scifact_doc_id": "21", "scifact_sentence_ids": [2], "map_relation": "background"},
            {"scifact_doc_id": "22", "scifact_sentence_ids": [4], "map_relation": "irrelevant"},
        ],
    )

    assert row["evidence"] == {"21": {"label": "CONTRADICT", "sentences": [2]}}
    assert row["_evidence_source"] == "prompt_top1_fallback"


def test_prompt_candidates_respect_post_truncation_evidence_count() -> None:
    candidates = _prompt_candidates(
        {
            "evidence_count": 1,
            "candidates": [
                {"scifact_doc_id": "31", "scifact_sentence_ids": [1]},
                {"scifact_doc_id": "32", "scifact_sentence_ids": [2]},
            ],
        }
    )

    assert [candidate["scifact_doc_id"] for candidate in candidates] == ["31"]


def test_scifact_official_style_eval_toy_case() -> None:
    predictions = [
        {
            "id": 52,
            "evidence": {
                "11": {"label": "SUPPORT", "sentences": [1, 11, 13]},
                "16": {"label": "CONTRADICT", "sentences": [18, 20]},
            },
        }
    ]
    gold = [
        {
            "id": 52,
            "claim": "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors.",
            "evidence": {
                "11": [
                    {"label": "SUPPORT", "sentences": [0, 1]},
                    {"label": "SUPPORT", "sentences": [11]},
                ],
                "15": [{"label": "SUPPORT", "sentences": [4]}],
            },
            "cited_doc_ids": [11, 15],
        }
    ]

    metrics = _evaluate(predictions, gold)

    assert metrics["abstract_label_only"]["f1"] == 0.5
    assert metrics["abstract_label_rationale"] == metrics["abstract"]
    assert metrics["sentence_selection_only"]["f1"] == metrics["sentence"]["f1"]
    assert metrics["sentence_selection_label"] == metrics["sentence"]
    assert metrics["abstract"]["precision"] == 0.5
    assert metrics["abstract"]["recall"] == 0.5
    assert metrics["abstract"]["f1"] == 0.5
    assert metrics["sentence"]["precision"] == 0.2
    assert metrics["sentence"]["recall"] == 0.25
    assert round(metrics["sentence"]["f1"], 6) == round(2 / 9, 6)


def test_scifact_label_only_does_not_require_a_complete_rationale() -> None:
    metrics = _evaluate(
        [{"id": 1, "evidence": {"11": {"label": "SUPPORT", "sentences": [9]}}}],
        [
            {
                "id": 1,
                "claim": "Supported claim.",
                "evidence": {"11": [{"label": "SUPPORT", "sentences": [1]}]},
            }
        ],
    )

    assert metrics["abstract_label_only"]["f1"] == 1.0
    assert metrics["abstract_label_rationale"]["f1"] == 0.0
    assert metrics["primary_comparison"] == {
        "abstract_label_only_f1": 1.0,
        "sentence_selection_label_f1": 0.0,
    }


def test_scifact_selection_only_does_not_require_the_correct_label() -> None:
    metrics = _evaluate(
        [{"id": 1, "evidence": {"11": {"label": "CONTRADICT", "sentences": [1]}}}],
        [
            {
                "id": 1,
                "claim": "Supported claim.",
                "evidence": {"11": [{"label": "SUPPORT", "sentences": [1]}]},
            }
        ],
    )

    assert metrics["sentence_selection_only"]["f1"] == 1.0
    assert metrics["sentence_selection_label"]["f1"] == 0.0
    assert metrics["abstract_label_only"]["f1"] == 0.0


def test_scifact_claim_label_metrics_credit_empty_nei_prediction() -> None:
    metrics = _evaluate(
        [{"id": 1, "evidence": {}}],
        [{"id": 1, "claim": "Unknown claim.", "evidence": {}, "cited_doc_ids": [10]}],
    )

    assert metrics["claim_label"]["accuracy"] == 1.0
    assert metrics["claim_label"]["per_class"]["NEI"]["f1"] == 1.0
