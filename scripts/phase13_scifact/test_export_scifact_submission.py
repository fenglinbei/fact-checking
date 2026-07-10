from __future__ import annotations

from scripts.phase13_scifact.export_scifact_submission import _evaluate, _submission_row


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

    assert metrics["abstract"]["precision"] == 0.5
    assert metrics["abstract"]["recall"] == 0.5
    assert metrics["abstract"]["f1"] == 0.5
    assert metrics["sentence"]["precision"] == 0.2
    assert metrics["sentence"]["recall"] == 0.25
    assert round(metrics["sentence"]["f1"], 6) == round(2 / 9, 6)


def test_scifact_claim_label_metrics_credit_empty_nei_prediction() -> None:
    metrics = _evaluate(
        [{"id": 1, "evidence": {}}],
        [{"id": 1, "claim": "Unknown claim.", "evidence": {}, "cited_doc_ids": [10]}],
    )

    assert metrics["claim_label"]["accuracy"] == 1.0
    assert metrics["claim_label"]["per_class"]["NEI"]["f1"] == 1.0
