from __future__ import annotations

from scripts.phase13_scifact.audit_scifact_retrieval import _audit_candidate_rows


def test_audit_counts_complete_alternative_rationale() -> None:
    raw = {
        "0": {
            "id": 0,
            "evidence": {
                "10": [
                    {"label": "SUPPORT", "sentences": [0, 1]},
                    {"label": "SUPPORT", "sentences": [3]},
                ]
            },
        }
    }
    rows = [
        {
            "event_id": "0",
            "candidates": [
                {
                    "candidate_uid": "scifact:10:3",
                    "canonical_text": "evidence",
                    "doc_id": "10",
                    "chunk_sent_indices": [3],
                    "from_baseline": False,
                    "from_atom_route": True,
                }
            ],
        }
    ]

    metrics = _audit_candidate_rows(rows, raw, top_k=20)

    assert metrics["micro_gold_doc_recall"] == 1.0
    assert metrics["gold_doc_complete_rationale_rate"] == 1.0
    assert metrics["claim_full_complete_rationale_rate"] == 1.0
    assert metrics["rows_with_atom_only"] == 1
