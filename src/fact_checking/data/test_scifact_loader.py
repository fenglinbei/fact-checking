from __future__ import annotations

import json
from pathlib import Path

from fact_checking.data.constants import label2id_for_schema, labels_for_schema
from fact_checking.data.io import load_split


def test_scifact2_label_schema() -> None:
    assert labels_for_schema("scifact2") == ["support", "contradict"]
    assert label2id_for_schema("scifact") == {"support": 0, "contradict": 1, "nei": 2}
    assert labels_for_schema("scifact3") == ["support", "contradict", "nei"]


def test_load_scifact_claims_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "claims_train.jsonl"
    rows = [
        {
            "id": 1,
            "claim": "Drug A reduces biomarker B.",
            "evidence": {
                "10": [{"label": "SUPPORT", "sentences": [0, 1]}],
            },
            "cited_doc_ids": [10, 11],
        },
        {
            "id": 2,
            "claim": "Protein C increases disease D.",
            "evidence": {
                "20": [{"label": "CONTRADICT", "sentences": [2]}],
            },
            "cited_doc_ids": [20],
        },
        {
            "id": 3,
            "claim": "Unlabeled test-style claim.",
            "evidence": {},
            "cited_doc_ids": [],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    samples = load_split(path, dataset="scifact", label_schema="scifact2")

    assert [sample.event_id for sample in samples] == ["1", "2", "3"]
    assert [sample.label for sample in samples] == ["support", "contradict", ""]
    assert samples[0].metadata["source_dataset"] == "scifact"
    assert samples[0].metadata["evidence"]["10"][0]["sentences"] == [0, 1]

    three_class_samples = load_split(path, dataset="scifact", label_schema="scifact3")
    assert [sample.label for sample in three_class_samples] == ["support", "contradict", "nei"]
    assert three_class_samples[2].metadata["has_gold_label"] is True


def test_scifact_test_split_stays_unlabeled_under_scifact3(tmp_path: Path) -> None:
    path = tmp_path / "claims_test.jsonl"
    path.write_text(
        json.dumps({"id": 9, "claim": "Held-out claim.", "evidence": {}, "cited_doc_ids": [10]}) + "\n",
        encoding="utf-8",
    )

    samples = load_split(path, dataset="scifact", label_schema="scifact3")

    assert samples[0].label == ""
    assert samples[0].metadata["has_gold_label"] is False
