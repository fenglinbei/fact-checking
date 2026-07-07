from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fact_checking.data.constants import (
    label2id_for_schema,
    label_definitions_for_schema,
    labels_for_schema,
    letter_order_for_schema,
    normalize_label_schema,
    task_name_for_schema,
)
from fact_checking.data.io import load_split


HOVER_ROOT = Path("data/raw/HoVer")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_hover2_label_schema_uses_official_binary_labels() -> None:
    assert normalize_label_schema("hover") == "hover2"
    assert normalize_label_schema("HoVer2") == "hover2"
    assert task_name_for_schema("hover2") == "HoVer"
    assert labels_for_schema("hover2") == ["supported", "not_supported"]
    assert label2id_for_schema("hover2") == {"supported": 0, "not_supported": 1}
    assert letter_order_for_schema("hover2") == ["A", "B"]
    assert set(label_definitions_for_schema("hover2")) == {"supported", "not_supported"}


def test_hover_loader_normalizes_labels_and_preserves_metadata(tmp_path: Path) -> None:
    raw_path = tmp_path / "train.json"
    _write_json(
        raw_path,
        [
            {
                "uid": "u1",
                "claim": "Claim one.",
                "label": "SUPPORTED",
                "supporting_facts": [["Page A", 0], ["Page B", 1]],
                "num_hops": 2,
                "hpqa_id": "hpqa-1",
            },
            {
                "uid": "u2",
                "claim": "Claim two.",
                "label": "NOT_SUPPORTED",
                "supporting_facts": [],
                "num_hops": 3,
                "hpqa_id": "hpqa-2",
            },
        ],
    )

    records = load_split(raw_path, dataset="hover", label_schema="hover2")

    assert [record.event_id for record in records] == ["u1", "u2"]
    assert [record.label for record in records] == ["supported", "not_supported"]
    assert records[0].claim == "Claim one."
    assert records[0].reports == []
    assert records[0].metadata["supporting_facts"] == [["Page A", 0], ["Page B", 1]]
    assert records[0].metadata["num_hops"] == 2
    assert records[0].metadata["hpqa_id"] == "hpqa-1"
    assert records[0].metadata["source_dataset"] == "hover"


def test_hover_loader_accepts_claim_only_test_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "test.json"
    _write_json(raw_path, [{"uid": "u3", "claim": "Claim-only test row."}])

    records = load_split(raw_path, dataset="hover", label_schema="hover2")

    assert len(records) == 1
    assert records[0].event_id == "u3"
    assert records[0].label == ""
    assert records[0].reports == []
    assert records[0].metadata["has_gold_label"] is False


def test_downloaded_hover_release_counts_if_available() -> None:
    if not HOVER_ROOT.exists():
        pytest.skip("HoVer raw data is not available in this checkout.")

    expected = {
        "train": (18171, {"supported": 11023, "not_supported": 7148}),
        "val": (4000, {"supported": 2000, "not_supported": 2000}),
        "test": (4000, {"": 4000}),
    }
    for split, (n_expected, counts_expected) in expected.items():
        records = load_split(HOVER_ROOT / f"{split}.json", dataset="hover", label_schema="hover2")
        assert len(records) == n_expected
        assert Counter(record.label for record in records) == counts_expected
