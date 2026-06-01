from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from fact_checking.data.io import load_split


RAWFC_ROOT = Path("data/raw/RAWFC")


def test_rawfc_loader_preserves_original_three_labels_and_counts() -> None:
    if not RAWFC_ROOT.exists():
        pytest.skip("RAWFC raw data is not available in this checkout.")

    expected = {
        "train": (1612, {"false": 514, "half": 537, "true": 561}),
        "val": (200, {"false": 66, "half": 67, "true": 67}),
        "test": (200, {"false": 66, "half": 67, "true": 67}),
    }
    for split, (n_expected, counts_expected) in expected.items():
        records = load_split(RAWFC_ROOT / f"{split}.json", dataset="rawfc", label_schema="rawfc3")
        assert len(records) == n_expected
        assert Counter(record.label for record in records) == counts_expected


def test_rawfc_loader_maps_evidence_to_closed_candidate_reports() -> None:
    if not RAWFC_ROOT.exists():
        pytest.skip("RAWFC raw data is not available in this checkout.")

    record = load_split(RAWFC_ROOT / "val.json", dataset="rawfc", label_schema="rawfc3")[0]

    assert record.event_id
    assert record.claim
    assert record.label in {"false", "half", "true"}
    assert record.explain
    assert record.reports
    assert {"report_id", "content", "rawfc_evidence_idx"} <= set(record.reports[0])
