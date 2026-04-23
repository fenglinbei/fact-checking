from __future__ import annotations

from fact_checking.data.constants import LABEL2ID


def normalize_gold_label(row: dict) -> str:
    gold_label = str(row.get("label", "")).strip().lower()
    if gold_label not in LABEL2ID:
        return ""
    return gold_label
