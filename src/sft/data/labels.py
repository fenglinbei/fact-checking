from __future__ import annotations

from fact_checking.data.constants import label2id_for_schema


def normalize_gold_label(row: dict, *, label_schema: str | None = None) -> str:
    gold_label = str(row.get("label", "")).strip().lower()
    if gold_label not in label2id_for_schema(label_schema):
        return ""
    return gold_label
