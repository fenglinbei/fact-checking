"""Oracle-free Stage A/B pipeline for LIAR-RAW."""

__all__ = ["LABELS", "LABEL2ID", "ID2LABEL"]

LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
