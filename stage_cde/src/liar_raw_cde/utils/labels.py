from __future__ import annotations

LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

ALIASES = {
    "pants on fire": "pants-fire",
    "pants fire": "pants-fire",
    "pants_fire": "pants-fire",
    "pants-fire": "pants-fire",
    "false": "false",
    "mostly true": "mostly-true",
    "mostly_true": "mostly-true",
    "mostly-true": "mostly-true",
    "barely true": "barely-true",
    "barely_true": "barely-true",
    "barely-true": "barely-true",
    "half true": "half-true",
    "half_true": "half-true",
    "half-true": "half-true",
    "true": "true",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in ALIASES:
        return ALIASES[normalized]
    raise KeyError(f"Unknown label: {label}")


def label_to_id(label: str) -> int:
    return LABEL_TO_ID[normalize_label(label)]


def id_to_label(label_id: int) -> str:
    return ID_TO_LABEL[int(label_id)]


def label_to_rank(label: str) -> int:
    return label_to_id(label) + 1
