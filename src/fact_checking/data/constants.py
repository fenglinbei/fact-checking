from __future__ import annotations

from dataclasses import dataclass


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

# Single-token aliases used to neutralize multi-token bias under generative SFT.
# Order parallels LABELS so LABELS[i] <-> LETTER_ORDER[i].
LABEL_LETTERS = {
    "pants-fire":  "A",
    "false":       "B",
    "barely-true": "C",
    "half-true":   "D",
    "mostly-true": "E",
    "true":        "F",
}
LETTER2LABEL = {letter: label for label, letter in LABEL_LETTERS.items()}
LETTER_ORDER = [LABEL_LETTERS[label] for label in LABELS]

LABEL_DEFINITIONS = {
    "pants-fire": "completely false and implausible",
    "false": "false based on the available evidence",
    "barely-true": "mostly false, with only a small element of truth",
    "half-true": "partly true and partly false",
    "mostly-true": "mostly true, with minor missing context or caveats",
    "true": "accurate based on the available evidence",
}

# 3-class coarsened labels for reduced task difficulty.
LABELS_3CLASS = ["false", "mixed", "true"]
LABEL_MAP_6TO3 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}


@dataclass(frozen=True)
class LabelSchema:
    name: str
    task_name: str
    labels: list[str]
    label_letters: dict[str, str]
    label_definitions: dict[str, str]

    @property
    def label2id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {idx: label for label, idx in self.label2id.items()}

    @property
    def letter2label(self) -> dict[str, str]:
        return {letter: label for label, letter in self.label_letters.items()}

    @property
    def letter_order(self) -> list[str]:
        return [self.label_letters[label] for label in self.labels]


RAWFC3_LABELS = ["false", "half", "true"]
RAWFC3_LABEL2ID = {label: idx for idx, label in enumerate(RAWFC3_LABELS)}
RAWFC3_ID2LABEL = {idx: label for label, idx in RAWFC3_LABEL2ID.items()}
RAWFC3_LABEL_LETTERS = {
    "false": "A",
    "half": "B",
    "true": "C",
}
RAWFC3_LETTER2LABEL = {letter: label for label, letter in RAWFC3_LABEL_LETTERS.items()}
RAWFC3_LETTER_ORDER = [RAWFC3_LABEL_LETTERS[label] for label in RAWFC3_LABELS]
RAWFC3_LABEL_DEFINITIONS = {
    "false": "false based on the available evidence",
    "half": "partly true and partly false",
    "true": "accurate based on the available evidence",
}
RAWFC_NUMERIC_LABELS = {
    1: "false",
    3: "half",
    5: "true",
    "1": "false",
    "3": "half",
    "5": "true",
}

HOVER2_LABELS = ["supported", "not_supported"]
HOVER2_LABEL2ID = {label: idx for idx, label in enumerate(HOVER2_LABELS)}
HOVER2_ID2LABEL = {idx: label for label, idx in HOVER2_LABEL2ID.items()}
HOVER2_LABEL_LETTERS = {
    "supported": "A",
    "not_supported": "B",
}
HOVER2_LETTER2LABEL = {letter: label for label, letter in HOVER2_LABEL_LETTERS.items()}
HOVER2_LETTER_ORDER = [HOVER2_LABEL_LETTERS[label] for label in HOVER2_LABELS]
HOVER2_LABEL_DEFINITIONS = {
    "supported": "the claim is supported by the retrieved Wikipedia facts",
    "not_supported": "the claim is not supported by the retrieved Wikipedia facts",
}

COVERAGE_LABELS = ["covered", "weak_covered", "uncovered"]
COVERAGE_LABEL2ID = {label: idx for idx, label in enumerate(COVERAGE_LABELS)}
COVERAGE_ID2LABEL = {idx: label for label, idx in COVERAGE_LABEL2ID.items()}
COVERAGE_LABEL_LETTERS = {
    "covered": "A",
    "weak_covered": "B",
    "uncovered": "C",
}
COVERAGE_LETTER2LABEL = {letter: label for label, letter in COVERAGE_LABEL_LETTERS.items()}
COVERAGE_LETTER_ORDER = [COVERAGE_LABEL_LETTERS[label] for label in COVERAGE_LABELS]
COVERAGE_LABEL_DEFINITIONS = {
    "covered": "the selected evidence is sufficient for deciding the claim",
    "weak_covered": "the selected evidence is partially useful but misses important support or context",
    "uncovered": "the selected evidence is insufficient for deciding the claim",
}

LABEL_SCHEMAS = {
    "liar6": LabelSchema(
        name="liar6",
        task_name="LIAR-RAW",
        labels=list(LABELS),
        label_letters=dict(LABEL_LETTERS),
        label_definitions=dict(LABEL_DEFINITIONS),
    ),
    "rawfc3": LabelSchema(
        name="rawfc3",
        task_name="RAWFC",
        labels=list(RAWFC3_LABELS),
        label_letters=dict(RAWFC3_LABEL_LETTERS),
        label_definitions=dict(RAWFC3_LABEL_DEFINITIONS),
    ),
    "hover2": LabelSchema(
        name="hover2",
        task_name="HoVer",
        labels=list(HOVER2_LABELS),
        label_letters=dict(HOVER2_LABEL_LETTERS),
        label_definitions=dict(HOVER2_LABEL_DEFINITIONS),
    ),
}

_SCHEMA_ALIASES = {
    "": "liar6",
    "default": "liar6",
    "liar": "liar6",
    "liar-raw": "liar6",
    "liar_raw": "liar6",
    "liar6": "liar6",
    "rawfc": "rawfc3",
    "rawfc3": "rawfc3",
    "hover": "hover2",
    "hover2": "hover2",
    "ho_ver": "hover2",
}


def normalize_label_schema(label_schema: str | None = None) -> str:
    key = str(label_schema or "").strip().lower().replace("-", "_")
    normalized = _SCHEMA_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unknown label_schema={label_schema!r}. Use one of {sorted(LABEL_SCHEMAS)}.")
    return normalized


def get_label_schema(label_schema: str | None = None) -> LabelSchema:
    return LABEL_SCHEMAS[normalize_label_schema(label_schema)]


def labels_for_schema(label_schema: str | None = None) -> list[str]:
    return list(get_label_schema(label_schema).labels)


def label2id_for_schema(label_schema: str | None = None) -> dict[str, int]:
    return dict(get_label_schema(label_schema).label2id)


def id2label_for_schema(label_schema: str | None = None) -> dict[int, str]:
    return dict(get_label_schema(label_schema).id2label)


def label_letters_for_schema(label_schema: str | None = None) -> dict[str, str]:
    return dict(get_label_schema(label_schema).label_letters)


def letter2label_for_schema(label_schema: str | None = None) -> dict[str, str]:
    return dict(get_label_schema(label_schema).letter2label)


def letter_order_for_schema(label_schema: str | None = None) -> list[str]:
    return list(get_label_schema(label_schema).letter_order)


def label_definitions_for_schema(label_schema: str | None = None) -> dict[str, str]:
    return dict(get_label_schema(label_schema).label_definitions)


def task_name_for_schema(label_schema: str | None = None) -> str:
    return get_label_schema(label_schema).task_name
