import re

from fact_checking.data.constants import label2id_for_schema, labels_for_schema, letter2label_for_schema, letter_order_for_schema

_LIAR_LABEL_PATTERNS = [
    ("pants-fire", r"\bpants\s*-?\s*fire\b"),
    ("barely-true", r"\bbarely\s*-?\s*true\b"),
    ("half-true", r"\bhalf\s*-?\s*true\b"),
    ("mostly-true", r"\bmostly\s*-?\s*true\b"),
    ("false", r"\bfalse\b"),
    ("true", r"\btrue\b"),
]

_RAWFC_LABEL_PATTERNS = [
    ("false", r"\bfalse\b"),
    ("half", r"\bhalf\b|\bhalf\s*-?\s*true\b|\bpartly\s+true\b"),
    ("true", r"\btrue\b"),
]

_HOVER_LABEL_PATTERNS = [
    ("not_supported", r"\bnot\s+supported\b|\bnot\s+support\b|\bnotsupported\b"),
    ("supported", r"\bsupported\b|\bsupport\b"),
]


def _normalize_for_match(text: str) -> str:
    clean = text.strip().lower()
    clean = re.sub(r"[_/]+", " ", clean)
    clean = re.sub(r"[^a-z\-\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean


def _allowed_letters_pattern(label_schema: str | None) -> str:
    letters = "".join(re.escape(letter) for letter in letter_order_for_schema(label_schema))
    return f"[{letters}]"


def _label_patterns(label_schema: str | None) -> list[tuple[str, str]]:
    labels = labels_for_schema(label_schema)
    if labels == ["false", "half", "true"]:
        return _RAWFC_LABEL_PATTERNS
    if labels == ["supported", "not_supported"]:
        return _HOVER_LABEL_PATTERNS
    return [item for item in _LIAR_LABEL_PATTERNS if item[0] in set(labels)]


def _parse_label_id(raw_text: str, *, label_schema: str | None = None) -> int:
    letter_pattern = _allowed_letters_pattern(label_schema)
    letter_line_pattern = re.compile(rf"(?mi)^\s*label\s*:\s*({letter_pattern})\b")
    bare_letter_pattern = re.compile(rf"(?is)^\s*({letter_pattern})\s*$")
    letter2label = letter2label_for_schema(label_schema)
    label2id = label2id_for_schema(label_schema)

    # 1. Letter-form output (Label: A/B/C/D/E/F) — primary path under label_format=letter.
    letter_match = letter_line_pattern.search(raw_text)
    if letter_match:
        letter = letter_match.group(1).upper()
        label = letter2label.get(letter)
        if label is not None:
            return label2id[label]

    # 2. Some constrained decoders return only the selected letter.
    bare_letter_match = bare_letter_pattern.match(raw_text)
    if bare_letter_match:
        letter = bare_letter_match.group(1).upper()
        label = letter2label.get(letter)
        if label is not None:
            return label2id[label]

    # 3. Legacy full-name 'Label: <name>' line.
    label_line_match = re.search(r"(?mi)^\s*label\s*:\s*([^\n\r]+)", raw_text)
    if label_line_match:
        clean = _normalize_for_match(label_line_match.group(1))
        for label, pattern in _label_patterns(label_schema):
            if re.search(pattern, clean):
                return label2id[label]

    # 4. Fallback: scan whole text for any label keyword.
    clean = _normalize_for_match(raw_text)
    for label, pattern in _label_patterns(label_schema):
        if re.search(pattern, clean):
            return label2id[label]

    return -1
