import re

from fact_checking.data.constants import LABEL2ID, LETTER2LABEL

_LABEL_PATTERNS = [
    ("pants-fire", r"\bpants\s*-?\s*fire\b"),
    ("barely-true", r"\bbarely\s*-?\s*true\b"),
    ("half-true", r"\bhalf\s*-?\s*true\b"),
    ("mostly-true", r"\bmostly\s*-?\s*true\b"),
    ("false", r"\bfalse\b"),
    ("true", r"\btrue\b"),
]

_LETTER_LINE_PATTERN = re.compile(r"(?mi)^\s*label\s*:\s*([A-F])\b")


def _normalize_for_match(text: str) -> str:
    clean = text.strip().lower()
    clean = re.sub(r"[_/]+", " ", clean)
    clean = re.sub(r"[^a-z\-\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean


def _parse_label_id(raw_text: str) -> int:
    # 1. Letter-form output (Label: A/B/C/D/E/F) — primary path under label_format=letter.
    letter_match = _LETTER_LINE_PATTERN.search(raw_text)
    if letter_match:
        letter = letter_match.group(1).upper()
        label = LETTER2LABEL.get(letter)
        if label is not None:
            return LABEL2ID[label]

    # 2. Legacy full-name 'Label: <name>' line.
    label_line_match = re.search(r"(?mi)^\s*label\s*:\s*([^\n\r]+)", raw_text)
    if label_line_match:
        clean = _normalize_for_match(label_line_match.group(1))
        for label, pattern in _LABEL_PATTERNS:
            if re.search(pattern, clean):
                return LABEL2ID[label]

    # 3. Fallback: scan whole text for any label keyword.
    clean = _normalize_for_match(raw_text)
    for label, pattern in _LABEL_PATTERNS:
        if re.search(pattern, clean):
            return LABEL2ID[label]

    return -1
