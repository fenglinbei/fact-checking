import re

from  fact_checking.data.constants import LABEL2ID

_LABEL_PATTERNS = [
    ("pants-fire", r"\bpants\s*-?\s*fire\b"),
    ("barely-true", r"\bbarely\s*-?\s*true\b"),
    ("half-true", r"\bhalf\s*-?\s*true\b"),
    ("mostly-true", r"\bmostly\s*-?\s*true\b"),
    ("false", r"\bfalse\b"),
    ("true", r"\btrue\b"),
]

def _parse_label_id(raw_text: str) -> int:
    label_line_match = re.search(r"(?mi)^\s*label\s*:\s*([^\n\r]+)", raw_text)
    if label_line_match:
        label_candidate = label_line_match.group(1)
        label_from_line = _parse_label_id(label_candidate)
        if label_from_line >= 0:
            return label_from_line

    clean = raw_text.strip().lower()
    clean = re.sub(r"[_/]+", " ", clean)
    clean = re.sub(r"[^a-z\-\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)

    for label, pattern in _LABEL_PATTERNS:
        if re.search(pattern, clean):
            return LABEL2ID[label]

    return -1