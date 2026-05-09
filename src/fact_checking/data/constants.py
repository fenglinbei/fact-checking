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