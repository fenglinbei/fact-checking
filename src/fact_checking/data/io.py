from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from fact_checking.data.constants import LABEL2ID
from fact_checking.data.types import SampleRecord, SentenceRecord
from fact_checking.utils.text import clean_text, robust_sentence_split


def load_split(path: str | Path) -> list[SampleRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[SampleRecord] = []
    for item in payload:
        label = clean_text(str(item["label"])).lower()
        if label not in LABEL2ID:
            raise ValueError(f"Unknown label: {label!r} in {path}")
        records.append(
            SampleRecord(
                event_id=str(item["event_id"]),
                claim=clean_text(str(item["claim"])),
                label=label,
                explain=clean_text(str(item.get("explain", ""))),
                reports=item.get("reports", []),
            )
        )
    return records

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def iter_sentences(sample: SampleRecord, min_char_len: int = 10) -> Iterable[SentenceRecord]:
    for report in sample.reports:
        report_id = report.get("report_id", "unknown")
        link = report.get("link")
        domain = report.get("domain")
        content = clean_text(str(report.get("content", "")))
        for sent_idx, sent in enumerate(robust_sentence_split(content)):
            if len(sent) < min_char_len:
                continue
            yield SentenceRecord(
                event_id=sample.event_id,
                report_id=report_id,
                sent_idx=sent_idx,
                text=sent,
                link=link,
                domain=domain,
                raw=report,
            )