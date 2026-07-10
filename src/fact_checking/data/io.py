from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fact_checking.data.constants import RAWFC_NUMERIC_LABELS, label2id_for_schema, normalize_label_schema
from fact_checking.data.types import SampleRecord, SentenceRecord
from fact_checking.utils.text import clean_text, robust_sentence_split


COVERAGE_METADATA_KEYS = (
    "coverage_label",
    "coverage_score",
    "coverage_version",
    "coverage",
)


def load_split(
    path: str | Path,
    *,
    dataset: str | None = None,
    label_schema: str | None = None,
) -> list[SampleRecord]:
    path = Path(path)
    dataset_name = _normalize_dataset(dataset, path=path, label_schema=label_schema)
    if dataset_name == "rawfc":
        return _load_rawfc_split(path, label_schema=label_schema or "rawfc3")
    if dataset_name == "hover":
        return _load_hover_split(path, label_schema=label_schema or "hover2")
    if dataset_name == "scifact":
        return _load_scifact_split(path, label_schema=label_schema or "scifact3")
    return _load_liar_raw_split(path, label_schema=label_schema or "liar6")


def _normalize_dataset(dataset: str | None, *, path: Path, label_schema: str | None) -> str:
    raw = str(dataset or "").strip().lower().replace("-", "_")
    if raw in {"rawfc", "raw_fc"}:
        return "rawfc"
    if raw in {"hover", "ho_ver"}:
        return "hover"
    if raw in {"scifact", "sci_fact"}:
        return "scifact"
    if raw in {"", "liar", "liar_raw", "liar6"}:
        if label_schema and normalize_label_schema(label_schema) == "rawfc3":
            return "rawfc"
        if label_schema and normalize_label_schema(label_schema) == "hover2":
            return "hover"
        if label_schema and normalize_label_schema(label_schema) in {"scifact2", "scifact3"}:
            return "scifact"
        if any(part.lower() == "rawfc" for part in path.parts):
            return "rawfc"
        if any(part.lower() == "hover" for part in path.parts):
            return "hover"
        if any(part.lower() == "scifact" for part in path.parts):
            return "scifact"
        return "liar_raw"
    raise ValueError(f"Unsupported dataset={dataset!r}. Use liar_raw, rawfc, hover, or scifact.")


def _load_liar_raw_split(path: Path, *, label_schema: str) -> list[SampleRecord]:
    label2id = label2id_for_schema(label_schema)
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[SampleRecord] = []
    for item in payload:
        label = clean_text(str(item["label"])).lower()
        if label not in label2id:
            raise ValueError(f"Unknown label: {label!r} in {path}")
        records.append(
            SampleRecord(
                event_id=str(item["event_id"]),
                claim=clean_text(str(item["claim"])),
                label=label,
                explain=clean_text(str(item.get("explain", ""))),
                reports=item.get("reports", []),
                metadata=_metadata_from_raw_row(item),
            )
        )
    return records


def _load_rawfc_split(path: Path, *, label_schema: str) -> list[SampleRecord]:
    label2id = label2id_for_schema(label_schema)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[SampleRecord] = []
    for item in payload:
        label = _rawfc_label_name(item.get("label"), path=path)
        if label not in label2id:
            raise ValueError(f"Unknown RAWFC label: {label!r} in {path}")
        evidence_items = item.get("evidence") or []
        if not isinstance(evidence_items, list):
            evidence_items = [evidence_items]
        reports: list[dict[str, Any]] = []
        for idx, evidence in enumerate(evidence_items):
            content = clean_text(str(evidence))
            if not content:
                continue
            reports.append(
                {
                    "report_id": idx,
                    "content": content,
                    "domain": None,
                    "link": None,
                    "rawfc_evidence_idx": idx,
                }
            )
        records.append(
            SampleRecord(
                event_id=str(item["id"]),
                claim=clean_text(str(item["claim"])),
                label=label,
                explain=clean_text(str(item.get("explanation", ""))),
                reports=reports,
                metadata=_metadata_from_raw_row(item),
            )
        )
    return records


def _load_hover_split(path: Path, *, label_schema: str) -> list[SampleRecord]:
    label2id = label2id_for_schema(label_schema)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[SampleRecord] = []
    for item in payload:
        label = _hover_label_name(item.get("label"), path=path)
        if label and label not in label2id:
            raise ValueError(f"Unknown HoVer label: {label!r} in {path}")
        reports = item.get("reports") or []
        records.append(
            SampleRecord(
                event_id=str(item["uid"]),
                claim=clean_text(str(item["claim"])),
                label=label,
                explain=clean_text(str(item.get("explain", ""))),
                reports=reports if isinstance(reports, list) else [],
                metadata=_hover_metadata_from_raw_row(item, has_gold_label=bool(label)),
            )
        )
    return records


def _load_scifact_split(path: Path, *, label_schema: str) -> list[SampleRecord]:
    label2id = label2id_for_schema(label_schema)
    include_nei = normalize_label_schema(label_schema) == "scifact3" and not _is_scifact_test_path(path)
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label, label_conflict = _scifact_claim_label(
                item.get("evidence") or {},
                empty_label="nei" if include_nei else "",
            )
            if label and label not in label2id:
                raise ValueError(f"Unknown SciFact label: {label!r} in {path}:{line_no}")
            records.append(
                SampleRecord(
                    event_id=str(item["id"]),
                    claim=clean_text(str(item["claim"])),
                    label=label,
                    explain="",
                    reports=[],
                    metadata=_scifact_metadata_from_raw_row(
                        item,
                        label=label,
                        label_conflict=label_conflict,
                    ),
                )
            )
    return records


def _metadata_from_raw_row(item: dict[str, Any]) -> dict[str, Any]:
    return {key: item[key] for key in COVERAGE_METADATA_KEYS if key in item}


def _rawfc_label_name(value: Any, *, path: Path) -> str:
    if value in RAWFC_NUMERIC_LABELS:
        return RAWFC_NUMERIC_LABELS[value]
    text = clean_text(str(value)).lower()
    if text in RAWFC_NUMERIC_LABELS:
        return RAWFC_NUMERIC_LABELS[text]
    if text in {"true", "false", "half"}:
        return text
    raise ValueError(f"Unknown RAWFC numeric label: {value!r} in {path}")


def _hover_label_name(value: Any, *, path: Path) -> str:
    if value is None:
        return ""
    text = clean_text(str(value)).lower().replace("-", "_")
    if text in {"", "none", "null"}:
        return ""
    if text in {"supported", "support"}:
        return "supported"
    if text in {"not_supported", "notsupported", "not support", "not supported"}:
        return "not_supported"
    raise ValueError(f"Unknown HoVer label: {value!r} in {path}")


def _hover_metadata_from_raw_row(item: dict[str, Any], *, has_gold_label: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_dataset": "hover",
        "has_gold_label": has_gold_label,
    }
    for key in ("supporting_facts", "num_hops", "hpqa_id"):
        if key in item:
            metadata[key] = item[key]
    return metadata


def _scifact_claim_label(evidence: Any, *, empty_label: str = "") -> tuple[str, bool]:
    if not isinstance(evidence, dict) or not evidence:
        return str(empty_label), False
    labels: list[str] = []
    for rationale_rows in evidence.values():
        if not isinstance(rationale_rows, list):
            continue
        for rationale in rationale_rows:
            if not isinstance(rationale, dict):
                continue
            label = _scifact_label_name(rationale.get("label"))
            if label:
                labels.append(label)
    unique = list(dict.fromkeys(labels))
    if not unique:
        return "", False
    return unique[0], len(set(unique)) > 1


def _scifact_label_name(value: Any) -> str:
    text = clean_text(str(value or "")).strip().lower().replace("-", "_")
    if text in {"", "none", "null"}:
        return ""
    if text in {"support", "supports", "supported"}:
        return "support"
    if text in {"contradict", "contradicts", "contradicted", "refute", "refutes", "refuted"}:
        return "contradict"
    raise ValueError(f"Unknown SciFact label: {value!r}")


def _scifact_metadata_from_raw_row(
    item: dict[str, Any],
    *,
    label: str,
    label_conflict: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_dataset": "scifact",
        "has_gold_label": bool(label),
        "scifact_label_conflict": bool(label_conflict),
    }
    for key in ("evidence", "cited_doc_ids"):
        if key in item:
            metadata[key] = item[key]
    return metadata


def _is_scifact_test_path(path: Path) -> bool:
    return path.name.strip().lower() in {"claims_test.jsonl", "test.jsonl"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coerce_evidence_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _iter_content_sentences(sample: SampleRecord, min_char_len: int) -> Iterable[SentenceRecord]:
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


def _iter_tokenized_sentences(sample: SampleRecord, min_char_len: int) -> Iterable[SentenceRecord]:
    for report_order, report in enumerate(sample.reports):
        report_id = report.get("report_id", "unknown")
        link = report.get("link")
        domain = report.get("domain")
        tokenized = report.get("tokenized")
        if not isinstance(tokenized, list):
            continue
        for sent_idx, item in enumerate(tokenized):
            if not isinstance(item, dict):
                continue
            text = clean_text(str(item.get("sent") or item.get("sentence") or item.get("text") or ""))
            if len(text) < min_char_len:
                continue
            yield SentenceRecord(
                event_id=sample.event_id,
                report_id=report_id,
                sent_idx=sent_idx,
                text=text,
                link=link,
                domain=domain,
                raw={
                    "raw_sentence_source": "tokenized",
                    "raw_is_evidence": _coerce_evidence_label(item.get("is_evidence", 0)),
                    "raw_evidence_label": item.get("is_evidence", 0),
                    "raw_report_order": report_order,
                    "raw_sent_order": sent_idx,
                },
            )


def iter_sentences(
    sample: SampleRecord,
    min_char_len: int = 10,
    source: str = "content",
) -> Iterable[SentenceRecord]:
    source = str(source or "content").strip().lower()
    if source in {"tokenized", "raw_tokenized", "raw"}:
        yield from _iter_tokenized_sentences(sample, min_char_len)
        return
    if source != "content":
        raise ValueError(f"Unsupported sentence source: {source!r}. Use content or tokenized.")
    yield from _iter_content_sentences(sample, min_char_len)
