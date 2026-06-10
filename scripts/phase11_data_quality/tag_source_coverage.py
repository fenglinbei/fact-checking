#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from fact_checking.data.io import iter_sentences, load_split


COVERED = "covered"
WEAK_COVERED = "weak_covered"
UNCOVERED = "uncovered"
VALID_LABELS = {COVERED, WEAK_COVERED, UNCOVERED}
DEFAULT_COVERAGE_VERSION = "source_coverage_v2"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"

DEFAULT_INPUTS_BY_DATASET = {
    "liar_raw": {
        "train": "data/raw/LIAR-RAW/train.json",
        "val": "data/raw/LIAR-RAW/val.json",
        "test": "data/raw/LIAR-RAW/test.json",
    },
    "rawfc": {
        "train": "data/raw/RAWFC/train.json",
        "val": "data/raw/RAWFC/val.json",
        "test": "data/raw/RAWFC/test.json",
    },
}

STOP_TERMS = {
    "said",
    "says",
    "claim",
    "claims",
    "show",
    "shows",
    "however",
    "indeed",
    "based",
    "available",
    "evidence",
    "statistics",
    "statement",
    "fact",
    "true",
    "false",
    "mostly",
    "barely",
    "half",
    "pants",
    "fire",
}

METRIC_TERMS = {
    "employment-population ratio",
    "employment population ratio",
    "labor force participation rate",
    "labor force participation",
    "labor participation rate",
    "participation rate",
    "unemployment rate",
    "inflation rate",
    "poverty rate",
    "tax rate",
    "approval rating",
    "gross domestic product",
    "gdp",
    "nonfarm payroll employment",
    "unemployment",
    "medicaid",
    "medicare",
    "social security",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
DECADE_RE = re.compile(r"(?:['`’]?\d0s|\b\d0s\b)", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\$?\d[\d,]*(?:\.\d+)?%?|(?:one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|thousand|million|billion))\b",
    re.IGNORECASE,
)
CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.'-]*)(?:\s+(?:[A-Z][A-Za-z0-9&.'-]*))*\b")

TEXT_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "on",
    "in",
    "for",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "not",
    "but",
    "if",
    "than",
    "then",
    "into",
    "their",
    "there",
    "about",
    "literally",
}


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in TEXT_STOPWORDS]


def lexical_overlap_f1(query: str, sentence: str) -> float:
    q_ctr = Counter(content_tokens(query))
    s_ctr = Counter(content_tokens(sentence))
    q_len = sum(q_ctr.values())
    s_len = sum(s_ctr.values())
    if q_len == 0 or s_len == 0:
        return 0.0
    overlap = sum(min(q_ctr[k], s_ctr[k]) for k in q_ctr.keys() & s_ctr.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / s_len
    recall = overlap / q_len
    return float(2.0 * precision * recall / max(1e-8, precision + recall))


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


def write_jsonl_atomic(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def load_resume_rows(*, checkpoint_path: Path, output_path: Path, split: str) -> dict[str, dict[str, Any]]:
    source_path: Path | None = None
    if checkpoint_path.exists():
        source_path = checkpoint_path
    elif output_path.exists():
        source_path = output_path
    if source_path is None:
        return {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_rows(source_path):
        if str(row.get("split") or "") != split:
            continue
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        rows_by_id[event_id] = row
    return rows_by_id


@dataclass(frozen=True)
class SentenceItem:
    report_id: str
    sent_idx: int
    text: str
    link: str | None
    domain: str | None
    raw_is_evidence: bool | None = None


@dataclass(frozen=True)
class Anchors:
    numbers: tuple[str, ...]
    years: tuple[str, ...]
    metrics: tuple[str, ...]
    entities: tuple[str, ...]
    salient_terms: tuple[str, ...]


@dataclass(frozen=True)
class LLMReviewContext:
    row_index: int
    record: Any
    anchors: Anchors
    top_items: list[tuple[SentenceItem, dict[str, float]]]
    rule: dict[str, Any]
    review_reasons: list[str]


@dataclass(frozen=True)
class LLMRunPlan:
    enabled: bool
    status: str
    policy: str
    review_count: int
    selected_model: str | None
    model_reason: str
    base_url: str
    api_key_env: str
    api_key_available: bool
    pro_max_reviews: int


@dataclass(frozen=True)
class OkapiBM25:
    tokenized_docs: tuple[tuple[str, ...], ...]
    doc_freq: dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: list[str], *, k1: float = 1.5, b: float = 0.75) -> "OkapiBM25":
        tokenized = tuple(tuple(content_tokens(doc)) for doc in docs)
        df: dict[str, int] = {}
        for toks in tokenized:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        avgdl = float(sum(len(toks) for toks in tokenized) / max(len(tokenized), 1))
        return cls(tokenized_docs=tokenized, doc_freq=df, avgdl=max(avgdl, 1.0), k1=float(k1), b=float(b))

    def scores(self, query: str) -> list[float]:
        q_terms = list(dict.fromkeys(content_tokens(query)))
        n_docs = len(self.tokenized_docs)
        if not q_terms or n_docs == 0:
            return [0.0 for _ in range(n_docs)]
        out = [0.0 for _ in range(n_docs)]
        doc_counters = [Counter(toks) for toks in self.tokenized_docs]
        for term in q_terms:
            df = self.doc_freq.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, toks in enumerate(self.tokenized_docs):
                tf = doc_counters[idx].get(term, 0)
                if tf <= 0:
                    continue
                dl = max(len(toks), 1)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                out[idx] += float(idf * (tf * (self.k1 + 1.0) / denom))
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag raw claim report sets as covered / weak_covered / uncovered.")
    p.add_argument("--input", default=None, help="Raw split JSON. Defaults to the selected dataset raw split.")
    p.add_argument("--dataset", default="liar_raw", choices=["liar_raw", "rawfc"])
    p.add_argument("--label-schema", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default=None, help="Defaults to outputs/data_quality/source_coverage/<dataset>.")
    p.add_argument("--coverage-version", default=DEFAULT_COVERAGE_VERSION)
    p.add_argument("--sentence-source", default="content", choices=["content", "tokenized"])
    p.add_argument("--sentence-min-char-len", type=int, default=10)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--event-id", action="append", default=None, help="Optional event_id filter; may be repeated.")
    p.add_argument("--covered-threshold", type=float, default=0.72)
    p.add_argument("--weak-threshold", type=float, default=0.38)
    p.add_argument("--embedding-model", default=None, help="Optional HF embedding model. Omit for BM25+lexical only.")
    p.add_argument("--embedding-device", default="cuda")
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--embedding-max-length", type=int, default=256)
    p.add_argument("--embedding-precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--env-file", default=".env", help="Project env file used to load DEEPSEEK_API_KEY if present.")
    p.add_argument("--llm-base-url", default=DEFAULT_DEEPSEEK_BASE_URL, help="OpenAI-compatible chat base URL.")
    p.add_argument("--llm-model-policy", default="auto", choices=["auto", "off"])
    p.add_argument("--llm-model", default=None, help="Manual model override; skips auto pro/flash selection.")
    p.add_argument("--llm-pro-model", default=DEFAULT_DEEPSEEK_PRO_MODEL)
    p.add_argument("--llm-flash-model", default=DEFAULT_DEEPSEEK_FLASH_MODEL)
    p.add_argument("--llm-pro-max-reviews", type=int, default=500)
    p.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--llm-timeout", type=float, default=120.0)
    p.add_argument("--llm-max-tokens", type=int, default=512)
    p.add_argument("--llm-min-confidence", type=float, default=0.65)
    p.add_argument("--llm-workers", type=int, default=1, help="Concurrent LLM review requests.")
    p.add_argument("--llm-retries", type=int, default=3)
    p.add_argument("--llm-retry-backoff", type=float, default=2.0)
    p.add_argument("--llm-retry-statuses", default="429,500,502,503,504")
    p.add_argument("--llm-thinking", default="disabled", choices=["default", "enabled", "disabled"])
    p.add_argument("--llm-reasoning-effort", default="high", choices=["high", "max"])
    p.add_argument("--llm-boundary-margin", type=float, default=0.025)
    p.add_argument("--llm-embedding-threshold", type=float, default=0.75)
    p.add_argument("--llm-critical-weak-threshold", type=float, default=0.60)
    p.add_argument("--checkpoint-dir", default=None, help="Defaults to <output-dir>/.checkpoints.")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint/final sidecar.")
    p.add_argument("--keep-llm-errors", action="store_true", help="Do not retry existing non-ok LLM judgments on resume.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def default_input_path(dataset: str, split: str) -> str:
    try:
        return DEFAULT_INPUTS_BY_DATASET[dataset][split]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset/split for default input: dataset={dataset!r} split={split!r}") from exc


def main() -> None:
    args = parse_args()
    started_at = time.time()
    env_loaded = load_env_file(Path(args.env_file)) if args.env_file else False
    input_path = Path(args.input or default_input_path(str(args.dataset), str(args.split)))
    records = load_split(input_path, dataset=str(args.dataset), label_schema=args.label_schema)
    if args.event_id:
        wanted = set(str(x) for x in args.event_id)
        records = [record for record in records if record.event_id in wanted]
    if args.sample_limit is not None:
        records = records[: int(args.sample_limit)]

    embedder = _build_embedder(args) if args.embedding_model else None
    out_dir = Path(args.output_dir or f"outputs/data_quality/source_coverage/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"source_coverage_{args.split}.jsonl"
    summary_path = out_dir / f"source_coverage_summary_{args.split}.json"
    manifest_path = out_dir / f"source_coverage_manifest_{args.split}.json"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / ".checkpoints"
    checkpoint_path = checkpoint_dir / f"source_coverage_{args.split}.jsonl"
    checkpoint_every = max(int(args.checkpoint_every), 1)

    rows: list[dict[str, Any]] = []
    review_contexts: list[LLMReviewContext] = []
    existing_rows = {} if args.no_resume else load_resume_rows(checkpoint_path=checkpoint_path, output_path=output_path, split=str(args.split))
    restored_rows = 0
    newly_tagged_rows = 0
    iterator = tqdm(records, desc=f"source coverage [{args.split}]", unit="claim", disable=bool(args.no_progress))
    for record in iterator:
        row_index = len(rows)
        row = existing_rows.get(str(record.event_id))
        if row is not None:
            row = dict(row)
            restored_rows += 1
            review_context = resume_review_context(
                row=row,
                row_index=row_index,
                record=record,
                args=args,
            )
        else:
            row, review_context = tag_record(record, row_index=row_index, args=args, embedder=embedder)
            newly_tagged_rows += 1
        rows.append(row)
        if review_context is not None:
            review_contexts.append(review_context)
        if newly_tagged_rows and newly_tagged_rows % checkpoint_every == 0:
            write_jsonl_atomic(rows, checkpoint_path)

    llm_plan = resolve_llm_run_plan(args, review_count=len(review_contexts))
    write_jsonl_atomic(rows, checkpoint_path)
    llm_stats = apply_llm_reviews(
        rows=rows,
        review_contexts=review_contexts,
        args=args,
        plan=llm_plan,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_jsonl_atomic(rows, checkpoint_path)

    summary = build_summary(rows)
    summary.update(
        {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
            "input": str(input_path),
            "dataset": str(args.dataset),
            "label_schema": args.label_schema,
            "split": str(args.split),
            "coverage_version": str(args.coverage_version),
            "sentence_source": str(args.sentence_source),
            "sentence_min_char_len": int(args.sentence_min_char_len),
            "top_k": int(args.top_k),
            "sample_limit": args.sample_limit,
            "event_id_filter": list(args.event_id or []),
            "resume": {
                "enabled": not bool(args.no_resume),
                "source_rows": len(existing_rows),
                "restored_rows": restored_rows,
                "newly_tagged_rows": newly_tagged_rows,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_every": checkpoint_every,
                "keep_llm_errors": bool(args.keep_llm_errors),
            },
            "embedding_model": str(args.embedding_model or ""),
            "embedding_device": str(args.embedding_device),
            "embedding_precision": str(args.embedding_precision),
            "env": {
                "env_file": str(args.env_file or ""),
                "loaded": bool(env_loaded),
                "api_key_env": str(args.llm_api_key_env),
                "api_key_available": bool(llm_plan.api_key_available),
            },
            "llm": llm_plan_to_manifest(llm_plan, llm_stats),
            "outputs": {
                "annotations": str(output_path),
                "summary": str(summary_path),
                "manifest": str(manifest_path),
            },
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
    )
    save_json(summary, summary_path)
    manifest = {
        "coverage_version": str(args.coverage_version),
        "created_at": summary["created_at"],
        "dataset": str(args.dataset),
        "split": str(args.split),
        "input": str(input_path),
        "outputs": summary["outputs"],
        "thresholds": {
            "covered_threshold": float(args.covered_threshold),
            "weak_threshold": float(args.weak_threshold),
            "llm_min_confidence": float(args.llm_min_confidence),
            "llm_workers": max(int(args.llm_workers), 1),
            "llm_retries": max(int(args.llm_retries), 0),
            "llm_retry_backoff": float(args.llm_retry_backoff),
            "llm_retry_statuses": parse_retry_statuses(str(args.llm_retry_statuses)),
            "llm_boundary_margin": float(args.llm_boundary_margin),
            "llm_embedding_threshold": float(args.llm_embedding_threshold),
            "llm_critical_weak_threshold": float(args.llm_critical_weak_threshold),
            "llm_thinking": str(args.llm_thinking),
            "llm_reasoning_effort": str(args.llm_reasoning_effort),
        },
        "embedding": {
            "enabled": bool(embedder is not None),
            "model": str(args.embedding_model or ""),
            "device": str(args.embedding_device),
            "batch_size": int(args.embedding_batch_size),
            "max_length": int(args.embedding_max_length),
            "precision": str(args.embedding_precision),
        },
        "llm": summary["llm"],
        "resume": summary["resume"],
        "command": summary["command"],
        "counts": {
            "n_rows": summary["n_rows"],
            "coverage_counts": summary["coverage_counts"],
            "rule_coverage_counts": summary["rule_coverage_counts"],
            "coverage_by_gold_label": summary["coverage_by_gold_label"],
        },
    }
    save_json(manifest, manifest_path)
    print(f"Wrote source coverage labels: {output_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps(summary.get("coverage_counts", {}), ensure_ascii=False, sort_keys=True))


def tag_record(
    record: Any,
    *,
    row_index: int,
    args: argparse.Namespace,
    embedder: Any | None,
) -> tuple[dict[str, Any], LLMReviewContext | None]:
    sentences = sentence_items(record, source=str(args.sentence_source), min_char_len=int(args.sentence_min_char_len))
    anchors = extract_anchors(str(record.claim), str(record.explain))
    ranked = rank_sentences(
        query=f"{record.claim}\n{record.explain}",
        sentences=sentences,
        embedder=embedder,
    )
    top_items = ranked[: max(int(args.top_k), 1)]
    all_text = "\n".join(item.text for item in sentences)
    top_text = "\n".join(item.text for item, _scores in top_items)
    all_coverage = anchor_coverage(anchors, all_text)
    top_coverage = anchor_coverage(anchors, top_text)
    rule = rule_decision(
        anchors=anchors,
        all_coverage=all_coverage,
        top_coverage=top_coverage,
        ranked=ranked,
        covered_threshold=float(args.covered_threshold),
        weak_threshold=float(args.weak_threshold),
    )
    review_reasons = llm_review_reasons(
        rule=rule,
        ranked=ranked,
        args=args,
    )
    final_label = str(rule["coverage_label"])
    llm_judgment = {
        "status": "pending" if review_reasons else "not_requested",
        "review_needed": bool(review_reasons),
        "review_reasons": review_reasons,
    }
    row = {
        "event_id": str(record.event_id),
        "split": str(args.split),
        "claim": str(record.claim),
        "gold_label": str(record.label),
        "coverage_label": final_label,
        "rule_coverage_label": rule["coverage_label"],
        "decision_source": "llm" if final_label != rule["coverage_label"] else "rule",
        "coverage_score": rule["coverage_score"],
        "weak_score": rule["weak_score"],
        "critical_missing": rule["critical_missing"],
        "rule": rule,
        "n_reports": len(record.reports or []),
        "n_sentences": len(sentences),
        "anchors": anchors_to_dict(anchors),
        "all_report_coverage": all_coverage,
        "top_evidence_coverage": top_coverage,
        "retrieval": {
            "top_k": int(args.top_k),
            "best_bm25": ranked[0][1]["bm25"] if ranked else 0.0,
            "best_lexical": ranked[0][1]["lexical"] if ranked else 0.0,
            "best_embedding": ranked[0][1].get("embedding", 0.0) if ranked else 0.0,
            "best_hybrid": ranked[0][1]["hybrid"] if ranked else 0.0,
            "embedding_enabled": bool(embedder is not None),
        },
        "top_evidence": [
            {
                "rank": idx + 1,
                "report_id": item.report_id,
                "sent_idx": item.sent_idx,
                "text": item.text,
                "link": item.link,
                "domain": item.domain,
                "raw_is_evidence": item.raw_is_evidence,
                "scores": scores,
                "anchor_hits": anchor_hits(anchors, item.text),
            }
            for idx, (item, scores) in enumerate(top_items)
        ],
        "llm_judgment": llm_judgment,
    }
    if not review_reasons:
        return row, None
    return row, LLMReviewContext(
        row_index=row_index,
        record=record,
        anchors=anchors,
        top_items=top_items,
        rule=rule,
        review_reasons=review_reasons,
    )


def resume_review_context(
    *,
    row: dict[str, Any],
    row_index: int,
    record: Any,
    args: argparse.Namespace,
) -> LLMReviewContext | None:
    judgment = row.get("llm_judgment") if isinstance(row.get("llm_judgment"), dict) else {}
    if not judgment.get("review_needed"):
        return None
    status = str(judgment.get("status") or "")
    if status == "ok":
        return None
    if args.keep_llm_errors and status not in {"", "pending"}:
        return None
    review_reasons = judgment.get("review_reasons")
    if not isinstance(review_reasons, list):
        review_reasons = []
    anchors = anchors_from_row(row)
    top_items = top_items_from_row(row)
    rule = rule_from_row(row)
    return LLMReviewContext(
        row_index=row_index,
        record=record,
        anchors=anchors,
        top_items=top_items,
        rule=rule,
        review_reasons=[str(reason) for reason in review_reasons],
    )


def anchors_from_row(row: dict[str, Any]) -> Anchors:
    data = row.get("anchors") if isinstance(row.get("anchors"), dict) else {}
    return Anchors(
        numbers=tuple(str(x) for x in data.get("numbers", []) if x is not None),
        years=tuple(str(x) for x in data.get("years", []) if x is not None),
        metrics=tuple(str(x) for x in data.get("metrics", []) if x is not None),
        entities=tuple(str(x) for x in data.get("entities", []) if x is not None),
        salient_terms=tuple(str(x) for x in data.get("salient_terms", []) if x is not None),
    )


def top_items_from_row(row: dict[str, Any]) -> list[tuple[SentenceItem, dict[str, float]]]:
    out: list[tuple[SentenceItem, dict[str, float]]] = []
    values = row.get("top_evidence") if isinstance(row.get("top_evidence"), list) else []
    for value in values:
        if not isinstance(value, dict):
            continue
        scores_raw = value.get("scores") if isinstance(value.get("scores"), dict) else {}
        scores = {str(key): safe_float(val, 0.0) for key, val in scores_raw.items()}
        out.append(
            (
                SentenceItem(
                    report_id=str(value.get("report_id") or ""),
                    sent_idx=int(safe_float(value.get("sent_idx"), 0.0)),
                    text=str(value.get("text") or ""),
                    link=value.get("link") if value.get("link") is None else str(value.get("link")),
                    domain=value.get("domain") if value.get("domain") is None else str(value.get("domain")),
                    raw_is_evidence=value.get("raw_is_evidence")
                    if isinstance(value.get("raw_is_evidence"), bool) or value.get("raw_is_evidence") is None
                    else bool(value.get("raw_is_evidence")),
                ),
                scores,
            )
        )
    return out


def rule_from_row(row: dict[str, Any]) -> dict[str, Any]:
    rule = row.get("rule") if isinstance(row.get("rule"), dict) else {}
    out = dict(rule)
    out.setdefault("coverage_label", row.get("rule_coverage_label") or row.get("coverage_label") or UNCOVERED)
    out.setdefault("coverage_score", row.get("coverage_score") or 0.0)
    out.setdefault("weak_score", row.get("weak_score") or 0.0)
    out.setdefault("critical_missing", row.get("critical_missing") or [])
    return out


def sentence_items(record: Any, *, source: str, min_char_len: int) -> list[SentenceItem]:
    items: list[SentenceItem] = []
    for sent in iter_sentences(record, min_char_len=int(min_char_len), source=source):
        raw = sent.raw if isinstance(sent.raw, dict) else {}
        raw_is_evidence = raw.get("raw_is_evidence")
        items.append(
            SentenceItem(
                report_id=str(sent.report_id),
                sent_idx=int(sent.sent_idx),
                text=str(sent.text),
                link=sent.link,
                domain=sent.domain,
                raw_is_evidence=bool(raw_is_evidence) if raw_is_evidence is not None else None,
            )
        )
    return items


def extract_anchors(claim: str, explain: str) -> Anchors:
    text = f"{claim}\n{explain}"
    claim_numbers = set(normalize_number(x) for x in NUMBER_RE.findall(claim))
    numbers = sorted(
        {
            normalize_number(x)
            for x in NUMBER_RE.findall(text)
            if normalize_number(x)
        }
    )
    years = sorted({x for x in YEAR_RE.findall(text)})
    decade_numbers = sorted({normalize_decade(x) for x in DECADE_RE.findall(text) if normalize_decade(x)})
    for decade in decade_numbers:
        if decade not in numbers:
            numbers.append(decade)
    metrics = extract_metric_phrases(text)
    entities = extract_entities(text)
    salient_terms = extract_salient_terms(claim, explain)
    # Keep explicit claim numbers even if the explanation omits them.
    for number in sorted(claim_numbers):
        if number and number not in numbers:
            numbers.append(number)
    return Anchors(
        numbers=tuple(sorted(set(numbers))),
        years=tuple(sorted(set(years))),
        metrics=tuple(metrics),
        entities=tuple(entities),
        salient_terms=tuple(salient_terms),
    )


def extract_metric_phrases(text: str) -> list[str]:
    lowered = normalize_text(text)
    metrics: set[str] = {term for term in METRIC_TERMS if term in lowered}
    pattern = re.compile(
        r"\b[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,4}\s+"
        r"(?:rate|ratio|share|percentage|count|number|total)\b"
    )
    for match in pattern.finditer(lowered):
        phrase = normalize_metric_phrase(match.group(0))
        toks = phrase.split()
        if 2 <= len(toks) <= 6 and not all(tok in STOP_TERMS for tok in toks):
            metrics.add(phrase)
    return sorted(metrics)


def extract_entities(text: str) -> list[str]:
    entities: set[str] = set()
    for match in CAPITALIZED_RE.finditer(text):
        value = clean_phrase(match.group(0))
        if not value or value in {"i", "the", "a"}:
            continue
        toks = value.split()
        if len(toks) == 1 and len(toks[0]) <= 2:
            continue
        if value in {"we", "however"}:
            continue
        entities.add(value)
    return sorted(entities)


def extract_salient_terms(claim: str, explain: str, *, limit: int = 32) -> list[str]:
    claim_tokens = set(content_tokens(claim))
    terms: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for order, tok in enumerate(content_tokens(explain)):
        if tok in seen or tok in STOP_TERMS:
            continue
        if len(tok) < 4 and not any(ch.isdigit() for ch in tok):
            continue
        # Keep claim-overlap terms, but let explanation-only terms dominate.
        priority = 1 if tok not in claim_tokens else 0
        terms.append((priority, order, tok))
        seen.add(tok)
    terms_sorted = [tok for _priority, _order, tok in sorted(terms, key=lambda item: (-item[0], item[1]))]
    return terms_sorted[:limit]


def rank_sentences(
    *,
    query: str,
    sentences: list[SentenceItem],
    embedder: Any | None,
) -> list[tuple[SentenceItem, dict[str, float]]]:
    if not sentences:
        return []
    docs = [item.text for item in sentences]
    bm25 = OkapiBM25.build(docs).scores(query)
    lexical = [lexical_overlap_f1(query, doc) for doc in docs]
    embedding = [0.0 for _ in docs]
    if embedder is not None:
        query_vec = embedder.encode([query], is_query=True)
        doc_vec = embedder.encode(docs, is_query=False)
        if query_vec.shape[0] == 1 and doc_vec.shape[0] == len(docs):
            embedding = [float(x) for x in (doc_vec @ query_vec[0]).tolist()]
    bm25_scaled = minmax(bm25)
    lexical_scaled = minmax(lexical)
    embedding_scaled = minmax(embedding)
    if embedder is not None:
        hybrid = [
            0.45 * bm25_scaled[idx] + 0.20 * lexical_scaled[idx] + 0.35 * embedding_scaled[idx]
            for idx in range(len(docs))
        ]
    else:
        hybrid = [
            0.70 * bm25_scaled[idx] + 0.30 * lexical_scaled[idx]
            for idx in range(len(docs))
        ]
    rows: list[tuple[SentenceItem, dict[str, float]]] = []
    for idx, item in enumerate(sentences):
        rows.append(
            (
                item,
                {
                    "bm25": round(float(bm25[idx]), 6),
                    "bm25_scaled": round(float(bm25_scaled[idx]), 6),
                    "lexical": round(float(lexical[idx]), 6),
                    "lexical_scaled": round(float(lexical_scaled[idx]), 6),
                    "embedding": round(float(embedding[idx]), 6),
                    "embedding_scaled": round(float(embedding_scaled[idx]), 6),
                    "hybrid": round(float(hybrid[idx]), 6),
                },
            )
        )
    rows.sort(key=lambda pair: (-pair[1]["hybrid"], -pair[1]["bm25"], pair[0].report_id, pair[0].sent_idx))
    return rows


def anchor_coverage(anchors: Anchors, text: str) -> dict[str, Any]:
    return {
        "numbers": coverage_for_values(anchors.numbers, text, kind="number"),
        "years": coverage_for_values(anchors.years, text, kind="number"),
        "metrics": coverage_for_values(anchors.metrics, text, kind="phrase"),
        "entities": coverage_for_values(anchors.entities, text, kind="phrase"),
        "salient_terms": coverage_for_values(anchors.salient_terms, text, kind="token"),
    }


def coverage_for_values(values: tuple[str, ...], text: str, *, kind: str) -> dict[str, Any]:
    if not values:
        return {"total": 0, "covered": 0, "ratio": 1.0, "missing": [], "covered_values": []}
    covered: list[str] = []
    missing: list[str] = []
    for value in values:
        if value_covered(value, text, kind=kind):
            covered.append(value)
        else:
            missing.append(value)
    return {
        "total": len(values),
        "covered": len(covered),
        "ratio": round(float(len(covered) / max(len(values), 1)), 6),
        "missing": missing,
        "covered_values": covered,
    }


def value_covered(value: str, text: str, *, kind: str) -> bool:
    if not value:
        return False
    lowered = normalize_text(text)
    if kind == "number":
        return normalize_number(value) in number_forms(lowered)
    if kind == "token":
        return value.lower() in set(tokenize(lowered))
    value_norm = normalize_text(value)
    if value_norm in lowered:
        return True
    value_tokens = content_tokens(value_norm)
    if not value_tokens:
        return False
    text_tokens = set(content_tokens(lowered))
    overlap = sum(1 for tok in value_tokens if tok in text_tokens)
    return overlap / max(len(value_tokens), 1) >= 0.75


def anchor_hits(anchors: Anchors, text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for key, values, kind in [
        ("numbers", anchors.numbers, "number"),
        ("years", anchors.years, "number"),
        ("metrics", anchors.metrics, "phrase"),
        ("entities", anchors.entities, "phrase"),
        ("salient_terms", anchors.salient_terms, "token"),
    ]:
        matched = [value for value in values if value_covered(value, text, kind=kind)]
        if matched:
            hits[key] = matched
    return hits


def rule_decision(
    *,
    anchors: Anchors,
    all_coverage: dict[str, Any],
    top_coverage: dict[str, Any],
    ranked: list[tuple[SentenceItem, dict[str, float]]],
    covered_threshold: float,
    weak_threshold: float,
) -> dict[str, Any]:
    all_score = weighted_anchor_score(all_coverage, anchors=anchors)
    top_score = weighted_anchor_score(top_coverage, anchors=anchors)
    best = ranked[0][1] if ranked else {}
    retrieval_signal = max(float(best.get("hybrid", 0.0)), 0.0)
    coverage_score = 0.72 * all_score + 0.20 * top_score + 0.08 * retrieval_signal
    weak_score = 0.55 * all_score + 0.30 * top_score + 0.15 * retrieval_signal
    critical_missing = critical_missing_values(anchors, all_coverage)
    numbers_ok = bool(not anchors.years and not anchors.numbers) or float(all_coverage["years"]["ratio"]) >= 0.75
    if anchors.years and len(anchors.years) <= 2:
        numbers_ok = float(all_coverage["years"]["ratio"]) >= 1.0
    if anchors.numbers and not anchors.years and len(anchors.numbers) <= 2:
        numbers_ok = float(all_coverage["numbers"]["ratio"]) >= 1.0
    metrics_ok = not anchors.metrics or float(all_coverage["metrics"]["ratio"]) >= 0.75
    if coverage_score >= covered_threshold and numbers_ok and metrics_ok and not critical_missing:
        label = COVERED
    elif hard_critical_missing(anchors, all_coverage) and weak_score < 0.70:
        label = UNCOVERED
    elif weak_score >= weak_threshold or partial_anchor_hit(all_coverage):
        label = WEAK_COVERED
    else:
        label = UNCOVERED
    return {
        "coverage_label": label,
        "coverage_score": round(float(coverage_score), 6),
        "weak_score": round(float(weak_score), 6),
        "all_anchor_score": round(float(all_score), 6),
        "top_anchor_score": round(float(top_score), 6),
        "retrieval_signal": round(float(retrieval_signal), 6),
        "numbers_ok": bool(numbers_ok),
        "metrics_ok": bool(metrics_ok),
        "critical_missing": critical_missing,
    }


def weighted_anchor_score(coverage: dict[str, Any], *, anchors: Anchors) -> float:
    pieces: list[tuple[float, float]] = []
    if anchors.metrics:
        pieces.append((0.35, float(coverage["metrics"]["ratio"])))
    if anchors.years or anchors.numbers:
        year_ratio = float(coverage["years"]["ratio"]) if anchors.years else 1.0
        number_ratio = float(coverage["numbers"]["ratio"]) if anchors.numbers else 1.0
        pieces.append((0.35, min(year_ratio, number_ratio)))
    if anchors.salient_terms:
        pieces.append((0.20, float(coverage["salient_terms"]["ratio"])))
    if anchors.entities:
        pieces.append((0.10, float(coverage["entities"]["ratio"])))
    if not pieces:
        return 0.0
    total_weight = sum(weight for weight, _score in pieces)
    return float(sum(weight * score for weight, score in pieces) / max(total_weight, 1e-8))


def critical_missing_values(anchors: Anchors, coverage: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if anchors.metrics and float(coverage["metrics"]["ratio"]) < 0.75:
        missing.extend(f"metric:{value}" for value in coverage["metrics"]["missing"])
    if anchors.years and float(coverage["years"]["ratio"]) < 0.75:
        missing.extend(f"year:{value}" for value in coverage["years"]["missing"])
    if anchors.numbers and not anchors.years and float(coverage["numbers"]["ratio"]) < 0.75:
        missing.extend(f"number:{value}" for value in coverage["numbers"]["missing"])
    return missing[:20]


def hard_critical_missing(anchors: Anchors, coverage: dict[str, Any]) -> bool:
    if anchors.years and float(coverage["years"]["ratio"]) < 0.75:
        return True
    if anchors.metrics and float(coverage["metrics"]["ratio"]) < 0.75:
        return True
    if anchors.numbers and not anchors.years and float(coverage["numbers"]["ratio"]) < 0.75:
        return True
    return False


def partial_anchor_hit(coverage: dict[str, Any]) -> bool:
    return any(
        float(coverage[key]["ratio"]) > 0.0 and int(coverage[key]["covered"]) > 0
        for key in ("metrics", "years", "numbers", "salient_terms", "entities")
        if coverage.get(key)
    )


def llm_review_reasons(
    *,
    rule: dict[str, Any],
    ranked: list[tuple[SentenceItem, dict[str, float]]],
    args: argparse.Namespace,
) -> list[str]:
    label = str(rule.get("coverage_label") or UNCOVERED)
    coverage_score = safe_float(rule.get("coverage_score"), 0.0)
    weak_score = safe_float(rule.get("weak_score"), 0.0)
    margin = max(safe_float(args.llm_boundary_margin, 0.08), 0.0)
    reasons: list[str] = []
    if label == WEAK_COVERED:
        reasons.append("rule_label_weak_covered")
    if abs(coverage_score - float(args.covered_threshold)) <= margin:
        reasons.append("near_covered_threshold")
    if abs(weak_score - float(args.weak_threshold)) <= margin:
        reasons.append("near_weak_threshold")
    if rule.get("critical_missing") and weak_score >= float(args.llm_critical_weak_threshold):
        reasons.append("critical_missing_but_weak_signal")
    if ranked:
        best_scores = ranked[0][1]
        embedding_score = safe_float(best_scores.get("embedding"), 0.0)
        embedding_enabled = embedding_score != 0.0 or safe_float(best_scores.get("embedding_scaled"), 0.0) != 0.0
        embedding_threshold = float(args.llm_embedding_threshold)
        uncovered_boundary = bool(reasons)
        covered_boundary = bool(reasons)
        if embedding_enabled and label == UNCOVERED and uncovered_boundary and embedding_score >= embedding_threshold:
            reasons.append("embedding_rule_disagreement")
        if (
            embedding_enabled
            and label == COVERED
            and covered_boundary
            and embedding_score < max(embedding_threshold * 0.60, 0.0)
        ):
            reasons.append("low_embedding_for_covered_rule")
    return list(dict.fromkeys(reasons))


def load_env_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    loaded = False
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            if key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
            loaded = True
    return loaded


def resolve_llm_run_plan(args: argparse.Namespace, *, review_count: int) -> LLMRunPlan:
    policy = str(args.llm_model_policy or "auto").strip().lower()
    base_url = str(args.llm_base_url or "").strip()
    api_key_env = str(args.llm_api_key_env or "DEEPSEEK_API_KEY").strip()
    api_key_available = bool(api_key_env and os.environ.get(api_key_env))
    manual_model = str(args.llm_model or "").strip()
    pro_max_reviews = max(int(args.llm_pro_max_reviews), 0)
    if review_count <= 0:
        return LLMRunPlan(
            enabled=False,
            status="skipped_no_review_candidates",
            policy=policy,
            review_count=review_count,
            selected_model=None,
            model_reason="no samples required LLM review",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key_available=api_key_available,
            pro_max_reviews=pro_max_reviews,
        )
    if policy == "off":
        return LLMRunPlan(
            enabled=False,
            status="skipped_policy_off",
            policy=policy,
            review_count=review_count,
            selected_model=None,
            model_reason="llm model policy is off",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key_available=api_key_available,
            pro_max_reviews=pro_max_reviews,
        )
    if not base_url:
        return LLMRunPlan(
            enabled=False,
            status="skipped_missing_base_url",
            policy=policy,
            review_count=review_count,
            selected_model=None,
            model_reason="missing llm base url",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key_available=api_key_available,
            pro_max_reviews=pro_max_reviews,
        )
    if not api_key_available:
        return LLMRunPlan(
            enabled=False,
            status="skipped_missing_api_key",
            policy=policy,
            review_count=review_count,
            selected_model=None,
            model_reason=f"missing API key in environment variable {api_key_env}",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key_available=False,
            pro_max_reviews=pro_max_reviews,
        )
    if manual_model:
        selected_model = manual_model
        reason = "manual --llm-model override"
    elif review_count <= pro_max_reviews:
        selected_model = str(args.llm_pro_model)
        reason = f"review_count <= {pro_max_reviews}; selected pro model"
    else:
        selected_model = str(args.llm_flash_model)
        reason = f"review_count > {pro_max_reviews}; selected flash model"
    return LLMRunPlan(
        enabled=True,
        status="enabled",
        policy=policy,
        review_count=review_count,
        selected_model=selected_model,
        model_reason=reason,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key_available=True,
        pro_max_reviews=pro_max_reviews,
    )


def apply_llm_reviews(
    *,
    rows: list[dict[str, Any]],
    review_contexts: list[LLMReviewContext],
    args: argparse.Namespace,
    plan: LLMRunPlan,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "review_candidates": len(review_contexts),
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "applied_overrides": 0,
    }
    if not plan.enabled:
        for context in review_contexts:
            row = rows[context.row_index]
            row["llm_judgment"] = {
                "status": plan.status,
                "review_needed": True,
                "review_reasons": context.review_reasons,
                "model": plan.selected_model,
                "model_policy": plan.policy,
                "model_reason": plan.model_reason,
            }
        stats["skipped"] = len(review_contexts)
        return stats

    api_key = os.environ.get(plan.api_key_env, "")
    workers = max(int(args.llm_workers), 1)

    def run_context(context: LLMReviewContext) -> tuple[LLMReviewContext, dict[str, Any], str, bool]:
        llm = maybe_llm_judge(
            record=context.record,
            anchors=context.anchors,
            top_items=context.top_items,
            rule=context.rule,
            args=args,
            model=str(plan.selected_model),
            api_key=api_key,
        )
        rule_label = str(context.rule.get("coverage_label") or UNCOVERED)
        final_label = combine_rule_and_llm(context.rule, llm, min_confidence=float(args.llm_min_confidence))
        applied = final_label != rule_label
        return context, llm, final_label, applied

    progress = tqdm(
        total=len(review_contexts),
        desc=f"LLM review [{plan.selected_model}, workers={workers}]",
        unit="claim",
        disable=bool(args.no_progress),
    )

    def apply_result(context: LLMReviewContext, llm: dict[str, Any], final_label: str, applied: bool) -> None:
        row = rows[context.row_index]
        stats["attempted"] += 1
        if llm.get("status") == "ok":
            stats["succeeded"] += 1
        else:
            stats["failed"] += 1
        if applied:
            stats["applied_overrides"] += 1
        row["coverage_label"] = final_label
        row["decision_source"] = "llm" if applied else "rule"
        row["llm_judgment"] = {
            **llm,
            "review_needed": True,
            "review_reasons": context.review_reasons,
            "model": plan.selected_model,
            "model_policy": plan.policy,
            "model_reason": plan.model_reason,
            "applied": bool(applied),
            "min_confidence": float(args.llm_min_confidence),
        }
        if checkpoint_path is not None and stats["attempted"] % max(int(checkpoint_every), 1) == 0:
            write_jsonl_atomic(rows, checkpoint_path)

    try:
        if workers == 1:
            for context in review_contexts:
                try:
                    apply_result(*run_context(context))
                except Exception as exc:
                    llm = {
                        "status": "error",
                        "error": f"unexpected_worker_exception:{type(exc).__name__}: {exc}",
                        "coverage_label": None,
                    }
                    final_label = str(context.rule.get("coverage_label") or UNCOVERED)
                    apply_result(context, llm, final_label, False)
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_context, context): context for context in review_contexts}
                for future in as_completed(futures):
                    context = futures[future]
                    try:
                        apply_result(*future.result())
                    except Exception as exc:
                        llm = {
                            "status": "error",
                            "error": f"unexpected_worker_exception:{type(exc).__name__}: {exc}",
                            "coverage_label": None,
                        }
                        final_label = str(context.rule.get("coverage_label") or UNCOVERED)
                        apply_result(context, llm, final_label, False)
                    progress.update(1)
    finally:
        progress.close()
    if checkpoint_path is not None:
        write_jsonl_atomic(rows, checkpoint_path)
    return stats


def llm_plan_to_manifest(plan: LLMRunPlan, stats: dict[str, int]) -> dict[str, Any]:
    return {
        "enabled": bool(plan.enabled),
        "status": plan.status,
        "model_policy": plan.policy,
        "review_count": int(plan.review_count),
        "selected_model": plan.selected_model,
        "model_reason": plan.model_reason,
        "base_url": plan.base_url,
        "api_key_env": plan.api_key_env,
        "api_key_available": bool(plan.api_key_available),
        "api_key_source": "environment" if plan.api_key_available else "missing",
        "pro_max_reviews": int(plan.pro_max_reviews),
        "stats": stats,
    }


def maybe_llm_judge(
    *,
    record: Any,
    anchors: Anchors,
    top_items: list[tuple[SentenceItem, dict[str, float]]],
    rule: dict[str, Any],
    args: argparse.Namespace,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    prompt = build_llm_prompt(record=record, anchors=anchors, top_items=top_items, rule=rule)
    payload = {
        "model": str(model),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict data-quality judge for fact-checking datasets. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": int(args.llm_max_tokens),
        "response_format": {"type": "json_object"},
    }
    thinking = str(args.llm_thinking or "disabled")
    if thinking != "default":
        payload["thinking"] = {"type": thinking}
        if thinking == "enabled":
            payload["thinking"]["reasoning_effort"] = str(args.llm_reasoning_effort)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{str(args.llm_base_url).rstrip('/')}/chat/completions"
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    retry_statuses = set(parse_retry_statuses(str(args.llm_retry_statuses)))
    max_retries = max(int(args.llm_retries), 0)
    raw: dict[str, Any] | None = None
    last_error: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=float(args.llm_timeout)) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            last_error = {
                "status": "error",
                "error": f"HTTP Error {exc.code}: {exc.reason}",
                "http_status": int(exc.code),
                "error_body": error_body[:1000],
                "attempts": attempt + 1,
                "retryable": int(exc.code) in retry_statuses,
                "coverage_label": None,
            }
            if int(exc.code) not in retry_statuses or attempt >= max_retries:
                return last_error
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            ConnectionError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_error = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempt + 1,
                "retryable": True,
                "coverage_label": None,
            }
            if attempt >= max_retries:
                return last_error
        sleep_before_retry(attempt=attempt, args=args)
    if raw is None:
        return last_error or {"status": "error", "error": "unknown llm request failure", "coverage_label": None}
    content = str(((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    parsed = parse_json_object(content)
    if not parsed:
        return {"status": "parse_error", "raw_content": content[:1000], "coverage_label": None}
    label = str(parsed.get("coverage_label") or "").strip()
    if label not in VALID_LABELS:
        return {"status": "invalid_label", "raw_content": content[:1000], "coverage_label": None}
    return {
        "status": "ok",
        "coverage_label": label,
        "confidence": safe_float(parsed.get("confidence"), 0.0),
        "rationale": str(parsed.get("rationale") or ""),
        "missing_evidence": parsed.get("missing_evidence") if isinstance(parsed.get("missing_evidence"), list) else [],
        "raw_content": content[:1000],
    }


def build_llm_prompt(
    *,
    record: Any,
    anchors: Anchors,
    top_items: list[tuple[SentenceItem, dict[str, float]]],
    rule: dict[str, Any],
) -> str:
    evidence_lines = []
    for idx, (item, scores) in enumerate(top_items, start=1):
        evidence_lines.append(
            f"[{idx}] report_id={item.report_id} sent_idx={item.sent_idx} "
            f"bm25={scores['bm25']:.3f} hybrid={scores['hybrid']:.3f}\n{item.text}"
        )
    return (
        "Task: decide whether the provided report evidence set contains enough information "
        "to justify the explanation for the claim.\n\n"
        "Labels:\n"
        "- covered: the evidence set contains the key facts needed by the explanation.\n"
        "- weak_covered: the evidence set is topically related or partially covers key facts, but misses important details.\n"
        "- uncovered: the evidence set lacks the key facts needed by the explanation.\n\n"
        "Return JSON with keys: coverage_label, confidence, rationale, missing_evidence.\n\n"
        f"Claim:\n{record.claim}\n\n"
        f"Gold label:\n{record.label}\n\n"
        f"Explanation:\n{record.explain}\n\n"
        f"Extracted anchors:\n{json.dumps(anchors_to_dict(anchors), ensure_ascii=False)}\n\n"
        f"Rule precheck:\n{json.dumps(rule, ensure_ascii=False)}\n\n"
        "Top report evidence:\n"
        + "\n\n".join(evidence_lines)
    )


def combine_rule_and_llm(rule: dict[str, Any], llm: dict[str, Any] | None, *, min_confidence: float) -> str:
    rule_label = str(rule.get("coverage_label") or UNCOVERED)
    if not llm or llm.get("status") != "ok":
        return rule_label
    label = str(llm.get("coverage_label") or "")
    confidence = safe_float(llm.get("confidence"), 0.0)
    if label in VALID_LABELS and confidence >= min_confidence:
        return label
    return rule_label


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {COVERED: 0, WEAK_COVERED: 0, UNCOVERED: 0}
    rule_counts: dict[str, int] = {COVERED: 0, WEAK_COVERED: 0, UNCOVERED: 0}
    by_gold: dict[str, dict[str, int]] = {}
    decision_sources: dict[str, int] = {}
    llm_status_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("coverage_label") or UNCOVERED)
        rule_label = str(row.get("rule_coverage_label") or UNCOVERED)
        counts[label] = counts.get(label, 0) + 1
        rule_counts[rule_label] = rule_counts.get(rule_label, 0) + 1
        gold = str(row.get("gold_label") or "")
        by_gold.setdefault(gold, {COVERED: 0, WEAK_COVERED: 0, UNCOVERED: 0})
        by_gold[gold][label] = by_gold[gold].get(label, 0) + 1
        source = str(row.get("decision_source") or "unknown")
        decision_sources[source] = decision_sources.get(source, 0) + 1
        llm = row.get("llm_judgment") if isinstance(row.get("llm_judgment"), dict) else {}
        llm_status = str(llm.get("status") or "missing")
        llm_status_counts[llm_status] = llm_status_counts.get(llm_status, 0) + 1
    return {
        "n_rows": len(rows),
        "coverage_counts": counts,
        "rule_coverage_counts": rule_counts,
        "coverage_by_gold_label": by_gold,
        "decision_source_counts": decision_sources,
        "llm_status_counts": llm_status_counts,
    }


def anchors_to_dict(anchors: Anchors) -> dict[str, list[str]]:
    return {
        "numbers": list(anchors.numbers),
        "years": list(anchors.years),
        "metrics": list(anchors.metrics),
        "entities": list(anchors.entities),
        "salient_terms": list(anchors.salient_terms),
    }


def _build_embedder(args: argparse.Namespace) -> TextEmbedder:
    from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder

    return TextEmbedder(
        EmbedderConfig(
            model_name=str(args.embedding_model),
            device=str(args.embedding_device),
            batch_size=int(args.embedding_batch_size),
            max_length=int(args.embedding_max_length),
            precision=str(args.embedding_precision),
        )
    )


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return [0.0 for _ in values]
    vmin = min(finite)
    vmax = max(finite)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
        return [0.0 for _ in values]
    return [float((float(value) - vmin) / (vmax - vmin)) for value in values]


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("’", "'").replace("`", "'").split())


def clean_phrase(text: str) -> str:
    return " ".join(TOKEN_RE.findall(normalize_text(text)))


def normalize_metric_phrase(text: str) -> str:
    phrase = clean_phrase(text)
    known_hits = [term for term in METRIC_TERMS if term in phrase]
    if known_hits:
        return sorted(known_hits, key=len, reverse=True)[0]
    drop_prefix = {
        "a",
        "an",
        "and",
        "the",
        "said",
        "says",
        "show",
        "shows",
        "cited",
        "have",
        "has",
        "had",
        "higher",
        "lower",
        "low",
        "high",
        "same",
        "she",
        "he",
        "they",
    }
    toks = phrase.split()
    while toks and toks[0] in drop_prefix:
        toks = toks[1:]
    return " ".join(toks)


def normalize_number(value: str) -> str:
    value = normalize_text(value).replace(",", "")
    value = value.strip("$")
    value = value.replace("percent", "%")
    if value.endswith("%"):
        return value
    return normalize_decade(value) or value


def normalize_decade(value: str) -> str:
    text = normalize_text(value).strip("'")
    if re.fullmatch(r"\d0s", text):
        return text
    return ""


def number_forms(text: str) -> set[str]:
    forms = {normalize_number(x) for x in NUMBER_RE.findall(text)}
    forms.update(YEAR_RE.findall(text))
    forms.update(normalize_decade(x) for x in DECADE_RE.findall(text))
    return {x for x in forms if x}


def parse_json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def parse_retry_statuses(value: str) -> list[int]:
    statuses: list[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            statuses.append(int(part))
        except ValueError:
            continue
    return statuses


def sleep_before_retry(*, attempt: int, args: argparse.Namespace) -> None:
    backoff = max(safe_float(getattr(args, "llm_retry_backoff", 2.0), 2.0), 0.0)
    if backoff <= 0.0:
        return
    delay = min(backoff * (2**attempt), 60.0)
    time.sleep(delay)


if __name__ == "__main__":
    main()
