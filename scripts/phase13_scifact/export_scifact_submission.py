#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SciFact official-format predictions from MREC traces.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--predictions", default=None, help="Optional label-token predictions JSONL.")
    parser.add_argument("--build-jsonl", default=None, help="Build JSONL used to map prediction sample_idx to event_id.")
    parser.add_argument("--max-sentences-per-doc", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_rows = _read_jsonl(Path(args.trace))
    raw_rows = _read_jsonl(Path(args.raw))
    pred_by_event = _load_predictions(args.predictions, args.build_jsonl)
    submissions = [
        _submission_row(
            trace,
            pred_label=pred_by_event.get(str(trace.get("event_id") or "")),
            max_sentences_per_doc=int(args.max_sentences_per_doc),
        )
        for trace in trace_rows
    ]
    _write_jsonl(Path(args.output), submissions)

    metrics = _evaluate(submissions, raw_rows)
    metrics.update(
        {
            "trace": str(args.trace),
            "raw": str(args.raw),
            "output": str(args.output),
            "predictions": str(args.predictions or ""),
            "build_jsonl": str(args.build_jsonl or ""),
            "prediction_source_counts": dict(
                Counter(str(row.get("_prediction_source") or "unknown") for row in submissions)
            ),
        }
    )
    for row in submissions:
        row.pop("_prediction_source", None)
    _write_jsonl(Path(args.output), submissions)
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote SciFact submission: {args.output}")
    if args.metrics_output:
        print(f"Wrote SciFact metrics: {args.metrics_output}")
    return 0


def _submission_row(
    trace: Mapping[str, Any],
    *,
    pred_label: str | None,
    max_sentences_per_doc: int,
) -> dict[str, Any]:
    event_id = int(trace.get("event_id"))
    if _is_nei_label(pred_label):
        return {"id": event_id, "evidence": {}, "_prediction_source": "verifier_prediction"}
    grouped: dict[str, dict[str, Any]] = {}
    selected = trace.get("selected_candidates") or []
    if not selected:
        pool = trace.get("candidate_pool") or []
        selected = []
        for idx in trace.get("selector_ordered_indices") or trace.get("selected_indices") or []:
            try:
                pos = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= pos < len(pool):
                selected.append(pool[pos])
    source = "verifier_prediction" if pred_label else "evidence_map_relation_fallback"
    for candidate in selected:
        if not isinstance(candidate, Mapping):
            continue
        doc_id = str(candidate.get("scifact_doc_id") or candidate.get("doc_id") or candidate.get("report_id") or "")
        if not doc_id:
            continue
        item = grouped.setdefault(doc_id, {"label": None, "sentences": [], "_relations": []})
        for sent_idx in candidate.get("scifact_sentence_ids") or candidate.get("chunk_sent_indices") or [candidate.get("sent_idx")]:
            try:
                value = int(sent_idx)
            except (TypeError, ValueError):
                continue
            if value not in item["sentences"]:
                item["sentences"].append(value)
        relation = str(candidate.get("map_relation") or "").lower()
        if relation:
            item["_relations"].append(relation)
    evidence: dict[str, dict[str, Any]] = {}
    for doc_id, item in grouped.items():
        sentences = sorted(int(x) for x in item["sentences"])[: max(1, int(max_sentences_per_doc))]
        if not sentences:
            continue
        label = _official_label(pred_label) if pred_label else _fallback_doc_label(item.get("_relations") or [])
        evidence[doc_id] = {"label": label, "sentences": sentences}
    return {"id": event_id, "evidence": evidence, "_prediction_source": source}


def _fallback_doc_label(relations: list[str]) -> str:
    if any(rel in {"refute", "contradict", "contradicts", "refutes"} for rel in relations):
        return "CONTRADICT"
    return "SUPPORT"


def _official_label(label: str | None) -> str:
    text = str(label or "").strip().lower()
    if text in {"contradict", "contradicts", "refute", "refutes"}:
        return "CONTRADICT"
    if text in {"support", "supports", "supported"}:
        return "SUPPORT"
    raise ValueError(f"Unsupported SciFact verifier label: {label!r}")


def _is_nei_label(label: str | None) -> bool:
    text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text in {"nei", "not_enough_info", "insufficient", "unknown"}


def _load_predictions(predictions: str | None, build_jsonl: str | None) -> dict[str, str]:
    if not predictions or not Path(predictions).exists():
        return {}
    build_rows = _read_jsonl(Path(build_jsonl)) if build_jsonl and Path(build_jsonl).exists() else []
    event_order = [str(row.get("event_id") or "") for row in build_rows]
    out: dict[str, str] = {}
    for row in _read_jsonl(Path(predictions)):
        event_id = str(row.get("event_id") or "")
        if not event_id and "sample_idx" in row:
            try:
                event_id = event_order[int(row["sample_idx"])]
            except (IndexError, TypeError, ValueError):
                event_id = ""
        pred_label = str(row.get("pred_label") or "").strip().lower()
        if event_id and pred_label:
            out[event_id] = pred_label
    return out


def _evaluate(pred_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {int(row["id"]): row for row in raw_rows if isinstance(row.get("evidence"), dict)}
    abstract_correct = 0
    abstract_pred = 0
    abstract_gold = 0
    sentence_correct = 0
    sentence_pred = 0
    sentence_gold = 0
    claim_gold: list[str] = []
    claim_pred: list[str] = []
    for pred in pred_rows:
        claim_id = int(pred["id"])
        evidence = pred.get("evidence") or {}
        gold = gold_by_id.get(claim_id)
        if gold is None:
            abstract_pred += len(evidence)
            sentence_pred += sum(len((item or {}).get("sentences") or []) for item in evidence.values())
            continue
        gold_evidence = gold.get("evidence") or {}
        claim_gold.append(_claim_label(gold_evidence))
        claim_pred.append(_claim_label(evidence))
        abstract_pred += len(evidence)
        abstract_gold += len(gold_evidence)
        sentence_pred += sum(len((item or {}).get("sentences") or []) for item in evidence.values())
        sentence_gold += len(_gold_sentence_keys(gold_evidence))
        for doc_id, pred_doc in evidence.items():
            gold_rationales = gold_evidence.get(str(doc_id)) or []
            if not gold_rationales:
                try:
                    gold_rationales = gold_evidence.get(int(doc_id)) or []
                except (TypeError, ValueError):
                    gold_rationales = []
            if not gold_rationales:
                continue
            pred_label = str(pred_doc.get("label") or "")
            pred_sentence_list = [int(x) for x in pred_doc.get("sentences") or []]
            pred_sentences_all = set(pred_sentence_list)
            pred_sentences_abstract = set(pred_sentence_list[:3])
            gold_label = _gold_doc_label(gold_rationales)
            if pred_label == gold_label and _contains_gold_set(gold_rationales, pred_sentences_abstract):
                abstract_correct += 1
            if pred_label == gold_label:
                for sent_idx in pred_sentences_all:
                    if _sentence_credit(gold_rationales, sent_idx, pred_sentences_all):
                        sentence_correct += 1
    return {
        "abstract": _prf(abstract_correct, abstract_pred, abstract_gold),
        "sentence": _prf(sentence_correct, sentence_pred, sentence_gold),
        "claim_label": _classification_metrics(claim_gold, claim_pred),
        "counts": {
            "abstract_correct": abstract_correct,
            "abstract_pred": abstract_pred,
            "abstract_gold": abstract_gold,
            "sentence_correct": sentence_correct,
            "sentence_pred": sentence_pred,
            "sentence_gold": sentence_gold,
        },
    }


def _gold_doc_label(rationales: list[Mapping[str, Any]]) -> str:
    for rationale in rationales:
        label = str(rationale.get("label") or "").strip().upper()
        if label in {"SUPPORT", "CONTRADICT"}:
            return label
    return "SUPPORT"


def _claim_label(evidence: Mapping[str, Any]) -> str:
    if not evidence:
        return "NEI"
    labels: list[str] = []
    for item in evidence.values():
        if isinstance(item, Mapping):
            label = str(item.get("label") or "").strip().upper()
            if label:
                labels.append(label)
        elif isinstance(item, list):
            labels.extend(str(row.get("label") or "").strip().upper() for row in item if isinstance(row, Mapping))
    if "CONTRADICT" in labels:
        return "CONTRADICT"
    return "SUPPORT"


def _classification_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = ["SUPPORT", "CONTRADICT", "NEI"]
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        per_class[label] = _prf(tp, tp + fp, tp + fn)
    return {
        "accuracy": _ratio(sum(g == p for g, p in zip(gold, pred)), len(gold)),
        "macro_f1": _ratio(sum(row["f1"] for row in per_class.values()), len(labels)),
        "per_class": per_class,
        "n": len(gold),
    }


def _contains_gold_set(rationales: list[Mapping[str, Any]], pred_sentences: set[int]) -> bool:
    for rationale in rationales:
        gold = {int(x) for x in rationale.get("sentences") or []}
        if gold and gold.issubset(pred_sentences):
            return True
    return False


def _sentence_credit(rationales: list[Mapping[str, Any]], sent_idx: int, pred_sentences: set[int]) -> bool:
    for rationale in rationales:
        gold = {int(x) for x in rationale.get("sentences") or []}
        if sent_idx in gold and gold.issubset(pred_sentences):
            return True
    return False


def _gold_sentence_keys(evidence: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for doc_id, rationales in evidence.items():
        for rationale in rationales or []:
            for sent_idx in rationale.get("sentences") or []:
                out.add(f"{doc_id}:{int(sent_idx)}")
    return out


def _prf(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = float(correct / predicted) if predicted else 0.0
    recall = float(correct / gold) if gold else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
