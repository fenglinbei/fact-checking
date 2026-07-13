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
    parser.add_argument(
        "--build-jsonl",
        default=None,
        help="Build JSONL used for prediction id mapping and final prompt evidence selection.",
    )
    parser.add_argument("--max-sentences-per-doc", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_rows = _read_jsonl(Path(args.trace))
    raw_rows = _read_jsonl(Path(args.raw))
    build_rows = _read_jsonl(Path(args.build_jsonl)) if args.build_jsonl and Path(args.build_jsonl).exists() else []
    build_by_event = {str(row.get("event_id") or ""): row for row in build_rows}
    trace_event_ids = [str(row.get("event_id") or "") for row in trace_rows]
    if args.predictions and Path(args.predictions).exists() and not build_rows:
        raise ValueError("Verifier predictions require --build-jsonl for sample_idx and prompt evidence mapping.")
    if build_rows:
        if len(build_by_event) != len(build_rows):
            raise ValueError("Build JSONL contains missing or duplicate event_id values.")
        missing_build = [event_id for event_id in trace_event_ids if event_id not in build_by_event]
        if missing_build:
            raise ValueError(f"Build JSONL is missing {len(missing_build)} trace events, sample={missing_build[:5]}.")
    pred_by_event = _load_predictions(args.predictions, build_rows)
    if args.predictions and Path(args.predictions).exists():
        missing_predictions = [event_id for event_id in trace_event_ids if event_id not in pred_by_event]
        if missing_predictions:
            raise ValueError(
                f"Predictions are missing {len(missing_predictions)} trace events, sample={missing_predictions[:5]}."
            )
    submissions = [
        _submission_row(
            trace,
            pred_label=pred_by_event.get(str(trace.get("event_id") or "")),
            max_sentences_per_doc=int(args.max_sentences_per_doc),
            prompt_candidates=_prompt_candidates(
                build_by_event.get(str(trace.get("event_id") or ""))
            ) if build_rows else None,
        )
        for trace in trace_rows
    ]

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
            "evidence_source_counts": dict(
                Counter(str(row.get("_evidence_source") or "unknown") for row in submissions)
            ),
            "evidence_policy": (
                "prompt_relation_aligned_mrec_top1" if build_rows else "trace_selected_candidates"
            ),
        }
    )
    for row in submissions:
        row.pop("_prediction_source", None)
        row.pop("_evidence_source", None)
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
    prompt_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_id = int(trace.get("event_id"))
    if _is_nei_label(pred_label):
        return {
            "id": event_id,
            "evidence": {},
            "_prediction_source": "verifier_prediction",
            "_evidence_source": "nei_empty",
        }
    grouped: dict[str, dict[str, Any]] = {}
    if prompt_candidates is not None and pred_label:
        selected, evidence_source = _select_prompt_submission_candidates(prompt_candidates, pred_label)
    else:
        selected = _trace_selected_candidates(trace)
        evidence_source = "trace_selected_candidates"
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
    return {
        "id": event_id,
        "evidence": evidence,
        "_prediction_source": source,
        "_evidence_source": evidence_source,
    }


def _prompt_candidates(build_row: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(build_row, Mapping):
        return []
    candidates = [dict(row) for row in build_row.get("candidates") or [] if isinstance(row, Mapping)]
    try:
        evidence_count = int(build_row.get("evidence_count", len(candidates)))
    except (TypeError, ValueError):
        evidence_count = len(candidates)
    return candidates[: max(0, evidence_count)]


def _select_prompt_submission_candidates(
    candidates: list[dict[str, Any]],
    pred_label: str,
) -> tuple[list[dict[str, Any]], str]:
    valid = [candidate for candidate in candidates if _has_scifact_location(candidate)]
    aligned = [candidate for candidate in valid if _relation_matches_label(candidate, pred_label)]
    if aligned:
        return aligned[:1], "prompt_relation_aligned_top1"
    if valid:
        return valid[:1], "prompt_top1_fallback"
    raise ValueError("Non-NEI SciFact prediction has no prompt candidate with document and sentence ids.")


def _trace_selected_candidates(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selected = [row for row in trace.get("selected_candidates") or [] if isinstance(row, Mapping)]
    if selected:
        return selected
    pool = trace.get("candidate_pool") or []
    out: list[Mapping[str, Any]] = []
    for idx in trace.get("selector_ordered_indices") or trace.get("selected_indices") or []:
        try:
            pos = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < len(pool) and isinstance(pool[pos], Mapping):
            out.append(pool[pos])
    return out


def _relation_matches_label(candidate: Mapping[str, Any], pred_label: str) -> bool:
    relation = str(candidate.get("map_relation") or "").strip().lower()
    label = str(pred_label or "").strip().lower()
    if label in {"support", "supports", "supported"}:
        return relation in {"support", "supports", "supported"}
    if label in {"contradict", "contradicts", "refute", "refutes"}:
        return relation in {"contradict", "contradicts", "refute", "refutes"}
    return False


def _has_scifact_location(candidate: Mapping[str, Any]) -> bool:
    doc_id = str(candidate.get("scifact_doc_id") or candidate.get("doc_id") or candidate.get("report_id") or "")
    sentence_ids = (
        candidate.get("scifact_sentence_ids")
        or candidate.get("chunk_sent_indices")
        or [candidate.get("sent_idx")]
    )
    if not isinstance(sentence_ids, (list, tuple, set)):
        sentence_ids = [sentence_ids]
    return bool(doc_id and any(value is not None for value in sentence_ids))


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


def _load_predictions(predictions: str | None, build_rows: list[dict[str, Any]]) -> dict[str, str]:
    if not predictions or not Path(predictions).exists():
        return {}
    event_order = [str(row.get("event_id") or "") for row in build_rows]
    out: dict[str, str] = {}
    for row in _read_jsonl(Path(predictions)):
        event_id = str(row.get("event_id") or "")
        if not event_id and "sample_idx" in row:
            try:
                event_id = event_order[int(row["sample_idx"])]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction sample_idx={row.get('sample_idx')!r}.") from exc
        pred_label = str(row.get("pred_label") or "").strip().lower()
        if event_id and pred_label:
            out[event_id] = pred_label
    return out


def _evaluate(pred_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {int(row["id"]): row for row in raw_rows if isinstance(row.get("evidence"), dict)}
    abstract_label_only_correct = 0
    abstract_label_rationale_correct = 0
    abstract_pred = 0
    abstract_gold = 0
    sentence_selection_only_correct = 0
    sentence_selection_label_correct = 0
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
            for sent_idx in pred_sentences_all:
                if _sentence_credit(gold_rationales, sent_idx, pred_sentences_all):
                    sentence_selection_only_correct += 1
            if pred_label == gold_label:
                abstract_label_only_correct += 1
                if _contains_gold_set(gold_rationales, pred_sentences_abstract):
                    abstract_label_rationale_correct += 1
                for sent_idx in pred_sentences_all:
                    if _sentence_credit(gold_rationales, sent_idx, pred_sentences_all):
                        sentence_selection_label_correct += 1
    abstract_label_only = _prf(abstract_label_only_correct, abstract_pred, abstract_gold)
    abstract_label_rationale = _prf(abstract_label_rationale_correct, abstract_pred, abstract_gold)
    sentence_selection_only = _prf(sentence_selection_only_correct, sentence_pred, sentence_gold)
    sentence_selection_label = _prf(sentence_selection_label_correct, sentence_pred, sentence_gold)
    return {
        # Preserve historical aliases while exposing the metric names used by
        # SciFact papers explicitly.
        "abstract": abstract_label_rationale,
        "sentence": sentence_selection_label,
        "abstract_label_only": abstract_label_only,
        "abstract_label_rationale": abstract_label_rationale,
        "sentence_selection_only": sentence_selection_only,
        "sentence_selection_label": sentence_selection_label,
        "primary_comparison": {
            "abstract_label_only_f1": abstract_label_only["f1"],
            "sentence_selection_label_f1": sentence_selection_label["f1"],
        },
        "claim_label": _classification_metrics(claim_gold, claim_pred),
        "counts": {
            "abstract_correct": abstract_label_rationale_correct,
            "abstract_label_only_correct": abstract_label_only_correct,
            "abstract_label_rationale_correct": abstract_label_rationale_correct,
            "abstract_pred": abstract_pred,
            "abstract_gold": abstract_gold,
            "sentence_correct": sentence_selection_label_correct,
            "sentence_selection_only_correct": sentence_selection_only_correct,
            "sentence_selection_label_correct": sentence_selection_label_correct,
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
