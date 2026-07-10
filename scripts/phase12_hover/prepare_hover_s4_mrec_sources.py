#!/usr/bin/env python3
"""Prepare HoVer S4 MREC source files from S3 candidate pools."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from fact_checking.data.io import load_split
from fact_checking.utils.io import save_json, write_jsonl
from fact_checking.utils.text import clean_text
from scripts.phase12_hover.build_hover_gold_sentence_verifier_data import _supporting_fact_pairs


DEFAULT_S3_OUTPUT_DIR = "outputs/sentence_trace_method/hover__ministral3_8b__bm25_page_mmr_sentence_minmax9_9"
DEFAULT_OUTPUT_DIR = "outputs/selectors/atom_anchor/hover_abc_v0_1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HoVer S4 MREC source artifacts.")
    p.add_argument("--train-raw", default="data/raw/HoVer/train.json")
    p.add_argument("--val-raw", default="data/raw/HoVer/val.json")
    p.add_argument("--s3-output-dir", default=DEFAULT_S3_OUTPUT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    s3_dir = Path(args.s3_output_dir)
    reports: dict[str, Any] = {}
    for split, raw_path in (("train", Path(args.train_raw)), ("val", Path(args.val_raw))):
        retrieval_path = s3_dir / "retrieval" / f"retrieval_{split}.jsonl"
        reports[split] = build_s4_sources_for_split(
            split=split,
            raw_path=raw_path,
            retrieval_path=retrieval_path,
            output_dir=output_dir,
            sample_limit=args.sample_limit,
        )
    manifest = {
        "status": "completed",
        "dataset": "hover",
        "label_schema": "hover2",
        "source_s3_output_dir": str(s3_dir),
        "output_dir": str(output_dir),
        "splits": reports,
        "notes": [
            "This prepares HoVer S4 source supervision; it does not train the learned marginal proxy.",
            "Claim atoms use deterministic fallback atoms unless replaced by an API atomization pass later.",
            "Evidence maps mark gold title and gold sentence hits from HoVer supporting_facts.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, output_dir / "manifest.json")
    print(f"Wrote HoVer S4 MREC source artifacts to {output_dir}")
    for split, report in reports.items():
        print(
            f"{split}: rows={report['n_rows']} gold_sentence_candidates={report['gold_sentence_candidate_count']} "
            f"proxy_pairs={report['proxy_pair_count']}"
        )


def build_s4_sources_for_split(
    *,
    split: str,
    raw_path: Path,
    retrieval_path: Path,
    output_dir: Path,
    sample_limit: int | None,
) -> dict[str, Any]:
    raw_by_id = {sample.event_id: sample for sample in load_split(raw_path, dataset="hover", label_schema="hover2")}
    retrieval_rows = list(_read_jsonl(retrieval_path))
    if sample_limit is not None:
        retrieval_rows = retrieval_rows[: int(sample_limit)]

    atom_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in retrieval_rows:
        event_id = str(row.get("event_id") or "")
        sample = raw_by_id.get(event_id)
        if sample is None:
            counts["missing_raw_sample"] += 1
            continue
        facts = _supporting_fact_pairs(sample.metadata.get("supporting_facts"))
        gold_titles = {title for title, _idx in facts}
        gold_sentences = {(title, int(idx)) for title, idx in facts}
        candidates = [dict(candidate) for candidate in row.get("candidates") or []]
        atoms = _fallback_atoms(sample.claim)

        labels = []
        gold_indices: list[int] = []
        non_gold_indices: list[int] = []
        for idx, candidate in enumerate(candidates):
            title = clean_text(str(candidate.get("hover_page_title") or candidate.get("report_id") or "")).replace("_", " ")
            sent_idx = int(candidate.get("hover_sent_idx", candidate.get("sent_idx", -1)))
            is_gold_title = title in gold_titles
            is_gold_sentence = (title, sent_idx) in gold_sentences
            if is_gold_sentence:
                gold_indices.append(idx)
            else:
                non_gold_indices.append(idx)
            labels.append(
                {
                    "candidate_index": idx,
                    "hover_page_title": title,
                    "hover_sent_idx": sent_idx,
                    "is_gold_title": bool(is_gold_title),
                    "is_gold_sentence": bool(is_gold_sentence),
                    "gold_match_type": "sentence" if is_gold_sentence else ("title" if is_gold_title else "none"),
                }
            )

        for winner in gold_indices:
            for loser in non_gold_indices:
                proxy_rows.append(
                    {
                        "event_id": event_id,
                        "winner_candidate_index": int(winner),
                        "loser_candidate_index": int(loser),
                        "reason": "gold_sentence_over_non_gold",
                    }
                )

        atom_rows.append(
            {
                "event_id": event_id,
                "claim": sample.claim,
                "label": sample.label,
                "claim_atoms": [{"atom_id": f"a{i}", "text": atom} for i, atom in enumerate(atoms)],
                "atomization_source": "fallback",
            }
        )
        pool_rows.append(
            {
                "event_id": event_id,
                "claim": sample.claim,
                "label": sample.label,
                "candidates": candidates,
            }
        )
        evidence_rows.append(
            {
                "event_id": event_id,
                "claim": sample.claim,
                "label": sample.label,
                "supporting_facts": [[title, idx] for title, idx in facts],
                "candidate_labels": labels,
                "claim_atoms": [{"atom_id": f"a{i}", "text": atom} for i, atom in enumerate(atoms)],
            }
        )
        counts["rows"] += 1
        counts["gold_sentence_candidate_count"] += len(gold_indices)
        counts["gold_title_candidate_count"] += sum(1 for label in labels if label["is_gold_title"])

    paths = {
        "claim_atoms": output_dir / "01_claim_atoms" / f"{split}.jsonl",
        "candidate_pool": output_dir / "02_candidate_pool" / f"{split}.jsonl",
        "evidence_map": output_dir / "04_evidence_map" / f"{split}.jsonl",
        "proxy_pairs": output_dir / "05_mrec_v0_2_learned_marginal_proxy_fullpool" / f"{split}_proxy_pairs.jsonl",
    }
    for path, rows in (
        (paths["claim_atoms"], atom_rows),
        (paths["candidate_pool"], pool_rows),
        (paths["evidence_map"], evidence_rows),
        (paths["proxy_pairs"], proxy_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, path)

    report = {
        "split": split,
        "raw_path": str(raw_path),
        "retrieval_path": str(retrieval_path),
        "n_rows": int(counts["rows"]),
        "gold_sentence_candidate_count": int(counts["gold_sentence_candidate_count"]),
        "gold_title_candidate_count": int(counts["gold_title_candidate_count"]),
        "proxy_pair_count": len(proxy_rows),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    save_json(report, output_dir / f"{split}_report.json")
    return report


def _fallback_atoms(claim: str) -> list[str]:
    parts = []
    for chunk in str(claim).replace(";", ".").split("."):
        text = clean_text(chunk)
        if text:
            parts.append(text)
    return parts or [clean_text(claim)]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    main()
