#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit atom-anchor source artifacts across splits.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "audit_mode": "source_only",
        "splits": {},
        "verifier": {"build_report": {"val_only": False}},
    }
    for split in args.splits:
        split = str(split).strip()
        if not split:
            continue
        report["splits"][split] = _audit_split(root, split)

    output = Path(args.output) if args.output else root / "quality_audit_after_fix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote atom-anchor quality audit: {output}")
    return 0


def _audit_split(root: Path, split: str) -> dict[str, Any]:
    claim_atoms = root / "01_claim_atoms" / f"claim_atoms_{split}.jsonl"
    retrieval_trace = root / "02_atom_retrieval" / f"retrieval_trace_{split}.jsonl"
    atom_union_pool = root / "03_atom_union" / f"atom_union_candidate_pool_{split}.jsonl"
    candidate_pool = root / "04_evidence_map" / f"evidence_map_candidate_pool_{split}.jsonl"
    annotations = root / "04_evidence_map" / f"deepseek_evidence_map_annotations_{split}.jsonl"
    errors = root / "04_evidence_map" / f"deepseek_evidence_map_errors_{split}.jsonl"
    features = root / "04_evidence_map" / f"candidate_evidence_map_features_{split}.jsonl"

    feature_rows = _read_jsonl(features)
    candidate_rows = _read_jsonl(candidate_pool)
    annotation_rows = _read_jsonl(annotations)
    candidate_event_ids = _event_ids(candidate_rows)
    annotation_event_ids = _event_ids(annotation_rows)
    parse_counts = Counter(
        str(row.get("evidence_map_parse_status") or row.get("parse_status") or "missing")
        for row in feature_rows
    )
    fallback_missing_annotation = sum(
        1
        for row in feature_rows
        if str(row.get("evidence_map_parse_status") or row.get("parse_status") or "") == "fallback_missing_annotation"
    )
    return {
        "counts": {
            "claim_atoms": _count_jsonl(claim_atoms),
            "retrieval_trace": _count_jsonl(retrieval_trace),
            "atom_union_pool": _count_jsonl(atom_union_pool),
            "candidate_pool": len(candidate_rows),
            "annotations": len(annotation_rows),
            "features": len(feature_rows),
            "unresolved_errors": _count_jsonl(errors, missing_ok=True),
        },
        "missing_annotations": sorted(candidate_event_ids - annotation_event_ids),
        "feature_fallback_missing_annotation": int(fallback_missing_annotation),
        "feature_parse_status_counts": dict(sorted(parse_counts.items())),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _count_jsonl(path: Path, *, missing_ok: bool = False) -> int:
    if missing_ok and not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _event_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("event_id") or "") for row in rows if str(row.get("event_id") or "")}


if __name__ == "__main__":
    raise SystemExit(main())
