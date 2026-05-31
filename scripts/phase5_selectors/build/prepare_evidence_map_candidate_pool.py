#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fact_checking.selectors.count_amplified_stance_bucket_selector import oracle_ordered_keys
from fact_checking.selectors.evidence_map_selector import prepare_evidence_map_candidate_rows
from fact_checking.selectors.evidence_quality import canonical_candidate_key
from fact_checking.utils.io import read_jsonl, save_json, write_jsonl


DEFAULT_INPUT = "outputs/selectors/direct_evidence_cross_encoder/v0_4d_val_default_query_fusion/candidate_fusion_scores_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/selectors/evidence_map_selector/v0_5a_val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare evidence-map annotation candidate pool rows.")
    p.add_argument("--input-fusion-file", default=None)
    p.add_argument("--input-candidate-file", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--candidate-source", default="fusion", choices=["fusion", "qd_union"])
    p.add_argument("--candidate-top-n", type=int, default=20)
    p.add_argument("--oracle-results", default=None, help="Optional oracle_results JSONL used only for eval/trace metadata.")
    p.add_argument("--sample-limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = str(args.input_candidate_file or args.input_fusion_file or DEFAULT_INPUT)
    rows = read_jsonl(input_path)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    if not rows:
        raise ValueError(f"No rows loaded from {input_path}")
    oracle_path = str(args.oracle_results or "")
    oracle_by_event = _load_oracle_by_event(oracle_path) if oracle_path else {}
    if oracle_by_event:
        rows = _attach_oracle_metadata(rows, oracle_by_event)
    prepared = prepare_evidence_map_candidate_rows(
        rows,
        candidate_top_n=int(args.candidate_top_n),
        candidate_source=str(args.candidate_source),
    )
    output_path = out_dir / f"evidence_map_candidate_pool_{args.split}.jsonl"
    write_jsonl(prepared, output_path)
    manifest: dict[str, Any] = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "input_candidate_file": input_path,
        "oracle_results": oracle_path,
        "candidate_source": str(args.candidate_source),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "candidate_top_n": int(args.candidate_top_n),
        "sample_limit": int(args.sample_limit) if args.sample_limit is not None else None,
        "n_events": len(prepared),
        "n_candidates": sum(len(row.get("candidates") or []) for row in prepared),
        "outputs": {"candidate_pool": str(output_path)},
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "prepare_manifest.json")
    print(f"Wrote evidence-map candidate pool: {output_path}")


def _load_oracle_by_event(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("event_id") or ""): row
        for row in read_jsonl(path)
        if str(row.get("event_id") or "")
    }


def _attach_oracle_metadata(
    rows: list[dict[str, Any]],
    oracle_by_event: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        event_id = str(item.get("event_id") or "")
        oracle_row = oracle_by_event.get(event_id)
        if oracle_row is not None:
            keys = oracle_ordered_keys(oracle_row)
            item["gold_label"] = str(oracle_row.get("gold_label") or item.get("gold_label") or "")
            item["oracle_ordered_keys"] = keys
            item["oracle_selected_count"] = int(len(keys))
            key_to_step = {key: idx for idx, key in enumerate(keys)}
            candidates: list[dict[str, Any]] = []
            for candidate in item.get("candidates") or []:
                c = dict(candidate)
                key = canonical_candidate_key(c)
                selected = key in key_to_step
                c["oracle_selected"] = bool(selected)
                c["oracle_step"] = int(key_to_step[key]) if selected else -1
                candidates.append(c)
            item["candidates"] = candidates
        out.append(item)
    return out


if __name__ == "__main__":
    main()
