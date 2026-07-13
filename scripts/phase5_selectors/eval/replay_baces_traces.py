#!/usr/bin/env python3
"""Independently replay-audit an existing canonical BACES trace JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.baces_trace import replay_trace  # noqa: E402


SCHEMA_VERSION = "baces_trace_replay_file_audit_v0_1"
DERIVED_FIELDS = (
    "candidate_pool_fingerprint",
    "selected_set_fingerprint",
    "display_order_fingerprint",
    "trace_fingerprint",
    "k_core",
    "k_sel",
    "terminal_ordinal_coverage_units",
    "core_weighted_coverage_acquisition_time",
    "display_weighted_coverage_acquisition_time",
    "core_padded_prefix_auc",
    "display_padded_prefix_auc",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    trace_path = Path(args.trace)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    seen_event_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            if args.limit is not None and row_count >= args.limit:
                break
            row_count += 1
            event_id = ""
            try:
                trace = json.loads(raw_line)
                if not isinstance(trace, Mapping):
                    raise TypeError("trace row must be a JSON object")
                event_id = str(trace.get("event_id") or "").strip()
                if not event_id:
                    raise ValueError("trace row has no event_id")
                if event_id in seen_event_ids:
                    raise ValueError(f"duplicate event_id: {event_id!r}")
                seen_event_ids.add(event_id)
                replay = replay_trace(trace)
                audit = {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": event_id,
                    "trace_line_number": line_number,
                    "status": "ok" if replay["ok"] else "replay_error",
                    "ok": bool(replay["ok"]),
                    "errors": list(replay["errors"]),
                    **{field: replay.get(field) for field in DERIVED_FIELDS},
                }
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                audit = {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": event_id,
                    "trace_line_number": line_number,
                    "status": "input_error",
                    "ok": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            status = str(audit["status"])
            status_counts[status] += 1
            if not audit["ok"]:
                failures.append(
                    {
                        "event_id": event_id,
                        "trace_line_number": line_number,
                        "status": status,
                        "errors": list(audit["errors"]),
                    }
                )
            output.write(
                json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n"
            )

    if row_count == 0:
        failures.append(
            {
                "event_id": "",
                "trace_line_number": None,
                "status": "input_error",
                "errors": ["no non-empty trace rows were processed"],
            }
        )
        status_counts["input_error"] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "trace": str(trace_path),
        "output_jsonl": str(output_path),
        "limit": args.limit,
        "row_count": row_count,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_count": len(failures),
        "all_replays_ok": not failures and status_counts.get("ok", 0) == row_count,
        "failures": failures,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Replayed {row_count} BACES traces; "
        f"status={dict(sorted(status_counts.items()))}"
    )
    print(f"Audit JSONL: {output_path}")
    print(f"Summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
