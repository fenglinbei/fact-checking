#!/usr/bin/env python3
"""Build canonical BACES traces and immediately replay-audit every row."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, BinaryIO, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.selectors.baces_trace import (  # noqa: E402
    TRACE_SCHEMA_VERSION,
    build_exact_trace,
    replay_trace,
)


AUDIT_SCHEMA_VERSION = "baces_trace_replay_audit_v0_1"
SUMMARY_SCHEMA_VERSION = "baces_trace_build_summary_v0_1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        "--input",
        dest="features",
        required=True,
        help="Evidence-map feature JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for baces_trace.jsonl, baces_replay_audit.jsonl, and "
            "baces_summary.json. Explicit output paths override these defaults."
        ),
    )
    parser.add_argument(
        "--trace-output",
        "--output-jsonl",
        dest="trace_output",
        help="Canonical trace JSONL path.",
    )
    parser.add_argument(
        "--audit-output",
        "--audit-jsonl",
        dest="audit_output",
        help="Per-row replay audit JSONL path.",
    )
    parser.add_argument("--summary-json", help="Build/replay summary JSON path.")
    parser.add_argument("--k-min", type=int, required=True)
    parser.add_argument("--k-max", type=int, required=True)
    parser.add_argument(
        "--token-budget",
        type=int,
        help="Optional non-negative additive evidence-token budget.",
    )
    parser.add_argument(
        "--cost-trace",
        help=(
            "Optional selector trace JSONL supplying candidate_pool[*].mrec_token_cost "
            "by event_id and candidate_uid. Use this for prompt-matched BACES runs."
        ),
    )
    parser.add_argument(
        "--cost-tokenizer-id",
        default="",
        help="Tokenizer/model identifier recorded in the canonical cost contract.",
    )
    parser.add_argument(
        "--cost-tokenizer-revision",
        default="",
        help="Optional tokenizer revision recorded in the canonical cost contract.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most the first N non-empty feature rows.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.k_min < 0:
        parser.error("--k-min must be non-negative")
    if args.k_max < 0:
        parser.error("--k-max must be non-negative")
    if args.k_min > args.k_max:
        parser.error("--k-min must be less than or equal to --k-max")
    if args.token_budget is not None and args.token_budget < 0:
        parser.error("--token-budget must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if not args.output_dir and not (
        args.trace_output and args.audit_output and args.summary_json
    ):
        parser.error(
            "provide --output-dir or all of --trace-output, --audit-output, "
            "and --summary-json"
        )
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    features_path = Path(args.features)
    cost_trace_path = Path(args.cost_trace) if args.cost_trace else None
    cost_offsets = (
        _index_jsonl_offsets(cost_trace_path, artifact="cost trace")
        if cost_trace_path is not None
        else {}
    )
    trace_path, audit_path, summary_path = _output_paths(args)
    for path in (trace_path, audit_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    trace_count = 0
    replay_ok_count = 0
    failures: list[dict[str, Any]] = []
    k_core_values: list[int] = []
    k_sel_values: list[int] = []
    fill_values: list[int] = []

    with ExitStack() as stack:
        feature_handle = stack.enter_context(features_path.open(encoding="utf-8"))
        trace_handle = stack.enter_context(trace_path.open("w", encoding="utf-8"))
        audit_handle = stack.enter_context(audit_path.open("w", encoding="utf-8"))
        cost_handle = (
            stack.enter_context(cost_trace_path.open("rb"))
            if cost_trace_path is not None
            else None
        )
        seen_event_ids: set[str] = set()
        for line_number, raw_line in enumerate(feature_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            event_id = ""
            try:
                feature_row = json.loads(line)
                if not isinstance(feature_row, Mapping):
                    raise TypeError("feature row must decode to a JSON object")
                event_id = str(feature_row.get("event_id") or "").strip()
                if not event_id:
                    raise ValueError("feature row has no non-empty event_id")
                if event_id in seen_event_ids:
                    raise ValueError(f"duplicate event_id: {event_id!r}")
                seen_event_ids.add(event_id)
                cost_overrides = None
                if cost_handle is not None:
                    if event_id not in cost_offsets:
                        raise ValueError(
                            f"cost trace is missing event_id {event_id!r}"
                        )
                    cost_row = _read_jsonl_row_at(
                        cost_handle,
                        cost_offsets[event_id],
                        artifact=f"cost trace:{event_id}",
                    )
                    cost_overrides = _mrec_cost_overrides(
                        feature_row=feature_row,
                        cost_row=cost_row,
                        event_id=event_id,
                    )
                feature_for_trace = dict(feature_row)
                if args.cost_tokenizer_id:
                    feature_for_trace["cost_tokenizer_id"] = str(
                        args.cost_tokenizer_id
                    )
                if args.cost_tokenizer_revision:
                    feature_for_trace["cost_tokenizer_revision"] = str(
                        args.cost_tokenizer_revision
                    )
                trace = build_exact_trace(
                    feature_for_trace,
                    k_min=int(args.k_min),
                    k_max=int(args.k_max),
                    token_budget=(
                        None if args.token_budget is None else int(args.token_budget)
                    ),
                    cost_overrides=cost_overrides,
                )
                trace["cost_source"] = (
                    "cost_trace.candidate_pool.mrec_token_cost"
                    if cost_trace_path is not None
                    else "feature_candidate_cost"
                )
                trace["cost_trace_path"] = (
                    str(cost_trace_path) if cost_trace_path is not None else None
                )
                replay = replay_trace(trace)
                audit = _audit_row(
                    event_id=event_id,
                    line_number=line_number,
                    replay=replay,
                )
                if not replay["ok"]:
                    failures.append(
                        {
                            "event_id": event_id,
                            "line_number": line_number,
                            "stage": "replay",
                            "errors": list(replay["errors"]),
                        }
                    )
                else:
                    replay_ok_count += 1
                _write_jsonl_row(trace_handle, trace)
                trace_count += 1
                k_core_values.append(int(trace["k_core"]))
                k_sel_values.append(int(trace["k_sel"]))
                fill_values.append(int(trace["zero_gain_fill_count"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                audit = {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "event_id": event_id,
                    "feature_line_number": line_number,
                    "ok": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
                failures.append(
                    {
                        "event_id": event_id,
                        "line_number": line_number,
                        "stage": "build",
                        "errors": list(audit["errors"]),
                    }
                )
            _write_jsonl_row(audit_handle, audit)

    if processed == 0:
        failures.append(
            {
                "event_id": "",
                "line_number": None,
                "stage": "input",
                "errors": ["no non-empty feature rows were processed"],
            }
        )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "features": str(features_path),
        "trace_output": str(trace_path),
        "audit_output": str(audit_path),
        "params": {
            "k_min": int(args.k_min),
            "k_max": int(args.k_max),
            "token_budget": (
                None if args.token_budget is None else int(args.token_budget)
            ),
            "cost_trace": (
                str(cost_trace_path) if cost_trace_path is not None else None
            ),
            "cost_source": (
                "candidate_pool.mrec_token_cost"
                if cost_trace_path is not None
                else "feature_candidate_cost"
            ),
            "cost_tokenizer_id": str(args.cost_tokenizer_id or ""),
            "cost_tokenizer_revision": str(args.cost_tokenizer_revision or ""),
            "limit": None if args.limit is None else int(args.limit),
        },
        "feature_rows_processed": processed,
        "trace_rows_written": trace_count,
        "replay_ok_count": replay_ok_count,
        "failure_count": len(failures),
        "all_replays_ok": not failures and trace_count == processed,
        "min_count_unreachable_count": sum(
            1 for k_sel in k_sel_values if k_sel < int(args.k_min)
        ),
        "mean_k_core": _mean(k_core_values),
        "mean_k_sel": _mean(k_sel_values),
        "mean_zero_gain_fill_count": _mean(fill_values),
        "failures": failures,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"BACES traces: {trace_count}/{processed}; replay OK: "
        f"{replay_ok_count}; failures: {len(failures)}"
    )
    print(f"Trace JSONL: {trace_path}")
    print(f"Replay audit: {audit_path}")
    print(f"Summary: {summary_path}")
    return 1 if failures or trace_count != processed else 0


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir) if args.output_dir else None
    trace_path = (
        Path(args.trace_output)
        if args.trace_output
        else output_dir / "baces_trace.jsonl"  # type: ignore[operator]
    )
    audit_path = (
        Path(args.audit_output)
        if args.audit_output
        else output_dir / "baces_replay_audit.jsonl"  # type: ignore[operator]
    )
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else output_dir / "baces_summary.json"  # type: ignore[operator]
    )
    return trace_path, audit_path, summary_path


def _audit_row(
    *,
    event_id: str,
    line_number: int,
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_id": event_id,
        "feature_line_number": line_number,
        "ok": bool(replay.get("ok")),
        "errors": list(replay.get("errors") or []),
        "candidate_pool_fingerprint": str(
            replay.get("candidate_pool_fingerprint") or ""
        ),
        "selected_set_fingerprint": str(
            replay.get("selected_set_fingerprint") or ""
        ),
        "display_order_fingerprint": str(
            replay.get("display_order_fingerprint") or ""
        ),
        "trace_fingerprint": str(replay.get("trace_fingerprint") or ""),
        "k_core": replay.get("k_core"),
        "k_sel": replay.get("k_sel"),
        "terminal_ordinal_coverage_units": replay.get(
            "terminal_ordinal_coverage_units"
        ),
        "core_weighted_coverage_acquisition_time": replay.get(
            "core_weighted_coverage_acquisition_time"
        ),
        "display_weighted_coverage_acquisition_time": replay.get(
            "display_weighted_coverage_acquisition_time"
        ),
        "core_padded_prefix_auc": replay.get("core_padded_prefix_auc"),
        "display_padded_prefix_auc": replay.get("display_padded_prefix_auc"),
    }


def _index_jsonl_offsets(path: Path, *, artifact: str) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {artifact}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise TypeError(f"{artifact}:{line_number} must be a JSON object")
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                raise ValueError(f"{artifact}:{line_number} has no event_id")
            if event_id in offsets:
                raise ValueError(f"duplicate event_id {event_id!r} in {artifact}")
            offsets[event_id] = offset
    return offsets


def _read_jsonl_row_at(
    handle: BinaryIO, offset: int, *, artifact: str
) -> dict[str, Any]:
    handle.seek(offset)
    raw_line = handle.readline()
    try:
        row = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid indexed JSON in {artifact}: {exc}") from exc
    if not isinstance(row, dict):
        raise TypeError(f"{artifact} must be a JSON object")
    return row


def _mrec_cost_overrides(
    *,
    feature_row: Mapping[str, Any],
    cost_row: Mapping[str, Any],
    event_id: str,
) -> dict[str, int]:
    if str(cost_row.get("event_id") or "").strip() != event_id:
        raise ValueError(f"cost trace event_id mismatch for {event_id!r}")
    feature_candidates = feature_row.get("candidates")
    cost_candidates = cost_row.get("candidate_pool")
    if not isinstance(feature_candidates, list):
        raise TypeError(f"features:{event_id}: candidates must be a list")
    if not isinstance(cost_candidates, list):
        raise TypeError(f"cost trace:{event_id}: candidate_pool must be a list")

    feature_uids: list[str] = []
    for index, candidate in enumerate(feature_candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"features:{event_id}: candidates[{index}] is not an object")
        uid = str(candidate.get("candidate_uid") or "").strip()
        if not uid:
            raise ValueError(f"features:{event_id}: candidates[{index}] has no candidate_uid")
        feature_uids.append(uid)

    overrides: dict[str, int] = {}
    for index, candidate in enumerate(cost_candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(
                f"cost trace:{event_id}: candidate_pool[{index}] is not an object"
            )
        uid = str(candidate.get("candidate_uid") or "").strip()
        value = candidate.get("mrec_token_cost")
        if not uid:
            raise ValueError(
                f"cost trace:{event_id}: candidate_pool[{index}] has no candidate_uid"
            )
        if uid in overrides:
            raise ValueError(f"cost trace:{event_id}: duplicate candidate_uid {uid!r}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"cost trace:{event_id}:{uid}: mrec_token_cost must be a "
                "non-negative integer"
            )
        overrides[uid] = int(value)

    if len(feature_uids) != len(set(feature_uids)):
        raise ValueError(f"features:{event_id}: duplicate candidate_uid")
    if set(feature_uids) != set(overrides):
        raise ValueError(
            f"features/cost trace UID sets differ for {event_id}: "
            f"feature_only={sorted(set(feature_uids) - set(overrides))[:10]}, "
            f"cost_only={sorted(set(overrides) - set(feature_uids))[:10]}"
        )
    return overrides


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
