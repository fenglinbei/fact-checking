#!/usr/bin/env python3
"""Expand one maximal prompt-feasible plan per selector into a K-prefix lattice."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, TextIO


SCHEMA_VERSION = "capacity_prefix_plan_lattice_v0_1"
CELL_PLAN_SCHEMA_VERSION = "capacity_prefix_selection_plan_v0_1"
CAPACITY_POLICY = "prompt_feasible_strict_ordered_prefix_v0_1"


class PrefixLatticeError(ValueError):
    """Raised when maximal plans cannot define one strict prefix lattice."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-plan",
        action="append",
        required=True,
        help="SELECTOR=PATH maximal plan; repeat once per selector.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--source-controller", default="ordinal_replay_minmax5_10")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.min_k <= 0 or args.max_k < args.min_k:
        parser.error("require 0 < --min-k <= --max-k")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    selector_plans = _parse_selector_plans(args.selector_plan)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise PrefixLatticeError(
            f"output directory already exists; pass --overwrite: {output_dir}"
        )
    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise PrefixLatticeError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    promoted = False
    try:
        manifest = materialize_prefix_lattice(
            selector_plans=selector_plans,
            output_dir=staging,
            logical_output_dir=output_dir,
            split=str(args.split),
            min_k=int(args.min_k),
            max_k=int(args.max_k),
            source_controller=str(args.source_controller),
        )
        _promote_directory(staging, output_dir, overwrite=bool(args.overwrite))
        promoted = True
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)
    print(
        f"Materialized {manifest['cell_count']} prefix-plan cells x "
        f"{manifest['event_count']} events at {output_dir}"
    )
    return 0


def materialize_prefix_lattice(
    *,
    selector_plans: Mapping[str, Path],
    output_dir: Path,
    logical_output_dir: Path,
    split: str,
    min_k: int,
    max_k: int,
    source_controller: str,
) -> dict[str, Any]:
    if not selector_plans:
        raise PrefixLatticeError("no selector plans were provided")
    capacity_values = list(range(min_k, max_k + 1))
    controller_levels = [_capacity_level(k) for k in capacity_values]
    cells: list[dict[str, Any]] = []
    common_event_sha: str | None = None
    common_event_count: int | None = None

    for selector, source_plan in selector_plans.items():
        if not source_plan.is_file():
            raise PrefixLatticeError(f"missing maximal plan for {selector}: {source_plan}")
        summaries = _expand_selector_plan(
            selector=selector,
            source_plan=source_plan,
            output_dir=output_dir,
            logical_output_dir=logical_output_dir,
            split=split,
            capacity_values=capacity_values,
            max_k=max_k,
            source_controller=source_controller,
        )
        selector_event_sha = str(summaries[0]["event_id_sequence_sha256"])
        selector_event_count = int(summaries[0]["row_count"])
        if any(
            str(summary["event_id_sequence_sha256"]) != selector_event_sha
            or int(summary["row_count"]) != selector_event_count
            for summary in summaries
        ):
            raise PrefixLatticeError(f"{selector}: capacity cells have inconsistent events")
        if common_event_sha is None:
            common_event_sha = selector_event_sha
            common_event_count = selector_event_count
        elif selector_event_sha != common_event_sha or selector_event_count != common_event_count:
            raise PrefixLatticeError("selector maximal plans have different event sequences")
        cells.extend(summaries)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "matrix_kind": "ordered_prompt_feasible_prefix_plan_lattice",
        "capacity_policy": CAPACITY_POLICY,
        "split": split,
        "source_controller": source_controller,
        "selector_levels": list(selector_plans),
        "controller_levels": controller_levels,
        "capacity_values": capacity_values,
        "event_count": int(common_event_count or 0),
        "event_id_sequence_sha256": common_event_sha,
        "cell_count": len(cells),
        "all_ready": bool(cells) and all(bool(cell["ready"]) for cell in cells),
        "output_dir": str(logical_output_dir),
        "source_maximal_plans": {
            selector: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for selector, path in selector_plans.items()
        },
        "cells": cells,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _expand_selector_plan(
    *,
    selector: str,
    source_plan: Path,
    output_dir: Path,
    logical_output_dir: Path,
    split: str,
    capacity_values: list[int],
    max_k: int,
    source_controller: str,
) -> list[dict[str, Any]]:
    cell_paths: dict[int, Path] = {}
    temp_paths: dict[int, Path] = {}
    row_counts = {k: 0 for k in capacity_values}
    exact_counts = {k: 0 for k in capacity_values}
    candidate_exhausted_counts = {k: 0 for k in capacity_values}
    prompt_drop_counts = {k: 0 for k in capacity_values}
    realized_sums = {k: 0 for k in capacity_values}
    event_hashers = {k: hashlib.sha256() for k in capacity_values}
    promoted = False
    try:
        with ExitStack() as stack:
            handles: dict[int, TextIO] = {}
            for k in capacity_values:
                cell_id = _cell_id(selector, k)
                cell_dir = output_dir / cell_id
                cell_dir.mkdir(parents=True, exist_ok=True)
                path = cell_dir / f"selection_plan_{split}.jsonl"
                temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
                cell_paths[k] = path
                temp_paths[k] = temp
                handles[k] = stack.enter_context(temp.open("w", encoding="utf-8"))

            source_handle = stack.enter_context(source_plan.open(encoding="utf-8"))
            seen_events: set[str] = set()
            for source_row in _iter_jsonl(source_handle, artifact=str(source_plan)):
                event_id = str(source_row.get("event_id") or "").strip()
                if not event_id or event_id in seen_events:
                    raise PrefixLatticeError(
                        f"{selector}: empty or duplicate event_id={event_id!r}"
                    )
                seen_events.add(event_id)
                if int(source_row.get("requested_prefix_k", -1)) != max_k:
                    raise PrefixLatticeError(
                        f"{selector}:{event_id}: maximal plan requested K is not {max_k}"
                    )
                maximal_indices = _int_list(
                    source_row.get("selected_indices"),
                    f"{selector}:{event_id}:selected_indices",
                )
                maximal_uids = _string_list(
                    source_row.get("selected_candidate_uids"),
                    f"{selector}:{event_id}:selected_candidate_uids",
                )
                maximal_evidence_ids = _string_list(
                    source_row.get("selected_evidence_ids"),
                    f"{selector}:{event_id}:selected_evidence_ids",
                )
                if not maximal_indices or not (
                    len(maximal_indices) == len(maximal_uids) == len(maximal_evidence_ids)
                ):
                    raise PrefixLatticeError(
                        f"{selector}:{event_id}: invalid maximal selected prefix"
                    )
                if len(maximal_indices) != len(set(maximal_indices)) or len(
                    maximal_uids
                ) != len(set(maximal_uids)) or len(maximal_evidence_ids) != len(
                    set(maximal_evidence_ids)
                ):
                    raise PrefixLatticeError(
                        f"{selector}:{event_id}: maximal selected prefix contains duplicates"
                    )
                feasible_max = int(source_row.get("prompt_feasible_prefix_k", -1))
                available_max = int(source_row.get("available_prefix_k", -1))
                if feasible_max != len(maximal_indices) or not (
                    0 < feasible_max <= available_max <= max_k
                ):
                    raise PrefixLatticeError(
                        f"{selector}:{event_id}: inconsistent maximal feasibility metadata"
                    )

                previous_indices: list[int] | None = None
                for k in capacity_values:
                    available_k = min(k, available_max)
                    realized_k = min(k, feasible_max)
                    selected_indices = maximal_indices[:realized_k]
                    selected_uids = maximal_uids[:realized_k]
                    selected_evidence_ids = maximal_evidence_ids[:realized_k]
                    if previous_indices is not None:
                        if selected_indices[: len(previous_indices)] != previous_indices:
                            raise PrefixLatticeError(
                                f"{selector}:{event_id}: K={k} violates adjacent prefix nesting"
                            )
                        if len(selected_indices) - len(previous_indices) not in (0, 1):
                            raise PrefixLatticeError(
                                f"{selector}:{event_id}: K={k} changes capacity by more than one"
                            )
                    previous_indices = selected_indices
                    row = {
                        "schema_version": CELL_PLAN_SCHEMA_VERSION,
                        "projection_version": CAPACITY_POLICY,
                        "event_id": event_id,
                        "selector_level": selector,
                        "controller_level": _capacity_level(k),
                        "capacity_policy": CAPACITY_POLICY,
                        "source_controller": source_controller,
                        "trace_order_field": source_row.get("trace_order_field"),
                        "requested_prefix_k": k,
                        "available_prefix_k": available_k,
                        "prompt_feasible_prefix_k": realized_k,
                        "feasible_max_prefix_k": feasible_max,
                        "selected_indices": selected_indices,
                        "selected_candidate_uids": selected_uids,
                        "selected_evidence_ids": selected_evidence_ids,
                        "exact_policy_k": realized_k == k,
                        "candidate_exhausted": available_k < k,
                        "prompt_tail_drop_count": available_k - realized_k,
                        "source_max_plan": str(source_plan),
                    }
                    handles[k].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    row_counts[k] += 1
                    exact_counts[k] += int(bool(row["exact_policy_k"]))
                    candidate_exhausted_counts[k] += int(bool(row["candidate_exhausted"]))
                    prompt_drop_counts[k] += int(int(row["prompt_tail_drop_count"]) > 0)
                    realized_sums[k] += realized_k
                    event_hashers[k].update(event_id.encode("utf-8"))
                    event_hashers[k].update(b"\0")

        for k in capacity_values:
            temp_paths[k].replace(cell_paths[k])
        promoted = True
    finally:
        if not promoted:
            for path in temp_paths.values():
                path.unlink(missing_ok=True)

    summaries: list[dict[str, Any]] = []
    for k in capacity_values:
        cell_id = _cell_id(selector, k)
        row_count = row_counts[k]
        path = cell_paths[k]
        summary = {
            "cell_id": cell_id,
            "selector_level": selector,
            "controller_level": _capacity_level(k),
            "capacity_k": k,
            "capacity_policy": CAPACITY_POLICY,
            "source_order_cell": f"{selector}__{source_controller}",
            "source_max_plan": str(source_plan),
            "plan_file": f"{cell_id}/selection_plan_{split}.jsonl",
            "plan_file_absolute": str(
                logical_output_dir / cell_id / f"selection_plan_{split}.jsonl"
            ),
            "plan_sha256": _sha256_file(path),
            "row_count": row_count,
            "event_id_sequence_sha256": event_hashers[k].hexdigest(),
            "exact_policy_k_event_count": exact_counts[k],
            "candidate_exhausted_event_count": candidate_exhausted_counts[k],
            "prompt_tail_drop_event_count": prompt_drop_counts[k],
            "mean_realized_k": realized_sums[k] / row_count if row_count else 0.0,
            "ready": row_count > 0,
        }
        _write_json(output_dir / cell_id / "summary.json", summary)
        summaries.append(summary)
    return summaries


def _parse_selector_plans(values: Iterable[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        selector, separator, raw_path = str(value).partition("=")
        selector = selector.strip()
        raw_path = raw_path.strip()
        if not separator or not selector or not raw_path:
            raise PrefixLatticeError(
                f"--selector-plan must be SELECTOR=PATH, got {value!r}"
            )
        if selector in out:
            raise PrefixLatticeError(f"duplicate selector plan: {selector}")
        out[selector] = Path(raw_path)
    return out


def _capacity_level(k: int) -> str:
    return f"prefix_k{k:02d}"


def _cell_id(selector: str, k: int) -> str:
    return f"{selector}__{_capacity_level(k)}"


def _iter_jsonl(handle: TextIO, *, artifact: str) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrefixLatticeError(
                f"invalid JSON at {artifact}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PrefixLatticeError(f"{artifact}:{line_number} is not an object")
        yield row


def _int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise PrefixLatticeError(f"{context} must be an integer array")
    return [int(item) for item in value]


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PrefixLatticeError(f"{context} must be a string array")
    return [str(item) for item in value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _promote_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise PrefixLatticeError(f"refusing to replace existing output: {target}")
        backup = target.with_name(f".{target.name}.old.{os.getpid()}")
        if backup.exists():
            raise PrefixLatticeError(f"backup directory already exists: {backup}")
        target.replace(backup)
    try:
        staging.replace(target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
