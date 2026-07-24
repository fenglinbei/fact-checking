from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.project_role_rescue_prompt_feasible import (
    ProjectionError,
    materialize_projection,
    project_cell,
    project_row,
)


def test_project_row_accepts_diagnostic_tail_drop_and_reindexes() -> None:
    trace = _trace("e1")
    build = _build("e1", trace, visible_count=2, was_truncated=True)
    row = project_row(trace_row=trace, build_row=build)

    assert row["selected_candidate_uids"] == ["u0", "u1"]
    assert row["selected_indices"] == [0, 1]
    assert [item["candidate_idx"] for item in row["candidate_pool"]] == [0, 1]
    assert [step["step"] for step in row["mrec_steps"]] == [1, 2]
    assert row["intrinsic_role_rescue_metadata"] == trace["role_rescue_metadata"]
    meta = row["realization_metadata"]
    assert meta["diagnostic_was_truncated"] is True
    assert meta["prompt_tail_drop_count"] == 1
    assert meta["visible_prefix_verified"] is True
    assert set(row["role_rescue_metadata"]["visible_realized_atom_role_slots"]) == {
        "cor:A1|support"
    }


def test_project_row_rejects_text_truncation_and_nonprefix() -> None:
    trace = _trace("e1")
    text_truncated = _build("e1", trace, visible_count=2, was_truncated=True)
    text_truncated["evidence_text_truncated"] = True
    with pytest.raises(ProjectionError, match="evidence_text_truncated"):
        project_row(trace_row=trace, build_row=text_truncated)

    nonprefix = _build("e1", trace, visible_count=2, was_truncated=True)
    nonprefix["candidates"][1]["candidate_uid"] = "u2"
    with pytest.raises(ProjectionError, match="not selected prefix"):
        project_row(trace_row=trace, build_row=nonprefix)


def test_project_cell_joins_shuffled_diagnostic_rows_by_event_id(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    build_path = tmp_path / "build.jsonl"
    output_path = tmp_path / "out.jsonl"
    traces = [_trace("e1"), _trace("e2")]
    builds = [
        _build("e2", traces[1], visible_count=3, was_truncated=False),
        _build("e1", traces[0], visible_count=2, was_truncated=True),
    ]
    _write_jsonl(trace_path, traces)
    _write_jsonl(build_path, builds)

    summary = project_cell(
        cell_id="full",
        source_trace_path=trace_path,
        source_build_path=build_path,
        output_trace_path=output_path,
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["event_id"] for row in rows] == ["e1", "e2"]
    assert [row["selected_count"] for row in rows] == [2, 3]
    assert summary["projected_drop_event_count"] == 1


def test_project_row_can_apply_shared_prompt_feasible_count() -> None:
    trace = _trace("e1")
    build = _build("e1", trace, visible_count=3, was_truncated=False)
    row = project_row(
        trace_row=trace,
        build_row=build,
        visible_count_cap=2,
    )

    assert row["selected_candidate_uids"] == ["u0", "u1"]
    meta = row["realization_metadata"]
    assert meta["diagnostic_visible_count"] == 3
    assert meta["shared_prompt_feasible_count"] == 2
    assert meta["shared_count_tail_drop_count"] == 1
    assert meta["diagnostic_surface_matches_projection"] is False


def test_materializer_matches_visible_count_across_requested_cells(tmp_path: Path) -> None:
    role_dir = tmp_path / "roles"
    build_root = tmp_path / "diagnostic"
    output_dir = tmp_path / "projected"
    traces = {}
    manifest_cells = {}
    for cell, visible_count in (("random", 3), ("full", 2)):
        trace = _trace("e1")
        trace_path = role_dir / cell / "selection_trace_val.jsonl"
        build_path = build_root / cell / "build" / "build_val.jsonl"
        _write_jsonl(trace_path, [trace])
        _write_jsonl(
            build_path,
            [_build("e1", trace, visible_count=visible_count, was_truncated=visible_count < 3)],
        )
        traces[cell] = trace_path
        manifest_cells[cell] = {"trace": str(trace_path)}
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / "manifest.json").write_text(
        json.dumps({"row_count": 1, "cells": manifest_cells}),
        encoding="utf-8",
    )

    manifest = materialize_projection(
        role_dir=role_dir,
        diagnostic_build_root=build_root,
        output_dir=output_dir,
        split="val",
        shared_count_cells=["random", "full"],
    )
    counts = []
    for cell in ("random", "full"):
        row = json.loads(
            (output_dir / cell / "selection_trace_val.jsonl").read_text().strip()
        )
        counts.append(row["selected_count"])
    assert counts == [2, 2]
    assert manifest["shared_count_policy"] == "event_wise_min_diagnostic_visible_count"


def test_materializer_emits_matrix_manifest_and_external_gate(tmp_path: Path) -> None:
    role_dir = tmp_path / "roles"
    build_root = tmp_path / "diagnostic"
    output_dir = tmp_path / "projected"
    trace_path = role_dir / "full" / "selection_trace_val.jsonl"
    build_path = build_root / "full" / "build" / "build_val.jsonl"
    trace = _trace("e1")
    _write_jsonl(trace_path, [trace])
    _write_jsonl(build_path, [_build("e1", trace, visible_count=2, was_truncated=True)])
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / "manifest.json").write_text(
        json.dumps(
            {
                "row_count": 1,
                "cells": {"full": {"trace": str(trace_path)}},
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize_projection(
        role_dir=role_dir,
        diagnostic_build_root=build_root,
        output_dir=output_dir,
        split="val",
        external_cells=["native_gate_anchor"],
    )
    assert manifest["all_ready"] is True
    assert [cell["cell_id"] for cell in manifest["cells"]] == [
        "full",
        "native_gate_anchor",
    ]
    assert manifest["cells"][1]["projection_kind"] == "external_prepare_cell"
    assert isinstance(manifest["cells"], list)


def _trace(event_id: str) -> dict:
    candidates = []
    steps = []
    scores = []
    for idx in range(3):
        candidate = {
            "candidate_uid": f"u{idx}",
            "evidence_id": f"E{idx}",
            "candidate_idx": idx,
            "selector_candidate_idx": idx,
            "source_candidate_idx": 10 + idx,
            "text": f"evidence {idx}",
        }
        candidates.append(candidate)
        steps.append(
            {
                "step": idx + 1,
                "candidate_idx": idx,
                "selector_candidate_idx": idx,
                "candidate_uid": f"u{idx}",
                "cue_text": "atom",
            }
        )
        scores.append({"candidate_idx": idx, "candidate_uid": f"u{idx}"})
    return {
        "schema_version": "role_rescue_trace_v0_1",
        "event_id": event_id,
        "selector_name": "role_rescue_full_v0_1",
        "candidate_pool": candidates,
        "candidate_scores": scores,
        "selector_ordered_indices": [0, 1, 2],
        "display_ordered_indices": [0, 1, 2],
        "selected_indices": [0, 1, 2],
        "selected_candidates": deepcopy(candidates),
        "selected_candidate_uids": ["u0", "u1", "u2"],
        "selected_count": 3,
        "mrec_steps": steps,
        "role_rescue_metadata": {
            "selected_source_indices": [10, 11, 12],
            "realized_atom_role_slots": {
                "cor:A1|support": {"source_candidate_idx": 11, "candidate_uid": "u1"},
                "ctx:A1": {"source_candidate_idx": 12, "candidate_uid": "u2"},
            },
        },
    }


def _build(
    event_id: str, trace: dict, *, visible_count: int, was_truncated: bool
) -> dict:
    return {
        "event_id": event_id,
        "candidates": deepcopy(trace["candidate_pool"]),
        "evidence_count": visible_count,
        "evidence_count_before": len(trace["candidate_pool"]),
        "was_truncated": was_truncated,
        "evidence_text_truncated": False,
        "prompt": f"prompt for {event_id} with {visible_count}",
        "prompt_input_ids": [1, 2, visible_count],
        "prompt_token_count": 3,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
