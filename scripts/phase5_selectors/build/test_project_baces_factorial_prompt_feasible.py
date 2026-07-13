from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.build_baces_factorial_traces import (
    CONTROLLER_LEVELS,
    SELECTOR_LEVELS,
    build_event_factorial_rows,
)
from scripts.phase5_selectors.build.project_baces_factorial_prompt_feasible import (
    PROJECTION_VERSION,
    ProjectionError,
    _validate_source_manifest,
    main,
    project_row,
)
from scripts.phase5_selectors.build.test_build_baces_factorial_traces import (
    _artifacts,
)


def test_projection_freezes_realized_prefix_and_recomputes_display() -> None:
    trace = _trace_fixture()
    build = _build_fixture(trace, visible_count=3, was_truncated=True)

    projected = project_row(
        trace_row=trace,
        build_row=build,
        source_trace_path="source-trace.jsonl",
        source_build_path="source-build.jsonl",
    )

    assert projected["controller_selected_indices"] == trace["selected_indices"]
    assert projected["controller_selected_count"] == 5
    assert projected["selected_indices"] == trace["selected_indices"][:3]
    assert projected["selected_candidate_uids"] == trace["selected_candidate_uids"][:3]
    assert projected["selected_count"] == 3
    assert projected["baces_display"]["length"] == 3
    assert projected["baces_display"]["terminal_utility"] == 2
    assert projected["baces_display"]["acquisition_time"] == 2
    assert projected["baces_display"]["token_cost"] == 9
    assert projected["baces_display"]["padded_auc_horizon10"] == 20
    assert len(projected["baces_display_steps"]) == 3
    assert len(projected["mrec_steps"]) == 3
    assert projected["realization_policy"] == PROJECTION_VERSION
    assert projected["realization_metadata"]["prompt_tail_drop_count"] == 2
    assert projected["realization_metadata"]["visible_prefix_verified"] is True


def test_projection_no_drop_preserves_selected_slate_and_display() -> None:
    trace = _trace_fixture()
    build = _build_fixture(trace, visible_count=5, was_truncated=False)

    projected = project_row(
        trace_row=trace,
        build_row=build,
        source_trace_path="source-trace.jsonl",
        source_build_path="source-build.jsonl",
    )

    assert projected["selected_indices"] == trace["selected_indices"]
    assert projected["selected_candidate_uids"] == trace["selected_candidate_uids"]
    assert projected["baces_display"] == trace["baces_display"]
    assert projected["realization_metadata"]["prompt_tail_drop_count"] == 0


def test_projection_rejects_nonprefix_or_evidence_text_truncation() -> None:
    trace = _trace_fixture()
    nonprefix = _build_fixture(trace, visible_count=3, was_truncated=True)
    nonprefix["candidates"][0], nonprefix["candidates"][1] = (
        nonprefix["candidates"][1],
        nonprefix["candidates"][0],
    )
    with pytest.raises(ProjectionError, match="not the controller-selected prefix"):
        project_row(
            trace_row=trace,
            build_row=nonprefix,
            source_trace_path="trace",
            source_build_path="build",
        )

    text_truncated = _build_fixture(trace, visible_count=1, was_truncated=True)
    text_truncated["evidence_text_truncated"] = True
    with pytest.raises(ProjectionError, match="must be exactly false"):
        project_row(
            trace_row=trace,
            build_row=text_truncated,
            source_trace_path="trace",
            source_build_path="build",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("ordered_indices", [1, 0, 2, 3, 4], "selected index aliases disagree"),
        (
            "selected_candidate_uids",
            ["poison", "u2", "u3", "u4", "u5"],
            "selected UID aliases disagree",
        ),
    ],
)
def test_projection_rejects_poisoned_selected_aliases(
    field: str, replacement: list, message: str
) -> None:
    trace = _trace_fixture()
    trace[field] = replacement
    build = _build_fixture(_trace_fixture(), visible_count=3, was_truncated=True)

    with pytest.raises(ProjectionError, match=message):
        project_row(
            trace_row=trace,
            build_row=build,
            source_trace_path="trace",
            source_build_path="build",
        )


def test_source_manifest_requires_complete_unique_ready_factorial_grid() -> None:
    manifest = _manifest_fixture(event_count=2)

    cells, event_count = _validate_source_manifest(manifest)

    assert len(cells) == len(SELECTOR_LEVELS) * len(CONTROLLER_LEVELS) == 18
    assert event_count == 2

    duplicate = copy.deepcopy(manifest)
    duplicate["cells"][-1] = copy.deepcopy(duplicate["cells"][0])
    with pytest.raises(ProjectionError, match="duplicate cell_id"):
        _validate_source_manifest(duplicate)

    wrong_count = copy.deepcopy(manifest)
    wrong_count["cells"][0]["row_count"] = 1
    with pytest.raises(ProjectionError, match="differs from event_count"):
        _validate_source_manifest(wrong_count)

    incomplete = copy.deepcopy(manifest)
    incomplete["cells"].pop()
    incomplete["cell_count"] -= 1
    with pytest.raises(ProjectionError, match="not a complete"):
        _validate_source_manifest(incomplete)


def test_cli_promotes_complete_tree_and_restores_previous_tree_on_late_failure(
    tmp_path: Path,
) -> None:
    factorial_dir = tmp_path / "factorial"
    build_root = tmp_path / "builds"
    output_dir = tmp_path / "projected"
    factorial_dir.mkdir()
    manifest = _manifest_fixture(event_count=1)
    (factorial_dir / "manifest.json").write_text(json.dumps(manifest))
    feature, learned, reference = _artifacts()
    rows = build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )
    last_build_path: Path | None = None
    for (selector, controller), trace in rows.items():
        cell_id = f"{selector}__{controller}"
        trace_dir = factorial_dir / cell_id
        trace_dir.mkdir()
        (trace_dir / "selection_trace_val.jsonl").write_text(
            json.dumps(trace) + "\n"
        )
        build_dir = build_root / cell_id / "build"
        build_dir.mkdir(parents=True)
        build = _build_fixture(
            trace,
            visible_count=len(trace["selected_indices"]),
            was_truncated=False,
        )
        last_build_path = build_dir / "build_val.jsonl"
        last_build_path.write_text(json.dumps(build) + "\n")

    argv = [
        "--factorial-dir",
        str(factorial_dir),
        "--build-root",
        str(build_root),
        "--output-dir",
        str(output_dir),
        "--split",
        "val",
    ]
    assert main(argv) == 0
    frozen_manifest = (output_dir / "manifest.json").read_bytes()

    with pytest.raises(ProjectionError, match="already exists"):
        main(argv)

    assert last_build_path is not None
    poisoned = json.loads(last_build_path.read_text())
    poisoned["evidence_text_truncated"] = True
    last_build_path.write_text(json.dumps(poisoned) + "\n")
    with pytest.raises(ProjectionError, match="must be exactly false"):
        main([*argv, "--overwrite"])
    assert (output_dir / "manifest.json").read_bytes() == frozen_manifest
    assert not list(tmp_path.glob(".projected.tmp.*"))
    assert not list(tmp_path.glob(".projected.backup.*"))


def _trace_fixture() -> dict:
    feature, learned, reference = _artifacts()
    return build_event_factorial_rows(
        feature_row=feature,
        learned_row=learned,
        reference_row=reference,
    )[("retrieval_source", "fixed5")]


def _build_fixture(
    trace: dict, *, visible_count: int, was_truncated: bool
) -> dict:
    candidates = copy.deepcopy(trace["selected_candidates"])
    return {
        "event_id": trace["event_id"],
        "candidates": candidates,
        "evidence_count": visible_count,
        "evidence_count_before": len(candidates),
        "prompt_evidence_selected_count_before_prompt_truncation": len(candidates),
        "was_truncated": was_truncated,
        "evidence_text_truncated": False,
        "prompt_token_count": 100,
    }


def _manifest_fixture(*, event_count: int) -> dict:
    cells = []
    for selector in SELECTOR_LEVELS:
        for controller in CONTROLLER_LEVELS:
            cell_id = f"{selector}__{controller}"
            cells.append(
                {
                    "cell_id": cell_id,
                    "selector_level": selector,
                    "controller_level": controller,
                    "trace_file": f"{cell_id}/selection_trace_val.jsonl",
                    "row_count": event_count,
                    "ready": True,
                }
            )
    return {
        "selector_levels": list(SELECTOR_LEVELS),
        "controller_levels": list(CONTROLLER_LEVELS),
        "cell_count": len(cells),
        "event_count": event_count,
        "all_ready": True,
        "cells": cells,
    }
