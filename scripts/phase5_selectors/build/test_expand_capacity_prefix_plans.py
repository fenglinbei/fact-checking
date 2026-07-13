from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.expand_capacity_prefix_plans import (
    PrefixLatticeError,
    materialize_prefix_lattice,
)


def _write_max_plan(path: Path, *, selector: str) -> None:
    rows = [
        {
            "schema_version": "capacity_prefix_selection_plan_v0_1",
            "event_id": "event-1",
            "trace_order_field": "selector_full_ordered_indices",
            "requested_prefix_k": 10,
            "available_prefix_k": 10,
            "prompt_feasible_prefix_k": 7,
            "selected_indices": [0, 1, 2, 3, 4, 5, 6],
            "selected_candidate_uids": [f"{selector}-u{i}" for i in range(7)],
            "selected_evidence_ids": [f"{selector}-E{i}" for i in range(7)],
        },
        {
            "schema_version": "capacity_prefix_selection_plan_v0_1",
            "event_id": "event-2",
            "trace_order_field": "selector_full_ordered_indices",
            "requested_prefix_k": 10,
            "available_prefix_k": 3,
            "prompt_feasible_prefix_k": 3,
            "selected_indices": [2, 1, 0],
            "selected_candidate_uids": [f"{selector}-v{i}" for i in range(3)],
            "selected_evidence_ids": [f"{selector}-V{i}" for i in range(3)],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materialize_prefix_lattice_expands_and_plateaus(tmp_path: Path) -> None:
    baces = tmp_path / "baces.jsonl"
    learned = tmp_path / "learned.jsonl"
    _write_max_plan(baces, selector="baces_exact")
    _write_max_plan(learned, selector="learned_marginal")
    output = tmp_path / "plans"
    output.mkdir()

    manifest = materialize_prefix_lattice(
        selector_plans={
            "baces_exact": baces,
            "learned_marginal": learned,
        },
        output_dir=output,
        logical_output_dir=output,
        split="val",
        min_k=1,
        max_k=10,
        source_controller="ordinal_replay_minmax5_10",
    )

    assert manifest["cell_count"] == 20
    assert manifest["event_count"] == 2
    assert manifest["all_ready"] is True

    k7 = _rows(output / "baces_exact__prefix_k07" / "selection_plan_val.jsonl")
    k8 = _rows(output / "baces_exact__prefix_k08" / "selection_plan_val.jsonl")
    k10 = _rows(output / "baces_exact__prefix_k10" / "selection_plan_val.jsonl")
    assert len(k7[0]["selected_indices"]) == 7
    assert k8[0]["selected_indices"] == k7[0]["selected_indices"]
    assert k8[0]["prompt_tail_drop_count"] == 1
    assert k10[0]["prompt_tail_drop_count"] == 3
    assert k10[1]["selected_indices"] == [2, 1, 0]
    assert k10[1]["candidate_exhausted"] is True
    assert k10[1]["prompt_tail_drop_count"] == 0

    for k in range(1, 10):
        left = _rows(
            output / f"learned_marginal__prefix_k{k:02d}" / "selection_plan_val.jsonl"
        )
        right = _rows(
            output / f"learned_marginal__prefix_k{k + 1:02d}" / "selection_plan_val.jsonl"
        )
        for left_row, right_row in zip(left, right):
            assert left_row["selected_indices"] == right_row["selected_indices"][
                : len(left_row["selected_indices"])
            ]


def test_materialize_prefix_lattice_rejects_duplicate_maximal_prefix(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    _write_max_plan(path, selector="baces_exact")
    rows = _rows(path)
    rows[0]["selected_indices"][1] = rows[0]["selected_indices"][0]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "plans"
    output.mkdir()

    with pytest.raises(PrefixLatticeError, match="contains duplicates"):
        materialize_prefix_lattice(
            selector_plans={"baces_exact": path},
            output_dir=output,
            logical_output_dir=output,
            split="val",
            min_k=1,
            max_k=10,
            source_controller="ordinal_replay_minmax5_10",
        )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
