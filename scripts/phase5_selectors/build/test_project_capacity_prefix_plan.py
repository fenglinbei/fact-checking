from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.build.project_capacity_prefix_plan import (
    PrefixPlanError,
    project_prefix_plan,
    project_prefix_row,
)


def _trace() -> dict:
    return {
        "event_id": "event-1",
        "candidate_pool": [
            {"candidate_uid": "u0", "evidence_id": "E0", "text": "zero"},
            {"candidate_uid": "u1", "evidence_id": "E1", "text": "one"},
            {"candidate_uid": "u2", "evidence_id": "E2", "text": "two"},
        ],
        "selector_full_ordered_indices": [2, 0, 1],
    }


def _build(*, visible_count: int, was_truncated: bool) -> dict:
    trace = _trace()
    requested = [trace["candidate_pool"][idx] for idx in [2, 0, 1]]
    return {
        "event_id": "event-1",
        "candidates": requested,
        "evidence_count_before": 3,
        "prompt_evidence_selected_count_before_prompt_truncation": 3,
        "evidence_count": visible_count,
        "was_truncated": was_truncated,
        "evidence_text_truncated": False,
        "prompt_token_count": 100,
    }


def test_project_prefix_row_freezes_visible_suffix_drop() -> None:
    plan = project_prefix_row(
        trace_row=_trace(),
        build_row=_build(visible_count=2, was_truncated=True),
        requested_prefix_k=3,
        trace_order_field="selector_full_ordered_indices",
    )

    assert plan["requested_prefix_k"] == 3
    assert plan["available_prefix_k"] == 3
    assert plan["prompt_feasible_prefix_k"] == 2
    assert plan["selected_indices"] == [2, 0]
    assert plan["selected_candidate_uids"] == ["u2", "u0"]
    assert plan["selected_evidence_ids"] == ["E2", "E0"]
    assert plan["prompt_tail_drop_count"] == 1
    assert plan["exact_policy_k"] is False
    assert plan["candidate_exhausted"] is False


def test_project_prefix_row_distinguishes_candidate_exhaustion() -> None:
    trace = _trace()
    build = _build(visible_count=3, was_truncated=False)
    plan = project_prefix_row(
        trace_row=trace,
        build_row=build,
        requested_prefix_k=5,
        trace_order_field="selector_full_ordered_indices",
    )

    assert plan["available_prefix_k"] == 3
    assert plan["prompt_feasible_prefix_k"] == 3
    assert plan["candidate_exhausted"] is True
    assert plan["exact_policy_k"] is False
    assert plan["prompt_tail_drop_count"] == 0


def test_project_prefix_row_rejects_nonprefix_and_text_truncation() -> None:
    nonprefix = _build(visible_count=2, was_truncated=True)
    nonprefix["candidates"][0], nonprefix["candidates"][1] = (
        nonprefix["candidates"][1],
        nonprefix["candidates"][0],
    )
    with pytest.raises(PrefixPlanError, match="not the requested ordered prefix"):
        project_prefix_row(
            trace_row=_trace(),
            build_row=nonprefix,
            requested_prefix_k=3,
            trace_order_field="selector_full_ordered_indices",
        )

    text_truncated = _build(visible_count=1, was_truncated=True)
    text_truncated["evidence_text_truncated"] = True
    with pytest.raises(PrefixPlanError, match="must be exactly false"):
        project_prefix_row(
            trace_row=_trace(),
            build_row=text_truncated,
            requested_prefix_k=3,
            trace_order_field="selector_full_ordered_indices",
        )


def test_project_prefix_plan_streams_aligned_rows_and_writes_summary(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    build_path = tmp_path / "build.jsonl"
    plan_path = tmp_path / "plan.jsonl"
    trace_path.write_text(json.dumps(_trace()) + "\n", encoding="utf-8")
    build_path.write_text(
        json.dumps(_build(visible_count=2, was_truncated=True)) + "\n",
        encoding="utf-8",
    )

    summary = project_prefix_plan(
        source_trace_path=trace_path,
        source_build_path=build_path,
        output_plan_path=plan_path,
        requested_prefix_k=3,
    )

    row = json.loads(plan_path.read_text(encoding="utf-8"))
    assert row["selected_indices"] == [2, 0]
    assert summary["row_count"] == 1
    assert summary["prompt_tail_drop_event_count"] == 1
    assert summary["exact_policy_k_event_count"] == 0
    assert summary["output_plan_sha256"]
