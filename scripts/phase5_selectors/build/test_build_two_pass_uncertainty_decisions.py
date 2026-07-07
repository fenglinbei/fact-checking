from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_two_pass_uncertainty_decisions.py")
SPEC = importlib.util.spec_from_file_location("build_two_pass_uncertainty_decisions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
build_two_pass_uncertainty_decisions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_two_pass_uncertainty_decisions)


def test_select_two_pass_decision_keeps_confident_initial_prefix() -> None:
    decision = build_two_pass_uncertainty_decisions._select_decision_from_score_trace(
        event_id="event-1",
        split="val",
        threshold=0.50,
        score_trace=[
            {
                "role": "initial",
                "selected_indices": [0, 1],
                "pred_margin": 0.81,
                "pred_label": "true",
                "prompt_token_count": 320,
                "was_truncated": False,
            },
            {
                "role": "expanded",
                "selected_indices": [0, 1, 2],
                "pred_margin": 0.83,
                "pred_label": "true",
                "prompt_token_count": 410,
                "was_truncated": False,
            },
        ],
    )

    assert decision["selected_indices"] == [0, 1]
    assert decision["initial_indices"] == [0, 1]
    assert decision["prompt_evidence_expanded"] is False
    assert decision["stop_reason"] == "confident_initial"
    assert decision["uncertainty_margin"] == 0.81


def test_select_two_pass_decision_expands_to_shortest_confident_prefix() -> None:
    decision = build_two_pass_uncertainty_decisions._select_decision_from_score_trace(
        event_id="event-2",
        split="val",
        threshold=0.50,
        score_trace=[
            {
                "role": "initial",
                "selected_indices": [0],
                "pred_margin": 0.12,
                "pred_label": "false",
                "prompt_token_count": 260,
                "was_truncated": False,
            },
            {
                "role": "expanded",
                "selected_indices": [0, 1],
                "pred_margin": 0.55,
                "pred_label": "false",
                "prompt_token_count": 360,
                "was_truncated": False,
            },
            {
                "role": "expanded",
                "selected_indices": [0, 1, 2],
                "pred_margin": 0.90,
                "pred_label": "false",
                "prompt_token_count": 460,
                "was_truncated": False,
            },
        ],
    )

    assert decision["selected_indices"] == [0, 1]
    assert decision["initial_indices"] == [0]
    assert decision["prompt_evidence_expanded"] is True
    assert decision["stop_reason"] == "expanded_confident"
    assert decision["uncertainty_margin"] == 0.55


def test_select_two_pass_decision_uses_best_available_when_no_prefix_is_confident() -> None:
    decision = build_two_pass_uncertainty_decisions._select_decision_from_score_trace(
        event_id="event-3",
        split="test",
        threshold=0.80,
        score_trace=[
            {
                "role": "initial",
                "selected_indices": [0],
                "pred_margin": 0.10,
                "pred_label": "half",
                "prompt_token_count": 260,
                "was_truncated": False,
            },
            {
                "role": "expanded",
                "selected_indices": [0, 1],
                "pred_margin": 0.44,
                "pred_label": "half",
                "prompt_token_count": 360,
                "was_truncated": False,
            },
            {
                "role": "expanded",
                "selected_indices": [0, 1, 2],
                "pred_margin": 0.44,
                "pred_label": "half",
                "prompt_token_count": 460,
                "was_truncated": False,
            },
        ],
    )

    assert decision["selected_indices"] == [0, 1]
    assert decision["prompt_evidence_expanded"] is True
    assert decision["stop_reason"] == "best_available"
    assert decision["uncertainty_margin"] == 0.44
