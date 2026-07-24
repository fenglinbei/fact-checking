from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.phase5_selectors.analyze.decide_structure_only_reservation_branch import (
    EXPECTED_CHECKPOINT,
    EXPECTED_EVENT_COUNT,
    EXPECTED_EVENT_SEQUENCE_SHA256,
    EXPECTED_VO_ADAPTER_SHA256,
    EXPECTED_VS_ADAPTER_SHA256,
    DecisionContractError,
    decide_branch,
)


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/phase5_selectors/analyze/decide_structure_only_reservation_branch.py"


def _metric(macro_f1: float, *, accuracy: float = 0.4, loss: float = 1.5) -> dict:
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "eval_ce_loss": loss,
        "num_samples": EXPECTED_EVENT_COUNT,
    }


def _verifier(adapter_sha: str, metrics: dict) -> dict:
    return {
        "adapter_sha256": adapter_sha,
        "checkpoint": EXPECTED_CHECKPOINT,
        "event_count": EXPECTED_EVENT_COUNT,
        "event_id_sequence_sha256": EXPECTED_EVENT_SEQUENCE_SHA256,
        "metrics": metrics,
    }


def _summaries(
    *,
    delta_vs: float = 0.010,
    delta_vo: float = 0.010,
    delta_rs: float = 0.010,
) -> tuple[dict, dict]:
    vs_o = 0.34
    vs_s = vs_o + delta_vs
    vo_o = 0.35
    vo_s = vo_o + delta_vo
    vo_r = vo_s - delta_rs
    stateful_metric = _metric(vo_s, accuracy=0.42, loss=1.4)
    matched_mean = (vs_s + vo_o) / 2.0
    crossed_mean = (vs_o + vo_s) / 2.0

    os_summary = {
        "schema_version": "structure-only-matched-verifier-crossover-summary-v0.1",
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support",
        "split": "val",
        "checkpoint": EXPECTED_CHECKPOINT,
        "event_count": EXPECTED_EVENT_COUNT,
        "event_id_sequence_sha256": EXPECTED_EVENT_SEQUENCE_SHA256,
        "prompt_cells": {"O": "one_shot__fixed5", "S": "stateful__fixed5"},
        "verifiers": {
            "V_S": _verifier(
                EXPECTED_VS_ADAPTER_SHA256,
                {
                    "one_shot__fixed5": _metric(vs_o),
                    "stateful__fixed5": _metric(vs_s),
                },
            ),
            "V_O": _verifier(
                EXPECTED_VO_ADAPTER_SHA256,
                {
                    "one_shot__fixed5": _metric(vo_o),
                    "stateful__fixed5": copy.deepcopy(stateful_metric),
                },
            ),
        },
        "macro_f1_matrix": {
            "V_S": {"O": vs_o, "S": vs_s},
            "V_O": {"O": vo_o, "S": vo_s},
        },
        "contrasts": {
            "prompt_S_minus_O_under_V_S": delta_vs,
            "prompt_S_minus_O_under_V_O": delta_vo,
            "verifier_V_O_minus_V_S_on_O": vo_o - vs_o,
            "verifier_V_O_minus_V_S_on_S": vo_s - vs_s,
            "matched_mean": matched_mean,
            "crossed_mean": crossed_mean,
            "matched_mean_minus_crossed_mean": matched_mean - crossed_mean,
            "difference_in_differences": delta_vs - delta_vo,
        },
        "interpretation_contract": {
            "causal_claim_allowed": False,
            "significance_included": False,
        },
    }
    rs_summary = {
        "schema_version": "vo-retrieval-stateful-diagnostic-summary-v0.1",
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support_single_verifier",
        "split": "val",
        "checkpoint": EXPECTED_CHECKPOINT,
        "event_count": EXPECTED_EVENT_COUNT,
        "event_id_sequence_sha256": EXPECTED_EVENT_SEQUENCE_SHA256,
        "input_cells": {"R": "retrieval__fixed5", "S": "stateful__fixed5"},
        "verifier": _verifier(
            EXPECTED_VO_ADAPTER_SHA256,
            {
                "retrieval__fixed5": _metric(vo_r),
                "stateful__fixed5": copy.deepcopy(stateful_metric),
            },
        ),
        "metrics": {
            "V_O_on_R": _metric(vo_r),
            "V_O_on_S": copy.deepcopy(stateful_metric),
        },
        "primary_contrast": {
            "name": "V_O(S)-V_O(R)",
            "metric": "macro_f1",
            "value": delta_rs,
            "higher_is_better": True,
        },
        "contrasts": {
            "V_O(S)-V_O(R)_macro_f1": delta_rs,
            "V_O(S)-V_O(R)_accuracy": 0.02,
            "V_O(S)-V_O(R)_eval_ce_loss": -0.1,
        },
        "interpretation_contract": {
            "matched_training_claim_allowed": False,
            "causal_claim_allowed": False,
            "significance_included": False,
        },
    }
    return os_summary, rs_summary


@pytest.mark.parametrize(
    ("delta_vs", "delta_vo", "delta_rs", "expected"),
    [
        (0.010, 0.008, 0.006, "no_map"),
        (0.010, 0.008, 0.004, "v_r"),
        (0.010, 0.004, 0.020, "paired_seed43"),
        (0.004, 0.010, 0.020, "paired_seed43"),
        (0.005, 0.005, 0.005, "no_map"),
    ],
)
def test_branch_rule(
    delta_vs: float,
    delta_vo: float,
    delta_rs: float,
    expected: str,
) -> None:
    os_summary, rs_summary = _summaries(
        delta_vs=delta_vs,
        delta_vo=delta_vo,
        delta_rs=delta_rs,
    )

    result = decide_branch(os_summary=os_summary, rs_summary=rs_summary)

    assert result["recommended_branch"] == expected
    assert result["status"] == "ready"
    assert result["interpretation"]["margin_is_significance_threshold"] is False


def test_fails_closed_when_cross_summary_stateful_metric_differs() -> None:
    os_summary, rs_summary = _summaries()
    rs_summary["metrics"]["V_O_on_S"]["accuracy"] += 0.01
    rs_summary["verifier"]["metrics"]["stateful__fixed5"]["accuracy"] += 0.01
    rs_summary["contrasts"]["V_O(S)-V_O(R)_accuracy"] += 0.01

    with pytest.raises(DecisionContractError, match="cross_summary.*accuracy"):
        decide_branch(os_summary=os_summary, rs_summary=rs_summary)


def test_fails_closed_when_contrast_does_not_recompute() -> None:
    os_summary, rs_summary = _summaries()
    os_summary["contrasts"]["prompt_S_minus_O_under_V_O"] = 0.2

    with pytest.raises(DecisionContractError, match="inconsistent"):
        decide_branch(os_summary=os_summary, rs_summary=rs_summary)


def test_fails_closed_for_wrong_checkpoint_sha() -> None:
    os_summary, rs_summary = _summaries()
    rs_summary["verifier"]["adapter_sha256"] = "0" * 64

    with pytest.raises(DecisionContractError, match="adapter_sha256"):
        decide_branch(os_summary=os_summary, rs_summary=rs_summary)


def test_cli_returns_blocked_json_and_nonzero_for_invalid_input(tmp_path: Path) -> None:
    os_summary, rs_summary = _summaries()
    os_summary["status"] = "pending"
    os_path = tmp_path / "os.json"
    rs_path = tmp_path / "rs.json"
    os_path.write_text(json.dumps(os_summary), encoding="utf-8")
    rs_path.write_text(json.dumps(rs_summary), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--os-summary",
            str(os_path),
            "--rs-summary",
            str(rs_path),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["recommended_branch"] is None
    assert payload["reason_codes"] == ["input_contract_invalid"]


def test_inputs_are_not_mutated() -> None:
    os_summary, rs_summary = _summaries()
    original_os = copy.deepcopy(os_summary)
    original_rs = copy.deepcopy(rs_summary)

    decide_branch(os_summary=os_summary, rs_summary=rs_summary)

    assert os_summary == original_os
    assert rs_summary == original_rs
