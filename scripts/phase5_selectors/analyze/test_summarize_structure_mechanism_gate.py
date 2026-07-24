from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.phase5_selectors.analyze.summarize_structure_mechanism_gate import (
    MechanismGateSummaryError,
    main,
    render_markdown,
    summarize_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION_PATH = (
    REPO_ROOT / "configs/validation/structure_only_mechanism_gate_v0_1.json"
)
COMPARISON_CELLS = {
    "S_minus_O": ("one_shot__fixed5", "stateful__fixed5"),
    "S_minus_H": ("hard_structure__fixed5", "stateful__fixed5"),
    "H_minus_R": ("retrieval__fixed5", "hard_structure__fixed5"),
    **{
        f"S_minus_shuffle_seed{seed}": (
            f"shuffle_seed{seed}",
            "stateful__fixed5",
        )
        for seed in range(5)
    },
}


def _preregistration() -> dict:
    return json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))


def _row(
    name: str,
    *,
    delta: float = 0.006,
    ci: tuple[float, float] = (0.001, 0.011),
    p_value: float = 0.01,
) -> dict:
    old_cell, new_cell = COMPARISON_CELLS[name]
    old_value = 0.4
    return {
        "name": name,
        "n": 100,
        "old_predictions": (
            f"/tmp/mechanism-gate/{old_cell}/label_token/val_predictions.jsonl"
        ),
        "new_predictions": (
            f"/tmp/mechanism-gate/{new_cell}/label_token/val_predictions.jsonl"
        ),
        "old_metrics": {"macro_f1": old_value},
        "new_metrics": {"macro_f1": old_value + delta},
        "delta": {"macro_f1": delta},
        "bootstrap": {
            "macro_f1": {"samples": 20_000, "ci95": list(ci)},
        },
        "paired_randomization": {
            "macro_f1": {
                "samples": 20_000,
                "p_value_two_sided": p_value,
            }
        },
    }


def _payload(
    *,
    deltas: dict[str, float] | None = None,
    secondary_p: float = 0.01,
    secondary_adjusted_p: float = 0.03,
) -> dict:
    deltas = deltas or {}
    rows = []
    for name in COMPARISON_CELLS:
        delta = deltas.get(name, 0.006)
        ci = (0.001, 0.011) if delta > 0 else (-0.011, 0.003)
        p_value = secondary_p if name in {
            "S_minus_H",
            "H_minus_R",
            "S_minus_shuffle_seed0",
        } else 0.01
        rows.append(_row(name, delta=delta, ci=ci, p_value=p_value))
    secondary_names = [
        "S_minus_H",
        "H_minus_R",
        "S_minus_shuffle_seed0",
    ]
    return {
        "method": {"delta_direction": "new - old"},
        "settings": {
            "bootstrap_samples": 20_000,
            "randomization_samples": 20_000,
            "seed": 20_260_717,
        },
        "comparisons": rows,
        "multiple_testing": {
            "primary_comparison": "S_minus_O",
            "primary_inference": "unadjusted preregistered primary",
            "secondary_family": secondary_names,
            "secondary_method": "Holm step-down",
            "metric": "macro_f1",
            "alpha": 0.05,
            "holm_adjusted_p_values": {
                name: secondary_adjusted_p for name in secondary_names
            },
            "holm_reject": {
                name: secondary_adjusted_p <= 0.05 for name in secondary_names
            },
            "diagnostic_comparisons_excluded_from_family": [
                f"S_minus_shuffle_seed{seed}" for seed in range(1, 5)
            ],
        },
    }


def _results_by_name(summary: dict) -> dict[str, dict]:
    return {row["name"]: row for row in summary["comparisons"]}


def test_all_green_comparisons_retain_full_story() -> None:
    summary = summarize_gate(_payload(), _preregistration())

    results = _results_by_name(summary)
    assert {row["classification"] for row in results.values()} == {"green"}
    assert results["S_minus_O"]["holm_adjusted_p_value"] is None
    assert results["S_minus_H"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert summary["order_robustness"]["positive_seed_count"] == 5
    assert summary["order_robustness"]["robust"] is True
    assert summary["full_story_supported"] is True
    assert summary["dropped_claims"] == []
    assert summary["overall_recommendation"] == (
        "retain_full_story_and_proceed_to_confirmatory_stage"
    )


def test_secondary_numerical_green_is_yellow_when_holm_does_not_reject() -> None:
    summary = summarize_gate(
        _payload(secondary_p=0.04, secondary_adjusted_p=0.12),
        _preregistration(),
    )

    results = _results_by_name(summary)
    assert results["S_minus_O"]["classification"] == "green"
    assert results["S_minus_H"]["classification"] == "yellow"
    assert results["H_minus_R"]["classification"] == "yellow"
    assert results["S_minus_shuffle_seed0"]["classification"] == "yellow"
    assert summary["claims"]["structure_induced_rule_benefit"]["decision"] == "drop"
    assert summary["full_story_supported"] is False


def test_order_claim_requires_four_of_five_positive_deltas() -> None:
    summary = summarize_gate(
        _payload(
            deltas={
                "S_minus_shuffle_seed3": -0.001,
                "S_minus_shuffle_seed4": -0.001,
            }
        ),
        _preregistration(),
    )

    assert summary["order_robustness"]["positive_seed_count"] == 3
    assert summary["order_robustness"]["robust"] is False
    assert summary["order_robustness"]["seed0_green"] is True
    assert summary["claims"]["presentation_order_benefit"]["decision"] == "drop"
    assert summary["full_story_supported"] is False


def test_positive_but_subthreshold_primary_is_yellow_and_drops_state_claim() -> None:
    payload = _payload(deltas={"S_minus_O": 0.004})
    row = next(row for row in payload["comparisons"] if row["name"] == "S_minus_O")
    row["bootstrap"]["macro_f1"]["ci95"] = [0.0001, 0.009]

    summary = summarize_gate(payload, _preregistration())

    result = _results_by_name(summary)["S_minus_O"]
    assert result["classification"] == "yellow"
    assert summary["claims"]["state_conditioned_rescoring_benefit"]["decision"] == "drop"


def test_missing_comparison_fails_closed() -> None:
    payload = _payload()
    payload["comparisons"] = payload["comparisons"][:-1]

    with pytest.raises(MechanismGateSummaryError, match="comparison set"):
        summarize_gate(payload, _preregistration())


def test_global_delta_direction_fails_closed() -> None:
    payload = _payload()
    payload["method"]["delta_direction"] = "old - new"

    with pytest.raises(MechanismGateSummaryError, match="delta_direction"):
        summarize_gate(payload, _preregistration())


def test_swapped_prediction_cells_fail_closed() -> None:
    payload = _payload()
    row = next(row for row in payload["comparisons"] if row["name"] == "S_minus_O")
    row["old_predictions"], row["new_predictions"] = (
        row["new_predictions"],
        row["old_predictions"],
    )

    with pytest.raises(MechanismGateSummaryError, match="direction mismatch"):
        summarize_gate(payload, _preregistration())


def test_incorrect_holm_annotation_fails_closed() -> None:
    payload = _payload()
    payload["multiple_testing"]["holm_adjusted_p_values"]["S_minus_H"] = 0.02

    with pytest.raises(MechanismGateSummaryError, match="Holm annotation"):
        summarize_gate(payload, _preregistration())


def test_inconsistent_delta_fails_closed() -> None:
    payload = _payload()
    row = next(row for row in payload["comparisons"] if row["name"] == "S_minus_O")
    row["delta"]["macro_f1"] = 0.02

    with pytest.raises(MechanismGateSummaryError, match="inconsistent"):
        summarize_gate(payload, _preregistration())


def test_main_writes_json_and_concise_markdown(tmp_path: Path) -> None:
    paired_path = tmp_path / "paired.json"
    preregistration_path = tmp_path / "preregistration.json"
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    paired_path.write_text(json.dumps(_payload()), encoding="utf-8")
    preregistration_path.write_text(
        json.dumps(_preregistration()), encoding="utf-8"
    )

    assert main(
        [
            "--paired-json",
            str(paired_path),
            "--preregistration",
            str(preregistration_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert summary["full_story_supported"] is True
    assert "| S_minus_O |" in markdown
    assert "Full story: **SUPPORTED**" in markdown
    assert render_markdown(summary) == markdown
