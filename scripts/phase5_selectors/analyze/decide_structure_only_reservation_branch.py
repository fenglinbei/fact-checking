#!/usr/bin/env python3
"""Choose the next structure-only reservation branch from frozen val summaries.

The decision is deliberately limited to the current LIAR-RAW checkpoint-800
gate.  It reads two completed diagnostic summaries and prints one JSON object;
it never edits experiment artifacts or launches a command.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "structure-only-reservation-branch-decision-v0.1"
OS_SCHEMA_VERSION = "structure-only-matched-verifier-crossover-summary-v0.1"
RS_SCHEMA_VERSION = "vo-retrieval-stateful-diagnostic-summary-v0.1"
EXPECTED_SPLIT = "val"
EXPECTED_CHECKPOINT = "checkpoint-800"
EXPECTED_EVENT_COUNT = 1234
EXPECTED_EVENT_SEQUENCE_SHA256 = (
    "65038f1f222b7d990642970ebf7281434abdb17fe61ec1e14ed0c937e8ee6549"
)
EXPECTED_VS_ADAPTER_SHA256 = (
    "7b7512cd8f5a37d7087be935c3d768db04a29dd3bd479131bd1c5c7681b9374a"
)
EXPECTED_VO_ADAPTER_SHA256 = (
    "24e661e8efec049f19e4427a4488de57ce4dc7aec97e412315029412f8779aa3"
)

# This is a directional guard band, not a significance threshold.  It prevents
# sub-half-point changes from automatically selecting a mechanism ablation.
MIN_DIRECTIONAL_MACRO_F1_MARGIN = 0.005
CONSISTENCY_TOLERANCE = 1e-9
CROSS_SUMMARY_METRIC_TOLERANCE = 1e-6


class DecisionContractError(ValueError):
    """Raised when the inputs do not satisfy the frozen decision contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DecisionContractError(f"missing summary: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DecisionContractError(f"invalid JSON summary: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecisionContractError(f"summary must be a JSON object: {path}")
    return payload


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionContractError(f"{context} must be an object")
    return value


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise DecisionContractError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionContractError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise DecisionContractError(f"{context} must be a finite number")
    return result


def _macro_f1(value: Any, *, context: str) -> float:
    result = _finite_float(value, context=context)
    if not 0.0 <= result <= 1.0:
        raise DecisionContractError(f"{context} must be in [0, 1]")
    return result


def _require_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise DecisionContractError(
            f"{context} must be {expected!r}, got {actual!r}"
        )


def _require_close(
    actual: Any,
    expected: float,
    *,
    context: str,
    tolerance: float = CONSISTENCY_TOLERANCE,
) -> float:
    result = _finite_float(actual, context=context)
    if not math.isclose(
        result,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise DecisionContractError(
            f"{context} is inconsistent: expected {expected:+.12f}, "
            f"got {result:+.12f}"
        )
    return result


def _validate_common_contract(
    summary: Mapping[str, Any],
    *,
    schema_version: str,
    scope: str,
    context: str,
) -> None:
    _require_equal(
        summary.get("schema_version"), schema_version, context=f"{context}.schema_version"
    )
    _require_equal(summary.get("status"), "complete", context=f"{context}.status")
    _require_equal(summary.get("scope"), scope, context=f"{context}.scope")
    _require_equal(summary.get("split"), EXPECTED_SPLIT, context=f"{context}.split")
    _require_equal(
        summary.get("checkpoint"),
        EXPECTED_CHECKPOINT,
        context=f"{context}.checkpoint",
    )
    _require_equal(
        summary.get("event_count"),
        EXPECTED_EVENT_COUNT,
        context=f"{context}.event_count",
    )
    _require_equal(
        summary.get("event_id_sequence_sha256"),
        EXPECTED_EVENT_SEQUENCE_SHA256,
        context=f"{context}.event_id_sequence_sha256",
    )


def _validate_metric_cell(
    cell: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, float | int]:
    sample_count = cell.get("num_samples")
    _require_equal(
        sample_count,
        EXPECTED_EVENT_COUNT,
        context=f"{context}.num_samples",
    )
    return {
        "macro_f1": _macro_f1(cell.get("macro_f1"), context=f"{context}.macro_f1"),
        "accuracy": _macro_f1(cell.get("accuracy"), context=f"{context}.accuracy"),
        "eval_ce_loss": _finite_float(
            cell.get("eval_ce_loss"), context=f"{context}.eval_ce_loss"
        ),
        "num_samples": EXPECTED_EVENT_COUNT,
    }


def _validate_os_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    context = "os_summary"
    _validate_common_contract(
        summary,
        schema_version=OS_SCHEMA_VERSION,
        scope="frozen_val_only_fixed_k5_common_support",
        context=context,
    )
    _require_equal(
        summary.get("prompt_cells"),
        {"O": "one_shot__fixed5", "S": "stateful__fixed5"},
        context=f"{context}.prompt_cells",
    )

    verifiers = _mapping(summary.get("verifiers"), context=f"{context}.verifiers")
    if set(verifiers) != {"V_S", "V_O"}:
        raise DecisionContractError(
            f"{context}.verifiers must contain exactly V_S and V_O"
        )
    verifier_s = _mapping(verifiers["V_S"], context=f"{context}.verifiers.V_S")
    verifier_o = _mapping(verifiers["V_O"], context=f"{context}.verifiers.V_O")
    _require_equal(
        verifier_s.get("adapter_sha256"),
        EXPECTED_VS_ADAPTER_SHA256,
        context=f"{context}.verifiers.V_S.adapter_sha256",
    )
    _require_equal(
        verifier_o.get("adapter_sha256"),
        EXPECTED_VO_ADAPTER_SHA256,
        context=f"{context}.verifiers.V_O.adapter_sha256",
    )
    for verifier_id, verifier in (("V_S", verifier_s), ("V_O", verifier_o)):
        _require_equal(
            verifier.get("checkpoint"),
            EXPECTED_CHECKPOINT,
            context=f"{context}.verifiers.{verifier_id}.checkpoint",
        )
        _require_equal(
            verifier.get("event_count"),
            EXPECTED_EVENT_COUNT,
            context=f"{context}.verifiers.{verifier_id}.event_count",
        )
        _require_equal(
            verifier.get("event_id_sequence_sha256"),
            EXPECTED_EVENT_SEQUENCE_SHA256,
            context=f"{context}.verifiers.{verifier_id}.event_id_sequence_sha256",
        )

    matrix = _mapping(summary.get("macro_f1_matrix"), context=f"{context}.macro_f1_matrix")
    if set(matrix) != {"V_S", "V_O"}:
        raise DecisionContractError(
            f"{context}.macro_f1_matrix must contain exactly V_S and V_O"
        )
    values: dict[str, dict[str, float]] = {}
    for verifier_id in ("V_S", "V_O"):
        row = _mapping(matrix[verifier_id], context=f"{context}.macro_f1_matrix.{verifier_id}")
        if set(row) != {"O", "S"}:
            raise DecisionContractError(
                f"{context}.macro_f1_matrix.{verifier_id} must contain exactly O and S"
            )
        values[verifier_id] = {
            "O": _macro_f1(
                row["O"], context=f"{context}.macro_f1_matrix.{verifier_id}.O"
            ),
            "S": _macro_f1(
                row["S"], context=f"{context}.macro_f1_matrix.{verifier_id}.S"
            ),
        }

    nested_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for verifier_id, verifier in (("V_S", verifier_s), ("V_O", verifier_o)):
        verifier_metrics = _mapping(
            verifier.get("metrics"),
            context=f"{context}.verifiers.{verifier_id}.metrics",
        )
        if set(verifier_metrics) != {"one_shot__fixed5", "stateful__fixed5"}:
            raise DecisionContractError(
                f"{context}.verifiers.{verifier_id}.metrics must contain exactly "
                "one_shot__fixed5 and stateful__fixed5"
            )
        nested_metrics[verifier_id] = {}
        for short_cell, cell_id in (
            ("O", "one_shot__fixed5"),
            ("S", "stateful__fixed5"),
        ):
            nested_cell = _validate_metric_cell(
                _mapping(
                    verifier_metrics[cell_id],
                    context=f"{context}.verifiers.{verifier_id}.metrics.{cell_id}",
                ),
                context=f"{context}.verifiers.{verifier_id}.metrics.{cell_id}",
            )
            _require_close(
                nested_cell["macro_f1"],
                values[verifier_id][short_cell],
                context=f"{context}.{verifier_id}_{short_cell}_matrix_join",
            )
            nested_metrics[verifier_id][short_cell] = nested_cell

    delta_vs = values["V_S"]["S"] - values["V_S"]["O"]
    delta_vo = values["V_O"]["S"] - values["V_O"]["O"]
    matched_mean = (values["V_S"]["S"] + values["V_O"]["O"]) / 2.0
    crossed_mean = (values["V_S"]["O"] + values["V_O"]["S"]) / 2.0
    derived_contrasts = {
        "prompt_S_minus_O_under_V_S": delta_vs,
        "prompt_S_minus_O_under_V_O": delta_vo,
        "verifier_V_O_minus_V_S_on_O": (
            values["V_O"]["O"] - values["V_S"]["O"]
        ),
        "verifier_V_O_minus_V_S_on_S": (
            values["V_O"]["S"] - values["V_S"]["S"]
        ),
        "matched_mean": matched_mean,
        "crossed_mean": crossed_mean,
        "matched_mean_minus_crossed_mean": matched_mean - crossed_mean,
        "difference_in_differences": delta_vs - delta_vo,
    }
    contrasts = _mapping(summary.get("contrasts"), context=f"{context}.contrasts")
    for key, expected in derived_contrasts.items():
        _require_close(
            contrasts.get(key), expected, context=f"{context}.contrasts.{key}"
        )

    interpretation = _mapping(
        summary.get("interpretation_contract"),
        context=f"{context}.interpretation_contract",
    )
    _require_equal(
        interpretation.get("causal_claim_allowed"),
        False,
        context=f"{context}.interpretation_contract.causal_claim_allowed",
    )
    _require_equal(
        interpretation.get("significance_included"),
        False,
        context=f"{context}.interpretation_contract.significance_included",
    )
    return {
        "delta_vs": delta_vs,
        "delta_vo": delta_vo,
        "matched_minus_crossed": derived_contrasts[
            "matched_mean_minus_crossed_mean"
        ],
        "vo_stateful_metrics": nested_metrics["V_O"]["S"],
    }


def _validate_rs_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    context = "rs_summary"
    _validate_common_contract(
        summary,
        schema_version=RS_SCHEMA_VERSION,
        scope="frozen_val_only_fixed_k5_common_support_single_verifier",
        context=context,
    )
    _require_equal(
        summary.get("input_cells"),
        {"R": "retrieval__fixed5", "S": "stateful__fixed5"},
        context=f"{context}.input_cells",
    )
    verifier_o = _mapping(summary.get("verifier"), context=f"{context}.verifier")
    _require_equal(
        verifier_o.get("adapter_sha256"),
        EXPECTED_VO_ADAPTER_SHA256,
        context=f"{context}.verifier.adapter_sha256",
    )
    _require_equal(
        verifier_o.get("checkpoint"),
        EXPECTED_CHECKPOINT,
        context=f"{context}.verifier.checkpoint",
    )
    _require_equal(
        verifier_o.get("event_count"),
        EXPECTED_EVENT_COUNT,
        context=f"{context}.verifier.event_count",
    )
    _require_equal(
        verifier_o.get("event_id_sequence_sha256"),
        EXPECTED_EVENT_SEQUENCE_SHA256,
        context=f"{context}.verifier.event_id_sequence_sha256",
    )

    metrics = _mapping(summary.get("metrics"), context=f"{context}.metrics")
    if set(metrics) != {"V_O_on_R", "V_O_on_S"}:
        raise DecisionContractError(
            f"{context}.metrics must contain exactly V_O_on_R and V_O_on_S"
        )
    metrics_r = _validate_metric_cell(
        _mapping(metrics["V_O_on_R"], context=f"{context}.metrics.V_O_on_R"),
        context=f"{context}.metrics.V_O_on_R",
    )
    metrics_s = _validate_metric_cell(
        _mapping(metrics["V_O_on_S"], context=f"{context}.metrics.V_O_on_S"),
        context=f"{context}.metrics.V_O_on_S",
    )
    delta = float(metrics_s["macro_f1"]) - float(metrics_r["macro_f1"])

    verifier_metrics = _mapping(
        verifier_o.get("metrics"), context=f"{context}.verifier.metrics"
    )
    if set(verifier_metrics) != {"retrieval__fixed5", "stateful__fixed5"}:
        raise DecisionContractError(
            f"{context}.verifier.metrics must contain exactly retrieval__fixed5 "
            "and stateful__fixed5"
        )
    for short_name, cell_id, top_level in (
        ("R", "retrieval__fixed5", metrics_r),
        ("S", "stateful__fixed5", metrics_s),
    ):
        nested = _validate_metric_cell(
            _mapping(
                verifier_metrics[cell_id],
                context=f"{context}.verifier.metrics.{cell_id}",
            ),
            context=f"{context}.verifier.metrics.{cell_id}",
        )
        for metric_name in ("macro_f1", "accuracy", "eval_ce_loss"):
            _require_close(
                nested[metric_name],
                float(top_level[metric_name]),
                context=f"{context}.verifier_{short_name}_{metric_name}_join",
            )

    primary = _mapping(
        summary.get("primary_contrast"), context=f"{context}.primary_contrast"
    )
    _require_equal(
        primary.get("name"), "V_O(S)-V_O(R)", context=f"{context}.primary_contrast.name"
    )
    _require_equal(
        primary.get("metric"), "macro_f1", context=f"{context}.primary_contrast.metric"
    )
    _require_equal(
        primary.get("higher_is_better"),
        True,
        context=f"{context}.primary_contrast.higher_is_better",
    )
    _require_close(
        primary.get("value"), delta, context=f"{context}.primary_contrast.value"
    )
    contrasts = _mapping(summary.get("contrasts"), context=f"{context}.contrasts")
    _require_close(
        contrasts.get("V_O(S)-V_O(R)_macro_f1"),
        delta,
        context=f"{context}.contrasts.V_O(S)-V_O(R)_macro_f1",
    )
    _require_close(
        contrasts.get("V_O(S)-V_O(R)_accuracy"),
        float(metrics_s["accuracy"]) - float(metrics_r["accuracy"]),
        context=f"{context}.contrasts.V_O(S)-V_O(R)_accuracy",
    )
    _require_close(
        contrasts.get("V_O(S)-V_O(R)_eval_ce_loss"),
        float(metrics_s["eval_ce_loss"]) - float(metrics_r["eval_ce_loss"]),
        context=f"{context}.contrasts.V_O(S)-V_O(R)_eval_ce_loss",
    )

    interpretation = _mapping(
        summary.get("interpretation_contract"),
        context=f"{context}.interpretation_contract",
    )
    for key in (
        "matched_training_claim_allowed",
        "causal_claim_allowed",
        "significance_included",
    ):
        _require_equal(
            interpretation.get(key),
            False,
            context=f"{context}.interpretation_contract.{key}",
        )
    return {"delta": delta, "vo_stateful_metrics": metrics_s}


def decide_branch(
    *,
    os_summary: Mapping[str, Any],
    rs_summary: Mapping[str, Any],
) -> dict[str, Any]:
    os_values = _validate_os_summary(os_summary)
    rs_values = _validate_rs_summary(rs_summary)

    for metric_name in ("macro_f1", "accuracy", "eval_ce_loss"):
        _require_close(
            rs_values["vo_stateful_metrics"][metric_name],
            float(os_values["vo_stateful_metrics"][metric_name]),
            context=f"cross_summary.V_O_on_stateful.{metric_name}",
            tolerance=CROSS_SUMMARY_METRIC_TOLERANCE,
        )

    delta_vs = float(os_values["delta_vs"])
    delta_vo = float(os_values["delta_vo"])
    delta_rs = float(rs_values["delta"])
    margin = MIN_DIRECTIONAL_MACRO_F1_MARGIN

    if delta_vs < margin or delta_vo < margin:
        branch = "paired_seed43"
        reason_codes = ["os_cross_verifier_direction_not_clear"]
        reasons = [
            "S must exceed O by at least 0.005 Macro-F1 under both V_S and V_O; "
            f"observed V_S={delta_vs:+.6f}, V_O={delta_vo:+.6f}.",
            "Repeat the matched S/O pair at seed 43 before spending a full run on "
            "a downstream mechanism ablation.",
        ]
    elif delta_rs >= margin:
        branch = "no_map"
        reason_codes = [
            "os_cross_verifier_direction_clear",
            "structure_exceeds_retrieval_under_frozen_vo",
        ]
        reasons = [
            "S exceeds O by at least 0.005 Macro-F1 under both frozen verifiers "
            f"(V_S={delta_vs:+.6f}, V_O={delta_vo:+.6f}).",
            "On the same frozen V_O and common support, S also exceeds R by at least "
            f"0.005 Macro-F1 ({delta_rs:+.6f}); the no-map ablation is therefore the "
            "next unresolved mechanism question.",
        ]
    else:
        branch = "v_r"
        reason_codes = [
            "os_cross_verifier_direction_clear",
            "structure_does_not_clear_retrieval_guard_band_under_frozen_vo",
        ]
        reasons = [
            "S exceeds O by at least 0.005 Macro-F1 under both frozen verifiers "
            f"(V_S={delta_vs:+.6f}, V_O={delta_vo:+.6f}).",
            "S does not exceed R by the 0.005 Macro-F1 guard band under V_O "
            f"({delta_rs:+.6f}); train V_R to distinguish retrieval strength from "
            "verifier/input mismatch before attempting the no-map ablation.",
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "recommended_branch": branch,
        "reason_codes": reason_codes,
        "reasons": reasons,
        "observed_macro_f1_contrasts": {
            "S_minus_O_under_V_S": delta_vs,
            "S_minus_O_under_V_O": delta_vo,
            "S_minus_R_under_V_O": delta_rs,
            "matched_mean_minus_crossed_mean": float(
                os_values["matched_minus_crossed"]
            ),
        },
        "thresholds": {
            "minimum_directional_macro_f1_margin": margin,
            "consistency_tolerance": CONSISTENCY_TOLERANCE,
            "cross_summary_metric_tolerance": CROSS_SUMMARY_METRIC_TOLERANCE,
        },
        "interpretation": {
            "claim_level": "directional_validation_only",
            "margin_is_significance_threshold": False,
            "test_split_used": False,
        },
    }


def _blocked_payload(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "recommended_branch": None,
        "reason_codes": ["input_contract_invalid"],
        "reasons": [reason],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os-summary", required=True)
    parser.add_argument("--rs-summary", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = decide_branch(
            os_summary=_load_json(Path(args.os_summary)),
            rs_summary=_load_json(Path(args.rs_summary)),
        )
    except DecisionContractError as exc:
        print(json.dumps(_blocked_payload(str(exc)), ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
