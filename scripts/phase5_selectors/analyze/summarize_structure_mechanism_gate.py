#!/usr/bin/env python3
"""Turn preregistered paired statistics into the Stage-C mechanism verdict.

The input is the JSON produced by ``paired_significance.py`` after
``annotate_paired_significance_holm.py`` has added the preregistered Holm
family.  This script intentionally fails closed: it refuses incomplete
comparison sets, reversed prediction paths, non-preregistered resampling
settings, and inconsistent Holm annotations instead of guessing a verdict.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "structure-only-mechanism-gate-summary-v0.1"
EXPECTED_PREREGISTRATION_SCHEMA = (
    "structure_only_mechanism_gate_preregistration_v0_1"
)
EXPECTED_PRIMARY = "S_minus_O"
EXPECTED_SECONDARY = (
    "S_minus_H",
    "H_minus_R",
    "S_minus_shuffle_seed0",
)
EXPECTED_DIAGNOSTICS = tuple(
    f"S_minus_shuffle_seed{seed}" for seed in range(1, 5)
)
EXPECTED_GREEN_RULE = (
    "macro-F1 delta >= 0.005 and paired-bootstrap 95% CI lower bound > 0"
)
EXPECTED_YELLOW_RULE = "macro-F1 delta > 0 but the green rule is not met"
EXPECTED_RED_RULE = "macro-F1 delta <= 0"
GREEN_MIN_DELTA = 0.005
ORDER_SEED_NAMES = tuple(
    f"S_minus_shuffle_seed{seed}" for seed in range(5)
)
ORDER_MIN_POSITIVE = 4
CLAIM_ORDER = (
    "structure_induced_rule_benefit",
    "learned_utility_benefit",
    "state_conditioned_rescoring_benefit",
    "presentation_order_benefit",
)


class MechanismGateSummaryError(ValueError):
    """Raised when the preregistration or paired statistics are inconsistent."""


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    old_cell: str
    new_cell: str
    family: str


@dataclass(frozen=True)
class GateContract:
    scope: str
    metric: str
    alpha: float
    bootstrap_samples: int
    randomization_samples: int
    seed: int
    registered_cells: tuple[str, ...]
    comparisons: tuple[ComparisonSpec, ...]

    @property
    def comparison_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.comparisons)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-json", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(list(argv) if argv is not None else None)


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MechanismGateSummaryError(f"{context} must be an object")
    return value


def _require_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise MechanismGateSummaryError(f"{context} must be an array")
    return value


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise MechanismGateSummaryError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MechanismGateSummaryError(
            f"{context} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise MechanismGateSummaryError(f"{context} must be a finite number")
    return result


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise MechanismGateSummaryError(f"{context} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MechanismGateSummaryError(
            f"{context} must be a positive integer"
        ) from exc
    if result <= 0 or result != value:
        raise MechanismGateSummaryError(f"{context} must be a positive integer")
    return result


def _comparison_specs(
    preregistration: Mapping[str, Any],
) -> tuple[ComparisonSpec, ...]:
    comparisons = _require_mapping(
        preregistration.get("comparisons"), context="preregistration.comparisons"
    )
    primary_rows = _require_list(
        comparisons.get("primary"), context="comparisons.primary"
    )
    secondary_rows = _require_list(
        comparisons.get("secondary_holm_family"),
        context="comparisons.secondary_holm_family",
    )
    diagnostic_names = _require_list(
        comparisons.get("shuffle_robustness_diagnostics"),
        context="comparisons.shuffle_robustness_diagnostics",
    )

    if len(primary_rows) != 1:
        raise MechanismGateSummaryError(
            "the v0.1 gate must register exactly one primary comparison"
        )

    def parse_rows(rows: Sequence[Any], family: str) -> list[ComparisonSpec]:
        result = []
        for index, raw in enumerate(rows):
            row = _require_mapping(raw, context=f"comparisons.{family}[{index}]")
            try:
                name = str(row["name"])
                old_cell = str(row["old"])
                new_cell = str(row["new"])
            except KeyError as exc:
                raise MechanismGateSummaryError(
                    f"comparisons.{family}[{index}] is missing {exc.args[0]}"
                ) from exc
            if not name or not old_cell or not new_cell or old_cell == new_cell:
                raise MechanismGateSummaryError(
                    f"comparisons.{family}[{index}] has invalid name/old/new"
                )
            result.append(ComparisonSpec(name, old_cell, new_cell, family))
        return result

    specs = parse_rows(primary_rows, "primary")
    specs.extend(parse_rows(secondary_rows, "secondary_holm_family"))

    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise MechanismGateSummaryError("preregistered comparison names are duplicated")
    shuffle_zero = by_name.get("S_minus_shuffle_seed0")
    if shuffle_zero is None:
        raise MechanismGateSummaryError(
            "secondary family is missing S_minus_shuffle_seed0"
        )
    for value in diagnostic_names:
        name = str(value)
        if not name:
            raise MechanismGateSummaryError("diagnostic comparison name is empty")
        suffix = name.removeprefix("S_minus_shuffle_seed")
        if not suffix.isdigit():
            raise MechanismGateSummaryError(
                f"cannot infer shuffle cell direction for diagnostic {name}"
            )
        if shuffle_zero.old_cell.count("seed0") != 1:
            raise MechanismGateSummaryError(
                "cannot derive diagnostic shuffle cells from the seed0 cell ID"
            )
        specs.append(
            ComparisonSpec(
                name=name,
                old_cell=shuffle_zero.old_cell.replace(
                    "seed0", f"seed{int(suffix)}"
                ),
                new_cell=shuffle_zero.new_cell,
                family="shuffle_robustness_diagnostic",
            )
        )

    names = tuple(spec.name for spec in specs)
    expected = (EXPECTED_PRIMARY, *EXPECTED_SECONDARY, *EXPECTED_DIAGNOSTICS)
    if names != expected:
        raise MechanismGateSummaryError(
            "preregistered comparisons do not match the v0.1 gate: "
            f"expected {expected}, got {names}"
        )
    return tuple(specs)


def parse_preregistration(preregistration: Mapping[str, Any]) -> GateContract:
    schema = preregistration.get("schema_version")
    if schema != EXPECTED_PREREGISTRATION_SCHEMA:
        raise MechanismGateSummaryError(
            "unexpected preregistration schema: "
            f"expected {EXPECTED_PREREGISTRATION_SCHEMA!r}, got {schema!r}"
        )
    decision = _require_mapping(
        preregistration.get("decision_rule"), context="decision_rule"
    )
    expected_rules = {
        "green": EXPECTED_GREEN_RULE,
        "yellow": EXPECTED_YELLOW_RULE,
        "red": EXPECTED_RED_RULE,
    }
    for key, expected in expected_rules.items():
        if decision.get(key) != expected:
            raise MechanismGateSummaryError(
                f"decision_rule.{key} does not match the implemented v0.1 rule"
            )
    if decision.get("secondary_green_requires_holm") is not True:
        raise MechanismGateSummaryError(
            "decision_rule.secondary_green_requires_holm must be true"
        )
    if decision.get("order_robustness") != (
        "at least four of five shuffle-seed macro-F1 deltas must be positive "
        "for a robust order claim"
    ):
        raise MechanismGateSummaryError(
            "decision_rule.order_robustness does not match the implemented 4-of-5 rule"
        )

    inference = _require_mapping(
        preregistration.get("inference"), context="inference"
    )
    metric = str(inference.get("primary_metric") or "")
    if metric != "macro_f1":
        raise MechanismGateSummaryError(
            f"the v0.1 gate requires primary_metric=macro_f1, got {metric!r}"
        )
    alpha = _finite_float(inference.get("alpha"), context="inference.alpha")
    if not 0.0 < alpha < 1.0:
        raise MechanismGateSummaryError("inference.alpha must be in (0, 1)")
    bootstrap_samples = _positive_int(
        inference.get("bootstrap_samples"), context="inference.bootstrap_samples"
    )
    randomization_samples = _positive_int(
        inference.get("randomization_samples"),
        context="inference.randomization_samples",
    )
    seed = _positive_int(inference.get("seed"), context="inference.seed")

    selector = _require_mapping(
        preregistration.get("selector_contract"), context="selector_contract"
    )
    raw_cells = _require_list(selector.get("cells"), context="selector_contract.cells")
    cells = tuple(str(value) for value in raw_cells)
    if not cells or len(set(cells)) != len(cells) or any(not cell for cell in cells):
        raise MechanismGateSummaryError(
            "selector_contract.cells must contain unique non-empty cell IDs"
        )
    specs = _comparison_specs(preregistration)
    registered = set(cells)
    for spec in specs:
        if spec.old_cell not in registered or spec.new_cell not in registered:
            raise MechanismGateSummaryError(
                f"{spec.name} references an unregistered cell: "
                f"{spec.old_cell} -> {spec.new_cell}"
            )
    return GateContract(
        scope=str(preregistration.get("scope") or ""),
        metric=metric,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        seed=seed,
        registered_cells=cells,
        comparisons=specs,
    )


def holm_adjust(p_values: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise MechanismGateSummaryError(f"invalid p-value for {name}: {value}")
        running = max(running, (total - index) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def _prediction_cell(
    row: Mapping[str, Any],
    *,
    side: str,
    registered_cells: Sequence[str],
    comparison: str,
) -> str:
    explicit_key = f"{side}_cell"
    if explicit_key in row:
        cell = str(row[explicit_key])
        if cell not in registered_cells:
            raise MechanismGateSummaryError(
                f"{comparison}.{explicit_key} is not a registered cell: {cell!r}"
            )
        return cell
    path_key = f"{side}_predictions"
    value = row.get(path_key)
    if not isinstance(value, str) or not value:
        raise MechanismGateSummaryError(
            f"{comparison}.{path_key} must be a non-empty path"
        )
    path_parts = tuple(part for part in value.replace("\\", "/").split("/") if part)
    matches = [cell for cell in registered_cells if cell in path_parts]
    if len(matches) != 1:
        raise MechanismGateSummaryError(
            f"cannot identify exactly one registered cell in {comparison}.{path_key}: "
            f"found {matches}"
        )
    return matches[0]


def _metric_result(
    row: Mapping[str, Any],
    *,
    spec: ComparisonSpec,
    contract: GateContract,
    adjusted_p: float | None,
) -> dict[str, Any]:
    metric = contract.metric
    delta_block = _require_mapping(row.get("delta"), context=f"{spec.name}.delta")
    old_block = _require_mapping(
        row.get("old_metrics"), context=f"{spec.name}.old_metrics"
    )
    new_block = _require_mapping(
        row.get("new_metrics"), context=f"{spec.name}.new_metrics"
    )
    delta = _finite_float(delta_block.get(metric), context=f"{spec.name}.delta.{metric}")
    old_value = _finite_float(
        old_block.get(metric), context=f"{spec.name}.old_metrics.{metric}"
    )
    new_value = _finite_float(
        new_block.get(metric), context=f"{spec.name}.new_metrics.{metric}"
    )
    if not math.isclose(delta, new_value - old_value, rel_tol=0.0, abs_tol=1e-12):
        raise MechanismGateSummaryError(
            f"{spec.name} delta is inconsistent with new_metrics - old_metrics"
        )

    bootstrap = _require_mapping(
        row.get("bootstrap"), context=f"{spec.name}.bootstrap"
    )
    bootstrap_metric = _require_mapping(
        bootstrap.get(metric), context=f"{spec.name}.bootstrap.{metric}"
    )
    if _positive_int(
        bootstrap_metric.get("samples"),
        context=f"{spec.name}.bootstrap.{metric}.samples",
    ) != contract.bootstrap_samples:
        raise MechanismGateSummaryError(
            f"{spec.name} bootstrap sample count is not preregistered"
        )
    ci = _require_list(
        bootstrap_metric.get("ci95"), context=f"{spec.name}.bootstrap.{metric}.ci95"
    )
    if len(ci) != 2:
        raise MechanismGateSummaryError(f"{spec.name} bootstrap CI must have two bounds")
    ci_lower = _finite_float(ci[0], context=f"{spec.name}.ci95[0]")
    ci_upper = _finite_float(ci[1], context=f"{spec.name}.ci95[1]")
    if ci_lower > ci_upper:
        raise MechanismGateSummaryError(f"{spec.name} bootstrap CI is reversed")

    randomization = _require_mapping(
        row.get("paired_randomization"),
        context=f"{spec.name}.paired_randomization",
    )
    randomization_metric = _require_mapping(
        randomization.get(metric),
        context=f"{spec.name}.paired_randomization.{metric}",
    )
    if _positive_int(
        randomization_metric.get("samples"),
        context=f"{spec.name}.paired_randomization.{metric}.samples",
    ) != contract.randomization_samples:
        raise MechanismGateSummaryError(
            f"{spec.name} randomization sample count is not preregistered"
        )
    p_value = _finite_float(
        randomization_metric.get("p_value_two_sided"),
        context=f"{spec.name}.paired_randomization.{metric}.p_value_two_sided",
    )
    if not 0.0 <= p_value <= 1.0:
        raise MechanismGateSummaryError(f"{spec.name} p-value must be in [0, 1]")

    requires_holm = spec.family == "secondary_holm_family"
    numerical_green = delta >= GREEN_MIN_DELTA and ci_lower > 0.0
    holm_passed = adjusted_p is not None and adjusted_p <= contract.alpha
    if delta <= 0.0:
        classification = "red"
    elif numerical_green and (not requires_holm or holm_passed):
        classification = "green"
    else:
        classification = "yellow"
    return {
        "name": spec.name,
        "family": spec.family,
        "old_cell": spec.old_cell,
        "new_cell": spec.new_cell,
        "n": _positive_int(row.get("n"), context=f"{spec.name}.n"),
        "old_macro_f1": old_value,
        "new_macro_f1": new_value,
        "delta_macro_f1": delta,
        "bootstrap_ci95": [ci_lower, ci_upper],
        "paired_randomization_p_value_two_sided": p_value,
        "holm_adjusted_p_value": adjusted_p,
        "holm_reject": holm_passed if requires_holm else None,
        "classification": classification,
        "green_checks": {
            "delta_at_least_0_005": delta >= GREEN_MIN_DELTA,
            "bootstrap_ci_lower_above_zero": ci_lower > 0.0,
            "holm_required": requires_holm,
            "holm_passed": holm_passed if requires_holm else None,
        },
    }


def _validate_multiple_testing(
    paired: Mapping[str, Any],
    *,
    rows_by_name: Mapping[str, Mapping[str, Any]],
    contract: GateContract,
) -> dict[str, float]:
    block = _require_mapping(
        paired.get("multiple_testing"), context="multiple_testing"
    )
    if block.get("primary_comparison") != EXPECTED_PRIMARY:
        raise MechanismGateSummaryError(
            "multiple_testing.primary_comparison does not match preregistration"
        )
    if block.get("metric") != contract.metric:
        raise MechanismGateSummaryError(
            "multiple_testing.metric does not match preregistration"
        )
    annotated_alpha = _finite_float(
        block.get("alpha"), context="multiple_testing.alpha"
    )
    if not math.isclose(annotated_alpha, contract.alpha, rel_tol=0.0, abs_tol=1e-15):
        raise MechanismGateSummaryError(
            "multiple_testing.alpha does not match preregistration"
        )
    if tuple(block.get("secondary_family") or ()) != EXPECTED_SECONDARY:
        raise MechanismGateSummaryError(
            "multiple_testing.secondary_family does not match preregistration"
        )
    if tuple(block.get("diagnostic_comparisons_excluded_from_family") or ()) != (
        EXPECTED_DIAGNOSTICS
    ):
        raise MechanismGateSummaryError(
            "multiple_testing diagnostic family does not match preregistration"
        )
    if block.get("secondary_method") != "Holm step-down":
        raise MechanismGateSummaryError(
            "multiple_testing.secondary_method must be Holm step-down"
        )

    raw_p_values = []
    for name in EXPECTED_SECONDARY:
        row = rows_by_name[name]
        randomization = _require_mapping(
            row.get("paired_randomization"), context=f"{name}.paired_randomization"
        )
        metric = _require_mapping(
            randomization.get(contract.metric),
            context=f"{name}.paired_randomization.{contract.metric}",
        )
        value = _finite_float(
            metric.get("p_value_two_sided"),
            context=f"{name}.paired_randomization.{contract.metric}.p_value_two_sided",
        )
        raw_p_values.append((name, value))
    expected_adjusted = holm_adjust(raw_p_values)

    annotated_values = _require_mapping(
        block.get("holm_adjusted_p_values"),
        context="multiple_testing.holm_adjusted_p_values",
    )
    reject_values = _require_mapping(
        block.get("holm_reject"), context="multiple_testing.holm_reject"
    )
    if set(annotated_values) != set(EXPECTED_SECONDARY):
        raise MechanismGateSummaryError(
            "Holm adjusted p-value keys do not match the secondary family"
        )
    if set(reject_values) != set(EXPECTED_SECONDARY):
        raise MechanismGateSummaryError(
            "Holm reject keys do not match the secondary family"
        )
    for name, expected in expected_adjusted.items():
        observed = _finite_float(
            annotated_values[name],
            context=f"multiple_testing.holm_adjusted_p_values.{name}",
        )
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise MechanismGateSummaryError(
                f"Holm annotation for {name} is inconsistent: "
                f"expected {expected}, got {observed}"
            )
        expected_reject = expected <= contract.alpha
        if reject_values[name] is not expected_reject:
            raise MechanismGateSummaryError(
                f"Holm reject annotation for {name} is inconsistent"
            )
    return expected_adjusted


def summarize_gate(
    paired: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    contract = parse_preregistration(preregistration)
    method = _require_mapping(paired.get("method"), context="method")
    if method.get("delta_direction") != "new - old":
        raise MechanismGateSummaryError(
            "paired method must declare delta_direction='new - old'"
        )
    settings = _require_mapping(paired.get("settings"), context="settings")
    expected_settings = {
        "bootstrap_samples": contract.bootstrap_samples,
        "randomization_samples": contract.randomization_samples,
        "seed": contract.seed,
    }
    for key, expected in expected_settings.items():
        if _positive_int(settings.get(key), context=f"settings.{key}") != expected:
            raise MechanismGateSummaryError(
                f"settings.{key} does not match preregistration ({expected})"
            )

    raw_rows = _require_list(paired.get("comparisons"), context="comparisons")
    rows_by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        row = _require_mapping(raw, context=f"comparisons[{index}]")
        name = str(row.get("name") or "")
        if not name:
            raise MechanismGateSummaryError(f"comparisons[{index}] has no name")
        if name in rows_by_name:
            raise MechanismGateSummaryError(f"duplicate comparison: {name}")
        rows_by_name[name] = row
    expected_names = set(contract.comparison_names)
    actual_names = set(rows_by_name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise MechanismGateSummaryError(
            f"paired comparison set is incomplete or undeclared; missing={missing}, "
            f"extra={extra}"
        )

    for spec in contract.comparisons:
        row = rows_by_name[spec.name]
        old_cell = _prediction_cell(
            row,
            side="old",
            registered_cells=contract.registered_cells,
            comparison=spec.name,
        )
        new_cell = _prediction_cell(
            row,
            side="new",
            registered_cells=contract.registered_cells,
            comparison=spec.name,
        )
        if (old_cell, new_cell) != (spec.old_cell, spec.new_cell):
            raise MechanismGateSummaryError(
                f"{spec.name} direction mismatch: expected "
                f"{spec.old_cell} -> {spec.new_cell}, got {old_cell} -> {new_cell}"
            )

    adjusted = _validate_multiple_testing(
        paired, rows_by_name=rows_by_name, contract=contract
    )
    results = []
    for spec in contract.comparisons:
        results.append(
            _metric_result(
                rows_by_name[spec.name],
                spec=spec,
                contract=contract,
                adjusted_p=adjusted.get(spec.name),
            )
        )
    result_by_name = {row["name"]: row for row in results}
    sample_sizes = {int(row["n"]) for row in results}
    if len(sample_sizes) != 1:
        raise MechanismGateSummaryError(
            f"comparisons do not share one sample size: {sorted(sample_sizes)}"
        )
    sample_size = next(iter(sample_sizes))

    positive_order_seeds = [
        name
        for name in ORDER_SEED_NAMES
        if result_by_name[name]["delta_macro_f1"] > 0.0
    ]
    order_robust = len(positive_order_seeds) >= ORDER_MIN_POSITIVE
    order_seed0_green = (
        result_by_name["S_minus_shuffle_seed0"]["classification"] == "green"
    )
    claims = {
        "structure_induced_rule_benefit": {
            "comparison": "H_minus_R",
            "decision": (
                "retain"
                if result_by_name["H_minus_R"]["classification"] == "green"
                else "drop"
            ),
        },
        "learned_utility_benefit": {
            "comparison": "S_minus_H",
            "decision": (
                "retain"
                if result_by_name["S_minus_H"]["classification"] == "green"
                else "drop"
            ),
        },
        "state_conditioned_rescoring_benefit": {
            "comparison": "S_minus_O",
            "decision": (
                "retain"
                if result_by_name["S_minus_O"]["classification"] == "green"
                else "drop"
            ),
        },
        "presentation_order_benefit": {
            "comparison": "S_minus_shuffle_seed0",
            "decision": "retain" if order_seed0_green and order_robust else "drop",
            "requires_seed0_green": True,
            "requires_at_least_four_positive_seeds": True,
        },
    }
    retained = [name for name, row in claims.items() if row["decision"] == "retain"]
    dropped = [name for name, row in claims.items() if row["decision"] == "drop"]
    full_story_supported = not dropped

    if claims["structure_induced_rule_benefit"]["decision"] == "drop":
        recommendation = "do_not_scale_clean_mechanism_story"
    elif claims["learned_utility_benefit"]["decision"] == "drop":
        recommendation = "retain_hard_structure_rule_drop_learned_utility_claim"
    elif claims["state_conditioned_rescoring_benefit"]["decision"] == "drop":
        recommendation = "drop_state_conditioned_rescoring_claim"
    elif claims["presentation_order_benefit"]["decision"] == "drop":
        recommendation = "drop_presentation_order_claim"
    else:
        recommendation = "retain_full_story_and_proceed_to_confirmatory_stage"

    return {
        "schema_version": SCHEMA_VERSION,
        "preregistration_schema_version": preregistration["schema_version"],
        "scope": contract.scope,
        "metric": contract.metric,
        "sample_size": sample_size,
        "decision_thresholds": {
            "green_min_delta_macro_f1": GREEN_MIN_DELTA,
            "green_requires_bootstrap_ci_lower_above_zero": True,
            "secondary_green_requires_holm_adjusted_p_at_most_alpha": True,
            "alpha": contract.alpha,
            "order_minimum_positive_seeds": ORDER_MIN_POSITIVE,
            "order_total_seeds": len(ORDER_SEED_NAMES),
        },
        "comparisons": results,
        "order_robustness": {
            "seed_comparisons": list(ORDER_SEED_NAMES),
            "positive_seed_comparisons": positive_order_seeds,
            "positive_seed_count": len(positive_order_seeds),
            "required_positive_seed_count": ORDER_MIN_POSITIVE,
            "robust": order_robust,
            "seed0_green": order_seed0_green,
        },
        "claims": claims,
        "retained_claims": retained,
        "dropped_claims": dropped,
        "full_story_supported": full_story_supported,
        "overall_recommendation": recommendation,
        "interpretation_boundary": (
            "Validation-only frozen-verifier input-organization gate; a retained "
            "claim still requires the preregistered confirmatory stage."
        ),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    verdict = "SUPPORTED" if summary["full_story_supported"] else "NOT SUPPORTED"
    lines = [
        "# Structure-only mechanism gate: Stage C",
        "",
        f"- Scope: {summary['scope']}",
        f"- Common paired support: n={summary['sample_size']}",
        f"- Full story: **{verdict}**",
        f"- Recommendation: `{summary['overall_recommendation']}`",
        "",
        "| comparison | family | delta Macro-F1 | bootstrap 95% CI | paired p | Holm p | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["comparisons"]:
        lower, upper = row["bootstrap_ci95"]
        holm = row["holm_adjusted_p_value"]
        holm_text = "-" if holm is None else f"{holm:.6f}"
        lines.append(
            f"| {row['name']} | {row['family']} | "
            f"{row['delta_macro_f1']:+.6f} | [{lower:+.6f}, {upper:+.6f}] | "
            f"{row['paired_randomization_p_value_two_sided']:.6f} | "
            f"{holm_text} | {row['classification']} |"
        )
    order = summary["order_robustness"]
    lines.extend(
        [
            "",
            "## Claim decisions",
            "",
        ]
    )
    for name in CLAIM_ORDER:
        claim = summary["claims"][name]
        lines.append(f"- `{name}`: **{claim['decision']}**")
    lines.extend(
        [
            "",
            (
                f"Order robustness: {order['positive_seed_count']}/"
                f"{len(order['seed_comparisons'])} shuffle deltas are positive; "
                f"robust={str(order['robust']).lower()}, "
                f"seed0_green={str(order['seed0_green']).lower()}."
            ),
            "",
            f"> {summary['interpretation_boundary']}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paired_path = Path(args.paired_json)
    preregistration_path = Path(args.preregistration)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md) if args.output_md else None
    protected = {paired_path.resolve(), preregistration_path.resolve()}
    if output_json.resolve() in protected or (
        output_md is not None and output_md.resolve() in protected
    ):
        raise MechanismGateSummaryError(
            "output paths must not overwrite paired statistics or preregistration"
        )
    if output_md is not None and output_md.resolve() == output_json.resolve():
        raise MechanismGateSummaryError("JSON and Markdown outputs must differ")

    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    summary = summarize_gate(
        _require_mapping(paired, context="paired JSON"),
        _require_mapping(preregistration, context="preregistration JSON"),
    )
    _write_atomic(output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if output_md is not None:
        _write_atomic(output_md, render_markdown(summary))
    print(output_json)
    if output_md is not None:
        print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
