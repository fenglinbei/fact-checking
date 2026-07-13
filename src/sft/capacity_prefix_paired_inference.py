"""Materialize paired inference for frozen BACES capacity-prefix contrasts.

The event is the statistical unit.  The checked-in contrast registry is
post-validation: results on an already inspected validation split are
diagnostic.  This materializer never promotes a split to confirmatory from its
name; prospective use requires a separate pre-inference external contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sft import capacity_prefix_analysis as capacity
from sft import paired_factorial_inference as paired


SCHEMA_VERSION = "baces_capacity_paired_inference_v0_1"
REGISTRY_SCHEMA_VERSION = "baces_capacity_contrast_registry_v0_1"
VALIDATION_ANALYSIS_STATUS = "post_hoc_validation_frozen_verifier_diagnostic"
UNCONTRACTED_ANALYSIS_STATUS = (
    "frozen_split_diagnostic_no_external_preinference_contract"
)
METRIC_NAMES = (
    "macro_f1",
    "accuracy",
    "class_balanced_nll_mean",
    "raw_nll_mean",
    "evidence_count_mean",
    "prompt_token_count_mean",
)
HIGHER_IS_BETTER = {"macro_f1", "accuracy"}
LOWER_IS_BETTER = {
    "class_balanced_nll_mean",
    "raw_nll_mean",
    "evidence_count_mean",
    "prompt_token_count_mean",
}
FULL_SUPPORT = "full_n_deployable"
STRICT_SUPPORT = "strict_full_grid_common_support"
SUPPORTED_SUPPORTS = {FULL_SUPPORT, STRICT_SUPPORT}
ARTIFACT_NAMES = (
    "contrast_registry.snapshot.json",
    "action_intervals.jsonl",
    "action_intervals.csv",
    "comparisons.jsonl",
    "comparisons.csv",
    "classwise_effects.jsonl",
    "classwise_effects.csv",
    "report.md",
    "bootstrap_samples.npz",
    "permutation_null.npz",
)


class CapacityPairedInferenceError(ValueError):
    """Raised when a source, registry, or paired-inference contract drifts."""


@dataclass(frozen=True)
class ActionData:
    action_id: str
    kind: str
    selector_level: str
    requested_k: np.ndarray
    realized_k: np.ndarray
    prompt_token_count: np.ndarray
    prompt_hashes: tuple[str, ...]
    pred_ids: np.ndarray
    raw_nll: np.ndarray
    source: Mapping[str, Any]


@dataclass(frozen=True)
class SupportData:
    support_id: str
    indices: np.ndarray
    event_ids: tuple[str, ...]
    gold_ids: np.ndarray
    event_id_sequence_sha256: str


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    family_id: str
    tier: str
    current_validation_role: str
    future_held_out_role: str
    selection_basis: str
    selection_uses_observed_validation_outcome: bool
    support_id: str
    a_action_id: str
    b_action_id: str


def _analysis_status(evaluation_split: str) -> str:
    if evaluation_split == "val":
        return VALIDATION_ANALYSIS_STATUS
    return UNCONTRACTED_ANALYSIS_STATUS


def _claim_gate(
    *, spec: ComparisonSpec, evaluation_split: str
) -> tuple[str, bool, str]:
    """Return a fail-closed claim role for this materializer.

    A separate pre-inference contract is intentionally outside this artifact.
    Consequently, no split handled here is automatically confirmatory.
    """

    if evaluation_split == "val":
        return (
            spec.current_validation_role,
            False,
            "retrospective_inspected_validation",
        )
    return (
        "diagnostic_no_external_preinference_contract",
        False,
        "automatic_promotion_disabled_external_contract_required",
    )


def _require_complete_class_bootstrap(
    *, support_id: str, ordinary_missing: int, stratified_missing: int
) -> None:
    if ordinary_missing or stratified_missing:
        raise CapacityPairedInferenceError(
            "Class-balanced bootstrap changed its registered class support "
            f"for {support_id}: ordinary_missing={ordinary_missing}, "
            f"gold_stratified_missing={stratified_missing}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CapacityPairedInferenceError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CapacityPairedInferenceError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CapacityPairedInferenceError(f"Expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise CapacityPairedInferenceError(f"Required JSONL file does not exist: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CapacityPairedInferenceError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise CapacityPairedInferenceError(f"Expected object in {path}:{line_number}")
            yield line_number, row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise CapacityPairedInferenceError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _action_id(reference: Mapping[str, Any]) -> str:
    kind = str(reference.get("kind") or "")
    selector = str(reference.get("selector_level") or "")
    if not selector:
        raise CapacityPairedInferenceError(f"Action reference has no selector: {reference}")
    if kind == "fixed_capacity":
        try:
            requested_k = int(reference["requested_k"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CapacityPairedInferenceError(
                f"Fixed-capacity reference has invalid requested_k: {reference}"
            ) from exc
        if requested_k <= 0:
            raise CapacityPairedInferenceError(f"requested_k must be positive: {reference}")
        return f"fixed::{selector}::k{requested_k:02d}"
    if kind == "trace_policy":
        policy_id = str(reference.get("policy_id") or "")
        if not policy_id:
            raise CapacityPairedInferenceError(f"Trace-policy reference has no policy_id: {reference}")
        return f"policy::{policy_id}::{selector}"
    raise CapacityPairedInferenceError(f"Unsupported action kind={kind!r}: {reference}")


def _load_registry(path: Path) -> tuple[dict[str, Any], list[ComparisonSpec]]:
    registry = _read_json(path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise CapacityPairedInferenceError(f"Unsupported contrast registry schema: {path}")
    timing = registry.get("registration_timing")
    if not isinstance(timing, Mapping) or timing.get("current_validation") != (
        "post_hoc_rule_based_after_result_inspection"
    ):
        raise CapacityPairedInferenceError(
            "Registry must state that the inspected validation split is post-hoc"
        )
    estimand = registry.get("estimand")
    if not isinstance(estimand, Mapping) or estimand.get("primary_support") != FULL_SUPPORT:
        raise CapacityPairedInferenceError("Registry primary support must be full_n_deployable")
    if estimand.get("primary_endpoint") != "macro_f1" or estimand.get("tests") != "two_sided":
        raise CapacityPairedInferenceError("Registry must freeze Macro-F1 and two-sided tests")
    if (
        estimand.get("causal_selector_mechanism_claim_allowed") is not False
        or estimand.get("end_to_end_policy_training_claim_allowed") is not False
        or not str(estimand.get("system_scope") or "")
    ):
        raise CapacityPairedInferenceError(
            "Registry must freeze the non-causal frozen-verifier system estimand"
        )
    confirmation_gate = registry.get("prospective_confirmation_gate")
    if (
        not isinstance(confirmation_gate, Mapping)
        or confirmation_gate.get("automatic_promotion_from_split_name") is not False
        or confirmation_gate.get("current_implementation_default") != "diagnostic_only"
        or not str(confirmation_gate.get("required_future_contract") or "")
    ):
        raise CapacityPairedInferenceError(
            "Registry must disable automatic confirmatory promotion"
        )
    multiplicity = registry.get("multiplicity")
    if not isinstance(multiplicity, Mapping) or multiplicity.get("method") != (
        "holm_step_down_fwer"
    ):
        raise CapacityPairedInferenceError("Registry must freeze Holm step-down correction")
    adjusted = [str(value) for value in multiplicity.get("adjusted_endpoints", [])]
    if adjusted != ["macro_f1", "class_balanced_nll_mean", "accuracy"]:
        raise CapacityPairedInferenceError(
            f"Unexpected adjusted endpoint registry: {adjusted}"
        )

    raw_families = registry.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise CapacityPairedInferenceError("Registry has no families")
    families: dict[str, Mapping[str, Any]] = {}
    for raw_family in raw_families:
        if not isinstance(raw_family, Mapping):
            raise CapacityPairedInferenceError("Registry family must be an object")
        family_id = str(raw_family.get("family_id") or "")
        if not family_id or family_id in families:
            raise CapacityPairedInferenceError(f"Duplicate/empty family_id={family_id!r}")
        families[family_id] = raw_family

    raw_contrasts = registry.get("contrasts")
    if not isinstance(raw_contrasts, list) or not raw_contrasts:
        raise CapacityPairedInferenceError("Registry has no contrasts")
    specs: list[ComparisonSpec] = []
    seen_ids: set[str] = set()
    for raw in raw_contrasts:
        if not isinstance(raw, Mapping):
            raise CapacityPairedInferenceError("Registry contrast must be an object")
        comparison_id = str(raw.get("comparison_id") or "")
        family_id = str(raw.get("family_id") or "")
        if not comparison_id or comparison_id in seen_ids or family_id not in families:
            raise CapacityPairedInferenceError(
                f"Invalid contrast identity: comparison={comparison_id!r}, family={family_id!r}"
            )
        seen_ids.add(comparison_id)
        family = families[family_id]
        supports = [str(value) for value in raw.get("supports", [])]
        if not supports or len(supports) != len(set(supports)) or not set(supports) <= SUPPORTED_SUPPORTS:
            raise CapacityPairedInferenceError(
                f"Invalid supports for comparison={comparison_id}: {supports}"
            )
        a_action_id = _action_id(raw.get("a") or {})
        b_action_id = _action_id(raw.get("b") or {})
        if a_action_id == b_action_id:
            raise CapacityPairedInferenceError(f"Degenerate action identity in {comparison_id}")
        uses_observed = raw.get("selection_uses_observed_validation_outcome")
        if not isinstance(uses_observed, bool):
            raise CapacityPairedInferenceError(
                f"Contrast must declare selection_uses_observed_validation_outcome: {comparison_id}"
            )
        if family_id == "prospective_capacity_primary":
            references = [raw.get("a") or {}, raw.get("b") or {}]
            if uses_observed or any(
                str(reference.get("kind") or "") != "fixed_capacity"
                or str(reference.get("selector_level") or "") != "baces_exact"
                or int(reference.get("requested_k", -1)) not in {1, 5, 10}
                for reference in references
            ):
                raise CapacityPairedInferenceError(
                    "Primary capacity contrasts must be outcome-independent BACES exact K1/K5/K10"
                )
        for support_id in supports:
            specs.append(
                ComparisonSpec(
                    comparison_id=comparison_id,
                    family_id=family_id,
                    tier=str(family.get("tier") or ""),
                    current_validation_role=str(
                        family.get("current_validation_role") or ""
                    ),
                    future_held_out_role=str(family.get("future_held_out_role") or ""),
                    selection_basis=str(raw.get("selection_basis") or ""),
                    selection_uses_observed_validation_outcome=uses_observed,
                    support_id=support_id,
                    a_action_id=a_action_id,
                    b_action_id=b_action_id,
                )
            )
    if len(seen_ids) != 9 or len(specs) != 14:
        raise CapacityPairedInferenceError(
            f"Frozen v0.1 registry must contain 9 contrasts / 14 support rows, "
            f"found {len(seen_ids)} / {len(specs)}"
        )
    return registry, specs


def _verify_analysis_contract(
    *,
    analysis_manifest_path: Path,
    matrix_manifest_path: Path,
) -> dict[str, Any]:
    analysis_manifest_path = analysis_manifest_path.resolve()
    matrix_manifest_path = matrix_manifest_path.resolve()
    manifest = _read_json(analysis_manifest_path)
    if (
        manifest.get("schema_version") != capacity.SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise CapacityPairedInferenceError(
            f"Capacity analysis is not a complete supported artifact: {analysis_manifest_path}"
        )
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise CapacityPairedInferenceError("Capacity analysis manifest has no source contract")
    declared_matrix = Path(str(source.get("matrix_manifest") or "")).resolve()
    if (
        declared_matrix != matrix_manifest_path
        or not matrix_manifest_path.is_file()
        or paired._sha256_file(matrix_manifest_path)
        != str(source.get("matrix_manifest_sha256") or "")
    ):
        raise CapacityPairedInferenceError("Capacity analysis/matrix provenance mismatch")
    implementation = manifest.get("implementation")
    if not isinstance(implementation, Mapping):
        raise CapacityPairedInferenceError("Capacity analysis has no implementation contract")
    implementation_path = Path(str(implementation.get("path") or "")).resolve()
    if (
        implementation_path != Path(capacity.__file__).resolve()
        or not implementation_path.is_file()
        or paired._sha256_file(implementation_path)
        != str(implementation.get("sha256") or "")
    ):
        raise CapacityPairedInferenceError("Capacity analysis implementation drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CapacityPairedInferenceError("Capacity analysis artifact map is invalid")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise CapacityPairedInferenceError(f"Invalid capacity artifact entry: {name}")
        path = analysis_manifest_path.parent / str(artifact.get("path") or "")
        if not path.is_file() or paired._sha256_file(path) != str(artifact.get("sha256") or ""):
            raise CapacityPairedInferenceError(f"Capacity analysis artifact drift: {path}")

    matrix = _read_json(matrix_manifest_path)
    if matrix.get("status") != "complete" or bool(matrix.get("diagnostic_only", True)):
        raise CapacityPairedInferenceError("Paired inference requires a formal complete matrix")
    gate_path = matrix_manifest_path.parent / str(matrix.get("equivalence_gate") or "")
    if (
        not gate_path.is_file()
        or paired._sha256_file(gate_path) != str(matrix.get("equivalence_gate_sha256") or "")
        or _read_json(gate_path).get("passed") is not True
    ):
        raise CapacityPairedInferenceError("Formal native equivalence gate is missing or failed")
    return manifest


def _load_curve_rows(
    *, analysis_manifest_path: Path, analysis_manifest: Mapping[str, Any]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    artifact = (analysis_manifest.get("artifacts") or {}).get("capacity_curves.jsonl")
    if not isinstance(artifact, Mapping):
        raise CapacityPairedInferenceError("Capacity analysis has no capacity_curves.jsonl")
    path = analysis_manifest_path.parent / str(artifact.get("path") or "")
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for _, row in _iter_jsonl(path):
        key = (
            str(row.get("selector_level") or ""),
            str(row.get("support") or ""),
            int(row.get("requested_k", -1)),
        )
        if key in output:
            raise CapacityPairedInferenceError(f"Duplicate capacity curve row: {key}")
        output[key] = row
    return output


def _class_balanced_mean(raw_nll: np.ndarray, gold_ids: np.ndarray, n_labels: int) -> float:
    values = []
    for label_id in range(n_labels):
        selected = np.asarray(raw_nll)[np.asarray(gold_ids) == label_id]
        if len(selected) == 0:
            raise CapacityPairedInferenceError(
                f"Class-balanced NLL support is missing label_id={label_id}"
            )
        values.append(float(np.mean(selected)))
    return float(np.mean(values))


def _point_metrics(
    *, action: ActionData, support: SupportData, n_labels: int
) -> tuple[dict[str, float], np.ndarray]:
    indices = support.indices
    point, class_f1 = paired._point_metrics(
        support.gold_ids, action.pred_ids[indices], n_labels=n_labels
    )
    raw_nll = action.raw_nll[indices]
    return (
        {
            **point,
            "class_balanced_nll_mean": _class_balanced_mean(
                raw_nll, support.gold_ids, n_labels
            ),
            "raw_nll_mean": float(np.mean(raw_nll)),
            "evidence_count_mean": float(np.mean(action.realized_k[indices])),
            "prompt_token_count_mean": float(
                np.mean(action.prompt_token_count[indices])
            ),
        },
        class_f1,
    )


def _build_fixed_actions(
    *,
    observations: Mapping[str, Mapping[int, Sequence[capacity.PrefixObservation]]],
    event_ids: Sequence[str],
) -> dict[str, ActionData]:
    actions: dict[str, ActionData] = {}
    for selector, by_k in sorted(observations.items()):
        for requested_k, rows in sorted(by_k.items()):
            if len(rows) != len(event_ids):
                raise CapacityPairedInferenceError(
                    f"Observation count mismatch for selector={selector}, K={requested_k}"
                )
            action_id = f"fixed::{selector}::k{requested_k:02d}"
            raw_nll = np.asarray(
                [capacity._raw_nll(row.logits, row.gold_id) for row in rows],
                dtype=np.float64,
            )
            actions[action_id] = ActionData(
                action_id=action_id,
                kind="fixed_capacity",
                selector_level=selector,
                requested_k=np.full(len(rows), requested_k, dtype=np.int64),
                realized_k=np.asarray([row.realized_k for row in rows], dtype=np.float64),
                prompt_token_count=np.asarray(
                    [row.prompt_token_count for row in rows], dtype=np.float64
                ),
                prompt_hashes=tuple(row.prompt_hash for row in rows),
                pred_ids=np.asarray(
                    [int(np.argmax(row.logits)) for row in rows], dtype=np.int64
                ),
                raw_nll=raw_nll,
                source={
                    "kind": "fixed_capacity",
                    "selector_level": selector,
                    "requested_k": requested_k,
                },
            )
    return actions


def _build_policy_actions(
    *,
    registry_specs: Sequence[ComparisonSpec],
    analysis_manifest: Mapping[str, Any],
    observations: Mapping[str, Mapping[int, Sequence[capacity.PrefixObservation]]],
    event_ids: Sequence[str],
    evaluation_split: str,
) -> tuple[dict[str, ActionData], list[dict[str, Any]]]:
    requested_policy_ids = sorted(
        {
            action_id.split("::", 2)[1]
            for spec in registry_specs
            for action_id in (spec.a_action_id, spec.b_action_id)
            if action_id.startswith("policy::")
        }
    )
    if not requested_policy_ids:
        return {}, []
    declared_sources = {
        str(source.get("policy_id") or ""): source
        for source in analysis_manifest.get("policy_sources", [])
        if isinstance(source, Mapping)
    }
    paths: dict[str, Path] = {}
    for policy_id in requested_policy_ids:
        source = declared_sources.get(policy_id)
        if source is None:
            raise CapacityPairedInferenceError(
                f"Capacity analysis did not bind policy={policy_id}"
            )
        path = Path(str(source.get("path") or "")).resolve()
        if not path.is_file() or paired._sha256_file(path) != str(source.get("sha256") or ""):
            raise CapacityPairedInferenceError(f"Policy artifact drift: {path}")
        paths[policy_id] = path
    try:
        policies, verified_sources = capacity._load_policies(
            paths,
            event_ids=event_ids,
            selectors=sorted(observations),
            evaluation_split=evaluation_split,
        )
    except capacity.CapacityAnalysisError as exc:
        raise CapacityPairedInferenceError(str(exc)) from exc
    verified_by_id = {str(source["policy_id"]): source for source in verified_sources}
    for policy_id in requested_policy_ids:
        verified = verified_by_id[policy_id]
        declared = declared_sources[policy_id]
        for field in (
            "sha256",
            "sidecar_sha256",
            "verification_status",
            "uses_gold",
            "uses_verifier_logits",
            "deployable_ex_ante",
        ):
            if verified.get(field) != declared.get(field):
                raise CapacityPairedInferenceError(
                    f"Policy provenance disagrees with capacity analysis: "
                    f"policy={policy_id}, field={field}"
                )
        if (
            verified.get("verification_status") != "verified_trace_policy_sidecar"
            or verified.get("uses_gold") is not False
            or verified.get("uses_verifier_logits") is not False
            or verified.get("deployable_ex_ante") is not True
        ):
            raise CapacityPairedInferenceError(
                f"Registered trace policy is not verified ex-ante: {policy_id}"
            )

    actions: dict[str, ActionData] = {}
    for policy_id in requested_policy_ids:
        assignments = policies[policy_id]
        for selector in sorted(observations):
            rows: list[capacity.PrefixObservation] = []
            selected_k_values: list[int] = []
            for sample_idx, event_id in enumerate(event_ids):
                key = (selector, event_id)
                if key not in assignments:
                    continue
                selected_k = int(assignments[key])
                if selected_k not in observations[selector]:
                    raise CapacityPairedInferenceError(
                        f"Policy selected unavailable K={selected_k}: {policy_id}, {key}"
                    )
                rows.append(observations[selector][selected_k][sample_idx])
                selected_k_values.append(selected_k)
            if not rows:
                continue
            if len(rows) != len(event_ids):
                raise CapacityPairedInferenceError(
                    f"Policy {policy_id} is incomplete for selector={selector}"
                )
            action_id = f"policy::{policy_id}::{selector}"
            actions[action_id] = ActionData(
                action_id=action_id,
                kind="trace_policy",
                selector_level=selector,
                requested_k=np.asarray(selected_k_values, dtype=np.int64),
                realized_k=np.asarray([row.realized_k for row in rows], dtype=np.float64),
                prompt_token_count=np.asarray(
                    [row.prompt_token_count for row in rows], dtype=np.float64
                ),
                prompt_hashes=tuple(row.prompt_hash for row in rows),
                pred_ids=np.asarray(
                    [int(np.argmax(row.logits)) for row in rows], dtype=np.int64
                ),
                raw_nll=np.asarray(
                    [capacity._raw_nll(row.logits, row.gold_id) for row in rows],
                    dtype=np.float64,
                ),
                source=verified_by_id[policy_id],
            )
    return actions, verified_sources


def _build_supports(
    *,
    event_ids: Sequence[str],
    gold_ids: np.ndarray,
    observations: Mapping[str, Mapping[int, Sequence[capacity.PrefixObservation]]],
    curve_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, SupportData]:
    full_indices = np.arange(len(event_ids), dtype=np.int64)
    strict_mask = np.ones(len(event_ids), dtype=bool)
    for by_k in observations.values():
        for rows in by_k.values():
            strict_mask &= np.asarray(
                [row.realized_k == row.requested_k for row in rows], dtype=bool
            )
    strict_indices = np.flatnonzero(strict_mask).astype(np.int64)
    supports: dict[str, SupportData] = {}
    for support_id, indices in (
        (FULL_SUPPORT, full_indices),
        (STRICT_SUPPORT, strict_indices),
    ):
        support_events = tuple(event_ids[index] for index in indices)
        support = SupportData(
            support_id=support_id,
            indices=indices,
            event_ids=support_events,
            gold_ids=np.asarray(gold_ids[indices], dtype=np.int64),
            event_id_sequence_sha256=capacity._event_sequence_sha256(support_events),
        )
        if len(support.event_ids) == 0:
            raise CapacityPairedInferenceError(f"Support is empty: {support_id}")
        curve_hashes = {
            str(row.get("support_event_id_sequence_sha256") or "")
            for (selector, row_support, requested_k), row in curve_rows.items()
            if row_support == support_id
        }
        curve_counts = {
            int(row.get("sample_count", -1))
            for (selector, row_support, requested_k), row in curve_rows.items()
            if row_support == support_id
        }
        if curve_hashes != {support.event_id_sequence_sha256} or curve_counts != {
            len(support.event_ids)
        }:
            raise CapacityPairedInferenceError(
                f"Recomputed support disagrees with capacity curves: {support_id}"
            )
        supports[support_id] = support
    return supports


def _verify_curve_parity(
    *,
    fixed_actions: Mapping[str, ActionData],
    supports: Mapping[str, SupportData],
    curve_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    n_labels: int,
) -> None:
    metric_map = {
        "macro_f1": "macro_f1",
        "accuracy": "accuracy",
        "class_balanced_nll_mean": "class_balanced_nll_mean",
        "raw_nll_mean": "raw_nll_mean",
        "evidence_count_mean": "mean_realized_k",
        "prompt_token_count_mean": "mean_prompt_token_count",
    }
    for action in fixed_actions.values():
        requested_k = int(action.requested_k[0])
        for support_id, support in supports.items():
            row = curve_rows.get((action.selector_level, support_id, requested_k))
            if row is None:
                raise CapacityPairedInferenceError(
                    f"Missing capacity curve row for {action.action_id}, support={support_id}"
                )
            point, _ = _point_metrics(action=action, support=support, n_labels=n_labels)
            for metric_name, curve_name in metric_map.items():
                if not math.isclose(
                    point[metric_name],
                    float(row.get(curve_name, math.nan)),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise CapacityPairedInferenceError(
                        f"Capacity curve parity failed: action={action.action_id}, "
                        f"support={support_id}, metric={metric_name}"
                    )


def _class_balanced_batch_mean(
    *, gold_batch: np.ndarray, value_batch: np.ndarray, n_labels: int
) -> tuple[np.ndarray, int]:
    gold = np.asarray(gold_batch, dtype=np.int64)
    values = np.asarray(value_batch, dtype=np.float64)
    if gold.shape != values.shape or gold.ndim != 2:
        raise CapacityPairedInferenceError(
            f"Expected matching 2-D gold/value batches, got {gold.shape}/{values.shape}"
        )
    output = np.zeros(gold.shape[0], dtype=np.float64)
    represented = np.zeros(gold.shape[0], dtype=np.int64)
    missing_replicates = np.zeros(gold.shape[0], dtype=bool)
    for label_id in range(n_labels):
        mask = gold == label_id
        counts = np.sum(mask, axis=1)
        present = counts > 0
        missing_replicates |= ~present
        if np.any(present):
            sums = np.sum(np.where(mask, values, 0.0), axis=1)
            output[present] += sums[present] / counts[present]
            represented[present] += 1
    if np.any(represented == 0):
        raise CapacityPairedInferenceError("Bootstrap replicate contains no represented class")
    return output / represented, int(np.sum(missing_replicates))


def bootstrap_action_scores(
    *,
    support: SupportData,
    actions: Sequence[ActionData],
    n_labels: int,
    n_resamples: int,
    seed: int,
    stratified: bool,
    chunk_size: int = 64,
) -> tuple[dict[str, np.ndarray], int]:
    if n_resamples <= 0:
        raise CapacityPairedInferenceError("bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    values = {
        metric_name: np.empty((n_resamples, len(actions)), dtype=np.float64)
        for metric_name in METRIC_NAMES
    }
    support_indices = support.indices
    missing_class_replicates = 0
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = paired._sample_indices(
            rng,
            gold_ids=support.gold_ids,
            batch_size=stop - start,
            stratified=stratified,
        )
        gold_batch = support.gold_ids[indices]
        for action_index, action in enumerate(actions):
            pred = action.pred_ids[support_indices]
            pred_batch = pred[indices]
            accuracy, macro_f1 = paired._metric_arrays(
                gold_batch, pred_batch, n_labels=n_labels
            )
            raw_nll_batch = action.raw_nll[support_indices][indices]
            balanced, missing = _class_balanced_batch_mean(
                gold_batch=gold_batch,
                value_batch=raw_nll_batch,
                n_labels=n_labels,
            )
            if action_index == 0:
                missing_class_replicates += missing
            values["accuracy"][start:stop, action_index] = accuracy
            values["macro_f1"][start:stop, action_index] = macro_f1
            values["class_balanced_nll_mean"][start:stop, action_index] = balanced
            values["raw_nll_mean"][start:stop, action_index] = np.mean(
                raw_nll_batch, axis=1
            )
            values["evidence_count_mean"][start:stop, action_index] = np.mean(
                action.realized_k[support_indices][indices], axis=1
            )
            values["prompt_token_count_mean"][start:stop, action_index] = np.mean(
                action.prompt_token_count[support_indices][indices], axis=1
            )
    return values, missing_class_replicates


def _paired_rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    nonzero = values[values != 0.0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.empty(len(nonzero), dtype=np.float64)
    order = np.argsort(np.abs(nonzero), kind="mergesort")
    sorted_abs = np.abs(nonzero)[order]
    start = 0
    while start < len(nonzero):
        stop = start + 1
        while stop < len(nonzero) and sorted_abs[stop] == sorted_abs[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    total = positive + negative
    return 0.0 if total == 0.0 else float((positive - negative) / total)


def _linear_effects(
    differences: np.ndarray, *, lower_is_better: bool
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if standard_deviation == 0.0:
        dz_status = "degenerate_zero" if mean == 0.0 else "degenerate_nonzero_constant"
    else:
        dz_status = "finite"
    positive = float(np.mean(values > 0.0))
    negative = float(np.mean(values < 0.0))
    ties = float(np.mean(values == 0.0))
    a_better = negative + 0.5 * ties if lower_is_better else positive + 0.5 * ties
    return {
        "mean_event_difference_a_minus_b": mean,
        "median_event_difference_a_minus_b": float(np.median(values)),
        "event_difference_standard_deviation": standard_deviation,
        "cohens_dz_a_minus_b": paired._cohens_dz(values),
        "cohens_dz_status": dz_status,
        "paired_rank_biserial_a_minus_b": _paired_rank_biserial(values),
        "positive_rate": positive,
        "negative_rate": negative,
        "tie_rate": ties,
        "common_language_probability_a_better": a_better,
    }


def paired_permutation_null(
    *,
    support: SupportData,
    a_action: ActionData,
    b_action: ActionData,
    n_labels: int,
    n_resamples: int,
    seed: int,
    chunk_size: int = 64,
) -> np.ndarray:
    if n_resamples <= 0:
        raise CapacityPairedInferenceError("permutation sample count must be positive")
    rng = np.random.default_rng(seed)
    sample_count = len(support.event_ids)
    output = np.empty((n_resamples, len(METRIC_NAMES)), dtype=np.float64)
    indices = support.indices
    a_pred = a_action.pred_ids[indices]
    b_pred = b_action.pred_ids[indices]
    raw_nll_difference = a_action.raw_nll[indices] - b_action.raw_nll[indices]
    evidence_difference = a_action.realized_k[indices] - b_action.realized_k[indices]
    token_difference = (
        a_action.prompt_token_count[indices] - b_action.prompt_token_count[indices]
    )
    gold_template = support.gold_ids[None, :]
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        size = stop - start
        swap = rng.integers(0, 2, size=(size, sample_count), dtype=np.int8).astype(bool)
        permuted_a = np.where(swap, b_pred[None, :], a_pred[None, :])
        permuted_b = np.where(swap, a_pred[None, :], b_pred[None, :])
        gold_batch = np.broadcast_to(gold_template, permuted_a.shape)
        a_accuracy, a_macro = paired._metric_arrays(
            gold_batch, permuted_a, n_labels=n_labels
        )
        b_accuracy, b_macro = paired._metric_arrays(
            gold_batch, permuted_b, n_labels=n_labels
        )
        signs = np.where(swap, -1.0, 1.0)
        signed_nll = signs * raw_nll_difference
        balanced_delta, missing = _class_balanced_batch_mean(
            gold_batch=gold_batch,
            value_batch=signed_nll,
            n_labels=n_labels,
        )
        if missing:
            raise CapacityPairedInferenceError(
                f"Permutation support unexpectedly misses a class: {support.support_id}"
            )
        output[start:stop, 0] = a_macro - b_macro
        output[start:stop, 1] = a_accuracy - b_accuracy
        output[start:stop, 2] = balanced_delta
        output[start:stop, 3] = np.mean(signed_nll, axis=1)
        output[start:stop, 4] = np.mean(signs * evidence_difference, axis=1)
        output[start:stop, 5] = np.mean(signs * token_difference, axis=1)
    return output


def _action_interval_rows(
    *,
    support: SupportData,
    actions: Sequence[ActionData],
    ordinary: Mapping[str, np.ndarray],
    stratified: Mapping[str, np.ndarray],
    alpha: float,
    n_labels: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(actions):
        point, _ = _point_metrics(action=action, support=support, n_labels=n_labels)
        for metric_name in METRIC_NAMES:
            rows.append(
                {
                    "support_id": support.support_id,
                    "support_event_id_sequence_sha256": support.event_id_sequence_sha256,
                    "sample_count": len(support.event_ids),
                    "action_id": action.action_id,
                    "action_kind": action.kind,
                    "selector_level": action.selector_level,
                    "metric": metric_name,
                    "point_estimate": point[metric_name],
                    "ordinary_event_bootstrap_ci": paired._interval(
                        ordinary[metric_name][:, action_index], alpha=alpha
                    ),
                    "gold_stratified_event_bootstrap_ci": paired._interval(
                        stratified[metric_name][:, action_index], alpha=alpha
                    ),
                }
            )
    return rows


def _metric_event_differences(
    *,
    metric_name: str,
    support: SupportData,
    a_action: ActionData,
    b_action: ActionData,
    n_labels: int,
) -> np.ndarray | None:
    indices = support.indices
    if metric_name == "class_balanced_nll_mean":
        weights = capacity.balanced_class_weights(support.gold_ids, n_labels=n_labels)
        return weights[support.gold_ids] * (
            a_action.raw_nll[indices] - b_action.raw_nll[indices]
        )
    if metric_name == "raw_nll_mean":
        return a_action.raw_nll[indices] - b_action.raw_nll[indices]
    if metric_name == "evidence_count_mean":
        return a_action.realized_k[indices] - b_action.realized_k[indices]
    if metric_name == "prompt_token_count_mean":
        return (
            a_action.prompt_token_count[indices]
            - b_action.prompt_token_count[indices]
        )
    return None


def _comparison_rows(
    *,
    specs: Sequence[ComparisonSpec],
    actions: Mapping[str, ActionData],
    supports: Mapping[str, SupportData],
    bootstrap_by_support: Mapping[str, Mapping[str, Any]],
    permutation_samples: int,
    base_seed: int,
    alpha: float,
    n_labels: int,
    evaluation_split: str,
    adjusted_metrics: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, list[int]]:
    rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    null_arrays: list[np.ndarray] = []
    seeds: list[int] = []
    for spec in specs:
        support = supports[spec.support_id]
        bootstrap = bootstrap_by_support[spec.support_id]
        action_ids = bootstrap["action_ids"]
        action_index = {action_id: index for index, action_id in enumerate(action_ids)}
        a_action = actions[spec.a_action_id]
        b_action = actions[spec.b_action_id]
        a_index = action_index[spec.a_action_id]
        b_index = action_index[spec.b_action_id]
        a_point, a_class_f1 = _point_metrics(
            action=a_action, support=support, n_labels=n_labels
        )
        b_point, b_class_f1 = _point_metrics(
            action=b_action, support=support, n_labels=n_labels
        )
        seed = paired._stable_seed(
            base_seed,
            spec.family_id,
            spec.comparison_id,
            spec.support_id,
            "permutation",
        )
        null = paired_permutation_null(
            support=support,
            a_action=a_action,
            b_action=b_action,
            n_labels=n_labels,
            n_resamples=permutation_samples,
            seed=seed,
        )
        null_arrays.append(null)
        seeds.append(seed)
        metric_payload: dict[str, Any] = {}
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            a_value = float(a_point[metric_name])
            b_value = float(b_point[metric_name])
            delta = a_value - b_value
            ordinary_delta = (
                bootstrap["ordinary"][metric_name][:, a_index]
                - bootstrap["ordinary"][metric_name][:, b_index]
            )
            stratified_delta = (
                bootstrap["stratified"][metric_name][:, a_index]
                - bootstrap["stratified"][metric_name][:, b_index]
            )
            lower_is_better = metric_name in LOWER_IS_BETTER
            ordinary_benefit = -ordinary_delta if lower_is_better else ordinary_delta
            stratified_benefit = -stratified_delta if lower_is_better else stratified_delta
            payload: dict[str, Any] = {
                "a_value": a_value,
                "b_value": b_value,
                "delta_a_minus_b": delta,
                "benefit_delta_a_over_b": -delta if lower_is_better else delta,
                "benefit_orientation": "lower_is_better"
                if lower_is_better
                else "higher_is_better",
                "delta_percentage_points": 100.0 * delta
                if metric_name in {"macro_f1", "accuracy"}
                else None,
                "ordinary_paired_bootstrap": {
                    "ci_delta_a_minus_b": paired._interval(ordinary_delta, alpha=alpha),
                    "standard_error": float(np.std(ordinary_delta, ddof=1)),
                    "support_probability_a_better": paired._bootstrap_support(
                        ordinary_benefit
                    ),
                },
                "gold_stratified_paired_bootstrap": {
                    "ci_delta_a_minus_b": paired._interval(
                        stratified_delta, alpha=alpha
                    ),
                    "standard_error": float(np.std(stratified_delta, ddof=1)),
                    "support_probability_a_better": paired._bootstrap_support(
                        stratified_benefit
                    ),
                },
                "paired_permutation": paired._permutation_summary(
                    null[:, metric_index],
                    observed_delta=delta,
                    n_resamples=permutation_samples,
                    seed=seed,
                ),
            }
            if metric_name == "accuracy":
                payload["paired_effects"] = paired._accuracy_pair_effects(
                    support.gold_ids,
                    a_action.pred_ids[support.indices],
                    b_action.pred_ids[support.indices],
                )
            else:
                differences = _metric_event_differences(
                    metric_name=metric_name,
                    support=support,
                    a_action=a_action,
                    b_action=b_action,
                    n_labels=n_labels,
                )
                if differences is not None:
                    payload["paired_effects"] = _linear_effects(
                        differences, lower_is_better=lower_is_better
                    )
            metric_payload[metric_name] = payload

        indices = support.indices
        prompt_changed = np.asarray(
            [
                a_action.prompt_hashes[index] != b_action.prompt_hashes[index]
                for index in indices
            ],
            dtype=bool,
        )
        predictions_differ = (
            a_action.pred_ids[indices] != b_action.pred_ids[indices]
        )
        analysis_role, valid_confirmatory, confirmatory_gate_status = _claim_gate(
            spec=spec,
            evaluation_split=evaluation_split,
        )
        row = {
            "comparison_id": spec.comparison_id,
            "comparison_support_id": f"{spec.comparison_id}::{spec.support_id}",
            "family_id": spec.family_id,
            "tier": spec.tier,
            "analysis_role": analysis_role,
            "current_validation_role": spec.current_validation_role,
            "future_held_out_role": spec.future_held_out_role,
            "declared_future_held_out_role": spec.future_held_out_role,
            "confirmatory_gate_status": confirmatory_gate_status,
            "valid_for_confirmatory_claim": valid_confirmatory,
            "selection_basis": spec.selection_basis,
            "selection_uses_observed_validation_outcome": (
                spec.selection_uses_observed_validation_outcome
            ),
            "support_id": spec.support_id,
            "support_event_id_sequence_sha256": support.event_id_sequence_sha256,
            "sample_count": len(support.event_ids),
            "direction": "a_minus_b",
            "a_action_id": a_action.action_id,
            "b_action_id": b_action.action_id,
            "a_selector_level": a_action.selector_level,
            "b_selector_level": b_action.selector_level,
            "prompt_changed_count": int(np.sum(prompt_changed)),
            "prompt_changed_rate": float(np.mean(prompt_changed)),
            "prediction_disagreement_count": int(np.sum(predictions_differ)),
            "prediction_disagreement_rate": float(np.mean(predictions_differ)),
            "metrics": metric_payload,
        }
        rows.append(row)
        for label_id in range(n_labels):
            classwise_rows.append(
                {
                    "comparison_id": spec.comparison_id,
                    "family_id": spec.family_id,
                    "tier": spec.tier,
                    "support_id": spec.support_id,
                    "a_action_id": a_action.action_id,
                    "b_action_id": b_action.action_id,
                    "label_id": label_id,
                    "a_f1": float(a_class_f1[label_id]),
                    "b_f1": float(b_class_f1[label_id]),
                    "delta_f1_a_minus_b": float(
                        a_class_f1[label_id] - b_class_f1[label_id]
                    ),
                    "descriptive_only": True,
                }
            )

    for family_id in sorted({row["family_id"] for row in rows}):
        for support_id in sorted({row["support_id"] for row in rows}):
            family_rows = [
                row
                for row in rows
                if row["family_id"] == family_id and row["support_id"] == support_id
            ]
            if not family_rows:
                continue
            for metric_name in adjusted_metrics:
                raw_values: list[float] = []
                sources: list[str] = []
                for row in family_rows:
                    metric = row["metrics"][metric_name]
                    if metric_name == "accuracy":
                        raw_value = float(
                            metric["paired_effects"][
                                "mcnemar_exact_p_value_two_sided"
                            ]
                        )
                        source = "two_sided_exact_mcnemar"
                    else:
                        raw_value = float(metric["paired_permutation"]["p_value"])
                        source = "two_sided_paired_permutation"
                    raw_values.append(raw_value)
                    sources.append(source)
                adjusted_values = paired.holm_adjust(raw_values)
                for row, raw_value, adjusted_value, source in zip(
                    family_rows, raw_values, adjusted_values, sources
                ):
                    row["metrics"][metric_name]["multiplicity"] = {
                        "family_id": family_id,
                        "support_id": support_id,
                        "family_size": len(family_rows),
                        "endpoint": metric_name,
                        "method": "holm_step_down_fwer",
                        "scope": "within_family_and_endpoint",
                        "p_value_source": source,
                        "p_value_raw": raw_value,
                        "p_value_holm": adjusted_value,
                        "alpha": alpha,
                        "reject_raw": bool(raw_value <= alpha),
                        "reject_holm": bool(adjusted_value <= alpha),
                        "confirmatory_interpretation_allowed": bool(
                            row["valid_for_confirmatory_claim"]
                        ),
                    }

    primary_full_rows = [
        row
        for row in rows
        if row["family_id"] == "prospective_capacity_primary"
        and row["support_id"] == FULL_SUPPORT
    ]
    if len(primary_full_rows) != 2:
        raise CapacityPairedInferenceError(
            f"Expected two primary full-N rows, found {len(primary_full_rows)}"
        )
    family_alpha = alpha / len(primary_full_rows)
    for row in primary_full_rows:
        spec = next(
            value
            for value in specs
            if value.comparison_id == row["comparison_id"]
            and value.support_id == row["support_id"]
        )
        bootstrap = bootstrap_by_support[row["support_id"]]
        action_index = {
            action_id: index
            for index, action_id in enumerate(bootstrap["action_ids"])
        }
        delta = (
            bootstrap["ordinary"]["macro_f1"][:, action_index[spec.a_action_id]]
            - bootstrap["ordinary"]["macro_f1"][:, action_index[spec.b_action_id]]
        )
        row["metrics"]["macro_f1"]["bonferroni_familywise_ci"] = paired._interval(
            delta, alpha=family_alpha
        )
    return rows, classwise_rows, np.stack(null_arrays, axis=0), seeds


def _flatten_action_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        ordinary = row["ordinary_event_bootstrap_ci"]
        stratified = row["gold_stratified_event_bootstrap_ci"]
        output.append(
            {
                "support_id": row["support_id"],
                "sample_count": row["sample_count"],
                "action_id": row["action_id"],
                "action_kind": row["action_kind"],
                "selector_level": row["selector_level"],
                "metric": row["metric"],
                "point_estimate": row["point_estimate"],
                "ordinary_ci_low": ordinary["low"],
                "ordinary_ci_high": ordinary["high"],
                "stratified_ci_low": stratified["low"],
                "stratified_ci_high": stratified["high"],
            }
        )
    return output


def _flatten_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for metric_name in METRIC_NAMES:
            metric = row["metrics"][metric_name]
            ordinary = metric["ordinary_paired_bootstrap"]
            stratified = metric["gold_stratified_paired_bootstrap"]
            permutation = metric["paired_permutation"]
            multiplicity = metric.get("multiplicity") or {}
            effects = metric.get("paired_effects") or {}
            output.append(
                {
                    "comparison_id": row["comparison_id"],
                    "family_id": row["family_id"],
                    "tier": row["tier"],
                    "analysis_role": row["analysis_role"],
                    "confirmatory_gate_status": row[
                        "confirmatory_gate_status"
                    ],
                    "valid_for_confirmatory_claim": row[
                        "valid_for_confirmatory_claim"
                    ],
                    "support_id": row["support_id"],
                    "sample_count": row["sample_count"],
                    "metric": metric_name,
                    "a_action_id": row["a_action_id"],
                    "b_action_id": row["b_action_id"],
                    "a_value": metric["a_value"],
                    "b_value": metric["b_value"],
                    "delta_a_minus_b": metric["delta_a_minus_b"],
                    "benefit_delta_a_over_b": metric["benefit_delta_a_over_b"],
                    "delta_percentage_points": metric["delta_percentage_points"],
                    "ordinary_ci_low": ordinary["ci_delta_a_minus_b"]["low"],
                    "ordinary_ci_high": ordinary["ci_delta_a_minus_b"]["high"],
                    "ordinary_bootstrap_se": ordinary["standard_error"],
                    "ordinary_support_probability_a_better": ordinary[
                        "support_probability_a_better"
                    ],
                    "stratified_ci_low": stratified["ci_delta_a_minus_b"]["low"],
                    "stratified_ci_high": stratified["ci_delta_a_minus_b"]["high"],
                    "permutation_p_value": permutation["p_value"],
                    "permutation_mcse": permutation[
                        "monte_carlo_standard_error"
                    ],
                    "p_value_source_for_holm": multiplicity.get("p_value_source"),
                    "p_value_raw_for_holm": multiplicity.get("p_value_raw"),
                    "p_value_holm": multiplicity.get("p_value_holm"),
                    "reject_holm": multiplicity.get("reject_holm"),
                    "mcnemar_exact_p_value": effects.get(
                        "mcnemar_exact_p_value_two_sided"
                    ),
                    "matched_pairs_odds_ratio_haldane_anscombe": effects.get(
                        "matched_pairs_odds_ratio_haldane_anscombe"
                    ),
                    "cohens_dz_a_minus_b": effects.get("cohens_dz_a_minus_b"),
                    "paired_rank_biserial_a_minus_b": effects.get(
                        "paired_rank_biserial_a_minus_b"
                    ),
                    "median_event_difference_a_minus_b": effects.get(
                        "median_event_difference_a_minus_b"
                    ),
                    "common_language_probability_a_better": effects.get(
                        "common_language_probability_a_better"
                    ),
                    "prompt_changed_count": row["prompt_changed_count"],
                    "prompt_changed_rate": row["prompt_changed_rate"],
                    "prediction_disagreement_count": row[
                        "prediction_disagreement_count"
                    ],
                    "prediction_disagreement_rate": row[
                        "prediction_disagreement_rate"
                    ],
                }
            )
    return output


def _render_report(
    *,
    registry: Mapping[str, Any],
    evaluation_split: str,
    supports: Mapping[str, SupportData],
    comparison_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    permutation_samples: int,
    alpha: float,
) -> str:
    timing = registry["registration_timing"]
    confirmation_gate = registry["prospective_confirmation_gate"]
    estimand = registry["estimand"]
    lines = [
        "# BACES capacity-prefix paired inference",
        "",
        f"- split: `{evaluation_split}`",
        f"- analysis status: `{_analysis_status(evaluation_split)}`",
        f"- registry: `{registry['registry_id']}`",
        f"- current validation timing: `{timing['current_validation']}`",
        f"- future held-out timing: `{timing['future_held_out_test']}`",
        f"- ordinary paired event bootstrap: {bootstrap_samples}",
        f"- gold-stratified paired event bootstrap: {bootstrap_samples}",
        f"- paired permutation samples per comparison/support: {permutation_samples}",
        f"- marginal CI level: {100.0 * (1.0 - alpha):.1f}%",
        "- primary endpoint: Macro-F1; delta direction: A minus B",
        "- Holm scope: within scientific family, support, and endpoint",
        f"- estimand scope: {estimand['system_scope']}",
        "- automatic confirmatory promotion from split name: disabled",
        "",
        "The current validation surface was inspected before this registry was frozen. "
        "Therefore every validation p-value below is retrospective and cannot support a "
        "prospective confirmatory claim. This runner also keeps every non-validation run "
        "diagnostic by default. Prospective confirmation requires a separate pre-inference "
        f"contract: {confirmation_gate['required_future_contract']}",
        "",
        "The current LIAR-RAW test split is not treated as untouched because it has already "
        "been inspected in prior project evaluation.",
        "",
        "## Supports",
        "",
        "| support | N | event sequence SHA256 | role |",
        "|---|---:|---|---|",
    ]
    for support_id in (FULL_SUPPORT, STRICT_SUPPORT):
        support = supports[support_id]
        role = "primary deployment estimand" if support_id == FULL_SUPPORT else "exact-K sensitivity"
        lines.append(
            f"| {support_id} | {len(support.event_ids)} | "
            f"`{support.event_id_sequence_sha256}` | {role} |"
        )

    lines.extend(
        [
            "",
            "## Full-N registered comparisons",
            "",
            "| comparison | family / tier | role | Macro-F1 delta (pp) | ordinary CI (pp) | permutation p | Holm p | Accuracy delta (pp) | balanced-NLL delta | NLL CI | NLL Holm p | token delta | prompt/pred changes |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        if row["support_id"] != FULL_SUPPORT:
            continue
        macro = row["metrics"]["macro_f1"]
        accuracy = row["metrics"]["accuracy"]
        nll = row["metrics"]["class_balanced_nll_mean"]
        tokens = row["metrics"]["prompt_token_count_mean"]
        macro_ci = macro["ordinary_paired_bootstrap"]["ci_delta_a_minus_b"]
        nll_ci = nll["ordinary_paired_bootstrap"]["ci_delta_a_minus_b"]
        lines.append(
            f"| {row['comparison_id']} | {row['family_id']} / {row['tier']} | "
            f"{row['analysis_role']} | {macro['delta_percentage_points']:.3f} | "
            f"[{100 * macro_ci['low']:.3f}, {100 * macro_ci['high']:.3f}] | "
            f"{macro['paired_permutation']['p_value']:.6f} | "
            f"{macro['multiplicity']['p_value_holm']:.6f} | "
            f"{accuracy['delta_percentage_points']:.3f} | "
            f"{nll['delta_a_minus_b']:.6f} | "
            f"[{nll_ci['low']:.6f}, {nll_ci['high']:.6f}] | "
            f"{nll['multiplicity']['p_value_holm']:.6f} | "
            f"{tokens['delta_a_minus_b']:.3f} | "
            f"{row['prompt_changed_count']}/{row['prediction_disagreement_count']} |"
        )

    lines.extend(
        [
            "",
            "## Strict full-grid common-support sensitivity",
            "",
            "| comparison | N | Macro-F1 delta (pp) | ordinary CI (pp) | permutation p | Holm p | balanced-NLL delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        if row["support_id"] != STRICT_SUPPORT:
            continue
        macro = row["metrics"]["macro_f1"]
        nll = row["metrics"]["class_balanced_nll_mean"]
        macro_ci = macro["ordinary_paired_bootstrap"]["ci_delta_a_minus_b"]
        lines.append(
            f"| {row['comparison_id']} | {row['sample_count']} | "
            f"{macro['delta_percentage_points']:.3f} | "
            f"[{100 * macro_ci['low']:.3f}, {100 * macro_ci['high']:.3f}] | "
            f"{macro['paired_permutation']['p_value']:.6f} | "
            f"{macro['multiplicity']['p_value_holm']:.6f} | "
            f"{nll['delta_a_minus_b']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(registry["interpretation_boundary"]),
            "",
            "Marginal bootstrap intervals are not simultaneous intervals. The two "
            "primary full-N Macro-F1 rows additionally store Bonferroni family-wise "
            "intervals, while inferential rejection follows Holm-adjusted p-values. "
            "Accuracy uses exact two-sided McNemar p-values for Holm; Macro-F1 and NLL "
            "use two-sided paired randomization. Classwise effects are descriptive. "
            "This analysis does not estimate verifier-seed, map, retrieval, checkpoint-"
            "selection, or test-set uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_existing(
    *,
    output_dir: Path,
    analysis_manifest_sha256: str,
    matrix_manifest_sha256: str,
    registry_sha256: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    source = manifest.get("source") or {}
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or source.get("capacity_analysis_manifest_sha256")
        != analysis_manifest_sha256
        or source.get("matrix_manifest_sha256") != matrix_manifest_sha256
        or source.get("contrast_registry_sha256") != registry_sha256
        or manifest.get("settings") != dict(settings)
    ):
        raise CapacityPairedInferenceError(
            f"Existing capacity paired artifact is incompatible: {output_dir}; pass --force"
        )
    dependencies = manifest.get("implementation_dependencies")
    if not isinstance(dependencies, list):
        raise CapacityPairedInferenceError("Existing artifact has no implementation dependencies")
    expected_paths = {
        Path(__file__).resolve(),
        Path(capacity.__file__).resolve(),
        Path(paired.__file__).resolve(),
    }
    actual_paths: set[Path] = set()
    for dependency in dependencies:
        path = Path(str((dependency or {}).get("path") or "")).resolve()
        actual_paths.add(path)
        if not path.is_file() or paired._sha256_file(path) != str(
            (dependency or {}).get("sha256") or ""
        ):
            raise CapacityPairedInferenceError(f"Implementation dependency drift: {path}")
    if actual_paths != expected_paths:
        raise CapacityPairedInferenceError("Implementation dependency set drift")
    artifacts = manifest.get("artifacts") or {}
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise CapacityPairedInferenceError("Existing artifact key set drift")
    for name, artifact in artifacts.items():
        path = output_dir / str((artifact or {}).get("path") or "")
        if not path.is_file() or paired._sha256_file(path) != str(
            (artifact or {}).get("sha256") or ""
        ):
            raise CapacityPairedInferenceError(f"Existing artifact drift ({name}): {path}")
    return manifest


def materialize_capacity_paired_inference(
    *,
    matrix_manifest_path: Path,
    capacity_analysis_manifest_path: Path,
    contrast_registry_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 20_000,
    seed: int = 20_260_713,
    alpha: float = 0.05,
    force: bool = False,
) -> dict[str, Any]:
    if bootstrap_samples <= 0 or permutation_samples <= 0:
        raise CapacityPairedInferenceError("Resample counts must be positive")
    if not 0.0 < alpha < 1.0:
        raise CapacityPairedInferenceError(f"alpha must be in (0,1), got {alpha}")
    matrix_manifest_path = matrix_manifest_path.resolve()
    capacity_analysis_manifest_path = capacity_analysis_manifest_path.resolve()
    contrast_registry_path = contrast_registry_path.resolve()
    output_dir = output_dir.resolve()
    registry, specs = _load_registry(contrast_registry_path)
    analysis_manifest = _verify_analysis_contract(
        analysis_manifest_path=capacity_analysis_manifest_path,
        matrix_manifest_path=matrix_manifest_path,
    )
    curve_rows = _load_curve_rows(
        analysis_manifest_path=capacity_analysis_manifest_path,
        analysis_manifest=analysis_manifest,
    )
    try:
        matrix, labels, event_ids, gold_ids, observations, source = (
            capacity._load_prefix_source(matrix_manifest_path)
        )
    except capacity.CapacityAnalysisError as exc:
        raise CapacityPairedInferenceError(str(exc)) from exc
    evaluation_split = str(matrix.get("split") or "")
    if not evaluation_split or evaluation_split != str(analysis_manifest.get("split") or ""):
        raise CapacityPairedInferenceError("Matrix/capacity-analysis split mismatch")
    fixed_actions = _build_fixed_actions(
        observations=observations, event_ids=event_ids
    )
    supports = _build_supports(
        event_ids=event_ids,
        gold_ids=gold_ids,
        observations=observations,
        curve_rows=curve_rows,
    )
    _verify_curve_parity(
        fixed_actions=fixed_actions,
        supports=supports,
        curve_rows=curve_rows,
        n_labels=len(labels),
    )
    policy_actions, policy_sources = _build_policy_actions(
        registry_specs=specs,
        analysis_manifest=analysis_manifest,
        observations=observations,
        event_ids=event_ids,
        evaluation_split=evaluation_split,
    )
    actions = {**fixed_actions, **policy_actions}
    required_actions = {
        action_id
        for spec in specs
        for action_id in (spec.a_action_id, spec.b_action_id)
    }
    missing_actions = required_actions - set(actions)
    if missing_actions:
        raise CapacityPairedInferenceError(
            f"Registry references unavailable actions: {sorted(missing_actions)}"
        )

    adjusted_metrics = [
        str(value)
        for value in registry["multiplicity"]["adjusted_endpoints"]
    ]
    bootstrap_by_support: dict[str, dict[str, Any]] = {}
    action_rows: list[dict[str, Any]] = []
    bootstrap_settings: dict[str, Any] = {}
    for support_id in (FULL_SUPPORT, STRICT_SUPPORT):
        support_specs = [spec for spec in specs if spec.support_id == support_id]
        action_ids = sorted(
            {
                action_id
                for spec in support_specs
                for action_id in (spec.a_action_id, spec.b_action_id)
            }
        )
        support_actions = [actions[action_id] for action_id in action_ids]
        ordinary_seed = paired._stable_seed(
            seed, support_id, "all_actions", "bootstrap", "ordinary"
        )
        stratified_seed = paired._stable_seed(
            seed, support_id, "all_actions", "bootstrap", "gold_stratified"
        )
        ordinary, ordinary_missing = bootstrap_action_scores(
            support=supports[support_id],
            actions=support_actions,
            n_labels=len(labels),
            n_resamples=bootstrap_samples,
            seed=ordinary_seed,
            stratified=False,
        )
        stratified, stratified_missing = bootstrap_action_scores(
            support=supports[support_id],
            actions=support_actions,
            n_labels=len(labels),
            n_resamples=bootstrap_samples,
            seed=stratified_seed,
            stratified=True,
        )
        _require_complete_class_bootstrap(
            support_id=support_id,
            ordinary_missing=ordinary_missing,
            stratified_missing=stratified_missing,
        )
        bootstrap_by_support[support_id] = {
            "action_ids": action_ids,
            "ordinary": ordinary,
            "stratified": stratified,
            "ordinary_missing_class_replicates": ordinary_missing,
            "stratified_missing_class_replicates": stratified_missing,
        }
        bootstrap_settings[support_id] = {
            "ordinary_seed": ordinary_seed,
            "gold_stratified_seed": stratified_seed,
            "ordinary_missing_class_replicates": ordinary_missing,
            "gold_stratified_missing_class_replicates": stratified_missing,
        }
        action_rows.extend(
            _action_interval_rows(
                support=supports[support_id],
                actions=support_actions,
                ordinary=ordinary,
                stratified=stratified,
                alpha=alpha,
                n_labels=len(labels),
            )
        )

    settings = {
        "alpha": float(alpha),
        "bootstrap_samples": int(bootstrap_samples),
        "permutation_samples": int(permutation_samples),
        "base_seed": int(seed),
        "seed_rule": "uint32(first8_sha256(base_seed:parts))",
        "bootstrap": {
            "method": "paired_percentile_event_bootstrap",
            "ordinary_class_balanced_nll": (
                "mean_of_within_replicate_class_mean_raw_nll_over_all_registered_classes;"
                "fail_closed_if_any_class_is_missing"
            ),
            "sensitivity_stratification": "gold_id",
            "supports": bootstrap_settings,
        },
        "permutation": {
            "method": "paired_event_level_complete_outcome_swap_monte_carlo",
            "alternative": "two_sided",
            "plus_one_correction": True,
        },
        "metrics": list(METRIC_NAMES),
        "multiplicity": dict(registry["multiplicity"]),
    }
    analysis_sha = paired._sha256_file(capacity_analysis_manifest_path)
    matrix_sha = paired._sha256_file(matrix_manifest_path)
    registry_sha = paired._sha256_file(contrast_registry_path)
    if output_dir.exists() and not force:
        return _validate_existing(
            output_dir=output_dir,
            analysis_manifest_sha256=analysis_sha,
            matrix_manifest_sha256=matrix_sha,
            registry_sha256=registry_sha,
            settings=settings,
        )

    comparison_rows, classwise_rows, permutation_null, permutation_seeds = (
        _comparison_rows(
            specs=specs,
            actions=actions,
            supports=supports,
            bootstrap_by_support=bootstrap_by_support,
            permutation_samples=permutation_samples,
            base_seed=seed,
            alpha=alpha,
            n_labels=len(labels),
            evaluation_split=evaluation_split,
            adjusted_metrics=adjusted_metrics,
        )
    )
    if any(row["valid_for_confirmatory_claim"] for row in comparison_rows):
        raise CapacityPairedInferenceError(
            "This materializer must never mark a result confirmatory without an "
            "external pre-inference contract"
        )

    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copyfile(contrast_registry_path, staging / "contrast_registry.snapshot.json")
        paired._write_jsonl(staging / "action_intervals.jsonl", action_rows)
        _write_csv(staging / "action_intervals.csv", _flatten_action_rows(action_rows))
        paired._write_jsonl(staging / "comparisons.jsonl", comparison_rows)
        _write_csv(
            staging / "comparisons.csv",
            _flatten_comparison_rows(comparison_rows),
        )
        paired._write_jsonl(staging / "classwise_effects.jsonl", classwise_rows)
        _write_csv(staging / "classwise_effects.csv", classwise_rows)
        (staging / "report.md").write_text(
            _render_report(
                registry=registry,
                evaluation_split=evaluation_split,
                supports=supports,
                comparison_rows=comparison_rows,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                alpha=alpha,
            ),
            encoding="utf-8",
        )
        bootstrap_payload: dict[str, np.ndarray] = {
            "metric_names": np.asarray(METRIC_NAMES),
        }
        for support_id, bootstrap in bootstrap_by_support.items():
            prefix = support_id
            bootstrap_payload[f"{prefix}__action_ids"] = np.asarray(
                bootstrap["action_ids"]
            )
            for metric_name in METRIC_NAMES:
                bootstrap_payload[f"{prefix}__ordinary__{metric_name}"] = bootstrap[
                    "ordinary"
                ][metric_name]
                bootstrap_payload[
                    f"{prefix}__gold_stratified__{metric_name}"
                ] = bootstrap["stratified"][metric_name]
        with (staging / "bootstrap_samples.npz").open("wb") as handle:
            np.savez_compressed(handle, **bootstrap_payload)
        with (staging / "permutation_null.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                comparison_support_ids=np.asarray(
                    [
                        f"{spec.comparison_id}::{spec.support_id}"
                        for spec in specs
                    ]
                ),
                metric_names=np.asarray(METRIC_NAMES),
                seeds=np.asarray(permutation_seeds, dtype=np.uint64),
                null_deltas=permutation_null,
            )

        artifacts = {
            name: {
                "path": name,
                "sha256": paired._sha256_file(staging / name),
                "size": (staging / name).stat().st_size,
            }
            for name in ARTIFACT_NAMES
        }
        dependencies = [
            {
                "path": str(path),
                "sha256": paired._sha256_file(path),
            }
            for path in (
                Path(__file__).resolve(),
                Path(capacity.__file__).resolve(),
                Path(paired.__file__).resolve(),
            )
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at": paired._utc_now(),
            "analysis_status": _analysis_status(evaluation_split),
            "split": evaluation_split,
            "registry_id": registry["registry_id"],
            "registration_timing": registry["registration_timing"],
            "prospective_confirmation_gate": registry[
                "prospective_confirmation_gate"
            ],
            "estimand": registry["estimand"],
            "sample_count": len(event_ids),
            "support_count": len(supports),
            "action_count": len(required_actions),
            "registered_contrast_count": len({spec.comparison_id for spec in specs}),
            "comparison_support_row_count": len(specs),
            "labels": labels,
            "supports": {
                support_id: {
                    "sample_count": len(support.event_ids),
                    "event_id_sequence_sha256": support.event_id_sequence_sha256,
                    "gold_id_counts": {
                        str(label_id): int(np.sum(support.gold_ids == label_id))
                        for label_id in range(len(labels))
                    },
                }
                for support_id, support in supports.items()
            },
            "comparison_registry": [spec.__dict__ for spec in specs],
            "settings": settings,
            "source": {
                "matrix_manifest": str(matrix_manifest_path),
                "matrix_manifest_sha256": matrix_sha,
                "capacity_analysis_manifest": str(
                    capacity_analysis_manifest_path
                ),
                "capacity_analysis_manifest_sha256": analysis_sha,
                "contrast_registry": str(contrast_registry_path),
                "contrast_registry_sha256": registry_sha,
                "prefix_source": source,
                "policy_sources": policy_sources,
            },
            "implementation_dependencies": dependencies,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "bit_generator": "PCG64",
            },
            "interpretation_boundary": (
                "The validation split was inspected before the contrast registry was "
                "frozen. Validation intervals and p-values are retrospective diagnostics. "
                "No split is automatically promoted to confirmatory; that requires a "
                "separate pre-inference external contract. The current LIAR-RAW test split "
                "is not untouched. Holm correction does not repair outcome-dependent K "
                "selection, checkpoint selection, or missing verifier/map/retrieval seed "
                "uncertainty. Effects describe one frozen-verifier system response and are "
                "not causal selector-mechanism or end-to-end training effects."
            ),
            "artifacts": artifacts,
        }
        paired._write_json(staging / "manifest.json", manifest)
        paired._promote_directory(staging, output_dir, force=force)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--capacity-analysis-manifest", required=True)
    parser.add_argument("--contrast-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutation-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = materialize_capacity_paired_inference(
        matrix_manifest_path=Path(args.matrix_manifest),
        capacity_analysis_manifest_path=Path(args.capacity_analysis_manifest),
        contrast_registry_path=Path(args.contrast_registry),
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
        alpha=args.alpha,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis_status": result["analysis_status"],
                "registered_contrast_count": result[
                    "registered_contrast_count"
                ],
                "comparison_support_row_count": result[
                    "comparison_support_row_count"
                ],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
