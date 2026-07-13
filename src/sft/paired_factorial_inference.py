"""Materialize paired uncertainty and effect-size artifacts for the BACES factorial.

The statistical unit is the fact-checking event.  Raw-logit prompt deduplication is
only an inference optimization and never changes the resampling unit used here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "baces_factorial_paired_inference_v0_1"
ANALYSIS_STATUS = "post_hoc_validation_frozen_verifier_diagnostic"
SELECTOR_LEVELS = (
    "retrieval_source",
    "map_quality_static",
    "ordinal_coverage_greedy",
    "state_free_structural",
    "baces_exact",
    "learned_marginal",
)
COMPARATOR_LEVELS = tuple(value for value in SELECTOR_LEVELS if value != "baces_exact")
CONTROLLER_LEVELS = (
    "fixed5",
    "ordinal_replay_minmax5_10",
    "matched_token_cap",
)
METRIC_NAMES = (
    "macro_f1",
    "accuracy",
    "evidence_count_mean",
    "prompt_token_count_mean",
)
EXPECTED_IDENTICAL_PROMPT_PAIRS = (
    (
        "baces_exact__fixed5",
        "baces_exact__ordinal_replay_minmax5_10",
    ),
    (
        "ordinal_coverage_greedy__fixed5",
        "ordinal_coverage_greedy__ordinal_replay_minmax5_10",
    ),
)


class PairedInferenceError(ValueError):
    """Raised when a paired-inference input or artifact contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(path)


def _promote_directory(staging: Path, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to replace existing artifact directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.old.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


@dataclass(frozen=True)
class CellData:
    cell_id: str
    selector_level: str
    controller_level: str
    predictions_path: Path
    predictions_sha256: str
    pred_ids: np.ndarray
    evidence_count: np.ndarray
    prompt_token_count: np.ndarray
    prompt_hashes: tuple[str, ...]
    point_metrics: Mapping[str, float]
    class_f1: np.ndarray


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    family_id: str
    role: str
    a_cell_id: str
    b_cell_id: str
    direction: str = "a_minus_b"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PairedInferenceError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PairedInferenceError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PairedInferenceError(f"Expected a JSON object in {path}")
    return value


def _resolve_declared_path(value: str, *, anchor: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, anchor / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _event_sequence_sha256(event_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for event_id in event_ids:
        digest.update(event_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = ":".join([str(int(base_seed)), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _metric_arrays(
    gold_ids: np.ndarray,
    pred_ids: np.ndarray,
    *,
    n_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row accuracy and Macro-F1 for matching 2-D batches."""
    gold = np.asarray(gold_ids, dtype=np.int64)
    pred = np.asarray(pred_ids, dtype=np.int64)
    if gold.shape != pred.shape or gold.ndim != 2:
        raise PairedInferenceError(
            f"Expected matching 2-D gold/pred arrays, got {gold.shape} and {pred.shape}"
        )
    batch_size, sample_count = gold.shape
    offsets = np.arange(batch_size, dtype=np.int64)[:, None] * (n_labels * n_labels)
    codes = gold * n_labels + pred + offsets
    matrices = np.bincount(
        codes.ravel(), minlength=batch_size * n_labels * n_labels
    ).reshape(batch_size, n_labels, n_labels)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2).astype(np.float64)
    gold_count = matrices.sum(axis=2, dtype=np.float64)
    pred_count = matrices.sum(axis=1, dtype=np.float64)
    denominator = gold_count + pred_count
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    accuracy = np.sum(true_positive, axis=1) / float(sample_count)
    return accuracy, class_f1.mean(axis=1)


def _point_metrics(
    gold_ids: np.ndarray,
    pred_ids: np.ndarray,
    *,
    n_labels: int,
) -> tuple[dict[str, float], np.ndarray]:
    accuracy, macro_f1 = _metric_arrays(
        np.asarray(gold_ids, dtype=np.int64)[None, :],
        np.asarray(pred_ids, dtype=np.int64)[None, :],
        n_labels=n_labels,
    )
    matrix = np.bincount(
        np.asarray(gold_ids, dtype=np.int64) * n_labels
        + np.asarray(pred_ids, dtype=np.int64),
        minlength=n_labels * n_labels,
    ).reshape(n_labels, n_labels)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=1, dtype=np.float64) + matrix.sum(
        axis=0, dtype=np.float64
    )
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return {
        "accuracy": float(accuracy[0]),
        "macro_f1": float(macro_f1[0]),
    }, class_f1


def _iter_prediction_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PairedInferenceError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise PairedInferenceError(f"Expected object in {path}:{line_number}")
            yield line_number, row


def _load_cell(
    *,
    matrix_dir: Path,
    cell_manifest: Mapping[str, Any],
    labels: Sequence[str],
    scoring_fingerprint: str,
    expected_event_ids: Sequence[str] | None,
    expected_gold_ids: np.ndarray | None,
) -> tuple[CellData, list[str], np.ndarray]:
    cell_id = str(cell_manifest.get("cell_id") or "")
    selector = str(cell_manifest.get("selector_level") or "")
    controller = str(cell_manifest.get("controller_level") or "")
    if cell_id != f"{selector}__{controller}":
        raise PairedInferenceError(
            f"Cell identity mismatch: cell_id={cell_id!r}, selector={selector!r}, "
            f"controller={controller!r}"
        )
    predictions_path = matrix_dir / str(cell_manifest.get("predictions_file") or "")
    expected_predictions_sha = str(cell_manifest.get("predictions_sha256") or "")
    if not predictions_path.is_file():
        raise PairedInferenceError(f"Missing predictions for {cell_id}: {predictions_path}")
    actual_predictions_sha = _sha256_file(predictions_path)
    if actual_predictions_sha != expected_predictions_sha:
        raise PairedInferenceError(
            f"Prediction SHA mismatch for {cell_id}: expected={expected_predictions_sha}, "
            f"actual={actual_predictions_sha}"
        )

    metrics_path = matrix_dir / str(cell_manifest.get("metrics_file") or "")
    expected_metrics_sha = str(cell_manifest.get("metrics_sha256") or "")
    if not metrics_path.is_file() or _sha256_file(metrics_path) != expected_metrics_sha:
        raise PairedInferenceError(f"Metrics artifact drift for {cell_id}: {metrics_path}")
    metrics = _read_json(metrics_path)
    if int(metrics.get("num_samples", -1)) <= 0 or float(
        metrics.get("parse_error_rate", math.nan)
    ) != 0.0:
        raise PairedInferenceError(f"Cell {cell_id} is not a complete parse-error-free result")

    event_ids: list[str] = []
    gold_ids: list[int] = []
    pred_ids: list[int] = []
    evidence_count: list[float] = []
    prompt_token_count: list[float] = []
    prompt_hashes: list[str] = []
    seen_events: set[str] = set()
    n_labels = len(labels)
    for row_index, (line_number, row) in enumerate(_iter_prediction_rows(predictions_path)):
        sample_idx = int(row.get("sample_idx", -1))
        if sample_idx != row_index:
            raise PairedInferenceError(
                f"{cell_id}: sample_idx must be contiguous and ordered at "
                f"{predictions_path}:{line_number}; expected={row_index}, actual={sample_idx}"
            )
        event_id = str(row.get("event_id") or "")
        if not event_id or event_id in seen_events:
            raise PairedInferenceError(f"{cell_id}: missing/duplicate event_id={event_id!r}")
        seen_events.add(event_id)
        gold_id = int(row.get("gold_id", -1))
        pred_id = int(row.get("pred_id", -1))
        if not 0 <= gold_id < n_labels or not 0 <= pred_id < n_labels:
            raise PairedInferenceError(
                f"{cell_id}: invalid gold/pred at event={event_id}: {gold_id}/{pred_id}"
            )
        if str(row.get("gold_label") or "") != labels[gold_id]:
            raise PairedInferenceError(f"{cell_id}: gold label mismatch at event={event_id}")
        if str(row.get("pred_label") or "") != labels[pred_id]:
            raise PairedInferenceError(f"{cell_id}: prediction label mismatch at event={event_id}")
        if str(row.get("cell_id") or "") != cell_id:
            raise PairedInferenceError(f"{cell_id}: row cell identity mismatch at event={event_id}")
        if str(row.get("selector_level") or "") != selector or str(
            row.get("controller_level") or ""
        ) != controller:
            raise PairedInferenceError(f"{cell_id}: row factor identity mismatch at event={event_id}")
        if str(row.get("scoring_fingerprint") or "") != scoring_fingerprint:
            raise PairedInferenceError(f"{cell_id}: scoring fingerprint mismatch at event={event_id}")
        evidence_value = float(row.get("evidence_count", math.nan))
        token_value = float(row.get("prompt_token_count", math.nan))
        if not math.isfinite(evidence_value) or not math.isfinite(token_value):
            raise PairedInferenceError(f"{cell_id}: non-finite resource value at event={event_id}")
        prompt_hash = str(row.get("prompt_input_ids_sha256") or "")
        if not prompt_hash:
            raise PairedInferenceError(f"{cell_id}: missing prompt hash at event={event_id}")
        event_ids.append(event_id)
        gold_ids.append(gold_id)
        pred_ids.append(pred_id)
        evidence_count.append(evidence_value)
        prompt_token_count.append(token_value)
        prompt_hashes.append(prompt_hash)

    gold_array = np.asarray(gold_ids, dtype=np.int64)
    pred_array = np.asarray(pred_ids, dtype=np.int64)
    if expected_event_ids is not None and list(expected_event_ids) != event_ids:
        raise PairedInferenceError(f"Event sequence mismatch for {cell_id}")
    if expected_gold_ids is not None and not np.array_equal(expected_gold_ids, gold_array):
        raise PairedInferenceError(f"Gold sequence mismatch for {cell_id}")
    point, class_f1 = _point_metrics(gold_array, pred_array, n_labels=n_labels)
    for metric_name in ("accuracy", "macro_f1"):
        expected_value = float(cell_manifest.get(metric_name, math.nan))
        if not math.isclose(point[metric_name], expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise PairedInferenceError(
                f"Metric parity failure for {cell_id}.{metric_name}: "
                f"recomputed={point[metric_name]}, manifest={expected_value}"
            )
    if len(event_ids) != int(metrics["num_samples"]):
        raise PairedInferenceError(f"Prediction/metrics sample count mismatch for {cell_id}")
    point = {
        **point,
        "evidence_count_mean": float(np.mean(evidence_count)),
        "prompt_token_count_mean": float(np.mean(prompt_token_count)),
    }
    return (
        CellData(
            cell_id=cell_id,
            selector_level=selector,
            controller_level=controller,
            predictions_path=predictions_path,
            predictions_sha256=actual_predictions_sha,
            pred_ids=pred_array,
            evidence_count=np.asarray(evidence_count, dtype=np.float64),
            prompt_token_count=np.asarray(prompt_token_count, dtype=np.float64),
            prompt_hashes=tuple(prompt_hashes),
            point_metrics=point,
            class_f1=class_f1,
        ),
        event_ids,
        gold_array,
    )


def _load_source(
    matrix_manifest_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], list[str], np.ndarray, list[CellData]]:
    matrix_manifest_path = matrix_manifest_path.resolve()
    matrix_dir = matrix_manifest_path.parent
    matrix = _read_json(matrix_manifest_path)
    if matrix.get("status") != "complete":
        raise PairedInferenceError(f"Matrix is not complete: {matrix_manifest_path}")
    cells_manifest = matrix.get("cells")
    if not isinstance(cells_manifest, list) or len(cells_manifest) != int(
        matrix.get("cell_count", -1)
    ):
        raise PairedInferenceError("Matrix cell_count disagrees with cells")
    if len(cells_manifest) != len(SELECTOR_LEVELS) * len(CONTROLLER_LEVELS):
        raise PairedInferenceError(f"Expected the frozen 6x3 grid, found {len(cells_manifest)} cells")

    raw_manifest_path = _resolve_declared_path(
        str(matrix.get("raw_logits_manifest") or ""), anchor=matrix_dir
    )
    expected_raw_sha = str(matrix.get("raw_logits_manifest_sha256") or "")
    if not raw_manifest_path.is_file() or _sha256_file(raw_manifest_path) != expected_raw_sha:
        raise PairedInferenceError(f"Raw-logits manifest drift: {raw_manifest_path}")
    raw_manifest = _read_json(raw_manifest_path)
    labels = [str(value) for value in raw_manifest.get("labels", [])]
    if len(labels) != int(raw_manifest.get("num_labels", -1)) or not labels:
        raise PairedInferenceError("Raw-logits manifest has an invalid label contract")
    scoring_fingerprint = str(matrix.get("raw_logits_scoring_fingerprint") or "")
    if scoring_fingerprint != str(raw_manifest.get("scoring_fingerprint") or ""):
        raise PairedInferenceError("Matrix/raw scoring fingerprint mismatch")

    if not bool(matrix.get("diagnostic_only", False)):
        gate_path = matrix_dir / str(matrix.get("equivalence_gate") or "")
        if not gate_path.is_file() or _sha256_file(gate_path) != str(
            matrix.get("equivalence_gate_sha256") or ""
        ):
            raise PairedInferenceError(f"Equivalence gate drift: {gate_path}")
        if not bool(_read_json(gate_path).get("passed", False)):
            raise PairedInferenceError("Formal source matrix has a failed equivalence gate")

    expected_event_ids: list[str] | None = None
    expected_gold_ids: np.ndarray | None = None
    cells: list[CellData] = []
    for cell_manifest in cells_manifest:
        cell, event_ids, gold_ids = _load_cell(
            matrix_dir=matrix_dir,
            cell_manifest=cell_manifest,
            labels=labels,
            scoring_fingerprint=scoring_fingerprint,
            expected_event_ids=expected_event_ids,
            expected_gold_ids=expected_gold_ids,
        )
        if expected_event_ids is None:
            expected_event_ids = event_ids
            expected_gold_ids = gold_ids
        cells.append(cell)
    assert expected_event_ids is not None and expected_gold_ids is not None

    actual_grid = {(cell.selector_level, cell.controller_level) for cell in cells}
    expected_grid = {
        (selector, controller)
        for selector in SELECTOR_LEVELS
        for controller in CONTROLLER_LEVELS
    }
    if actual_grid != expected_grid:
        raise PairedInferenceError(
            f"Factorial grid mismatch: missing={sorted(expected_grid - actual_grid)}, "
            f"extra={sorted(actual_grid - expected_grid)}"
        )
    cells_by_id = {cell.cell_id: cell for cell in cells}
    if len(cells_by_id) != len(cells):
        raise PairedInferenceError("Duplicate cell_id in matrix")
    ordered_cells = [
        cells_by_id[f"{selector}__{controller}"]
        for selector in SELECTOR_LEVELS
        for controller in CONTROLLER_LEVELS
    ]
    return matrix, matrix_dir, raw_manifest, expected_event_ids, expected_gold_ids, ordered_cells


def build_comparison_registry() -> list[ComparisonSpec]:
    specs: list[ComparisonSpec] = []
    matched = "matched_token_cap"
    for comparator in COMPARATOR_LEVELS:
        specs.append(
            ComparisonSpec(
                comparison_id=f"matched__baces_exact_minus_{comparator}",
                family_id="primary_matched_selector",
                role="primary",
                a_cell_id=f"baces_exact__{matched}",
                b_cell_id=f"{comparator}__{matched}",
            )
        )

    controller_pairs = (
        ("ordinal_replay_minmax5_10", "fixed5"),
        ("matched_token_cap", "fixed5"),
        ("matched_token_cap", "ordinal_replay_minmax5_10"),
    )
    for a_controller, b_controller in controller_pairs:
        specs.append(
            ComparisonSpec(
                comparison_id=f"baces_exact__{a_controller}_minus_{b_controller}",
                family_id="secondary_baces_controller",
                role="secondary",
                a_cell_id=f"baces_exact__{a_controller}",
                b_cell_id=f"baces_exact__{b_controller}",
            )
        )

    for controller in ("fixed5", "ordinal_replay_minmax5_10"):
        for comparator in COMPARATOR_LEVELS:
            specs.append(
                ComparisonSpec(
                    comparison_id=f"{controller}__baces_exact_minus_{comparator}",
                    family_id="exploratory_factorial",
                    role="exploratory",
                    a_cell_id=f"baces_exact__{controller}",
                    b_cell_id=f"{comparator}__{controller}",
                )
            )
    for selector in COMPARATOR_LEVELS:
        for a_controller, b_controller in controller_pairs:
            specs.append(
                ComparisonSpec(
                    comparison_id=f"{selector}__{a_controller}_minus_{b_controller}",
                    family_id="exploratory_factorial",
                    role="exploratory",
                    a_cell_id=f"{selector}__{a_controller}",
                    b_cell_id=f"{selector}__{b_controller}",
                )
            )
    if len(specs) != 33 or len({spec.comparison_id for spec in specs}) != len(specs):
        raise AssertionError("Frozen comparison registry must contain 33 unique contrasts")
    return specs


def _sample_indices(
    rng: np.random.Generator,
    *,
    gold_ids: np.ndarray,
    batch_size: int,
    stratified: bool,
) -> np.ndarray:
    sample_count = len(gold_ids)
    if not stratified:
        return rng.integers(0, sample_count, size=(batch_size, sample_count))
    chunks = []
    for label_id in range(int(np.max(gold_ids)) + 1):
        source = np.flatnonzero(gold_ids == label_id)
        if len(source) == 0:
            raise PairedInferenceError(f"Cannot stratify absent gold label {label_id}")
        chunks.append(rng.choice(source, size=(batch_size, len(source)), replace=True))
    return np.concatenate(chunks, axis=1)


def bootstrap_cell_scores(
    *,
    gold_ids: np.ndarray,
    cells: Sequence[CellData],
    n_labels: int,
    n_resamples: int,
    seed: int,
    stratified: bool,
    chunk_size: int = 64,
) -> tuple[dict[str, np.ndarray], int]:
    if n_resamples <= 0:
        raise PairedInferenceError("bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    cell_count = len(cells)
    values = {
        name: np.empty((n_resamples, cell_count), dtype=np.float64)
        for name in METRIC_NAMES
    }
    missing_class_replicates = 0
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = _sample_indices(
            rng,
            gold_ids=gold_ids,
            batch_size=stop - start,
            stratified=stratified,
        )
        gold_batch = gold_ids[indices]
        present_counts = np.stack(
            [np.sum(gold_batch == label_id, axis=1) for label_id in range(n_labels)], axis=1
        )
        missing_class_replicates += int(np.sum(np.any(present_counts == 0, axis=1)))
        for cell_index, cell in enumerate(cells):
            pred_batch = cell.pred_ids[indices]
            accuracy, macro_f1 = _metric_arrays(
                gold_batch, pred_batch, n_labels=n_labels
            )
            values["accuracy"][start:stop, cell_index] = accuracy
            values["macro_f1"][start:stop, cell_index] = macro_f1
            values["evidence_count_mean"][start:stop, cell_index] = np.mean(
                cell.evidence_count[indices], axis=1
            )
            values["prompt_token_count_mean"][start:stop, cell_index] = np.mean(
                cell.prompt_token_count[indices], axis=1
            )
    return values, missing_class_replicates


def _interval(values: np.ndarray, *, alpha: float) -> dict[str, float]:
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"low": float(low), "high": float(high)}


def _bootstrap_support(values: np.ndarray) -> float:
    return float(np.mean(values > 0.0) + 0.5 * np.mean(values == 0.0))


def _cohens_dz(differences: np.ndarray) -> float | None:
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < 2:
        return None
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0:
        return 0.0 if float(np.mean(values)) == 0.0 else None
    return float(np.mean(values) / standard_deviation)


def _mcnemar_exact_p(a_correct_only: int, b_correct_only: int) -> float:
    discordant = int(a_correct_only) + int(b_correct_only)
    if discordant == 0:
        return 1.0
    tail_limit = min(int(a_correct_only), int(b_correct_only))
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail_limit + 1)
    ]
    maximum = max(log_terms)
    tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in log_terms)
    return float(min(1.0, 2.0 * tail))


def _accuracy_pair_effects(
    gold_ids: np.ndarray,
    a_pred: np.ndarray,
    b_pred: np.ndarray,
) -> dict[str, Any]:
    a_correct = a_pred == gold_ids
    b_correct = b_pred == gold_ids
    both_correct = int(np.sum(a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    raw_odds_ratio: float | None = None
    odds_status = "finite"
    if b_only == 0:
        odds_status = "not_estimable_no_discordance" if a_only == 0 else "positive_infinity"
    else:
        raw_odds_ratio = float(a_only / b_only)
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant_correctness_count": discordant,
        "discordant_correctness_rate": float(discordant / len(gold_ids)),
        "discordant_net_advantage": float((a_only - b_only) / discordant)
        if discordant
        else 0.0,
        "a_win_probability_given_discordant": float(a_only / discordant)
        if discordant
        else 0.5,
        "matched_pairs_odds_ratio": raw_odds_ratio,
        "matched_pairs_odds_ratio_status": odds_status,
        "matched_pairs_odds_ratio_haldane_anscombe": float(
            (a_only + 0.5) / (b_only + 0.5)
        ),
        "mcnemar_exact_p_value_two_sided": _mcnemar_exact_p(a_only, b_only),
        "prediction_disagreement_count": int(np.sum(a_pred != b_pred)),
        "prediction_disagreement_rate": float(np.mean(a_pred != b_pred)),
    }


def paired_permutation_null(
    *,
    gold_ids: np.ndarray,
    a_cell: CellData,
    b_cell: CellData,
    n_labels: int,
    n_resamples: int,
    seed: int,
    chunk_size: int = 64,
) -> np.ndarray:
    if n_resamples <= 0:
        raise PairedInferenceError("permutation sample count must be positive")
    rng = np.random.default_rng(seed)
    sample_count = len(gold_ids)
    output = np.empty((n_resamples, len(METRIC_NAMES)), dtype=np.float64)
    resource_differences = (
        a_cell.evidence_count - b_cell.evidence_count,
        a_cell.prompt_token_count - b_cell.prompt_token_count,
    )
    gold_batch_template = gold_ids[None, :]
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        size = stop - start
        swap = rng.integers(0, 2, size=(size, sample_count), dtype=np.int8).astype(bool)
        a_pred = np.where(swap, b_cell.pred_ids[None, :], a_cell.pred_ids[None, :])
        b_pred = np.where(swap, a_cell.pred_ids[None, :], b_cell.pred_ids[None, :])
        gold_batch = np.broadcast_to(gold_batch_template, a_pred.shape)
        a_accuracy, a_macro = _metric_arrays(gold_batch, a_pred, n_labels=n_labels)
        b_accuracy, b_macro = _metric_arrays(gold_batch, b_pred, n_labels=n_labels)
        output[start:stop, 0] = a_macro - b_macro
        output[start:stop, 1] = a_accuracy - b_accuracy
        signs = np.where(swap, -1.0, 1.0)
        output[start:stop, 2] = np.mean(signs * resource_differences[0], axis=1)
        output[start:stop, 3] = np.mean(signs * resource_differences[1], axis=1)
    return output


def _permutation_summary(
    null_values: np.ndarray,
    *,
    observed_delta: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if observed_delta == 0.0:
        extreme_count = n_resamples
    else:
        extreme_count = int(
            np.sum(np.abs(null_values) + 1e-15 >= abs(float(observed_delta)))
        )
    p_value = float((extreme_count + 1) / (n_resamples + 1))
    return {
        "method": "paired_event_level_outcome_swap_monte_carlo",
        "alternative": "two_sided",
        "plus_one_correction": True,
        "samples": int(n_resamples),
        "seed": int(seed),
        "extreme_count": extreme_count,
        "p_value": p_value,
        "monte_carlo_standard_error": float(
            math.sqrt(p_value * (1.0 - p_value) / (n_resamples + 1))
        ),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise PairedInferenceError(f"Invalid p-values for Holm adjustment: {values}")
    count = len(values)
    order = sorted(range(count), key=lambda index: (values[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _cell_interval_rows(
    *,
    cells: Sequence[CellData],
    ordinary: Mapping[str, np.ndarray],
    stratified: Mapping[str, np.ndarray],
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        for metric_name in METRIC_NAMES:
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "selector_level": cell.selector_level,
                    "controller_level": cell.controller_level,
                    "metric": metric_name,
                    "point_estimate": float(cell.point_metrics[metric_name]),
                    "ordinary_paired_bootstrap_ci": _interval(
                        ordinary[metric_name][:, cell_index], alpha=alpha
                    ),
                    "gold_stratified_paired_bootstrap_ci": _interval(
                        stratified[metric_name][:, cell_index], alpha=alpha
                    ),
                }
            )
    return rows


def _comparison_rows(
    *,
    specs: Sequence[ComparisonSpec],
    cells: Sequence[CellData],
    gold_ids: np.ndarray,
    ordinary: Mapping[str, np.ndarray],
    stratified: Mapping[str, np.ndarray],
    permutation_samples: int,
    base_seed: int,
    alpha: float,
    n_labels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, list[int]]:
    cell_index = {cell.cell_id: index for index, cell in enumerate(cells)}
    cells_by_id = {cell.cell_id: cell for cell in cells}
    rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    null_arrays: list[np.ndarray] = []
    permutation_seeds: list[int] = []
    family_sizes: dict[str, int] = {}
    for spec in specs:
        family_sizes[spec.family_id] = family_sizes.get(spec.family_id, 0) + 1

    for spec in specs:
        a_cell = cells_by_id[spec.a_cell_id]
        b_cell = cells_by_id[spec.b_cell_id]
        a_index = cell_index[spec.a_cell_id]
        b_index = cell_index[spec.b_cell_id]
        seed = _stable_seed(base_seed, spec.family_id, spec.comparison_id, "permutation")
        null = paired_permutation_null(
            gold_ids=gold_ids,
            a_cell=a_cell,
            b_cell=b_cell,
            n_labels=n_labels,
            n_resamples=permutation_samples,
            seed=seed,
        )
        null_arrays.append(null)
        permutation_seeds.append(seed)
        metric_payload: dict[str, Any] = {}
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            a_value = float(a_cell.point_metrics[metric_name])
            b_value = float(b_cell.point_metrics[metric_name])
            delta = a_value - b_value
            ordinary_delta = (
                ordinary[metric_name][:, a_index] - ordinary[metric_name][:, b_index]
            )
            stratified_delta = (
                stratified[metric_name][:, a_index]
                - stratified[metric_name][:, b_index]
            )
            payload: dict[str, Any] = {
                "a_value": a_value,
                "b_value": b_value,
                "delta_a_minus_b": delta,
                "delta_percentage_points": 100.0 * delta
                if metric_name in {"macro_f1", "accuracy"}
                else None,
                "ordinary_paired_bootstrap": {
                    "ci": _interval(ordinary_delta, alpha=alpha),
                    "standard_error": float(np.std(ordinary_delta, ddof=1)),
                    "support_probability_a_gt_b": _bootstrap_support(ordinary_delta),
                },
                "gold_stratified_paired_bootstrap": {
                    "ci": _interval(stratified_delta, alpha=alpha),
                    "standard_error": float(np.std(stratified_delta, ddof=1)),
                    "support_probability_a_gt_b": _bootstrap_support(stratified_delta),
                },
                "paired_permutation": _permutation_summary(
                    null[:, metric_index],
                    observed_delta=delta,
                    n_resamples=permutation_samples,
                    seed=seed,
                ),
            }
            if metric_name == "accuracy":
                payload["paired_effects"] = _accuracy_pair_effects(
                    gold_ids, a_cell.pred_ids, b_cell.pred_ids
                )
            elif metric_name == "evidence_count_mean":
                differences = a_cell.evidence_count - b_cell.evidence_count
                payload["paired_effects"] = {
                    "median_event_difference": float(np.median(differences)),
                    "cohens_dz": _cohens_dz(differences),
                }
            elif metric_name == "prompt_token_count_mean":
                differences = a_cell.prompt_token_count - b_cell.prompt_token_count
                payload["paired_effects"] = {
                    "median_event_difference": float(np.median(differences)),
                    "cohens_dz": _cohens_dz(differences),
                }
            metric_payload[metric_name] = payload

        primary_family_size = family_sizes[spec.family_id]
        if spec.family_id == "primary_matched_selector":
            family_alpha = alpha / primary_family_size
            for metric_name in ("macro_f1", "accuracy"):
                delta_samples = (
                    ordinary[metric_name][:, a_index] - ordinary[metric_name][:, b_index]
                )
                metric_payload[metric_name]["bonferroni_familywise_ci"] = _interval(
                    delta_samples, alpha=family_alpha
                )

        accuracy_effects = metric_payload["accuracy"]["paired_effects"]
        row = {
            "comparison_id": spec.comparison_id,
            "family_id": spec.family_id,
            "family_size": primary_family_size,
            "role": spec.role,
            "direction": spec.direction,
            "a_cell_id": spec.a_cell_id,
            "b_cell_id": spec.b_cell_id,
            "a_selector_level": a_cell.selector_level,
            "a_controller_level": a_cell.controller_level,
            "b_selector_level": b_cell.selector_level,
            "b_controller_level": b_cell.controller_level,
            "sample_count": len(gold_ids),
            "metrics": metric_payload,
            "prediction_disagreement_count": accuracy_effects[
                "prediction_disagreement_count"
            ],
        }
        rows.append(row)
        for label_id in range(n_labels):
            classwise_rows.append(
                {
                    "comparison_id": spec.comparison_id,
                    "family_id": spec.family_id,
                    "role": spec.role,
                    "a_cell_id": spec.a_cell_id,
                    "b_cell_id": spec.b_cell_id,
                    "label_id": label_id,
                    "a_f1": float(a_cell.class_f1[label_id]),
                    "b_f1": float(b_cell.class_f1[label_id]),
                    "delta_f1_a_minus_b": float(
                        a_cell.class_f1[label_id] - b_cell.class_f1[label_id]
                    ),
                    "descriptive_only": True,
                }
            )

    for family_id in sorted({row["family_id"] for row in rows}):
        family_rows = [row for row in rows if row["family_id"] == family_id]
        for metric_name in METRIC_NAMES:
            raw_values = []
            sources = []
            for row in family_rows:
                metric = row["metrics"][metric_name]
                if metric_name == "accuracy":
                    raw_value = float(
                        metric["paired_effects"]["mcnemar_exact_p_value_two_sided"]
                    )
                    source = "two_sided_exact_mcnemar"
                else:
                    raw_value = float(metric["paired_permutation"]["p_value"])
                    source = "two_sided_paired_permutation"
                raw_values.append(raw_value)
                sources.append(source)
            adjusted_values = holm_adjust(raw_values)
            for row, raw_value, adjusted_value, source in zip(
                family_rows, raw_values, adjusted_values, sources
            ):
                row["metrics"][metric_name]["multiplicity"] = {
                    "family_id": family_id,
                    "family_size": len(family_rows),
                    "method": "holm_step_down_fwer",
                    "p_value_source": source,
                    "p_value_raw": raw_value,
                    "p_value_holm": adjusted_value,
                    "alpha": alpha,
                    "reject_raw": bool(raw_value <= alpha),
                    "reject_holm": bool(adjusted_value <= alpha),
                }
    return rows, classwise_rows, np.stack(null_arrays, axis=0), permutation_seeds


def _flatten_cell_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        ordinary = row["ordinary_paired_bootstrap_ci"]
        stratified = row["gold_stratified_paired_bootstrap_ci"]
        output.append(
            {
                "cell_id": row["cell_id"],
                "selector_level": row["selector_level"],
                "controller_level": row["controller_level"],
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
            multiplicity = metric["multiplicity"]
            paired_effects = metric.get("paired_effects", {})
            output.append(
                {
                    "comparison_id": row["comparison_id"],
                    "family_id": row["family_id"],
                    "family_size": row["family_size"],
                    "role": row["role"],
                    "metric": metric_name,
                    "direction": row["direction"],
                    "a_cell_id": row["a_cell_id"],
                    "b_cell_id": row["b_cell_id"],
                    "sample_count": row["sample_count"],
                    "a_value": metric["a_value"],
                    "b_value": metric["b_value"],
                    "delta_a_minus_b": metric["delta_a_minus_b"],
                    "delta_percentage_points": metric["delta_percentage_points"],
                    "ordinary_ci_low": ordinary["ci"]["low"],
                    "ordinary_ci_high": ordinary["ci"]["high"],
                    "ordinary_bootstrap_se": ordinary["standard_error"],
                    "ordinary_support_a_gt_b": ordinary[
                        "support_probability_a_gt_b"
                    ],
                    "stratified_ci_low": stratified["ci"]["low"],
                    "stratified_ci_high": stratified["ci"]["high"],
                    "permutation_p_value": permutation["p_value"],
                    "permutation_mcse": permutation["monte_carlo_standard_error"],
                    "p_value_source_for_holm": multiplicity["p_value_source"],
                    "p_value_raw_for_holm": multiplicity["p_value_raw"],
                    "p_value_holm": multiplicity["p_value_holm"],
                    "reject_holm": multiplicity["reject_holm"],
                    "mcnemar_exact_p_value": paired_effects.get(
                        "mcnemar_exact_p_value_two_sided"
                    ),
                    "a_only_correct": paired_effects.get("a_only_correct"),
                    "b_only_correct": paired_effects.get("b_only_correct"),
                    "matched_pairs_odds_ratio": paired_effects.get(
                        "matched_pairs_odds_ratio"
                    ),
                    "matched_pairs_odds_ratio_haldane_anscombe": paired_effects.get(
                        "matched_pairs_odds_ratio_haldane_anscombe"
                    ),
                    "cohens_dz": paired_effects.get("cohens_dz"),
                    "median_event_difference": paired_effects.get(
                        "median_event_difference"
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PairedInferenceError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    *,
    cell_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    permutation_samples: int,
    alpha: float,
    sanity_checks: Sequence[Mapping[str, Any]],
) -> str:
    macro_cells = {row["cell_id"]: row for row in cell_rows if row["metric"] == "macro_f1"}
    lines = [
        "# BACES Factorial Paired Inference",
        "",
        f"- analysis status: `{ANALYSIS_STATUS}`",
        f"- ordinary paired bootstrap samples: {bootstrap_samples}",
        f"- gold-stratified paired bootstrap samples: {bootstrap_samples}",
        f"- paired permutation samples per contrast: {permutation_samples}",
        f"- marginal CI level: {100.0 * (1.0 - alpha):.1f}%",
        "- delta direction: A minus B",
        "",
        "## Cell Macro-F1",
        "",
        "| cell | Macro-F1 | ordinary paired CI | gold-stratified CI |",
        "|---|---:|---:|---:|",
    ]
    for cell_id in [
        f"{selector}__{controller}"
        for selector in SELECTOR_LEVELS
        for controller in CONTROLLER_LEVELS
    ]:
        row = macro_cells[cell_id]
        ordinary = row["ordinary_paired_bootstrap_ci"]
        stratified = row["gold_stratified_paired_bootstrap_ci"]
        lines.append(
            f"| {cell_id} | {row['point_estimate']:.6f} | "
            f"[{ordinary['low']:.6f}, {ordinary['high']:.6f}] | "
            f"[{stratified['low']:.6f}, {stratified['high']:.6f}] |"
        )

    lines.extend(
        [
            "",
            "## Primary: matched-token BACES vs comparator",
            "",
            "| comparison | Macro-F1 delta (pp) | ordinary 95% CI | stratified 95% CI | Bonferroni family-wise CI | permutation p | Holm p | Accuracy delta (pp) | matched OR (H-A) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        if row["family_id"] != "primary_matched_selector":
            continue
        macro = row["metrics"]["macro_f1"]
        accuracy = row["metrics"]["accuracy"]
        ordinary_ci = macro["ordinary_paired_bootstrap"]["ci"]
        stratified_ci = macro["gold_stratified_paired_bootstrap"]["ci"]
        familywise_ci = macro["bonferroni_familywise_ci"]
        matched_or = accuracy["paired_effects"][
            "matched_pairs_odds_ratio_haldane_anscombe"
        ]
        lines.append(
            f"| {row['comparison_id']} | {macro['delta_percentage_points']:.3f} | "
            f"[{100 * ordinary_ci['low']:.3f}, {100 * ordinary_ci['high']:.3f}] | "
            f"[{100 * stratified_ci['low']:.3f}, {100 * stratified_ci['high']:.3f}] | "
            f"[{100 * familywise_ci['low']:.3f}, {100 * familywise_ci['high']:.3f}] | "
            f"{macro['paired_permutation']['p_value']:.6f} | "
            f"{macro['multiplicity']['p_value_holm']:.6f} | "
            f"{accuracy['delta_percentage_points']:.3f} | {matched_or:.3f} |"
        )

    lines.extend(
        [
            "",
            "## BACES controller effects",
            "",
            "| comparison | Macro-F1 delta (pp) | ordinary 95% CI | stratified 95% CI | permutation p | Holm p | Accuracy delta (pp) | evidence delta / dz | token delta / dz |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        if row["family_id"] != "secondary_baces_controller":
            continue
        macro = row["metrics"]["macro_f1"]
        accuracy = row["metrics"]["accuracy"]
        evidence = row["metrics"]["evidence_count_mean"]
        tokens = row["metrics"]["prompt_token_count_mean"]
        ordinary_ci = macro["ordinary_paired_bootstrap"]["ci"]
        stratified_ci = macro["gold_stratified_paired_bootstrap"]["ci"]
        evidence_dz = evidence["paired_effects"]["cohens_dz"]
        token_dz = tokens["paired_effects"]["cohens_dz"]
        evidence_dz_text = "NA" if evidence_dz is None else f"{evidence_dz:.3f}"
        token_dz_text = "NA" if token_dz is None else f"{token_dz:.3f}"
        lines.append(
            f"| {row['comparison_id']} | {macro['delta_percentage_points']:.3f} | "
            f"[{100 * ordinary_ci['low']:.3f}, {100 * ordinary_ci['high']:.3f}] | "
            f"[{100 * stratified_ci['low']:.3f}, {100 * stratified_ci['high']:.3f}] | "
            f"{macro['paired_permutation']['p_value']:.6f} | "
            f"{macro['multiplicity']['p_value_holm']:.6f} | "
            f"{accuracy['delta_percentage_points']:.3f} | "
            f"{evidence['delta_a_minus_b']:.3f} / {evidence_dz_text} | "
            f"{tokens['delta_a_minus_b']:.3f} / {token_dz_text} |"
        )

    lines.extend(["", "## Sanity gates", ""])
    for check in sanity_checks:
        lines.append(
            f"- `{check['a_cell_id']}` vs `{check['b_cell_id']}`: "
            f"identical prompts={str(check['prompt_sequences_identical']).lower()}, "
            f"identical predictions={str(check['predictions_identical']).lower()}"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These intervals and tests quantify event-level sampling uncertainty for one fixed validation set, evidence-map cache, checkpoint, and frozen verifier. They do not remove verifier-training-distribution bias, verifier seed variance, map/retrieval variance, or post-hoc comparison-selection bias. Marginal bootstrap CIs are not multiplicity-adjusted; Holm correction applies to the declared p-value families. The primary family additionally records Bonferroni family-wise bootstrap intervals.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_existing(
    *,
    output_dir: Path,
    source_matrix_sha256: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    existing = _read_json(manifest_path)
    if (
        existing.get("schema_version") != SCHEMA_VERSION
        or existing.get("status") != "complete"
        or str((existing.get("source", {}) or {}).get("matrix_manifest_sha256") or "")
        != source_matrix_sha256
        or existing.get("settings") != dict(settings)
    ):
        raise PairedInferenceError(
            f"Existing paired artifact is incompatible: {output_dir}; pass --force"
        )
    artifacts = existing.get("artifacts", {}) or {}
    for name, artifact in artifacts.items():
        path = output_dir / str((artifact or {}).get("path") or "")
        expected_sha = str((artifact or {}).get("sha256") or "")
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise PairedInferenceError(f"Existing paired artifact drifted ({name}): {path}")
    return existing


def materialize_paired_inference(
    *,
    matrix_manifest_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 20_000,
    seed: int = 20_260_713,
    alpha: float = 0.05,
    force: bool = False,
) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0:
        raise PairedInferenceError(f"alpha must be in (0,1), got {alpha}")
    matrix, matrix_dir, raw_manifest, event_ids, gold_ids, cells = _load_source(
        matrix_manifest_path
    )
    matrix_manifest_path = matrix_manifest_path.resolve()
    output_dir = output_dir.resolve()
    ordinary_seed = _stable_seed(seed, "all_cells", "bootstrap", "ordinary")
    stratified_seed = _stable_seed(seed, "all_cells", "bootstrap", "gold_stratified")
    settings = {
        "alpha": float(alpha),
        "bootstrap": {
            "ordinary_samples": int(bootstrap_samples),
            "ordinary_seed": int(ordinary_seed),
            "stratified_samples": int(bootstrap_samples),
            "stratified_seed": int(stratified_seed),
            "ci_method": "paired_percentile",
            "primary_resampling_unit": "event",
            "sensitivity_stratification": "gold_id",
        },
        "permutation": {
            "samples_per_comparison": int(permutation_samples),
            "method": "paired_event_level_outcome_swap_monte_carlo",
            "alternative": "two_sided",
            "plus_one_correction": True,
        },
        "base_seed": int(seed),
        "derived_seed_rule": "uint32(first8_sha256(base_seed:family:comparison:operation))",
        "metrics": list(METRIC_NAMES),
        "multiplicity": {
            "method": "holm_step_down_fwer",
            "families": [
                "primary_matched_selector",
                "secondary_baces_controller",
                "exploratory_factorial",
            ],
            "separate_by_metric": True,
            "accuracy_p_value_source": "two_sided_exact_mcnemar",
            "other_metric_p_value_source": "two_sided_paired_permutation",
        },
    }
    source_matrix_sha = _sha256_file(matrix_manifest_path)
    if output_dir.exists() and not force:
        return _validate_existing(
            output_dir=output_dir,
            source_matrix_sha256=source_matrix_sha,
            settings=settings,
        )

    cells_by_id = {cell.cell_id: cell for cell in cells}
    sanity_checks: list[dict[str, Any]] = []
    for a_cell_id, b_cell_id in EXPECTED_IDENTICAL_PROMPT_PAIRS:
        a_cell = cells_by_id[a_cell_id]
        b_cell = cells_by_id[b_cell_id]
        prompts_identical = a_cell.prompt_hashes == b_cell.prompt_hashes
        predictions_identical = np.array_equal(a_cell.pred_ids, b_cell.pred_ids)
        if not prompts_identical or not predictions_identical:
            raise PairedInferenceError(
                f"Frozen identical-prompt sanity gate failed: {a_cell_id} vs {b_cell_id}"
            )
        sanity_checks.append(
            {
                "a_cell_id": a_cell_id,
                "b_cell_id": b_cell_id,
                "prompt_sequences_identical": prompts_identical,
                "predictions_identical": predictions_identical,
            }
        )

    n_labels = len(raw_manifest["labels"])
    ordinary, ordinary_missing = bootstrap_cell_scores(
        gold_ids=gold_ids,
        cells=cells,
        n_labels=n_labels,
        n_resamples=bootstrap_samples,
        seed=ordinary_seed,
        stratified=False,
    )
    stratified, stratified_missing = bootstrap_cell_scores(
        gold_ids=gold_ids,
        cells=cells,
        n_labels=n_labels,
        n_resamples=bootstrap_samples,
        seed=stratified_seed,
        stratified=True,
    )
    cell_rows = _cell_interval_rows(
        cells=cells,
        ordinary=ordinary,
        stratified=stratified,
        alpha=alpha,
    )
    specs = build_comparison_registry()
    comparison_rows, classwise_rows, permutation_null, permutation_seeds = _comparison_rows(
        specs=specs,
        cells=cells,
        gold_ids=gold_ids,
        ordinary=ordinary,
        stratified=stratified,
        permutation_samples=permutation_samples,
        base_seed=seed,
        alpha=alpha,
        n_labels=n_labels,
    )

    # Degenerate identical-prompt comparisons are an end-to-end resampling sanity check.
    for a_cell_id, b_cell_id in EXPECTED_IDENTICAL_PROMPT_PAIRS:
        matching = [
            row
            for row in comparison_rows
            if {row["a_cell_id"], row["b_cell_id"]} == {a_cell_id, b_cell_id}
        ]
        if len(matching) != 1:
            raise PairedInferenceError(
                f"Expected one registered sanity comparison for {a_cell_id} vs {b_cell_id}"
            )
        for metric_name in METRIC_NAMES:
            metric = matching[0]["metrics"][metric_name]
            ordinary_ci = metric["ordinary_paired_bootstrap"]["ci"]
            if (
                metric["delta_a_minus_b"] != 0.0
                or ordinary_ci != {"low": 0.0, "high": 0.0}
                or metric["paired_permutation"]["p_value"] != 1.0
            ):
                raise PairedInferenceError(
                    f"Degenerate resampling sanity gate failed for {matching[0]['comparison_id']} "
                    f"metric={metric_name}"
                )

    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(staging / "cell_intervals.jsonl", cell_rows)
        _write_csv(staging / "cell_intervals.csv", _flatten_cell_rows(cell_rows))
        _write_jsonl(staging / "comparisons.jsonl", comparison_rows)
        _write_csv(staging / "comparisons.csv", _flatten_comparison_rows(comparison_rows))
        _write_jsonl(staging / "classwise_effects.jsonl", classwise_rows)
        _write_csv(staging / "classwise_effects.csv", classwise_rows)
        report = _render_report(
            cell_rows=cell_rows,
            comparison_rows=comparison_rows,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            alpha=alpha,
            sanity_checks=sanity_checks,
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        with (staging / "bootstrap_samples.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                cell_ids=np.asarray([cell.cell_id for cell in cells]),
                metric_names=np.asarray(METRIC_NAMES),
                ordinary_macro_f1=ordinary["macro_f1"],
                ordinary_accuracy=ordinary["accuracy"],
                ordinary_evidence_count_mean=ordinary["evidence_count_mean"],
                ordinary_prompt_token_count_mean=ordinary["prompt_token_count_mean"],
                stratified_macro_f1=stratified["macro_f1"],
                stratified_accuracy=stratified["accuracy"],
                stratified_evidence_count_mean=stratified["evidence_count_mean"],
                stratified_prompt_token_count_mean=stratified["prompt_token_count_mean"],
            )
        with (staging / "permutation_null.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                comparison_ids=np.asarray([spec.comparison_id for spec in specs]),
                metric_names=np.asarray(METRIC_NAMES),
                seeds=np.asarray(permutation_seeds, dtype=np.uint64),
                null_deltas=permutation_null,
            )

        artifact_names = (
            "cell_intervals.jsonl",
            "cell_intervals.csv",
            "comparisons.jsonl",
            "comparisons.csv",
            "classwise_effects.jsonl",
            "classwise_effects.csv",
            "report.md",
            "bootstrap_samples.npz",
            "permutation_null.npz",
        )
        artifacts = {
            name: {
                "path": name,
                "sha256": _sha256_file(staging / name),
                "size": (staging / name).stat().st_size,
            }
            for name in artifact_names
        }
        gate_path = matrix_dir / str(matrix.get("equivalence_gate") or "")
        source_payload = {
            "matrix_manifest": str(matrix_manifest_path),
            "matrix_manifest_sha256": source_matrix_sha,
            "matrix_schema_version": matrix.get("schema_version"),
            "matrix_diagnostic_only": bool(matrix.get("diagnostic_only", False)),
            "equivalence_gate": str(gate_path) if gate_path.is_file() else None,
            "equivalence_gate_sha256": _sha256_file(gate_path)
            if gate_path.is_file()
            else None,
            "raw_logits_manifest": str(
                _resolve_declared_path(
                    str(matrix.get("raw_logits_manifest") or ""), anchor=matrix_dir
                )
            ),
            "raw_logits_manifest_sha256": matrix.get("raw_logits_manifest_sha256"),
            "raw_logits_scoring_fingerprint": matrix.get(
                "raw_logits_scoring_fingerprint"
            ),
            "raw_logits_execution_fingerprint": matrix.get(
                "raw_logits_execution_fingerprint"
            ),
            "checkpoint": matrix.get("checkpoint"),
            "prediction_artifacts": [
                {
                    "cell_id": cell.cell_id,
                    "path": str(cell.predictions_path),
                    "sha256": cell.predictions_sha256,
                }
                for cell in cells
            ],
        }
        implementation_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_status": ANALYSIS_STATUS,
            "split": matrix.get("split"),
            "sample_count": len(event_ids),
            "cell_count": len(cells),
            "comparison_count": len(specs),
            "alignment": {
                "primary_key": "event_id",
                "secondary_order_check": "sample_idx",
                "event_id_sequence_sha256": _event_sequence_sha256(event_ids),
                "gold_id_counts": {
                    str(label_id): int(np.sum(gold_ids == label_id))
                    for label_id in range(n_labels)
                },
                "all_cells_exactly_aligned": True,
            },
            "labels": list(raw_manifest["labels"]),
            "comparison_registry": [spec.__dict__ for spec in specs],
            "settings": settings,
            "bootstrap_audit": {
                "ordinary_missing_gold_class_replicates": ordinary_missing,
                "stratified_missing_gold_class_replicates": stratified_missing,
            },
            "sanity_checks": sanity_checks,
            "source": source_payload,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "bit_generator": "PCG64",
            },
            "implementation": {
                "path": str(implementation_path),
                "sha256": _sha256_file(implementation_path),
            },
            "interpretation_boundary": (
                "Event-level post-hoc validation uncertainty under one fixed evidence-map "
                "cache, checkpoint, and frozen verifier; not a causal selector comparison "
                "and not verifier-seed, map, retrieval, or test-set uncertainty."
            ),
            "artifacts": artifacts,
        }
        _write_json(staging / "manifest.json", manifest)
        _promote_directory(staging, output_dir, force=force)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutation-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = materialize_paired_inference(
        matrix_manifest_path=Path(args.matrix_manifest),
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
                "comparison_count": result["comparison_count"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
