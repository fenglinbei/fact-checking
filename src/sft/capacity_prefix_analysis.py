"""Materialize prefix-capacity oracles and policy regret from frozen raw logits.

The analysis unit is one fact-checking event under one frozen selector order.  A
prefix cell requests a capacity K, while ``evidence_count`` records the capacity
that was actually prompt-feasible.  Requested capacities that map to the same
prompt are deduplicated before the oracle is selected.

This is a post-hoc diagnostic: gold labels define the oracle and are never an
input to a deployable capacity policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "capacity_prefix_oracle_regret_v0_1"
ANALYSIS_STATUS = "post_hoc_validation_frozen_verifier_diagnostic"
_PREFIX_CONTROLLER_RE = re.compile(r"^prefix_k(\d+)$")
PROMPT_CACHE_KEY_VERSION = "label_token_prompt_ids_plus_prefix_v0_1"
_REQUESTED_K_FIELDS = (
    "requested_prefix_k",
    "capacity_k",
    "capacity_requested_k",
)


class CapacityAnalysisError(ValueError):
    """Raised when the prefix matrix or policy contract is violated."""


@dataclass(frozen=True)
class PrefixObservation:
    selector_level: str
    event_id: str
    sample_idx: int
    requested_k: int
    realized_k: int
    prompt_token_count: int
    prompt_hash: str
    unique_idx: int
    gold_id: int
    logits: np.ndarray

    @property
    def prompt_identity(self) -> tuple[int, str]:
        return self.unique_idx, self.prompt_hash


@dataclass(frozen=True)
class PrefixAction:
    observation: PrefixObservation
    requested_k_aliases: tuple[int, ...]
    raw_nll: float
    balanced_nll: float
    token_penalty: float
    objective: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_cache_key(prompt_input_ids: Sequence[int], label_prefix: str) -> str:
    return _sha256_json(
        {
            "cache_key_version": PROMPT_CACHE_KEY_VERSION,
            "label_prefix": label_prefix,
            "prompt_input_ids": [int(token_id) for token_id in prompt_input_ids],
        }
    )


def _event_sequence_sha256(event_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for event_id in event_ids:
        digest.update(str(event_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CapacityAnalysisError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CapacityAnalysisError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CapacityAnalysisError(f"Expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise CapacityAnalysisError(f"Required JSONL file does not exist: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CapacityAnalysisError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise CapacityAnalysisError(f"Expected object in {path}:{line_number}")
            yield line_number, value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_declared_path(value: str, *, anchor: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, anchor / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


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


def _requested_k(*cells: Mapping[str, Any]) -> int | None:
    values: list[int] = []
    for cell in cells:
        for field in _REQUESTED_K_FIELDS:
            if cell.get(field) is not None:
                values.append(int(cell[field]))
        controller = str(cell.get("controller_level") or "")
        match = _PREFIX_CONTROLLER_RE.fullmatch(controller)
        if match:
            values.append(int(match.group(1)))
    if not values:
        return None
    if len(set(values)) != 1:
        raise CapacityAnalysisError(f"Conflicting requested K declarations: {values}")
    if values[0] <= 0:
        raise CapacityAnalysisError(f"requested K must be positive, got {values[0]}")
    return values[0]


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload[key])
    except (KeyError, OSError, ValueError) as exc:
        raise CapacityAnalysisError(f"Cannot load {key!r} from raw logits: {path}") from exc


def _verify_raw_cache_index(
    *,
    raw_manifest_path: Path,
    raw_manifest: Mapping[str, Any],
    input_manifest_path: Path,
    input_manifest: Mapping[str, Any],
    raw_gold_ids: np.ndarray,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = raw_manifest_path.parent / str(raw_manifest.get("index_file") or "")
    if not index_path.is_file() or _sha256_file(index_path) != str(
        raw_manifest.get("index_sha256") or ""
    ):
        raise CapacityAnalysisError(f"Raw-logits index drift: {index_path}")
    index_rows = [row for _, row in _iter_jsonl(index_path)]

    unique_rows_path = input_manifest_path.parent / str(
        input_manifest.get("unique_rows_file") or ""
    )
    if not unique_rows_path.is_file() or _sha256_file(unique_rows_path) != str(
        input_manifest.get("unique_rows_sha256") or ""
    ):
        raise CapacityAnalysisError(f"Prepared unique-row cache drift: {unique_rows_path}")
    unique_rows = [row for _, row in _iter_jsonl(unique_rows_path)]
    expected_count = len(raw_gold_ids)
    if (
        len(index_rows) != expected_count
        or len(unique_rows) != expected_count
        or int(raw_manifest.get("num_unique_prompts", -1)) != expected_count
        or int(input_manifest.get("unique_prompt_count", -1)) != expected_count
    ):
        raise CapacityAnalysisError(
            "Raw index/prepared unique-row count mismatch: "
            f"index={len(index_rows)}, unique={len(unique_rows)}, expected={expected_count}"
        )
    identity_fields = (
        "event_id",
        "gold_id",
        "prompt_cache_key",
        "prompt_input_ids_sha256",
    )
    for unique_idx, (index_row, unique_row) in enumerate(zip(index_rows, unique_rows)):
        if int(index_row.get("unique_idx", -1)) != unique_idx or int(
            unique_row.get("unique_idx", -1)
        ) != unique_idx:
            raise CapacityAnalysisError(
                f"Raw/prepared unique indices are not exactly contiguous at {unique_idx}"
            )
        for field in identity_fields:
            if index_row.get(field) != unique_row.get(field):
                raise CapacityAnalysisError(
                    f"Raw index/prepared unique-row {field} mismatch at unique_idx={unique_idx}"
                )
        if int(index_row.get("gold_id", -1)) != int(raw_gold_ids[unique_idx]):
            raise CapacityAnalysisError(
                f"Raw index/NPZ gold mismatch at unique_idx={unique_idx}"
            )
    return index_path, unique_rows_path, index_rows, unique_rows


def _load_prefix_source(
    matrix_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[str],
    list[str],
    np.ndarray,
    dict[str, dict[int, list[PrefixObservation]]],
    dict[str, Any],
]:
    matrix_manifest_path = matrix_manifest_path.resolve()
    matrix_dir = matrix_manifest_path.parent
    matrix = _read_json(matrix_manifest_path)
    if matrix.get("status") != "complete":
        raise CapacityAnalysisError(f"Prefix matrix is not complete: {matrix_manifest_path}")

    raw_manifest_path = _resolve_declared_path(
        str(matrix.get("raw_logits_manifest") or ""), anchor=matrix_dir
    )
    expected_raw_manifest_sha = str(matrix.get("raw_logits_manifest_sha256") or "")
    if not raw_manifest_path.is_file() or _sha256_file(
        raw_manifest_path
    ) != expected_raw_manifest_sha:
        raise CapacityAnalysisError(f"Raw-logits manifest drift: {raw_manifest_path}")
    raw_manifest = _read_json(raw_manifest_path)
    if raw_manifest.get("status") != "complete":
        raise CapacityAnalysisError(f"Raw-logits cache is incomplete: {raw_manifest_path}")
    labels = [str(label) for label in raw_manifest.get("labels", [])]
    if not labels or len(labels) != int(raw_manifest.get("num_labels", -1)):
        raise CapacityAnalysisError("Raw-logits manifest has an invalid label contract")
    if str(matrix.get("raw_logits_scoring_fingerprint") or "") != str(
        raw_manifest.get("scoring_fingerprint") or ""
    ):
        raise CapacityAnalysisError("Matrix/raw-logits scoring fingerprint mismatch")
    logits_path = raw_manifest_path.parent / str(raw_manifest.get("raw_logits_file") or "")
    if not logits_path.is_file() or _sha256_file(logits_path) != str(
        raw_manifest.get("raw_logits_sha256") or ""
    ):
        raise CapacityAnalysisError(f"Raw-logits NPZ drift: {logits_path}")
    logits = _load_npz_array(logits_path, "label_logits").astype(np.float64, copy=False)
    raw_gold_ids = _load_npz_array(logits_path, "gold_ids").astype(np.int64, copy=False)
    raw_unique_indices = _load_npz_array(logits_path, "unique_indices").astype(
        np.int64, copy=False
    )
    if (
        logits.ndim != 2
        or logits.shape[1] != len(labels)
        or raw_gold_ids.shape != (len(logits),)
        or raw_unique_indices.shape != (len(logits),)
        or not np.array_equal(raw_unique_indices, np.arange(len(logits), dtype=np.int64))
    ):
        raise CapacityAnalysisError(
            "Invalid raw-logit arrays: "
            f"logits={logits.shape}, gold={raw_gold_ids.shape}, "
            f"unique_indices={raw_unique_indices.shape}"
        )
    if not np.all(np.isfinite(logits)):
        raise CapacityAnalysisError("Raw logits contain non-finite values")
    if str(matrix.get("raw_logits_sha256") or "") != str(
        raw_manifest.get("raw_logits_sha256") or ""
    ):
        raise CapacityAnalysisError("Matrix/raw-logits NPZ SHA mismatch")

    input_manifest_path = _resolve_declared_path(
        str(matrix.get("input_manifest") or ""), anchor=matrix_dir
    )
    expected_input_sha = str(matrix.get("input_manifest_sha256") or "")
    if not input_manifest_path.is_file() or _sha256_file(input_manifest_path) != expected_input_sha:
        raise CapacityAnalysisError(f"Prepared-input manifest drift: {input_manifest_path}")
    if str(raw_manifest.get("input_manifest_sha256") or "") != expected_input_sha:
        raise CapacityAnalysisError("Prepared-input/raw-logits provenance mismatch")
    input_manifest = _read_json(input_manifest_path)
    index_path, unique_rows_path, raw_index_rows, prepared_unique_rows = (
        _verify_raw_cache_index(
            raw_manifest_path=raw_manifest_path,
            raw_manifest=raw_manifest,
            input_manifest_path=input_manifest_path,
            input_manifest=input_manifest,
            raw_gold_ids=raw_gold_ids,
        )
    )
    input_cells = {
        str(cell.get("cell_id") or ""): cell for cell in input_manifest.get("cells", [])
    }
    if not input_cells or "" in input_cells:
        raise CapacityAnalysisError("Prepared-input manifest has invalid cells")
    source_matrix_path: Path | None = None
    source_matrix_cells: dict[str, Mapping[str, Any]] = {}
    source_matrix_sha256 = str(input_manifest.get("matrix_manifest_sha256") or "")
    prefix_gate_path: Path | None = None
    prefix_gate_sha256 = ""
    source_matrix_schema = str(input_manifest.get("matrix_schema_version") or "")
    if source_matrix_schema == "baces_capacity_prefix_matrix_v0_1":
        source_matrix_path = _resolve_declared_path(
            str(input_manifest.get("matrix_manifest") or ""),
            anchor=input_manifest_path.parent,
        )
        if not source_matrix_path.is_file() or _sha256_file(
            source_matrix_path
        ) != source_matrix_sha256:
            raise CapacityAnalysisError(
                f"Source capacity-matrix manifest drift: {source_matrix_path}"
            )
        source_matrix = _read_json(source_matrix_path)
        raw_source_cells = source_matrix.get("cells")
        if not isinstance(raw_source_cells, list):
            raise CapacityAnalysisError("Source capacity matrix has invalid cells")
        source_matrix_cells = {
            str(cell.get("cell_id") or ""): cell
            for cell in raw_source_cells
            if isinstance(cell, Mapping)
        }
        if "" in source_matrix_cells or len(source_matrix_cells) != len(
            raw_source_cells
        ):
            raise CapacityAnalysisError("Source capacity matrix has duplicate/empty cells")
        prefix_gate_path = source_matrix_path.parent / str(
            source_matrix.get("prefix_integrity_gate") or ""
        )
        prefix_gate_sha256 = str(
            source_matrix.get("prefix_integrity_gate_sha256") or ""
        )
        if not prefix_gate_path.is_file() or _sha256_file(
            prefix_gate_path
        ) != prefix_gate_sha256:
            raise CapacityAnalysisError(f"Prefix integrity gate drift: {prefix_gate_path}")
        prefix_gate = _read_json(prefix_gate_path)
        if prefix_gate.get("passed") is not True:
            raise CapacityAnalysisError("Prefix integrity gate is not passed=true")

    matrix_cells = matrix.get("cells")
    if not isinstance(matrix_cells, list):
        raise CapacityAnalysisError("Prefix matrix cells must be a list")
    prefix_cells: list[tuple[Mapping[str, Any], Mapping[str, Any], int]] = []
    for matrix_cell in matrix_cells:
        cell_id = str(matrix_cell.get("cell_id") or "")
        input_cell = input_cells.get(cell_id)
        if input_cell is None:
            raise CapacityAnalysisError(f"Matrix cell is absent from prepared input: {cell_id}")
        requested_k = _requested_k(matrix_cell, input_cell)
        if requested_k is not None:
            prefix_cells.append((matrix_cell, input_cell, requested_k))
    if not prefix_cells:
        raise CapacityAnalysisError("Matrix has no prefix_k cells")

    observations: dict[str, dict[int, list[PrefixObservation]]] = {}
    expected_event_ids: list[str] | None = None
    expected_gold_ids: list[int] | None = None
    seen_selector_k: set[tuple[str, int]] = set()
    for matrix_cell, input_cell, requested_k in prefix_cells:
        cell_id = str(matrix_cell["cell_id"])
        selector = str(matrix_cell.get("selector_level") or input_cell.get("selector_level") or "")
        if not selector:
            raise CapacityAnalysisError(f"Missing selector_level for {cell_id}")
        controller = str(
            matrix_cell.get("controller_level")
            or input_cell.get("controller_level")
            or ""
        )
        if not controller:
            raise CapacityAnalysisError(f"Missing controller_level for {cell_id}")
        if (selector, requested_k) in seen_selector_k:
            raise CapacityAnalysisError(f"Duplicate prefix cell for {selector}, K={requested_k}")
        seen_selector_k.add((selector, requested_k))

        mapping_path = input_manifest_path.parent / str(input_cell.get("mapping_file") or "")
        if not mapping_path.is_file() or _sha256_file(mapping_path) != str(
            input_cell.get("mapping_sha256") or ""
        ):
            raise CapacityAnalysisError(f"Prepared mapping drift for {cell_id}: {mapping_path}")
        mappings = [row for _, row in _iter_jsonl(mapping_path)]
        mappings.sort(key=lambda row: int(row.get("cell_sample_idx", -1)))

        source_build_rows: list[dict[str, Any]] | None = None
        if source_matrix_cells:
            source_cell = source_matrix_cells.get(cell_id)
            if source_cell is None:
                raise CapacityAnalysisError(
                    f"Prepared cell is absent from source capacity matrix: {cell_id}"
                )
            source_build_path = _resolve_declared_path(
                str(input_cell.get("source_build") or ""),
                anchor=input_manifest_path.parent,
            )
            expected_source_build_sha = str(
                input_cell.get("source_build_sha256") or ""
            )
            declared_matrix_build = _resolve_declared_path(
                str(source_cell.get("build_file") or ""),
                anchor=source_matrix_path.parent if source_matrix_path else matrix_dir,
            )
            if (
                source_build_path != declared_matrix_build
                or not source_build_path.is_file()
                or _sha256_file(source_build_path) != expected_source_build_sha
                or expected_source_build_sha
                != str(source_cell.get("build_sha256") or "")
            ):
                raise CapacityAnalysisError(
                    f"Prepared/source-matrix build provenance mismatch for {cell_id}"
                )
            source_build_rows = [row for _, row in _iter_jsonl(source_build_path)]

        predictions_path = matrix_dir / str(matrix_cell.get("predictions_file") or "")
        if not predictions_path.is_file() or _sha256_file(predictions_path) != str(
            matrix_cell.get("predictions_sha256") or ""
        ):
            raise CapacityAnalysisError(
                f"Prediction artifact drift for {cell_id}: {predictions_path}"
            )
        predictions = [row for _, row in _iter_jsonl(predictions_path)]
        predictions.sort(key=lambda row: int(row.get("sample_idx", -1)))
        if (
            len(predictions) != len(mappings)
            or not predictions
            or (
                source_build_rows is not None
                and len(source_build_rows) != len(mappings)
            )
        ):
            raise CapacityAnalysisError(f"Prediction/mapping row-count mismatch for {cell_id}")

        cell_events: list[str] = []
        cell_gold: list[int] = []
        cell_observations: list[PrefixObservation] = []
        for sample_idx, (mapping, prediction) in enumerate(zip(mappings, predictions)):
            if int(mapping.get("cell_sample_idx", -1)) != sample_idx or int(
                prediction.get("sample_idx", -1)
            ) != sample_idx:
                raise CapacityAnalysisError(f"Non-contiguous sample_idx in {cell_id}")
            event_id = str(mapping.get("event_id") or "")
            gold_id = int(mapping.get("gold_id", -1))
            unique_idx = int(mapping.get("unique_idx", -1))
            if (
                not event_id
                or event_id != str(prediction.get("event_id") or "")
                or gold_id != int(prediction.get("gold_id", -1))
                or unique_idx != int(prediction.get("raw_logits_unique_idx", -1))
                or str(mapping.get("cell_id") or "") not in ("", cell_id)
                or str(prediction.get("cell_id") or "") != cell_id
                or str(prediction.get("selector_level") or "") != selector
                or str(prediction.get("controller_level") or "") != controller
                or str(prediction.get("scoring_fingerprint") or "")
                != str(raw_manifest.get("scoring_fingerprint") or "")
            ):
                raise CapacityAnalysisError(
                    f"Prediction/mapping identity mismatch in {cell_id}, sample_idx={sample_idx}"
                )
            if not 0 <= gold_id < len(labels) or not 0 <= unique_idx < len(logits):
                raise CapacityAnalysisError(
                    f"Invalid gold/raw index in {cell_id}, event={event_id}: {gold_id}/{unique_idx}"
                )
            if int(raw_gold_ids[unique_idx]) != gold_id:
                raise CapacityAnalysisError(
                    f"Raw-logit gold mismatch in {cell_id}, event={event_id}"
                )
            unique_row = prepared_unique_rows[unique_idx]
            raw_index_row = raw_index_rows[unique_idx]
            mapping_identity = {
                "event_id": event_id,
                "gold_id": gold_id,
                "prompt_cache_key": str(mapping.get("prompt_cache_key") or ""),
                "prompt_input_ids_sha256": str(
                    mapping.get("prompt_input_ids_sha256") or ""
                ),
                "evidence_count": int(mapping.get("evidence_count", -1)),
                "prompt_token_count": int(mapping.get("prompt_token_count", -1)),
            }
            unique_identity = {
                "event_id": str(unique_row.get("event_id") or ""),
                "gold_id": int(unique_row.get("gold_id", -1)),
                "prompt_cache_key": str(unique_row.get("prompt_cache_key") or ""),
                "prompt_input_ids_sha256": str(
                    unique_row.get("prompt_input_ids_sha256") or ""
                ),
                "evidence_count": int(unique_row.get("evidence_count", -1)),
                "prompt_token_count": int(unique_row.get("prompt_token_count", -1)),
            }
            if mapping_identity != unique_identity:
                raise CapacityAnalysisError(
                    f"Mapping/unique-row identity mismatch in {cell_id}, "
                    f"event={event_id}, unique_idx={unique_idx}"
                )
            if any(
                raw_index_row.get(field) != unique_row.get(field)
                for field in (
                    "event_id",
                    "gold_id",
                    "prompt_cache_key",
                    "prompt_input_ids_sha256",
                )
            ):
                raise CapacityAnalysisError(
                    f"Raw index/unique-row identity drift at unique_idx={unique_idx}"
                )
            if (
                str(prediction.get("prompt_cache_key") or "")
                != mapping_identity["prompt_cache_key"]
                or str(prediction.get("prompt_input_ids_sha256") or "")
                != mapping_identity["prompt_input_ids_sha256"]
                or int(prediction.get("evidence_count", -1))
                != mapping_identity["evidence_count"]
                or int(prediction.get("prompt_token_count", -1))
                != mapping_identity["prompt_token_count"]
            ):
                raise CapacityAnalysisError(
                    f"Prediction/unique-row identity mismatch in {cell_id}, "
                    f"event={event_id}, unique_idx={unique_idx}"
                )
            if source_build_rows is not None:
                source_build = source_build_rows[sample_idx]
                raw_prompt_ids = source_build.get("prompt_input_ids")
                if not isinstance(raw_prompt_ids, list) or not raw_prompt_ids:
                    raise CapacityAnalysisError(
                        f"Source build has invalid prompt IDs in {cell_id}, event={event_id}"
                    )
                try:
                    prompt_ids = [int(token_id) for token_id in raw_prompt_ids]
                except (TypeError, ValueError) as exc:
                    raise CapacityAnalysisError(
                        f"Source build has non-integer prompt IDs in {cell_id}, "
                        f"event={event_id}"
                    ) from exc
                source_identity = {
                    "event_id": str(source_build.get("event_id") or ""),
                    "gold_id": int(source_build.get("gold_id", -1)),
                    "prompt_cache_key": _prompt_cache_key(
                        prompt_ids, str(input_manifest.get("label_prefix") or "")
                    ),
                    "prompt_input_ids_sha256": _sha256_json(prompt_ids),
                    "evidence_count": int(source_build.get("evidence_count", -1)),
                    "prompt_token_count": int(
                        source_build.get("prompt_token_count", -1)
                    ),
                }
                if source_identity != mapping_identity:
                    raise CapacityAnalysisError(
                        f"Source-build/mapping identity mismatch in {cell_id}, "
                        f"event={event_id}, unique_idx={unique_idx}"
                    )
            realized_k = int(prediction.get("evidence_count", -1))
            prompt_tokens = int(prediction.get("prompt_token_count", -1))
            prompt_hash = str(prediction.get("prompt_input_ids_sha256") or "")
            if realized_k < 0 or realized_k > requested_k or prompt_tokens <= 0 or not prompt_hash:
                raise CapacityAnalysisError(
                    f"Invalid realized prefix in {cell_id}, event={event_id}: "
                    f"requested={requested_k}, realized={realized_k}, tokens={prompt_tokens}"
                )
            if (
                realized_k != int(mapping.get("evidence_count", -1))
                or prompt_tokens != int(mapping.get("prompt_token_count", -1))
                or prompt_hash != str(mapping.get("prompt_input_ids_sha256") or "")
            ):
                raise CapacityAnalysisError(
                    f"Prediction/mapping resource mismatch in {cell_id}, event={event_id}"
                )
            cell_events.append(event_id)
            cell_gold.append(gold_id)
            cell_observations.append(
                PrefixObservation(
                    selector_level=selector,
                    event_id=event_id,
                    sample_idx=sample_idx,
                    requested_k=requested_k,
                    realized_k=realized_k,
                    prompt_token_count=prompt_tokens,
                    prompt_hash=prompt_hash,
                    unique_idx=unique_idx,
                    gold_id=gold_id,
                    logits=logits[unique_idx],
                )
            )
        if expected_event_ids is None:
            expected_event_ids = cell_events
            expected_gold_ids = cell_gold
        elif cell_events != expected_event_ids or cell_gold != expected_gold_ids:
            raise CapacityAnalysisError(f"Event/gold sequence mismatch for {cell_id}")
        observations.setdefault(selector, {})[requested_k] = cell_observations

    assert expected_event_ids is not None and expected_gold_ids is not None
    expected_grid: tuple[int, ...] | None = None
    for selector, by_k in observations.items():
        grid = tuple(sorted(by_k))
        if expected_grid is None:
            expected_grid = grid
        elif grid != expected_grid:
            raise CapacityAnalysisError(
                f"Prefix K grid mismatch for {selector}: expected={expected_grid}, actual={grid}"
            )
        for sample_idx, event_id in enumerate(expected_event_ids):
            event_rows = [by_k[k][sample_idx] for k in grid]
            realized = [row.realized_k for row in event_rows]
            tokens = [row.prompt_token_count for row in event_rows]
            if realized != sorted(realized) or tokens != sorted(tokens):
                raise CapacityAnalysisError(
                    f"Prefix resources are not monotone for selector={selector}, event={event_id}: "
                    f"realized={realized}, tokens={tokens}"
                )
            exact_flags = [
                row.realized_k == row.requested_k for row in event_rows
            ]
            if exact_flags != sorted(exact_flags, reverse=True):
                raise CapacityAnalysisError(
                    f"Exact-K support is not an initial prefix for selector={selector}, "
                    f"event={event_id}: exact={exact_flags}"
                )
            identity_by_realized: dict[int, tuple[int, str]] = {}
            resources_by_identity: dict[tuple[int, str], tuple[int, int]] = {}
            for row in event_rows:
                previous = identity_by_realized.setdefault(row.realized_k, row.prompt_identity)
                if previous != row.prompt_identity:
                    raise CapacityAnalysisError(
                        f"Equal realized K maps to different prompts for selector={selector}, "
                        f"event={event_id}, realized_k={row.realized_k}"
                    )
                resources = (row.realized_k, row.prompt_token_count)
                previous_resources = resources_by_identity.setdefault(
                    row.prompt_identity, resources
                )
                if previous_resources != resources:
                    raise CapacityAnalysisError(
                        f"Equal prompt identity maps to different resources for "
                        f"selector={selector}, event={event_id}: "
                        f"{previous_resources} != {resources}"
                    )

    source = {
        "matrix_manifest": str(matrix_manifest_path),
        "matrix_manifest_sha256": _sha256_file(matrix_manifest_path),
        "matrix_schema_version": matrix.get("schema_version"),
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": expected_input_sha,
        "raw_logits_manifest": str(raw_manifest_path),
        "raw_logits_manifest_sha256": expected_raw_manifest_sha,
        "raw_logits_file": str(logits_path),
        "raw_logits_sha256": str(raw_manifest.get("raw_logits_sha256") or ""),
        "raw_logits_index": str(index_path),
        "raw_logits_index_sha256": str(raw_manifest.get("index_sha256") or ""),
        "prepared_unique_rows": str(unique_rows_path),
        "prepared_unique_rows_sha256": str(
            input_manifest.get("unique_rows_sha256") or ""
        ),
        "source_capacity_matrix_manifest": (
            str(source_matrix_path) if source_matrix_path is not None else None
        ),
        "source_capacity_matrix_manifest_sha256": source_matrix_sha256 or None,
        "prefix_integrity_gate": (
            str(prefix_gate_path) if prefix_gate_path is not None else None
        ),
        "prefix_integrity_gate_sha256": prefix_gate_sha256 or None,
        "scoring_fingerprint": str(raw_manifest.get("scoring_fingerprint") or ""),
        "execution_fingerprint": str(raw_manifest.get("execution_fingerprint") or ""),
    }
    return (
        matrix,
        labels,
        expected_event_ids,
        np.asarray(expected_gold_ids, dtype=np.int64),
        observations,
        source,
    )


def balanced_class_weights(gold_ids: np.ndarray, *, n_labels: int) -> np.ndarray:
    """Return inverse-frequency weights normalized to mean one over events."""
    gold = np.asarray(gold_ids, dtype=np.int64)
    if gold.ndim != 1 or len(gold) == 0 or np.any(gold < 0) or np.any(gold >= n_labels):
        raise CapacityAnalysisError("gold_ids must be a non-empty 1-D array in label range")
    counts = np.bincount(gold, minlength=n_labels)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise CapacityAnalysisError(
            f"Class-balanced NLL requires every declared class; missing label IDs={missing}"
        )
    return len(gold) / (float(n_labels) * counts.astype(np.float64))


def _observed_class_weights(
    gold_ids: np.ndarray, *, n_labels: int
) -> tuple[np.ndarray, int]:
    """Balance over classes represented on a possibly restricted support."""
    gold = np.asarray(gold_ids, dtype=np.int64)
    counts = np.bincount(gold, minlength=n_labels)
    present = counts > 0
    present_count = int(np.sum(present))
    if present_count == 0:
        raise CapacityAnalysisError("Restricted support contains no events")
    weights = np.zeros(n_labels, dtype=np.float64)
    weights[present] = len(gold) / (
        float(present_count) * counts[present].astype(np.float64)
    )
    return weights, present_count


def _raw_nll(logits: np.ndarray, gold_id: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    maximum = float(np.max(values))
    log_normalizer = maximum + math.log(float(np.exp(values - maximum).sum()))
    return log_normalizer - float(values[gold_id])


def _event_actions(
    rows: Sequence[PrefixObservation],
    *,
    class_weight: float,
    token_penalty_per_1k: float,
) -> tuple[list[PrefixAction], dict[int, PrefixAction]]:
    grouped: dict[tuple[int, str], list[PrefixObservation]] = {}
    for row in sorted(rows, key=lambda item: item.requested_k):
        grouped.setdefault(row.prompt_identity, []).append(row)
    actions: list[PrefixAction] = []
    by_requested_k: dict[int, PrefixAction] = {}
    for duplicates in grouped.values():
        canonical = min(
            duplicates,
            key=lambda item: (item.prompt_token_count, item.realized_k, item.requested_k),
        )
        aliases = tuple(sorted(item.requested_k for item in duplicates))
        nll = _raw_nll(canonical.logits, canonical.gold_id)
        balanced = float(class_weight) * nll
        token_penalty = float(token_penalty_per_1k) * canonical.prompt_token_count / 1000.0
        action = PrefixAction(
            observation=canonical,
            requested_k_aliases=aliases,
            raw_nll=nll,
            balanced_nll=balanced,
            token_penalty=token_penalty,
            objective=balanced + token_penalty,
        )
        actions.append(action)
        for requested_k in aliases:
            by_requested_k[requested_k] = action
    actions.sort(key=lambda item: item.observation.requested_k)
    return actions, by_requested_k


def _select_oracle(actions: Sequence[PrefixAction], *, tie_atol: float) -> PrefixAction:
    minimum = min(action.objective for action in actions)
    tied = [action for action in actions if action.objective <= minimum + tie_atol]
    return min(
        tied,
        key=lambda action: (
            action.observation.prompt_token_count,
            action.observation.realized_k,
            action.observation.requested_k,
        ),
    )


def _load_policies(
    policy_paths: Mapping[str, Path],
    *,
    event_ids: Sequence[str],
    selectors: Sequence[str],
    evaluation_split: str,
) -> tuple[dict[str, dict[tuple[str, str], int]], list[dict[str, Any]]]:
    known_events = set(event_ids)
    known_selectors = set(selectors)
    policies: dict[str, dict[tuple[str, str], int]] = {}
    sources: list[dict[str, Any]] = []
    for policy_id, path_value in sorted(policy_paths.items()):
        path = Path(path_value).resolve()
        if not policy_id:
            raise CapacityAnalysisError("policy_id must not be empty")
        assignments: dict[tuple[str, str], int] = {}
        covered_selectors: set[str] = set()
        for line_number, row in _iter_jsonl(path):
            event_id = str(row.get("event_id") or "")
            selector = str(row.get("selector_level") or "")
            try:
                selected_k = int(row["selected_k"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CapacityAnalysisError(
                    f"Invalid selected_k in {path}:{line_number}"
                ) from exc
            if event_id not in known_events or selector not in known_selectors or selected_k <= 0:
                raise CapacityAnalysisError(
                    f"Invalid policy assignment in {path}:{line_number}: "
                    f"event={event_id!r}, selector={selector!r}, K={selected_k}"
                )
            if row.get("policy_id") not in (None, "", policy_id):
                raise CapacityAnalysisError(f"policy_id mismatch in {path}:{line_number}")
            key = (selector, event_id)
            if key in assignments:
                raise CapacityAnalysisError(f"Duplicate policy assignment in {path}: {key}")
            assignments[key] = selected_k
            covered_selectors.add(selector)
        if not assignments:
            raise CapacityAnalysisError(f"Policy file is empty: {path}")
        for selector in covered_selectors:
            missing = [
                event_id
                for event_id in event_ids
                if (selector, event_id) not in assignments
            ]
            if missing:
                raise CapacityAnalysisError(
                    f"Policy {policy_id} is incomplete for selector={selector}; "
                    f"missing={missing[:5]}"
                )
        policies[policy_id] = assignments
        source = {
            "policy_id": policy_id,
            "path": str(path),
            "sha256": _sha256_file(path),
            "assignment_count": len(assignments),
            "selectors": sorted(covered_selectors),
            "verification_status": "unverified_external_policy",
            "uses_gold": None,
            "uses_verifier_logits": None,
            "fit_split": None,
            "decision_source_split": None,
            "cross_fit_status": "unknown",
            "deployable_without_gold": False,
            "deployable_ex_ante": False,
        }
        sidecar_path = path.with_name(f"{path.stem}.summary.json")
        if sidecar_path.is_file():
            source.update(
                _verified_trace_policy_source(
                    policy_id=policy_id,
                    policy_path=path,
                    sidecar_path=sidecar_path,
                    assignment_count=len(assignments),
                    covered_selectors=covered_selectors,
                    event_ids=event_ids,
                    evaluation_split=evaluation_split,
                )
            )
        sources.append(source)
    return policies, sources


def _verified_trace_policy_source(
    *,
    policy_id: str,
    policy_path: Path,
    sidecar_path: Path,
    assignment_count: int,
    covered_selectors: set[str],
    event_ids: Sequence[str],
    evaluation_split: str,
) -> dict[str, Any]:
    summary = _read_json(sidecar_path)
    if summary.get("schema_version") != "capacity_policy_from_traces_v0_1":
        raise CapacityAnalysisError(
            f"Unsupported policy sidecar schema: {sidecar_path}"
        )
    declared_policy_path = _resolve_declared_path(
        str(summary.get("output_policy") or ""), anchor=sidecar_path.parent
    )
    if (
        str(summary.get("policy_id") or "") != policy_id
        or declared_policy_path != policy_path
        or str(summary.get("output_policy_sha256") or "")
        != _sha256_file(policy_path)
        or int(summary.get("assignment_count", -1)) != assignment_count
        or int(summary.get("event_count", -1)) != len(event_ids)
        or int(summary.get("selector_count", -1)) != len(covered_selectors)
        or str(summary.get("event_id_sequence_sha256") or "")
        != _event_sequence_sha256(event_ids)
    ):
        raise CapacityAnalysisError(
            f"Policy sidecar does not bind the supplied assignments: {sidecar_path}"
        )
    raw_sources = summary.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(covered_selectors):
        raise CapacityAnalysisError(f"Invalid policy source traces: {sidecar_path}")
    source_selectors: set[str] = set()
    verified_sources: list[dict[str, Any]] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise CapacityAnalysisError(f"Invalid policy source entry: {sidecar_path}")
        selector = str(raw_source.get("selector_level") or "")
        trace_path = _resolve_declared_path(
            str(raw_source.get("path") or ""), anchor=sidecar_path.parent
        )
        if (
            selector not in covered_selectors
            or selector in source_selectors
            or not trace_path.is_file()
            or _sha256_file(trace_path) != str(raw_source.get("sha256") or "")
            or int(raw_source.get("event_count", -1)) != len(event_ids)
            or int(raw_source.get("assignment_count", -1)) != len(event_ids)
        ):
            raise CapacityAnalysisError(
                f"Policy source trace provenance mismatch: {sidecar_path}"
            )
        source_selectors.add(selector)
        verified_sources.append(
            {
                "selector_level": selector,
                "path": str(trace_path),
                "sha256": str(raw_source.get("sha256") or ""),
                "event_count": len(event_ids),
            }
        )
    if source_selectors != covered_selectors:
        raise CapacityAnalysisError(
            f"Policy source selectors disagree with assignments: {sidecar_path}"
        )
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CapacityAnalysisError(f"Policy sidecar has no provenance contract: {sidecar_path}")
    uses_gold = provenance.get("uses_gold")
    uses_logits = provenance.get("uses_verifier_logits")
    deployable = provenance.get("deployable_without_gold")
    deployable_ex_ante = provenance.get("deployable_ex_ante", False)
    if (
        (uses_gold is not None and not isinstance(uses_gold, bool))
        or (uses_logits is not None and not isinstance(uses_logits, bool))
        or not isinstance(deployable, bool)
        or not isinstance(deployable_ex_ante, bool)
    ):
        raise CapacityAnalysisError(
            f"Policy provenance booleans/nulls are invalid: {sidecar_path}"
        )
    decision_split = provenance.get("decision_source_split")
    if decision_split is not None and str(decision_split) != str(
        summary.get("split") or ""
    ):
        raise CapacityAnalysisError(
            f"Policy decision split disagrees with sidecar split: {sidecar_path}"
        )
    declared_status = str(provenance.get("verification_status") or "")
    if not declared_status:
        raise CapacityAnalysisError(
            f"Policy provenance has no verification_status: {sidecar_path}"
        )
    factorial_contract = summary.get("factorial_manifest")
    verified_factorial_contract: dict[str, Any] | None = None
    if factorial_contract is not None:
        if not isinstance(factorial_contract, Mapping):
            raise CapacityAnalysisError(
                f"Policy factorial_manifest contract is invalid: {sidecar_path}"
            )
        factorial_path = _resolve_declared_path(
            str(factorial_contract.get("path") or ""), anchor=sidecar_path.parent
        )
        generator = factorial_contract.get("generator_implementation")
        if not isinstance(generator, Mapping):
            raise CapacityAnalysisError(
                f"Policy factorial generator contract is invalid: {sidecar_path}"
            )
        generator_path = _resolve_declared_path(
            str(generator.get("path") or ""), anchor=sidecar_path.parent
        )
        factorial_cells = factorial_contract.get("cells")
        if not isinstance(factorial_cells, list) or len(factorial_cells) != len(
            verified_sources
        ):
            raise CapacityAnalysisError(
                f"Policy factorial cell contract is invalid: {sidecar_path}"
            )
        factorial_cells_by_selector = {
            str(cell.get("selector_level") or ""): cell
            for cell in factorial_cells
            if isinstance(cell, Mapping)
        }
        source_by_selector = {
            str(source["selector_level"]): source for source in verified_sources
        }
        if set(factorial_cells_by_selector) != set(source_by_selector):
            raise CapacityAnalysisError(
                f"Policy factorial/source selectors disagree: {sidecar_path}"
            )
        for selector, cell in factorial_cells_by_selector.items():
            source = source_by_selector[selector]
            if (
                Path(str(cell.get("trace_path") or "")).resolve()
                != Path(str(source["path"])).resolve()
                or str(cell.get("trace_sha256") or "") != str(source["sha256"])
                or int(cell.get("policy_event_count", -1)) != len(event_ids)
            ):
                raise CapacityAnalysisError(
                    f"Policy factorial/source trace binding mismatch: {sidecar_path}"
                )
        if (
            not factorial_path.is_file()
            or _sha256_file(factorial_path)
            != str(factorial_contract.get("sha256") or "")
            or str(factorial_contract.get("controller_level") or "")
            != str(provenance.get("expected_controller") or "")
            or str(factorial_contract.get("split") or "")
            != str(summary.get("split") or "")
            or not generator_path.is_file()
            or _sha256_file(generator_path) != str(generator.get("sha256") or "")
        ):
            raise CapacityAnalysisError(
                f"Policy factorial provenance drift: {sidecar_path}"
            )
        verified_factorial_contract = {
            "path": str(factorial_path),
            "sha256": str(factorial_contract.get("sha256") or ""),
            "controller_level": factorial_contract.get("controller_level"),
            "controller_contract": factorial_contract.get("controller_contract"),
            "generator_implementation": {
                "path": str(generator_path),
                "sha256": str(generator.get("sha256") or ""),
            },
        }
    verified_controller = declared_status == (
        "verified_known_structure_only_factorial_controller"
    )
    if verified_controller and (
        verified_factorial_contract is None
        or uses_gold is not False
        or uses_logits is not False
        or not deployable_ex_ante
    ):
        raise CapacityAnalysisError(
            f"Verified controller lacks a closed provenance contract: {sidecar_path}"
        )
    verification_status = (
        "verified_trace_policy_sidecar"
        if verified_controller
        else "trace_policy_sidecar_integrity_verified_provenance_unknown"
    )
    return {
        "verification_status": verification_status,
        "declared_verification_status": declared_status,
        "sidecar_path": str(sidecar_path.resolve()),
        "sidecar_sha256": _sha256_file(sidecar_path),
        "policy_family": provenance.get("policy_family"),
        "uses_gold": uses_gold,
        "uses_verifier_logits": uses_logits,
        "fit_split": provenance.get("fit_split"),
        "decision_source_split": decision_split,
        "evaluation_split": evaluation_split,
        "cross_fit_status": provenance.get("cross_fit_status"),
        "deployable_without_gold": bool(deployable and uses_gold is False),
        "deployable_ex_ante": bool(
            deployable_ex_ante
            and uses_gold is False
            and uses_logits is False
            and verified_controller
        ),
        "expected_controller": provenance.get("expected_controller"),
        "trace_order_field": provenance.get("trace_order_field"),
        "factorial_manifest": verified_factorial_contract,
        "source_traces": sorted(
            verified_sources, key=lambda row: str(row["selector_level"])
        ),
    }


def _generated_fixed_policies(
    *,
    observations: Mapping[str, Mapping[int, Sequence[PrefixObservation]]],
    event_ids: Sequence[str],
) -> tuple[dict[str, dict[tuple[str, str], int]], list[dict[str, Any]]]:
    selectors = sorted(observations)
    grid = sorted(observations[selectors[0]])
    policies: dict[str, dict[tuple[str, str], int]] = {}
    sources: list[dict[str, Any]] = []
    for requested_k in grid:
        policy_id = f"fixed_k{requested_k:02d}"
        assignments = {
            (selector, event_id): requested_k
            for selector in selectors
            for event_id in event_ids
        }
        policies[policy_id] = assignments
        sources.append(
            {
                "policy_id": policy_id,
                "kind": "generated_fixed_capacity",
                "selected_k": requested_k,
                "assignment_count": len(assignments),
                "selectors": selectors,
                "verification_status": "verified_generated_fixed_policy",
                "uses_gold": False,
                "uses_verifier_logits": False,
                "fit_split": None,
                "decision_source_split": None,
                "cross_fit_status": "not_applicable_no_fitting",
                "deployable_without_gold": True,
                "deployable_ex_ante": True,
            }
        )
    return policies, sources


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {
        str(value): int(sum(item == value for item in values))
        for value in sorted(set(values))
    }


def _classification_metrics(
    gold_ids: np.ndarray,
    pred_ids: np.ndarray,
    *,
    n_labels: int,
) -> tuple[float, float]:
    gold = np.asarray(gold_ids, dtype=np.int64)
    pred = np.asarray(pred_ids, dtype=np.int64)
    if gold.shape != pred.shape or gold.ndim != 1 or len(gold) == 0:
        raise CapacityAnalysisError(
            f"Expected matching non-empty 1-D gold/pred arrays, got {gold.shape}/{pred.shape}"
        )
    matrix = np.bincount(
        gold * n_labels + pred, minlength=n_labels * n_labels
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
    return float(np.mean(gold == pred)), float(np.mean(class_f1))


def _capacity_curve_rows(
    *,
    observations: Mapping[str, Mapping[int, Sequence[PrefixObservation]]],
    labels: Sequence[str],
    event_ids: Sequence[str],
    token_penalty_per_1k: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_labels = len(labels)
    selectors = sorted(observations)
    common_grid = sorted(observations[selectors[0]])
    strict_common_indices = np.asarray(
        [
            sample_idx
            for sample_idx in range(len(event_ids))
            if all(
                observations[selector][requested_k][sample_idx].realized_k
                == requested_k
                for selector in selectors
                for requested_k in common_grid
            )
        ],
        dtype=np.int64,
    )
    for selector in selectors:
        by_k = observations[selector]
        grid = sorted(by_k)
        full_indices = np.arange(len(event_ids), dtype=np.int64)
        support_specs = (
            ("full_n_deployable", full_indices),
            ("strict_full_grid_common_support", strict_common_indices),
        )
        for support, indices in support_specs:
            support_event_ids = [event_ids[int(index)] for index in indices]
            if len(indices) == 0:
                for requested_k in grid:
                    rows.append(
                        {
                            "selector_level": selector,
                            "support": support,
                            "requested_k": requested_k,
                            "sample_count": 0,
                            "support_event_id_sequence_sha256": _event_sequence_sha256([]),
                            "accuracy": None,
                            "macro_f1": None,
                            "raw_nll_mean": None,
                            "class_balanced_nll_mean": None,
                            "token_penalty_mean": None,
                            "objective_mean": None,
                            "class_balanced_label_count": 0,
                            "mean_realized_k": None,
                            "mean_prompt_token_count": None,
                            "exact_support_count": 0,
                            "exact_support_rate": None,
                            "unique_prompt_count": 0,
                        }
                    )
                continue
            support_gold = np.asarray(
                [by_k[grid[0]][int(index)].gold_id for index in indices],
                dtype=np.int64,
            )
            support_weights, balanced_label_count = _observed_class_weights(
                support_gold, n_labels=n_labels
            )
            for requested_k in grid:
                observations_k = [by_k[requested_k][int(index)] for index in indices]
                logits = np.stack([observation.logits for observation in observations_k])
                pred_ids = np.argmax(logits, axis=1).astype(np.int64)
                accuracy, macro_f1 = _classification_metrics(
                    support_gold, pred_ids, n_labels=n_labels
                )
                raw_nlls = np.asarray(
                    [
                        _raw_nll(observation.logits, observation.gold_id)
                        for observation in observations_k
                    ],
                    dtype=np.float64,
                )
                event_weights = support_weights[support_gold]
                realized = np.asarray(
                    [observation.realized_k for observation in observations_k],
                    dtype=np.int64,
                )
                prompt_tokens = np.asarray(
                    [
                        observation.prompt_token_count
                        for observation in observations_k
                    ],
                    dtype=np.int64,
                )
                balanced_nll_mean = float(np.mean(raw_nlls * event_weights))
                token_penalty_mean = (
                    float(token_penalty_per_1k)
                    * float(np.mean(prompt_tokens))
                    / 1000.0
                )
                rows.append(
                    {
                        "selector_level": selector,
                        "support": support,
                        "requested_k": requested_k,
                        "sample_count": len(indices),
                        "support_event_id_sequence_sha256": _event_sequence_sha256(
                            support_event_ids
                        ),
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "raw_nll_mean": float(np.mean(raw_nlls)),
                        "class_balanced_nll_mean": balanced_nll_mean,
                        "token_penalty_mean": token_penalty_mean,
                        "objective_mean": balanced_nll_mean + token_penalty_mean,
                        "class_balanced_label_count": balanced_label_count,
                        "mean_realized_k": float(np.mean(realized)),
                        "mean_prompt_token_count": float(np.mean(prompt_tokens)),
                        "exact_support_count": int(np.sum(realized == requested_k)),
                        "exact_support_rate": float(
                            np.mean(realized == requested_k)
                        ),
                        "unique_prompt_count": len(
                            {observation.prompt_identity for observation in observations_k}
                        ),
                    }
                )
    return rows


def _summarize(
    oracle_rows: Sequence[Mapping[str, Any]],
    regret_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    policy_sources: Sequence[Mapping[str, Any]],
    tie_atol: float,
) -> dict[str, Any]:
    selector_summaries: list[dict[str, Any]] = []
    for selector in sorted({str(row["selector_level"]) for row in oracle_rows}):
        rows = [row for row in oracle_rows if row["selector_level"] == selector]
        requested = [int(row["oracle_requested_k"]) for row in rows]
        realized = [int(row["oracle_realized_k"]) for row in rows]
        feasible = [int(row["capacity_feasible_max_k"]) for row in rows]
        max_grid_k = max(int(row["grid_max_requested_k"]) for row in rows)
        selector_summaries.append(
            {
                "selector_level": selector,
                "event_count": len(rows),
                "oracle_requested_k_mean": float(np.mean(requested)),
                "oracle_requested_k_median": float(np.median(requested)),
                "oracle_requested_k_distribution": _distribution(requested),
                "oracle_realized_k_mean": float(np.mean(realized)),
                "oracle_realized_k_distribution": _distribution(realized),
                "oracle_balanced_nll_mean": float(
                    np.mean([float(row["oracle_balanced_nll"]) for row in rows])
                ),
                "oracle_objective_mean": float(
                    np.mean([float(row["oracle_objective"]) for row in rows])
                ),
                "capacity_feasible_max_k_mean": float(np.mean(feasible)),
                "full_grid_exact_support_count": int(
                    sum(value >= max_grid_k for value in feasible)
                ),
                "full_grid_exact_support_rate": float(
                    np.mean([value >= max_grid_k for value in feasible])
                ),
                "oracle_at_min_action_rate": float(
                    np.mean([bool(row["oracle_at_min_action"]) for row in rows])
                ),
                "oracle_at_max_action_rate": float(
                    np.mean([bool(row["oracle_at_max_action"]) for row in rows])
                ),
            }
        )

    policy_source_by_id = {
        str(source.get("policy_id") or ""): source for source in policy_sources
    }
    if "" in policy_source_by_id or len(policy_source_by_id) != len(policy_sources):
        raise CapacityAnalysisError("Policy source metadata has duplicate or empty IDs")
    policy_summaries: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["policy_id"]), str(row["selector_level"])) for row in regret_rows}
    )
    for policy_id, selector in keys:
        rows = [
            row
            for row in regret_rows
            if row["policy_id"] == policy_id and row["selector_level"] == selector
        ]
        relations = [str(row["capacity_relation"]) for row in rows]
        regrets = np.asarray([float(row["objective_regret"]) for row in rows])
        regret_by_relation = {}
        source = policy_source_by_id.get(policy_id)
        if source is None:
            raise CapacityAnalysisError(f"Missing policy source metadata: {policy_id}")
        for relation in ("underfill", "capacity_correct", "overfill"):
            relation_regrets = [
                float(row["objective_regret"])
                for row in rows
                if row["capacity_relation"] == relation
            ]
            regret_by_relation[relation] = {
                "count": len(relation_regrets),
                "mean": float(np.mean(relation_regrets)) if relation_regrets else None,
                "sum": float(np.sum(relation_regrets)) if relation_regrets else 0.0,
            }
        policy_summaries.append(
            {
                "policy_id": policy_id,
                "selector_level": selector,
                "event_count": len(rows),
                "policy_canonical_k_mean": float(
                    np.mean([int(row["policy_canonical_requested_k"]) for row in rows])
                ),
                "oracle_requested_k_mean": float(
                    np.mean([int(row["oracle_requested_k"]) for row in rows])
                ),
                "underfill_count": relations.count("underfill"),
                "underfill_rate": float(np.mean([value == "underfill" for value in relations])),
                "capacity_correct_count": relations.count("capacity_correct"),
                "capacity_correct_rate": float(
                    np.mean([value == "capacity_correct" for value in relations])
                ),
                "overfill_count": relations.count("overfill"),
                "overfill_rate": float(np.mean([value == "overfill" for value in relations])),
                "objective_regret_mean": float(np.mean(regrets)),
                "objective_regret_median": float(np.median(regrets)),
                "objective_regret_p90": float(np.quantile(regrets, 0.9)),
                "objective_regret_max": float(np.max(regrets)),
                "objective_regret_by_relation": regret_by_relation,
                "balanced_nll_delta_mean": float(
                    np.mean([float(row["balanced_nll_delta"]) for row in rows])
                ),
                "prompt_token_delta_mean": float(
                    np.mean([int(row["prompt_token_delta"]) for row in rows])
                ),
                "verification_status": source.get("verification_status"),
                "uses_gold": source.get("uses_gold"),
                "uses_verifier_logits": source.get("uses_verifier_logits"),
                "fit_split": source.get("fit_split"),
                "decision_source_split": source.get("decision_source_split"),
                "cross_fit_status": source.get("cross_fit_status"),
                "deployable_without_gold": bool(
                    source.get("deployable_without_gold")
                ),
                "deployable_ex_ante": bool(source.get("deployable_ex_ante")),
            }
        )

    capacity_optima: list[dict[str, Any]] = []
    curve_keys = sorted(
        {
            (str(row["selector_level"]), str(row["support"]))
            for row in curve_rows
        }
    )
    for selector, support in curve_keys:
        rows = [
            row
            for row in curve_rows
            if row["selector_level"] == selector
            and row["support"] == support
            and int(row["sample_count"]) > 0
        ]
        if not rows:
            continue
        nll_minimum = min(float(row["class_balanced_nll_mean"]) for row in rows)
        nll_best = min(
            (
                row
                for row in rows
                if float(row["class_balanced_nll_mean"])
                <= nll_minimum + tie_atol
            ),
            key=lambda row: (
                float(row["mean_prompt_token_count"]),
                int(row["requested_k"]),
            ),
        )
        objective_minimum = min(float(row["objective_mean"]) for row in rows)
        objective_best = min(
            (
                row
                for row in rows
                if float(row["objective_mean"]) <= objective_minimum + tie_atol
            ),
            key=lambda row: (
                float(row["mean_prompt_token_count"]),
                int(row["requested_k"]),
            ),
        )
        macro_maximum = max(float(row["macro_f1"]) for row in rows)
        macro_best = min(
            (
                row
                for row in rows
                if float(row["macro_f1"]) >= macro_maximum - tie_atol
            ),
            key=lambda row: (
                float(row["mean_prompt_token_count"]),
                int(row["requested_k"]),
            ),
        )
        capacity_optima.append(
            {
                "selector_level": selector,
                "support": support,
                "sample_count": int(nll_best["sample_count"]),
                "support_event_id_sequence_sha256": nll_best[
                    "support_event_id_sequence_sha256"
                ],
                "balanced_nll_optimal_k": int(nll_best["requested_k"]),
                "balanced_nll_optimal_value": float(
                    nll_best["class_balanced_nll_mean"]
                ),
                "balanced_nll_optimal_macro_f1": float(nll_best["macro_f1"]),
                "objective_optimal_k": int(objective_best["requested_k"]),
                "objective_optimal_value": float(objective_best["objective_mean"]),
                "objective_optimal_balanced_nll": float(
                    objective_best["class_balanced_nll_mean"]
                ),
                "objective_optimal_token_penalty": float(
                    objective_best["token_penalty_mean"]
                ),
                "objective_optimal_macro_f1": float(objective_best["macro_f1"]),
                "macro_f1_optimal_k": int(macro_best["requested_k"]),
                "macro_f1_optimal_value": float(macro_best["macro_f1"]),
                "macro_f1_optimal_balanced_nll": float(
                    macro_best["class_balanced_nll_mean"]
                ),
                "tie_atol": float(tie_atol),
                "tie_break": "fewer_mean_prompt_tokens_then_lower_requested_k",
            }
        )
    return {
        "selectors": selector_summaries,
        "policies": policy_summaries,
        "capacity_curves": list(curve_rows),
        "capacity_optima": capacity_optima,
    }


def _render_report(
    *,
    aggregate: Mapping[str, Any],
    class_balance: Mapping[str, Any],
    token_penalty_per_1k: float,
) -> str:
    lines = [
        "# Capacity prefix oracle and policy regret",
        "",
        "> Post-hoc frozen-verifier diagnostic. Gold labels define the oracle; this is not a "
        "deployable policy and not a causal estimate of evidence capacity.",
        "",
        "## Objective",
        "",
        "Per-event oracle minimizes inverse-frequency class-balanced NLL plus "
        f"`{token_penalty_per_1k:g} * prompt_tokens / 1000`. Identical prompts are "
        "deduplicated; ties prefer fewer prompt tokens, lower realized K, then lower requested K.",
        "Capacity-curve balanced NLL recomputes inverse-frequency weights within each fixed "
        "support; all K on the strict curve use the same full-grid-exact event subset.",
        "On a restricted support, balanced NLL balances only labels represented there, while "
        "Macro-F1 still averages all declared labels. Compare K values within one support; do "
        "not read absolute full-N versus strict-support metric differences as capacity effects.",
        "",
        "## Class balance",
        "",
        "| label | count | weight |",
        "|---|---:|---:|",
    ]
    for row in class_balance["labels"]:
        lines.append(f"| {row['label']} | {row['count']} | {row['weight']:.6f} |")
    lines.extend(
        [
            "",
            "## Selector oracle summary",
            "",
            "| selector | events | oracle K mean | realized K mean | full-grid exact support | "
            "oracle NLL | min/max action rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["selectors"]:
        lines.append(
            f"| {row['selector_level']} | {row['event_count']} | "
            f"{row['oracle_requested_k_mean']:.3f} | {row['oracle_realized_k_mean']:.3f} | "
            f"{row['full_grid_exact_support_rate']:.3%} | "
            f"{row['oracle_balanced_nll_mean']:.6f} | "
            f"{row['oracle_at_min_action_rate']:.3%}/{row['oracle_at_max_action_rate']:.3%} |"
        )
    for support, title in (
        ("full_n_deployable", "Full-N deployable capacity curve"),
        (
            "strict_full_grid_common_support",
            "Strict full-grid common-support capacity curve",
        ),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| selector | requested K | N | accuracy | Macro-F1 | raw NLL | "
                "balanced NLL | token penalty | objective | realized K | tokens | "
                "exact support |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate["capacity_curves"]:
            if row["support"] != support:
                continue
            if int(row["sample_count"]) == 0:
                lines.append(
                    f"| {row['selector_level']} | {row['requested_k']} | 0 | NA | NA | "
                    "NA | NA | NA | NA | NA | NA | NA |"
                )
                continue
            lines.append(
                f"| {row['selector_level']} | {row['requested_k']} | "
                f"{row['sample_count']} | {row['accuracy']:.6f} | "
                f"{row['macro_f1']:.6f} | {row['raw_nll_mean']:.6f} | "
                f"{row['class_balanced_nll_mean']:.6f} | "
                f"{row['token_penalty_mean']:.6f} | {row['objective_mean']:.6f} | "
                f"{row['mean_realized_k']:.3f} | "
                f"{row['mean_prompt_token_count']:.1f} | "
                f"{row['exact_support_rate']:.3%} |"
            )
    lines.extend(
        [
            "",
            "## Capacity optima on the frozen validation surface",
            "",
            "| selector | support | N | objective optimum K | objective | "
            "balanced-NLL optimum K | balanced NLL | Macro-F1 optimum K | Macro-F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["capacity_optima"]:
        lines.append(
            f"| {row['selector_level']} | {row['support']} | {row['sample_count']} | "
            f"{row['objective_optimal_k']} | {row['objective_optimal_value']:.6f} | "
            f"{row['balanced_nll_optimal_k']} | "
            f"{row['balanced_nll_optimal_value']:.6f} | "
            f"{row['macro_f1_optimal_k']} | {row['macro_f1_optimal_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Policy regret summary",
            "",
            "| policy | selector | provenance | ex-ante deployable | events | under/correct/over | "
            "mean regret | p90 regret | balanced NLL delta | token delta |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if not aggregate["policies"]:
        lines.append("| _no policy assignments supplied_ | | | | | | | | | |")
    for row in aggregate["policies"]:
        lines.append(
            f"| {row['policy_id']} | {row['selector_level']} | "
            f"{row['verification_status']} | {str(row['deployable_ex_ante']).lower()} | "
            f"{row['event_count']} | "
            f"{row['underfill_rate']:.1%}/{row['capacity_correct_rate']:.1%}/"
            f"{row['overfill_rate']:.1%} | {row['objective_regret_mean']:.6f} | "
            f"{row['objective_regret_p90']:.6f} | {row['balanced_nll_delta_mean']:.6f} | "
            f"{row['prompt_token_delta_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "`underfill` and `overfill` compare the policy's canonical prompt action with the "
            "oracle action. Requested K values that produce the same prompt are aliases and do "
            "not create artificial overfill regret.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_existing(
    *,
    output_dir: Path,
    source_sha: str,
    settings: Mapping[str, Any],
    policy_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    implementation_sha256 = _sha256_file(Path(__file__).resolve())
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or str((manifest.get("source", {}) or {}).get("matrix_manifest_sha256") or "")
        != source_sha
        or manifest.get("settings") != dict(settings)
        or manifest.get("policy_sources") != list(policy_sources)
        or str((manifest.get("implementation", {}) or {}).get("sha256") or "")
        != implementation_sha256
    ):
        raise CapacityAnalysisError(
            f"Existing capacity artifact is incompatible: {output_dir}; pass --force"
        )
    for artifact in (manifest.get("artifacts", {}) or {}).values():
        path = output_dir / str((artifact or {}).get("path") or "")
        if not path.is_file() or _sha256_file(path) != str((artifact or {}).get("sha256") or ""):
            raise CapacityAnalysisError(f"Existing capacity artifact drifted: {path}")
    return manifest


def materialize_capacity_analysis(
    *,
    matrix_manifest_path: Path,
    output_dir: Path,
    policy_paths: Mapping[str, Path] | None = None,
    include_fixed_policies: bool = False,
    token_penalty_per_1k: float = 0.0,
    tie_atol: float = 1e-12,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize per-event oracle K, prefix scores, and supplied-policy regret."""
    if not math.isfinite(token_penalty_per_1k) or token_penalty_per_1k < 0:
        raise CapacityAnalysisError("token_penalty_per_1k must be finite and non-negative")
    if not math.isfinite(tie_atol) or tie_atol < 0:
        raise CapacityAnalysisError("tie_atol must be finite and non-negative")
    (
        matrix,
        labels,
        event_ids,
        gold_ids,
        observations,
        source,
    ) = _load_prefix_source(Path(matrix_manifest_path))
    weights = balanced_class_weights(gold_ids, n_labels=len(labels))
    counts = np.bincount(gold_ids, minlength=len(labels))
    class_balance = {
        "method": "inverse_frequency_N_over_Cn",
        "normalization": "mean_event_weight_equals_one",
        "labels": [
            {
                "label_id": label_id,
                "label": labels[label_id],
                "count": int(counts[label_id]),
                "weight": float(weights[label_id]),
            }
            for label_id in range(len(labels))
        ],
    }
    policies, policy_sources = _load_policies(
        policy_paths or {},
        event_ids=event_ids,
        selectors=sorted(observations),
        evaluation_split=str(matrix.get("split") or ""),
    )
    if include_fixed_policies:
        fixed_policies, fixed_sources = _generated_fixed_policies(
            observations=observations,
            event_ids=event_ids,
        )
        overlap = sorted(set(policies) & set(fixed_policies))
        if overlap:
            raise CapacityAnalysisError(
                f"Explicit policy IDs conflict with generated fixed policies: {overlap}"
            )
        policies.update(fixed_policies)
        policy_sources.extend(fixed_sources)
    settings = {
        "objective": "inverse_frequency_class_balanced_nll_plus_token_penalty",
        "token_penalty_per_1k": float(token_penalty_per_1k),
        "tie_atol": float(tie_atol),
        "tie_break": ["prompt_token_count", "realized_k", "requested_k"],
        "duplicate_action_key": ["raw_logits_unique_idx", "prompt_input_ids_sha256"],
        "capacity_relation_basis": "canonical_requested_k_after_prompt_deduplication",
        "include_fixed_policies": bool(include_fixed_policies),
        "capacity_curve_supports": [
            "full_n_deployable",
            "strict_full_grid_common_support",
        ],
        "capacity_curve_metrics": [
            "accuracy",
            "macro_f1",
            "raw_nll_mean",
            "class_balanced_nll_mean",
            "token_penalty_mean",
            "objective_mean",
            "mean_realized_k",
            "mean_prompt_token_count",
            "exact_support_rate",
        ],
        "capacity_curve_class_balance": (
            "inverse_frequency_within_each_support_over_represented_labels"
        ),
    }
    output_dir = Path(output_dir).resolve()
    source_sha = str(source["matrix_manifest_sha256"])
    if output_dir.exists() and not force:
        return _validate_existing(
            output_dir=output_dir,
            source_sha=source_sha,
            settings=settings,
            policy_sources=policy_sources,
        )

    oracle_rows: list[dict[str, Any]] = []
    prefix_score_rows: list[dict[str, Any]] = []
    action_lookup: dict[tuple[str, str, int], PrefixAction] = {}
    oracle_lookup: dict[tuple[str, str], PrefixAction] = {}
    selectors = sorted(observations)
    for selector in selectors:
        by_k = observations[selector]
        grid = sorted(by_k)
        for sample_idx, event_id in enumerate(event_ids):
            event_observations = [by_k[k][sample_idx] for k in grid]
            actions, by_requested_k = _event_actions(
                event_observations,
                class_weight=float(weights[int(gold_ids[sample_idx])]),
                token_penalty_per_1k=token_penalty_per_1k,
            )
            oracle = _select_oracle(actions, tie_atol=tie_atol)
            oracle_lookup[(selector, event_id)] = oracle
            for requested_k, action in by_requested_k.items():
                action_lookup[(selector, event_id, requested_k)] = action
            feasible_max = max(
                (
                    row.requested_k
                    for row in event_observations
                    if row.realized_k == row.requested_k
                ),
                default=0,
            )
            oracle_index = actions.index(oracle)
            oracle_row = {
                "sample_idx": sample_idx,
                "event_id": event_id,
                "selector_level": selector,
                "gold_id": int(gold_ids[sample_idx]),
                "gold_label": labels[int(gold_ids[sample_idx])],
                "class_weight": float(weights[int(gold_ids[sample_idx])]),
                "grid_min_requested_k": grid[0],
                "grid_max_requested_k": grid[-1],
                "capacity_feasible_max_k": feasible_max,
                "requested_action_count": len(grid),
                "unique_prompt_action_count": len(actions),
                "oracle_requested_k": oracle.observation.requested_k,
                "oracle_requested_k_aliases": list(oracle.requested_k_aliases),
                "oracle_realized_k": oracle.observation.realized_k,
                "oracle_prompt_token_count": oracle.observation.prompt_token_count,
                "oracle_raw_logits_unique_idx": oracle.observation.unique_idx,
                "oracle_raw_nll": oracle.raw_nll,
                "oracle_balanced_nll": oracle.balanced_nll,
                "oracle_token_penalty": oracle.token_penalty,
                "oracle_objective": oracle.objective,
                "oracle_at_min_action": oracle_index == 0,
                "oracle_at_max_action": oracle_index == len(actions) - 1,
            }
            oracle_rows.append(oracle_row)
            action_indices = {id(action): index for index, action in enumerate(actions)}
            for observation in event_observations:
                action = by_requested_k[observation.requested_k]
                gold_probability = math.exp(-action.raw_nll)
                prefix_score_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "event_id": event_id,
                        "selector_level": selector,
                        "gold_id": observation.gold_id,
                        "gold_label": labels[observation.gold_id],
                        "requested_k": observation.requested_k,
                        "canonical_requested_k": action.observation.requested_k,
                        "requested_k_aliases": list(action.requested_k_aliases),
                        "realized_k": observation.realized_k,
                        "capacity_is_exact_k": observation.realized_k
                        == observation.requested_k,
                        "prompt_token_count": observation.prompt_token_count,
                        "raw_logits_unique_idx": observation.unique_idx,
                        "prompt_input_ids_sha256": observation.prompt_hash,
                        "raw_nll": action.raw_nll,
                        "gold_probability": gold_probability,
                        "class_weight": float(weights[observation.gold_id]),
                        "balanced_nll": action.balanced_nll,
                        "token_penalty": action.token_penalty,
                        "objective": action.objective,
                        "is_duplicate_prompt_alias": observation.requested_k
                        != action.observation.requested_k,
                        "is_oracle_action": action is oracle,
                        "is_oracle_canonical": action is oracle
                        and observation.requested_k
                        == action.observation.requested_k,
                        "action_index": action_indices[id(action)],
                    }
                )

    policy_source_by_id = {
        str(source.get("policy_id") or ""): source for source in policy_sources
    }
    if "" in policy_source_by_id or len(policy_source_by_id) != len(policy_sources):
        raise CapacityAnalysisError("Policy source metadata has duplicate or empty IDs")
    regret_rows: list[dict[str, Any]] = []
    for policy_id, assignments in sorted(policies.items()):
        policy_source = policy_source_by_id.get(policy_id)
        if policy_source is None:
            raise CapacityAnalysisError(f"Missing policy source metadata: {policy_id}")
        for (selector, event_id), selected_k in sorted(assignments.items()):
            action_key = (selector, event_id, selected_k)
            if action_key not in action_lookup:
                raise CapacityAnalysisError(
                    f"Policy {policy_id} selects unsupported K: selector={selector}, "
                    f"event={event_id}, K={selected_k}"
                )
            policy = action_lookup[action_key]
            oracle = oracle_lookup[(selector, event_id)]
            policy_k = policy.observation.requested_k
            oracle_k = oracle.observation.requested_k
            if policy_k < oracle_k:
                relation = "underfill"
            elif policy_k > oracle_k:
                relation = "overfill"
            else:
                relation = "capacity_correct"
            objective_regret = policy.objective - oracle.objective
            if objective_regret < -tie_atol:
                raise CapacityAnalysisError(
                    f"Oracle invariant failed for selector={selector}, event={event_id}: "
                    f"regret={objective_regret}"
                )
            objective_regret = max(0.0, objective_regret)
            regret_rows.append(
                {
                    "policy_id": policy_id,
                    "policy_verification_status": policy_source.get(
                        "verification_status"
                    ),
                    "policy_uses_gold": policy_source.get("uses_gold"),
                    "policy_uses_verifier_logits": policy_source.get(
                        "uses_verifier_logits"
                    ),
                    "policy_fit_split": policy_source.get("fit_split"),
                    "policy_decision_source_split": policy_source.get(
                        "decision_source_split"
                    ),
                    "policy_deployable_without_gold": bool(
                        policy_source.get("deployable_without_gold")
                    ),
                    "policy_deployable_ex_ante": bool(
                        policy_source.get("deployable_ex_ante")
                    ),
                    "event_id": event_id,
                    "selector_level": selector,
                    "selected_k_input": selected_k,
                    "policy_canonical_requested_k": policy_k,
                    "policy_requested_k_aliases": list(policy.requested_k_aliases),
                    "policy_realized_k": policy.observation.realized_k,
                    "policy_prompt_token_count": policy.observation.prompt_token_count,
                    "policy_raw_nll": policy.raw_nll,
                    "policy_balanced_nll": policy.balanced_nll,
                    "policy_objective": policy.objective,
                    "oracle_requested_k": oracle_k,
                    "oracle_requested_k_aliases": list(oracle.requested_k_aliases),
                    "oracle_realized_k": oracle.observation.realized_k,
                    "oracle_prompt_token_count": oracle.observation.prompt_token_count,
                    "oracle_raw_nll": oracle.raw_nll,
                    "oracle_balanced_nll": oracle.balanced_nll,
                    "oracle_objective": oracle.objective,
                    "capacity_relation": relation,
                    "objective_regret": objective_regret,
                    "raw_nll_delta": policy.raw_nll - oracle.raw_nll,
                    "balanced_nll_delta": policy.balanced_nll - oracle.balanced_nll,
                    "prompt_token_delta": policy.observation.prompt_token_count
                    - oracle.observation.prompt_token_count,
                    "realized_k_delta": policy.observation.realized_k
                    - oracle.observation.realized_k,
                }
            )

    curve_rows = _capacity_curve_rows(
        observations=observations,
        labels=labels,
        event_ids=event_ids,
        token_penalty_per_1k=token_penalty_per_1k,
    )
    aggregate = _summarize(
        oracle_rows,
        regret_rows,
        curve_rows,
        policy_sources=policy_sources,
        tie_atol=tie_atol,
    )
    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(staging / "prefix_scores.jsonl", prefix_score_rows)
        _write_jsonl(staging / "capacity_curves.jsonl", curve_rows)
        _write_jsonl(staging / "oracle_events.jsonl", oracle_rows)
        _write_jsonl(staging / "policy_regret.jsonl", regret_rows)
        _write_json(staging / "aggregate.json", aggregate)
        (staging / "report.md").write_text(
            _render_report(
                aggregate=aggregate,
                class_balance=class_balance,
                token_penalty_per_1k=token_penalty_per_1k,
            ),
            encoding="utf-8",
        )
        artifact_names = (
            "prefix_scores.jsonl",
            "capacity_curves.jsonl",
            "oracle_events.jsonl",
            "policy_regret.jsonl",
            "aggregate.json",
            "report.md",
        )
        artifacts = {
            name: {
                "path": name,
                "sha256": _sha256_file(staging / name),
                "size": (staging / name).stat().st_size,
            }
            for name in artifact_names
        }
        implementation_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_status": ANALYSIS_STATUS,
            "split": matrix.get("split"),
            "event_count": len(event_ids),
            "selector_count": len(selectors),
            "oracle_row_count": len(oracle_rows),
            "capacity_curve_row_count": len(curve_rows),
            "policy_regret_row_count": len(regret_rows),
            "labels": labels,
            "class_balance": class_balance,
            "alignment": {
                "primary_key": ["selector_level", "event_id"],
                "event_id_sequence_sha256": _event_sequence_sha256(event_ids),
                "all_prefix_cells_exactly_aligned": True,
            },
            "settings": settings,
            "source": source,
            "policy_sources": policy_sources,
            "artifacts": artifacts,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "implementation": {
                "path": str(implementation_path),
                "sha256": _sha256_file(implementation_path),
            },
            "interpretation_boundary": (
                "Gold-defined per-event capacity oracle under one frozen verifier, selector "
                "order, prompt renderer, map cache, and split. It is not deployable, not a "
                "causal capacity effect, and does not estimate verifier-seed or test uncertainty."
            ),
        }
        _write_json(staging / "manifest.json", manifest)
        _promote_directory(staging, output_dir, force=force)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _parse_policy_args(values: Sequence[str]) -> dict[str, Path]:
    policies: dict[str, Path] = {}
    for value in values:
        policy_id, separator, path = value.partition("=")
        if not separator or not policy_id or not path:
            raise argparse.ArgumentTypeError(
                f"--policy must be POLICY_ID=JSONL_PATH, got {value!r}"
            )
        if policy_id in policies:
            raise argparse.ArgumentTypeError(f"Duplicate --policy ID: {policy_id}")
        policies[policy_id] = Path(path)
    return policies


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="POLICY_ID=JSONL_PATH",
        help="JSONL rows require event_id, selector_level, and selected_k",
    )
    parser.add_argument(
        "--include-fixed-policies",
        action="store_true",
        help="Evaluate every requested K as a generated fixed-capacity policy.",
    )
    parser.add_argument("--token-penalty-per-1k", type=float, default=0.0)
    parser.add_argument("--tie-atol", type=float, default=1e-12)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    materialize_capacity_analysis(
        matrix_manifest_path=Path(args.matrix_manifest),
        output_dir=Path(args.output_dir),
        policy_paths=_parse_policy_args(args.policy),
        include_fixed_policies=bool(args.include_fixed_policies),
        token_penalty_per_1k=args.token_penalty_per_1k,
        tie_atol=args.tie_atol,
        force=args.force,
    )


if __name__ == "__main__":
    main()
