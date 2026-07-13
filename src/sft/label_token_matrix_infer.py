from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from tqdm.auto import tqdm

from fact_checking.data.constants import labels_for_schema, letter_order_for_schema
from fact_checking.utils.logging import init_logger
from sft.data.io import load_prebuilt_samples, save_eval_artifacts
from sft.dataset.loaders import build_dataloader
from sft.eval import build_eval_metrics
from sft.infer_common import build_inference_context, build_serializable_metrics
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.label_token_multi_infer import _load_model
from sft.label_token_trainer import (
    _build_label_token_ids,
    _checkpoint_selection_score,
    _coverage_label_token_enabled,
    _forward_label_logits,
    _true_side_macro_f1,
)
from sft.runtime.deps import flash_attn2_available


logger = init_logger(__name__)

INPUT_SCHEMA_VERSION = "deduplicated_label_token_matrix_input_v0_2"
RAW_LOGITS_SCHEMA_VERSION = "deduplicated_raw_label_logits_v0_2"
MATRIX_RESULT_SCHEMA_VERSION = "deduplicated_label_token_matrix_result_v0_2"
CACHE_KEY_VERSION = "label_token_prompt_ids_plus_prefix_v0_1"


class MatrixValidationError(RuntimeError):
    """Raised when a frozen matrix artifact violates the runner contract."""


class EquivalenceGateError(MatrixValidationError):
    """Raised after writing a failed native-equivalence gate report."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise MatrixValidationError(f"Expected a JSON object in {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise MatrixValidationError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise MatrixValidationError(
                    f"Expected a JSON object in {path}:{line_number}"
                )
            yield row


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


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


def _choice_text(label_prefix: str, letter: str) -> str:
    return letter if label_prefix.endswith((" ", "\n", "\t")) else " " + letter


def _prompt_ids_sha256(prompt_input_ids: Sequence[int]) -> str:
    return _sha256_json([int(token_id) for token_id in prompt_input_ids])


def _prompt_cache_key(prompt_input_ids: Sequence[int], label_prefix: str) -> str:
    return _sha256_json(
        {
            "cache_key_version": CACHE_KEY_VERSION,
            "label_prefix": label_prefix,
            "prompt_input_ids": [int(token_id) for token_id in prompt_input_ids],
        }
    )


def _event_sequence_sha256(event_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for event_id in event_ids:
        digest.update(event_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_factorial_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not bool(manifest.get("all_ready", False)):
        raise MatrixValidationError("Factorial manifest is not all_ready=true")
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise MatrixValidationError("Factorial manifest has no cells")
    cells = [dict(cell) for cell in raw_cells if isinstance(cell, dict)]
    if len(cells) != len(raw_cells):
        raise MatrixValidationError("Factorial manifest contains a non-object cell")

    cell_ids = [str(cell.get("cell_id") or "") for cell in cells]
    if any(not cell_id for cell_id in cell_ids) or len(set(cell_ids)) != len(cell_ids):
        raise MatrixValidationError("Factorial cell IDs must be non-empty and unique")
    for cell in cells:
        if not bool(cell.get("ready", False)):
            raise MatrixValidationError(f"Factorial cell is not ready: {cell['cell_id']}")

    selector_levels = [str(value) for value in manifest.get("selector_levels", [])]
    controller_levels = [str(value) for value in manifest.get("controller_levels", [])]
    if selector_levels and controller_levels:
        expected = {
            (selector, controller)
            for selector in selector_levels
            for controller in controller_levels
        }
        actual_pairs = [
            (str(cell.get("selector_level")), str(cell.get("controller_level")))
            for cell in cells
        ]
        actual = set(actual_pairs)
        if len(actual_pairs) != len(actual):
            raise MatrixValidationError(
                "Factorial grid contains duplicate selector/controller coordinates"
            )
        if actual != expected:
            raise MatrixValidationError(
                "Factorial grid is incomplete or contains extra cells: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    declared_count = int(manifest.get("cell_count", len(cells)))
    if declared_count != len(cells):
        raise MatrixValidationError(
            f"cell_count={declared_count} differs from cells length={len(cells)}"
        )
    return cells


_CELL_METADATA_FIELDS = (
    "capacity_k",
    "capacity_policy",
    "source_order_cell",
)


def _cell_metadata(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: cell.get(field)
        for field in _CELL_METADATA_FIELDS
        if cell.get(field) is not None
    }


def _compact_build_row(
    row: Mapping[str, Any],
    *,
    label_prefix: str,
    expected_label_schema: str | None,
    cell_id: str,
    sample_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        raise MatrixValidationError(f"{cell_id}[{sample_idx}] has no event_id")
    prompt = str(row.get("prompt") or "")
    if not prompt:
        raise MatrixValidationError(f"{cell_id}[{sample_idx}] has an empty prompt")
    raw_prompt_ids = row.get("prompt_input_ids")
    if not isinstance(raw_prompt_ids, list) or not raw_prompt_ids:
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] must contain non-empty prompt_input_ids"
        )
    try:
        prompt_input_ids = [int(token_id) for token_id in raw_prompt_ids]
    except (TypeError, ValueError) as exc:
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] contains a non-integer prompt token"
        ) from exc
    prompt_token_count = int(row.get("prompt_token_count", -1))
    if prompt_token_count != len(prompt_input_ids):
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] prompt_token_count={prompt_token_count} "
            f"but len(prompt_input_ids)={len(prompt_input_ids)}"
        )
    if bool(row.get("prompt_add_special_tokens", False)):
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] must use prompt_add_special_tokens=false"
        )
    if not bool(row.get("preserve_prompt_prefix", True)):
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] must use preserve_prompt_prefix=true"
        )
    if bool(row.get("was_truncated", False)) or bool(
        row.get("evidence_text_truncated", False)
    ):
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] is truncated; raw-logit reuse is fail-closed"
        )

    label_schema = str(row.get("label_schema") or "liar6")
    if expected_label_schema and label_schema != expected_label_schema:
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] label_schema={label_schema!r}; "
            f"expected {expected_label_schema!r}"
        )
    labels = labels_for_schema(label_schema)
    letters = letter_order_for_schema(label_schema)
    gold_id = int(row.get("gold_id", -1))
    gold_label = str(row.get("gold_label") or "")
    if not 0 <= gold_id < len(labels) or gold_label != labels[gold_id]:
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] has inconsistent gold label: "
            f"gold_id={gold_id}, gold_label={gold_label!r}"
        )
    target = str(row.get("target") or "")
    expected_target = f"{label_prefix}{_choice_text(label_prefix, letters[gold_id])}"
    if target != expected_target:
        raise MatrixValidationError(
            f"{cell_id}[{sample_idx}] target={target!r}; expected {expected_target!r}"
        )

    prompt_ids_sha = _prompt_ids_sha256(prompt_input_ids)
    prompt_text_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    target_sha = hashlib.sha256(target.encode("utf-8")).hexdigest()
    cache_key = _prompt_cache_key(prompt_input_ids, label_prefix)
    compact = {
        "event_id": event_id,
        "prompt": prompt,
        "target": target,
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "gold_id": gold_id,
        "gold_label": gold_label,
        "gold_explain": str(row.get("gold_explain", row.get("explain", ""))),
        "prompt_token_count": prompt_token_count,
        "target_token_count": int(row.get("target_token_count", 0)),
        "evidence_count": int(row.get("evidence_count", 0)),
        "was_truncated": False,
        "claim": str(row.get("claim") or ""),
        "label_schema": label_schema,
        "prompt_input_ids": prompt_input_ids,
        "coverage_label": str(row.get("coverage_label") or ""),
        "prompt_cache_key": cache_key,
        "prompt_input_ids_sha256": prompt_ids_sha,
        "prompt_text_sha256": prompt_text_sha,
        "target_sha256": target_sha,
    }
    mapping = {
        "cell_id": cell_id,
        "cell_sample_idx": int(sample_idx),
        "event_id": event_id,
        "prompt_cache_key": cache_key,
        "prompt_input_ids_sha256": prompt_ids_sha,
        "prompt_text_sha256": prompt_text_sha,
        "target_sha256": target_sha,
        "gold_id": gold_id,
        "gold_label": gold_label,
        "evidence_count": int(row.get("evidence_count", 0)),
        "prompt_token_count": prompt_token_count,
    }
    return compact, mapping


def _strict_duplicate_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "prompt_input_ids_sha256": row["prompt_input_ids_sha256"],
        "prompt_text_sha256": row["prompt_text_sha256"],
        "target_sha256": row["target_sha256"],
        "gold_id": int(row["gold_id"]),
        "gold_label": row["gold_label"],
        "gold_explain": row["gold_explain"],
        "label_schema": row["label_schema"],
        "coverage_label": row.get("coverage_label", ""),
        "prompt_add_special_tokens": bool(row["prompt_add_special_tokens"]),
        "preserve_prompt_prefix": bool(row["preserve_prompt_prefix"]),
    }


def prepare_matrix(
    *,
    matrix_manifest_path: Path,
    build_root: Path,
    output_dir: Path,
    split: str,
    label_prefix: str,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze unique model inputs and per-cell mappings without loading a model."""
    matrix_manifest = _read_json(matrix_manifest_path)
    cells = _validate_factorial_manifest(matrix_manifest)
    expected_schema = str(matrix_manifest.get("label_schema") or "") or None
    expected_rows = int(
        matrix_manifest.get("event_count", matrix_manifest.get("source_event_count", 0))
    )
    if expected_rows <= 0:
        raise MatrixValidationError("Factorial manifest has no positive event_count")
    expected_event_sha = str(matrix_manifest.get("event_id_sequence_sha256") or "")

    target_dir = output_dir / "input"
    if target_dir.exists() and not force:
        existing = _read_json(target_dir / "manifest.json")
        compatible = (
            existing.get("schema_version") == INPUT_SCHEMA_VERSION
            and existing.get("status") == "complete"
            and existing.get("split") == split
            and existing.get("label_prefix") == label_prefix
            and Path(str(existing.get("build_root") or "")).resolve() == build_root.resolve()
            and existing.get("matrix_manifest_sha256") == _sha256_file(matrix_manifest_path)
        )
        if not compatible:
            raise MatrixValidationError(
                f"Prepared matrix input is incompatible: {target_dir}; "
                "pass --force-prepare to replace"
            )
        _load_prepared_input(output_dir)
        for cell in existing.get("cells", []):
            source_build = Path(str(cell["source_build"]))
            if _sha256_file(source_build) != str(cell["source_build_sha256"]):
                raise MatrixValidationError(
                    f"Frozen source build has drifted for {cell['cell_id']}; "
                    "pass --force-prepare only after auditing the change"
                )
        logger.info("[matrix-prepare] compatible frozen input already exists at %s", target_dir)
        return existing
    staging = output_dir / f".input.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "cells").mkdir(parents=True, exist_ok=False)

    unique_by_key: dict[str, dict[str, Any]] = {}
    signature_by_key: dict[str, dict[str, Any]] = {}
    cell_mappings: dict[str, list[dict[str, Any]]] = {}
    cell_artifacts: list[dict[str, Any]] = []
    reference_count = 0
    common_event_ids: list[str] | None = None
    common_label_schema: str | None = expected_schema

    try:
        for cell in cells:
            cell_id = str(cell["cell_id"])
            build_path = build_root / cell_id / "build" / f"build_{split}.jsonl"
            if not build_path.exists():
                raise FileNotFoundError(f"Missing build artifact for {cell_id}: {build_path}")
            mappings: list[dict[str, Any]] = []
            event_ids: list[str] = []
            seen_events: set[str] = set()
            build_hasher = hashlib.sha256()
            with build_path.open("rb") as handle:
                for sample_idx, raw_line in enumerate(handle):
                    if not raw_line.strip():
                        raise MatrixValidationError(
                            f"Blank line in frozen build artifact {build_path}:{sample_idx + 1}"
                        )
                    build_hasher.update(raw_line)
                    try:
                        raw_row = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise MatrixValidationError(
                            f"Invalid JSON in {build_path}:{sample_idx + 1}: {exc}"
                        ) from exc
                    if not isinstance(raw_row, dict):
                        raise MatrixValidationError(
                            f"Expected object in {build_path}:{sample_idx + 1}"
                        )
                    compact, mapping = _compact_build_row(
                        raw_row,
                        label_prefix=label_prefix,
                        expected_label_schema=expected_schema,
                        cell_id=cell_id,
                        sample_idx=sample_idx,
                    )
                    row_schema = str(compact["label_schema"])
                    if common_label_schema is None:
                        common_label_schema = row_schema
                    elif row_schema != common_label_schema:
                        raise MatrixValidationError(
                            f"{cell_id}[{sample_idx}] label_schema={row_schema!r}; "
                            f"matrix schema is {common_label_schema!r}"
                        )
                    event_id = str(compact["event_id"])
                    if event_id in seen_events:
                        raise MatrixValidationError(
                            f"{cell_id} contains duplicate event_id={event_id!r}"
                        )
                    seen_events.add(event_id)
                    event_ids.append(event_id)
                    cache_key = str(compact["prompt_cache_key"])
                    signature = _strict_duplicate_signature(compact)
                    if cache_key in unique_by_key:
                        if signature_by_key[cache_key] != signature:
                            raise MatrixValidationError(
                                "Equal model input IDs have conflicting strict metadata: "
                                f"cell={cell_id}, sample_idx={sample_idx}, key={cache_key}"
                            )
                    else:
                        unique_by_key[cache_key] = compact
                        signature_by_key[cache_key] = signature
                    mappings.append(mapping)

            source_build_sha256 = build_hasher.hexdigest()
            declared_build_sha256 = str(cell.get("build_sha256") or "")
            if declared_build_sha256 and source_build_sha256 != declared_build_sha256:
                raise MatrixValidationError(
                    f"{cell_id} source build SHA disagrees with the matrix manifest: "
                    f"{source_build_sha256} != {declared_build_sha256}"
                )
            declared_build_file = str(cell.get("build_file") or "")
            if declared_build_file:
                declared_path = Path(declared_build_file)
                if not declared_path.is_absolute():
                    declared_path = matrix_manifest_path.parent / declared_path
                if declared_path.resolve() != build_path.resolve():
                    raise MatrixValidationError(
                        f"{cell_id} source build path disagrees with the matrix manifest: "
                        f"{build_path.resolve()} != {declared_path.resolve()}"
                    )

            if len(mappings) != expected_rows:
                raise MatrixValidationError(
                    f"{cell_id} has {len(mappings)} build rows; expected {expected_rows}"
                )
            declared_rows = int(cell.get("row_count", expected_rows))
            if declared_rows != len(mappings):
                raise MatrixValidationError(
                    f"{cell_id} manifest row_count={declared_rows}; build rows={len(mappings)}"
                )
            event_sha = _event_sequence_sha256(event_ids)
            if expected_event_sha and event_sha != expected_event_sha:
                raise MatrixValidationError(
                    f"{cell_id} event sequence SHA mismatch: {event_sha} != {expected_event_sha}"
                )
            if common_event_ids is None:
                common_event_ids = event_ids
            elif event_ids != common_event_ids:
                raise MatrixValidationError(f"{cell_id} event order differs from the first cell")

            cell_mappings[cell_id] = mappings
            reference_count += len(mappings)
            cell_artifacts.append(
                {
                    "cell_id": cell_id,
                    "selector_level": cell.get("selector_level"),
                    "controller_level": cell.get("controller_level"),
                    **_cell_metadata(cell),
                    "row_count": len(mappings),
                    "source_build": str(build_path),
                    "source_build_sha256": source_build_sha256,
                    "event_id_sequence_sha256": event_sha,
                    "mapping_file": f"cells/{cell_id}.jsonl",
                }
            )

        ordered_unique = sorted(
            unique_by_key.values(),
            key=lambda row: (len(row["prompt_input_ids"]), str(row["prompt_cache_key"])),
        )
        unique_idx_by_key: dict[str, int] = {}
        for unique_idx, row in enumerate(ordered_unique):
            row["unique_idx"] = unique_idx
            unique_idx_by_key[str(row["prompt_cache_key"])] = unique_idx
        for cell_id, mappings in cell_mappings.items():
            for mapping in mappings:
                mapping["unique_idx"] = unique_idx_by_key[str(mapping["prompt_cache_key"])]
            _write_jsonl(staging / "cells" / f"{cell_id}.jsonl", mappings)
        _write_jsonl(staging / "unique_rows.jsonl", ordered_unique)

        for cell_artifact in cell_artifacts:
            mapping_path = staging / str(cell_artifact["mapping_file"])
            cell_artifact["mapping_sha256"] = _sha256_file(mapping_path)
        unique_path = staging / "unique_rows.jsonl"
        unique_count = len(ordered_unique)
        duplicate_count = reference_count - unique_count
        label_schema = common_label_schema or "liar6"
        manifest = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "status": "complete",
            "created_at": _utc_now(),
            "split": split,
            "label_schema": label_schema,
            "label_prefix": label_prefix,
            "cache_key_version": CACHE_KEY_VERSION,
            "matrix_manifest": str(matrix_manifest_path),
            "matrix_manifest_sha256": _sha256_file(matrix_manifest_path),
            "matrix_schema_version": matrix_manifest.get("schema_version"),
            "build_root": str(build_root),
            "cell_count": len(cell_artifacts),
            "event_count": expected_rows,
            "reference_count": reference_count,
            "unique_prompt_count": unique_count,
            "duplicate_reference_count": duplicate_count,
            "reuse_rate": duplicate_count / reference_count if reference_count else 0.0,
            "theoretical_forward_reduction": (
                reference_count / unique_count if unique_count else 0.0
            ),
            "event_id_sequence_sha256": _event_sequence_sha256(common_event_ids or []),
            "unique_rows_file": "unique_rows.jsonl",
            "unique_rows_sha256": _sha256_file(unique_path),
            "strict_duplicate_fields": sorted(_strict_duplicate_signature(ordered_unique[0]).keys())
            if ordered_unique
            else [],
            "input_contract": {
                "prompt_add_special_tokens": False,
                "preserve_prompt_prefix": True,
                "was_truncated": False,
                "evidence_text_truncated": False,
                "cache_identity": "prompt_input_ids + label_prefix",
            },
            "cells": cell_artifacts,
        }
        _write_json(staging / "manifest.json", manifest)
        _promote_directory(staging, target_dir, force=force)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    logger.info(
        "[matrix-prepare] references=%s unique=%s duplicates=%s reuse=%.2f%% output=%s",
        manifest["reference_count"],
        manifest["unique_prompt_count"],
        manifest["duplicate_reference_count"],
        100.0 * float(manifest["reuse_rate"]),
        target_dir,
    )
    return manifest


def _artifact_file_identity(path: Path, *, force_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if force_hash or stat.st_size <= 256 * 1024 * 1024:
        payload["sha256"] = _sha256_file(path)
    return payload


def _existing_file_identities(
    root: Path,
    names: Sequence[str],
    *,
    force_hash_names: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    force_hash_names = force_hash_names or set()
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if path.is_file():
            out[name] = _artifact_file_identity(path, force_hash=name in force_hash_names)
    return out


def _model_weight_file_names(root: Path) -> list[str]:
    """Return exactly the weight files selected by a local HF checkpoint."""
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        index_payload = _read_json(index_path)
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise MatrixValidationError(f"Weight index has no weight_map: {index_path}")
        names = sorted({str(value) for value in weight_map.values()})
        missing = [name for name in names if not (root / name).is_file()]
        if missing:
            raise MatrixValidationError(
                f"Weight index {index_path} references missing shards: {missing}"
            )
        return names
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (root / name).is_file():
            return [name]
    raise MatrixValidationError(f"Cannot identify model weight files under {root}")


def _checkpoint_provenance(context, *, config_path: str | None) -> dict[str, Any]:
    checkpoint_dir = Path(context.checkpoint_dir).resolve()
    checkpoint_files = _existing_file_identities(
        checkpoint_dir,
        [
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_config.json",
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "generation_config.json",
        ],
        force_hash_names={"adapter_model.safetensors", "adapter_model.bin"},
    )
    adapter_sha = None
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if name in checkpoint_files:
            adapter_sha = checkpoint_files[name].get("sha256")
            break

    base_root = Path(context.model_name_or_path).resolve()
    base_files = _existing_file_identities(
        base_root,
        [
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "generation_config.json",
        ],
    )
    actual_model_root = base_root if context.is_peft_adapter else checkpoint_dir
    weight_names = _model_weight_file_names(actual_model_root)
    model_weight_files = _existing_file_identities(
        actual_model_root,
        weight_names,
        force_hash_names=set(weight_names),
    )
    if set(model_weight_files) != set(weight_names):
        raise MatrixValidationError(
            f"Failed to fingerprint all model weights under {actual_model_root}"
        )
    if context.is_peft_adapter:
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise MatrixValidationError(f"PEFT checkpoint has no adapter_config.json: {checkpoint_dir}")
        adapter_config = _read_json(adapter_config_path)
        declared_base = str(adapter_config.get("base_model_name_or_path") or "").strip()
        if not declared_base or Path(declared_base).resolve() != base_root:
            raise MatrixValidationError(
                "PEFT adapter base_model_name_or_path does not match the resolved verifier base: "
                f"declared={declared_base!r}, resolved={base_root}"
            )
    tokenizer_root = base_root if context.is_peft_adapter else checkpoint_dir
    tokenizer_files = _existing_file_identities(
        tokenizer_root,
        [
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer.model.v3",
            "tekken.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )
    resolved_config = Path(config_path) if config_path else Path(context.run_dir) / "config.resolved.yaml"
    if not resolved_config.exists() and not config_path:
        fallback = Path(context.run_dir).parent / "train.resolved.yaml"
        if fallback.exists():
            resolved_config = fallback
    payload = {
        "run_dir": str(Path(context.run_dir).resolve()),
        "checkpoint_name": context.checkpoint_name,
        "checkpoint_dir": str(checkpoint_dir),
        "is_peft_adapter": bool(context.is_peft_adapter),
        "adapter_sha256": adapter_sha,
        "checkpoint_files": checkpoint_files,
        "base_model_dir": str(base_root),
        "base_model_files": base_files,
        "model_weight_root": str(actual_model_root),
        "model_weight_files": model_weight_files,
        "tokenizer_dir": str(tokenizer_root),
        "tokenizer_files": tokenizer_files,
        "resolved_config": str(resolved_config),
        "resolved_config_sha256": _sha256_file(resolved_config)
        if resolved_config.is_file()
        else None,
    }
    payload["model_identity_sha256"] = _sha256_json(payload)
    return payload


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for package in ("torch", "accelerate", "transformers", "peft", "numpy"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not-installed"
    return out


def restore_gathered_logits(
    *,
    sample_indices: np.ndarray,
    label_logits: np.ndarray,
    gold_ids: np.ndarray,
    expected_count: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Restore stable unique order and validate repeated distributed samples."""
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    logits = np.asarray(label_logits, dtype=np.float32)
    gold = np.asarray(gold_ids, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or len(indices) != len(logits) or len(gold) != len(indices):
        raise MatrixValidationError(
            "Gathered sample_indices, label_logits, and gold_ids have incompatible shapes"
        )
    grouped: dict[int, tuple[np.ndarray, int]] = {}
    duplicate_count = 0
    for idx, row_logits, row_gold in zip(indices.tolist(), logits, gold.tolist()):
        if idx < 0:
            continue
        if not np.isfinite(row_logits).all():
            raise MatrixValidationError(
                f"Gathered unique_idx={idx} contains NaN or Inf raw logits"
            )
        if not 0 <= idx < expected_count:
            raise MatrixValidationError(
                f"Gathered unique_idx={idx} is outside [0, {expected_count})"
            )
        if idx in grouped:
            previous_logits, previous_gold = grouped[idx]
            if previous_gold != int(row_gold):
                raise MatrixValidationError(
                    f"Distributed duplicate unique_idx={idx} has conflicting gold IDs"
                )
            previous_argmax = int(np.argmax(previous_logits))
            repeated_argmax = int(np.argmax(row_logits))
            if previous_argmax != repeated_argmax:
                raise MatrixValidationError(
                    f"Distributed duplicate unique_idx={idx} has conflicting argmax: "
                    f"{previous_argmax} != {repeated_argmax}"
                )
            if not np.allclose(previous_logits, row_logits, atol=atol, rtol=rtol):
                max_diff = float(np.max(np.abs(previous_logits - row_logits)))
                raise MatrixValidationError(
                    f"Distributed duplicate unique_idx={idx} has conflicting logits "
                    f"(max_abs_diff={max_diff})"
                )
            duplicate_count += 1
            continue
        grouped[idx] = (row_logits.copy(), int(row_gold))
    missing = sorted(set(range(expected_count)) - set(grouped))
    if missing:
        raise MatrixValidationError(
            f"Raw-logit gather is incomplete: missing {len(missing)} indices, first={missing[:10]}"
        )
    ordered_indices = np.arange(expected_count, dtype=np.int64)
    ordered_logits = np.stack([grouped[idx][0] for idx in range(expected_count)]).astype(
        np.float32, copy=False
    )
    ordered_gold = np.asarray([grouped[idx][1] for idx in range(expected_count)], dtype=np.int64)
    return ordered_indices, ordered_logits, ordered_gold, duplicate_count


def _load_prepared_input(output_dir: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    input_dir = output_dir / "input"
    manifest_path = input_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION or manifest.get("status") != "complete":
        raise MatrixValidationError(f"Prepared input is not a complete {INPUT_SCHEMA_VERSION}: {manifest_path}")
    unique_path = input_dir / str(manifest["unique_rows_file"])
    actual_sha = _sha256_file(unique_path)
    if actual_sha != str(manifest["unique_rows_sha256"]):
        raise MatrixValidationError(
            f"Prepared unique_rows SHA mismatch: {actual_sha} != {manifest['unique_rows_sha256']}"
        )
    rows = _load_jsonl(unique_path)
    if len(rows) != int(manifest["unique_prompt_count"]):
        raise MatrixValidationError("Prepared unique row count disagrees with its manifest")
    for expected_idx, row in enumerate(rows):
        if int(row.get("unique_idx", -1)) != expected_idx:
            raise MatrixValidationError(
                f"unique_rows.jsonl is not indexed contiguously at row {expected_idx}"
            )
        prompt_ids = row.get("prompt_input_ids")
        if not isinstance(prompt_ids, list):
            raise MatrixValidationError(f"unique_idx={expected_idx} has no prompt_input_ids")
        expected_key = _prompt_cache_key(prompt_ids, str(manifest["label_prefix"]))
        if str(row.get("prompt_cache_key")) != expected_key:
            raise MatrixValidationError(f"unique_idx={expected_idx} has an invalid prompt cache key")
        if str(row.get("label_schema")) != str(manifest["label_schema"]):
            raise MatrixValidationError(
                f"unique_idx={expected_idx} label schema disagrees with prepared manifest"
            )
    mapping_total = 0
    for cell in manifest.get("cells", []):
        mapping_path = input_dir / str(cell["mapping_file"])
        if not mapping_path.is_file() or _sha256_file(mapping_path) != str(cell["mapping_sha256"]):
            raise MatrixValidationError(
                f"Prepared mapping is missing or has a SHA mismatch: {cell.get('cell_id')}"
            )
        mapping_total += int(cell["row_count"])
    if mapping_total != int(manifest["reference_count"]):
        raise MatrixValidationError("Prepared mapping row total disagrees with reference_count")
    return manifest, manifest_path, rows


def infer_raw_logits(
    *,
    output_dir: Path,
    run_dir: Path,
    checkpoint: str,
    split: str,
    config_path: str | None,
    per_device_eval_batch_size: int | None,
    dataloader_num_workers: int | None,
    expected_adapter_sha256: str | None,
    unsafe_unpinned_checkpoint: bool,
    expected_world_size: int | None,
    force: bool,
    duplicate_logit_atol: float,
    duplicate_logit_rtol: float,
) -> dict[str, Any] | None:
    input_manifest, input_manifest_path, unique_rows = _load_prepared_input(output_dir)
    if split != str(input_manifest["split"]):
        raise MatrixValidationError(
            f"infer split={split!r} disagrees with prepared split={input_manifest['split']!r}"
        )
    context = build_inference_context(
        run_dir=run_dir,
        checkpoint=checkpoint,
        split=split,
        config_path=config_path,
    )
    train_cfg = context.train_cfg
    if _coverage_label_token_enabled(train_cfg):
        raise MatrixValidationError(
            "coverage_label_token.enabled=true is outside this raw six-class logit cache contract"
        )
    label_cfg = train_cfg.get("label_token_ce", {}) or {}
    label_prefix = str(label_cfg.get("label_prefix", "Label:"))
    if label_prefix != str(input_manifest["label_prefix"]):
        raise MatrixValidationError(
            f"Verifier label_prefix={label_prefix!r} disagrees with prepared prefix="
            f"{input_manifest['label_prefix']!r}"
        )
    if context.label_schema != str(input_manifest["label_schema"]):
        raise MatrixValidationError(
            f"Verifier label_schema={context.label_schema!r} disagrees with prepared schema="
            f"{input_manifest['label_schema']!r}"
        )
    labels = labels_for_schema(context.label_schema)
    letter_order = letter_order_for_schema(context.label_schema)
    label_token_id_list, label_token_meta = _build_label_token_ids(
        context.tokenizer,
        label_prefix=label_prefix,
        letter_order=letter_order,
    )
    mixed_precision = "bf16" if bool(train_cfg.get("bf16", True)) else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    if expected_world_size is not None and int(accelerator.num_processes) != int(
        expected_world_size
    ):
        raise MatrixValidationError(
            f"Accelerate world_size={accelerator.num_processes}; expected {expected_world_size}. "
            "Check the launcher and --multi_gpu setting."
        )
    provenance_box: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        provenance_box[0] = _checkpoint_provenance(context, config_path=config_path)
    broadcast_object_list(provenance_box, from_process=0)
    checkpoint_provenance = provenance_box[0]
    if checkpoint_provenance is None:
        raise MatrixValidationError("Failed to broadcast checkpoint provenance")
    actual_adapter_sha = checkpoint_provenance.get("adapter_sha256")
    if context.is_peft_adapter and not expected_adapter_sha256 and not unsafe_unpinned_checkpoint:
        raise MatrixValidationError(
            "Formal PEFT matrix inference requires --expected-adapter-sha256; "
            "use --unsafe-unpinned-checkpoint only for diagnostics"
        )
    if expected_adapter_sha256 and actual_adapter_sha != expected_adapter_sha256:
        raise MatrixValidationError(
            "Adapter SHA-256 mismatch: "
            f"actual={actual_adapter_sha!r}, expected={expected_adapter_sha256!r}"
        )
    input_manifest_sha = _sha256_file(input_manifest_path)
    scoring_contract = {
        "schema_version": RAW_LOGITS_SCHEMA_VERSION,
        "input_manifest_sha256": input_manifest_sha,
        "unique_rows_sha256": input_manifest["unique_rows_sha256"],
        "model_identity_sha256": checkpoint_provenance["model_identity_sha256"],
        "label_schema": context.label_schema,
        "labels": labels,
        "letter_order": letter_order,
        "label_prefix": label_prefix,
        "prefix_token_ids": label_token_meta["prefix_token_ids"],
        "label_token_ids": label_token_meta["label_token_ids"],
        "max_length": int(context.max_length),
        "logit_adjust": {"enabled": False},
        "forward_contract": "last non-padding position, restricted raw label-token logits",
    }
    scoring_fingerprint = _sha256_json(scoring_contract)
    batch_size = int(
        per_device_eval_batch_size
        if per_device_eval_batch_size is not None
        else train_cfg.get("per_device_eval_batch_size", 1)
    )
    num_workers = int(
        dataloader_num_workers
        if dataloader_num_workers is not None
        else train_cfg.get("dataloader_num_workers", 0)
    )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Evaluation batch size must be positive and worker count non-negative")
    cuda_device: dict[str, Any] | None = None
    if torch.cuda.is_available():
        device_index = accelerator.device.index if accelerator.device.index is not None else 0
        properties = torch.cuda.get_device_properties(device_index)
        cuda_device = {
            "name": properties.name,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory": int(properties.total_memory),
        }
    attention_backend = (
        "flash_attention_2"
        if bool(train_cfg.get("use_flash_attention_2", True)) and flash_attn2_available()
        else "transformers_default"
    )
    gathered_devices = gather_object([cuda_device])
    if not isinstance(gathered_devices, list):
        gathered_devices = [gathered_devices]
    device_identities = [device for device in gathered_devices if device is not None]
    if device_identities and any(device != device_identities[0] for device in device_identities[1:]):
        raise MatrixValidationError(
            f"Heterogeneous CUDA devices are outside the frozen execution contract: {device_identities}"
        )
    execution_contract = {
        "scoring_fingerprint": scoring_fingerprint,
        "python_executable": str(Path(sys.executable).resolve()),
        "per_device_eval_batch_size": batch_size,
        "dataloader_num_workers": num_workers,
        "world_size": int(accelerator.num_processes),
        "mixed_precision": mixed_precision,
        "attention_backend": attention_backend,
        "cuda_devices": device_identities,
        "duplicate_logit_atol": float(duplicate_logit_atol),
        "duplicate_logit_rtol": float(duplicate_logit_rtol),
        "packages": _package_versions(),
    }
    execution_box: list[dict[str, Any] | None] = [
        execution_contract if accelerator.is_main_process else None
    ]
    broadcast_object_list(execution_box, from_process=0)
    if execution_box[0] is None:
        raise MatrixValidationError("Failed to broadcast execution contract")
    execution_contract = execution_box[0]
    execution_fingerprint = _sha256_json(execution_contract)

    cache_dir = output_dir / "raw_logits"
    existing_manifest_path = cache_dir / "manifest.json"
    if existing_manifest_path.exists() and not force:
        existing = _read_json(existing_manifest_path)
        if (
            existing.get("status") == "complete"
            and existing.get("scoring_fingerprint") == scoring_fingerprint
            and existing.get("execution_fingerprint") == execution_fingerprint
        ):
            _load_raw_logits_cache(
                output_dir,
                input_manifest_sha256=input_manifest_sha,
            )
            if accelerator.is_main_process:
                logger.info("[matrix-infer] compatible raw-logit cache already exists at %s", cache_dir)
            accelerator.wait_for_everyone()
            return existing if accelerator.is_main_process else None
        raise MatrixValidationError(
            "Existing raw-logit cache is incompatible with the requested scoring/execution "
            f"contract: {cache_dir}; pass --force-infer to replace"
        )

    samples = load_prebuilt_samples(unique_rows)
    if len(samples) != len(unique_rows):
        raise MatrixValidationError("Prepared unique rows were unexpectedly filtered as unlabeled")
    dataset = LabelTokenDataset(
        samples,
        context.tokenizer,
        max_length=context.max_length,
        label_prefix=label_prefix,
        label_schema=context.label_schema,
        coverage_enabled=False,
    )
    prefix_ids = [int(value) for value in label_token_meta["prefix_token_ids"]]
    for unique_idx, (source_row, tokenized_row) in enumerate(zip(unique_rows, dataset.tokenized)):
        expected_ids = [int(value) for value in source_row["prompt_input_ids"]] + prefix_ids
        if tokenized_row["input_ids"] != expected_ids:
            raise MatrixValidationError(
                f"LabelTokenDataset changed final model input IDs for unique_idx={unique_idx}"
            )
    collator = LabelTokenCollator(tokenizer=context.tokenizer, pad_to_multiple_of=8)
    dataloader = build_dataloader(
        dataset,
        collator=collator,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        use_length_bucket=False,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = _load_model(context=context, train_cfg=train_cfg, mixed_precision=mixed_precision)
    model, dataloader = accelerator.prepare(model, dataloader)
    label_token_ids = torch.tensor(label_token_id_list, dtype=torch.long)
    gathered_logits: list[np.ndarray] = []
    gathered_indices: list[np.ndarray] = []
    gathered_gold: list[np.ndarray] = []
    model.eval()
    progress = tqdm(
        total=len(dataloader),
        desc="deduplicated-raw-label-logits",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )
    with torch.no_grad():
        for batch in dataloader:
            logits = _forward_label_logits(model, batch, label_token_ids).float()
            indices = batch["sample_indices"].to(torch.long)
            gold_ids = batch["gold_ids"].to(torch.long)
            logits = accelerator.pad_across_processes(logits, dim=0, pad_index=0.0)
            indices = accelerator.pad_across_processes(indices, dim=0, pad_index=-1)
            gold_ids = accelerator.pad_across_processes(gold_ids, dim=0, pad_index=-1)
            gathered_logits.append(accelerator.gather(logits).cpu().numpy())
            gathered_indices.append(accelerator.gather(indices).cpu().numpy())
            gathered_gold.append(accelerator.gather(gold_ids).cpu().numpy())
            progress.update(1)
    progress.close()
    ordered_indices, ordered_logits, ordered_gold, distributed_duplicates = restore_gathered_logits(
        sample_indices=np.concatenate(gathered_indices),
        label_logits=np.concatenate(gathered_logits),
        gold_ids=np.concatenate(gathered_gold),
        expected_count=len(unique_rows),
        atol=duplicate_logit_atol,
        rtol=duplicate_logit_rtol,
    )

    accelerator.wait_for_everyone()
    manifest: dict[str, Any] | None = None
    if accelerator.is_main_process:
        staging = output_dir / f".raw_logits.tmp.{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        try:
            logits_path = staging / "raw_label_logits.npz"
            with logits_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    label_logits=ordered_logits,
                    gold_ids=ordered_gold,
                    unique_indices=ordered_indices,
                )
            index_rows = [
                {
                    "unique_idx": int(row["unique_idx"]),
                    "prompt_cache_key": str(row["prompt_cache_key"]),
                    "prompt_input_ids_sha256": str(row["prompt_input_ids_sha256"]),
                    "event_id": str(row["event_id"]),
                    "gold_id": int(row["gold_id"]),
                    "gold_label": str(row["gold_label"]),
                }
                for row in unique_rows
            ]
            index_path = staging / "raw_logits_index.jsonl"
            _write_jsonl(index_path, index_rows)
            ordinal_cfg = dict(label_cfg.get("ordinal_loss", {}) or {})
            manifest = {
                "schema_version": RAW_LOGITS_SCHEMA_VERSION,
                "status": "complete",
                "created_at": _utc_now(),
                "split": split,
                "num_unique_prompts": len(unique_rows),
                "num_labels": len(labels),
                "label_schema": context.label_schema,
                "labels": labels,
                "letter_order": letter_order,
                "label_token_meta": label_token_meta,
                "max_length": int(context.max_length),
                "logit_adjust": {"enabled": False},
                "input_manifest": str(input_manifest_path),
                "input_manifest_sha256": input_manifest_sha,
                "scoring_contract": scoring_contract,
                "scoring_fingerprint": scoring_fingerprint,
                "execution_contract": execution_contract,
                "execution_fingerprint": execution_fingerprint,
                "distributed_duplicate_count": distributed_duplicates,
                "duplicate_logit_atol": duplicate_logit_atol,
                "duplicate_logit_rtol": duplicate_logit_rtol,
                "checkpoint": checkpoint_provenance,
                "raw_logits_file": "raw_label_logits.npz",
                "raw_logits_sha256": _sha256_file(logits_path),
                "raw_logits_dtype": str(ordered_logits.dtype),
                "raw_logits_shape": list(ordered_logits.shape),
                "index_file": "raw_logits_index.jsonl",
                "index_sha256": _sha256_file(index_path),
                "loss_replay_contract": {
                    "native_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
                    "native_eval_global_step": 0,
                    "ordinal_loss": ordinal_cfg,
                    "class_weights": dict(label_cfg.get("class_weights", {}) or {}),
                },
                "packages": _package_versions(),
            }
            _write_json(staging / "manifest.json", manifest)
            _promote_directory(staging, cache_dir, force=force)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        logger.info(
            "[matrix-infer] cached raw logits shape=%s at %s",
            list(ordered_logits.shape),
            cache_dir,
        )
    accelerator.wait_for_everyone()
    return manifest


def _load_raw_logits_cache(
    output_dir: Path,
    *,
    input_manifest_sha256: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = output_dir / "raw_logits"
    manifest = _read_json(cache_dir / "manifest.json")
    if manifest.get("schema_version") != RAW_LOGITS_SCHEMA_VERSION or manifest.get("status") != "complete":
        raise MatrixValidationError(f"Raw-logit cache is incomplete: {cache_dir}")
    if str(manifest.get("input_manifest_sha256")) != input_manifest_sha256:
        raise MatrixValidationError("Raw-logit cache was not generated from the current prepared input")
    logits_path = cache_dir / str(manifest["raw_logits_file"])
    if _sha256_file(logits_path) != str(manifest["raw_logits_sha256"]):
        raise MatrixValidationError("raw_label_logits.npz SHA-256 mismatch")
    index_path = cache_dir / str(manifest["index_file"])
    if not index_path.is_file() or _sha256_file(index_path) != str(manifest["index_sha256"]):
        raise MatrixValidationError("raw_logits_index.jsonl is missing or has a SHA-256 mismatch")
    index_rows = _load_jsonl(index_path)
    with np.load(logits_path, allow_pickle=False) as arrays:
        logits = np.asarray(arrays["label_logits"], dtype=np.float32)
        gold = np.asarray(arrays["gold_ids"], dtype=np.int64)
        indices = np.asarray(arrays["unique_indices"], dtype=np.int64)
    expected = int(manifest["num_unique_prompts"])
    if logits.shape != (expected, int(manifest["num_labels"])):
        raise MatrixValidationError(
            f"Raw-logit matrix shape={logits.shape} disagrees with cache manifest"
        )
    if not np.isfinite(logits).all():
        raise MatrixValidationError("Raw-logit cache contains NaN or Inf")
    if not np.array_equal(indices, np.arange(expected, dtype=np.int64)):
        raise MatrixValidationError("Raw-logit unique_indices are not contiguous")
    if gold.shape != (expected,):
        raise MatrixValidationError("Raw-logit gold_ids shape mismatch")
    if len(index_rows) != expected or [int(row.get("unique_idx", -1)) for row in index_rows] != list(
        range(expected)
    ):
        raise MatrixValidationError("Raw-logit index is not complete and contiguous")
    return manifest, logits, gold, indices


def _native_batch1_losses(
    logits: np.ndarray,
    gold_ids: np.ndarray,
    *,
    ordinal_cfg: Mapping[str, Any],
) -> tuple[float, float, float]:
    logits_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    gold_tensor = torch.from_numpy(np.asarray(gold_ids, dtype=np.int64))
    ce = F.cross_entropy(logits_tensor, gold_tensor, reduction="none")
    ce_loss = float(ce.mean().item())
    ordinal_loss = 0.0
    if bool(ordinal_cfg.get("enabled", False)) and logits_tensor.shape[-1] > 1:
        ranks = torch.arange(logits_tensor.shape[-1], dtype=torch.float32)
        distances = (ranks.unsqueeze(0) - ranks[gold_tensor].unsqueeze(1)).abs()
        if bool(ordinal_cfg.get("normalize_distance", True)):
            distances = distances / float(logits_tensor.shape[-1] - 1)
        ordinal_per_sample = (torch.softmax(logits_tensor, dim=-1) * distances).sum(dim=-1)
        ordinal_loss = float(ordinal_per_sample.mean().item())
    # Native inference calls the evaluator at global_step=0. With warmup enabled,
    # effective ordinal alpha is zero; the frozen gate therefore has eval_loss=CE.
    effective_alpha = float(ordinal_cfg.get("alpha", 0.0))
    if bool(ordinal_cfg.get("enabled", False)) and float(
        ordinal_cfg.get("alpha_warmup_ratio", 0.0)
    ) > 0:
        effective_alpha = 0.0
    eval_loss = ce_loss + effective_alpha * ordinal_loss if bool(
        ordinal_cfg.get("enabled", False)
    ) else ce_loss
    return eval_loss, ce_loss, ordinal_loss


def build_cell_metrics_from_raw_logits(
    *,
    cell: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]],
    unique_rows: Sequence[Mapping[str, Any]],
    raw_logits: np.ndarray,
    raw_gold_ids: np.ndarray,
    raw_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    labels = [str(value) for value in raw_manifest["labels"]]
    letters = [str(value) for value in raw_manifest["letter_order"]]
    label_prefix = str(raw_manifest["label_token_meta"]["label_prefix"])
    ordered_mappings = sorted(mappings, key=lambda row: int(row["cell_sample_idx"]))
    expected_indices = list(range(len(ordered_mappings)))
    actual_indices = [int(row["cell_sample_idx"]) for row in ordered_mappings]
    if actual_indices != expected_indices:
        raise MatrixValidationError(
            f"{cell['cell_id']} mapping indices are not exactly 0..{len(mappings) - 1}"
        )
    unique_indices = np.asarray([int(row["unique_idx"]) for row in ordered_mappings], dtype=np.int64)
    if np.any(unique_indices < 0) or np.any(unique_indices >= len(unique_rows)):
        raise MatrixValidationError(f"{cell['cell_id']} mapping references an invalid unique_idx")
    logits = raw_logits[unique_indices]
    gold_ids = np.asarray([int(row["gold_id"]) for row in ordered_mappings], dtype=np.int64)
    if not np.array_equal(raw_gold_ids[unique_indices], gold_ids):
        raise MatrixValidationError(f"{cell['cell_id']} mapping gold IDs disagree with raw cache")
    pred_ids = np.argmax(logits, axis=1).astype(np.int64)
    records: list[dict[str, Any]] = []
    for mapping, unique_idx, pred_id, gold_id in zip(
        ordered_mappings,
        unique_indices.tolist(),
        pred_ids.tolist(),
        gold_ids.tolist(),
    ):
        unique_row = unique_rows[unique_idx]
        if str(mapping["prompt_cache_key"]) != str(unique_row["prompt_cache_key"]):
            raise MatrixValidationError(
                f"{cell['cell_id']} sample_idx={mapping['cell_sample_idx']} cache-key mismatch"
            )
        letter = letters[int(pred_id)]
        records.append(
            {
                "sample_idx": int(mapping["cell_sample_idx"]),
                "prompt": str(unique_row["prompt"]),
                "target": str(unique_row["target"]),
                "raw_output": f"{label_prefix}{_choice_text(label_prefix, letter)}",
                "pred_id": int(pred_id),
                "pred_label": labels[int(pred_id)],
                "gold_id": int(gold_id),
                "gold_label": str(mapping["gold_label"]),
                "gold_explain": str(unique_row.get("gold_explain", "")),
                "event_id": str(mapping["event_id"]),
                "cell_id": str(cell["cell_id"]),
                "selector_level": cell.get("selector_level"),
                "controller_level": cell.get("controller_level"),
                **_cell_metadata(cell),
                "raw_logits_unique_idx": int(unique_idx),
                "prompt_cache_key": str(mapping["prompt_cache_key"]),
                "prompt_input_ids_sha256": str(mapping["prompt_input_ids_sha256"]),
                "prompt_text_sha256": str(mapping["prompt_text_sha256"]),
                "evidence_count": int(mapping["evidence_count"]),
                "prompt_token_count": int(mapping["prompt_token_count"]),
                "scoring_fingerprint": str(raw_manifest["scoring_fingerprint"]),
            }
        )
    eval_metrics = build_eval_metrics(
        pred_ids,
        gold_ids,
        labels=labels,
        prediction_records=records,
        log_predictions_limit=0,
        log_prediction_examples=False,
    )
    loss_contract = raw_manifest.get("loss_replay_contract", {}) or {}
    native_batch_size = int(loss_contract.get("native_eval_batch_size", 1))
    if native_batch_size != 1:
        raise MatrixValidationError(
            "Exact native loss replay currently requires sft_train.per_device_eval_batch_size=1; "
            f"got {native_batch_size}"
        )
    eval_loss, ce_loss, ordinal_loss = _native_batch1_losses(
        logits,
        gold_ids,
        ordinal_cfg=loss_contract.get("ordinal_loss", {}) or {},
    )
    eval_metrics.update(
        {
            "eval_loss": eval_loss,
            "eval_ce_loss": ce_loss,
            "eval_ordinal_loss": ordinal_loss,
        }
    )
    metrics = build_serializable_metrics(eval_metrics)
    true_side = _true_side_macro_f1(eval_metrics)
    selection_score = _checkpoint_selection_score(eval_metrics, {"label_token_ce": {}})
    metrics.update(
        {
            "schema_version": MATRIX_RESULT_SCHEMA_VERSION,
            "label_schema": raw_manifest["label_schema"],
            "eval_backend": "deduplicated_raw_label_logits",
            "checkpoint": raw_manifest["checkpoint"]["checkpoint_name"],
            "split": raw_manifest["split"],
            "cell_id": cell["cell_id"],
            "selector_level": cell.get("selector_level"),
            "controller_level": cell.get("controller_level"),
            **_cell_metadata(cell),
            "true_side_macro_f1": true_side,
            "checkpoint_selection_score": selection_score,
            "logit_adjust": {"enabled": False},
            "raw_logits_scoring_fingerprint": raw_manifest["scoring_fingerprint"],
            "unique_prompt_count": int(len(set(unique_indices.tolist()))),
            "reused_reference_count": int(len(unique_indices) - len(set(unique_indices.tolist()))),
            "mean_evidence_count": float(
                np.mean([int(row["evidence_count"]) for row in ordered_mappings])
            ),
            "mean_prompt_token_count": float(
                np.mean([int(row["prompt_token_count"]) for row in ordered_mappings])
            ),
        }
    )
    return eval_metrics, metrics, records


def _flatten_numeric_metrics(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_numeric_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def compare_native_equivalence(
    *,
    candidate_records: Sequence[Mapping[str, Any]],
    candidate_metrics: Mapping[str, Any],
    reference_predictions_path: Path,
    reference_metrics_path: Path,
    reference_build_path: Path,
    expected_count: int,
    adapter_sha256: str | None,
    expected_adapter_sha256: str | None,
    reference_contract: Mapping[str, Any] | None = None,
    raw_manifest: Mapping[str, Any] | None = None,
    candidate_artifacts: Mapping[str, Any] | None = None,
    classification_atol: float = 1e-12,
    loss_atol: float = 1e-6,
) -> dict[str, Any]:
    reference_records = _load_jsonl(reference_predictions_path)
    reference_metrics = _read_json(reference_metrics_path)
    reference_build = _load_jsonl(reference_build_path)
    mismatches: list[dict[str, Any]] = []
    counts = {
        "expected": int(expected_count),
        "candidate": len(candidate_records),
        "reference_predictions": len(reference_records),
        "reference_build": len(reference_build),
    }
    if any(value != expected_count for key, value in counts.items() if key != "expected"):
        mismatches.append(
            {
                "type": "row_count",
                **counts,
            }
        )

    def _index_records(
        records: Sequence[Mapping[str, Any]], source: str
    ) -> dict[int, Mapping[str, Any]]:
        indexed: dict[int, Mapping[str, Any]] = {}
        for row in records:
            sample_idx = int(row.get("sample_idx", -1))
            if sample_idx in indexed:
                mismatches.append(
                    {"type": "duplicate_sample_idx", "source": source, "sample_idx": sample_idx}
                )
                continue
            indexed[sample_idx] = row
        expected_indices = set(range(expected_count))
        if set(indexed) != expected_indices:
            mismatches.append(
                {
                    "type": "sample_idx_domain",
                    "source": source,
                    "missing": sorted(expected_indices - set(indexed))[:20],
                    "extra": sorted(set(indexed) - expected_indices)[:20],
                }
            )
        return indexed

    candidate_by_idx = _index_records(candidate_records, "candidate")
    reference_by_idx = _index_records(reference_records, "reference")
    build_event_ids = [str(row.get("event_id") or "") for row in reference_build]
    if len(set(build_event_ids)) != len(build_event_ids) or any(not value for value in build_event_ids):
        mismatches.append({"type": "reference_build_event_domain"})

    if reference_contract is not None:
        if reference_contract.get("schema_version") != "native_label_token_reference_contract_v0_1":
            mismatches.append({"type": "reference_contract_schema"})
        contract_artifacts = reference_contract.get("artifacts", {}) or {}
        supplied_paths = {
            "predictions": reference_predictions_path,
            "metrics": reference_metrics_path,
            "build": reference_build_path,
        }
        for name, supplied_path in supplied_paths.items():
            artifact = contract_artifacts.get(name, {}) or {}
            declared_path_text = str(artifact.get("path") or "")
            declared_path = Path(declared_path_text) if declared_path_text else None
            declared_sha = str(artifact.get("sha256") or "")
            if declared_path is None or declared_path.resolve() != supplied_path.resolve():
                mismatches.append(
                    {
                        "type": "reference_contract_path",
                        "artifact": name,
                        "declared": declared_path_text,
                        "supplied": str(supplied_path),
                    }
                )
            actual_sha = _sha256_file(supplied_path)
            if not declared_sha or actual_sha != declared_sha:
                mismatches.append(
                    {
                        "type": "reference_contract_sha256",
                        "artifact": name,
                        "actual": actual_sha,
                        "declared": declared_sha,
                    }
                )
        contract_checkpoint = reference_contract.get("checkpoint", {}) or {}
        if adapter_sha256 != contract_checkpoint.get("adapter_sha256"):
            mismatches.append(
                {
                    "type": "reference_contract_adapter",
                    "candidate": adapter_sha256,
                    "declared": contract_checkpoint.get("adapter_sha256"),
                }
            )
        if raw_manifest is None:
            mismatches.append({"type": "missing_candidate_raw_manifest"})
        else:
            raw_checkpoint = raw_manifest.get("checkpoint", {}) or {}
            checkpoint_files = raw_checkpoint.get("checkpoint_files", {}) or {}
            base_files = raw_checkpoint.get("base_model_files", {}) or {}
            weight_files = raw_checkpoint.get("model_weight_files", {}) or {}
            tokenizer_files = raw_checkpoint.get("tokenizer_files", {}) or {}
            provenance_checks = {
                "adapter_config_sha256": (
                    (checkpoint_files.get("adapter_config.json") or {}).get("sha256"),
                    contract_checkpoint.get("adapter_config_sha256"),
                ),
                "base_model_path": (
                    str(Path(str(raw_checkpoint.get("base_model_dir") or "")).resolve()),
                    str(Path(str(contract_checkpoint.get("base_model_path") or "")).resolve()),
                ),
                "base_model_config_sha256": (
                    (base_files.get("config.json") or {}).get("sha256"),
                    contract_checkpoint.get("base_model_config_sha256"),
                ),
                "base_weight_index_sha256": (
                    (base_files.get("model.safetensors.index.json") or {}).get("sha256"),
                    contract_checkpoint.get("base_weight_index_sha256"),
                ),
                "run_dir": (
                    str(Path(str(raw_checkpoint.get("run_dir") or "")).resolve()),
                    str(Path(str(contract_checkpoint.get("run_dir") or "")).resolve()),
                ),
                "inference_config_sha256": (
                    raw_checkpoint.get("resolved_config_sha256"),
                    (contract_artifacts.get("inference_config") or {}).get("sha256"),
                ),
                "inference_config_path": (
                    str(Path(str(raw_checkpoint.get("resolved_config") or "")).resolve()),
                    str(
                        Path(
                            str((contract_artifacts.get("inference_config") or {}).get("path") or "")
                        ).resolve()
                    ),
                ),
            }
            for field, (actual, declared) in provenance_checks.items():
                if not declared or actual != declared:
                    mismatches.append(
                        {
                            "type": "reference_contract_provenance",
                            "field": field,
                            "actual": actual,
                            "declared": declared,
                        }
                    )
            for name, declared_sha in (
                contract_checkpoint.get("base_weight_shards", {}) or {}
            ).items():
                actual_sha = (weight_files.get(name) or {}).get("sha256")
                if actual_sha != declared_sha:
                    mismatches.append(
                        {
                            "type": "reference_contract_weight_shard",
                            "file": name,
                            "actual": actual_sha,
                            "declared": declared_sha,
                        }
                    )
            for name, declared_sha in (
                contract_checkpoint.get("tokenizer_files", {}) or {}
            ).items():
                actual_sha = (tokenizer_files.get(name) or {}).get("sha256")
                if actual_sha != declared_sha:
                    mismatches.append(
                        {
                            "type": "reference_contract_tokenizer",
                            "file": name,
                            "actual": actual_sha,
                            "declared": declared_sha,
                        }
                    )
            expected_contract = reference_contract.get("expected", {}) or {}
            execution = raw_manifest.get("execution_contract", {}) or {}
            raw_contract_checks = {
                "split": raw_manifest.get("split"),
                "label_schema": raw_manifest.get("label_schema"),
                "checkpoint": raw_checkpoint.get("checkpoint_name"),
                "python_executable": execution.get("python_executable"),
                "per_device_eval_batch_size": execution.get("per_device_eval_batch_size"),
                "mixed_precision": execution.get("mixed_precision"),
                "attention_backend": execution.get("attention_backend"),
                "world_size": execution.get("world_size"),
            }
            for field, actual in raw_contract_checks.items():
                declared = expected_contract.get(field)
                if declared is not None and actual != declared:
                    mismatches.append(
                        {
                            "type": "candidate_execution_contract",
                            "field": field,
                            "actual": actual,
                            "declared": declared,
                        }
                    )
            declared_packages = expected_contract.get("packages", {}) or {}
            actual_packages = execution.get("packages", {}) or {}
            for package, declared in declared_packages.items():
                actual = actual_packages.get(package)
                if actual != declared:
                    mismatches.append(
                        {
                            "type": "candidate_execution_contract",
                            "field": f"packages.{package}",
                            "actual": actual,
                            "declared": declared,
                        }
                    )

        expected_contract = reference_contract.get("expected", {}) or {}
        reference_contract_checks = {
            "num_samples": reference_metrics.get("num_samples"),
            "label_schema": reference_metrics.get("label_schema"),
            "eval_backend": reference_metrics.get("eval_backend"),
            "checkpoint": reference_metrics.get("checkpoint"),
            "split": reference_metrics.get("split"),
        }
        for field, actual in reference_contract_checks.items():
            declared = expected_contract.get(field)
            if declared is not None and actual != declared:
                mismatches.append(
                    {
                        "type": "reference_metrics_contract",
                        "field": field,
                        "actual": actual,
                        "declared": declared,
                    }
                )
        reference_adjust = reference_metrics.get("logit_adjust", {}) or {}
        declared_adjust = bool(expected_contract.get("logit_adjust_enabled", False))
        actual_reference_adjust = bool(reference_adjust.get("enabled", False))
        if declared_adjust or actual_reference_adjust != declared_adjust:
            mismatches.append(
                {
                    "type": "reference_logit_adjust_contract",
                    "actual": actual_reference_adjust,
                    "declared": declared_adjust,
                }
            )
        candidate_contract_checks = {
            "num_samples": candidate_metrics.get("num_samples"),
            "label_schema": candidate_metrics.get("label_schema"),
            "checkpoint": candidate_metrics.get("checkpoint"),
            "split": candidate_metrics.get("split"),
        }
        for field, actual in candidate_contract_checks.items():
            declared = expected_contract.get(field)
            if declared is not None and actual != declared:
                mismatches.append(
                    {
                        "type": "candidate_metrics_contract",
                        "field": field,
                        "actual": actual,
                        "declared": declared,
                    }
                )
        if candidate_metrics.get("eval_backend") != "deduplicated_raw_label_logits":
            mismatches.append({"type": "candidate_eval_backend"})
        actual_candidate_adjust = bool(
            (candidate_metrics.get("logit_adjust", {}) or {}).get("enabled", False)
        )
        if actual_candidate_adjust or actual_candidate_adjust != declared_adjust:
            mismatches.append(
                {
                    "type": "candidate_logit_adjust_contract",
                    "actual": actual_candidate_adjust,
                    "declared": declared_adjust,
                }
            )

    compare_count = min(expected_count, len(reference_build))
    exact_prediction_matches = 0
    exact_mapping_matches = 0
    for idx in range(compare_count):
        candidate = candidate_by_idx.get(idx)
        reference = reference_by_idx.get(idx)
        if candidate is None or reference is None:
            continue
        build_row = reference_build[idx]
        prediction_fields = ("sample_idx", "pred_id", "pred_label", "raw_output")
        prediction_diff = {
            field: {"candidate": candidate.get(field), "reference": reference.get(field)}
            for field in prediction_fields
            if candidate.get(field) != reference.get(field)
        }
        if prediction_diff:
            if len(mismatches) < 20:
                mismatches.append(
                    {"type": "prediction", "sample_idx": idx, "fields": prediction_diff}
                )
        else:
            exact_prediction_matches += 1

        build_prompt_ids = build_row.get("prompt_input_ids")
        build_ids_sha = (
            _prompt_ids_sha256(build_prompt_ids) if isinstance(build_prompt_ids, list) else None
        )
        mapping_checks = {
            "candidate_sample_idx": (candidate.get("sample_idx"), idx),
            "reference_sample_idx": (reference.get("sample_idx"), idx),
            "candidate_event_id": (candidate.get("event_id"), build_row.get("event_id")),
            "candidate_gold_id": (candidate.get("gold_id"), build_row.get("gold_id")),
            "reference_gold_id": (reference.get("gold_id"), build_row.get("gold_id")),
            "candidate_gold_label": (candidate.get("gold_label"), build_row.get("gold_label")),
            "reference_gold_label": (reference.get("gold_label"), build_row.get("gold_label")),
            "candidate_prompt": (candidate.get("prompt"), build_row.get("prompt")),
            "reference_prompt": (reference.get("prompt"), build_row.get("prompt")),
            "candidate_target": (candidate.get("target"), build_row.get("target")),
            "reference_target": (reference.get("target"), build_row.get("target")),
            "prompt_input_ids_sha256": (
                candidate.get("prompt_input_ids_sha256"),
                build_ids_sha,
            ),
        }
        mapping_diff = {
            field: {
                "candidate_sha256": hashlib.sha256(str(left).encode()).hexdigest()
                if "prompt" in field or "target" in field
                else left,
                "reference_sha256": hashlib.sha256(str(right).encode()).hexdigest()
                if "prompt" in field or "target" in field
                else right,
            }
            for field, (left, right) in mapping_checks.items()
            if left != right
        }
        if mapping_diff:
            if len(mismatches) < 20:
                mismatches.append(
                    {"type": "mapping", "sample_idx": idx, "fields": mapping_diff}
                )
        else:
            exact_mapping_matches += 1

    metric_keys = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "parse_error_rate",
        "true_side_macro_f1",
        "checkpoint_selection_score",
    )
    metric_diffs: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        if key not in candidate_metrics or key not in reference_metrics:
            metric_diffs[key] = {
                "candidate": float(candidate_metrics.get(key, float("nan"))),
                "reference": float(reference_metrics.get(key, float("nan"))),
            }
            continue
        left = float(candidate_metrics[key])
        right = float(reference_metrics[key])
        if not np.isclose(left, right, atol=classification_atol, rtol=0.0):
            metric_diffs[key] = {"candidate": left, "reference": right}
    candidate_flat = _flatten_numeric_metrics(candidate_metrics.get("per_class", {}) or {})
    reference_flat = _flatten_numeric_metrics(reference_metrics.get("per_class", {}) or {})
    for key in sorted(set(candidate_flat) | set(reference_flat)):
        left = candidate_flat.get(key, float("nan"))
        right = reference_flat.get(key, float("nan"))
        if not np.isclose(left, right, atol=classification_atol, rtol=0.0):
            metric_diffs[f"per_class.{key}"] = {"candidate": left, "reference": right}
    loss_diffs: dict[str, dict[str, float]] = {}
    for key in ("eval_loss", "eval_ce_loss", "eval_ordinal_loss"):
        left = float(candidate_metrics.get(key, float("nan")))
        right = float(reference_metrics.get(key, float("nan")))
        if not np.isclose(left, right, atol=loss_atol, rtol=0.0):
            loss_diffs[key] = {"candidate": left, "reference": right}
    if metric_diffs:
        mismatches.append({"type": "classification_metrics", "fields": metric_diffs})
    if loss_diffs:
        mismatches.append({"type": "loss_metrics", "fields": loss_diffs})
    if expected_adapter_sha256 and adapter_sha256 != expected_adapter_sha256:
        mismatches.append(
            {
                "type": "adapter_sha256",
                "candidate": adapter_sha256,
                "expected": expected_adapter_sha256,
            }
        )

    candidate_confusion = np.asarray(
        build_eval_metrics(
            np.asarray([int(row["pred_id"]) for row in candidate_records]),
            np.asarray([int(row["gold_id"]) for row in candidate_records]),
            labels=labels_for_schema(str(candidate_metrics.get("label_schema") or "liar6")),
            log_prediction_examples=False,
        )["confusion_matrix"]
    )
    reference_confusion = np.asarray(
        build_eval_metrics(
            np.asarray([int(row["pred_id"]) for row in reference_records]),
            np.asarray([int(row["gold_id"]) for row in reference_records]),
            labels=labels_for_schema(str(reference_metrics.get("label_schema") or "liar6")),
            log_prediction_examples=False,
        )["confusion_matrix"]
    )
    confusion_equal = np.array_equal(candidate_confusion, reference_confusion)
    if not confusion_equal:
        mismatches.append(
            {
                "type": "confusion_matrix",
                "candidate": candidate_confusion.tolist(),
                "reference": reference_confusion.tolist(),
            }
        )
    passed = not mismatches
    return {
        "schema_version": "native_label_token_equivalence_gate_v0_2",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "created_at": _utc_now(),
        "candidate_count": len(candidate_records),
        "reference_count": len(reference_records),
        "expected_count": int(expected_count),
        "mapping_exact_match_count": exact_mapping_matches,
        "prediction_exact_match_count": exact_prediction_matches,
        "confusion_matrix_exact_match": confusion_equal,
        "classification_atol": classification_atol,
        "loss_atol": loss_atol,
        "candidate_adapter_sha256": adapter_sha256,
        "expected_adapter_sha256": expected_adapter_sha256,
        "reference_contract": dict(reference_contract) if reference_contract is not None else None,
        "candidate_artifacts": dict(candidate_artifacts or {}),
        "reference_predictions": str(reference_predictions_path),
        "reference_predictions_sha256": _sha256_file(reference_predictions_path),
        "reference_metrics": str(reference_metrics_path),
        "reference_metrics_sha256": _sha256_file(reference_metrics_path),
        "reference_build": str(reference_build_path),
        "reference_build_sha256": _sha256_file(reference_build_path),
        "mismatches": mismatches,
    }


def _factorial_summary_rows(
    cell_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scalar_keys = (
        "num_samples",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "parse_error_rate",
        "eval_loss",
        "eval_ce_loss",
        "eval_ordinal_loss",
        "true_side_macro_f1",
        "checkpoint_selection_score",
        "unique_prompt_count",
        "reused_reference_count",
        "mean_evidence_count",
        "mean_prompt_token_count",
    )
    return [
        {
            "cell_id": metrics["cell_id"],
            "selector_level": metrics.get("selector_level"),
            "controller_level": metrics.get("controller_level"),
            **_cell_metadata(metrics),
            **{key: metrics.get(key) for key in scalar_keys},
            "metrics_file": f"cells/{metrics['cell_id']}/label_token/metrics.json",
            "metrics_sha256": (metrics.get("_artifact_hashes", {}) or {}).get("metrics_sha256"),
            "confusion_matrix_file": f"cells/{metrics['cell_id']}/label_token/confusion_matrix.json",
            "confusion_matrix_sha256": (metrics.get("_artifact_hashes", {}) or {}).get(
                "confusion_matrix_sha256"
            ),
            "predictions_file": (
                f"cells/{metrics['cell_id']}/label_token/{metrics.get('_predictions_filename')}"
            ),
            "predictions_sha256": (metrics.get("_artifact_hashes", {}) or {}).get(
                "predictions_sha256"
            ),
        }
        for metrics in cell_metrics
    ]


def _write_factorial_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fieldnames = list(rows[0].keys()) if rows else ["cell_id"]
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def fanout_matrix(
    *,
    output_dir: Path,
    force: bool,
    equivalence_gate_cell: str | None,
    equivalence_gate_predictions: Path | None,
    equivalence_gate_metrics: Path | None,
    equivalence_gate_build: Path | None,
    equivalence_gate_expected_adapter_sha256: str | None,
    equivalence_gate_reference_contract: Path | None,
    unsafe_skip_equivalence_gate: bool,
    classification_atol: float,
    loss_atol: float,
) -> dict[str, Any]:
    input_manifest, input_manifest_path, unique_rows = _load_prepared_input(output_dir)
    input_manifest_sha = _sha256_file(input_manifest_path)
    raw_manifest, raw_logits, raw_gold_ids, _ = _load_raw_logits_cache(
        output_dir,
        input_manifest_sha256=input_manifest_sha,
    )
    reference_contract: dict[str, Any] | None = None
    if equivalence_gate_reference_contract is not None:
        reference_contract = _read_json(equivalence_gate_reference_contract)
        contract_artifacts = reference_contract.get("artifacts", {}) or {}
        required_contract_paths = {
            name: str((contract_artifacts.get(name) or {}).get("path") or "")
            for name in ("predictions", "metrics", "build")
        }
        missing_contract_paths = [
            name for name, value in required_contract_paths.items() if not value
        ]
        if missing_contract_paths:
            raise MatrixValidationError(
                f"Native reference contract is missing artifact paths: {missing_contract_paths}"
            )
        equivalence_gate_cell = equivalence_gate_cell or str(reference_contract.get("cell_id") or "")
        equivalence_gate_predictions = equivalence_gate_predictions or Path(
            required_contract_paths["predictions"]
        )
        equivalence_gate_metrics = equivalence_gate_metrics or Path(
            required_contract_paths["metrics"]
        )
        equivalence_gate_build = equivalence_gate_build or Path(
            required_contract_paths["build"]
        )
        equivalence_gate_expected_adapter_sha256 = (
            equivalence_gate_expected_adapter_sha256
            or str((reference_contract.get("checkpoint", {}) or {}).get("adapter_sha256") or "")
        )
    gate_values = (
        equivalence_gate_cell,
        equivalence_gate_predictions,
        equivalence_gate_metrics,
        equivalence_gate_build,
        reference_contract,
    )
    gate_requested = all(value is not None and value != "" for value in gate_values)
    if any(value is not None and value != "" for value in gate_values) and not gate_requested:
        raise ValueError(
            "Formal native gate requires a cell, predictions, metrics, build, and "
            "--equivalence-gate-reference-contract together"
        )
    if not gate_requested and not unsafe_skip_equivalence_gate:
        raise ValueError(
            "Formal fan-out requires a native reference contract; pass "
            "--unsafe-skip-equivalence-gate only for diagnostic artifacts"
        )

    result_dir = output_dir / "materialized"
    if result_dir.exists() and not force:
        existing = _read_json(result_dir / "matrix_manifest.json")
        if (
            existing.get("schema_version") != MATRIX_RESULT_SCHEMA_VERSION
            or existing.get("status") != "complete"
            or existing.get("input_manifest_sha256") != input_manifest_sha
            or existing.get("raw_logits_scoring_fingerprint")
            != raw_manifest.get("scoring_fingerprint")
            or existing.get("raw_logits_execution_fingerprint")
            != raw_manifest.get("execution_fingerprint")
            or existing.get("raw_logits_sha256") != raw_manifest.get("raw_logits_sha256")
            or existing.get("raw_logits_manifest_sha256")
            != _sha256_file(output_dir / "raw_logits" / "manifest.json")
            or bool(existing.get("diagnostic_only", False)) != bool(not gate_requested)
        ):
            raise MatrixValidationError(
                f"Existing materialized matrix is incompatible: {result_dir}; "
                "pass --force-fanout to replace"
            )
        for cell in existing.get("cells", []):
            for path_key, sha_key in (
                ("metrics_file", "metrics_sha256"),
                ("confusion_matrix_file", "confusion_matrix_sha256"),
                ("predictions_file", "predictions_sha256"),
            ):
                artifact_path = result_dir / str(cell[path_key])
                if _sha256_file(artifact_path) != str(cell[sha_key]):
                    raise MatrixValidationError(
                        f"Existing materialized artifact has drifted: {artifact_path}"
                    )
        for path_key, sha_key in (
            ("factorial_metrics_jsonl", "factorial_metrics_jsonl_sha256"),
            ("factorial_metrics_csv", "factorial_metrics_csv_sha256"),
            ("equivalence_gate", "equivalence_gate_sha256"),
        ):
            relative_path = existing.get(path_key)
            expected_sha = existing.get(sha_key)
            if relative_path is None and expected_sha is None:
                continue
            artifact_path = result_dir / str(relative_path)
            if not artifact_path.is_file() or _sha256_file(artifact_path) != str(expected_sha):
                raise MatrixValidationError(
                    f"Existing materialized summary has drifted: {artifact_path}"
                )
        if gate_requested:
            existing_gate = _read_json(result_dir / str(existing["equivalence_gate"]))
            current_contract_sha = _sha256_file(Path(equivalence_gate_reference_contract))
            current_reference_contract = {
                "cell_id": str(equivalence_gate_cell),
                "classification_atol": float(classification_atol),
                "loss_atol": float(loss_atol),
                "expected_adapter_sha256": equivalence_gate_expected_adapter_sha256,
                "reference_contract_sha256": current_contract_sha,
                "reference_predictions": str(Path(equivalence_gate_predictions)),
                "reference_predictions_sha256": _sha256_file(Path(equivalence_gate_predictions)),
                "reference_metrics": str(Path(equivalence_gate_metrics)),
                "reference_metrics_sha256": _sha256_file(Path(equivalence_gate_metrics)),
                "reference_build": str(Path(equivalence_gate_build)),
                "reference_build_sha256": _sha256_file(Path(equivalence_gate_build)),
            }
            gate_reuse_matches = bool(existing_gate.get("passed", False))
            for key, requested_value in current_reference_contract.items():
                existing_value = existing_gate.get(key)
                if key in {"reference_predictions", "reference_metrics", "reference_build"}:
                    if Path(str(existing_value)).resolve() != Path(str(requested_value)).resolve():
                        gate_reuse_matches = False
                elif existing_value != requested_value:
                    gate_reuse_matches = False
            if not gate_reuse_matches:
                raise MatrixValidationError(
                    "Existing materialized gate does not match the requested gate semantics; "
                    "pass --force-fanout to rerun"
                )
        logger.info("[matrix-fanout] compatible materialized result already exists at %s", result_dir)
        return existing

    staging = output_dir / f".materialized.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    all_metrics: list[dict[str, Any]] = []
    gate_candidate: tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None = None
    gate_report: dict[str, Any] | None = None
    try:
        for cell in input_manifest["cells"]:
            cell_id = str(cell["cell_id"])
            mapping_path = output_dir / "input" / str(cell["mapping_file"])
            if _sha256_file(mapping_path) != str(cell["mapping_sha256"]):
                raise MatrixValidationError(f"Mapping SHA mismatch for {cell_id}")
            mappings = _load_jsonl(mapping_path)
            eval_metrics, metrics, records = build_cell_metrics_from_raw_logits(
                cell=cell,
                mappings=mappings,
                unique_rows=unique_rows,
                raw_logits=raw_logits,
                raw_gold_ids=raw_gold_ids,
                raw_manifest=raw_manifest,
            )
            cell_eval_dir = staging / "cells" / cell_id / "label_token"
            predictions_filename = f"{raw_manifest['split']}_predictions.jsonl"
            artifacts = save_eval_artifacts(
                eval_dir=cell_eval_dir,
                metrics=metrics,
                confusion_matrix=eval_metrics["confusion_matrix"],
                confusion_labels=eval_metrics["confusion_labels"],
                prediction_records=records,
                predictions_filename=predictions_filename,
                title=f"Deduplicated Label-Token Matrix ({cell_id})",
                labels=[str(value) for value in raw_manifest["labels"]],
            )
            artifact_hashes = {
                "metrics_sha256": _sha256_file(Path(artifacts["metrics_path"])),
                "confusion_matrix_sha256": _sha256_file(Path(artifacts["confusion_data_path"])),
                "predictions_sha256": _sha256_file(Path(artifacts["predictions_path"])),
            }
            metrics["_artifact_hashes"] = artifact_hashes
            metrics["_predictions_filename"] = predictions_filename
            all_metrics.append(metrics)
            if cell_id == equivalence_gate_cell:
                gate_candidate = (records, metrics, artifact_hashes)

        if gate_requested:
            if gate_candidate is None:
                raise MatrixValidationError(
                    f"Equivalence gate cell is absent from matrix: {equivalence_gate_cell}"
                )
            gate_report = compare_native_equivalence(
                candidate_records=gate_candidate[0],
                candidate_metrics=gate_candidate[1],
                reference_predictions_path=Path(equivalence_gate_predictions),
                reference_metrics_path=Path(equivalence_gate_metrics),
                reference_build_path=Path(equivalence_gate_build),
                expected_count=int(
                    next(
                        cell["row_count"]
                        for cell in input_manifest["cells"]
                        if str(cell["cell_id"]) == str(equivalence_gate_cell)
                    )
                ),
                adapter_sha256=raw_manifest["checkpoint"].get("adapter_sha256"),
                expected_adapter_sha256=equivalence_gate_expected_adapter_sha256,
                reference_contract=reference_contract,
                raw_manifest=raw_manifest,
                candidate_artifacts=gate_candidate[2],
                classification_atol=classification_atol,
                loss_atol=loss_atol,
            )
            gate_report["reference_contract_path"] = str(equivalence_gate_reference_contract)
            gate_report["reference_contract_sha256"] = _sha256_file(
                Path(equivalence_gate_reference_contract)
            )
            gate_report["cell_id"] = str(equivalence_gate_cell)
            if not bool(gate_report["passed"]):
                failure_dir = output_dir / "fanout_failures"
                failure_name = f"equivalence_gate_failed_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}.json"
                failure_path = failure_dir / failure_name
                _write_json(failure_path, gate_report)
                mismatch_types = sorted(
                    {str(item.get("type") or "unknown") for item in gate_report["mismatches"]}
                )
                raise EquivalenceGateError(
                    "Native equivalence gate failed: "
                    f"cell={equivalence_gate_cell}, "
                    f"mapping={gate_report['mapping_exact_match_count']}/"
                    f"{gate_report['expected_count']}, "
                    f"predictions={gate_report['prediction_exact_match_count']}/"
                    f"{gate_report['expected_count']}, "
                    f"candidate_adapter={gate_report['candidate_adapter_sha256']}, "
                    f"expected_adapter={gate_report['expected_adapter_sha256']}, "
                    f"mismatch_types={','.join(mismatch_types)}; see {failure_path}"
                )
            _write_json(staging / "equivalence_gate.json", gate_report)

        summary_rows = _factorial_summary_rows(all_metrics)
        _write_jsonl(staging / "factorial_metrics.jsonl", summary_rows)
        _write_factorial_csv(staging / "factorial_metrics.csv", summary_rows)
        result_manifest = {
            "schema_version": MATRIX_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "created_at": _utc_now(),
            "diagnostic_only": not gate_requested,
            "split": raw_manifest["split"],
            "cell_count": len(summary_rows),
            "input_manifest": str(input_manifest_path),
            "input_manifest_sha256": input_manifest_sha,
            "raw_logits_manifest": str(output_dir / "raw_logits" / "manifest.json"),
            "raw_logits_manifest_sha256": _sha256_file(
                output_dir / "raw_logits" / "manifest.json"
            ),
            "raw_logits_scoring_fingerprint": raw_manifest["scoring_fingerprint"],
            "raw_logits_execution_fingerprint": raw_manifest["execution_fingerprint"],
            "raw_logits_sha256": raw_manifest["raw_logits_sha256"],
            "checkpoint": raw_manifest["checkpoint"],
            "logit_adjust": {"enabled": False},
            "factorial_metrics_jsonl": "factorial_metrics.jsonl",
            "factorial_metrics_jsonl_sha256": _sha256_file(staging / "factorial_metrics.jsonl"),
            "factorial_metrics_csv": "factorial_metrics.csv",
            "factorial_metrics_csv_sha256": _sha256_file(staging / "factorial_metrics.csv"),
            "equivalence_gate": "equivalence_gate.json" if gate_requested else None,
            "equivalence_gate_sha256": _sha256_file(staging / "equivalence_gate.json")
            if gate_requested
            else None,
            "cells": summary_rows,
        }
        _write_json(staging / "matrix_manifest.json", result_manifest)
        _promote_directory(staging, result_dir, force=force)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    logger.info(
        "[matrix-fanout] materialized %s cells at %s (gate=%s)",
        len(result_manifest["cells"]),
        result_dir,
        "passed" if gate_requested else "unsafe-skipped",
    )
    return result_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate frozen label-token matrix prompts, run one raw-logit forward cache, "
            "and fan the cache back out to native-compatible cell artifacts."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze unique prompts and per-cell mappings")
    prepare.add_argument("--matrix-manifest", required=True)
    prepare.add_argument("--build-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--split", default="val", choices=["train", "val", "test"])
    prepare.add_argument("--label-prefix", default="Label:")
    prepare.add_argument("--force-prepare", action="store_true")

    infer = subparsers.add_parser("infer", help="Run one model load over unique raw label logits")
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--checkpoint", default="best")
    infer.add_argument("--split", default="val", choices=["train", "val", "test"])
    infer.add_argument("--config", default=None)
    infer.add_argument("--per-device-eval-batch-size", type=int, default=None)
    infer.add_argument("--dataloader-num-workers", type=int, default=None)
    infer.add_argument("--expected-world-size", type=int, default=None)
    infer.add_argument("--expected-adapter-sha256", default=None)
    infer.add_argument(
        "--unsafe-unpinned-checkpoint",
        action="store_true",
        help="Allow a PEFT cache without an expected adapter SHA (diagnostics only).",
    )
    infer.add_argument("--duplicate-logit-atol", type=float, default=1e-5)
    infer.add_argument("--duplicate-logit-rtol", type=float, default=1e-5)
    infer.add_argument("--force-infer", action="store_true")

    fanout = subparsers.add_parser("fanout", help="Materialize native-compatible per-cell artifacts")
    fanout.add_argument("--output-dir", required=True)
    fanout.add_argument("--force-fanout", action="store_true")
    fanout.add_argument("--equivalence-gate-cell", default=None)
    fanout.add_argument("--equivalence-gate-predictions", default=None)
    fanout.add_argument("--equivalence-gate-metrics", default=None)
    fanout.add_argument("--equivalence-gate-build", default=None)
    fanout.add_argument("--equivalence-gate-expected-adapter-sha256", default=None)
    fanout.add_argument("--equivalence-gate-reference-contract", default=None)
    fanout.add_argument(
        "--unsafe-skip-equivalence-gate",
        action="store_true",
        help="Materialize diagnostic_only=true results without the frozen native gate.",
    )
    fanout.add_argument("--classification-atol", type=float, default=1e-12)
    fanout.add_argument("--loss-atol", type=float, default=1e-6)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.mode == "prepare":
        prepare_matrix(
            matrix_manifest_path=Path(args.matrix_manifest),
            build_root=Path(args.build_root),
            output_dir=Path(args.output_dir),
            split=str(args.split),
            label_prefix=str(args.label_prefix),
            force=bool(args.force_prepare),
        )
        return
    if args.mode == "infer":
        infer_raw_logits(
            output_dir=Path(args.output_dir),
            run_dir=Path(args.run_dir),
            checkpoint=str(args.checkpoint),
            split=str(args.split),
            config_path=args.config,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            expected_adapter_sha256=args.expected_adapter_sha256,
            unsafe_unpinned_checkpoint=bool(args.unsafe_unpinned_checkpoint),
            expected_world_size=args.expected_world_size,
            force=bool(args.force_infer),
            duplicate_logit_atol=float(args.duplicate_logit_atol),
            duplicate_logit_rtol=float(args.duplicate_logit_rtol),
        )
        return
    if args.mode == "fanout":
        fanout_matrix(
            output_dir=Path(args.output_dir),
            force=bool(args.force_fanout),
            equivalence_gate_cell=args.equivalence_gate_cell,
            equivalence_gate_predictions=Path(args.equivalence_gate_predictions)
            if args.equivalence_gate_predictions
            else None,
            equivalence_gate_metrics=Path(args.equivalence_gate_metrics)
            if args.equivalence_gate_metrics
            else None,
            equivalence_gate_build=Path(args.equivalence_gate_build)
            if args.equivalence_gate_build
            else None,
            equivalence_gate_expected_adapter_sha256=args.equivalence_gate_expected_adapter_sha256,
            equivalence_gate_reference_contract=Path(args.equivalence_gate_reference_contract)
            if args.equivalence_gate_reference_contract
            else None,
            unsafe_skip_equivalence_gate=bool(args.unsafe_skip_equivalence_gate),
            classification_atol=float(args.classification_atol),
            loss_atol=float(args.loss_atol),
        )
        return
    raise AssertionError(f"Unhandled mode={args.mode}")


if __name__ == "__main__":
    main()
