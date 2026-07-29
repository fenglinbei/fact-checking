#!/usr/bin/env python3
"""Analyze the frozen 2 x 2 x 3 cross-verifier fine-tuning experiment.

The analyzer is intentionally independent from training and inference.  It
joins gold labels only after all twelve result files are complete, validates
the balanced backbone/assignment/seed grid, and keeps the claim as the
inferential unit.

The public entry point is :func:`analyze_results`; the small command-line
wrapper at the bottom exists only so the same implementation can also be run
directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BOOTSTRAP = 10_000
DEFAULT_PERMUTATIONS = 100_000
DEFAULT_SEED = 20260724
DEFAULT_TAU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SESOI = 0.01
FORMAL_SEEDS = (20260724, 20260725, 20260726)
EXPECTED_MAIN_CLAIMS = 1_250
EXPECTED_ORDER_CLAIMS = 1_152
EXPECTED_PREFIX_CLAIMS = 1_152
EXPECTED_PREFIX_POSITIONS = 6_996
EXPECTED_STRICT_PREFIX_POSITIONS = 2_020
EXPECTED_LOGICAL_COUNTS_PER_RUN = {
    "main": 2_500,
    "order_only": 2_304,
    "prefix": 13_992,
    "val_paired": 2_548,
    "val_claim_only": 1_274,
    "val_mismatched": 1_274,
}
EXPECTED_LOGICAL_RESULTS_PER_RUN = sum(
    EXPECTED_LOGICAL_COUNTS_PER_RUN.values()
)

BACKBONES = ("qwen3", "llama31")
ASSIGNMENTS = ("a", "b")
TEST_COMPARISON_TYPES = ("main", "order_only", "prefix")
ALLOWED_COMPARISON_TYPES = frozenset(
    TEST_COMPARISON_TYPES
    + ("val_paired", "val_claim_only", "val_mismatched")
)
TEST_ARMS = ("evitrace", "s4")
ALLOWED_ARMS = frozenset(TEST_ARMS + ("claim_only", "mismatched"))
PREFIX_RELATIONS = frozenset(
    ("same_order", "different_set", "same_set_different_order")
)

LIAR6_LABELS = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
LETTERS = ("A", "B", "C", "D", "E", "F")
LETTER_TO_LABEL = dict(zip(LETTERS, LIAR6_LABELS))
LABEL_TO_LETTER = {label: letter for letter, label in LETTER_TO_LABEL.items()}
LABEL_TO_ID = {label: index for index, label in enumerate(LIAR6_LABELS)}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GOLD_LEAK_KEYS = frozenset(
    ("gold", "gold_label", "target", "target_label", "ground_truth", "answer")
)


class FineTuneAnalysisError(RuntimeError):
    """Raised when a frozen analysis contract is incomplete or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FineTuneAnalysisError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FineTuneAnalysisError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise FineTuneAnalysisError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _resolve_path(value: str | Path, *, relative_to: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _file_metadata(
    path: Path,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _metadata_entry_path(
    entry: Any,
    *,
    relative_to: Path,
    name: str,
) -> Path:
    if isinstance(entry, (str, Path)):
        raw_path = entry
    elif isinstance(entry, Mapping):
        raw_path = entry.get("path")
    else:
        raw_path = None
    if not raw_path:
        raise FineTuneAnalysisError(f"Missing path for prepared file {name!r}")
    return _resolve_path(raw_path, relative_to=relative_to)


def _verify_file_entry(
    name: str,
    entry: Any,
    *,
    relative_to: Path,
) -> Path:
    path = _metadata_entry_path(entry, relative_to=relative_to, name=name)
    if not path.is_file():
        raise FineTuneAnalysisError(f"Prepared file does not exist: {path}")
    if isinstance(entry, Mapping):
        expected_sha = entry.get("sha256")
        if expected_sha and sha256_file(path) != str(expected_sha):
            raise FineTuneAnalysisError(
                f"{name}: prepared-file SHA-256 mismatch"
            )
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(
            expected_bytes
        ):
            raise FineTuneAnalysisError(f"{name}: prepared-file byte mismatch")
        expected_rows = entry.get("rows")
        if expected_rows is not None:
            with path.open("rb") as handle:
                observed_rows = sum(1 for line in handle if line.strip())
            if observed_rows != int(expected_rows):
                raise FineTuneAnalysisError(
                    f"{name}: prepared-file row-count mismatch "
                    f"({observed_rows} != {expected_rows})"
                )
    return path


def _prepared_file_entries(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    entries = manifest.get("prepared_files")
    if entries is None:
        entries = manifest.get("files")
    if not isinstance(entries, Mapping):
        raise FineTuneAnalysisError(
            "Prepared manifest has no prepared_files/files mapping"
        )
    return entries


def _find_prepared_entry(
    entries: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[str, Any]:
    for name in names:
        if name in entries:
            return name, entries[name]
    raise FineTuneAnalysisError(
        "Prepared manifest is missing one of: " + ", ".join(names)
    )


def _normalize_label(value: Any, *, context: str) -> str:
    text = str(value)
    if text in LETTER_TO_LABEL:
        return LETTER_TO_LABEL[text]
    if text in LABEL_TO_ID:
        return text
    raise FineTuneAnalysisError(f"{context}: invalid LIAR6 label {value!r}")


def _gold_rows_by_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
    require_unique: bool,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        if "gold_label" not in row:
            continue
        event_id = str(row.get("event_id") or "")
        if not event_id:
            raise FineTuneAnalysisError(
                f"{context}:{row_number}: gold row has no event_id"
            )
        gold_label = _normalize_label(
            row["gold_label"],
            context=f"{context}:{event_id}",
        )
        previous = output.get(event_id)
        if previous is not None:
            if require_unique:
                raise FineTuneAnalysisError(
                    f"{context}: duplicate gold event_id {event_id!r}"
                )
            if str(previous["gold_label"]) != gold_label:
                raise FineTuneAnalysisError(
                    f"{context}:{event_id}: inconsistent repeated gold label"
                )
            continue
        output[event_id] = {**row, "gold_label": gold_label}
    return output


def _walk_hashed_file_entries(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        return
    if "path" in value and "sha256" in value:
        yield prefix or "file", value
        return
    for name, child in value.items():
        child_prefix = f"{prefix}.{name}" if prefix else str(name)
        yield from _walk_hashed_file_entries(child, prefix=child_prefix)


def _model_prepared_files(
    manifest: Mapping[str, Any],
    backbone: str,
) -> Mapping[str, Any]:
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise FineTuneAnalysisError("Prepared manifest has no models mapping")
    model = models.get(backbone)
    if not isinstance(model, Mapping):
        raise FineTuneAnalysisError(
            f"Prepared manifest has no model entry for {backbone}"
        )
    files = model.get("files")
    if not isinstance(files, Mapping):
        raise FineTuneAnalysisError(
            f"Prepared manifest has no files mapping for {backbone}"
        )
    return files


def _assignment_training_summary(
    path: Path,
    *,
    assignment: str,
) -> tuple[dict[str, int], set[str]]:
    rows = load_jsonl(path)
    gold = _gold_rows_by_event(
        rows,
        context=str(path),
        require_unique=False,
    )
    if not gold:
        raise FineTuneAnalysisError(
            f"{path}: train assignment {assignment} exposes no analysis-stage "
            "gold labels from which to recompute priors"
        )
    counts = {
        letter: sum(
            str(row["gold_label"]) == LETTER_TO_LABEL[letter]
            for row in gold.values()
        )
        for letter in LETTERS
    }
    if any(value <= 0 for value in counts.values()):
        raise FineTuneAnalysisError(
            f"{path}: every LIAR6 class must occur in the training prior"
        )
    snippet_hashes: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        values = row.get("evidence_snippet_sha256s")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
        ):
            raise FineTuneAnalysisError(
                f"{path}:{row_number}: missing evidence_snippet_sha256s"
            )
        for value in values:
            digest = str(value)
            if not _SHA256.fullmatch(digest):
                raise FineTuneAnalysisError(
                    f"{path}:{row_number}: invalid evidence snippet SHA"
                )
            snippet_hashes.add(digest)
    if not snippet_hashes:
        raise FineTuneAnalysisError(
            f"{path}: training assignment selected-snippet set is empty"
        )
    return counts, snippet_hashes


def _load_val_gold(
    entries: Mapping[str, Any],
    verified_paths: Mapping[str, Path],
) -> tuple[dict[str, dict[str, Any]], str]:
    for name in ("gold_val", "val_gold", "independent_gold_val"):
        if name in entries:
            path = verified_paths[name]
            gold = _gold_rows_by_event(
                load_jsonl(path),
                context=str(path),
                require_unique=True,
            )
            if not gold:
                raise FineTuneAnalysisError(f"{path}: val gold file is empty")
            return gold, name
    candidates = sorted(
        str(name)
        for name in entries
        if "val_paired" in str(name).lower()
    )
    for name in candidates:
        path = verified_paths[name]
        rows = load_jsonl(path)
        gold = _gold_rows_by_event(
            rows,
            context=str(path),
            require_unique=False,
        )
        if gold:
            return gold, name
    raise FineTuneAnalysisError(
        "Prepared files contain neither gold_val/val_gold nor a "
        "gold-bearing val_paired registry"
    )


def _load_and_verify_prepared_manifest(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, float],
    dict[str, Any],
]:
    manifest = load_json(manifest_path)
    if manifest.get("complete") is not True:
        raise FineTuneAnalysisError(
            f"Prepared manifest is not complete: {manifest_path}"
        )
    entries = _prepared_file_entries(manifest)
    verified_paths: dict[str, Path] = {}
    for name, entry in entries.items():
        verified_paths[str(name)] = _verify_file_entry(
            str(name),
            entry,
            relative_to=manifest_path.parent,
        )
    all_verified_files: dict[str, Path] = {
        f"prepared_files.{name}": path
        for name, path in verified_paths.items()
    }
    for name, entry in _walk_hashed_file_entries(
        manifest.get("models", {}),
        prefix="models",
    ):
        all_verified_files[name] = _verify_file_entry(
            name,
            entry,
            relative_to=manifest_path.parent,
        )

    gold_name, _gold_entry = _find_prepared_entry(
        entries,
        ("gold_test", "test_gold", "independent_gold_test"),
    )
    gold_path = verified_paths[gold_name]
    gold_rows = load_jsonl(gold_path)
    if not gold_rows:
        raise FineTuneAnalysisError("Independent test-gold file is empty")
    gold_by_event = _gold_rows_by_event(
        gold_rows,
        context=str(gold_path),
        require_unique=True,
    )
    val_gold_by_event, val_gold_name = _load_val_gold(
        entries, verified_paths
    )

    assignment_counts: dict[str, dict[str, int]] = {}
    assignment_snippet_hashes: dict[str, dict[str, set[str]]] = {}
    assignment_files: dict[str, dict[str, str]] = {}
    expected_runtime_provenance: dict[str, dict[str, Any]] = {}
    for backbone in BACKBONES:
        models = manifest.get("models")
        model_entry = (
            models.get(backbone) if isinstance(models, Mapping) else None
        )
        if not isinstance(model_entry, Mapping):
            raise FineTuneAnalysisError(
                f"Prepared manifest is missing model entry {backbone}"
            )
        files = _model_prepared_files(manifest, backbone)
        registry_entry = files.get("eval_registry")
        if not isinstance(registry_entry, Mapping):
            raise FineTuneAnalysisError(
                f"Prepared manifest is missing {backbone}:eval_registry"
            )
        tokenizer_sha = str(model_entry.get("tokenizer_sha256") or "")
        if not _SHA256.fullmatch(tokenizer_sha):
            raise FineTuneAnalysisError(
                f"Prepared manifest has invalid {backbone} tokenizer SHA"
            )
        label_token_ids = model_entry.get("label_token_ids")
        if (
            not isinstance(label_token_ids, Mapping)
            or set(label_token_ids) != set(LETTERS)
        ):
            raise FineTuneAnalysisError(
                f"Prepared manifest has invalid {backbone} label-token IDs"
            )
        expected_runtime_provenance[backbone] = {
            "registry_sha256": str(registry_entry.get("sha256") or ""),
            "tokenizer_sha256": tokenizer_sha,
            "label_token_ids": {
                letter: int(label_token_ids[letter]) for letter in LETTERS
            },
        }
        if not _SHA256.fullmatch(
            expected_runtime_provenance[backbone]["registry_sha256"]
        ):
            raise FineTuneAnalysisError(
                f"Prepared manifest has invalid {backbone} eval-registry SHA"
            )
        assignment_snippet_hashes[backbone] = {}
        assignment_files[backbone] = {}
        for assignment in ASSIGNMENTS:
            name = f"train_assignment_{assignment}"
            entry = files.get(name)
            if not isinstance(entry, Mapping):
                raise FineTuneAnalysisError(
                    f"Prepared manifest is missing {backbone}:{name}"
                )
            path = _metadata_entry_path(
                entry,
                relative_to=manifest_path.parent,
                name=f"{backbone}:{name}",
            )
            assignment_files[backbone][assignment] = str(path)
            (
                counts,
                assignment_snippet_hashes[backbone][assignment],
            ) = _assignment_training_summary(
                path,
                assignment=f"{backbone}:{assignment}",
            )
            assignment_counts[f"{backbone}:{assignment}"] = counts
    unique_count_vectors = {
        tuple(counts[letter] for letter in LETTERS)
        for counts in assignment_counts.values()
    }
    if len(unique_count_vectors) != 1:
        raise FineTuneAnalysisError(
            "Tokenizer-specific training assignment label counts differ; "
            "arm reversal/tokenization must not alter the label prior"
        )
    canonical_counts = assignment_counts["qwen3:a"]
    total = sum(canonical_counts.values())
    assignment_prior = {
        letter: canonical_counts[letter] / total
        for letter in LETTERS
    }

    prepared_metadata: dict[str, Any] = {
        "gold_file_key": gold_name,
        "gold_file": _file_metadata(gold_path, rows=len(gold_rows)),
        "val_gold_file_key": val_gold_name,
        "val_gold_claim_count": len(val_gold_by_event),
        "training_assignment_files": assignment_files,
        "training_assignment_label_counts": assignment_counts,
        "training_assignment_selected_snippet_hashes": {
            backbone: {
                assignment: sorted(values)
                for assignment, values in assignments.items()
            }
            for backbone, assignments in assignment_snippet_hashes.items()
        },
        "training_label_prior": assignment_prior,
        "expected_runtime_provenance": expected_runtime_provenance,
        "verified_files": {
            name: _file_metadata(path)
            for name, path in sorted(all_verified_files.items())
        },
    }
    return (
        manifest,
        gold_by_event,
        val_gold_by_event,
        assignment_prior,
        prepared_metadata,
    )


def _verify_runtime_provenance(
    runtime: Mapping[str, Any],
    *,
    runtime_path: Path,
) -> dict[str, Any]:
    for name in (
        "base_model_sha256",
        "tokenizer_sha256",
        "adapter_sha256",
    ):
        if not _SHA256.fullmatch(str(runtime.get(name) or "")):
            raise FineTuneAnalysisError(
                f"{runtime_path}: {name} must be a SHA-256 digest"
            )
    marker_entry = runtime.get("training_complete")
    if not isinstance(marker_entry, Mapping):
        raise FineTuneAnalysisError(
            f"{runtime_path}: training_complete metadata is required"
        )
    marker_path = _verify_file_entry(
        "training_complete",
        marker_entry,
        relative_to=runtime_path.parent,
    )
    marker_payload = load_json(marker_path)
    marker_is_complete = any(
        marker_payload.get(name) is True
        for name in ("completed", "training_complete", "complete")
    )
    if not marker_is_complete:
        raise FineTuneAnalysisError(
            f"{marker_path}: training marker has no recognized true completion flag"
        )
    train_config_entry = runtime.get("train_config")
    if not isinstance(train_config_entry, Mapping):
        raise FineTuneAnalysisError(
            f"{runtime_path}: train_config metadata is required"
        )
    train_config_path = _verify_file_entry(
        "train_config",
        train_config_entry,
        relative_to=runtime_path.parent,
    )
    try:
        import yaml
    except ImportError as exc:
        raise FineTuneAnalysisError(
            "PyYAML is required to validate training-config consistency"
        ) from exc
    try:
        with train_config_path.open("r", encoding="utf-8") as handle:
            train_config_payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise FineTuneAnalysisError(
            f"{train_config_path}: cannot parse frozen training config"
        ) from exc
    if not isinstance(train_config_payload, Mapping):
        raise FineTuneAnalysisError(
            f"{train_config_path}: training config is not a mapping"
        )
    normalized_config = json.loads(
        json.dumps(train_config_payload, ensure_ascii=False)
    )
    for key in ("output_dir", "eval_output_dir", "prompt_stats_output_dir"):
        if key in normalized_config:
            normalized_config[key] = f"<{key}>"
    data = normalized_config.get("data")
    if isinstance(data, dict):
        for key in ("train_candidates", "val_candidates", "test_candidates"):
            if key in data:
                data[key] = f"<{key}>"
    sft_train = normalized_config.get("sft_train")
    if isinstance(sft_train, dict) and "seed" in sft_train:
        sft_train["seed"] = "<seed>"
    experiment = normalized_config.get("experiment")
    if isinstance(experiment, dict) and "name" in experiment:
        experiment["name"] = "<experiment_name>"
    normalized_config_sha = hashlib.sha256(
        json.dumps(
            normalized_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    code = runtime.get("code")
    if not isinstance(code, Mapping):
        raise FineTuneAnalysisError(
            f"{runtime_path}: code provenance mapping is required"
        )
    code_files: dict[str, Any] = {}
    for name in ("cross_verifier_finetune", "label_token_trainer"):
        entry = code.get(name)
        if not isinstance(entry, Mapping):
            raise FineTuneAnalysisError(
                f"{runtime_path}: missing code.{name} metadata"
            )
        path = _verify_file_entry(
            f"code.{name}",
            entry,
            relative_to=runtime_path.parent,
        )
        code_files[name] = _file_metadata(path)
    return {
        "training_complete": _file_metadata(marker_path),
        "train_config": _file_metadata(train_config_path),
        "normalized_train_config_sha256": normalized_config_sha,
        "code": code_files,
        "base_model_sha256": str(runtime["base_model_sha256"]),
        "tokenizer_sha256": str(runtime["tokenizer_sha256"]),
        "adapter_sha256": str(runtime["adapter_sha256"]),
    }


def _resolve_result_and_runtime(
    supplied_path: str | Path,
) -> tuple[Path, Path]:
    path = Path(supplied_path).resolve()
    if path.is_dir():
        runtime_path = path / "runtime_manifest.json"
        if not runtime_path.is_file():
            raise FineTuneAnalysisError(
                f"Missing runtime_manifest.json under {path}"
            )
        runtime = load_json(runtime_path)
        file_entry = (runtime.get("files") or {}).get("logical_results")
        if file_entry:
            result_path = _metadata_entry_path(
                file_entry,
                relative_to=runtime_path.parent,
                name="logical_results",
            )
        else:
            result_path = path / "logical_results.jsonl"
        return result_path.resolve(), runtime_path.resolve()
    if path.name == "runtime_manifest.json":
        runtime = load_json(path)
        file_entry = (runtime.get("files") or {}).get("logical_results")
        if file_entry:
            result_path = _metadata_entry_path(
                file_entry,
                relative_to=path.parent,
                name="logical_results",
            )
        else:
            result_path = path.with_name("logical_results.jsonl")
        return result_path.resolve(), path
    return path, path.with_name("runtime_manifest.json").resolve()


def _numeric_map(
    value: Any,
    *,
    context: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(LETTERS):
        raise FineTuneAnalysisError(
            f"{context}: expected exactly six A-F scores"
        )
    output = {letter: float(value[letter]) for letter in LETTERS}
    if any(not math.isfinite(number) for number in output.values()):
        raise FineTuneAnalysisError(f"{context}: non-finite label score")
    return output


def _softmax(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    denominator = sum(math.exp(value - maximum) for value in values.values())
    return {
        letter: math.exp(values[letter] - maximum) / denominator
        for letter in LETTERS
    }


def _validate_logical_row(
    row: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    row_number: int,
    result_path: Path,
) -> dict[str, Any]:
    context = f"{result_path}:{row_number}"
    leaked = sorted(_GOLD_LEAK_KEYS.intersection(row))
    if leaked:
        raise FineTuneAnalysisError(
            f"{context}: result row leaks test gold via {leaked}"
        )
    required = (
        "run_id",
        "backbone",
        "assignment_id",
        "seed",
        "logical_id",
        "event_id",
        "comparison_type",
        "evidence_arm",
        "k_visible",
        "input_ids_sha256",
        "token_count",
        "logits",
        "log_probs",
        "probabilities",
        "pred_label",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise FineTuneAnalysisError(f"{context}: missing fields {missing}")
    snippet_values = row.get("evidence_snippet_sha256s")
    row_comparison_type = str(row.get("comparison_type") or "")
    if row_comparison_type in TEST_COMPARISON_TYPES:
        if (
            not isinstance(snippet_values, Sequence)
            or isinstance(snippet_values, (str, bytes))
        ):
            raise FineTuneAnalysisError(
                f"{context}: test row requires evidence_snippet_sha256s"
            )
        normalized_snippet_hashes = [str(value) for value in snippet_values]
        if not normalized_snippet_hashes:
            raise FineTuneAnalysisError(
                f"{context}: test evidence snippet list is empty"
            )
        if any(
            not _SHA256.fullmatch(value)
            for value in normalized_snippet_hashes
        ):
            raise FineTuneAnalysisError(
                f"{context}: invalid evidence snippet SHA-256"
            )
    else:
        normalized_snippet_hashes = (
            [str(value) for value in snippet_values]
            if isinstance(snippet_values, Sequence)
            and not isinstance(snippet_values, (str, bytes))
            else []
        )
    for name in ("run_id", "backbone", "assignment_id"):
        if str(row[name]) != str(runtime[name]):
            raise FineTuneAnalysisError(
                f"{context}: {name} differs from runtime manifest"
            )
    if int(row["seed"]) != int(runtime["seed"]):
        raise FineTuneAnalysisError(
            f"{context}: seed differs from runtime manifest"
        )
    logical_id = str(row["logical_id"])
    event_id = str(row["event_id"])
    if not logical_id or not event_id:
        raise FineTuneAnalysisError(f"{context}: empty logical/event id")
    comparison_type = str(row["comparison_type"])
    evidence_arm = str(row["evidence_arm"])
    if comparison_type not in ALLOWED_COMPARISON_TYPES:
        raise FineTuneAnalysisError(
            f"{context}: unsupported comparison_type {comparison_type!r}"
        )
    if evidence_arm not in ALLOWED_ARMS:
        raise FineTuneAnalysisError(
            f"{context}: unsupported evidence_arm {evidence_arm!r}"
        )
    if comparison_type in TEST_COMPARISON_TYPES and evidence_arm not in TEST_ARMS:
        raise FineTuneAnalysisError(
            f"{context}: test row requires evitrace or s4 arm"
        )
    if comparison_type == "val_paired" and evidence_arm not in TEST_ARMS:
        raise FineTuneAnalysisError(
            f"{context}: val_paired row requires evitrace or s4 arm"
        )
    if comparison_type == "val_claim_only" and evidence_arm != "claim_only":
        raise FineTuneAnalysisError(
            f"{context}: val_claim_only row requires claim_only arm"
        )
    if comparison_type == "val_mismatched" and evidence_arm != "mismatched":
        raise FineTuneAnalysisError(
            f"{context}: val_mismatched row requires mismatched arm"
        )
    k_visible = int(row["k_visible"])
    if k_visible <= 0:
        raise FineTuneAnalysisError(f"{context}: k_visible must be positive")
    k_value = row.get("k")
    if k_value is not None:
        k_value = int(k_value)
        minimum_k = 0 if comparison_type == "val_claim_only" else 1
        maximum_k = (
            int(row.get("donor_k_visible") or k_value)
            if comparison_type == "val_mismatched"
            else k_visible
        )
        if k_value < minimum_k or k_value > maximum_k:
            raise FineTuneAnalysisError(
                f"{context}: k must be in [{minimum_k}, {maximum_k}]"
            )
    if comparison_type == "prefix" and k_value is None:
        raise FineTuneAnalysisError(f"{context}: prefix row requires k")
    relation = row.get("prefix_relation")
    if relation is not None:
        relation = str(relation)
        if (
            relation not in PREFIX_RELATIONS
            and relation != "not_applicable"
        ):
            raise FineTuneAnalysisError(
                f"{context}: invalid prefix_relation {relation!r}"
            )
    if comparison_type == "prefix" and relation is None:
        raise FineTuneAnalysisError(
            f"{context}: prefix row requires prefix_relation"
        )
    token_count = int(row["token_count"])
    if token_count <= 0:
        raise FineTuneAnalysisError(f"{context}: token_count must be positive")
    input_sha = str(row["input_ids_sha256"])
    if not _SHA256.fullmatch(input_sha):
        raise FineTuneAnalysisError(
            f"{context}: input_ids_sha256 is not a SHA-256 digest"
        )

    logits = _numeric_map(row["logits"], context=f"{context}:logits")
    log_probs = _numeric_map(
        row["log_probs"], context=f"{context}:log_probs"
    )
    probabilities = _numeric_map(
        row["probabilities"], context=f"{context}:probabilities"
    )
    probability_sum = sum(probabilities.values())
    log_probability_sum = sum(math.exp(value) for value in log_probs.values())
    if abs(probability_sum - 1.0) > 1.0e-5:
        raise FineTuneAnalysisError(
            f"{context}: probabilities do not sum to one"
        )
    if abs(log_probability_sum - 1.0) > 1.0e-5:
        raise FineTuneAnalysisError(
            f"{context}: log_probs are not normalized"
        )
    if any(
        abs(probabilities[letter] - math.exp(log_probs[letter])) > 1.0e-5
        for letter in LETTERS
    ):
        raise FineTuneAnalysisError(
            f"{context}: probabilities and log_probs disagree"
        )
    softmax_logits = _softmax(logits)
    if any(
        abs(probabilities[letter] - softmax_logits[letter]) > 1.0e-4
        for letter in LETTERS
    ):
        raise FineTuneAnalysisError(
            f"{context}: probabilities are not softmax(logits)"
        )
    pred_label = _normalize_label(
        row["pred_label"], context=f"{context}:pred_label"
    )
    pred_letter = LABEL_TO_LETTER[pred_label]
    maximum = max(log_probs.values())
    if log_probs[pred_letter] < maximum - 1.0e-8:
        raise FineTuneAnalysisError(
            f"{context}: pred_label is not an argmax label"
        )
    return {
        **row,
        "run_id": str(row["run_id"]),
        "backbone": str(row["backbone"]),
        "assignment_id": str(row["assignment_id"]),
        "seed": int(row["seed"]),
        "logical_id": logical_id,
        "event_id": event_id,
        "comparison_type": comparison_type,
        "evidence_arm": evidence_arm,
        "k": k_value,
        "k_visible": k_visible,
        "prefix_relation": relation,
        "token_count": token_count,
        "evidence_snippet_sha256s": normalized_snippet_hashes,
        "logits": logits,
        "log_probs": log_probs,
        "probabilities": probabilities,
        "pred_label": pred_label,
    }


def _load_result_run(
    supplied_path: str | Path,
    *,
    prepared_manifest_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result_path, runtime_path = _resolve_result_and_runtime(supplied_path)
    if not result_path.is_file():
        raise FineTuneAnalysisError(f"Missing logical results: {result_path}")
    if not runtime_path.is_file():
        raise FineTuneAnalysisError(f"Missing runtime manifest: {runtime_path}")
    runtime = load_json(runtime_path)
    if runtime.get("complete") is not True:
        raise FineTuneAnalysisError(
            f"Runtime manifest is incomplete: {runtime_path}"
        )
    for name in ("run_id", "backbone", "assignment_id", "seed"):
        if name not in runtime:
            raise FineTuneAnalysisError(f"{runtime_path}: missing {name}")
    run_id = str(runtime["run_id"])
    backbone = str(runtime["backbone"])
    assignment = str(runtime["assignment_id"])
    seed = int(runtime["seed"])
    if not run_id:
        raise FineTuneAnalysisError(f"{runtime_path}: empty run_id")
    if backbone not in BACKBONES:
        raise FineTuneAnalysisError(
            f"{runtime_path}: backbone must be one of {BACKBONES}"
        )
    if assignment not in ASSIGNMENTS:
        raise FineTuneAnalysisError(
            f"{runtime_path}: assignment_id must be one of {ASSIGNMENTS}"
        )
    provenance = _verify_runtime_provenance(
        runtime,
        runtime_path=runtime_path,
    )
    observed_prepared_sha = runtime.get(
        "prepared_manifest_sha",
        runtime.get("prepared_manifest_sha256"),
    )
    if str(observed_prepared_sha or "") != prepared_manifest_sha:
        raise FineTuneAnalysisError(
            f"{runtime_path}: prepared-manifest SHA mismatch"
        )
    counts = runtime.get("counts")
    if not isinstance(counts, Mapping) or "logical_results" not in counts:
        raise FineTuneAnalysisError(
            f"{runtime_path}: counts.logical_results is required"
        )
    file_entry = (runtime.get("files") or {}).get("logical_results")
    if file_entry is not None:
        recorded_path = _verify_file_entry(
            "logical_results",
            file_entry,
            relative_to=runtime_path.parent,
        )
        if recorded_path.resolve() != result_path.resolve():
            raise FineTuneAnalysisError(
                f"{runtime_path}: logical-results path mismatch"
            )

    raw_rows = load_jsonl(result_path)
    if len(raw_rows) != int(counts["logical_results"]):
        raise FineTuneAnalysisError(
            f"{runtime_path}: logical-results count mismatch"
        )
    rows = [
        _validate_logical_row(
            row,
            runtime=runtime,
            row_number=index,
            result_path=result_path,
        )
        for index, row in enumerate(raw_rows, start=1)
    ]
    observed_type_counts = {
        comparison_type: sum(
            str(row["comparison_type"]) == comparison_type for row in rows
        )
        for comparison_type in EXPECTED_LOGICAL_COUNTS_PER_RUN
    }
    if observed_type_counts != EXPECTED_LOGICAL_COUNTS_PER_RUN:
        raise FineTuneAnalysisError(
            f"{result_path}: per-type logical count mismatch; "
            f"observed={observed_type_counts}, "
            f"expected={EXPECTED_LOGICAL_COUNTS_PER_RUN}"
        )
    if len(rows) != EXPECTED_LOGICAL_RESULTS_PER_RUN:
        raise FineTuneAnalysisError(
            f"{result_path}: expected {EXPECTED_LOGICAL_RESULTS_PER_RUN} "
            f"logical results, found {len(rows)}"
        )
    logical_ids = [str(row["logical_id"]) for row in rows]
    if len(logical_ids) != len(set(logical_ids)):
        raise FineTuneAnalysisError(
            f"{result_path}: duplicate logical_id within run"
        )
    registry_sha = str(runtime.get("registry_sha256") or "")
    if not _SHA256.fullmatch(registry_sha):
        raise FineTuneAnalysisError(
            f"{runtime_path}: registry_sha256 must be a SHA-256 digest"
        )
    runtime_label_token_ids = runtime.get("label_token_ids")
    if (
        not isinstance(runtime_label_token_ids, Mapping)
        or set(runtime_label_token_ids) != set(LETTERS)
    ):
        raise FineTuneAnalysisError(
            f"{runtime_path}: label_token_ids must contain exactly A-F"
        )
    run_metadata = {
        "run_id": run_id,
        "backbone": backbone,
        "assignment_id": assignment,
        "seed": seed,
        "base_model_sha256": str(runtime["base_model_sha256"]),
        "adapter_sha256": str(runtime["adapter_sha256"]),
        "tokenizer_sha256": str(runtime["tokenizer_sha256"]),
        "registry_sha256": registry_sha,
        "label_token_ids": {
            letter: int(runtime_label_token_ids[letter]) for letter in LETTERS
        },
        "provenance": provenance,
        "logical_counts_by_type": observed_type_counts,
        "result_file": _file_metadata(result_path, rows=len(rows)),
        "runtime_manifest": _file_metadata(runtime_path),
    }
    return run_metadata, rows, runtime


def _logical_universe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["event_id"]),
        str(row["comparison_type"]),
        str(row["evidence_arm"]),
        row.get("k"),
        row.get("prefix_relation"),
        int(row["k_visible"]),
        tuple(str(value) for value in row.get("evidence_snippet_sha256s", ())),
    )


def _validate_run_grid(
    runs: Sequence[Mapping[str, Any]],
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[int]:
    if len(runs) != 12:
        raise FineTuneAnalysisError(
            f"Expected exactly 12 runs, found {len(runs)}"
        )
    run_ids = [str(run["run_id"]) for run in runs]
    if len(set(run_ids)) != 12:
        raise FineTuneAnalysisError("run_id values must be unique")
    cells: dict[tuple[str, str], set[int]] = defaultdict(set)
    triples: set[tuple[str, str, int]] = set()
    for run in runs:
        key = (
            str(run["backbone"]),
            str(run["assignment_id"]),
            int(run["seed"]),
        )
        if key in triples:
            raise FineTuneAnalysisError(f"Duplicate run grid cell: {key}")
        triples.add(key)
        cells[key[:2]].add(key[2])
    if set(cells) != {
        (backbone, assignment)
        for backbone in BACKBONES
        for assignment in ASSIGNMENTS
    }:
        raise FineTuneAnalysisError(
            "Run grid must contain qwen3/llama31 x a/b"
        )
    seed_sets = set()
    for cell, seeds in cells.items():
        if len(seeds) != 3:
            raise FineTuneAnalysisError(
                f"Run grid cell {cell} has {len(seeds)} seeds, expected 3"
            )
        seed_sets.add(tuple(sorted(seeds)))
    if len(seed_sets) != 1:
        raise FineTuneAnalysisError(
            "All backbone/assignment cells must use the same three seeds"
        )
    seeds = list(next(iter(seed_sets)))
    if tuple(seeds) != FORMAL_SEEDS:
        raise FineTuneAnalysisError(
            f"Formal run seeds must be exactly {FORMAL_SEEDS}, found {seeds}"
        )

    reference_universe: set[tuple[Any, ...]] | None = None
    for run_id in sorted(rows_by_run):
        universe = {_logical_universe_key(row) for row in rows_by_run[run_id]}
        if len(universe) != len(rows_by_run[run_id]):
            raise FineTuneAnalysisError(
                f"{run_id}: duplicate logical analysis identity"
            )
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            raise FineTuneAnalysisError(
                f"{run_id}: logical universe differs across the 12 runs"
            )
    return seeds


def _validate_provenance_consistency(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_backbone: dict[str, Any] = {}
    for backbone in BACKBONES:
        selected = [
            run for run in runs if str(run["backbone"]) == backbone
        ]
        base_hashes = {
            str(run["base_model_sha256"]) for run in selected
        }
        tokenizer_hashes = {
            str(run["tokenizer_sha256"]) for run in selected
        }
        normalized_config_hashes = {
            str(run["provenance"]["normalized_train_config_sha256"])
            for run in selected
        }
        if (
            len(base_hashes) != 1
            or len(tokenizer_hashes) != 1
            or len(normalized_config_hashes) != 1
        ):
            raise FineTuneAnalysisError(
                f"{backbone}: base/tokenizer/normalized-config provenance "
                "is inconsistent across assignment/seed runs"
            )
        by_backbone[backbone] = {
            "run_count": len(selected),
            "base_model_sha256": next(iter(base_hashes)),
            "tokenizer_sha256": next(iter(tokenizer_hashes)),
            "normalized_train_config_sha256": next(
                iter(normalized_config_hashes)
            ),
        }
    code_hashes: dict[str, str] = {}
    for code_name in ("cross_verifier_finetune", "label_token_trainer"):
        values = {
            str(run["provenance"]["code"][code_name]["sha256"])
            for run in runs
        }
        if len(values) != 1:
            raise FineTuneAnalysisError(
                f"code.{code_name}: hash differs across the 12 runs"
            )
        code_hashes[code_name] = next(iter(values))
    adapter_hashes = {str(run["adapter_sha256"]) for run in runs}
    if len(adapter_hashes) != len(runs):
        raise FineTuneAnalysisError(
            "Adapter SHA-256 values are not unique across the 12 run cells"
        )
    return {
        "per_backbone": by_backbone,
        "code_sha256": code_hashes,
        "unique_adapter_sha256_count": len(adapter_hashes),
        "raw_train_config_hashes_verified": {
            str(run["run_id"]): str(
                run["provenance"]["train_config"]["sha256"]
            )
            for run in runs
        },
        "normalization_excludes_only": [
            "output_dir",
            "eval_output_dir",
            "prompt_stats_output_dir",
            "data.train_candidates",
            "data.val_candidates",
            "data.test_candidates",
            "sft_train.seed",
            "experiment.name",
        ],
    }


def _macro_f1(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> float:
    if len(gold) != len(predicted):
        raise FineTuneAnalysisError("gold/prediction length mismatch")
    if not gold:
        raise FineTuneAnalysisError("Cannot compute Macro-F1 on no claims")
    scores: list[float] = []
    for label in LIAR6_LABELS:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(gold, predicted)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    return statistics.fmean(scores)


def exact_mcnemar_pvalue(wins: int, losses: int) -> float:
    wins = int(wins)
    losses = int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("McNemar discordant counts must be non-negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * float(tail))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = [float(value) for value in pvalues]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Holm adjustment requires p-values in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _conditional_win_interval(wins: int, losses: int) -> list[float]:
    discordant = wins + losses
    if discordant == 0:
        return [0.0, 1.0]
    try:
        from scipy.stats import binomtest

        interval = binomtest(wins, discordant, 0.5).proportion_ci(
            confidence_level=0.95,
            method="exact",
        )
        return [float(interval.low), float(interval.high)]
    except ImportError:
        # Wilson is only a dependency-free reporting fallback; the McNemar
        # p-value above remains exact.
        probability = wins / discordant
        z_value = 1.959963984540054
        denominator = 1 + z_value**2 / discordant
        center = (
            probability + z_value**2 / (2 * discordant)
        ) / denominator
        half_width = (
            z_value
            * math.sqrt(
                probability * (1 - probability) / discordant
                + z_value**2 / (4 * discordant**2)
            )
            / denominator
        )
        return [max(0.0, center - half_width), min(1.0, center + half_width)]


def compute_run_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute one run's paired metrics without pooling across runs."""

    if not pair_rows:
        raise FineTuneAnalysisError("Cannot analyze an empty paired run")
    gold = [str(row["gold_label"]) for row in pair_rows]
    evitrace = [str(row["evitrace_pred_label"]) for row in pair_rows]
    s4 = [str(row["s4_pred_label"]) for row in pair_rows]
    if any(
        label not in LABEL_TO_ID
        for label in gold + evitrace + s4
    ):
        raise FineTuneAnalysisError("Invalid LIAR6 label in paired rows")
    evi_correct = [
        actual == predicted for actual, predicted in zip(gold, evitrace)
    ]
    s4_correct = [
        actual == predicted for actual, predicted in zip(gold, s4)
    ]
    wins = sum(left and not right for left, right in zip(evi_correct, s4_correct))
    losses = sum(right and not left for left, right in zip(evi_correct, s4_correct))
    both_correct = sum(left and right for left, right in zip(evi_correct, s4_correct))
    both_wrong = sum(
        not left and not right for left, right in zip(evi_correct, s4_correct)
    )
    ties = both_correct + both_wrong
    discordant = wins + losses
    logprob_deltas = [
        float(row["evitrace_gold_logprob"])
        - float(row["s4_gold_logprob"])
        for row in pair_rows
    ]
    if any(not math.isfinite(value) for value in logprob_deltas):
        raise FineTuneAnalysisError("Non-finite gold log-probability delta")
    evitrace_accuracy = statistics.fmean(evi_correct)
    s4_accuracy = statistics.fmean(s4_correct)
    evitrace_macro_f1 = _macro_f1(gold, evitrace)
    s4_macro_f1 = _macro_f1(gold, s4)
    return {
        "n": len(pair_rows),
        "evitrace": {
            "accuracy": evitrace_accuracy,
            "macro_f1": evitrace_macro_f1,
        },
        "s4": {
            "accuracy": s4_accuracy,
            "macro_f1": s4_macro_f1,
        },
        "delta": {
            "accuracy": evitrace_accuracy - s4_accuracy,
            "macro_f1": evitrace_macro_f1 - s4_macro_f1,
            "gold_logprob_mean": statistics.fmean(logprob_deltas),
        },
        "wlt": {
            "evitrace_win": wins,
            "s4_win": losses,
            "tie": ties,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "evitrace_win_rate": wins / len(pair_rows),
            "s4_win_rate": losses / len(pair_rows),
            "tie_rate": ties / len(pair_rows),
            "conditional_evitrace_win_rate": (
                wins / discordant if discordant else 0.5
            ),
            "conditional_win_rate_ci95": _conditional_win_interval(
                wins, losses
            ),
            "exact_mcnemar_pvalue": exact_mcnemar_pvalue(wins, losses),
        },
        "gold_logprob_delta": {
            "mean": statistics.fmean(logprob_deltas),
            "median": statistics.median(logprob_deltas),
            "positive": sum(value > 1.0e-12 for value in logprob_deltas),
            "negative": sum(value < -1.0e-12 for value in logprob_deltas),
            "tie": sum(abs(value) <= 1.0e-12 for value in logprob_deltas),
        },
    }


def _build_pair_rows(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    runs_by_id: Mapping[str, Mapping[str, Any]],
    gold_by_event: Mapping[str, Mapping[str, Any]],
    *,
    comparison_type: str,
) -> list[dict[str, Any]]:
    if comparison_type not in ("main", "order_only"):
        raise ValueError("pair rows are only defined for main/order_only")
    all_pairs: list[dict[str, Any]] = []
    reference_events: set[str] | None = None
    for run_id in sorted(rows_by_run):
        selected = [
            row
            for row in rows_by_run[run_id]
            if row["comparison_type"] == comparison_type
        ]
        by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in selected:
            key = (str(row["event_id"]), str(row["evidence_arm"]))
            if key in by_key:
                raise FineTuneAnalysisError(
                    f"{run_id}:{comparison_type}: duplicate pair arm {key}"
                )
            by_key[key] = row
        events = {event_id for event_id, _arm in by_key}
        if not events:
            raise FineTuneAnalysisError(
                f"{run_id}: no {comparison_type} test rows"
            )
        if reference_events is None:
            reference_events = events
        elif events != reference_events:
            raise FineTuneAnalysisError(
                f"{comparison_type}: event universe differs across runs"
            )
        for event_id in sorted(events):
            if event_id not in gold_by_event:
                raise FineTuneAnalysisError(
                    f"{comparison_type}:{event_id}: absent from independent test gold"
                )
            arm_rows: dict[str, Mapping[str, Any]] = {}
            for arm in TEST_ARMS:
                row = by_key.get((event_id, arm))
                if row is None:
                    raise FineTuneAnalysisError(
                        f"{run_id}:{comparison_type}:{event_id}: missing {arm}"
                    )
                arm_rows[arm] = row
            evi = arm_rows["evitrace"]
            s4 = arm_rows["s4"]
            if int(evi["k_visible"]) != int(s4["k_visible"]):
                raise FineTuneAnalysisError(
                    f"{run_id}:{comparison_type}:{event_id}: unmatched K"
                )
            if (
                comparison_type == "order_only"
                and evi.get("prefix_relation") is not None
                and evi.get("prefix_relation") != "same_set_different_order"
            ):
                raise FineTuneAnalysisError(
                    f"{run_id}:{event_id}: order-only relation is not same-set"
                )
            gold_label = str(gold_by_event[event_id]["gold_label"])
            complexity = str(
                gold_by_event[event_id].get("complexity") or ""
            )
            if complexity not in {"single", "multi"}:
                raise FineTuneAnalysisError(
                    f"{event_id}: gold_test complexity must be single/multi"
                )
            gold_letter = LABEL_TO_LETTER[gold_label]
            run = runs_by_id[run_id]
            all_pairs.append(
                {
                    "run_id": run_id,
                    "backbone": str(run["backbone"]),
                    "assignment_id": str(run["assignment_id"]),
                    "seed": int(run["seed"]),
                    "event_id": event_id,
                    "comparison_type": comparison_type,
                    "gold_label": gold_label,
                    "complexity": complexity,
                    "evitrace_pred_label": str(evi["pred_label"]),
                    "s4_pred_label": str(s4["pred_label"]),
                    "evitrace_gold_logprob": float(
                        evi["log_probs"][gold_letter]
                    ),
                    "s4_gold_logprob": float(s4["log_probs"][gold_letter]),
                    "evitrace_token_count": int(evi["token_count"]),
                    "s4_token_count": int(s4["token_count"]),
                    "token_difference_evi_minus_s4": int(evi["token_count"])
                    - int(s4["token_count"]),
                    "k_visible": int(evi["k_visible"]),
                    "evitrace_evidence_snippet_sha256s": list(
                        evi["evidence_snippet_sha256s"]
                    ),
                    "s4_evidence_snippet_sha256s": list(
                        s4["evidence_snippet_sha256s"]
                    ),
                }
            )
    return all_pairs


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise FineTuneAnalysisError("Cannot average an empty metric")
    return statistics.fmean(materialized)


def _aggregate_run_metric_group(
    run_metrics: Mapping[str, Mapping[str, Any]],
    run_ids: Sequence[str],
) -> dict[str, Any]:
    if not run_ids:
        raise FineTuneAnalysisError("Cannot aggregate no runs")
    return {
        "run_count": len(run_ids),
        "evitrace": {
            "accuracy": _mean(
                run_metrics[run_id]["evitrace"]["accuracy"]
                for run_id in run_ids
            ),
            "macro_f1": _mean(
                run_metrics[run_id]["evitrace"]["macro_f1"]
                for run_id in run_ids
            ),
        },
        "s4": {
            "accuracy": _mean(
                run_metrics[run_id]["s4"]["accuracy"]
                for run_id in run_ids
            ),
            "macro_f1": _mean(
                run_metrics[run_id]["s4"]["macro_f1"]
                for run_id in run_ids
            ),
        },
        "delta": {
            "accuracy": _mean(
                run_metrics[run_id]["delta"]["accuracy"]
                for run_id in run_ids
            ),
            "macro_f1": _mean(
                run_metrics[run_id]["delta"]["macro_f1"]
                for run_id in run_ids
            ),
            "gold_logprob_mean": _mean(
                run_metrics[run_id]["delta"]["gold_logprob_mean"]
                for run_id in run_ids
            ),
        },
        "wlt_rates": {
            "evitrace_win": _mean(
                run_metrics[run_id]["wlt"]["evitrace_win_rate"]
                for run_id in run_ids
            ),
            "s4_win": _mean(
                run_metrics[run_id]["wlt"]["s4_win_rate"]
                for run_id in run_ids
            ),
            "tie": _mean(
                run_metrics[run_id]["wlt"]["tie_rate"]
                for run_id in run_ids
            ),
            "conditional_evitrace_win_rate": _mean(
                run_metrics[run_id]["wlt"][
                    "conditional_evitrace_win_rate"
                ]
                for run_id in run_ids
            ),
        },
    }


def _summarize_values(values: Sequence[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"n": 0}
    return {
        "n": len(materialized),
        "mean": statistics.fmean(materialized),
        "sd": (
            statistics.stdev(materialized)
            if len(materialized) > 1
            else 0.0
        ),
        "min": min(materialized),
        "max": max(materialized),
        "positive": sum(value > 1.0e-12 for value in materialized),
        "negative": sum(value < -1.0e-12 for value in materialized),
        "zero": sum(abs(value) <= 1.0e-12 for value in materialized),
    }


def hierarchical_point(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate run -> backbone -> equal-weight two-backbone panel."""

    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_meta: dict[str, tuple[str, str, int]] = {}
    for row in pair_rows:
        run_id = str(row["run_id"])
        meta = (
            str(row["backbone"]),
            str(row["assignment_id"]),
            int(row["seed"]),
        )
        previous = run_meta.setdefault(run_id, meta)
        if previous != meta:
            raise FineTuneAnalysisError(f"{run_id}: inconsistent run metadata")
        rows_by_run[run_id].append(row)
    run_metrics = {
        run_id: compute_run_metrics(rows)
        for run_id, rows in sorted(rows_by_run.items())
    }
    per_backbone: dict[str, Any] = {}
    for backbone in BACKBONES:
        run_ids = sorted(
            run_id
            for run_id, meta in run_meta.items()
            if meta[0] == backbone
        )
        if run_ids:
            per_backbone[backbone] = _aggregate_run_metric_group(
                run_metrics, run_ids
            )
    if set(per_backbone) != set(BACKBONES):
        raise FineTuneAnalysisError(
            "Hierarchical point requires both frozen backbones"
        )
    panel: dict[str, Any] = {
        "backbone_count": 2,
        "evitrace": {
            metric: _mean(
                per_backbone[backbone]["evitrace"][metric]
                for backbone in BACKBONES
            )
            for metric in ("accuracy", "macro_f1")
        },
        "s4": {
            metric: _mean(
                per_backbone[backbone]["s4"][metric]
                for backbone in BACKBONES
            )
            for metric in ("accuracy", "macro_f1")
        },
        "delta": {
            metric: _mean(
                per_backbone[backbone]["delta"][metric]
                for backbone in BACKBONES
            )
            for metric in ("accuracy", "macro_f1", "gold_logprob_mean")
        },
        "wlt_rates": {
            metric: _mean(
                per_backbone[backbone]["wlt_rates"][metric]
                for backbone in BACKBONES
            )
            for metric in (
                "evitrace_win",
                "s4_win",
                "tie",
                "conditional_evitrace_win_rate",
            )
        },
    }
    pooled_wins = sum(
        int(item["wlt"]["evitrace_win"]) for item in run_metrics.values()
    )
    pooled_losses = sum(
        int(item["wlt"]["s4_win"]) for item in run_metrics.values()
    )
    pooled_ties = sum(int(item["wlt"]["tie"]) for item in run_metrics.values())
    panel["wlt_pooled_descriptive"] = {
        "evitrace_win": pooled_wins,
        "s4_win": pooled_losses,
        "tie": pooled_ties,
        "conditional_evitrace_win_rate": (
            pooled_wins / (pooled_wins + pooled_losses)
            if pooled_wins + pooled_losses
            else 0.5
        ),
        "exact_mcnemar_pvalue_if_rows_were_independent": exact_mcnemar_pvalue(
            pooled_wins, pooled_losses
        ),
        "inferential_warning": (
            "Claims repeat across runs; use the shared-claim randomization "
            "test, not this pooled descriptive p-value, for inference."
        ),
    }

    per_assignment: dict[str, Any] = {}
    for assignment in ASSIGNMENTS:
        run_ids = sorted(
            run_id
            for run_id, meta in run_meta.items()
            if meta[1] == assignment
        )
        if run_ids:
            per_assignment[assignment] = _aggregate_run_metric_group(
                run_metrics, run_ids
            )
    per_seed: dict[str, Any] = {}
    for seed in sorted({meta[2] for meta in run_meta.values()}):
        run_ids = sorted(
            run_id
            for run_id, meta in run_meta.items()
            if meta[2] == seed
        )
        per_seed[str(seed)] = _aggregate_run_metric_group(
            run_metrics, run_ids
        )
    per_backbone_assignment: dict[str, Any] = {}
    for backbone in BACKBONES:
        for assignment in ASSIGNMENTS:
            run_ids = sorted(
                run_id
                for run_id, meta in run_meta.items()
                if meta[:2] == (backbone, assignment)
            )
            if run_ids:
                per_backbone_assignment[
                    f"{backbone}:{assignment}"
                ] = _aggregate_run_metric_group(run_metrics, run_ids)

    heterogeneity = {
        "run_macro_f1_delta": _summarize_values(
            [
                float(item["delta"]["macro_f1"])
                for item in run_metrics.values()
            ]
        ),
        "backbone_macro_f1_delta": _summarize_values(
            [
                float(item["delta"]["macro_f1"])
                for item in per_backbone.values()
            ]
        ),
        "assignment_macro_f1_delta": _summarize_values(
            [
                float(item["delta"]["macro_f1"])
                for item in per_assignment.values()
            ]
        ),
        "seed_macro_f1_delta": _summarize_values(
            [
                float(item["delta"]["macro_f1"])
                for item in per_seed.values()
            ]
        ),
        "mcnemar_pvalues_by_run": {
            run_id: float(metrics["wlt"]["exact_mcnemar_pvalue"])
            for run_id, metrics in run_metrics.items()
        },
    }
    return {
        "aggregation_order": (
            "assignment/seed run -> equal mean within backbone -> "
            "equal mean across qwen3 and llama31"
        ),
        "per_run": run_metrics,
        "per_backbone_assignment": per_backbone_assignment,
        "per_backbone": per_backbone,
        "per_assignment": per_assignment,
        "per_seed": per_seed,
        "panel": panel,
        "heterogeneity": heterogeneity,
    }


def _ordered_claim_matrix(
    pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows_by_run: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    run_backbone: dict[str, str] = {}
    labels_by_event: dict[str, str] = {}
    for row in pair_rows:
        run_id = str(row["run_id"])
        event_id = str(row["event_id"])
        if event_id in rows_by_run[run_id]:
            raise FineTuneAnalysisError(
                f"{run_id}:{event_id}: duplicate paired claim"
            )
        rows_by_run[run_id][event_id] = row
        previous_backbone = run_backbone.setdefault(
            run_id, str(row["backbone"])
        )
        if previous_backbone != str(row["backbone"]):
            raise FineTuneAnalysisError(
                f"{run_id}: inconsistent backbone in paired matrix"
            )
        previous_label = labels_by_event.setdefault(
            event_id, str(row["gold_label"])
        )
        if previous_label != str(row["gold_label"]):
            raise FineTuneAnalysisError(
                f"{event_id}: inconsistent gold label across runs"
            )
    run_ids = sorted(rows_by_run)
    event_ids = sorted(labels_by_event)
    for run_id in run_ids:
        if set(rows_by_run[run_id]) != set(event_ids):
            raise FineTuneAnalysisError(
                f"{run_id}: paired-claim universe differs"
            )
    gold = np.asarray(
        [LABEL_TO_ID[labels_by_event[event_id]] for event_id in event_ids],
        dtype=np.int8,
    )
    evi = np.asarray(
        [
            [
                LABEL_TO_ID[
                    str(rows_by_run[run_id][event_id]["evitrace_pred_label"])
                ]
                for event_id in event_ids
            ]
            for run_id in run_ids
        ],
        dtype=np.int8,
    )
    s4 = np.asarray(
        [
            [
                LABEL_TO_ID[
                    str(rows_by_run[run_id][event_id]["s4_pred_label"])
                ]
                for event_id in event_ids
            ]
            for run_id in run_ids
        ],
        dtype=np.int8,
    )
    logprob_delta = np.asarray(
        [
            [
                float(
                    rows_by_run[run_id][event_id][
                        "evitrace_gold_logprob"
                    ]
                )
                - float(
                    rows_by_run[run_id][event_id]["s4_gold_logprob"]
                )
                for event_id in event_ids
            ]
            for run_id in run_ids
        ],
        dtype=np.float64,
    )
    backbones = [run_backbone[run_id] for run_id in run_ids]
    return run_ids, backbones, event_ids, gold, evi, s4, logprob_delta


def _confusion_contributions(
    predictions: np.ndarray,
    gold: np.ndarray,
) -> np.ndarray:
    run_count, claim_count = predictions.shape
    contributions = np.zeros(
        (run_count, claim_count, len(LIAR6_LABELS) ** 2),
        dtype=np.float32,
    )
    for run_index in range(run_count):
        flat = gold.astype(np.int64) * len(LIAR6_LABELS) + predictions[
            run_index
        ].astype(np.int64)
        contributions[
            run_index,
            np.arange(claim_count),
            flat,
        ] = 1.0
    return contributions


def _macro_f1_from_confusions(
    confusions: np.ndarray,
) -> np.ndarray:
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1)
    false_positive = confusions.sum(axis=-2) - true_positive
    false_negative = confusions.sum(axis=-1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator != 0,
    )
    return scores.mean(axis=-1)


def _hierarchical_array_mean(
    values: np.ndarray,
    backbones: Sequence[str],
) -> np.ndarray:
    """Average the final run dimension within backbone, then across backbone."""

    if values.shape[-1] != len(backbones):
        raise FineTuneAnalysisError("Backbone/run array length mismatch")
    backbone_values = []
    for backbone in BACKBONES:
        indices = [
            index
            for index, observed in enumerate(backbones)
            if observed == backbone
        ]
        if not indices:
            raise FineTuneAnalysisError(
                f"No runs available for backbone {backbone}"
            )
        backbone_values.append(values[..., indices].mean(axis=-1))
    return np.stack(backbone_values, axis=-1).mean(axis=-1)


def _stratified_weight_chunks(
    gold: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 256,
) -> Iterable[np.ndarray]:
    observed_labels = sorted(set(int(value) for value in gold.tolist()))
    completed = 0
    while completed < iterations:
        current = min(chunk_size, iterations - completed)
        weights = np.zeros((current, len(gold)), dtype=np.float32)
        for label in observed_labels:
            indices = np.flatnonzero(gold == label)
            draws = rng.multinomial(
                len(indices),
                np.full(len(indices), 1.0 / len(indices)),
                size=current,
            )
            weights[:, indices] = draws.astype(np.float32)
        completed += current
        yield weights


def _percentile_interval(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    ]


def stratified_claim_bootstrap(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Label-stratified claim bootstrap with all 12 run rows kept together."""

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    (
        run_ids,
        backbones,
        event_ids,
        gold,
        evi,
        s4,
        logprob_delta,
    ) = _ordered_claim_matrix(pair_rows)
    evi_contributions = _confusion_contributions(evi, gold)
    s4_contributions = _confusion_contributions(s4, gold)
    evi_correct = (evi == gold[None, :]).astype(np.float32)
    s4_correct = (s4 == gold[None, :]).astype(np.float32)
    claim_count = len(event_ids)
    distributions: dict[str, list[float]] = {
        "macro_f1_delta": [],
        "accuracy_delta": [],
        "gold_logprob_delta": [],
    }
    rng = np.random.default_rng(int(seed))
    for weights in _stratified_weight_chunks(
        gold,
        iterations=iterations,
        rng=rng,
    ):
        evi_confusion = np.einsum(
            "bn,rnc->brc", weights, evi_contributions, optimize=True
        ).reshape(-1, len(run_ids), 6, 6)
        s4_confusion = np.einsum(
            "bn,rnc->brc", weights, s4_contributions, optimize=True
        ).reshape(-1, len(run_ids), 6, 6)
        evi_f1 = _macro_f1_from_confusions(evi_confusion)
        s4_f1 = _macro_f1_from_confusions(s4_confusion)
        macro_delta = _hierarchical_array_mean(
            evi_f1 - s4_f1, backbones
        )
        evi_accuracy = (
            weights @ evi_correct.T / float(claim_count)
        )
        s4_accuracy = (
            weights @ s4_correct.T / float(claim_count)
        )
        accuracy_delta = _hierarchical_array_mean(
            evi_accuracy - s4_accuracy, backbones
        )
        logp = (
            weights @ logprob_delta.T / float(claim_count)
        )
        hierarchical_logp = _hierarchical_array_mean(logp, backbones)
        distributions["macro_f1_delta"].extend(macro_delta.tolist())
        distributions["accuracy_delta"].extend(accuracy_delta.tolist())
        distributions["gold_logprob_delta"].extend(
            hierarchical_logp.tolist()
        )
    point = hierarchical_point(pair_rows)["panel"]["delta"]
    return {
        "iterations": iterations,
        "seed": int(seed),
        "claim_count": claim_count,
        "run_count_per_claim": len(run_ids),
        "stratification": "independent test gold label",
        "cluster": "event_id; all 12 run rows share each resample count",
        "point": {
            "macro_f1_delta": float(point["macro_f1"]),
            "accuracy_delta": float(point["accuracy"]),
            "gold_logprob_delta": float(point["gold_logprob_mean"]),
        },
        "ci95": {
            name: _percentile_interval(values)
            for name, values in distributions.items()
        },
        "macro_f1_delta_probabilities": {
            "gt_zero": _mean(
                value > 0.0 for value in distributions["macro_f1_delta"]
            ),
            "gt_positive_sesoi": _mean(
                value > SESOI for value in distributions["macro_f1_delta"]
            ),
            "inside_sesoi": _mean(
                -SESOI <= value <= SESOI
                for value in distributions["macro_f1_delta"]
            ),
            "lt_negative_sesoi": _mean(
                value < -SESOI for value in distributions["macro_f1_delta"]
            ),
        },
    }


def shared_claim_swap_randomization(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Randomize EviTrace/S4 once per claim, shared by all twelve runs."""

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("permutation iterations must be positive")
    (
        run_ids,
        backbones,
        event_ids,
        gold,
        evi,
        s4,
        _logprob_delta,
    ) = _ordered_claim_matrix(pair_rows)
    evi_contributions = _confusion_contributions(evi, gold)
    s4_contributions = _confusion_contributions(s4, gold)
    base_evi = evi_contributions.sum(axis=1)
    base_s4 = s4_contributions.sum(axis=1)
    swap_delta = (
        s4_contributions - evi_contributions
    ).transpose(1, 0, 2).reshape(len(event_ids), -1)
    observed_run_delta = _macro_f1_from_confusions(
        base_evi.reshape(len(run_ids), 6, 6)
    ) - _macro_f1_from_confusions(
        base_s4.reshape(len(run_ids), 6, 6)
    )
    observed = float(
        _hierarchical_array_mean(observed_run_delta, backbones)
    )
    rng = np.random.default_rng(int(seed))
    exceedances = 0
    completed = 0
    chunk_size = 512
    while completed < iterations:
        current = min(chunk_size, iterations - completed)
        swap_bits = rng.integers(
            0,
            2,
            size=(current, len(event_ids)),
            dtype=np.int8,
        ).astype(np.float32)
        changes = (swap_bits @ swap_delta).reshape(
            current, len(run_ids), 36
        )
        permuted_evi = (
            base_evi[None, :, :] + changes
        ).reshape(current, len(run_ids), 6, 6)
        permuted_s4 = (
            base_s4[None, :, :] - changes
        ).reshape(current, len(run_ids), 6, 6)
        run_delta = _macro_f1_from_confusions(
            permuted_evi
        ) - _macro_f1_from_confusions(permuted_s4)
        permuted = _hierarchical_array_mean(run_delta, backbones)
        exceedances += int(
            np.count_nonzero(
                np.abs(permuted) >= abs(observed) - 1.0e-15
            )
        )
        completed += current
    return {
        "iterations": iterations,
        "seed": int(seed),
        "claim_count": len(event_ids),
        "run_count_per_claim": len(run_ids),
        "observed_macro_f1_delta": observed,
        "two_sided_pvalue": (exceedances + 1) / (iterations + 1),
        "exceedances": exceedances,
        "same_swap_bit_across_all_runs_for_each_claim": True,
        "null_operation": (
            "For every permutation, one Bernoulli arm-swap bit is drawn per "
            "event_id and applied to that claim in all 12 runs."
        ),
    }


def assess_sesoi(
    point: float,
    ci95: Sequence[float],
    *,
    sesoi: float = SESOI,
) -> dict[str, Any]:
    if len(ci95) != 2 or float(ci95[0]) > float(ci95[1]):
        raise ValueError("ci95 must be an ordered [low, high] pair")
    low, high = (float(ci95[0]), float(ci95[1]))
    point = float(point)
    sesoi = abs(float(sesoi))
    if low > sesoi:
        category = "beneficial_beyond_sesoi"
    elif high < -sesoi:
        category = "harmful_beyond_sesoi"
    elif low >= -sesoi and high <= sesoi:
        category = "practically_equivalent_within_sesoi"
    else:
        category = "inconclusive_relative_to_sesoi"
    return {
        "point": point,
        "ci95": [low, high],
        "sesoi": [-sesoi, sesoi],
        "category": category,
    }


def compute_main_token_sensitivity(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: int = 64,
) -> dict[str, Any]:
    """Apply the fixed |token gap|<=64 rule per tokenizer and in intersection."""

    threshold = int(threshold)
    if threshold < 0:
        raise ValueError("token threshold must be non-negative")
    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_backbone: dict[str, str] = {}
    for row in pair_rows:
        run_id = str(row["run_id"])
        rows_by_run[run_id].append(row)
        run_backbone[run_id] = str(row["backbone"])
    if not rows_by_run:
        raise FineTuneAnalysisError("Token sensitivity requires main rows")
    eligible_by_run = {
        run_id: {
            str(row["event_id"])
            for row in rows
            if abs(int(row["token_difference_evi_minus_s4"])) <= threshold
        }
        for run_id, rows in rows_by_run.items()
    }
    eligible_by_backbone: dict[str, set[str]] = {}
    per_backbone_results: dict[str, Any] = {}
    for backbone in BACKBONES:
        run_ids = [
            run_id
            for run_id in rows_by_run
            if run_backbone[run_id] == backbone
        ]
        if not run_ids:
            raise FineTuneAnalysisError(
                f"Token sensitivity lacks backbone {backbone}"
            )
        eligible_by_backbone[backbone] = set.intersection(
            *(eligible_by_run[run_id] for run_id in run_ids)
        )
        backbone_rows = [
            row
            for row in pair_rows
            if str(row["backbone"]) == backbone
            and str(row["event_id"]) in eligible_by_backbone[backbone]
        ]
        backbone_rows_by_run: dict[
            str, list[Mapping[str, Any]]
        ] = defaultdict(list)
        for row in backbone_rows:
            backbone_rows_by_run[str(row["run_id"])].append(row)
        run_metrics = {
            run_id: compute_run_metrics(rows)
            for run_id, rows in backbone_rows_by_run.items()
        }
        per_backbone_results[backbone] = {
            "claim_count": len(eligible_by_backbone[backbone]),
            "point": _aggregate_run_metric_group(
                run_metrics,
                sorted(run_metrics),
            ),
        }
    intersection = set.intersection(*eligible_by_backbone.values())
    filtered = [
        row
        for row in pair_rows
        if str(row["event_id"]) in intersection
    ]
    return {
        "threshold": threshold,
        "definition": (
            "|token_count_EviTrace-token_count_S4|<=64 is first checked "
            "within every run sharing a tokenizer; the reported panel uses "
            "the claim intersection of the qwen3 and llama31 tokenizers"
        ),
        "eligible_claims_per_run": {
            run_id: len(events)
            for run_id, events in sorted(eligible_by_run.items())
        },
        "eligible_claims_per_backbone_tokenizer": {
            backbone: len(events)
            for backbone, events in sorted(eligible_by_backbone.items())
        },
        "per_backbone_tokenizer": per_backbone_results,
        "intersection_claim_count": len(intersection),
        "intersection_point": (
            hierarchical_point(filtered)["panel"] if intersection else None
        ),
    }


def compute_exact_snippet_sensitivity(
    pair_rows: Sequence[Mapping[str, Any]],
    selected_hashes_by_assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclude a claim per run if either test arm exactly matches train evidence."""

    if set(selected_hashes_by_assignment) == set(ASSIGNMENTS):
        train_sets = {
            backbone: {
                assignment: {
                    str(value)
                    for value in selected_hashes_by_assignment[assignment]
                }
                for assignment in ASSIGNMENTS
            }
            for backbone in BACKBONES
        }
    else:
        train_sets = {
            backbone: {
                assignment: {
                    str(value)
                    for value in selected_hashes_by_assignment[backbone][
                        assignment
                    ]
                }
                for assignment in ASSIGNMENTS
            }
            for backbone in BACKBONES
        }
    if any(
        not train_sets[backbone][assignment]
        for backbone in BACKBONES
        for assignment in ASSIGNMENTS
    ):
        raise FineTuneAnalysisError(
            "Exact-snippet sensitivity requires non-empty "
            "backbone-specific A/B train hash sets"
        )
    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        rows_by_run[str(row["run_id"])].append(row)
    retained: list[Mapping[str, Any]] = []
    counts: dict[str, Any] = {}
    for run_id, rows in sorted(rows_by_run.items()):
        assignments = {str(row["assignment_id"]) for row in rows}
        backbones = {str(row["backbone"]) for row in rows}
        if len(assignments) != 1:
            raise FineTuneAnalysisError(
                f"{run_id}: inconsistent assignment for snippet sensitivity"
            )
        assignment = next(iter(assignments))
        if len(backbones) != 1:
            raise FineTuneAnalysisError(
                f"{run_id}: inconsistent backbone for snippet sensitivity"
            )
        backbone = next(iter(backbones))
        train_hashes = train_sets[backbone][assignment]
        retained_events: set[str] = set()
        excluded_events: set[str] = set()
        for row in rows:
            evi_values = row.get("evitrace_evidence_snippet_sha256s")
            s4_values = row.get("s4_evidence_snippet_sha256s")
            if not isinstance(evi_values, Sequence) or not isinstance(
                s4_values, Sequence
            ):
                raise FineTuneAnalysisError(
                    f"{run_id}: paired row lacks snippet-hash fields"
                )
            overlap = train_hashes.intersection(
                {str(value) for value in evi_values}
                | {str(value) for value in s4_values}
            )
            event_id = str(row["event_id"])
            if overlap:
                excluded_events.add(event_id)
            else:
                retained_events.add(event_id)
                retained.append(row)
        if retained_events.intersection(excluded_events):
            raise FineTuneAnalysisError(
                f"{run_id}: an event was both retained and excluded"
            )
        if not retained_events:
            raise FineTuneAnalysisError(
                f"{run_id}: snippet sensitivity retained no claims"
            )
        counts[run_id] = {
            "assignment_id": assignment,
            "backbone": backbone,
            "original_claim_count": len(retained_events | excluded_events),
            "retained_claim_count": len(retained_events),
            "excluded_claim_count": len(excluded_events),
        }
    point = hierarchical_point(retained)
    delta = float(point["panel"]["delta"]["macro_f1"])
    direction = (
        "evitrace_positive"
        if delta > 1.0e-12
        else "s4_positive"
        if delta < -1.0e-12
        else "tie"
    )
    commitments = {
        backbone: {
            assignment: hashlib.sha256(
                json.dumps(
                    sorted(train_sets[backbone][assignment]),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for assignment in ASSIGNMENTS
        }
        for backbone in BACKBONES
    }
    return {
        "definition": (
            "Within each run, use that run's training assignment selected-"
            "snippet hash set and exclude an event if either EviTrace or S4 "
            "contains any exact training-selected snippet hash."
        ),
        "claim_counts_by_run": counts,
        "training_selected_hash_set_sha256": commitments,
        "point": point,
        "panel_macro_f1_direction": direction,
    }


def _build_val_paired_rows(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    runs_by_id: Mapping[str, Mapping[str, Any]],
    val_gold_by_event: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reference_events: set[str] | None = None
    for run_id in sorted(rows_by_run):
        selected = [
            row
            for row in rows_by_run[run_id]
            if row["comparison_type"] == "val_paired"
        ]
        by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in selected:
            key = (str(row["event_id"]), str(row["evidence_arm"]))
            if key in by_key:
                raise FineTuneAnalysisError(
                    f"{run_id}: duplicate val_paired arm {key}"
                )
            by_key[key] = row
        events = {event_id for event_id, _arm in by_key}
        if not events:
            raise FineTuneAnalysisError(f"{run_id}: no val_paired rows")
        if reference_events is None:
            reference_events = events
        elif events != reference_events:
            raise FineTuneAnalysisError(
                f"{run_id}: val_paired event universe differs"
            )
        run = runs_by_id[run_id]
        for event_id in sorted(events):
            gold_row = val_gold_by_event.get(event_id)
            if gold_row is None:
                raise FineTuneAnalysisError(
                    f"val_paired:{event_id}: absent from analysis-stage val gold"
                )
            arms: dict[str, Mapping[str, Any]] = {}
            for arm in TEST_ARMS:
                row = by_key.get((event_id, arm))
                if row is None:
                    raise FineTuneAnalysisError(
                        f"{run_id}:val_paired:{event_id}: missing {arm}"
                    )
                arms[arm] = row
            output.append(
                {
                    "run_id": run_id,
                    "backbone": str(run["backbone"]),
                    "assignment_id": str(run["assignment_id"]),
                    "seed": int(run["seed"]),
                    "event_id": event_id,
                    "gold_label": str(gold_row["gold_label"]),
                    "evitrace_logits": dict(arms["evitrace"]["logits"]),
                    "s4_logits": dict(arms["s4"]["logits"]),
                    "evitrace_raw_pred_label": str(
                        arms["evitrace"]["pred_label"]
                    ),
                    "s4_raw_pred_label": str(arms["s4"]["pred_label"]),
                }
            )
    return output


def compute_logit_adjustment_tau_grid(
    val_pair_rows: Sequence[Mapping[str, Any]],
    label_prior: Mapping[str, float],
    *,
    taus: Sequence[float] = DEFAULT_TAU_GRID,
) -> dict[str, Any]:
    """Val-only prior/logit-adjustment sensitivity.

    The adjustment follows the existing sign convention:
    ``adjusted_logits = raw_logits - tau * log(training_label_prior)``.
    """

    if not val_pair_rows:
        raise FineTuneAnalysisError("Tau grid requires val_paired rows")
    prior = {letter: float(label_prior[letter]) for letter in LETTERS}
    if any(value <= 0.0 or not math.isfinite(value) for value in prior.values()):
        raise FineTuneAnalysisError("Training label priors must be finite and positive")
    if abs(sum(prior.values()) - 1.0) > 1.0e-8:
        raise FineTuneAnalysisError("Training label priors must sum to one")
    log_prior = {letter: math.log(prior[letter]) for letter in LETTERS}
    grid: dict[str, Any] = {}
    for raw_tau in taus:
        tau = float(raw_tau)
        if tau < 0.0 or tau > 1.0:
            raise ValueError("tau must lie in [0, 1]")
        calibrated_rows: list[dict[str, Any]] = []
        for row in val_pair_rows:
            gold_label = str(row["gold_label"])
            gold_letter = LABEL_TO_LETTER[gold_label]
            arm_values: dict[str, tuple[str, float]] = {}
            for arm in TEST_ARMS:
                logits = {
                    letter: float(row[f"{arm}_logits"][letter])
                    - tau * log_prior[letter]
                    for letter in LETTERS
                }
                probabilities = _softmax(logits)
                pred_letter = max(
                    LETTERS, key=lambda letter: probabilities[letter]
                )
                arm_values[arm] = (
                    LETTER_TO_LABEL[pred_letter],
                    math.log(probabilities[gold_letter]),
                )
                if tau == 0.0 and arm_values[arm][0] != str(
                    row[f"{arm}_raw_pred_label"]
                ):
                    raise FineTuneAnalysisError(
                        "tau=0 prediction differs from frozen raw prediction"
                    )
            calibrated_rows.append(
                {
                    "run_id": str(row["run_id"]),
                    "backbone": str(row["backbone"]),
                    "assignment_id": str(row["assignment_id"]),
                    "seed": int(row["seed"]),
                    "event_id": str(row["event_id"]),
                    "gold_label": gold_label,
                    "evitrace_pred_label": arm_values["evitrace"][0],
                    "s4_pred_label": arm_values["s4"][0],
                    "evitrace_gold_logprob": arm_values["evitrace"][1],
                    "s4_gold_logprob": arm_values["s4"][1],
                }
            )
        key = f"{tau:g}"
        grid[key] = {
            "tau": tau,
            "claim_count": len(
                {str(row["event_id"]) for row in calibrated_rows}
            ),
            "point": hierarchical_point(calibrated_rows)["panel"],
        }
    return {
        "scope": "validation_only",
        "does_not_replace_raw_test_primary": True,
        "formula": "adjusted_logits = raw_logits - tau * log(label_prior)",
        "label_prior": prior,
        "log_label_prior": log_prior,
        "grid": grid,
    }


def _build_val_condition_rows(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    runs_by_id: Mapping[str, Mapping[str, Any]],
    val_gold_by_event: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    conditions: dict[str, list[dict[str, Any]]] = {
        "correct_evitrace": [],
        "correct_s4": [],
        "correct_pooled": [],
        "claim_only": [],
        "mismatched": [],
    }
    expected_events = set(val_gold_by_event)
    for run_id in sorted(rows_by_run):
        run = runs_by_id[run_id]
        indexes: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
        for comparison_type in (
            "val_paired",
            "val_claim_only",
            "val_mismatched",
        ):
            index: dict[tuple[str, str], Mapping[str, Any]] = {}
            for row in rows_by_run[run_id]:
                if row["comparison_type"] != comparison_type:
                    continue
                key = (str(row["event_id"]), str(row["evidence_arm"]))
                if key in index:
                    raise FineTuneAnalysisError(
                        f"{run_id}:{comparison_type}: duplicate {key}"
                    )
                index[key] = row
            indexes[comparison_type] = index
        observed_paired = {
            event_id
            for event_id, _arm in indexes["val_paired"]
        }
        observed_claim_only = {
            event_id
            for event_id, _arm in indexes["val_claim_only"]
        }
        observed_mismatched = {
            event_id
            for event_id, _arm in indexes["val_mismatched"]
        }
        if (
            observed_paired != expected_events
            or observed_claim_only != expected_events
            or observed_mismatched != expected_events
        ):
            raise FineTuneAnalysisError(
                f"{run_id}: validation condition event universe differs "
                "from val gold"
            )
        for event_id in sorted(expected_events):
            gold_label = str(val_gold_by_event[event_id]["gold_label"])
            gold_letter = LABEL_TO_LETTER[gold_label]
            source_rows = {
                "correct_evitrace": indexes["val_paired"].get(
                    (event_id, "evitrace")
                ),
                "correct_s4": indexes["val_paired"].get((event_id, "s4")),
                "claim_only": indexes["val_claim_only"].get(
                    (event_id, "claim_only")
                ),
                "mismatched": indexes["val_mismatched"].get(
                    (event_id, "mismatched")
                ),
            }
            if any(value is None for value in source_rows.values()):
                raise FineTuneAnalysisError(
                    f"{run_id}:{event_id}: incomplete validation conditions"
                )
            for condition, source in source_rows.items():
                assert source is not None
                item = {
                    "run_id": run_id,
                    "backbone": str(run["backbone"]),
                    "assignment_id": str(run["assignment_id"]),
                    "seed": int(run["seed"]),
                    "event_id": event_id,
                    "gold_label": gold_label,
                    "pred_label": str(source["pred_label"]),
                    "gold_logprob": float(
                        source["log_probs"][gold_letter]
                    ),
                    "probabilities": dict(source["probabilities"]),
                }
                conditions[condition].append(item)
                if condition in {"correct_evitrace", "correct_s4"}:
                    conditions["correct_pooled"].append(item)
    return conditions


def _ece(
    gold: Sequence[str],
    predicted: Sequence[str],
    confidences: Sequence[float],
    *,
    bins: int = 15,
) -> float:
    if not (len(gold) == len(predicted) == len(confidences)):
        raise FineTuneAnalysisError("ECE inputs have different lengths")
    total = len(gold)
    ece = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if confidence >= low
            and (
                confidence < high
                or (bin_index == bins - 1 and confidence <= high)
            )
        ]
        if not indices:
            continue
        accuracy = _mean(
            gold[index] == predicted[index] for index in indices
        )
        confidence = _mean(confidences[index] for index in indices)
        ece += len(indices) / total * abs(accuracy - confidence)
    return ece


def _classification_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise FineTuneAnalysisError("Classification metrics require rows")
    gold = [str(row["gold_label"]) for row in rows]
    predicted = [str(row["pred_label"]) for row in rows]
    confidences = [
        max(float(value) for value in row["probabilities"].values())
        for row in rows
    ]
    correct = [
        actual == guess for actual, guess in zip(gold, predicted)
    ]
    distribution_counts = {
        label: sum(guess == label for guess in predicted)
        for label in LIAR6_LABELS
    }
    per_class_recall: dict[str, Any] = {}
    for label in LIAR6_LABELS:
        support = sum(actual == label for actual in gold)
        recalled = sum(
            actual == label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        per_class_recall[label] = {
            "support": support,
            "recall": recalled / support if support else None,
        }
    return {
        "n": len(rows),
        "accuracy": statistics.fmean(correct),
        "macro_f1": _macro_f1(gold, predicted),
        "nll": -statistics.fmean(
            float(row["gold_logprob"]) for row in rows
        ),
        "ece_15_equal_width": _ece(gold, predicted, confidences),
        "predicted_label_distribution": {
            label: {
                "count": distribution_counts[label],
                "rate": distribution_counts[label] / len(rows),
            }
            for label in LIAR6_LABELS
        },
        "per_class_recall": per_class_recall,
    }


def _hierarchical_classification_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_backbone: dict[str, str] = {}
    for row in rows:
        run_id = str(row["run_id"])
        rows_by_run[run_id].append(row)
        run_backbone[run_id] = str(row["backbone"])
    per_run = {
        run_id: _classification_metrics(run_rows)
        for run_id, run_rows in sorted(rows_by_run.items())
    }
    scalar_names = ("accuracy", "macro_f1", "nll", "ece_15_equal_width")
    per_backbone: dict[str, Any] = {}
    for backbone in BACKBONES:
        run_ids = [
            run_id
            for run_id in sorted(per_run)
            if run_backbone[run_id] == backbone
        ]
        if not run_ids:
            raise FineTuneAnalysisError(
                f"Validation summary lacks backbone {backbone}"
            )
        per_backbone[backbone] = {
            **{
                name: _mean(per_run[run_id][name] for run_id in run_ids)
                for name in scalar_names
            },
            "predicted_label_distribution_rate": {
                label: _mean(
                    per_run[run_id]["predicted_label_distribution"][label][
                        "rate"
                    ]
                    for run_id in run_ids
                )
                for label in LIAR6_LABELS
            },
            "per_class_recall": {
                label: _mean(
                    per_run[run_id]["per_class_recall"][label]["recall"]
                    for run_id in run_ids
                )
                for label in LIAR6_LABELS
            },
        }
    panel = {
        **{
            name: _mean(
                per_backbone[backbone][name] for backbone in BACKBONES
            )
            for name in scalar_names
        },
        "predicted_label_distribution_rate": {
            label: _mean(
                per_backbone[backbone][
                    "predicted_label_distribution_rate"
                ][label]
                for backbone in BACKBONES
            )
            for label in LIAR6_LABELS
        },
        "per_class_recall": {
            label: _mean(
                per_backbone[backbone]["per_class_recall"][label]
                for backbone in BACKBONES
            )
            for label in LIAR6_LABELS
        },
    }
    pooled = _classification_metrics(rows)
    return {
        "aggregation": "run -> backbone -> equal-weight two-backbone panel",
        "per_run": per_run,
        "per_backbone": per_backbone,
        "panel": panel,
        "pooled_repeated_run_rows_descriptive": pooled,
    }


def _hierarchical_paired_scalar_summary(
    values_by_run: Mapping[str, Sequence[float]],
    run_backbone: Mapping[str, str],
) -> dict[str, Any]:
    per_run = {
        run_id: {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }
        for run_id, values in sorted(values_by_run.items())
    }
    per_backbone: dict[str, Any] = {}
    for backbone in BACKBONES:
        run_ids = [
            run_id
            for run_id in sorted(per_run)
            if run_backbone[run_id] == backbone
        ]
        per_backbone[backbone] = {
            "mean": _mean(per_run[run_id]["mean"] for run_id in run_ids),
            "median": _mean(per_run[run_id]["median"] for run_id in run_ids),
        }
    return {
        "per_run": per_run,
        "per_backbone": per_backbone,
        "panel": {
            name: _mean(
                per_backbone[backbone][name] for backbone in BACKBONES
            )
            for name in ("mean", "median")
        },
    }


def _validation_gold_logprob_contrasts(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    indexes: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    run_backbone: dict[str, str] = {}
    for condition, rows in condition_rows.items():
        if condition == "correct_pooled":
            continue
        indexes[condition] = {
            (str(row["run_id"]), str(row["event_id"])): row for row in rows
        }
        for row in rows:
            run_backbone[str(row["run_id"])] = str(row["backbone"])
    keys = set(indexes["claim_only"])
    if any(set(index) != keys for index in indexes.values()):
        raise FineTuneAnalysisError(
            "Validation gold-logp contrast universes differ"
        )
    output: dict[str, Any] = {}
    for correct_name in ("correct_evitrace", "correct_s4"):
        for diagnostic in ("claim_only", "mismatched"):
            values_by_run: dict[str, list[float]] = defaultdict(list)
            for run_id, event_id in sorted(keys):
                values_by_run[run_id].append(
                    float(indexes[correct_name][(run_id, event_id)][
                        "gold_logprob"
                    ])
                    - float(indexes[diagnostic][(run_id, event_id)][
                        "gold_logprob"
                    ])
                )
            output[f"{correct_name}_minus_{diagnostic}"] = (
                _hierarchical_paired_scalar_summary(
                    values_by_run, run_backbone
                )
            )
    for diagnostic in ("claim_only", "mismatched"):
        values_by_run = defaultdict(list)
        for run_id, event_id in sorted(keys):
            correct_mean = statistics.fmean(
                [
                    float(indexes["correct_evitrace"][(run_id, event_id)][
                        "gold_logprob"
                    ]),
                    float(indexes["correct_s4"][(run_id, event_id)][
                        "gold_logprob"
                    ]),
                ]
            )
            values_by_run[run_id].append(
                correct_mean
                - float(indexes[diagnostic][(run_id, event_id)][
                    "gold_logprob"
                ])
            )
        output[f"correct_pooled_minus_{diagnostic}"] = (
            _hierarchical_paired_scalar_summary(
                values_by_run, run_backbone
            )
        )
    return output


def summarize_validation_conditions(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "conditions": {
            condition: _hierarchical_classification_summary(rows)
            for condition, rows in condition_rows.items()
        },
        "metrics": (
            "predicted label distribution, per-class recall, Macro-F1, "
            "accuracy, NLL, and 15-bin equal-width ECE"
        ),
        "gold_logprob_contrasts": _validation_gold_logprob_contrasts(
            condition_rows
        ),
    }


def _prefix_curve_summary(
    paired_positions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(paired_positions, key=lambda row: int(row["k"]))
    if not ordered:
        raise FineTuneAnalysisError("Empty prefix curve")
    k_visible = int(ordered[0]["k_visible"])
    observed_positions = [int(row["k"]) for row in ordered]
    if observed_positions != list(range(1, k_visible + 1)):
        raise FineTuneAnalysisError(
            f"Incomplete prefix curve: {observed_positions} versus 1..{k_visible}"
        )
    gold_label = str(ordered[0]["gold_label"])
    output: dict[str, Any] = {
        "k_visible": k_visible,
        "positions": observed_positions,
        "prefix_relation_counts": {
            relation: sum(
                str(row["prefix_relation"]) == relation for row in ordered
            )
            for relation in sorted(PREFIX_RELATIONS)
        },
        "token_equal_at_every_position": all(
            int(row["evitrace_token_count"]) == int(row["s4_token_count"])
            for row in ordered
        ),
    }
    for arm in TEST_ARMS:
        correct = [
            str(row[f"{arm}_pred_label"]) == gold_label for row in ordered
        ]
        logprob = [
            float(row[f"{arm}_gold_logprob"]) for row in ordered
        ]
        first_correct = next(
            (index for index, value in enumerate(correct) if value),
            None,
        )
        stable_after_first = (
            first_correct is not None and all(correct[first_correct:])
        )
        stable_onset = next(
            (
                index
                for index in range(len(correct))
                if all(correct[index:])
            ),
            None,
        )
        stable_coverage = (
            (len(correct) - stable_onset) / len(correct)
            if stable_onset is not None
            else 0.0
        )
        output[arm] = {
            # This is a normalized discrete prefix AUC: equal weight 1/K for
            # every evidence-count position k=1,...,K.
            "normalized_gold_logprob_auc": statistics.fmean(logprob),
            "normalized_accuracy_auc": statistics.fmean(correct),
            "stable_correct": stable_after_first,
            "stable_correct_coverage": stable_coverage,
            "earliest_stable_correct_k": (
                stable_onset + 1 if stable_onset is not None else None
            ),
            "all_prefixes_correct": all(correct),
            "final_correct": correct[-1],
        }
    output["delta"] = {
        "normalized_gold_logprob_auc": (
            output["evitrace"]["normalized_gold_logprob_auc"]
            - output["s4"]["normalized_gold_logprob_auc"]
        ),
        "normalized_accuracy_auc": (
            output["evitrace"]["normalized_accuracy_auc"]
            - output["s4"]["normalized_accuracy_auc"]
        ),
        "stable_correct": float(output["evitrace"]["stable_correct"])
        - float(output["s4"]["stable_correct"]),
        "stable_correct_coverage": (
            output["evitrace"]["stable_correct_coverage"]
            - output["s4"]["stable_correct_coverage"]
        ),
    }
    return output


def _build_prefix_curves(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    runs_by_id: Mapping[str, Mapping[str, Any]],
    gold_by_event: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    all_positions: list[dict[str, Any]] = []
    reference_keys: set[tuple[str, int, str]] | None = None
    for run_id in sorted(rows_by_run):
        prefix_rows = [
            row
            for row in rows_by_run[run_id]
            if row["comparison_type"] == "prefix"
        ]
        by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        for row in prefix_rows:
            event_id = str(row["event_id"])
            k_value = int(row["k"])
            arm = str(row["evidence_arm"])
            key = (event_id, k_value, arm)
            if key in by_key:
                raise FineTuneAnalysisError(
                    f"{run_id}: duplicate prefix logical row {key}"
                )
            by_key[key] = row
        paired_keys = {
            (event_id, k_value)
            for event_id, k_value, _arm in by_key
        }
        if not paired_keys:
            raise FineTuneAnalysisError(f"{run_id}: no prefix test rows")
        logical_keys = {
            (
                event_id,
                k_value,
                str(row["prefix_relation"]),
            )
            for (event_id, k_value, _arm), row in by_key.items()
        }
        if reference_keys is None:
            reference_keys = logical_keys
        elif logical_keys != reference_keys:
            raise FineTuneAnalysisError(
                f"{run_id}: prefix position universe differs across runs"
            )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        run = runs_by_id[run_id]
        for event_id, k_value in sorted(paired_keys):
            if event_id not in gold_by_event:
                raise FineTuneAnalysisError(
                    f"prefix:{event_id}: absent from independent test gold"
                )
            arms: dict[str, Mapping[str, Any]] = {}
            for arm in TEST_ARMS:
                row = by_key.get((event_id, k_value, arm))
                if row is None:
                    raise FineTuneAnalysisError(
                        f"{run_id}:{event_id}:k={k_value} missing {arm}"
                    )
                arms[arm] = row
            evi = arms["evitrace"]
            s4 = arms["s4"]
            if evi["prefix_relation"] != s4["prefix_relation"]:
                raise FineTuneAnalysisError(
                    f"{run_id}:{event_id}:k={k_value}: relation differs by arm"
                )
            relation = str(evi["prefix_relation"])
            if int(evi["k_visible"]) != int(s4["k_visible"]):
                raise FineTuneAnalysisError(
                    f"{run_id}:{event_id}:{relation}: unmatched prefix horizon"
                )
            gold_label = str(gold_by_event[event_id]["gold_label"])
            gold_letter = LABEL_TO_LETTER[gold_label]
            position = {
                "run_id": run_id,
                "backbone": str(run["backbone"]),
                "assignment_id": str(run["assignment_id"]),
                "seed": int(run["seed"]),
                "event_id": event_id,
                "prefix_relation": relation,
                "k": k_value,
                "k_visible": int(evi["k_visible"]),
                "gold_label": gold_label,
                "evitrace_pred_label": str(evi["pred_label"]),
                "s4_pred_label": str(s4["pred_label"]),
                "evitrace_gold_logprob": float(
                    evi["log_probs"][gold_letter]
                ),
                "s4_gold_logprob": float(s4["log_probs"][gold_letter]),
                "evitrace_token_count": int(evi["token_count"]),
                "s4_token_count": int(s4["token_count"]),
            }
            grouped[event_id].append(position)
            all_positions.append(position)
        for event_id, positions in sorted(grouped.items()):
            summary = _prefix_curve_summary(positions)
            curves.append(
                {
                    "run_id": run_id,
                    "backbone": str(run["backbone"]),
                    "assignment_id": str(run["assignment_id"]),
                    "seed": int(run["seed"]),
                    "event_id": event_id,
                    "gold_label": str(
                        gold_by_event[event_id]["gold_label"]
                    ),
                    **summary,
                }
            )
    return curves, all_positions


_PREFIX_METRICS = (
    "normalized_gold_logprob_auc",
    "normalized_accuracy_auc",
    "stable_correct",
    "stable_correct_coverage",
)


def _prefix_linear_point(
    curves: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not curves:
        raise FineTuneAnalysisError("Prefix summary requires curves")
    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_backbone: dict[str, str] = {}
    for row in curves:
        run_id = str(row["run_id"])
        rows_by_run[run_id].append(row)
        run_backbone[run_id] = str(row["backbone"])
    per_run: dict[str, Any] = {}
    for run_id, rows in sorted(rows_by_run.items()):
        per_run[run_id] = {
            "n_curves": len(rows),
            "evitrace": {
                metric: _mean(
                    float(row["evitrace"][metric]) for row in rows
                )
                for metric in _PREFIX_METRICS
            },
            "s4": {
                metric: _mean(float(row["s4"][metric]) for row in rows)
                for metric in _PREFIX_METRICS
            },
            "delta": {
                metric: _mean(float(row["delta"][metric]) for row in rows)
                for metric in _PREFIX_METRICS
            },
        }
    per_backbone: dict[str, Any] = {}
    for backbone in BACKBONES:
        selected_ids = [
            run_id
            for run_id in sorted(per_run)
            if run_backbone[run_id] == backbone
        ]
        if not selected_ids:
            raise FineTuneAnalysisError(
                f"Prefix summary lacks backbone {backbone}"
            )
        per_backbone[backbone] = {
            arm: {
                metric: _mean(
                    per_run[run_id][arm][metric]
                    for run_id in selected_ids
                )
                for metric in _PREFIX_METRICS
            }
            for arm in ("evitrace", "s4", "delta")
        }
    panel = {
        arm: {
            metric: _mean(
                per_backbone[backbone][arm][metric]
                for backbone in BACKBONES
            )
            for metric in _PREFIX_METRICS
        }
        for arm in ("evitrace", "s4", "delta")
    }
    return {
        "curve_count": len(curves),
        "claim_count": len({str(row["event_id"]) for row in curves}),
        "token_equal_at_every_position_curve_count": sum(
            bool(row["token_equal_at_every_position"]) for row in curves
        ),
        "per_run": per_run,
        "per_backbone": per_backbone,
        "panel": panel,
    }


def _prefix_bootstrap(
    curves: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not curves:
        raise FineTuneAnalysisError("Prefix bootstrap requires curves")
    run_ids = sorted({str(row["run_id"]) for row in curves})
    event_ids = sorted({str(row["event_id"]) for row in curves})
    backbones_by_run: dict[str, str] = {}
    labels_by_event: dict[str, str] = {}
    event_run_rows: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in curves:
        run_id = str(row["run_id"])
        event_id = str(row["event_id"])
        backbones_by_run[run_id] = str(row["backbone"])
        labels_by_event[event_id] = str(row["gold_label"])
        event_run_rows[(event_id, run_id)].append(row)
    if any(
        (event_id, run_id) not in event_run_rows
        for event_id in event_ids
        for run_id in run_ids
    ):
        raise FineTuneAnalysisError(
            "Prefix bootstrap requires every claim in every run"
        )
    gold = np.asarray(
        [LABEL_TO_ID[labels_by_event[event_id]] for event_id in event_ids],
        dtype=np.int8,
    )
    matrices: dict[str, np.ndarray] = {}
    for metric in _PREFIX_METRICS:
        matrices[metric] = np.asarray(
            [
                [
                    _mean(
                        float(row["delta"][metric])
                        for row in event_run_rows[(event_id, run_id)]
                    )
                    for event_id in event_ids
                ]
                for run_id in run_ids
            ],
            dtype=np.float64,
        )
    backbones = [backbones_by_run[run_id] for run_id in run_ids]
    distributions = {metric: [] for metric in _PREFIX_METRICS}
    rng = np.random.default_rng(int(seed))
    for weights in _stratified_weight_chunks(
        gold,
        iterations=int(iterations),
        rng=rng,
    ):
        for metric, matrix in matrices.items():
            run_values = weights @ matrix.T / float(len(event_ids))
            values = _hierarchical_array_mean(run_values, backbones)
            distributions[metric].extend(values.tolist())
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "claim_count": len(event_ids),
        "ci95_delta": {
            metric: _percentile_interval(values)
            for metric, values in distributions.items()
        },
        "cluster": "event_id; all run/relation curve rows retained together",
    }


_POSITION_METRICS = ("gold_logprob", "accuracy")


def _position_values(row: Mapping[str, Any], arm: str) -> dict[str, float]:
    return {
        "gold_logprob": float(row[f"{arm}_gold_logprob"]),
        "accuracy": float(
            str(row[f"{arm}_pred_label"]) == str(row["gold_label"])
        ),
    }


def _position_linear_point(
    positions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not positions:
        raise FineTuneAnalysisError("Positional summary requires paired positions")
    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_backbone: dict[str, str] = {}
    for row in positions:
        run_id = str(row["run_id"])
        rows_by_run[run_id].append(row)
        run_backbone[run_id] = str(row["backbone"])
    per_run: dict[str, Any] = {}
    for run_id, rows in sorted(rows_by_run.items()):
        arm_metrics = {
            arm: {
                metric: _mean(
                    _position_values(row, arm)[metric] for row in rows
                )
                for metric in _POSITION_METRICS
            }
            for arm in TEST_ARMS
        }
        wins = sum(
            _position_values(row, "evitrace")["accuracy"]
            > _position_values(row, "s4")["accuracy"]
            for row in rows
        )
        losses = sum(
            _position_values(row, "s4")["accuracy"]
            > _position_values(row, "evitrace")["accuracy"]
            for row in rows
        )
        per_run[run_id] = {
            "paired_position_count": len(rows),
            **arm_metrics,
            "delta": {
                metric: arm_metrics["evitrace"][metric]
                - arm_metrics["s4"][metric]
                for metric in _POSITION_METRICS
            },
            "wlt": {
                "evitrace_win": wins,
                "s4_win": losses,
                "tie": len(rows) - wins - losses,
            },
        }
    per_backbone: dict[str, Any] = {}
    for backbone in BACKBONES:
        run_ids = [
            run_id
            for run_id in sorted(per_run)
            if run_backbone[run_id] == backbone
        ]
        if not run_ids:
            raise FineTuneAnalysisError(
                f"Positional summary lacks backbone {backbone}"
            )
        per_backbone[backbone] = {
            arm: {
                metric: _mean(
                    per_run[run_id][arm][metric] for run_id in run_ids
                )
                for metric in _POSITION_METRICS
            }
            for arm in ("evitrace", "s4", "delta")
        }
    panel = {
        arm: {
            metric: _mean(
                per_backbone[backbone][arm][metric]
                for backbone in BACKBONES
            )
            for metric in _POSITION_METRICS
        }
        for arm in ("evitrace", "s4", "delta")
    }
    return {
        "paired_position_count_unique": len(
            {
                (str(row["event_id"]), int(row["k"]))
                for row in positions
            }
        ),
        "paired_position_run_rows": len(positions),
        "claim_count": len({str(row["event_id"]) for row in positions}),
        "token_equal_position_run_rows": sum(
            int(row["evitrace_token_count"]) == int(row["s4_token_count"])
            for row in positions
        ),
        "per_run": per_run,
        "per_backbone": per_backbone,
        "panel": panel,
    }


def _position_bootstrap(
    positions: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    run_ids = sorted({str(row["run_id"]) for row in positions})
    event_ids = sorted({str(row["event_id"]) for row in positions})
    run_backbone: dict[str, str] = {}
    labels_by_event: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in positions:
        run_id = str(row["run_id"])
        event_id = str(row["event_id"])
        run_backbone[run_id] = str(row["backbone"])
        labels_by_event[event_id] = str(row["gold_label"])
        grouped[(run_id, event_id)].append(row)
    if any(
        (run_id, event_id) not in grouped
        for run_id in run_ids
        for event_id in event_ids
    ):
        raise FineTuneAnalysisError(
            "Positional bootstrap requires every selected claim in every run"
        )
    counts = np.asarray(
        [
            [len(grouped[(run_id, event_id)]) for event_id in event_ids]
            for run_id in run_ids
        ],
        dtype=np.float64,
    )
    delta_sums: dict[str, np.ndarray] = {}
    for metric in _POSITION_METRICS:
        delta_sums[metric] = np.asarray(
            [
                [
                    sum(
                        _position_values(row, "evitrace")[metric]
                        - _position_values(row, "s4")[metric]
                        for row in grouped[(run_id, event_id)]
                    )
                    for event_id in event_ids
                ]
                for run_id in run_ids
            ],
            dtype=np.float64,
        )
    gold = np.asarray(
        [LABEL_TO_ID[labels_by_event[event_id]] for event_id in event_ids],
        dtype=np.int8,
    )
    backbones = [run_backbone[run_id] for run_id in run_ids]
    distributions = {metric: [] for metric in _POSITION_METRICS}
    rng = np.random.default_rng(int(seed))
    for weights in _stratified_weight_chunks(
        gold,
        iterations=int(iterations),
        rng=rng,
    ):
        denominators = weights @ counts.T
        if np.any(denominators <= 0):
            raise FineTuneAnalysisError(
                "Positional bootstrap produced an empty run denominator"
            )
        for metric, sums in delta_sums.items():
            run_values = (weights @ sums.T) / denominators
            values = _hierarchical_array_mean(run_values, backbones)
            distributions[metric].extend(values.tolist())
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "claim_count": len(event_ids),
        "cluster": "event_id; all selected k positions and 12 runs retained",
        "ci95_delta": {
            metric: _percentile_interval(values)
            for metric, values in distributions.items()
        },
    }


def summarize_prefix_curves(
    curves: Sequence[Mapping[str, Any]],
    paired_positions: Sequence[Mapping[str, Any]],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Summarize full curves plus relation-specific paired positions."""

    overall = _prefix_linear_point(curves)
    overall["bootstrap"] = _prefix_bootstrap(
        curves, iterations=int(bootstrap), seed=int(seed)
    )
    by_relation: dict[str, Any] = {}
    for offset, relation in enumerate(sorted(PREFIX_RELATIONS)):
        subset = [
            row
            for row in paired_positions
            if row["prefix_relation"] == relation
        ]
        if subset:
            relation_result = _position_linear_point(subset)
            relation_result["bootstrap"] = _position_bootstrap(
                subset,
                iterations=int(bootstrap),
                seed=int(seed) + 37 * (offset + 1),
            )
            by_relation[relation] = relation_result
    strict = [
        row
        for row in paired_positions
        if row["prefix_relation"] == "same_set_different_order"
    ]
    if not strict:
        raise FineTuneAnalysisError(
            "No strict same-set-different-order paired prefix positions"
        )
    strict_result = _position_linear_point(strict)
    strict_result["bootstrap"] = _position_bootstrap(
        strict,
        iterations=int(bootstrap),
        seed=int(seed) + 401,
    )
    strict_result["definition"] = (
        "Individual (event_id,k) pairs whose prefix_relation is "
        "same_set_different_order. A claim need not have that relation at "
        "every prefix position to contribute its strict positions."
    )
    strict_result["token_count_is_not_a_membership_criterion"] = True
    return {
        "auc_definition": (
            "normalized discrete AUC = K^{-1} sum_{k=1}^K metric(k); "
            "each evidence-count position has equal weight"
        ),
        "stable_correct_definition": (
            "the prediction is correct at its first correct prefix and remains "
            "correct at every later prefix"
        ),
        "overall": overall,
        "paired_positions_by_prefix_relation": by_relation,
        "strict_positional_subset": strict_result,
    }


def _comparison_analysis(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    point = hierarchical_point(pair_rows)
    bootstrap_result = stratified_claim_bootstrap(
        pair_rows, iterations=bootstrap, seed=seed
    )
    randomization = shared_claim_swap_randomization(
        pair_rows, iterations=permutations, seed=seed + 1
    )
    macro_ci = bootstrap_result["ci95"]["macro_f1_delta"]
    return {
        "claim_count": len({str(row["event_id"]) for row in pair_rows}),
        "point": point,
        "bootstrap": bootstrap_result,
        "randomization": randomization,
        "sesoi": assess_sesoi(
            point["panel"]["delta"]["macro_f1"], macro_ci
        ),
    }


def _label_strata(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in LIAR6_LABELS:
        events = {
            str(row["event_id"])
            for row in pair_rows
            if row["gold_label"] == label
        }
        if not events:
            continue
        subset = [
            row for row in pair_rows if str(row["event_id"]) in events
        ]
        point = hierarchical_point(subset)["panel"]
        result[label] = {
            "claim_count": len(events),
            "accuracy_delta": point["delta"]["accuracy"],
            "gold_logprob_delta": point["delta"]["gold_logprob_mean"],
            "wlt_rates": point["wlt_rates"],
        }
    return result


def _gold_logprob_delta_summary(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    include_per_run: bool = True,
) -> dict[str, Any]:
    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    run_backbone: dict[str, str] = {}
    for row in pair_rows:
        run_id = str(row["run_id"])
        rows_by_run[run_id].append(row)
        run_backbone[run_id] = str(row["backbone"])
    per_run: dict[str, Any] = {}
    for run_id, rows in sorted(rows_by_run.items()):
        deltas = [
            float(row["evitrace_gold_logprob"])
            - float(row["s4_gold_logprob"])
            for row in rows
        ]
        per_run[run_id] = {
            "n": len(deltas),
            "mean": statistics.fmean(deltas),
            "median": statistics.median(deltas),
            "positive": sum(value > 1.0e-12 for value in deltas),
            "negative": sum(value < -1.0e-12 for value in deltas),
            "tie": sum(abs(value) <= 1.0e-12 for value in deltas),
            "positive_rate": _mean(value > 1.0e-12 for value in deltas),
            "negative_rate": _mean(value < -1.0e-12 for value in deltas),
            "tie_rate": _mean(abs(value) <= 1.0e-12 for value in deltas),
        }
    per_backbone: dict[str, Any] = {}
    scalar_names = (
        "mean",
        "median",
        "positive_rate",
        "negative_rate",
        "tie_rate",
    )
    for backbone in BACKBONES:
        run_ids = [
            run_id
            for run_id in sorted(per_run)
            if run_backbone[run_id] == backbone
        ]
        if not run_ids:
            raise FineTuneAnalysisError(
                f"Gold-logp summary lacks backbone {backbone}"
            )
        per_backbone[backbone] = {
            name: _mean(per_run[run_id][name] for run_id in run_ids)
            for name in scalar_names
        }
    panel = {
        name: _mean(
            per_backbone[backbone][name] for backbone in BACKBONES
        )
        for name in scalar_names
    }
    pooled = [
        float(row["evitrace_gold_logprob"])
        - float(row["s4_gold_logprob"])
        for row in pair_rows
    ]
    output = {
        "panel": panel,
        "per_backbone": per_backbone,
        "pooled_repeated_run_rows_descriptive": {
            "n": len(pooled),
            "mean": statistics.fmean(pooled),
            "median": statistics.median(pooled),
            "positive": sum(value > 1.0e-12 for value in pooled),
            "negative": sum(value < -1.0e-12 for value in pooled),
            "tie": sum(abs(value) <= 1.0e-12 for value in pooled),
        },
    }
    if include_per_run:
        output["per_run"] = per_run
    return output


def _gold_logprob_label_complexity_strata(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in LIAR6_LABELS:
        for complexity in ("single", "multi"):
            subset = [
                row
                for row in pair_rows
                if row["gold_label"] == label
                and row["complexity"] == complexity
            ]
            if not subset:
                continue
            key = f"{label}:{complexity}"
            result[key] = {
                "gold_label": label,
                "complexity": complexity,
                "claim_count": len(
                    {str(row["event_id"]) for row in subset}
                ),
                **_gold_logprob_delta_summary(
                    subset,
                    include_per_run=False,
                ),
            }
    return result


def _interpretation(
    primary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    main_category = str(primary["main"]["sesoi"]["category"])
    order_category = str(primary["order_only"]["sesoi"]["category"])
    main_p = float(
        primary["main"]["randomization"]["holm_pvalue"]
    )
    order_p = float(
        primary["order_only"]["randomization"]["holm_pvalue"]
    )
    main_positive = (
        main_category == "beneficial_beyond_sesoi" and main_p < 0.05
    )
    order_positive = (
        order_category == "beneficial_beyond_sesoi" and order_p < 0.05
    )
    if main_positive and order_positive:
        category = "main_and_order_positive_beyond_sesoi"
        wording = (
            "Across the two fine-tuned verifier backbones, EviTrace improves "
            "evidence selection/organization and decision-oriented ordering."
        )
    elif main_positive:
        category = "main_positive_order_not_confirmed"
        wording = (
            "Across the two fine-tuned verifier backbones, EviTrace improves "
            "evidence selection and overall organization; an ordering-specific "
            "benefit is not established."
        )
    else:
        category = "mixed_or_uncertain"
        wording = (
            "The fine-tuned cross-verifier experiment does not establish a "
            "practically meaningful EviTrace advantage."
        )
    return {
        "category": category,
        "paper_wording": wording,
        "main_holm_pvalue": main_p,
        "order_holm_pvalue": order_p,
        "prohibited_claims": (
            "This is model-based downstream utility, not human alignment, "
            "human fact-checking accuracy, causal explanation, or latent "
            "chain-of-thought validation."
        ),
    }


def _markdown_report(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# EviTrace Cross-Verifier Fine-Tuning Evaluation",
        "",
        "Twelve independently fine-tuned runs form a frozen "
        "`2 backbones × 2 assignments × 3 seeds` panel. Gold labels are joined "
        "only in this analysis. Claims—not repeated run rows—are the "
        "inferential unit.",
        "",
        "## Primary results",
        "",
        "| Comparison | Evi Macro-F1 | S4 Macro-F1 | ΔMacro-F1 | 95% CI | "
        "shared-swap p | Holm p | SESOI decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for comparison_type in ("main", "order_only"):
        item = metrics["primary"][comparison_type]
        panel = item["point"]["panel"]
        interval = item["bootstrap"]["ci95"]["macro_f1_delta"]
        randomization = item["randomization"]
        lines.append(
            "| {comparison} | {evi:.4f} | {s4:.4f} | {delta:+.4f} | "
            "[{low:+.4f}, {high:+.4f}] | {p:.6g} | {holm:.6g} | {sesoi} |".format(
                comparison=comparison_type,
                evi=panel["evitrace"]["macro_f1"],
                s4=panel["s4"]["macro_f1"],
                delta=panel["delta"]["macro_f1"],
                low=interval[0],
                high=interval[1],
                p=randomization["two_sided_pvalue"],
                holm=randomization["holm_pvalue"],
                sesoi=item["sesoi"]["category"],
            )
        )
    lines.extend(
        [
            "",
            "The bootstrap is stratified by the independent test gold label. "
            "Each resampled claim carries all 12 run rows. The randomization "
            "draws one arm-swap bit per claim and shares it across all runs. "
            "The smallest effect size of interest is ±0.01 Macro-F1.",
            "",
            "## Secondary paired results",
            "",
            "| Comparison | Evi Acc. | S4 Acc. | ΔAcc. | W/L/T (pooled, "
            "descriptive) | Conditional Evi win | Δ gold log p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison_type in ("main", "order_only"):
        panel = metrics["primary"][comparison_type]["point"]["panel"]
        pooled = panel["wlt_pooled_descriptive"]
        lines.append(
            "| {comparison} | {evi:.4f} | {s4:.4f} | {delta:+.4f} | "
            "{w}/{l}/{t} | {conditional:.4f} | {logp:+.4f} |".format(
                comparison=comparison_type,
                evi=panel["evitrace"]["accuracy"],
                s4=panel["s4"]["accuracy"],
                delta=panel["delta"]["accuracy"],
                w=pooled["evitrace_win"],
                l=pooled["s4_win"],
                t=pooled["tie"],
                conditional=panel["wlt_rates"][
                    "conditional_evitrace_win_rate"
                ],
                logp=panel["delta"]["gold_logprob_mean"],
            )
        )
    lines.extend(
        [
            "",
            "Run-specific exact McNemar tests and backbone/assignment/seed "
            "heterogeneity are retained in `metrics.json`. The pooled W/L/T "
            "rows repeat each claim 12 times and are descriptive only.",
            "",
            "### Gold-label log-probability deltas",
            "",
            "| Comparison | Mean | Median | Positive / negative / tie rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison_type, key in (
        ("main", "main_gold_logprob"),
        ("order_only", "order_only_gold_logprob"),
    ):
        panel = metrics["secondary"][key]["overall"]["panel"]
        lines.append(
            f"| {comparison_type} | {panel['mean']:+.4f} | "
            f"{panel['median']:+.4f} | {panel['positive_rate']:.3f} / "
            f"{panel['negative_rate']:.3f} / {panel['tie_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The full 12 `gold label × single/multi` strata for Main and "
            "Order are retained in `metrics.json`.",
            "",
            "### Exact train-selected snippet sensitivity",
            "",
        ]
    )
    for comparison_type in ("main", "order_only"):
        item = metrics["secondary"][
            "exact_train_selected_snippet_sensitivity"
        ][comparison_type]
        retained_counts = [
            value["retained_claim_count"]
            for value in item["claim_counts_by_run"].values()
        ]
        lines.append(
            f"- {comparison_type}: retained {min(retained_counts)}–"
            f"{max(retained_counts)} claims per run; panel ΔMacro-F1 "
            f"{item['point']['panel']['delta']['macro_f1']:+.4f} "
            f"({item['panel_macro_f1_direction']})."
        )
    lines.extend(
        [
            "",
            "## Validation diagnostics",
            "",
            "| Condition | Macro-F1 | Accuracy | NLL | ECE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    val_conditions = metrics["secondary"]["validation_diagnostics"][
        "conditions"
    ]
    for condition in (
        "correct_evitrace",
        "correct_s4",
        "correct_pooled",
        "claim_only",
        "mismatched",
    ):
        panel = val_conditions[condition]["panel"]
        lines.append(
            f"| {condition} | {panel['macro_f1']:.4f} | "
            f"{panel['accuracy']:.4f} | {panel['nll']:.4f} | "
            f"{panel['ece_15_equal_width']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Predicted-class distributions, per-class recall, and paired "
            "correct-evidence minus claim-only/mismatched gold-logp contrasts "
            "are retained in `metrics.json`.",
            "",
            "## Prefix utility",
            "",
            "Prefix AUC is the equally weighted discrete mean over evidence-count "
            "positions `k=1,…,K`; it is not a token-time integral.",
            "",
            "Count correction: prefix evaluation uses the 1,152 final-order "
            "eligible claims and 6,996 `(event_id,k)` positions. The earlier "
            "7,448 figure covered all 1,250 claims and included 452 positions "
            "from the 98 claims whose final orders are identical; those claims "
            "are not in this prefix comparison.",
            "",
            "| Population | Δ normalized gold-logp AUC | "
            "Δ normalized accuracy AUC | Δ stable-correct |",
            "|---|---:|---:|---:|",
        ]
    )
    prefix = metrics["prefix"]
    curve_delta = prefix["overall"]["panel"]["delta"]
    lines.append(
        "| all complete curves | "
        f"{curve_delta['normalized_gold_logprob_auc']:+.4f} | "
        f"{curve_delta['normalized_accuracy_auc']:+.4f} | "
        f"{curve_delta['stable_correct']:+.4f} |"
    )
    strict = prefix["strict_positional_subset"]
    strict_delta = strict["panel"]["delta"]
    lines.extend(
        [
            "",
            "The strict positional analysis is defined per `(event_id,k)` "
            "pair, not by requiring a whole claim curve to be strict.",
            "",
            "| Strict same-set position pairs | Δ gold log p | Δ accuracy |",
            "|---:|---:|---:|",
            f"| {strict['paired_position_count_unique']} | "
            f"{strict_delta['gold_logprob']:+.4f} | "
            f"{strict_delta['accuracy']:+.4f} |",
            "",
            "## Main token robustness",
            "",
        ]
    )
    token = metrics["secondary"]["main_token_sensitivity_abs_le_64"]
    token_point = token["intersection_point"]
    if token_point is None:
        lines.append("- The two-tokenizer |Δtokens|≤64 intersection is empty.")
    else:
        lines.append(
            f"- Two-tokenizer intersection: "
            f"{token['intersection_claim_count']} claims; "
            f"ΔMacro-F1={token_point['delta']['macro_f1']:+.4f}, "
            f"ΔAccuracy={token_point['delta']['accuracy']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Validation-only prior adjustment",
            "",
            "| τ | Val claims | ΔMacro-F1 | ΔAccuracy |",
            "|---:|---:|---:|---:|",
        ]
    )
    tau_grid = metrics["secondary"]["val_logit_adjustment_tau_grid"]["grid"]
    for tau, item in tau_grid.items():
        point = item["point"]
        lines.append(
            f"| {tau} | {item['claim_count']} | "
            f"{point['delta']['macro_f1']:+.4f} | "
            f"{point['delta']['accuracy']:+.4f} |"
        )
    lines.append(
        "\nThis τ grid is a validation-only calibration sensitivity and does "
        "not replace the raw-logit test primary result."
    )
    lines.extend(
        [
            "",
            "## Locked interpretation",
            "",
            f"- Category: `{metrics['interpretation']['category']}`",
            f"- {metrics['interpretation']['paper_wording']}",
            f"- {metrics['interpretation']['prohibited_claims']}",
            "",
        ]
    )
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _paper_table(metrics: Mapping[str, Any]) -> str:
    lines = [
        "% Auto-generated by cross_verifier_finetune_analysis.py.",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Comparison & Evi Macro-F1 & S4 Macro-F1 & $\\Delta$Macro-F1 "
        "& Holm $p$ & SESOI \\\\",
        "\\midrule",
    ]
    for comparison_type in ("main", "order_only"):
        item = metrics["primary"][comparison_type]
        panel = item["point"]["panel"]
        lines.append(
            f"{_latex_escape(comparison_type)} & "
            f"{panel['evitrace']['macro_f1']:.3f} & "
            f"{panel['s4']['macro_f1']:.3f} & "
            f"{panel['delta']['macro_f1']:+.3f} & "
            f"{item['randomization']['holm_pvalue']:.3g} & "
            f"{_latex_escape(item['sesoi']['category'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "% Equal-weight hierarchy: run -> backbone -> two-backbone panel.",
            "% 95% CIs and prefix/token sensitivity are in the appendix report.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_results(
    prepared_manifest: str | Path,
    result_paths: Sequence[str | Path],
    output_dir: str | Path,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Validate, analyze, and atomically complete the frozen 12-run panel."""

    bootstrap = int(bootstrap)
    permutations = int(permutations)
    seed = int(seed)
    if bootstrap <= 0 or permutations <= 0:
        raise ValueError("bootstrap and permutations must be positive")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    complete_path = output_path / "complete_manifest.json"
    _atomic_write_json(
        complete_path,
        {
            "schema_version": 1,
            "experiment": "evitrace_cross_verifier_finetune_v1",
            "complete": False,
            "status": "analysis_running_or_failed",
            "updated_at": utc_now(),
        },
    )

    manifest_path = Path(prepared_manifest).resolve()
    if not manifest_path.is_file():
        raise FineTuneAnalysisError(
            f"Prepared manifest does not exist: {manifest_path}"
        )
    prepared_sha = sha256_file(manifest_path)
    (
        prepared,
        gold_by_event,
        val_gold_by_event,
        training_label_prior,
        prepared_metadata,
    ) = _load_and_verify_prepared_manifest(manifest_path)

    supplied = [Path(path).resolve() for path in result_paths]
    if len(supplied) != 12 or len(set(supplied)) != 12:
        raise FineTuneAnalysisError(
            "result_paths must contain exactly 12 distinct paths"
        )
    run_metadata: list[dict[str, Any]] = []
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    resolved_results: set[Path] = set()
    for supplied_path in supplied:
        run, rows, _runtime = _load_result_run(
            supplied_path,
            prepared_manifest_sha=prepared_sha,
        )
        run_id = str(run["run_id"])
        result_path = Path(run["result_file"]["path"]).resolve()
        if result_path in resolved_results:
            raise FineTuneAnalysisError(
                f"Duplicate resolved logical-results file: {result_path}"
            )
        resolved_results.add(result_path)
        if run_id in rows_by_run:
            raise FineTuneAnalysisError(f"Duplicate run_id: {run_id}")
        if result_path == Path(
            prepared_metadata["gold_file"]["path"]
        ).resolve():
            raise FineTuneAnalysisError(
                "Independent gold cannot be a logical-result file"
            )
        run_metadata.append(run)
        rows_by_run[run_id] = rows
    seeds = _validate_run_grid(run_metadata, rows_by_run)
    expected_runtime = prepared_metadata["expected_runtime_provenance"]
    for run in run_metadata:
        backbone = str(run["backbone"])
        expected = expected_runtime[backbone]
        for field in ("registry_sha256", "tokenizer_sha256"):
            if str(run[field]) != str(expected[field]):
                raise FineTuneAnalysisError(
                    f"{run['run_id']}: {field} differs from the frozen "
                    f"prepared {backbone} contract"
                )
        if run["label_token_ids"] != expected["label_token_ids"]:
            raise FineTuneAnalysisError(
                f"{run['run_id']}: A-F label-token IDs differ from prepare"
            )
    provenance_audit = _validate_provenance_consistency(run_metadata)
    provenance_audit[
        "prepared_registry_tokenizer_and_label_tokens_verified"
    ] = True
    runs_by_id = {
        str(run["run_id"]): run for run in run_metadata
    }

    main_rows = _build_pair_rows(
        rows_by_run,
        runs_by_id,
        gold_by_event,
        comparison_type="main",
    )
    order_rows = _build_pair_rows(
        rows_by_run,
        runs_by_id,
        gold_by_event,
        comparison_type="order_only",
    )
    prefix_curves, prefix_positions = _build_prefix_curves(
        rows_by_run, runs_by_id, gold_by_event
    )
    val_pair_rows = _build_val_paired_rows(
        rows_by_run, runs_by_id, val_gold_by_event
    )
    val_condition_rows = _build_val_condition_rows(
        rows_by_run, runs_by_id, val_gold_by_event
    )
    selected_hashes_by_assignment = {
        backbone: {
            assignment: set(values)
            for assignment, values in assignments.items()
        }
        for backbone, assignments in prepared_metadata[
            "training_assignment_selected_snippet_hashes"
        ].items()
    }
    observed_canonical_counts = {
        "main_claims": len({str(row["event_id"]) for row in main_rows}),
        "order_only_claims": len(
            {str(row["event_id"]) for row in order_rows}
        ),
        "prefix_claims": len(
            {str(row["event_id"]) for row in prefix_curves}
        ),
        "prefix_positions": len(
            {
                (str(row["event_id"]), int(row["k"]))
                for row in prefix_positions
            }
        ),
        "strict_prefix_positions": len(
            {
                (str(row["event_id"]), int(row["k"]))
                for row in prefix_positions
                if row["prefix_relation"] == "same_set_different_order"
            }
        ),
    }
    expected_canonical_counts = {
        "main_claims": EXPECTED_MAIN_CLAIMS,
        "order_only_claims": EXPECTED_ORDER_CLAIMS,
        "prefix_claims": EXPECTED_PREFIX_CLAIMS,
        "prefix_positions": EXPECTED_PREFIX_POSITIONS,
        "strict_prefix_positions": EXPECTED_STRICT_PREFIX_POSITIONS,
    }
    if observed_canonical_counts != expected_canonical_counts:
        raise FineTuneAnalysisError(
            "Canonical test/prefix count gate failed: "
            f"observed={observed_canonical_counts}, "
            f"expected={expected_canonical_counts}"
        )
    primary = {
        "main": _comparison_analysis(
            main_rows,
            bootstrap=bootstrap,
            permutations=permutations,
            seed=seed + 101,
        ),
        "order_only": _comparison_analysis(
            order_rows,
            bootstrap=bootstrap,
            permutations=permutations,
            seed=seed + 211,
        ),
    }
    adjusted = holm_adjust(
        [
            primary[comparison]["randomization"]["two_sided_pvalue"]
            for comparison in ("main", "order_only")
        ]
    )
    for comparison, adjusted_p in zip(
        ("main", "order_only"), adjusted
    ):
        primary[comparison]["randomization"]["holm_pvalue"] = adjusted_p
        primary[comparison]["randomization"][
            "holm_family"
        ] = ["main_delta_macro_f1", "order_only_delta_macro_f1"]

    prefix = summarize_prefix_curves(
        prefix_curves,
        prefix_positions,
        bootstrap=bootstrap,
        seed=seed + 307,
    )
    prefix["count_contract"] = {
        "eligible_claims": EXPECTED_PREFIX_CLAIMS,
        "paired_positions": EXPECTED_PREFIX_POSITIONS,
        "strict_same_set_different_order_positions": (
            EXPECTED_STRICT_PREFIX_POSITIONS
        ),
        "correction": (
            "7,448 positions is the all-1,250-claim total. The formal "
            "1,152-claim prefix evaluation excludes 98 identical-final-order "
            "claims and their 452 positions, leaving 6,996."
        ),
    }
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_finetune_v1",
        "created_at": utc_now(),
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": prepared_sha,
        "prepared_experiment": prepared.get("experiment"),
        "gold_join": {
            "stage": "analysis_only",
            "source": prepared_metadata["gold_file"],
            "test_gold_claim_count": len(gold_by_event),
            "val_gold_file_key": prepared_metadata["val_gold_file_key"],
            "val_gold_claim_count": len(val_gold_by_event),
            "logical_results_contain_gold": False,
        },
        "design": {
            "backbones": list(BACKBONES),
            "assignments": list(ASSIGNMENTS),
            "seeds": seeds,
            "run_count": len(run_metadata),
            "aggregation": (
                "assignment/seed run -> equal mean within backbone -> "
                "equal mean across the two backbones"
            ),
            "inferential_unit": "claim/event_id",
            "primary_metric": "delta Macro-F1",
            "sesoi": [-SESOI, SESOI],
        },
        "runs": {
            str(run["run_id"]): run
            for run in sorted(run_metadata, key=lambda item: str(item["run_id"]))
        },
        "provenance_audit": provenance_audit,
        "primary": primary,
        "secondary": {
            "main_label_strata": _label_strata(main_rows),
            "order_only_label_strata": _label_strata(order_rows),
            "main_gold_logprob": {
                "overall": _gold_logprob_delta_summary(main_rows),
                "label_x_complexity": (
                    _gold_logprob_label_complexity_strata(main_rows)
                ),
            },
            "order_only_gold_logprob": {
                "overall": _gold_logprob_delta_summary(order_rows),
                "label_x_complexity": (
                    _gold_logprob_label_complexity_strata(order_rows)
                ),
            },
            "main_token_sensitivity_abs_le_64": (
                compute_main_token_sensitivity(main_rows)
            ),
            "exact_train_selected_snippet_sensitivity": {
                "main": compute_exact_snippet_sensitivity(
                    main_rows, selected_hashes_by_assignment
                ),
                "order_only": compute_exact_snippet_sensitivity(
                    order_rows, selected_hashes_by_assignment
                ),
            },
            "val_logit_adjustment_tau_grid": (
                compute_logit_adjustment_tau_grid(
                    val_pair_rows,
                    training_label_prior,
                )
            ),
            "validation_diagnostics": summarize_validation_conditions(
                val_condition_rows
            ),
        },
        "prefix": prefix,
    }
    metrics["interpretation"] = _interpretation(primary)

    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    paper_table_path = output_path / "paper_table.tex"
    _atomic_write_json(metrics_path, metrics)
    _atomic_write_text(report_path, _markdown_report(metrics))
    _atomic_write_text(paper_table_path, _paper_table(metrics))

    main_claim_count = len({str(row["event_id"]) for row in main_rows})
    order_claim_count = len({str(row["event_id"]) for row in order_rows})
    prefix_claim_count = len(
        {str(row["event_id"]) for row in prefix_curves}
    )
    final_manifest = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_finetune_v1",
        "created_at": utc_now(),
        "complete": True,
        "completion_is_effect_independent": True,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": prepared_sha,
        "counts": {
            "runs": 12,
            "training_complete_markers_verified": 12,
            "backbones": 2,
            "assignments_per_backbone": 2,
            "seeds_per_backbone_assignment": 3,
            "main_claims": main_claim_count,
            "order_only_claims": order_claim_count,
            "prefix_claims": prefix_claim_count,
            "main_paired_run_rows": len(main_rows),
            "order_only_paired_run_rows": len(order_rows),
            "prefix_curves": len(prefix_curves),
            "prefix_paired_position_run_rows": len(prefix_positions),
            "strict_prefix_position_pairs_unique": prefix[
                "strict_positional_subset"
            ]["paired_position_count_unique"],
            "val_paired_claims": len(
                {str(row["event_id"]) for row in val_pair_rows}
            ),
            "logical_results_per_run": EXPECTED_LOGICAL_RESULTS_PER_RUN,
            "logical_results_total": (
                12 * EXPECTED_LOGICAL_RESULTS_PER_RUN
            ),
            "logical_results_per_run_by_type": (
                EXPECTED_LOGICAL_COUNTS_PER_RUN
            ),
            "prefix_count_correction": {
                "formal_eligible_claims": 1_152,
                "formal_positions": 6_996,
                "all_1250_claim_positions_not_used": 7_448,
                "excluded_identical_order_claims": 98,
                "excluded_identical_order_positions": 452,
            },
        },
        "invariants": {
            "exact_2x2x3_run_grid": True,
            "same_logical_universe_all_runs": True,
            "independent_test_gold_joined_only_in_analysis": True,
            "no_gold_fields_in_logical_results": True,
            "all_six_way_logits_finite": True,
            "all_six_way_log_probs_normalized": True,
            "all_main_order_prefix_arms_paired": True,
            "label_stratified_claim_bootstrap": True,
            "one_shared_arm_swap_per_claim_across_12_runs": True,
            "holm_family_main_and_order_macro_f1": True,
            "sesoi_macro_f1_plus_minus_0_01": True,
            "complete_prefix_position_grids": True,
            "assignment_a_b_training_label_priors_identical": True,
            "tau_grid_is_validation_only": True,
            "main_token_sensitivity_is_fixed_abs_le_64": True,
            "strict_prefix_subset_is_per_event_k_pair": True,
            "canonical_main_order_prefix_counts_match": True,
            "prefix_6996_count_correction_disclosed": True,
            "exact_train_selected_snippet_fields_present": True,
            "exact_snippet_sensitivity_completed_main_and_order": True,
            "validation_three_condition_diagnostics_completed": True,
            "all_12_training_complete_markers_verified": True,
            "all_runtime_config_code_tokenizer_adapter_hashes_verified": True,
            "backbone_provenance_consistency_verified": True,
            "per_run_logical_counts_2500_2304_13992_5096": True,
        },
        "inputs": {
            "prepared_files": prepared_metadata["verified_files"],
            "runs": {
                str(run["run_id"]): {
                    "result_file": run["result_file"],
                    "runtime_manifest": run["runtime_manifest"],
                    "base_model_sha256": run["base_model_sha256"],
                    "adapter_sha256": run["adapter_sha256"],
                    "tokenizer_sha256": run["tokenizer_sha256"],
                    "provenance": run["provenance"],
                    "logical_counts_by_type": run[
                        "logical_counts_by_type"
                    ],
                }
                for run in sorted(
                    run_metadata, key=lambda item: str(item["run_id"])
                )
            },
        },
        "outputs": {
            "metrics": _file_metadata(metrics_path),
            "report": _file_metadata(report_path),
            "paper_table": _file_metadata(paper_table_path),
        },
        "interpretation_category": metrics["interpretation"]["category"],
    }
    _atomic_write_json(complete_path, final_manifest)
    return final_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze frozen 12-run EviTrace fine-tuned verifier results"
    )
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help=(
            "logical_results.jsonl, runtime_manifest.json, or run directory; "
            "repeat exactly 12 times"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument(
        "--permutations", type=int, default=DEFAULT_PERMUTATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = analyze_results(
        args.prepared_manifest,
        args.result,
        args.output_dir,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    print(
        "Analysis complete: "
        f"{manifest['outputs']['report']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
