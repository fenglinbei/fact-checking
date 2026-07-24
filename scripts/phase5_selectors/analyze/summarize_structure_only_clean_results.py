#!/usr/bin/env python3
"""Summarize only preregistered clean structure-only result roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_SCHEMA = "structure-only-clean-results-audit-input-v0.1"
OUTPUT_SCHEMA = "structure-only-clean-results-audit-summary-v0.1"
SECTION_ORDER = (
    "verifier_crossover_s_o",
    "liar_main",
    "rawfc_clean",
    "scifact_clean",
    "verifier_crossover_r_s",
    "liar_no_map",
)
SECTION_KINDS = {
    "verifier_crossover_s_o": "crossover",
    "liar_main": "standard",
    "rawfc_clean": "standard",
    "scifact_clean": "scifact",
    "verifier_crossover_r_s": "crossover",
    "liar_no_map": "standard",
}
HARD_FORBIDDEN_ROOT_FRAGMENTS = ("learned_marginal_proxy",)
NORMALIZED_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


class AuditInputError(ValueError):
    """Raised when the explicit audit manifest itself violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def _resolve_path(value: Any, repo_root: Path, *, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuditInputError(f"{context} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _positive_int(value: Any, *, context: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if result < (0 if allow_zero else 1):
        raise ValueError(f"{context} must be {'non-negative' if allow_zero else 'positive'}")
    return result


def _state() -> dict[str, Any]:
    return {
        "missing_artifacts": [],
        "pending_reasons": [],
        "invalid_artifacts": [],
        "artifacts": {},
    }


def _record_invalid(state: dict[str, Any], label: str, message: str) -> None:
    state["invalid_artifacts"].append({"artifact": label, "error": message})


def _validate_root(
    value: Any,
    *,
    label: str,
    required_fragments: Any,
    repo_root: Path,
    input_policy: Mapping[str, Any],
    state: dict[str, Any],
) -> Path | None:
    try:
        root = _resolve_path(value, repo_root, context=label)
    except AuditInputError as exc:
        _record_invalid(state, label, str(exc))
        return None
    lexical = str(root)
    resolved = str(root.resolve(strict=False))
    forbidden = set(HARD_FORBIDDEN_ROOT_FRAGMENTS)
    manifest_forbidden = input_policy.get("forbidden_root_fragments", [])
    if isinstance(manifest_forbidden, list):
        forbidden.update(str(item) for item in manifest_forbidden if str(item))
    for fragment in sorted(forbidden):
        if fragment in lexical or fragment in resolved:
            _record_invalid(state, label, f"forbidden root fragment {fragment!r}")
            return None
    if not isinstance(required_fragments, list) or not required_fragments:
        _record_invalid(state, label, "required root fragments must be a non-empty array")
        return None
    for fragment in required_fragments:
        if not isinstance(fragment, str) or not fragment or fragment not in lexical:
            _record_invalid(state, label, f"required root fragment is absent: {fragment!r}")
            return None
    if input_policy.get("require_within_repo") is True:
        try:
            root.resolve(strict=False).relative_to(repo_root.resolve(strict=False))
        except ValueError:
            _record_invalid(state, label, "root resolves outside the declared repository")
            return None
    return root


def _safe_artifact_path(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"artifact escapes its declared root: {relative}") from exc
    return path


def _read_json_artifact(
    path: Path,
    *,
    label: str,
    repo_root: Path,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    display = _display(path, repo_root)
    if not path.is_file():
        state["missing_artifacts"].append(display)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _record_invalid(state, label, f"cannot read JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        _record_invalid(state, label, "expected a JSON object")
        return None
    state["artifacts"][label] = {
        "path": display,
        "sha256": _sha256(path),
    }
    return payload


def _read_jsonl_export(
    path: Path,
    *,
    label: str,
    repo_root: Path,
    state: dict[str, Any],
) -> int | None:
    display = _display(path, repo_root)
    if not path.is_file():
        state["missing_artifacts"].append(display)
        return None
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                row_count += 1
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _record_invalid(state, label, f"cannot validate JSONL: {exc}")
        return None
    if row_count <= 0:
        _record_invalid(state, label, "export contains no records")
        return None
    state["artifacts"][label] = {
        "path": display,
        "sha256": _sha256(path),
        "row_count": row_count,
    }
    return row_count


def _training_result(
    run_root: Path,
    *,
    repo_root: Path,
    state: dict[str, Any],
    label: str = "training_complete",
    minimum_global_step: int = 1,
) -> dict[str, Any] | None:
    marker = _safe_artifact_path(run_root, "train/training_complete.json")
    payload = _read_json_artifact(
        marker, label=label, repo_root=repo_root, state=state
    )
    if payload is None:
        return None
    if payload.get("completed") is not True:
        state["pending_reasons"].append(
            f"{_display(marker, repo_root)} does not declare completed=true"
        )
        return None
    try:
        global_step = _positive_int(
            payload.get("global_step"), context=f"{label}.global_step"
        )
        if global_step < minimum_global_step:
            raise ValueError(
                f"{label}.global_step must be at least {minimum_global_step}"
            )
        best_score = payload.get("best_score")
        normalized_best = (
            _finite_float(best_score, context=f"{label}.best_score")
            if best_score is not None
            else None
        )
    except ValueError as exc:
        _record_invalid(state, label, str(exc))
        return None
    return {
        "completed": True,
        "global_step": global_step,
        "best_score": normalized_best,
    }


def _normalize_verifier_metrics(
    payload: Mapping[str, Any],
    *,
    split: str,
    checkpoint: str,
    label_schema: str,
    context: str,
    expected_num_samples: int | None = None,
) -> dict[str, Any]:
    if payload.get("split") != split:
        raise ValueError(f"{context}.split must be {split!r}")
    if payload.get("checkpoint") != checkpoint:
        raise ValueError(f"{context}.checkpoint must be {checkpoint!r}")
    if payload.get("label_schema") != label_schema:
        raise ValueError(f"{context}.label_schema must be {label_schema!r}")
    if payload.get("eval_backend") != "label_token_logits":
        raise ValueError(f"{context}.eval_backend must be 'label_token_logits'")
    metrics = {
        name: _finite_float(payload.get(name), context=f"{context}.{name}")
        for name in NORMALIZED_METRICS
    }
    metrics["num_samples"] = _positive_int(
        payload.get("num_samples"), context=f"{context}.num_samples"
    )
    if expected_num_samples is not None and metrics["num_samples"] != expected_num_samples:
        raise ValueError(
            f"{context}.num_samples must be {expected_num_samples}, "
            f"got {metrics['num_samples']}"
        )
    if payload.get("parse_error_rate") is not None:
        metrics["parse_error_rate"] = _finite_float(
            payload.get("parse_error_rate"), context=f"{context}.parse_error_rate"
        )
    metrics.update(
        {
            "split": split,
            "checkpoint": checkpoint,
            "label_schema": label_schema,
            "eval_backend": "label_token_logits",
        }
    )
    return metrics


def _finish_section(result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    result.update(state)
    if state["invalid_artifacts"]:
        result["status"] = "invalid"
    elif state["missing_artifacts"] or state["pending_reasons"]:
        result["status"] = "pending"
    else:
        result["status"] = "complete"
    return result


def _summarize_standard(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    input_policy: Mapping[str, Any],
) -> dict[str, Any]:
    state = _state()
    result: dict[str, Any] = {"kind": "standard", "training": None, "metrics": {}}
    root = _validate_root(
        spec.get("run_root"),
        label="run_root",
        required_fragments=spec.get("required_root_fragments"),
        repo_root=repo_root,
        input_policy=input_policy,
        state=state,
    )
    if root is None:
        return _finish_section(result, state)
    result["run_root"] = _display(root, repo_root)
    result["training"] = _training_result(root, repo_root=repo_root, state=state)
    checkpoint = str(spec.get("checkpoint") or "")
    label_schema = str(spec.get("label_schema") or "")
    splits = spec.get("splits")
    expected_counts = spec.get("expected_num_samples", {})
    if not isinstance(expected_counts, Mapping):
        _record_invalid(
            state, "expected_num_samples", "expected_num_samples must be an object"
        )
        expected_counts = {}
    if not isinstance(splits, list) or not splits:
        _record_invalid(state, "splits", "splits must be a non-empty array")
        return _finish_section(result, state)
    for raw_split in splits:
        split = str(raw_split)
        path = _safe_artifact_path(
            root, f"eval/{split}/{checkpoint}/label_token/metrics.json"
        )
        payload = _read_json_artifact(
            path,
            label=f"metrics_{split}",
            repo_root=repo_root,
            state=state,
        )
        if payload is None:
            continue
        try:
            result["metrics"][split] = _normalize_verifier_metrics(
                payload,
                split=split,
                checkpoint=checkpoint,
                label_schema=label_schema,
                context=f"metrics_{split}",
                expected_num_samples=(
                    int(expected_counts[split]) if split in expected_counts else None
                ),
            )
        except ValueError as exc:
            _record_invalid(state, f"metrics_{split}", str(exc))
    return _finish_section(result, state)


def _path_matches(value: Any, expected: Path, repo_root: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    actual = Path(value)
    if not actual.is_absolute():
        actual = repo_root / actual
    return actual.resolve(strict=False) == expected.resolve(strict=False)


def _prf(payload: Any, *, context: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    return {
        "precision": _finite_float(payload.get("precision"), context=f"{context}.precision"),
        "recall": _finite_float(payload.get("recall"), context=f"{context}.recall"),
        "f1": _finite_float(payload.get("f1"), context=f"{context}.f1"),
    }


def _summarize_scifact(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    input_policy: Mapping[str, Any],
) -> dict[str, Any]:
    state = _state()
    result: dict[str, Any] = {
        "kind": "scifact",
        "training": None,
        "verifier_metrics": {},
        "official_metrics": {},
        "exports": {},
    }
    root = _validate_root(
        spec.get("run_root"),
        label="run_root",
        required_fragments=spec.get("required_root_fragments"),
        repo_root=repo_root,
        input_policy=input_policy,
        state=state,
    )
    if root is None:
        return _finish_section(result, state)
    result["run_root"] = _display(root, repo_root)
    result["training"] = _training_result(root, repo_root=repo_root, state=state)
    checkpoint = str(spec.get("checkpoint") or "")
    label_schema = str(spec.get("label_schema") or "")

    val_metrics_path = _safe_artifact_path(
        root, f"eval/val/{checkpoint}/label_token/metrics.json"
    )
    val_metrics = _read_json_artifact(
        val_metrics_path,
        label="metrics_val",
        repo_root=repo_root,
        state=state,
    )
    if val_metrics is not None:
        try:
            result["verifier_metrics"]["val"] = _normalize_verifier_metrics(
                val_metrics,
                split="val",
                checkpoint=checkpoint,
                label_schema=label_schema,
                context="metrics_val",
                expected_num_samples=(
                    int(spec["expected_num_samples"]["val"])
                    if isinstance(spec.get("expected_num_samples"), Mapping)
                    and "val" in spec["expected_num_samples"]
                    else None
                ),
            )
        except ValueError as exc:
            _record_invalid(state, "metrics_val", str(exc))

    prediction_manifest_path = _safe_artifact_path(
        root, f"eval/test/{checkpoint}/label_token/prediction_manifest.json"
    )
    prediction_manifest = _read_json_artifact(
        prediction_manifest_path,
        label="test_prediction_manifest",
        repo_root=repo_root,
        state=state,
    )
    test_prediction_count: int | None = None
    if prediction_manifest is not None:
        try:
            if prediction_manifest.get("prediction_only") is not True:
                raise ValueError("test prediction manifest must set prediction_only=true")
            if prediction_manifest.get("split") != "test":
                raise ValueError("test prediction manifest split must be 'test'")
            if prediction_manifest.get("checkpoint") != checkpoint:
                raise ValueError(
                    f"test prediction manifest checkpoint must be {checkpoint!r}"
                )
            if prediction_manifest.get("label_schema") != label_schema:
                raise ValueError(
                    f"test prediction manifest label_schema must be {label_schema!r}"
                )
            test_prediction_count = _positive_int(
                prediction_manifest.get("num_samples"),
                context="test_prediction_manifest.num_samples",
            )
            result["exports"]["test_prediction_manifest"] = {
                "num_samples": test_prediction_count,
                "num_labeled_samples": _positive_int(
                    prediction_manifest.get("num_labeled_samples", 0),
                    context="test_prediction_manifest.num_labeled_samples",
                    allow_zero=True,
                ),
                "prediction_only": True,
            }
        except ValueError as exc:
            _record_invalid(state, "test_prediction_manifest", str(exc))

    official_path = _safe_artifact_path(
        root, "submission/scifact_official_style_metrics_val.json"
    )
    official = _read_json_artifact(
        official_path,
        label="official_metrics_val",
        repo_root=repo_root,
        state=state,
    )
    official_count: int | None = None
    if official is not None:
        try:
            expected_output = _safe_artifact_path(
                root, "submission/scifact_submission_val.jsonl"
            )
            expected_predictions = _safe_artifact_path(
                root, f"eval/val/{checkpoint}/label_token/val_predictions.jsonl"
            )
            expected_build = _safe_artifact_path(root, "build/build_val.jsonl")
            expected_trace = _resolve_path(
                spec.get("expected_trace"), repo_root, context="expected_trace"
            )
            for key, expected in (
                ("output", expected_output),
                ("predictions", expected_predictions),
                ("build_jsonl", expected_build),
                ("trace", expected_trace),
            ):
                if not _path_matches(official.get(key), expected, repo_root):
                    raise ValueError(
                        f"official_metrics_val.{key} does not point to the clean artifact"
                    )
            claim_label = official.get("claim_label")
            if not isinstance(claim_label, Mapping):
                raise ValueError("official_metrics_val.claim_label must be an object")
            official_count = _positive_int(
                claim_label.get("n"), context="official_metrics_val.claim_label.n"
            )
            result["official_metrics"] = {
                "claim_label": {
                    "accuracy": _finite_float(
                        claim_label.get("accuracy"),
                        context="official_metrics_val.claim_label.accuracy",
                    ),
                    "macro_f1": _finite_float(
                        claim_label.get("macro_f1"),
                        context="official_metrics_val.claim_label.macro_f1",
                    ),
                    "num_samples": official_count,
                },
                "abstract": _prf(official.get("abstract"), context="official_metrics_val.abstract"),
                "abstract_label_only": _prf(
                    official.get("abstract_label_only"),
                    context="official_metrics_val.abstract_label_only",
                ),
                "sentence": _prf(official.get("sentence"), context="official_metrics_val.sentence"),
                "sentence_selection_only": _prf(
                    official.get("sentence_selection_only"),
                    context="official_metrics_val.sentence_selection_only",
                ),
            }
        except (AuditInputError, ValueError) as exc:
            _record_invalid(state, "official_metrics_val", str(exc))

    val_export_path = _safe_artifact_path(
        root, "submission/scifact_submission_val.jsonl"
    )
    test_export_path = _safe_artifact_path(
        root, "submission/scifact_submission_test.jsonl"
    )
    val_export_count = _read_jsonl_export(
        val_export_path,
        label="submission_val",
        repo_root=repo_root,
        state=state,
    )
    test_export_count = _read_jsonl_export(
        test_export_path,
        label="submission_test",
        repo_root=repo_root,
        state=state,
    )
    if val_export_count is not None:
        result["exports"]["val"] = {"row_count": val_export_count}
    if test_export_count is not None:
        result["exports"]["test"] = {"row_count": test_export_count}
    expected_val_count = None
    if "val" in result["verifier_metrics"]:
        expected_val_count = result["verifier_metrics"]["val"]["num_samples"]
    for context, left, right in (
        ("val verifier/official sample count", expected_val_count, official_count),
        ("val official/export row count", official_count, val_export_count),
        ("test prediction/export row count", test_prediction_count, test_export_count),
    ):
        if left is not None and right is not None and left != right:
            _record_invalid(state, context, f"count mismatch: {left} != {right}")
    return _finish_section(result, state)


def _summarize_crossover(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    input_policy: Mapping[str, Any],
) -> dict[str, Any]:
    state = _state()
    result: dict[str, Any] = {
        "kind": "crossover",
        "training": {},
        "crossover": {},
    }
    summary_root = _validate_root(
        spec.get("summary_root"),
        label="summary_root",
        required_fragments=spec.get("required_summary_root_fragments"),
        repo_root=repo_root,
        input_policy=input_policy,
        state=state,
    )
    verifier_specs = spec.get("verifiers")
    if not isinstance(verifier_specs, Mapping) or len(verifier_specs) != 2:
        _record_invalid(state, "verifiers", "verifiers must contain exactly two entries")
        return _finish_section(result, state)
    roots: dict[str, Path] = {}
    for verifier_id, raw_verifier_spec in verifier_specs.items():
        if not isinstance(raw_verifier_spec, Mapping):
            _record_invalid(state, str(verifier_id), "verifier spec must be an object")
            continue
        root = _validate_root(
            raw_verifier_spec.get("run_root"),
            label=f"{verifier_id}.run_root",
            required_fragments=raw_verifier_spec.get("required_root_fragments"),
            repo_root=repo_root,
            input_policy=input_policy,
            state=state,
        )
        if root is None:
            continue
        roots[str(verifier_id)] = root
        result["training"][str(verifier_id)] = _training_result(
            root,
            repo_root=repo_root,
            state=state,
            label=f"{verifier_id}_training_complete",
            minimum_global_step=int(raw_verifier_spec.get("minimum_global_step", 1)),
        )
    if summary_root is None:
        return _finish_section(result, state)
    result["summary_root"] = _display(summary_root, repo_root)
    summary_path = _safe_artifact_path(summary_root, "summary.json")
    summary = _read_json_artifact(
        summary_path,
        label="crossover_summary",
        repo_root=repo_root,
        state=state,
    )
    if summary is None:
        return _finish_section(result, state)
    try:
        expected_schema = str(spec.get("summary_schema_version") or "")
        expected_checkpoint = str(spec.get("checkpoint") or "")
        expected_split = str(spec.get("split") or "")
        if summary.get("schema_version") != expected_schema:
            raise ValueError(f"summary schema must be {expected_schema!r}")
        if summary.get("status") != "complete":
            raise ValueError("crossover summary must declare status=complete")
        if summary.get("checkpoint") != expected_checkpoint:
            raise ValueError(f"summary checkpoint must be {expected_checkpoint!r}")
        if summary.get("split") != expected_split:
            raise ValueError(f"summary split must be {expected_split!r}")
        event_count = _positive_int(
            summary.get("event_count"), context="crossover_summary.event_count"
        )
        expected_event_count = spec.get("expected_event_count")
        if expected_event_count is not None and event_count != int(expected_event_count):
            raise ValueError(
                f"crossover summary event_count must be {expected_event_count}, "
                f"got {event_count}"
            )
        expected_event_sha = spec.get("expected_event_id_sequence_sha256")
        if expected_event_sha is not None and summary.get(
            "event_id_sequence_sha256"
        ) != expected_event_sha:
            raise ValueError(
                "crossover summary event_id_sequence_sha256 does not match the manifest"
            )
        expected_prompt_cells = spec.get("prompt_cells")
        if summary.get("prompt_cells") != expected_prompt_cells:
            raise ValueError("crossover summary prompt_cells do not match the manifest")
        expected_verifier_ids = set(str(key) for key in verifier_specs)
        summary_verifiers = summary.get("verifiers")
        if not isinstance(summary_verifiers, Mapping) or set(summary_verifiers) != expected_verifier_ids:
            raise ValueError("crossover summary verifier IDs do not match the manifest")
        for verifier_id, verifier_spec in verifier_specs.items():
            verifier_payload = summary_verifiers.get(verifier_id)
            if not isinstance(verifier_payload, Mapping):
                raise ValueError(f"summary verifier {verifier_id} must be an object")
            root = roots.get(str(verifier_id))
            if root is None:
                continue
            if not _path_matches(verifier_payload.get("run_dir"), root / "train", repo_root):
                raise ValueError(f"summary verifier {verifier_id} run_dir is not the clean run")
            output_dir = str(verifier_spec.get("summary_output_dir") or "")
            if not output_dir or not _path_matches(
                verifier_payload.get("root"), summary_root / output_dir, repo_root
            ):
                raise ValueError(
                    f"summary verifier {verifier_id} root is not the declared crossover output"
                )
            adapter_sha = verifier_payload.get("adapter_sha256")
            if not isinstance(adapter_sha, str) or len(adapter_sha) != 64:
                raise ValueError(f"summary verifier {verifier_id} lacks adapter SHA-256")
            expected_adapter_sha = verifier_spec.get("expected_adapter_sha256")
            if expected_adapter_sha is not None and adapter_sha != expected_adapter_sha:
                raise ValueError(
                    f"summary verifier {verifier_id} adapter SHA-256 does not match"
                )
        prompt_ids = set(str(key) for key in expected_prompt_cells)
        matrix = summary.get("macro_f1_matrix")
        if not isinstance(matrix, Mapping) or set(matrix) != expected_verifier_ids:
            raise ValueError("macro_f1_matrix verifier IDs do not match the manifest")
        normalized_matrix: dict[str, dict[str, float]] = {}
        for verifier_id in verifier_specs:
            row = matrix.get(verifier_id)
            if not isinstance(row, Mapping) or set(row) != prompt_ids:
                raise ValueError(f"macro_f1_matrix.{verifier_id} prompt IDs do not match")
            normalized_matrix[str(verifier_id)] = {
                prompt_id: _finite_float(
                    row.get(prompt_id),
                    context=f"macro_f1_matrix.{verifier_id}.{prompt_id}",
                )
                for prompt_id in expected_prompt_cells
            }
        raw_contrasts = summary.get("contrasts")
        if not isinstance(raw_contrasts, Mapping) or not raw_contrasts:
            raise ValueError("crossover summary contrasts must be a non-empty object")
        normalized_contrasts = {
            str(key): _finite_float(value, context=f"contrasts.{key}")
            for key, value in raw_contrasts.items()
        }
        result["crossover"] = {
            "split": expected_split,
            "checkpoint": expected_checkpoint,
            "event_count": event_count,
            "prompt_cells": dict(expected_prompt_cells),
            "macro_f1_matrix": normalized_matrix,
            "contrasts": normalized_contrasts,
        }
    except ValueError as exc:
        _record_invalid(state, "crossover_summary", str(exc))
    return _finish_section(result, state)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditInputError(f"missing audit input manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"invalid audit input manifest JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditInputError("audit input manifest must be a JSON object")
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise AuditInputError(f"audit input schema must be {INPUT_SCHEMA!r}")
    input_policy = payload.get("input_policy")
    if not isinstance(input_policy, Mapping):
        raise AuditInputError("input_policy must be an object")
    if input_policy.get("fallback_allowed") is not False:
        raise AuditInputError("fallback_allowed must be false")
    if input_policy.get("search_allowed") is not False:
        raise AuditInputError("search_allowed must be false")
    sections = payload.get("sections")
    if not isinstance(sections, Mapping) or set(sections) != set(SECTION_ORDER):
        raise AuditInputError(
            f"sections must contain exactly: {', '.join(SECTION_ORDER)}"
        )
    for section_name, expected_kind in SECTION_KINDS.items():
        section = sections.get(section_name)
        if not isinstance(section, Mapping) or section.get("kind") != expected_kind:
            raise AuditInputError(
                f"{section_name}.kind must be {expected_kind!r}"
            )
    return payload


def summarize_clean_results(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    input_policy = manifest["input_policy"]
    sections: dict[str, Any] = {}
    for section_name in SECTION_ORDER:
        spec = manifest["sections"][section_name]
        kind = spec["kind"]
        if kind == "standard":
            section = _summarize_standard(
                spec, repo_root=repo_root, input_policy=input_policy
            )
        elif kind == "scifact":
            section = _summarize_scifact(
                spec, repo_root=repo_root, input_policy=input_policy
            )
        else:
            section = _summarize_crossover(
                spec, repo_root=repo_root, input_policy=input_policy
            )
        sections[section_name] = section
    counts = {
        status: sum(section["status"] == status for section in sections.values())
        for status in ("complete", "pending", "invalid")
    }
    if counts["invalid"]:
        overall_status = "invalid"
    elif counts["complete"] == len(SECTION_ORDER):
        overall_status = "complete"
    elif counts["complete"]:
        overall_status = "partial"
    else:
        overall_status = "pending"
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": overall_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": {
            "path": _display(manifest_path, repo_root),
            "sha256": _sha256(manifest_path),
            "schema_version": INPUT_SCHEMA,
        },
        "provenance_policy": {
            "explicit_roots_only": True,
            "search_performed": False,
            "fallback_used": False,
            "legacy_proxy_artifacts_allowed": False,
        },
        "coverage": {"total": len(SECTION_ORDER), **counts},
        "sections": sections,
    }


def _fmt(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Structure-only clean results audit",
        "",
        f"- overall status: `{summary['status']}`",
        "- input policy: explicit clean roots only; search=false; fallback=false",
        "- legacy proxy artifacts: forbidden",
        "",
        "## Coverage",
        "",
        "| section | status | missing / pending | invalid |",
        "|---|---|---|---|",
    ]
    sections = summary["sections"]
    for name in SECTION_ORDER:
        section = sections[name]
        status = str(section["status"]).upper()
        pending = list(section.get("missing_artifacts", [])) + list(
            section.get("pending_reasons", [])
        )
        invalid = [
            f"{item['artifact']}: {item['error']}"
            for item in section.get("invalid_artifacts", [])
        ]
        lines.append(
            f"| {name} | **{status}** | {'; '.join(pending) or '-'} | "
            f"{'<br>'.join(invalid) or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Verifier metrics",
            "",
            "| section | split | status | accuracy | macro precision | macro recall | macro F1 | n |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    metric_rows = (
        ("liar_main", "test", "metrics"),
        ("rawfc_clean", "val", "metrics"),
        ("rawfc_clean", "test", "metrics"),
        ("scifact_clean", "val", "verifier_metrics"),
        ("liar_no_map", "val", "metrics"),
        ("liar_no_map", "test", "metrics"),
    )
    for section_name, split, metric_key in metric_rows:
        metrics = sections[section_name].get(metric_key, {}).get(split)
        if metrics is None:
            lines.append(
                f"| {section_name} | {split} | **PENDING** | - | - | - | - | - |"
            )
        else:
            lines.append(
                f"| {section_name} | {split} | available | "
                f"{_fmt(metrics['accuracy'])} | {_fmt(metrics['macro_precision'])} | "
                f"{_fmt(metrics['macro_recall'])} | {_fmt(metrics['macro_f1'])} | "
                f"{metrics['num_samples']} |"
            )

    official = sections["scifact_clean"].get("official_metrics", {})
    lines.extend(["", "## SciFact official-style metrics", ""])
    if not official:
        lines.append("**PENDING**: clean SciFact official metrics/export are incomplete.")
    else:
        claim = official["claim_label"]
        lines.extend(
            [
                "| metric family | precision | recall | F1 | accuracy | n |",
                "|---|---:|---:|---:|---:|---:|",
                f"| claim label | - | - | {_fmt(claim['macro_f1'])} | {_fmt(claim['accuracy'])} | {claim['num_samples']} |",
            ]
        )
        for family in ("abstract", "abstract_label_only", "sentence", "sentence_selection_only"):
            row = official[family]
            lines.append(
                f"| {family} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
                f"{_fmt(row['f1'])} | - | - |"
            )

    for section_name, title in (
        ("verifier_crossover_s_o", "V_S / V_O crossover"),
        ("verifier_crossover_r_s", "V_R / V_S crossover"),
    ):
        section = sections[section_name]
        lines.extend(["", f"## {title}", ""])
        crossover = section.get("crossover", {})
        matrix = crossover.get("macro_f1_matrix")
        if not matrix:
            missing = "; ".join(section.get("missing_artifacts", []))
            lines.append(f"**PENDING**: {missing or 'clean crossover summary is unavailable.'}")
            continue
        prompts = list(crossover["prompt_cells"])
        lines.extend(
            [
                f"- split: `{crossover['split']}`; checkpoint: `{crossover['checkpoint']}`; n={crossover['event_count']}",
                "",
                "| verifier | " + " | ".join(prompts) + " |",
                "|---|" + "---:|" * len(prompts),
            ]
        )
        for verifier_id, row in matrix.items():
            lines.append(
                f"| {verifier_id} | "
                + " | ".join(_fmt(row[prompt]) for prompt in prompts)
                + " |"
            )
        lines.extend(["", "Contrasts:", ""])
        for key, value in crossover["contrasts"].items():
            lines.append(f"- `{key}`: `{value:+.6f}`")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_summary(
    summary: Mapping[str, Any], *, output_json: Path, output_md: Path
) -> None:
    _atomic_write(
        output_json,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output_md, render_markdown(summary))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/validation/structure_only_clean_results_audit_v0_1.json",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    manifest_path = _resolve_path(args.manifest, repo_root, context="manifest")
    summary = summarize_clean_results(manifest_path, repo_root=repo_root)
    write_summary(
        summary,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    print(
        "[structure-only-clean-results-audit] "
        f"status={summary['status']} coverage={summary['coverage']} "
        f"json={args.output_json} markdown={args.output_md}"
    )
    return 2 if summary["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
