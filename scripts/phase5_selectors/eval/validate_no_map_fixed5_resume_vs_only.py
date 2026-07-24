#!/usr/bin/env python3
"""Fail-closed preflight for the no-map fixed-K5 V_S-only resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_CELLS = ("N_fixed5", "S_fixed5")
EXPECTED_EVENT_COUNT = 1234
EXPECTED_UNIQUE_PROMPTS = 2272
EXPECTED_EVENT_SHA256 = "65038f1f222b7d990642970ebf7281434abdb17fe61ec1e14ed0c937e8ee6549"


class ResumeContractError(RuntimeError):
    """Raised when an artifact cannot be safely reused by the resume path."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResumeContractError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeContractError(f"cannot read JSON artifact {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> str:
    _require(len(expected) == 64, f"{label} expected SHA-256 is not fixed")
    before = path.stat()
    actual = _sha256(path)
    after = path.stat()
    _require(
        (before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while hashing: {path}",
    )
    _require(actual == expected, f"{label} SHA-256 mismatch: expected={expected} actual={actual}")
    return actual


def _jsonl_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResumeContractError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                _require(isinstance(row, dict), f"expected JSON object at {path}:{line_number}")
                yield line_number, row
    except OSError as exc:
        raise ResumeContractError(f"cannot read JSONL artifact {path}: {exc}") from exc


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in _jsonl_rows(path))


def _validate_index_domain(path: Path, *, field: str, expected_count: int) -> None:
    values = [int(row.get(field, -1)) for _, row in _jsonl_rows(path)]
    _require(len(values) == expected_count, f"{path} rows={len(values)} expected={expected_count}")
    _require(values == list(range(expected_count)), f"{path} {field} is not exactly 0..{expected_count - 1}")


def _resolved_declared_path(value: object) -> Path:
    text = str(value or "")
    _require(bool(text), "manifest contains an empty artifact path")
    return Path(text).resolve()


def _validate_checkpoint(
    run_dir: Path,
    *,
    role: str,
    expected_adapter_sha256: str,
    expected_completion_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = run_dir / "checkpoint-800"
    adapter = checkpoint / "adapter_model.safetensors"
    _require((checkpoint / "adapter_config.json").is_file(), f"{role} checkpoint-800 adapter config is missing")
    _require((run_dir / "config.resolved.yaml").is_file(), f"{role} resolved runtime config is missing")
    adapter_sha = _require_sha(adapter, expected_adapter_sha256, f"{role} checkpoint-800 adapter")
    completion_path = run_dir / "training_complete.json"
    if role == "V_N":
        _require(not completion_path.exists(), "V_N must remain a capped run without training_complete.json")
        latest = _read_json(run_dir / "latest_state" / "trainer_state.json")
        _require(int(latest.get("global_step", -1)) == 800, "V_N latest_state global_step must equal 800")
        progress = 800
        completed = False
    else:
        _require(completion_path.is_file(), "V_S must have training_complete.json")
        if expected_completion_sha256:
            _require_sha(completion_path, expected_completion_sha256, "V_S training_complete")
        completion = _read_json(completion_path)
        _require(completion.get("completed") is True, "V_S training_complete.completed must be true")
        progress = int(completion.get("global_step", -1))
        _require(progress >= 800, "V_S completed training is below checkpoint-800")
        completed = True
    return {
        "role": role,
        "checkpoint": "checkpoint-800",
        "adapter_sha256": adapter_sha,
        "training_complete": completed,
        "progress_step": progress,
    }


def _validate_cap_manifest(path: Path, *, expected_sha256: str, expected_adapter_sha256: str) -> None:
    _require_sha(path, expected_sha256, "V_N cap manifest")
    payload = _read_json(path)
    contract = payload.get("contract") or {}
    _require(
        payload.get("status") == "capped"
        and payload.get("role") == "V_N"
        and int(payload.get("seed", -1)) == 42
        and payload.get("checkpoint") == "checkpoint-800"
        and int(payload.get("checkpoint_step", -1)) == 800
        and payload.get("training_complete_present") is False,
        "V_N cap manifest header violates the fixed-step contract",
    )
    _require(
        contract.get("status") == "ready"
        and contract.get("role") == "V_N"
        and contract.get("checkpoint") == "checkpoint-800"
        and int(contract.get("progress_step", -1)) == 800
        and contract.get("training_complete_present") is False
        and str((contract.get("adapter") or {}).get("sha256")) == expected_adapter_sha256,
        "V_N nested cap provenance is incompatible",
    )


def _validate_matrix(matrix_root: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_path = matrix_root / "manifest.json"
    manifest_sha = _require_sha(manifest_path, expected_manifest_sha256, "frozen matrix manifest")
    manifest = _read_json(manifest_path)
    cells = manifest.get("cells") or []
    checkpoint = manifest.get("checkpoint_contract") or {}
    _require(
        manifest.get("schema_version") == "no-map-structure-fixed5-matrix-v0.1"
        and manifest.get("matrix_kind") == "no_map_structure_matched_verifier_crossover"
        and manifest.get("all_ready") is True
        and manifest.get("split") == "val"
        and manifest.get("label_schema") == "liar6"
        and int(manifest.get("expected_k", -1)) == 5
        and int(manifest.get("event_count", -1)) == EXPECTED_EVENT_COUNT
        and int(manifest.get("cell_count", -1)) == 2
        and str(manifest.get("event_id_sequence_sha256")) == EXPECTED_EVENT_SHA256
        and [str(cell.get("cell_id")) for cell in cells] == list(EXPECTED_CELLS),
        "frozen matrix is not the exact val/K5/n=1234 N/S matrix",
    )
    _require(
        checkpoint.get("checkpoint") == "checkpoint-800"
        and checkpoint.get("selection") == "fixed_step"
        and checkpoint.get("split") == "val"
        and checkpoint.get("test_allowed") is False
        and checkpoint.get("best_alias_allowed") is False,
        "matrix checkpoint contract is not fixed checkpoint-800 val-only",
    )
    audit_path = matrix_root / "audit.json"
    _require_sha(audit_path, str(manifest.get("audit_sha256") or ""), "matrix input-difference audit")
    audit = _read_json(audit_path)
    _require(
        audit.get("schema_version") == "no-map-structure-fixed5-input-difference-audit-v0.1"
        and audit.get("status") == "complete"
        and audit.get("passed") is True
        and audit.get("split") == "val"
        and int(audit.get("expected_k", -1)) == 5
        and int(audit.get("event_count", -1)) == EXPECTED_EVENT_COUNT
        and str(audit.get("event_id_sequence_sha256")) == EXPECTED_EVENT_SHA256
        and audit.get("standard_clean_results_audit_slot_mutated") is False,
        "matrix input-difference audit is incomplete or out of scope",
    )
    for cell in cells:
        cell_id = str(cell["cell_id"])
        build_path = matrix_root / cell_id / "build" / "build_val.jsonl"
        _require(_resolved_declared_path(cell.get("build_file")) == build_path.resolve(), f"{cell_id} build path drifted")
        _require_sha(build_path, str(cell.get("build_sha256") or ""), f"{cell_id} build")
        _require(_count_jsonl(build_path) == EXPECTED_EVENT_COUNT, f"{cell_id} build must contain 1234 rows")
        _require(
            cell.get("ready") is True
            and int(cell.get("row_count", -1)) == EXPECTED_EVENT_COUNT
            and str(cell.get("event_id_sequence_sha256")) == EXPECTED_EVENT_SHA256,
            f"{cell_id} manifest row/event contract failed",
        )
    return {"manifest_sha256": manifest_sha, "event_id_sequence_sha256": EXPECTED_EVENT_SHA256}


def _validate_prepared_input(
    output_dir: Path,
    *,
    matrix_root: Path,
    expected_manifest_sha256: str,
    expected_matrix_sha256: str,
) -> dict[str, Any]:
    input_dir = output_dir / "input"
    manifest_path = input_dir / "manifest.json"
    manifest_sha = _require_sha(manifest_path, expected_manifest_sha256, f"{output_dir.name} prepared-input manifest")
    manifest = _read_json(manifest_path)
    cells = manifest.get("cells") or []
    _require(
        manifest.get("schema_version") == "deduplicated_label_token_matrix_input_v0_2"
        and manifest.get("status") == "complete"
        and manifest.get("split") == "val"
        and manifest.get("label_schema") == "liar6"
        and manifest.get("label_prefix") == "Label:"
        and int(manifest.get("event_count", -1)) == EXPECTED_EVENT_COUNT
        and int(manifest.get("cell_count", -1)) == 2
        and int(manifest.get("reference_count", -1)) == 2 * EXPECTED_EVENT_COUNT
        and int(manifest.get("unique_prompt_count", -1)) == EXPECTED_UNIQUE_PROMPTS
        and str(manifest.get("event_id_sequence_sha256")) == EXPECTED_EVENT_SHA256
        and str(manifest.get("matrix_manifest_sha256")) == expected_matrix_sha256
        and [str(cell.get("cell_id")) for cell in cells] == list(EXPECTED_CELLS),
        f"{output_dir.name} prepared input violates val/K5/n=1234 contract",
    )
    _require(
        _resolved_declared_path(manifest.get("matrix_manifest")) == (matrix_root / "manifest.json").resolve(),
        f"{output_dir.name} prepared input points to a different matrix",
    )
    unique_path = input_dir / str(manifest.get("unique_rows_file") or "")
    unique_sha = str(manifest.get("unique_rows_sha256") or "")
    _require_sha(unique_path, unique_sha, f"{output_dir.name} unique rows")
    _require(_count_jsonl(unique_path) == EXPECTED_UNIQUE_PROMPTS, f"{output_dir.name} unique rows must contain 2272 rows")
    for cell in cells:
        cell_id = str(cell["cell_id"])
        mapping_path = input_dir / str(cell.get("mapping_file") or "")
        _require_sha(mapping_path, str(cell.get("mapping_sha256") or ""), f"{output_dir.name} {cell_id} mapping")
        _validate_index_domain(mapping_path, field="cell_sample_idx", expected_count=EXPECTED_EVENT_COUNT)
        source_build = matrix_root / cell_id / "build" / "build_val.jsonl"
        _require(_resolved_declared_path(cell.get("source_build")) == source_build.resolve(), f"{output_dir.name} {cell_id} source build drifted")
        _require_sha(source_build, str(cell.get("source_build_sha256") or ""), f"{output_dir.name} {cell_id} source build")
        _require(
            int(cell.get("row_count", -1)) == EXPECTED_EVENT_COUNT
            and str(cell.get("event_id_sequence_sha256")) == EXPECTED_EVENT_SHA256,
            f"{output_dir.name} {cell_id} row/event contract failed",
        )
    return {
        "manifest_sha256": manifest_sha,
        "unique_rows_sha256": unique_sha,
        "cells": [
            {
                "cell_id": str(cell["cell_id"]),
                "mapping_sha256": str(cell["mapping_sha256"]),
                "source_build_sha256": str(cell["source_build_sha256"]),
            }
            for cell in cells
        ],
    }


def _validate_raw_logits(
    output_dir: Path,
    *,
    expected_input_sha256: str,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    raw_dir = output_dir / "raw_logits"
    manifest_path = raw_dir / "manifest.json"
    if expected_manifest_sha256:
        manifest_sha = _require_sha(manifest_path, expected_manifest_sha256, f"{output_dir.name} raw-logits manifest")
    else:
        manifest_sha = _sha256(manifest_path)
    manifest = _read_json(manifest_path)
    checkpoint = manifest.get("checkpoint") or {}
    _require(
        manifest.get("schema_version") == "deduplicated_raw_label_logits_v0_2"
        and manifest.get("status") == "complete"
        and manifest.get("split") == "val"
        and manifest.get("label_schema") == "liar6"
        and int(manifest.get("num_unique_prompts", -1)) == EXPECTED_UNIQUE_PROMPTS
        and int(manifest.get("num_labels", -1)) == 6
        and manifest.get("raw_logits_shape") == [EXPECTED_UNIQUE_PROMPTS, 6]
        and str(manifest.get("input_manifest_sha256")) == expected_input_sha256,
        f"{output_dir.name} raw-logits manifest violates the frozen input contract",
    )
    _require(
        checkpoint.get("checkpoint_name") == "checkpoint-800"
        and str(checkpoint.get("adapter_sha256")) == expected_adapter_sha256,
        f"{output_dir.name} raw logits came from a different checkpoint",
    )
    logits_path = raw_dir / str(manifest.get("raw_logits_file") or "")
    index_path = raw_dir / str(manifest.get("index_file") or "")
    _require_sha(logits_path, str(manifest.get("raw_logits_sha256") or ""), f"{output_dir.name} raw logits")
    _require_sha(index_path, str(manifest.get("index_sha256") or ""), f"{output_dir.name} raw-logits index")
    _validate_index_domain(index_path, field="unique_idx", expected_count=EXPECTED_UNIQUE_PROMPTS)
    return {
        "manifest_sha256": manifest_sha,
        "raw_logits_sha256": str(manifest["raw_logits_sha256"]),
        "scoring_fingerprint": str(manifest.get("scoring_fingerprint") or ""),
        "execution_fingerprint": str(manifest.get("execution_fingerprint") or ""),
    }


def _validate_materialized(
    output_dir: Path,
    *,
    expected_input_sha256: str,
    expected_raw_manifest_sha256: str,
    expected_raw_logits_sha256: str,
    expected_adapter_sha256: str,
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    result_dir = output_dir / "materialized"
    manifest_path = result_dir / "matrix_manifest.json"
    if expected_manifest_sha256:
        manifest_sha = _require_sha(manifest_path, expected_manifest_sha256, f"{output_dir.name} materialized manifest")
    else:
        manifest_sha = _sha256(manifest_path)
    manifest = _read_json(manifest_path)
    checkpoint = manifest.get("checkpoint") or {}
    cells = manifest.get("cells") or []
    _require(
        manifest.get("schema_version") == "deduplicated_label_token_matrix_result_v0_2"
        and manifest.get("status") == "complete"
        and manifest.get("diagnostic_only") is True
        and manifest.get("split") == "val"
        and int(manifest.get("cell_count", -1)) == 2
        and str(manifest.get("input_manifest_sha256")) == expected_input_sha256
        and str(manifest.get("raw_logits_manifest_sha256")) == expected_raw_manifest_sha256
        and str(manifest.get("raw_logits_sha256")) == expected_raw_logits_sha256
        and [str(cell.get("cell_id")) for cell in cells] == list(EXPECTED_CELLS),
        f"{output_dir.name} materialized manifest violates the frozen diagnostic contract",
    )
    _require(
        checkpoint.get("checkpoint_name") == "checkpoint-800"
        and str(checkpoint.get("adapter_sha256")) == expected_adapter_sha256,
        f"{output_dir.name} materialized cells came from a different checkpoint",
    )
    for file_key, sha_key in (
        ("factorial_metrics_jsonl", "factorial_metrics_jsonl_sha256"),
        ("factorial_metrics_csv", "factorial_metrics_csv_sha256"),
    ):
        artifact = result_dir / str(manifest.get(file_key) or "")
        _require_sha(artifact, str(manifest.get(sha_key) or ""), f"{output_dir.name} {file_key}")
    for cell in cells:
        cell_id = str(cell["cell_id"])
        _require(
            int(cell.get("num_samples", -1)) == EXPECTED_EVENT_COUNT
            and float(cell.get("mean_evidence_count", -1)) == 5.0
            and float(cell.get("parse_error_rate", -1)) == 0.0,
            f"{output_dir.name} {cell_id} metrics are not K5/n=1234/parse-clean",
        )
        for file_key, sha_key in (
            ("metrics_file", "metrics_sha256"),
            ("confusion_matrix_file", "confusion_matrix_sha256"),
            ("predictions_file", "predictions_sha256"),
        ):
            artifact = result_dir / str(cell.get(file_key) or "")
            _require_sha(artifact, str(cell.get(sha_key) or ""), f"{output_dir.name} {cell_id} {file_key}")
        predictions = result_dir / str(cell["predictions_file"])
        _validate_index_domain(predictions, field="sample_idx", expected_count=EXPECTED_EVENT_COUNT)
    return {"manifest_sha256": manifest_sha, "cells": list(EXPECTED_CELLS)}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    matrix = _validate_matrix(args.matrix_root, expected_manifest_sha256=args.expected_matrix_sha256)
    verifier_n = args.output_root / "verifier_n"
    verifier_s = args.output_root / "verifier_s"
    checkpoints = {
        "V_N": _validate_checkpoint(
            args.n_run_dir,
            role="V_N",
            expected_adapter_sha256=args.expected_n_adapter_sha256,
        ),
        "V_S": _validate_checkpoint(
            args.s_run_dir,
            role="V_S",
            expected_adapter_sha256=args.expected_s_adapter_sha256,
            expected_completion_sha256=args.expected_s_completion_sha256,
        ),
    }
    _validate_cap_manifest(
        args.n_cap_manifest,
        expected_sha256=args.expected_n_cap_manifest_sha256,
        expected_adapter_sha256=args.expected_n_adapter_sha256,
    )
    n_input = _validate_prepared_input(
        verifier_n,
        matrix_root=args.matrix_root,
        expected_manifest_sha256=args.expected_n_input_sha256,
        expected_matrix_sha256=args.expected_matrix_sha256,
    )
    s_input = _validate_prepared_input(
        verifier_s,
        matrix_root=args.matrix_root,
        expected_manifest_sha256=args.expected_s_input_sha256,
        expected_matrix_sha256=args.expected_matrix_sha256,
    )
    _require(
        n_input["unique_rows_sha256"] == s_input["unique_rows_sha256"]
        and n_input["cells"] == s_input["cells"],
        "V_N and V_S prepared inputs are not byte-identical on rows/mappings/build provenance",
    )
    n_raw = _validate_raw_logits(
        verifier_n,
        expected_input_sha256=args.expected_n_input_sha256,
        expected_adapter_sha256=args.expected_n_adapter_sha256,
        expected_manifest_sha256=args.expected_n_raw_manifest_sha256,
    )
    n_materialized = _validate_materialized(
        verifier_n,
        expected_input_sha256=args.expected_n_input_sha256,
        expected_raw_manifest_sha256=args.expected_n_raw_manifest_sha256,
        expected_raw_logits_sha256=n_raw["raw_logits_sha256"],
        expected_adapter_sha256=args.expected_n_adapter_sha256,
        expected_manifest_sha256=args.expected_n_materialized_manifest_sha256,
    )

    s_status: dict[str, Any] = {"status": "pending_raw_logits"}
    s_raw_dir = verifier_s / "raw_logits"
    s_materialized_dir = verifier_s / "materialized"
    if s_raw_dir.exists():
        _require((s_raw_dir / "manifest.json").is_file(), "V_S raw_logits exists without a complete manifest; refusing force replacement")
        s_raw = _validate_raw_logits(
            verifier_s,
            expected_input_sha256=args.expected_s_input_sha256,
            expected_adapter_sha256=args.expected_s_adapter_sha256,
            expected_manifest_sha256=None,
        )
        s_status = {"status": "raw_logits_complete", **s_raw}
        if s_materialized_dir.exists():
            _require((s_materialized_dir / "matrix_manifest.json").is_file(), "V_S materialized exists without a complete manifest")
            materialized = _validate_materialized(
                verifier_s,
                expected_input_sha256=args.expected_s_input_sha256,
                expected_raw_manifest_sha256=s_raw["manifest_sha256"],
                expected_raw_logits_sha256=s_raw["raw_logits_sha256"],
                expected_adapter_sha256=args.expected_s_adapter_sha256,
                expected_manifest_sha256=None,
            )
            s_status = {"status": "materialized_complete", **s_raw, "materialized": materialized}
    else:
        _require(not s_materialized_dir.exists(), "V_S materialized output exists without raw logits")
    if args.require_s_complete:
        _require(s_status["status"] == "materialized_complete", "V_S raw logits and fanout are not both complete")

    return {
        "schema_version": "no-map-fixed5-vs-only-resume-contract-v0.1",
        "status": "ready",
        "scope": "val_only_fixed_k5_common_support",
        "event_count": EXPECTED_EVENT_COUNT,
        "cells": list(EXPECTED_CELLS),
        "matrix": matrix,
        "checkpoints": checkpoints,
        "verifier_n_reused": {
            "input": n_input,
            "raw_logits": n_raw,
            "materialized": n_materialized,
        },
        "verifier_s": {"input": s_input, **s_status},
        "gpu_work_allowed": ["V_S checkpoint-800 raw-logit inference"],
        "gpu_work_forbidden": ["V_N inference", "training", "test inference"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-run-dir", type=Path, required=True)
    parser.add_argument("--s-run-dir", type=Path, required=True)
    parser.add_argument("--n-cap-manifest", type=Path, required=True)
    parser.add_argument("--expected-matrix-sha256", required=True)
    parser.add_argument("--expected-n-adapter-sha256", required=True)
    parser.add_argument("--expected-s-adapter-sha256", required=True)
    parser.add_argument("--expected-s-completion-sha256", required=True)
    parser.add_argument("--expected-n-cap-manifest-sha256", required=True)
    parser.add_argument("--expected-n-input-sha256", required=True)
    parser.add_argument("--expected-s-input-sha256", required=True)
    parser.add_argument("--expected-n-raw-manifest-sha256", required=True)
    parser.add_argument("--expected-n-materialized-manifest-sha256", required=True)
    parser.add_argument("--require-s-complete", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        payload = validate(args)
    except (ResumeContractError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
