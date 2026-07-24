#!/usr/bin/env python3
"""Fail-closed contracts for the clean no-map checkpoint-800 tail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ContractError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected JSON object: {path}")
    return payload


def _stable_sha(path: Path) -> dict[str, Any]:
    before = path.stat()
    if before.st_size <= 0:
        raise ContractError(f"empty artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity = (before.st_size, before.st_mtime_ns, before.st_ino)
    if identity != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ContractError(f"artifact changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "inode": before.st_ino,
    }


def _yaml_seed(path: Path, expected: int) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError(f"invalid YAML object: {path}")
    train = payload.get("sft_train")
    recorded = train.get("seed") if isinstance(train, Mapping) else None
    if recorded is not None and (type(recorded) is not int or recorded != expected):
        raise ContractError(f"{path}: seed={recorded!r}, expected {expected} or omitted")
    effective = expected if recorded is None else recorded
    if effective != expected:
        raise ContractError(f"{path}: effective seed mismatch")
    return {"recorded": recorded, "effective": effective, **_stable_sha(path)}


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                raise ContractError(f"blank JSONL line: {path}:{count + 1}")
            count += 1
    return count


def _expect_sha(path: Path, expected: str) -> dict[str, Any]:
    artifact = _stable_sha(path)
    if artifact["sha256"] != expected:
        raise ContractError(
            f"SHA mismatch for {path}: {artifact['sha256']} != {expected}"
        )
    return artifact


def validate_inputs(contract_path: Path) -> dict[str, Any]:
    contract = _read_json(contract_path)
    if contract.get("schema_version") != "no-map-fixed800-input-contract-v0.1":
        raise ContractError("unexpected input-contract schema")
    if contract.get("standard_clean_results_audit_slot_mutable") is not False:
        raise ContractError("contract must explicitly forbid clean-audit slot mutation")
    seed = int(contract.get("effective_seed", -1))
    if seed != 42:
        raise ContractError("clean no-map tail is frozen to effective seed 42")
    selector = str(contract.get("selector_name") or "")
    fingerprint = str(contract.get("weight_fingerprint") or "")
    if not selector or len(fingerprint) != 12:
        raise ContractError("missing selector or weight fingerprint")

    weights_cfg = contract["weights"]
    weight_file = Path(weights_cfg["file"])
    weight_manifest_path = Path(weights_cfg["manifest"])
    weight_file_artifact = _expect_sha(weight_file, str(weights_cfg["sha256"]))
    weight_manifest_artifact = _expect_sha(
        weight_manifest_path, str(weights_cfg["manifest_sha256"])
    )
    weight_manifest = _read_json(weight_manifest_path)
    supervision = weight_manifest.get("supervision_contract") or {}
    zero_reads = (
        "gold_label_read_count",
        "oracle_read_row_count",
        "teacher_read_count",
        "utility_read_count",
        "reward_read_count",
    )
    if (
        weight_manifest.get("training_supervision") != "structure_only"
        or weight_manifest.get("weight_fingerprint") != fingerprint
        or (weight_manifest.get("params") or {}).get("map_ablation_mode") != "no_map"
        or int(weight_manifest.get("n_train_rows", 0)) != 10065
        or int(weight_manifest.get("n_val_rows", 0)) != 1274
        or supervision.get("core_supervision_mode") != "structure_only"
        or any(int(supervision.get(key, -1)) != 0 for key in zero_reads)
    ):
        raise ContractError("weight manifest violates full structure-only/no-map contract")
    weights_payload = _read_json(weight_file)
    metadata = weights_payload.get("metadata") or {}
    if (
        metadata.get("map_ablation_mode") != "no_map"
        or metadata.get("supervision_mode") != "structure_only"
    ):
        raise ContractError("weights payload lacks no-map/structure-only provenance")

    trace_root = Path(contract["traces"]["root"])
    traces: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        expected = contract["traces"]["splits"][split]
        path = trace_root / f"selection_trace_{split}.jsonl"
        artifact = _expect_sha(path, str(expected["sha256"]))
        count = _line_count(path)
        manifest_path = trace_root / f"manifest_{split}.json"
        manifest = _read_json(manifest_path)
        if (
            count != int(expected["rows"])
            or int(manifest.get("n_input_rows", -1)) != count
            or int(manifest.get("n_trace_rows", -1)) != count
            or manifest.get("split") != split
            or manifest.get("selector_name") != selector
            or manifest.get("weight_fingerprint") != fingerprint
            or (manifest.get("params") or {}).get("map_ablation_mode") != "no_map"
            or (manifest.get("params") or {}).get("weight_file") != str(weight_file)
        ):
            raise ContractError(f"{split} trace manifest/content contract failed")
        traces[split] = {
            **artifact,
            "rows": count,
            "manifest": _stable_sha(manifest_path),
        }

    builds_cfg = contract["builds"]
    base_root = Path(builds_cfg["base_root"])
    run_root = Path(builds_cfg["run_root"])
    report_path = base_root / "build/build_report.json"
    report = _read_json(report_path)
    policy = report.get("prompt_evidence") or {}
    expected_policy = contract["training_prompt_policy"]
    if (
        report.get("val_only") is not False
        or sorted(report.get("built_splits") or []) != ["test", "train", "val"]
        or report.get("expected_selector_name") != selector
        or policy.get("policy") != expected_policy["policy"]
        or int(policy.get("min_evidence_count", -1)) != int(expected_policy["min_evidence_count"])
        or int(policy.get("max_evidence_count", -1)) != int(expected_policy["max_evidence_count"])
    ):
        raise ContractError("natural-K verifier build report violates minmax(5,10) contract")
    builds: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        expected = builds_cfg["splits"][split]
        base_path = base_root / f"build/build_{split}.jsonl"
        run_path = run_root / f"build/build_{split}.jsonl"
        base_artifact = _expect_sha(base_path, str(expected["sha256"]))
        run_artifact = _expect_sha(run_path, str(expected["sha256"]))
        rows = _line_count(base_path)
        split_report = (report.get("splits") or {}).get(split) or {}
        if (
            rows != int(expected["rows"])
            or int(split_report.get("n_source_rows", -1)) != rows
            or int(split_report.get("n_rows", -1)) != rows
            or split_report.get("source_path") != str(trace_root / f"selection_trace_{split}.jsonl")
            or split_report.get("selector_names") != {selector: rows}
        ):
            raise ContractError(f"{split} verifier build/report contract failed")
        builds[split] = {"rows": rows, "base": base_artifact, "run": run_artifact}
    configs = {
        "base": _yaml_seed(base_root / "train.resolved.yaml", seed),
        "run": _yaml_seed(run_root / "train.resolved.yaml", seed),
    }
    return {
        "schema_version": "no-map-fixed800-clean-input-audit-v0.1",
        "status": "ready",
        "effective_seed": seed,
        "weight_fingerprint": fingerprint,
        "selector_name": selector,
        "weights": {"file": weight_file_artifact, "manifest": weight_manifest_artifact},
        "traces": traces,
        "builds": builds,
        "configs": configs,
        "training_prompt_policy": expected_policy,
        "diagnostic_prompt_policy": {"policy": "fixed_topk", "k": 5},
        "standard_clean_results_audit_slot_mutated": False,
        "contract": _stable_sha(contract_path),
    }


def validate_checkpoint(run_root: Path, contract_path: Path, step: int) -> dict[str, Any]:
    contract = _read_json(contract_path)
    configured_root = Path(contract["builds"]["run_root"])
    if run_root.resolve() != configured_root.resolve():
        raise ContractError(f"run root differs from frozen input contract: {run_root}")
    train = run_root / "train"
    if (train / "training_complete.json").exists():
        raise ContractError("training_complete.json exists; fixed-step artifact is invalid")
    checkpoint = train / f"checkpoint-{step}"
    adapter = _stable_sha(checkpoint / "adapter_model.safetensors")
    adapter_config = _stable_sha(checkpoint / "adapter_config.json")
    parsed_adapter_config = _read_json(checkpoint / "adapter_config.json")
    if not parsed_adapter_config.get("base_model_name_or_path") or not parsed_adapter_config.get("peft_type"):
        raise ContractError("checkpoint adapter_config is incomplete")
    state_path = train / "latest_state/trainer_state.json"
    state = _read_json(state_path)
    progress = int(state.get("global_step", -1))
    if progress < step:
        raise ContractError(f"latest_state global_step={progress}, expected >= {step}")
    seed = int(contract["effective_seed"])
    return {
        "schema_version": "no-map-fixed800-checkpoint-contract-v0.1",
        "status": "ready",
        "role": "V_N",
        "checkpoint": f"checkpoint-{step}",
        "checkpoint_step": step,
        "progress_step": progress,
        "training_complete_present": False,
        "seed": _yaml_seed(run_root / "train.resolved.yaml", seed),
        "runtime_config": _yaml_seed(train / "config.resolved.yaml", seed),
        "adapter": adapter,
        "adapter_config": adapter_config,
        "latest_state": _stable_sha(state_path),
        "build_sha256": {
            split: str(contract["builds"]["splits"][split]["sha256"])
            for split in ("train", "val", "test")
        },
        "weight_fingerprint": str(contract["weight_fingerprint"]),
    }


def _iter_process_cmdlines(proc_root: Path) -> Iterable[tuple[int, tuple[str, ...]]]:
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = tuple(part.decode(errors="replace") for part in raw.split(b"\0") if part)
        yield int(entry.name), argv


def count_accelerate(proc_root: Path) -> dict[str, Any]:
    pids = []
    for pid, argv in _iter_process_cmdlines(proc_root):
        for index, token in enumerate(argv[:-1]):
            if Path(token).name == "accelerate" and argv[index + 1] == "launch":
                pids.append(pid)
                break
    return {"schema_version": "global-accelerate-audit-v0.1", "status": "ready", "count": len(pids), "pids": sorted(pids)}


def _emit(payload: Mapping[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inputs = sub.add_parser("inputs")
    inputs.add_argument("--contract", type=Path, required=True)
    inputs.set_defaults(func=lambda args: validate_inputs(args.contract))
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--run-root", type=Path, required=True)
    checkpoint.add_argument("--contract", type=Path, required=True)
    checkpoint.add_argument("--step", type=int, default=800)
    checkpoint.set_defaults(func=lambda args: validate_checkpoint(args.run_root, args.contract, args.step))
    accelerate = sub.add_parser("accelerate")
    accelerate.add_argument("--proc-root", type=Path, default=Path("/proc"))
    accelerate.set_defaults(func=lambda args: count_accelerate(args.proc_root))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return _emit(args.func(args))
    except (ContractError, FileNotFoundError, json.JSONDecodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
