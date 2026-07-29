#!/usr/bin/env python3
"""Frozen LIAR-RAW mixed-arm verifier fine-tuning experiment.

This is the single public entry point for the experiment.  The deliberately
separate phases keep test gold outside model-facing processes:

* ``prepare`` delegates artifact auditing and tokenizer-specific prompt export;
* ``train`` writes a fully resolved LoRA config and launches the existing
  label-token trainer (or only audits the launch with ``--dry-run``);
* ``infer`` obtains raw A--F logits with the frozen HF/PEFT forward path;
* ``analyze`` joins the separately stored gold data and computes panel results.

The implementation is fail-closed around the preregistered model, seed,
assignment, prompt-length, and completion contracts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/analysis/evitrace_cross_verifier_finetune_v1"
)
DEFAULT_PYTHON = Path("/data/liaozijie/conda/accelerate-fc/bin/python")
DEFAULT_ACCELERATE = Path("/data/liaozijie/conda/accelerate-fc/bin/accelerate")
DEFAULT_DEEPSPEED = (
    PROJECT_ROOT / "configs/deepspeed/deepspeed_zero2_bsz1_ga4.json"
)
DEFAULT_MODELS = {
    "qwen3": Path("/data/models/Qwen3-4B-Instruct-2507"),
    "llama31": Path("/data/models/Meta-Llama-3.1-8B-Instruct"),
}
FORMAL_SEEDS = (20260724, 20260725, 20260726)
ASSIGNMENTS = ("a", "b")
LETTERS = ("A", "B", "C", "D", "E", "F")
LABELS = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))
MAX_LENGTH = 2048
EXPECTED_MAIN = 1_250
EXPECTED_ORDER = 1_152
# Artifact audit correction: 7,448 is sum(K) over all 1,250 Main claims.
# The 1,152 final-order-eligible claims contribute 6,996 prefix pairs; the 98
# pre-excluded identical-order claims account for the remaining 452.
EXPECTED_PREFIX = 6_996
EXPECTED_VAL = 1_274
EXPECTED_EVAL_LOGICAL_RESULTS = (
    2 * EXPECTED_MAIN
    + 2 * EXPECTED_ORDER
    + 2 * EXPECTED_PREFIX
    + 2 * EXPECTED_VAL
    + EXPECTED_VAL
    + EXPECTED_VAL
)
EXPECTED_RUNS = 12


class FinetuneExperimentError(RuntimeError):
    """Raised when a frozen experiment contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FinetuneExperimentError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FinetuneExperimentError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise FinetuneExperimentError(
                    f"Expected object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def repair_truncated_jsonl_tail(path: str | Path) -> bool:
    """Atomically drop one interrupted, non-newline-terminated JSONL tail."""

    source = Path(path)
    if not source.is_file():
        return False
    data = source.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    lines = data.splitlines(keepends=True)
    if not lines:
        return False
    try:
        json.loads(lines[-1])
    except (json.JSONDecodeError, UnicodeDecodeError):
        repaired = b"".join(lines[:-1])
        temporary = source.with_name(f".{source.name}.repair.{os.getpid()}")
        temporary.write_bytes(repaired)
        os.replace(temporary, source)
        return True
    return False


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return count


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def file_metadata(path: str | Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        payload["rows"] = int(rows)
    return payload


def _verify_file_metadata(metadata: Mapping[str, Any], *, name: str) -> Path:
    path = Path(str(metadata.get("path") or "")).resolve()
    if not path.is_file():
        raise FinetuneExperimentError(f"Prepared file is missing ({name}): {path}")
    expected_sha = str(metadata.get("sha256") or "")
    if not expected_sha or sha256_file(path) != expected_sha:
        raise FinetuneExperimentError(f"Prepared file SHA mismatch ({name}): {path}")
    if metadata.get("bytes") is not None and path.stat().st_size != int(
        metadata["bytes"]
    ):
        raise FinetuneExperimentError(f"Prepared file size mismatch ({name}): {path}")
    return path


def _walk_file_metadata(value: object, prefix: str = "") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield prefix or "file", value
            return
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_file_metadata(child, child_prefix)


def load_prepared_manifest(
    path: str | Path,
    *,
    model_facing: bool = False,
) -> tuple[Path, dict[str, Any], str]:
    manifest_path = Path(path).resolve()
    manifest = read_json(manifest_path)
    if not bool(manifest.get("complete")):
        raise FinetuneExperimentError(
            f"Prepared manifest is not complete: {manifest_path}"
        )
    experiment = str(manifest.get("experiment") or "")
    if experiment and experiment != "evitrace_cross_verifier_finetune_v1":
        raise FinetuneExperimentError(
            f"Prepared manifest belongs to another experiment: {experiment}"
        )
    if not model_facing:
        checked = 0
        for name, metadata in _walk_file_metadata(
            {
                "prepared_files": manifest.get("prepared_files", {}),
                "models": manifest.get("models", {}),
            }
        ):
            _verify_file_metadata(metadata, name=name)
            checked += 1
        if checked == 0:
            raise FinetuneExperimentError("Prepared manifest contains no hashed files")
    # A model-facing process validates only the explicitly selected gold-free
    # registry later.  It must not hash train/val files because those contain
    # labels, nor the independent test-gold file.
    return manifest_path, manifest, sha256_file(manifest_path)


def _normalize_model_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "qwen3": "qwen3",
        "qwen34b": "qwen3",
        "qwen34binstruct2507": "qwen3",
        "llama31": "llama31",
        "llama318b": "llama31",
        "metallama318binstruct": "llama31",
    }
    if normalized not in aliases:
        raise FinetuneExperimentError(
            f"Unsupported backbone {value!r}; use qwen3 or llama31"
        )
    return aliases[normalized]


def _normalize_assignment(value: str) -> str:
    assignment = value.strip().lower().removeprefix("assignment_")
    if assignment not in ASSIGNMENTS:
        raise FinetuneExperimentError("--assignment must be a or b")
    return assignment


def _model_entry(manifest: Mapping[str, Any], model_name: str) -> Mapping[str, Any]:
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise FinetuneExperimentError("Prepared manifest has no models object")
    aliases = (
        model_name,
        "qwen3_4b_2507" if model_name == "qwen3" else "llama31_8b",
        "qwen" if model_name == "qwen3" else "llama",
    )
    for alias in aliases:
        entry = models.get(alias)
        if isinstance(entry, Mapping):
            return entry
    raise FinetuneExperimentError(
        f"Prepared manifest has no tokenizer-specific entry for {model_name}"
    )


def _find_file_metadata(
    container: Mapping[str, Any],
    candidates: Sequence[str],
) -> Mapping[str, Any]:
    sources: list[Mapping[str, Any]] = [container]
    for key in ("files", "prepared_files", "artifacts"):
        child = container.get(key)
        if isinstance(child, Mapping):
            sources.append(child)
    for source in sources:
        for candidate in candidates:
            value = source.get(candidate)
            if isinstance(value, Mapping) and value.get("path"):
                return value
    raise FinetuneExperimentError(
        f"None of the required prepared files exists: {', '.join(candidates)}"
    )


def _prepared_model_file(
    manifest: Mapping[str, Any],
    model_name: str,
    *candidates: str,
) -> Path:
    metadata = _find_file_metadata(_model_entry(manifest, model_name), candidates)
    return _verify_file_metadata(metadata, name=f"{model_name}:{candidates[0]}")


def _formal_run_id(model_name: str, assignment: str, seed: int) -> str:
    return f"{model_name}__assignment_{assignment}__seed_{seed}"


def _run_root(
    experiment_root: Path,
    model_name: str,
    assignment: str,
    seed: int,
    *,
    smoke: bool,
) -> Path:
    if smoke:
        return experiment_root / "smoke" / model_name
    return (
        experiment_root
        / "runs"
        / model_name
        / f"assignment_{assignment}"
        / f"seed_{seed}"
    )


def _copy_smoke_subset(
    source: Path,
    destination: Path,
    *,
    pair_by_event: bool,
    limit: int,
) -> int:
    rows = read_jsonl(source)
    if not pair_by_event:
        selected = rows[:limit]
    else:
        by_event: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_event.setdefault(str(row.get("event_id") or ""), []).append(row)
        selected = []
        for event_id in sorted(by_event):
            group = by_event[event_id]
            if {str(item.get("evidence_arm")) for item in group} == {
                "evitrace",
                "s4",
            }:
                selected.extend(group)
            if len(selected) >= limit:
                break
    if not selected:
        raise FinetuneExperimentError(f"Cannot build smoke subset from {source}")
    return atomic_write_jsonl(destination, selected)


def build_train_config(
    *,
    model_name: str,
    model_path: Path,
    train_path: Path,
    val_path: Path,
    run_root: Path,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    train_dir = run_root / "train"
    return {
        "label_schema": "liar6",
        "output_dir": str(train_dir.resolve()),
        "eval_output_dir": str((run_root / "eval").resolve()),
        "prompt_stats_output_dir": str((run_root / "prompt_stats").resolve()),
        "data": {
            "train_candidates": str(train_path.resolve()),
            "val_candidates": str(val_path.resolve()),
            # Required by generic inference context; formal inference uses its
            # own gold-free registry and never reads this placeholder.
            "test_candidates": str(val_path.resolve()),
        },
        "model_name_or_path": str(model_path.resolve()),
        "baseline": {
            "variant": f"evitrace_cross_verifier_finetune_v1_{model_name}",
            "chunking_strategy": "sentence_trace_method",
            "label_schema": "liar6",
            "model_name_or_path": str(model_path.resolve()),
        },
        "sft_train": {
            "seed": int(seed),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2.0e-5,
            "num_train_epochs": 2.0 if smoke else 12.0,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "bf16": True,
            "max_length": MAX_LENGTH,
            "logging_steps": 1 if smoke else 2,
            "save_steps": 1 if smoke else 100,
            "eval_steps": 1 if smoke else 100,
            "dataloader_num_workers": 0 if smoke else 4,
            "gradient_checkpointing": True,
            "use_flash_attention_2": True,
            "lr_scheduler_type": "cosine_with_restarts",
            "lr_scheduler_kwargs": {"num_cycles": 2},
            "max_grad_norm": 1.0,
            "padding": "longest",
            "use_length_bucket": True,
            "empty_cache_steps": 0,
            "empty_cache_on_eval": True,
            "empty_cache_on_save": True,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "early_stopping_patience": 8,
            "eval_log_predictions": 0,
            "label_schema": "liar6",
            "logit_adjust": {"enabled": False, "tau": 1.0},
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "modules_to_save": None,
            },
            "label_token_ce": {
                "label_prefix": "Label:",
                "early_stopping_metric": "arm_balanced_macro_f1",
                "class_weights": {
                    "pants-fire": 1.2,
                    "false": 1.0,
                    "barely-true": 1.5,
                    "half-true": 1.0,
                    "mostly-true": 1.0,
                    "true": 1.8,
                },
                "ordinal_loss": {
                    "enabled": True,
                    "alpha": 0.2,
                    "normalize_distance": True,
                    "alpha_warmup_ratio": 0.3,
                },
            },
            "resolved_output_dir": True,
            "save_latest_state": True,
            "resume_latest_state": True,
            "latest_state_save_steps": 1 if smoke else 100,
        },
        "tracking": {"enabled": False, "backend": "none"},
        "train": {"deepspeed_config": str(DEFAULT_DEEPSPEED.resolve())},
        "experiment": {
            "name": _formal_run_id(
                model_name,
                "a" if smoke else run_root.parent.name.removeprefix("assignment_"),
                seed,
            )
        },
    }


def _training_complete(train_dir: Path) -> bool:
    marker = train_dir / "training_complete.json"
    best = train_dir / "best"
    if not marker.is_file() or not best.is_dir():
        return False
    try:
        payload = read_json(marker)
    except (OSError, json.JSONDecodeError, FinetuneExperimentError):
        return False
    if "training_complete" in payload:
        marker_complete = payload.get("training_complete") is True
    elif "completed" in payload:
        marker_complete = payload.get("completed") is True
    else:
        marker_complete = payload.get("complete") is True
    if not marker_complete:
        return False
    if not (best / "adapter_config.json").is_file():
        return False
    return any(
        (best / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def prepare_phase(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from cross_verifier_finetune_prepare import prepare_dataset

    return prepare_dataset(args)


def train_phase(args: argparse.Namespace) -> dict[str, Any]:
    if args.force:
        raise FinetuneExperimentError(
            "Training --force is intentionally unsupported because an interrupted "
            "overwrite could leave a stale completion marker. Use the existing "
            "latest_state resume path, or choose a new --experiment-root."
        )
    manifest_path, manifest, manifest_sha = load_prepared_manifest(
        args.prepared_manifest
    )
    model_name = _normalize_model_name(args.model_name)
    assignment = _normalize_assignment(args.assignment)
    seed = int(args.seed)
    smoke = bool(args.smoke)
    if not smoke and seed not in FORMAL_SEEDS:
        raise FinetuneExperimentError(
            f"Formal runs require seed in {FORMAL_SEEDS}; received {seed}"
        )
    model_path = Path(args.model_path or DEFAULT_MODELS[model_name]).resolve()
    if model_path != DEFAULT_MODELS[model_name].resolve():
        raise FinetuneExperimentError(
            f"Frozen {model_name} path is {DEFAULT_MODELS[model_name]}, got {model_path}"
        )
    if not model_path.is_dir():
        raise FinetuneExperimentError(f"Model directory is missing: {model_path}")
    tokenizer_sha = _tokenizer_fingerprint(model_path)
    prepared_tokenizer_sha = _prepared_tokenizer_fingerprint(
        manifest,
        model_name,
    )
    if tokenizer_sha != prepared_tokenizer_sha:
        raise FinetuneExperimentError(
            f"{model_name} tokenizer/config drifted since prepare: "
            f"current={tokenizer_sha}, prepared={prepared_tokenizer_sha}"
        )
    if not DEFAULT_DEEPSPEED.is_file():
        raise FinetuneExperimentError(
            f"Frozen DeepSpeed config is missing: {DEFAULT_DEEPSPEED}"
        )

    train_source = _prepared_model_file(
        manifest,
        model_name,
        f"train_assignment_{assignment}",
        f"assignment_{assignment}_train",
    )
    val_source = _prepared_model_file(
        manifest,
        model_name,
        "val_paired",
        "paired_val",
    )
    experiment_root = Path(args.experiment_root).resolve()
    run_root = _run_root(
        experiment_root,
        model_name,
        assignment,
        seed,
        smoke=smoke,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    train_path = train_source
    val_path = val_source
    if smoke:
        smoke_data = run_root / "data"
        train_path = smoke_data / "train.jsonl"
        val_path = smoke_data / "val.jsonl"
        _copy_smoke_subset(
            train_source,
            train_path,
            pair_by_event=False,
            limit=int(args.smoke_train_rows),
        )
        _copy_smoke_subset(
            val_source,
            val_path,
            pair_by_event=True,
            limit=int(args.smoke_val_rows),
        )

    config = build_train_config(
        model_name=model_name,
        model_path=model_path,
        train_path=train_path,
        val_path=val_path,
        run_root=run_root,
        seed=seed,
        smoke=smoke,
    )
    config_path = run_root / "train.resolved.yaml"
    config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if config_path.is_file():
        if config_path.read_text(encoding="utf-8") != config_text:
            raise FinetuneExperimentError(
                f"Resolved config drift at resumable/completed run: {config_path}"
            )
    else:
        temporary = config_path.with_name(f".{config_path.name}.tmp.{os.getpid()}")
        temporary.write_text(config_text, encoding="utf-8")
        os.replace(temporary, config_path)

    launch_contract = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_finetune_v1",
        "created_at": utc_now(),
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": manifest_sha,
        "run_id": (
            f"smoke_{model_name}"
            if smoke
            else _formal_run_id(model_name, assignment, seed)
        ),
        "backbone": model_name,
        "assignment_id": assignment,
        "seed": seed,
        "smoke": smoke,
        "config": file_metadata(config_path),
        "train_data": file_metadata(train_path, rows=len(read_jsonl(train_path))),
        "val_data": file_metadata(val_path, rows=len(read_jsonl(val_path))),
        "world_size": 4,
        "effective_batch_size": 16,
        "deepspeed": file_metadata(DEFAULT_DEEPSPEED),
        "base_model_sha256": _base_model_fingerprint(
            model_name,
            model_path,
            experiment_root / "base_model_fingerprints",
        ),
        "tokenizer_sha256": tokenizer_sha,
        "code": {
            "cross_verifier_finetune": file_metadata(Path(__file__).resolve()),
            "label_token_trainer": file_metadata(
                SRC_ROOT / "sft/label_token_trainer.py"
            ),
        },
    }
    launch_contract_path = run_root / "launch_contract.json"
    if launch_contract_path.is_file():
        existing_contract = read_json(launch_contract_path)
        ignored = {"created_at"}
        existing_stable = {
            key: value
            for key, value in existing_contract.items()
            if key not in ignored
        }
        candidate_stable = {
            key: value for key, value in launch_contract.items() if key not in ignored
        }
        if existing_stable != candidate_stable:
            raise FinetuneExperimentError(
                f"Resume launch contract drift; use a new output root or explicitly "
                f"--force after auditing: {launch_contract_path}"
            )
        launch_contract = existing_contract
    else:
        atomic_write_json(launch_contract_path, launch_contract)

    train_dir = run_root / "train"
    if _training_complete(train_dir):
        print(f"Training already complete and hashable: {train_dir}")
        return launch_contract
    if args.dry_run:
        print(f"Dry-run config written: {config_path}")
        return launch_contract

    python_bin = Path(args.python_bin).resolve()
    if not python_bin.is_file():
        raise FinetuneExperimentError(f"Required executable is missing: {python_bin}")
    gpu_ids = [part.strip() for part in str(args.gpu_ids).split(",") if part.strip()]
    if (
        len(gpu_ids) != 4
        or len(set(gpu_ids)) != 4
        or any(not item.isdigit() for item in gpu_ids)
    ):
        raise FinetuneExperimentError(
            "--gpu-ids must name exactly four distinct physical GPUs"
        )
    command = [
        str(python_bin),
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_processes",
        "4",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        str(DEFAULT_DEEPSPEED.resolve()),
        "-m",
        "sft.label_token_trainer",
        "--config",
        str(config_path.resolve()),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["SAVE_LATEST_TRAIN_STATE"] = "true"
    env["RESUME_LATEST_TRAIN_STATE"] = "true"
    print("Launching:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise FinetuneExperimentError(
            f"Training launcher exited with code {completed.returncode}; "
            f"resume state, if published, remains at {train_dir / 'latest_state'}"
        )
    if not _training_complete(train_dir):
        raise FinetuneExperimentError(
            f"Trainer returned success without a valid training_complete marker: {train_dir}"
        )
    return launch_contract


def _weight_files(model_path: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_path / index_name
        if index_path.is_file():
            payload = read_json(index_path)
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, Mapping) or not weight_map:
                raise FinetuneExperimentError(f"Invalid model index: {index_path}")
            paths = [model_path / str(name) for name in sorted(set(weight_map.values()))]
            if any(not path.is_file() for path in paths):
                raise FinetuneExperimentError(f"Model index references missing shard: {index_path}")
            return paths
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_path / name
        if path.is_file():
            return [path]
    raise FinetuneExperimentError(f"No model weights found under {model_path}")


def _base_model_fingerprint(model_name: str, model_path: Path, cache_root: Path) -> str:
    cache_path = cache_root / f"{model_name}.json"
    weight_files = _weight_files(model_path)
    identity_files = [
        path
        for path in (
            model_path / "config.json",
            model_path / "generation_config.json",
            *weight_files,
        )
        if path.is_file()
    ]
    stats = [
        {
            "path": path.relative_to(model_path).as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in identity_files
    ]
    if cache_path.is_file():
        cached = read_json(cache_path)
        if (
            cached.get("model_path") == str(model_path)
            and cached.get("file_stats") == stats
            and cached.get("sha256")
        ):
            return str(cached["sha256"])
    entries = [
        {
            "path": path.relative_to(model_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in identity_files
    ]
    digest = sha256_json(entries)
    atomic_write_json(
        cache_path,
        {
            "schema_version": 1,
            "model_path": str(model_path),
            "file_stats": stats,
            "files": entries,
            "sha256": digest,
        },
    )
    return digest


def _adapter_fingerprint(checkpoint: Path) -> tuple[str, Path]:
    config_path = checkpoint / "adapter_config.json"
    if not config_path.is_file():
        raise FinetuneExperimentError(f"No PEFT adapter_config.json under {checkpoint}")
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return (
                sha256_json(
                    {
                        "adapter_config.json": sha256_file(config_path),
                        name: sha256_file(path),
                    }
                ),
                path,
            )
    raise FinetuneExperimentError(f"No PEFT adapter weights under {checkpoint}")


def _tokenizer_fingerprint(model_path: Path) -> str:
    # Keep this byte-for-byte compatible with prepare's tokenizer-directory
    # fingerprint.  Prompt IDs are frozen at prepare time, so train and infer
    # must fail if any non-weight model/tokenizer/config file has drifted.
    weight_suffixes = {
        ".bin",
        ".ckpt",
        ".gguf",
        ".h5",
        ".msgpack",
        ".pt",
        ".pth",
        ".safetensors",
    }
    entries = [
        {
            "path": path.relative_to(model_path).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(candidate for candidate in model_path.rglob("*") if candidate.is_file())
        if path.suffix.lower() not in weight_suffixes
    ]
    if not entries:
        raise FinetuneExperimentError(f"No tokenizer files under {model_path}")
    return sha256_json(entries)


def _prepared_tokenizer_fingerprint(
    manifest: Mapping[str, Any],
    model_name: str,
) -> str:
    models = manifest.get("models")
    model_entry = models.get(model_name) if isinstance(models, Mapping) else None
    if not isinstance(model_entry, Mapping):
        raise FinetuneExperimentError(
            f"Prepared manifest has no model entry for {model_name}"
        )
    digest = str(model_entry.get("tokenizer_sha256") or "")
    if len(digest) != 64:
        raise FinetuneExperimentError(
            f"Prepared tokenizer fingerprint is missing for {model_name}"
        )
    return digest


def _resolve_gpu_id(value: str) -> str:
    requested = value.strip().lower()
    visible_env = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    visible_physical: set[int] | None = None
    if visible_env and visible_env not in {"-1", "none", "void"}:
        pieces = [piece.strip() for piece in visible_env.split(",") if piece.strip()]
        if not pieces or any(not piece.isdigit() for piece in pieces):
            raise FinetuneExperimentError(
                "CUDA_VISIBLE_DEVICES must contain physical integer IDs for "
                "this frozen launcher"
            )
        visible_physical = {int(piece) for piece in pieces}
    if requested != "auto":
        if not requested.isdigit():
            raise FinetuneExperimentError("--gpu-id must be auto or one integer")
        if visible_physical is not None and int(requested) not in visible_physical:
            raise FinetuneExperimentError(
                f"Requested physical GPU {requested} is outside inherited "
                f"CUDA_VISIBLE_DEVICES={visible_env}"
            )
        return requested
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    candidates: list[tuple[int, int]] = []
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
                physical_id = int(pieces[0])
                if visible_physical is None or physical_id in visible_physical:
                    candidates.append((int(pieces[1]), physical_id))
    if not candidates:
        raise FinetuneExperimentError("Cannot select a GPU from nvidia-smi")
    return str(max(candidates, key=lambda pair: (pair[0], -pair[1]))[1])


def _registry_rows(
    manifest: Mapping[str, Any],
    model_name: str,
    explicit_path: str | None,
) -> tuple[Path, list[dict[str, Any]]]:
    if explicit_path:
        path = Path(explicit_path).resolve()
        if not path.is_file():
            raise FinetuneExperimentError(f"Eval registry is missing: {path}")
    else:
        path = _prepared_model_file(
            manifest,
            model_name,
            "eval_registry",
            "test_eval_registry",
            "logical_registry",
        )
    rows = read_jsonl(path)
    if not rows:
        raise FinetuneExperimentError(f"Eval registry is empty: {path}")
    forbidden = {"gold_label", "gold_id", "target"}
    leaked = sorted(
        {
            key
            for row in rows
            for key in forbidden
            if key in row and row.get(key) not in (None, "")
        }
    )
    if leaked:
        raise FinetuneExperimentError(
            f"Model-facing eval registry leaks gold fields: {leaked}"
        )
    return path, rows


def _normalize_prompt_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    prefix_ids = tokenizer(
        "Label:",
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if not prefix_ids:
        raise FinetuneExperimentError("Label: prefix tokenized to an empty sequence")
    logical: list[dict[str, Any]] = []
    unique: dict[str, list[int]] = {}
    seen_logical: set[str] = set()
    for raw in rows:
        logical_id = str(raw.get("logical_id") or "")
        if not logical_id or logical_id in seen_logical:
            raise FinetuneExperimentError(f"Missing/duplicate logical_id: {logical_id!r}")
        seen_logical.add(logical_id)
        prompt_ids_raw = raw.get("prompt_input_ids")
        if not isinstance(prompt_ids_raw, list) or not prompt_ids_raw:
            raise FinetuneExperimentError(f"{logical_id}: missing prompt_input_ids")
        prompt_ids = [int(value) for value in prompt_ids_raw]
        declared_sha = str(
            raw.get("prompt_input_ids_sha256")
            or raw.get("input_ids_sha256")
            or ""
        )
        prompt_sha = sha256_json(prompt_ids)
        if declared_sha and declared_sha != prompt_sha:
            raise FinetuneExperimentError(
                f"{logical_id}: prepared prompt_input_ids SHA mismatch"
            )
        input_ids = prompt_ids + [int(value) for value in prefix_ids]
        if len(input_ids) + 1 > MAX_LENGTH:
            raise FinetuneExperimentError(
                f"{logical_id}: {len(input_ids) + 1} tokens exceeds {MAX_LENGTH}; "
                "truncation is forbidden"
            )
        input_sha = sha256_json(input_ids)
        if input_sha in unique and unique[input_sha] != input_ids:
            raise FinetuneExperimentError("SHA-256 collision across prompt token IDs")
        unique.setdefault(input_sha, input_ids)
        metadata = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "prompt",
                "prompt_text",
                "prompt_input_ids",
                "prompt_input_ids_sha256",
                "input_ids",
                "input_ids_sha256",
            }
        }
        metadata.update(
            {
                "logical_id": logical_id,
                "input_ids_sha256": input_sha,
                "token_count": len(input_ids),
            }
        )
        logical.append(metadata)
    return logical, unique


def _normalize_logits(logits: Sequence[float]) -> tuple[dict[str, float], dict[str, float]]:
    values = [float(value) for value in logits]
    if len(values) != len(LETTERS) or any(not math.isfinite(value) for value in values):
        raise FinetuneExperimentError(f"Invalid A-F logits: {values}")
    maximum = max(values)
    denominator = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    log_probs = {
        letter: value - denominator for letter, value in zip(LETTERS, values)
    }
    return log_probs, {letter: math.exp(value) for letter, value in log_probs.items()}


def _load_resume_scores(
    path: Path,
    *,
    base_sha: str,
    adapter_sha: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    scores: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        input_sha = str(row.get("input_ids_sha256") or "")
        expected_key = sha256_json(
            {
                "base_model_sha256": base_sha,
                "adapter_sha256": adapter_sha,
                "input_ids_sha256": input_sha,
            }
        )
        if row.get("cache_key") != expected_key:
            raise FinetuneExperimentError(
                f"Resume cache key mismatch for prompt {input_sha}"
            )
        if input_sha in scores:
            raise FinetuneExperimentError(f"Duplicate resume score: {input_sha}")
        _normalize_logits([row["logits"][letter] for letter in LETTERS])
        scores[input_sha] = row
    return scores


def _load_hf_peft_model(model_path: Path, checkpoint: Path):
    import torch

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from peft import PeftModel
    from sft.runtime.deps import flash_attn2_available
    from sft.runtime.model_loading import load_causal_lm_compatible_model

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if flash_attn2_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    base = load_causal_lm_compatible_model(
        str(model_path),
        use_mistral3_text_only=False,
        **kwargs,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    model.to(torch.device("cuda"))
    return model


def infer_phase(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path, manifest, manifest_sha = load_prepared_manifest(
        args.prepared_manifest,
        model_facing=True,
    )
    model_name = _normalize_model_name(args.model_name)
    assignment = _normalize_assignment(args.assignment)
    seed = int(args.seed)
    if seed not in FORMAL_SEEDS and not args.smoke:
        raise FinetuneExperimentError(
            f"Formal inference requires seed in {FORMAL_SEEDS}; received {seed}"
        )
    model_path = Path(args.model_path or DEFAULT_MODELS[model_name]).resolve()
    if model_path != DEFAULT_MODELS[model_name].resolve():
        raise FinetuneExperimentError("Inference model path differs from frozen model")
    experiment_root = Path(args.experiment_root).resolve()
    run_root = _run_root(
        experiment_root,
        model_name,
        assignment,
        seed,
        smoke=bool(args.smoke),
    )
    train_dir = run_root / "train"
    if not _training_complete(train_dir):
        raise FinetuneExperimentError(
            f"Run is not training_complete; inference is forbidden: {train_dir}"
        )
    checkpoint = train_dir / "best"
    adapter_sha, adapter_path = _adapter_fingerprint(checkpoint)
    base_sha = _base_model_fingerprint(
        model_name,
        model_path,
        experiment_root / "base_model_fingerprints",
    )
    tokenizer_sha = _tokenizer_fingerprint(model_path)
    prepared_tokenizer_sha = _prepared_tokenizer_fingerprint(
        manifest,
        model_name,
    )
    if tokenizer_sha != prepared_tokenizer_sha:
        raise FinetuneExperimentError(
            f"{model_name} tokenizer/config drifted since prepare: "
            f"current={tokenizer_sha}, prepared={prepared_tokenizer_sha}"
        )
    launch_contract_path = run_root / "launch_contract.json"
    if not launch_contract_path.is_file():
        raise FinetuneExperimentError(
            f"Run has no frozen launch contract: {launch_contract_path}"
        )
    launch_contract = read_json(launch_contract_path)
    launch_code = launch_contract.get("code")
    expected_code_paths = {
        "cross_verifier_finetune": Path(__file__).resolve(),
        "label_token_trainer": SRC_ROOT / "sft/label_token_trainer.py",
    }
    code_matches = isinstance(launch_code, Mapping) and all(
        isinstance(launch_code.get(name), Mapping)
        and launch_code[name].get("sha256") == sha256_file(path)
        for name, path in expected_code_paths.items()
    )
    if (
        launch_contract.get("prepared_manifest_sha256") != manifest_sha
        or launch_contract.get("base_model_sha256") != base_sha
        or launch_contract.get("tokenizer_sha256") != tokenizer_sha
        or launch_contract.get("config", {}).get("sha256")
        != sha256_file(run_root / "train.resolved.yaml")
        or not code_matches
    ):
        raise FinetuneExperimentError(
            f"Training/inference provenance drift for {run_root}"
        )
    if args.registry and not args.smoke:
        raise FinetuneExperimentError(
            "Formal inference must use the frozen eval_registry from the "
            "prepared manifest; --registry is smoke-only"
        )
    registry_path, registry_rows = _registry_rows(
        manifest,
        model_name,
        args.registry,
    )
    if args.max_logical_rows is not None:
        if not args.smoke:
            raise FinetuneExperimentError(
                "--max-logical-rows is restricted to non-statistical smoke inference"
            )
        if int(args.max_logical_rows) <= 0:
            raise FinetuneExperimentError("--max-logical-rows must be positive")
        registry_rows = registry_rows[: int(args.max_logical_rows)]

    if not args.dry_run:
        selected_gpu = _resolve_gpu_id(args.gpu_id)
        # This must precede importing sft.label_token_trainer, transformers,
        # PEFT, or torch so CUDA device enumeration cannot escape the scheduler
        # allocation.
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from sft.label_token_trainer import _build_label_token_ids
    from sft.runtime.model_loading import load_compatible_tokenizer

    tokenizer = load_compatible_tokenizer(str(model_path), trust_remote_code=True)
    label_ids, token_meta = _build_label_token_ids(
        tokenizer,
        label_prefix="Label:",
        letter_order=list(LETTERS),
    )
    if len(label_ids) != 6 or len(set(label_ids)) != 6:
        raise FinetuneExperimentError(f"A-F token IDs are not unique: {label_ids}")
    for letter in LETTERS:
        ids = tokenizer(f" {letter}", add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        if len(ids) != 1:
            raise FinetuneExperimentError(
                f"Frozen single-token label contract failed for {letter}: {ids}"
            )
    logical_rows, unique_prompts = _normalize_prompt_rows(registry_rows, tokenizer)

    run_id = (
        f"smoke_{model_name}"
        if args.smoke
        else _formal_run_id(model_name, assignment, seed)
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else experiment_root / "inference" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".infer.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise FinetuneExperimentError(
            f"Another inference process holds the run lock: {lock_path}"
        ) from exc
    unique_path = output_dir / "unique_logits.jsonl"
    logical_path = output_dir / "logical_results.jsonl"
    runtime_path = output_dir / "runtime_manifest.json"

    if args.force:
        for stale in (runtime_path, logical_path, unique_path):
            stale.unlink(missing_ok=True)
    if runtime_path.is_file() and not args.force:
        runtime = read_json(runtime_path)
        if runtime.get("complete"):
            invariants = (
                runtime.get("prepared_manifest_sha256") == manifest_sha
                and runtime.get("base_model_sha256") == base_sha
                and runtime.get("adapter_sha256") == adapter_sha
                and runtime.get("registry_sha256") == sha256_file(registry_path)
                and logical_path.is_file()
                and unique_path.is_file()
                and runtime.get("files", {}).get("logical_results", {}).get("sha256")
                == sha256_file(logical_path)
                and runtime.get("files", {}).get("unique_logits", {}).get("sha256")
                == sha256_file(unique_path)
            )
            if invariants:
                print(f"Inference already complete and hash-valid: {output_dir}")
                return runtime
            raise FinetuneExperimentError(
                f"Existing complete runtime manifest failed hash validation: {runtime_path}"
            )

    if repair_truncated_jsonl_tail(unique_path):
        print(f"Recovered one interrupted JSONL tail: {unique_path}", flush=True)
    existing = _load_resume_scores(
        unique_path,
        base_sha=base_sha,
        adapter_sha=adapter_sha,
    )
    unknown = set(existing) - set(unique_prompts)
    if unknown:
        raise FinetuneExperimentError(
            f"Resume cache contains {len(unknown)} prompts outside the registry"
        )
    pending = [
        (digest, ids)
        for digest, ids in sorted(unique_prompts.items())
        if digest not in existing
    ]
    if args.dry_run:
        dry = {
            "schema_version": 1,
            "complete": False,
            "dry_run": True,
            "run_id": run_id,
            "unique_prompt_count": len(unique_prompts),
            "pending_prompt_count": len(pending),
            "logical_result_count": len(logical_rows),
            "base_model_sha256": base_sha,
            "adapter_sha256": adapter_sha,
            "tokenizer_sha256": tokenizer_sha,
        }
        atomic_write_json(output_dir / "dry_run.json", dry)
        print(json.dumps(dry, indent=2))
        return dry

    import torch

    model = _load_hf_peft_model(model_path, checkpoint)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise FinetuneExperimentError("--batch-size must be positive")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise FinetuneExperimentError("Tokenizer has neither pad nor EOS token")
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device="cuda")
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        max_len = max(len(ids) for _digest, ids in chunk)
        input_rows = [
            ids + [int(pad_id)] * (max_len - len(ids)) for _digest, ids in chunk
        ]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for _digest, ids in chunk]
        input_tensor = torch.tensor(input_rows, dtype=torch.long, device="cuda")
        mask_tensor = torch.tensor(masks, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                use_cache=False,
            )
            positions = mask_tensor.sum(dim=1) - 1
            batch_indices = torch.arange(len(chunk), device="cuda")
            next_logits = outputs.logits[batch_indices, positions]
            restricted = next_logits.index_select(dim=-1, index=label_tensor).float().cpu()
        completed_rows: list[dict[str, Any]] = []
        for (input_sha, input_ids), values in zip(chunk, restricted.tolist()):
            log_probs, probabilities = _normalize_logits(values)
            pred_letter = max(LETTERS, key=lambda letter: log_probs[letter])
            cache_key = sha256_json(
                {
                    "base_model_sha256": base_sha,
                    "adapter_sha256": adapter_sha,
                    "input_ids_sha256": input_sha,
                }
            )
            row = {
                "cache_key": cache_key,
                "base_model_sha256": base_sha,
                "adapter_sha256": adapter_sha,
                "input_ids_sha256": input_sha,
                "token_count": len(input_ids),
                "logits": {
                    letter: float(value) for letter, value in zip(LETTERS, values)
                },
                "log_probs": log_probs,
                "probabilities": probabilities,
                "pred_letter": pred_letter,
                "pred_label": LETTER_TO_LABEL[pred_letter],
            }
            completed_rows.append(row)
            existing[input_sha] = row
        append_jsonl(unique_path, completed_rows)
        print(
            f"[{run_id}] scored {min(offset + len(chunk), len(pending))}/"
            f"{len(pending)} pending unique prompts",
            flush=True,
        )

    if set(existing) != set(unique_prompts):
        raise FinetuneExperimentError("Unique prompt inference is incomplete")
    expanded: list[dict[str, Any]] = []
    for logical in logical_rows:
        score = existing[str(logical["input_ids_sha256"])]
        row = dict(logical)
        row.update(
            {
                "run_id": run_id,
                "backbone": model_name,
                "assignment_id": assignment,
                "seed": seed,
                "logits": score["logits"],
                "log_probs": score["log_probs"],
                "probabilities": score["probabilities"],
                "pred_letter": score["pred_letter"],
                "pred_label": score["pred_label"],
            }
        )
        expanded.append(row)
    expanded.sort(key=lambda row: str(row["logical_id"]))
    logical_count = atomic_write_jsonl(logical_path, expanded)

    type_arm_counts: dict[str, int] = {}
    for row in expanded:
        key = f"{row.get('comparison_type')}::{row.get('evidence_arm')}"
        type_arm_counts[key] = type_arm_counts.get(key, 0) + 1
    if not args.smoke:
        expected_type_arm_counts = {
            "main::evitrace": EXPECTED_MAIN,
            "main::s4": EXPECTED_MAIN,
            "order_only::evitrace": EXPECTED_ORDER,
            "order_only::s4": EXPECTED_ORDER,
            "prefix::evitrace": EXPECTED_PREFIX,
            "prefix::s4": EXPECTED_PREFIX,
            "val_paired::evitrace": EXPECTED_VAL,
            "val_paired::s4": EXPECTED_VAL,
            "val_claim_only::claim_only": EXPECTED_VAL,
            "val_mismatched::mismatched": EXPECTED_VAL,
        }
        if type_arm_counts != expected_type_arm_counts:
            raise FinetuneExperimentError(
                "Formal logical-result inventory drift: "
                f"observed={type_arm_counts}, expected={expected_type_arm_counts}"
            )
        if logical_count != EXPECTED_EVAL_LOGICAL_RESULTS:
            raise FinetuneExperimentError(
                f"Expected {EXPECTED_EVAL_LOGICAL_RESULTS} formal logical results, "
                f"found {logical_count}"
            )
    runtime = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_finetune_v1",
        "created_at": utc_now(),
        "complete": True,
        "completion_is_effect_independent": True,
        "run_id": run_id,
        "backbone": model_name,
        "assignment_id": assignment,
        "seed": seed,
        "smoke": bool(args.smoke),
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": manifest_sha,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "model_path": str(model_path),
        "base_model_sha256": base_sha,
        "checkpoint_path": str(checkpoint),
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha,
        "adapter_config": file_metadata(checkpoint / "adapter_config.json"),
        "training_complete": file_metadata(train_dir / "training_complete.json"),
        "train_config": file_metadata(run_root / "train.resolved.yaml"),
        "launch_contract": file_metadata(launch_contract_path),
        "tokenizer_sha256": tokenizer_sha,
        "label_token_ids": dict(zip(LETTERS, label_ids)),
        "label_token_meta": token_meta,
        "max_length": MAX_LENGTH,
        "scoring": "hf_peft_last_nonpadding_raw_label_token_logits",
        "logit_adjustment": {"enabled": False},
        "counts": {
            "unique_logits": len(existing),
            "logical_results": logical_count,
            "by_type_arm": type_arm_counts,
        },
        "files": {
            "unique_logits": file_metadata(unique_path, rows=len(existing)),
            "logical_results": file_metadata(logical_path, rows=logical_count),
        },
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "peft")
        },
        "code": {
            "cross_verifier_finetune": file_metadata(Path(__file__).resolve()),
            "label_token_trainer": file_metadata(
                SRC_ROOT / "sft/label_token_trainer.py"
            ),
        },
    }
    atomic_write_json(runtime_path, runtime)
    print(f"Inference complete: {runtime_path}")
    return runtime


def analyze_phase(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from cross_verifier_finetune_analysis import analyze_results

    return analyze_results(
        prepared_manifest=Path(args.prepared_manifest).resolve(),
        result_paths=[Path(path).resolve() for path in args.result],
        output_dir=Path(args.output_dir).resolve(),
        bootstrap=int(args.bootstrap),
        permutations=int(args.randomization),
        seed=int(args.seed),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LIAR-RAW fair mixed-arm fine-tuned verifier experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Audit canonical artifacts and export tokenizer-specific datasets",
    )
    prepare.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "prepared"))
    prepare.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    prepare.add_argument("--max-length", type=int, default=MAX_LENGTH)
    prepare.add_argument(
        "--qwen-model-path",
        dest="qwen_model_path",
        default=str(DEFAULT_MODELS["qwen3"]),
    )
    prepare.add_argument(
        "--llama-model-path",
        dest="llama_model_path",
        default=str(DEFAULT_MODELS["llama31"]),
    )
    for split in ("train", "val", "test"):
        prepare.add_argument(f"--build-{split}", default=None)
        prepare.add_argument(f"--evitrace-{split}", default=None)
        prepare.add_argument(f"--s4-{split}", default=None)
    prepare.set_defaults(handler=prepare_phase)

    train = subparsers.add_parser(
        "train",
        help="Write a frozen LoRA config and launch/resume one run",
    )
    train.add_argument("--prepared-manifest", required=True)
    train.add_argument("--model-name", required=True)
    train.add_argument("--model-path", default=None)
    train.add_argument("--assignment", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--experiment-root", default=str(DEFAULT_OUTPUT_ROOT))
    train.add_argument("--gpu-ids", default="0,1,2,3")
    train.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    train.add_argument("--accelerate-bin", default=str(DEFAULT_ACCELERATE))
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--force", action="store_true")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--smoke-train-rows", type=int, default=32)
    train.add_argument("--smoke-val-rows", type=int, default=24)
    train.set_defaults(handler=train_phase)

    infer = subparsers.add_parser(
        "infer",
        help="Run deduplicated HF/PEFT A-F forward scoring for one trained run",
    )
    infer.add_argument("--prepared-manifest", required=True)
    infer.add_argument("--model-name", required=True)
    infer.add_argument("--model-path", default=None)
    infer.add_argument("--assignment", required=True)
    infer.add_argument("--seed", type=int, required=True)
    infer.add_argument("--experiment-root", default=str(DEFAULT_OUTPUT_ROOT))
    infer.add_argument("--registry", default=None)
    infer.add_argument("--output-dir", default=None)
    infer.add_argument("--gpu-id", default="auto")
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--max-logical-rows", type=int, default=None)
    infer.add_argument("--dry-run", action="store_true")
    infer.add_argument("--force", action="store_true")
    infer.add_argument("--smoke", action="store_true")
    infer.set_defaults(handler=infer_phase)

    analyze = subparsers.add_parser(
        "analyze",
        help="Join independent gold and analyze the frozen 12-run panel",
    )
    analyze.add_argument("--prepared-manifest", required=True)
    analyze.add_argument("--result", action="append", required=True)
    analyze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "analysis"))
    analyze.add_argument("--bootstrap", type=int, default=10_000)
    analyze.add_argument("--randomization", type=int, default=100_000)
    analyze.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    analyze.set_defaults(handler=analyze_phase)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "seed", FORMAL_SEEDS[0]) not in FORMAL_SEEDS:
        if not getattr(args, "smoke", False):
            raise FinetuneExperimentError(
                f"Seed must be one of {FORMAL_SEEDS}; received {args.seed}"
            )
    if hasattr(args, "max_length") and int(args.max_length) != MAX_LENGTH:
        raise FinetuneExperimentError(
            f"Frozen prepare max length is {MAX_LENGTH}; received {args.max_length}"
        )
    if hasattr(args, "bootstrap") and int(args.bootstrap) <= 0:
        raise FinetuneExperimentError("--bootstrap must be positive")
    if hasattr(args, "randomization") and int(args.randomization) <= 0:
        raise FinetuneExperimentError("--randomization must be positive")
    args.handler(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinetuneExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
