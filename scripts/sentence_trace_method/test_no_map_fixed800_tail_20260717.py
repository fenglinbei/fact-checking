from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TAIL = ROOT / "scripts/sentence_trace_method/run_no_map_fixed800_tail_20260717.sh"
CONTRACT = ROOT / "scripts/sentence_trace_method/no_map_fixed800_tail_contract.py"
CROSSOVER = ROOT / "scripts/phase5_selectors/eval/run_no_map_structure_fixed5_crossover_step800.sh"
TZ8 = timezone(timedelta(hours=8))


def _epoch(hour: int, minute: int, second: int = 0) -> int:
    return int(datetime(2026, 7, 17, hour, minute, second, tzinfo=TZ8).timestamp())


def _fake_helper(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json, sys
command = sys.argv[1]
if command == 'accelerate':
    print(json.dumps({'status':'ready','count':0,'pids':[]}))
elif command == 'inputs':
    print(json.dumps({'schema_version':'fake-input','status':'ready','effective_seed':42}))
else:
    print(json.dumps({'status':'invalid'})); raise SystemExit(2)
""",
        encoding="utf-8",
    )


def _tail_env(tmp_path: Path, *, now: int) -> dict[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    helper = tmp_path / "helper.py"
    _fake_helper(helper)
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    crossover = tmp_path / "crossover.sh"
    crossover.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    input_contract = tmp_path / "input.json"
    input_contract.write_text("{}\n", encoding="utf-8")
    env = dict(os.environ)
    for name in (
        "CASE_NAME", "CASE_ROOT", "LORA_ROOT", "TRAIN_CASE_ROOT", "RUN_DIR",
        "CONFIG", "CONFIG_PATH", "BASE_CASE_NAME", "CASE_SUFFIX", "LORA_SUFFIX",
        "OUTPUT_ROOT", "MREC_POLICY_CONFIG", "MAP_ABLATION_MODE",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "TRAIN_ACCELERATE_BIN": "/usr/bin/env",
            "CONTRACT_HELPER": str(helper),
            "PROCESS_HELPER": str(ROOT / "scripts/sentence_trace_method/fixed_step_salvage_process.py"),
            "INPUT_CONTRACT": str(input_contract),
            "TRAIN_WRAPPER": str(wrapper),
            "CROSSOVER_WRAPPER": str(crossover),
            "OLD_DECISION_LOCK": str(tmp_path / "old.lock"),
            "TAIL_DIR": str(tmp_path / "tail"),
            "TAIL_LOCK": str(tmp_path / "tail.lock"),
            "EVENTS_FILE": str(tmp_path / "events.tsv"),
            "FINAL_MANIFEST": str(tmp_path / "manifest.json"),
            "INPUT_AUDIT": str(tmp_path / "input_audit.json"),
            "CAPPED_MANIFEST": str(tmp_path / "cap.json"),
            "TRAIN_LOG": str(tmp_path / "train.log"),
            "DIAGNOSTIC_LOG": str(tmp_path / "diagnostic.log"),
            "NO_MAP_BASE_ROOT": str(tmp_path / "base"),
            "NO_MAP_RUN_ROOT": str(tmp_path / "run"),
            "NO_MAP_TRACE_ROOT": str(tmp_path / "traces"),
            "N_FIXED5_SOURCE_ROOT": str(tmp_path / "n-fixed5"),
            "MATRIX_ROOT": str(tmp_path / "matrix"),
            "DIAGNOSTIC_ROOT": str(tmp_path / "diagnostic"),
            "DRY_RUN": "true",
            "TAIL_NOW_EPOCH": str(now),
        }
    )
    return env


def _run_tail(tmp_path: Path, *, now: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TAIL)], cwd=ROOT, env=_tail_env(tmp_path, now=now),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def test_dry_run_passes_lock_inputs_and_never_invokes_wrapper(tmp_path: Path) -> None:
    result = _run_tail(tmp_path, now=_epoch(10, 34))
    assert result.returncode == 0, result.stdout
    assert "stage=old_decision_lock status=acquired" in result.stdout
    assert "stage=training status=dry_run" in result.stdout
    assert "stage=diagnostic status=dry_run" in result.stdout
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["standard_clean_results_audit_slot_mutated"] is False


def test_start_deadline_and_65_minute_gate_are_fail_closed(tmp_path: Path) -> None:
    late = _run_tail(tmp_path / "late", now=_epoch(10, 35, 1))
    assert late.returncode == 0
    assert "status=skipped_deadline" in late.stdout
    short_env = _tail_env(tmp_path / "short", now=_epoch(10, 35))
    short_env["MIN_START_REMAINING_SECONDS"] = "3901"
    short = subprocess.run(
        ["bash", str(TAIL)], cwd=ROOT, env=short_env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert short.returncode == 0
    assert "status=skipped_insufficient_window" in short.stdout


def test_dry_run_refuses_live_old_lock(tmp_path: Path) -> None:
    env = _tail_env(tmp_path, now=_epoch(10, 34))
    lock_path = Path(env["OLD_DECISION_LOCK"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(TAIL)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
    assert result.returncode == 73
    assert "status=blocked_live_lock" in result.stdout


def test_tail_refuses_stale_no_map_train_or_eval(tmp_path: Path) -> None:
    env = _tail_env(tmp_path, now=_epoch(10, 34))
    (Path(env["NO_MAP_RUN_ROOT"]) / "train").mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(TAIL)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert result.returncode == 4
    assert "status=blocked_stale" in result.stdout


def _write_checkpoint(root: Path, *, training_complete: bool = False) -> Path:
    checkpoint = root / "train/checkpoint-800"
    latest = root / "train/latest_state"
    checkpoint.mkdir(parents=True)
    latest.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"stable-no-map-adapter")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/model", "peft_type": "LORA"}) + "\n",
        encoding="utf-8",
    )
    (latest / "trainer_state.json").write_text('{"global_step": 800}\n', encoding="utf-8")
    (root / "train.resolved.yaml").write_text("sft_train: {}\n", encoding="utf-8")
    (root / "train/config.resolved.yaml").write_text("sft_train: {}\n", encoding="utf-8")
    if training_complete:
        (root / "train/training_complete.json").write_text('{"completed":true}\n', encoding="utf-8")
    return checkpoint


def test_checkpoint_contract_records_cap_provenance_and_rejects_complete(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_checkpoint(run)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "effective_seed": 42,
                "weight_fingerprint": "123456789abc",
                "builds": {
                    "run_root": str(run),
                    "splits": {split: {"sha256": split[0] * 64} for split in ("train", "val", "test")},
                },
            }
        ) + "\n",
        encoding="utf-8",
    )
    ready = subprocess.run(
        [sys.executable, str(CONTRACT), "checkpoint", "--run-root", str(run), "--contract", str(contract)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert ready.returncode == 0, ready.stdout
    payload = json.loads(ready.stdout)
    assert payload["role"] == "V_N"
    assert payload["checkpoint"] == "checkpoint-800"
    assert payload["progress_step"] == 800
    assert payload["seed"]["effective"] == 42
    assert payload["training_complete_present"] is False
    (run / "train/training_complete.json").write_text('{"completed":true}\n', encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, str(CONTRACT), "checkpoint", "--run-root", str(run), "--contract", str(contract)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert rejected.returncode == 2
    assert "training_complete" in rejected.stdout


def test_script_declares_exact_cap_and_insufficient_diagnostic_time_contract() -> None:
    text = TAIL.read_text(encoding="utf-8")
    assert 'CHECKPOINT_DEADLINE_ISO="${CHECKPOINT_DEADLINE_ISO:-2026-07-17T11:31:00+08:00}"' in text
    assert 'MIN_DIAGNOSTIC_REMAINING_SECONDS="${MIN_DIAGNOSTIC_REMAINING_SECONDS:-480}"' in text
    assert "signal_accelerate" in text
    assert "checkpoint-800 was not stably ready" in text
    assert "capped_no_diagnostic" in text
    assert 'CLEANUP_ISO="${CLEANUP_ISO:-2026-07-17T11:39:00+08:00}"' in text
    assert 'HARD_STOP_ISO="${HARD_STOP_ISO:-2026-07-17T11:40:00+08:00}"' in text


def test_crossover_dry_run_is_exact_n_s_fixed5_step800_contract(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "no-map-structure-fixed5-matrix-v0.1",
                "matrix_kind": "no_map_structure_matched_verifier_crossover",
                "split": "val", "expected_k": 5, "event_count": 1234,
                "cell_count": 2, "all_ready": True,
                "checkpoint_contract": {"checkpoint": "checkpoint-800", "test_allowed": False, "best_alias_allowed": False},
                "cells": [{"cell_id": "N_fixed5"}, {"cell_id": "S_fixed5"}],
            }
        ) + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "true", "MATRIX_ROOT": str(matrix),
            "MATRIX_MANIFEST": str(matrix / "manifest.json"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "N_EXPECTED_ADAPTER_SHA256": "1" * 64,
            "S_EXPECTED_ADAPTER_SHA256": "2" * 64,
        }
    )
    result = subprocess.run(
        ["bash", str(CROSSOVER)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "checkpoint=checkpoint-800 events=1234 cells=N_fixed5,S_fixed5" in result.stdout
    assert "verifier_n" in result.stdout and "verifier_s" in result.stdout
    assert "summarize_no_map_structure_fixed5_crossover.py" in result.stdout
