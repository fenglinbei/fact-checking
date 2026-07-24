from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/sentence_trace_method/fixed_step_salvage_process.py"
ORCHESTRATOR = ROOT / "scripts/sentence_trace_method/run_fixed_step_salvage_20260717.sh"
TZ8 = timezone(timedelta(hours=8))


def _write_process(
    proc_root: Path,
    pid: int,
    ppid: int,
    starttime: int,
    argv: list[str],
) -> None:
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True)
    fields_after_comm = ["S", str(ppid), *("0" for _ in range(17)), str(starttime)]
    (process_root / "stat").write_text(
        f"{pid} (fake process) {' '.join(fields_after_comm)}\n",
        encoding="utf-8",
    )
    (process_root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")


def _run_helper(proc_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), "--proc-root", str(proc_root), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _build_tree(proc_root: Path) -> tuple[str, str]:
    config = "outputs/run/train.resolved.yaml"
    module = "sft.hami_cuda_bootstrap"
    _write_process(proc_root, 100, 1, 1000, ["timeout", "100s", "bash", "wrapper.sh"])
    _write_process(proc_root, 101, 100, 1001, ["bash", "wrapper.sh"])
    _write_process(
        proc_root,
        102,
        101,
        1002,
        [
            "/env/bin/python",
            "/env/bin/accelerate",
            "launch",
            "-m",
            module,
            "--config",
            config,
        ],
    )
    _write_process(
        proc_root,
        103,
        102,
        1003,
        ["python", "-m", module, "--config", config],
    )
    _write_process(
        proc_root,
        200,
        1,
        2000,
        ["accelerate", "launch", "-m", module, "--config", config],
    )
    return config, module


def test_identify_selects_only_exact_accelerate_descendant(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    config, module = _build_tree(proc_root)
    result = _run_helper(
        proc_root,
        "identify",
        "--root-pid",
        "100",
        "--root-starttime",
        "1000",
        "--config",
        config,
        "--module",
        module,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["pid"] == 102
    assert payload["starttime"] == 1002


def test_identify_fails_closed_on_ambiguity_and_pid_reuse(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    config, module = _build_tree(proc_root)
    _write_process(
        proc_root,
        104,
        101,
        1004,
        ["accelerate", "launch", "-m", module, "--config", config],
    )
    ambiguous = _run_helper(
        proc_root,
        "identify",
        "--root-pid",
        "100",
        "--root-starttime",
        "1000",
        "--config",
        config,
        "--module",
        module,
    )
    assert ambiguous.returncode == 4, ambiguous.stdout
    assert json.loads(ambiguous.stdout)["status"] == "ambiguous"

    reused = _run_helper(
        proc_root,
        "identify",
        "--root-pid",
        "100",
        "--root-starttime",
        "999",
        "--config",
        config,
        "--module",
        module,
    )
    assert reused.returncode == 5, reused.stdout
    assert json.loads(reused.stdout)["status"] == "root_identity_changed"


def test_signal_dry_run_revalidates_candidate_identity(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    config, module = _build_tree(proc_root)
    result = _run_helper(
        proc_root,
        "signal",
        "--root-pid",
        "100",
        "--root-starttime",
        "1000",
        "--config",
        config,
        "--module",
        module,
        "--pid",
        "102",
        "--starttime",
        "1002",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "would_signal"
    assert payload["signal_sent"] is False


def _write_fixed_s_contract(root: Path) -> str:
    train = root / "train"
    checkpoint = train / "checkpoint-800"
    latest = train / "latest_state"
    checkpoint.mkdir(parents=True)
    latest.mkdir()
    adapter = b"fixed-seed43-S-adapter"
    (checkpoint / "adapter_model.safetensors").write_bytes(adapter)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/model", "peft_type": "LORA"}) + "\n",
        encoding="utf-8",
    )
    (latest / "trainer_state.json").write_text(
        json.dumps({"global_step": 1200}) + "\n",
        encoding="utf-8",
    )
    (train / "config.resolved.yaml").write_text("dataset: fake\n", encoding="utf-8")
    (root / "train.resolved.yaml").write_text(
        "sft_train:\n  seed: 43\n",
        encoding="utf-8",
    )
    return hashlib.sha256(adapter).hexdigest()


def test_orchestrator_dry_run_preserves_fixed_step_semantics_and_order(
    tmp_path: Path,
) -> None:
    s_root = tmp_path / "seed-s"
    s_sha = _write_fixed_s_contract(s_root)
    run_dir = tmp_path / "salvage"
    now = int(datetime(2026, 7, 17, 8, 0, tzinfo=TZ8).timestamp())
    env = dict(os.environ)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "DRY_RUN": "true",
            "SALVAGE_NOW_EPOCH": str(now),
            "SALVAGE_RUN_DIR": str(run_dir),
            "SALVAGE_LOCK": str(tmp_path / "salvage.lock"),
            "OLD_DECISION_LOCK": str(tmp_path / "decision.lock"),
            "EVENTS_FILE": str(tmp_path / "events.tsv"),
            "FINAL_MANIFEST": str(tmp_path / "final.json"),
            "O_CAPPED_MANIFEST": str(tmp_path / "o-capped.json"),
            "O_LOG": str(tmp_path / "o.log"),
            "S_RUN_ROOT": str(s_root),
            "O_RUN_ROOT": str(tmp_path / "seed-o"),
            "S_EXPECTED_ADAPTER_SHA256": s_sha,
        }
    )
    result = subprocess.run(
        ["bash", str(ORCHESTRATOR)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    expected = [
        "stage=seed43_o status=dry_run",
        "stage=seed43_crossover status=dry_run",
        "stage=scifact_clean status=dry_run",
        "stage=rawfc_clean status=dry_run",
        "stage=clean_results_audit status=dry_run",
    ]
    offsets = [result.stdout.index(item) for item in expected]
    assert offsets == sorted(offsets)
    assert not (s_root / "train/training_complete.json").exists()
    assert not (tmp_path / "seed-o/train/training_complete.json").exists()
    assert not Path(env["O_CAPPED_MANIFEST"]).exists()
