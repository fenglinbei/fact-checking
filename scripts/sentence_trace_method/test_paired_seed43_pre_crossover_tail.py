from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts/sentence_trace_method/run_paired_seed43_pre_crossover_tail.sh"
CROSSOVER = (
    ROOT
    / "scripts/phase5_selectors/eval/run_structure_only_matched_verifier_crossover_step800.sh"
)
EXACT_OUTPUT = (
    "outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/"
    "matched_verifier_crossover_seed43_step800_val"
)
TZ8 = timezone(timedelta(hours=8))


def _epoch(hour: int, minute: int) -> str:
    return str(int(datetime(2026, 7, 17, hour, minute, tzinfo=TZ8).timestamp()))


def _write_seed_contract(case_root: Path) -> Path:
    train = case_root / "train"
    checkpoint = train / "checkpoint-800"
    checkpoint.mkdir(parents=True)
    (train / "training_complete.json").write_text(
        json.dumps({"completed": True, "global_step": 900}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"seed43-adapter")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/models/base",
                "peft_type": "LORA",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (train / "config.resolved.yaml").write_text("dataset: fake\n", encoding="utf-8")
    (case_root / "train.resolved.yaml").write_text(
        "sft_train:\n  seed: 43\n",
        encoding="utf-8",
    )
    return train


def _write_rawfc_complete(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "train/training_complete.json").write_text(
        json.dumps({"completed": True, "global_step": 100}) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "num_samples": 200,
        "accuracy": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1": 0.5,
    }
    for split in ("val", "test"):
        path = root / f"eval/{split}/best/label_token/metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")


def _write_scifact_complete(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "train/training_complete.json").write_text(
        json.dumps({"completed": True, "global_step": 100}) + "\n",
        encoding="utf-8",
    )
    val_metrics = root / "eval/val/best/label_token/metrics.json"
    val_metrics.parent.mkdir(parents=True)
    val_metrics.write_text(
        json.dumps({"num_samples": 300, "macro_f1": 0.5}) + "\n",
        encoding="utf-8",
    )
    test_manifest = root / "eval/test/best/label_token/prediction_manifest.json"
    test_manifest.parent.mkdir(parents=True)
    test_manifest.write_text(
        json.dumps({"split": "test", "num_samples": 300, "prediction_only": True})
        + "\n",
        encoding="utf-8",
    )
    submission = root / "submission"
    submission.mkdir()
    (submission / "scifact_official_style_metrics_val.json").write_text(
        json.dumps(
            {
                key: {"f1": 0.5}
                for key in (
                    "sentence_selection_only",
                    "sentence",
                    "abstract_label_only",
                    "abstract",
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = "".join('{"id": 1}\n' for _ in range(300))
    (submission / "scifact_submission_val.jsonl").write_text(rows, encoding="utf-8")
    (submission / "scifact_submission_test.jsonl").write_text(rows, encoding="utf-8")


def _write_fake_audit(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'mkdir -p "$OUTPUT_ROOT"\n'
        "printf '%s\\n' '"
        '{"schema_version":"structure-only-clean-results-audit-summary-v0.1",'
        '"coverage":{"total":6,"invalid":0},'
        '"provenance_policy":{"explicit_roots_only":true,"fallback_used":false}}'
        "' > \"$OUTPUT_ROOT/summary.json\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_crossover_summary(root: Path, s_sha: str, o_sha: str) -> None:
    summary = {
        "schema_version": "structure-only-matched-verifier-crossover-summary-v0.1",
        "status": "complete",
        "scope": "frozen_val_only_fixed_k5_common_support",
        "split": "val",
        "checkpoint": "checkpoint-800",
        "event_count": 1234,
        "event_id_sequence_sha256": (
            "65038f1f222b7d990642970ebf7281434abdb17fe61ec1e14ed0c937e8ee6549"
        ),
        "verifiers": {
            verifier: {
                "adapter_sha256": sha,
                "metrics": {
                    cell: {"num_samples": 1234}
                    for cell in ("one_shot__fixed5", "stateful__fixed5")
                },
            }
            for verifier, sha in (("V_S", s_sha), ("V_O", o_sha))
        },
    }
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")


def _base_env(tmp_path: Path, now: str, *, dry_run: str = "true") -> dict[str, str]:
    seed_s = _write_seed_contract(tmp_path / "seed-s")
    seed_o = _write_seed_contract(tmp_path / "seed-o")
    tail = tmp_path / "tail"
    tail.mkdir()
    sentinel = tail / "enabled"
    sentinel.write_text("enabled\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "DRY_RUN": dry_run,
            "HOOK_PHASE": "pre",
            "TAIL_BASE_DIR": str(tail),
            "TAIL_SENTINEL": str(sentinel),
            "TAIL_LOCK_FILE": str(tail / "tail.lock"),
            "TAIL_EVENTS_FILE": str(tail / "events.tsv"),
            "TAIL_COMPLETE_FILE": str(tail / "complete.json"),
            "TAIL_NOW_EPOCH": now,
            "SEED43_S_TRAIN_DIR": str(seed_s),
            "SEED43_O_TRAIN_DIR": str(seed_o),
            "RAWFC_RUN_ROOT": str(tmp_path / "rawfc"),
            "SCIFACT_RUN_ROOT": str(tmp_path / "scifact"),
            "CROSSOVER_OUTPUT_ROOT": str(tmp_path / "crossover"),
            "CLEAN_AUDIT_OUTPUT_ROOT": str(tmp_path / "audit"),
            "POLL_SECONDS": "1",
        }
    )
    return env


def _run_hook(tmp_path: Path, now: str, **updates: str) -> subprocess.CompletedProcess[str]:
    env = _base_env(tmp_path, now)
    env.update(updates)
    return subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_before_1020_plans_rawfc_then_scifact_and_never_vr(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, _epoch(10, 0))
    assert result.returncode == 0, result.stdout
    assert "stage=rawfc_clean status=starting" in result.stdout
    assert "stage=scifact_clean status=starting" in result.stdout
    assert result.stdout.index("stage=rawfc_clean") < result.stdout.index("stage=scifact_clean")
    assert "v_r:full" not in result.stdout
    assert "stage=seed43_crossover status=ready" in result.stdout


def test_1020_to_1055_plans_only_scifact(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, _epoch(10, 30))
    assert result.returncode == 0, result.stdout
    assert "stage=rawfc_clean status=starting" not in result.stdout
    assert "stage=scifact_clean status=starting" in result.stdout


def test_after_1055_starts_no_full_training(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, _epoch(11, 0))
    assert result.returncode == 0, result.stdout
    assert "stage=rawfc_clean status=starting" not in result.stdout
    assert "stage=scifact_clean status=starting" not in result.stdout
    assert "stage=full_training status=skipped_deadline" in result.stdout
    assert "stage=seed43_crossover status=ready" in result.stdout


def test_less_than_900_seconds_skips_crossover_and_returns_distinct_code(
    tmp_path: Path,
) -> None:
    result = _run_hook(tmp_path, _epoch(11, 30))
    assert result.returncode == 75, result.stdout
    assert "stage=seed43_crossover status=skipped_budget" in result.stdout
    assert "stage=clean_results_audit status=dry_run" in result.stdout
    assert (tmp_path / "tail/enabled").is_file()


def test_valid_complete_task_is_not_repeated(tmp_path: Path) -> None:
    env = _base_env(tmp_path, _epoch(10, 0))
    _write_rawfc_complete(Path(env["RAWFC_RUN_ROOT"]))
    result = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "stage=rawfc_clean status=skipped_complete" in result.stdout
    assert "stage=rawfc_clean status=starting" not in result.stdout
    assert "stage=scifact_clean status=starting" in result.stdout


def test_valid_completed_crossover_is_not_repeated(tmp_path: Path) -> None:
    env = _base_env(tmp_path, _epoch(11, 0))
    import hashlib

    s_adapter = Path(env["SEED43_S_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors"
    o_adapter = Path(env["SEED43_O_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors"
    _write_crossover_summary(
        Path(env["CROSSOVER_OUTPUT_ROOT"]),
        hashlib.sha256(s_adapter.read_bytes()).hexdigest(),
        hashlib.sha256(o_adapter.read_bytes()).hexdigest(),
    )
    result = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 76, result.stdout
    assert "stage=seed43_crossover status=skipped_complete" in result.stdout


def test_seed_contract_fails_closed_when_checkpoint_is_missing(tmp_path: Path) -> None:
    env = _base_env(tmp_path, _epoch(10, 0))
    (Path(env["SEED43_O_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors").unlink()
    result = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 4, result.stdout
    assert "stage=seed43_pair status=invalid" in result.stdout


def test_finalize_requires_matching_crossover_then_consumes_sentinel(
    tmp_path: Path,
) -> None:
    env = _base_env(tmp_path, _epoch(11, 0), dry_run="false")
    env["HOOK_PHASE"] = "finalize"
    s_adapter = Path(env["SEED43_S_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors"
    o_adapter = Path(env["SEED43_O_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors"
    import hashlib

    s_sha = hashlib.sha256(s_adapter.read_bytes()).hexdigest()
    o_sha = hashlib.sha256(o_adapter.read_bytes()).hexdigest()
    crossover_root = Path(env["CROSSOVER_OUTPUT_ROOT"])
    _write_crossover_summary(crossover_root, s_sha, o_sha)
    fake_audit = tmp_path / "fake-audit.sh"
    _write_fake_audit(fake_audit)
    env["AUDIT_WRAPPER"] = str(fake_audit)
    result = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert not Path(env["TAIL_SENTINEL"]).exists()
    assert Path(env["TAIL_SENTINEL"] + ".consumed").is_file()
    complete = json.loads(Path(env["TAIL_COMPLETE_FILE"]).read_text(encoding="utf-8"))
    assert complete["status"] == "complete"


@pytest.mark.parametrize(
    ("task", "now", "exit_code"),
    [
        ("rawfc_clean", _epoch(10, 0), 7),
        ("scifact_clean", _epoch(10, 30), 8),
    ],
)
def test_failed_full_task_preserves_sentinel_after_crossover_and_retry_consumes(
    tmp_path: Path,
    task: str,
    now: str,
    exit_code: int,
) -> None:
    env = _base_env(tmp_path, now, dry_run="false")
    if task == "rawfc_clean":
        env["SCIFACT_SAFE_START_ISO"] = "2026-07-17T09:00:00+08:00"
    fake_queue = tmp_path / "fake-queue.sh"
    fake_queue.write_text(
        f"#!/usr/bin/env bash\nexit {exit_code}\n",
        encoding="utf-8",
    )
    fake_queue.chmod(0o755)
    env["QUEUE_WRAPPER"] = str(fake_queue)
    fake_audit = tmp_path / "fake-audit.sh"
    _write_fake_audit(fake_audit)
    env["AUDIT_WRAPPER"] = str(fake_audit)

    pre = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert pre.returncode == 0, pre.stdout
    failure_marker = tmp_path / "tail/seed43_pre_crossover_tail_failure.json"
    marker = json.loads(failure_marker.read_text(encoding="utf-8"))
    assert marker["failed_tasks"][task]["last_exit_code"] == exit_code

    import hashlib

    s_adapter = Path(env["SEED43_S_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors"
    o_adapter = Path(env["SEED43_O_TRAIN_DIR"]) / "checkpoint-800/adapter_model.safetensors"
    _write_crossover_summary(
        Path(env["CROSSOVER_OUTPUT_ROOT"]),
        hashlib.sha256(s_adapter.read_bytes()).hexdigest(),
        hashlib.sha256(o_adapter.read_bytes()).hexdigest(),
    )
    finalize_env = dict(env, HOOK_PHASE="finalize")
    finalize = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=finalize_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert finalize.returncode == 0, finalize.stdout
    assert Path(env["TAIL_SENTINEL"]).is_file()
    assert failure_marker.is_file()
    degraded = json.loads(Path(env["TAIL_COMPLETE_FILE"]).read_text(encoding="utf-8"))
    assert degraded["status"] == "degraded_complete"

    if task == "rawfc_clean":
        _write_rawfc_complete(Path(env["RAWFC_RUN_ROOT"]))
    else:
        _write_scifact_complete(Path(env["SCIFACT_RUN_ROOT"]))
    retry = subprocess.run(
        ["bash", str(HOOK)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert retry.returncode == 76, retry.stdout
    assert "stage=failure_marker status=resolved" in retry.stdout
    assert not failure_marker.exists()
    assert not Path(env["TAIL_SENTINEL"]).exists()
    assert Path(env["TAIL_SENTINEL"] + ".consumed").is_file()
    complete = json.loads(Path(env["TAIL_COMPLETE_FILE"]).read_text(encoding="utf-8"))
    assert complete["status"] == "complete"


def _run_crossover_with_fake_hook(
    tmp_path: Path,
    *,
    output_root: str,
    sentinel: bool = True,
    active: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    hook_log = tmp_path / "hook.log"
    fake_hook = tmp_path / "fake-hook.sh"
    fake_hook.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$HOOK_PHASE" >> "$HOOK_LOG"\n',
        encoding="utf-8",
    )
    fake_hook.chmod(0o755)
    sentinel_path = tmp_path / "enabled"
    if sentinel:
        sentinel_path.write_text("enabled\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "true",
            "PHASES": "prepare",
            "OUTPUT_ROOT": output_root,
            "SEED43_TAIL_SENTINEL": str(sentinel_path),
            "SEED43_TAIL_LOCK_FILE": str(tmp_path / "tail.lock"),
            "SEED43_PRE_CROSSOVER_HOOK": str(fake_hook),
            "HOOK_LOG": str(hook_log),
            "S_RUN_DIR": str(tmp_path / "seed-s/train"),
            "O_RUN_DIR": str(tmp_path / "seed-o/train"),
            "SEED43_TAIL_HOOK_ACTIVE": "true" if active else "false",
        }
    )
    result = subprocess.run(
        ["bash", str(CROSSOVER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result, hook_log


def test_generic_wrapper_hook_is_exact_and_runs_pre_then_finalize(tmp_path: Path) -> None:
    result, hook_log = _run_crossover_with_fake_hook(
        tmp_path,
        output_root=EXACT_OUTPUT,
    )
    assert result.returncode == 0, result.stdout
    assert hook_log.read_text(encoding="utf-8").splitlines() == ["pre", "finalize"]


@pytest.mark.parametrize(
    ("output_root", "sentinel", "active"),
    [
        ("outputs/not-the-seed43-crossover", True, False),
        (EXACT_OUTPUT, False, False),
        (EXACT_OUTPUT, True, True),
    ],
)
def test_generic_wrapper_does_not_trigger_outside_narrow_contract(
    tmp_path: Path,
    output_root: str,
    sentinel: bool,
    active: bool,
) -> None:
    result, hook_log = _run_crossover_with_fake_hook(
        tmp_path,
        output_root=output_root,
        sentinel=sentinel,
        active=active,
    )
    assert result.returncode == 0, result.stdout
    assert not hook_log.exists()


def test_hook_never_queues_vr() -> None:
    text = HOOK.read_text(encoding="utf-8")
    assert "v_r:full" not in text
    assert 'run_full_task v_r' not in text
