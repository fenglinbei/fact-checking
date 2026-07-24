from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TAIL = ROOT / "scripts/sentence_trace_method/run_rawfc_seed43_best_export_tail_20260717.sh"
CONTRACT = ROOT / "scripts/sentence_trace_method/rawfc_seed43_tail_contract.py"
TZ8 = timezone(timedelta(hours=8))


def _epoch(hour: int, minute: int, second: int = 0) -> str:
    return str(
        int(datetime(2026, 7, 17, hour, minute, second, tzinfo=TZ8).timestamp())
    )


def _write_seed43_root(root: Path, *, completed: bool = False) -> str:
    best = root / "train/best"
    best.mkdir(parents=True)
    adapter = best / "adapter_model.safetensors"
    adapter.write_bytes(b"auditable-seed43-best-adapter")
    (root / "train.resolved.yaml").write_text(
        "sft_train:\n  seed: 43\n", encoding="utf-8"
    )
    if completed:
        (root / "train/training_complete.json").write_text(
            '{"completed":true,"global_step":650}\n', encoding="utf-8"
        )
    return hashlib.sha256(adapter.read_bytes()).hexdigest()


def _write_fake_infer(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "split=\n"
        "run_dir=\n"
        "while (( $# > 0 )); do\n"
        "  case \"$1\" in\n"
        "    --split) split=\"$2\"; shift 2 ;;\n"
        "    --run-dir) run_dir=\"$2\"; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "root=\"${run_dir%/train}\"\n"
        "out=\"$root/eval/$split/best/label_token\"\n"
        "mkdir -p \"$out\"\n"
        "printf '%s\\n' \"$split\" >> \"$FAKE_CALLS\"\n"
        "printf '%s\\n' '{\"num_samples\":200,\"accuracy\":0.6,"
        "\"macro_precision\":0.6,\"macro_recall\":0.6,"
        "\"macro_f1\":0.6,\"parse_error_rate\":0}' > \"$out/metrics.json\"\n"
        "for i in $(seq 0 199); do\n"
        "  printf '{\"sample_idx\":%s,\"pred_label\":\"half\"}\\n' \"$i\"\n"
        "done > \"$out/${split}_predictions.jsonl\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _base_env(tmp_path: Path, *, now: str, completed: bool = False) -> dict[str, str]:
    seed43_root = tmp_path / "seed43-run"
    expected_sha = _write_seed43_root(seed43_root, completed=completed)
    fake_infer = tmp_path / "fake-infer.sh"
    _write_fake_infer(fake_infer)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "INFER_BIN": str(fake_infer),
            "CONTRACT_HELPER": str(CONTRACT),
            "OLD_DECISION_LOCK": str(tmp_path / "old-decision.lock"),
            "TAIL_RUN_DIR": str(tmp_path / "tail"),
            "SEED43_RUN_ROOT": str(seed43_root),
            "EXPECTED_ADAPTER_SHA": expected_sha,
            "RAWFC_SEED43_PROC_ROOT": str(proc_root),
            "TAIL_NOW_EPOCH": now,
            "MIN_EXPORT_BUDGET_SECONDS": "600",
            "EXPORT_FINISH_MARGIN_SECONDS": "60",
            "EXPORT_CUDA_VISIBLE_DEVICES": "0",
            "FAKE_CALLS": str(tmp_path / "calls.txt"),
        }
    )
    return env


def _run(tmp_path: Path, *, now: str, **updates: str) -> subprocess.CompletedProcess[str]:
    env = _base_env(tmp_path, now=now)
    env.update(updates)
    return subprocess.run(
        ["bash", str(TAIL)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_best_export_tail_runs_direct_val_then_test_and_records_contracts(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, now=_epoch(10, 30))
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").splitlines() == [
        "val",
        "test",
    ]
    manifest = json.loads((tmp_path / "tail/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["hard_stop"] == "2026-07-17T11:40:00+08:00"
    assert manifest["export_deadline"] == "2026-07-17T11:39:00+08:00"
    assert manifest["preflight"]["seed"]["effective"] == 43
    assert manifest["preflight"]["training_complete"]["missing"] is True
    assert manifest["preflight"]["adapter"]["matched"] is True
    for split in ("val", "test"):
        audit = manifest["exports"][split]
        assert audit["status"] == "complete"
        assert audit["metrics"]["num_samples"] == 200
        assert audit["metrics"]["parse_failures"] == 0
        assert audit["predictions"]["num_predictions"] == 200
        assert len(audit["metrics"]["sha256"]) == 64
        assert len(audit["predictions"]["sha256"]) == 64
    assert [item["audit"]["count"] for item in manifest["accelerate_audits"]] == [
        0,
        0,
        0,
        0,
    ]


def test_best_export_tail_blocks_adapter_sha_mismatch(tmp_path: Path) -> None:
    result = _run(tmp_path, now=_epoch(10, 30), EXPECTED_ADAPTER_SHA="0" * 64)
    assert result.returncode == 6, result.stdout
    assert "status=blocked_preflight" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_best_export_tail_blocks_when_training_complete_exists(tmp_path: Path) -> None:
    env = _base_env(tmp_path, now=_epoch(10, 30), completed=True)
    result = subprocess.run(
        ["bash", str(TAIL)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 6, result.stdout
    assert "status=blocked_preflight" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_best_export_tail_skips_after_latest_safe_start(tmp_path: Path) -> None:
    result = _run(tmp_path, now=_epoch(11, 29, 1))
    assert result.returncode == 0, result.stdout
    assert "status=skipped_insufficient_window" in result.stdout
    assert not (tmp_path / "calls.txt").exists()
    manifest = json.loads((tmp_path / "tail/manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest_start"] == "2026-07-17T11:29:00+08:00"


def test_best_export_tail_dry_run_never_invokes_inference(tmp_path: Path) -> None:
    result = _run(tmp_path, now=_epoch(10, 30), DRY_RUN="true")
    assert result.returncode == 0, result.stdout
    assert "status=dry_run" in result.stdout
    assert "-m sft.label_token_infer" in result.stdout
    assert not (tmp_path / "calls.txt").exists()

