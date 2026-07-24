from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/experiment/mrec_v0.2/"
    "rawfc_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_seed43.yaml"
)
BASE_CONFIG = (
    ROOT
    / "configs/experiment/mrec_v0.2/"
    "rawfc_learned_marginal_structure_only_fullpool_minmax5_10_baseline20.yaml"
)
WRAPPER = (
    ROOT
    / "scripts/sentence_trace_method/"
    "run_rawfc_ministral3_atom_anchor_v0_2_structure_only_fullpool_minmax5_10_"
    "baseline20_seed43_lora_r16a32_d010_lr1e5_ep12_eval50.sh"
)
TAIL = ROOT / "scripts/sentence_trace_method/run_rawfc_seed43_full_tail_20260717.sh"
CONTRACT = ROOT / "scripts/sentence_trace_method/rawfc_seed43_tail_contract.py"
CONFIG_LOADER = ROOT / "scripts/sentence_trace_method/mrec_policy_config.py"
TZ8 = timezone(timedelta(hours=8))
POLLUTING = (
    "CASE_NAME",
    "CASE_ROOT",
    "LORA_ROOT",
    "TRAIN_CASE_ROOT",
    "RUN_DIR",
    "CONFIG",
    "CONFIG_PATH",
    "BASE_CASE_NAME",
    "CASE_SUFFIX",
    "LORA_SUFFIX",
    "OUTPUT_ROOT",
    "MREC_POLICY_CONFIG",
)


def _epoch(hour: int, minute: int, second: int = 0) -> str:
    return str(
        int(datetime(2026, 7, 17, hour, minute, second, tzinfo=TZ8).timestamp())
    )


def _load_policy(path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "rawfc_seed43_test_loader"}
    exec(compile(CONFIG_LOADER.read_text(encoding="utf-8"), str(CONFIG_LOADER), "exec"), namespace)
    return namespace["_load_yaml"](path)  # type: ignore[operator,no-any-return]


def _write_builds(root: Path, *, mismatch_split: str | None = None) -> None:
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        suffix = "-mismatch" if split == mismatch_split else ""
        (build / f"build_{split}.jsonl").write_text(
            f'{{"id":"{split}{suffix}"}}\n', encoding="utf-8"
        )


def _copy_builds(source: Path, target: Path, *, mismatch_split: str | None = None) -> None:
    build = target / "build"
    build.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        payload = (source / "build" / f"build_{split}.jsonl").read_bytes()
        if split == mismatch_split:
            payload += b'{"mismatch":true}\n'
        (build / f"build_{split}.jsonl").write_bytes(payload)


def _write_full(root: Path, *, seed: int | None, num_samples: int = 200) -> None:
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    (train / "training_complete.json").write_text(
        json.dumps({"completed": True, "global_step": 500}) + "\n",
        encoding="utf-8",
    )
    seed_line = "" if seed is None else f"  seed: {seed}\n"
    (root / "train.resolved.yaml").write_text(
        f"sft_train:\n{seed_line}", encoding="utf-8"
    )
    metrics = {
        "num_samples": num_samples,
        "accuracy": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1": 0.5,
    }
    for split in ("val", "test"):
        path = root / f"eval/{split}/best/label_token/metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")


def _write_fake_wrapper(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$MODE" >> "$FAKE_CALLS"\n'
        'mkdir -p "$RAWFC_SEED43_BASE_ROOT/build" "$RAWFC_SEED43_RUN_ROOT/build"\n'
        "for split in train val test; do\n"
        '  cp "$RAWFC_SEED43_CANONICAL_BASE_ROOT/build/build_${split}.jsonl" '
        '"$RAWFC_SEED43_BASE_ROOT/build/build_${split}.jsonl"\n'
        '  cp "$RAWFC_SEED43_CANONICAL_RUN_ROOT/build/build_${split}.jsonl" '
        '"$RAWFC_SEED43_RUN_ROOT/build/build_${split}.jsonl"\n'
        '  if [[ "${FAKE_MISMATCH_SPLIT:-}" == "$split" ]]; then\n'
        "    printf '%s\\n' '{\"mismatch\":true}' >> "
        '"$RAWFC_SEED43_BASE_ROOT/build/build_${split}.jsonl"\n'
        "    printf '%s\\n' '{\"mismatch\":true}' >> "
        '"$RAWFC_SEED43_RUN_ROOT/build/build_${split}.jsonl"\n'
        "  fi\n"
        "done\n"
        "printf 'sft_train:\\n  seed: 43\\n' > "
        '"$RAWFC_SEED43_BASE_ROOT/train.resolved.yaml"\n'
        "printf 'sft_train:\\n  seed: 43\\n' > "
        '"$RAWFC_SEED43_RUN_ROOT/train.resolved.yaml"\n'
        'if [[ "$MODE" == "full" ]]; then\n'
        '  mkdir -p "$RAWFC_SEED43_RUN_ROOT/train"\n'
        "  printf '%s\\n' '{\"completed\":true,\"global_step\":500}' > "
        '"$RAWFC_SEED43_RUN_ROOT/train/training_complete.json"\n'
        "  for split in val test; do\n"
        '    mkdir -p "$RAWFC_SEED43_RUN_ROOT/eval/${split}/best/label_token"\n'
        "    printf '%s\\n' "
        "'{\"num_samples\":200,\"accuracy\":0.5,\"macro_precision\":0.5,"
        "\"macro_recall\":0.5,\"macro_f1\":0.5}' > "
        '"$RAWFC_SEED43_RUN_ROOT/eval/${split}/best/label_token/metrics.json"\n'
        "  done\n"
        "fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _base_env(tmp_path: Path, *, now: str, dry_run: str = "false") -> dict[str, str]:
    canonical_base = tmp_path / "canonical-base"
    canonical_run = tmp_path / "canonical-run"
    _write_builds(canonical_base)
    _copy_builds(canonical_base, canonical_run)
    _write_full(canonical_run, seed=None)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    fake_wrapper = tmp_path / "fake-wrapper.sh"
    _write_fake_wrapper(fake_wrapper)
    env = dict(os.environ)
    for key in POLLUTING:
        env.pop(key, None)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "CONTRACT_HELPER": str(CONTRACT),
            "SEED43_WRAPPER": str(fake_wrapper),
            "OLD_DECISION_LOCK": str(tmp_path / "old-decision.lock"),
            "TAIL_RUN_DIR": str(tmp_path / "tail"),
            "CANONICAL_BASE_ROOT": str(canonical_base),
            "CANONICAL_RUN_ROOT": str(canonical_run),
            "SEED43_BASE_ROOT": str(tmp_path / "seed43-base"),
            "SEED43_RUN_ROOT": str(tmp_path / "seed43-run"),
            "RAWFC_SEED43_PROC_ROOT": str(proc_root),
            "TAIL_NOW_EPOCH": now,
            "TAIL_CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "DRY_RUN": dry_run,
            "FAKE_CALLS": str(tmp_path / "calls.txt"),
        }
    )
    return env


def _run_tail(tmp_path: Path, *, now: str, dry_run: str = "false", **updates: str) -> subprocess.CompletedProcess[str]:
    env = _base_env(tmp_path, now=now, dry_run=dry_run)
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


def test_seed43_config_changes_only_labels_seed_tau_and_swanlab() -> None:
    base = _load_policy(BASE_CONFIG)
    actual = _load_policy(CONFIG)
    expected = copy.deepcopy(base)
    expected["experiment"].update(  # type: ignore[union-attr]
        {
            "name": "rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_seed43",
            "case_suffix": "__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_seed43",
            "run_label": "rawfc-atom-anchor-v0.2-structure-only-fullpool-minmax5-10-baseline20-seed43",
            "run_header_label": "rawfc-atom-anchor-v0.2-structure-only-fullpool-minmax5-10-baseline20-seed43-full",
        }
    )
    expected["sft_train"]["seed"] = 43  # type: ignore[index]
    expected["eval"]["run_tau_eval"] = "false"  # type: ignore[index]
    expected["swanlab"]["experiment_name"] = (  # type: ignore[index]
        "rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_"
        "structure_only_fullpool_minmax5_10_baseline20_seed43"
    )
    assert actual == expected


def test_seed43_wrapper_dry_run_has_exact_root_port_cache_and_no_tau() -> None:
    env = dict(os.environ)
    for key in POLLUTING:
        env.pop(key, None)
    env.update(
        {
            "PYTHON_BIN": "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python",
            "MODE": "train",
            "DRY_RUN": "true",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }
    )
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert str(CONFIG.relative_to(ROOT)) in result.stdout
    assert "port=29683" in result.stdout
    assert "rawfc_structure_only_baseline20_seed43" in result.stdout
    assert "baseline20_seed43_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in result.stdout
    assert "RUN_TAU_EVAL=false" in result.stdout


def test_seed43_wrapper_refuses_polluting_case_root() -> None:
    env = dict(os.environ)
    for key in POLLUTING:
        env.pop(key, None)
    env.update({"CASE_ROOT": "/tmp/wrong-run", "DRY_RUN": "true", "MODE": "train"})
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing inherited CASE_ROOT" in result.stdout


def test_tail_dry_run_emits_build_then_full_without_invoking_wrapper(tmp_path: Path) -> None:
    result = _run_tail(tmp_path, now=_epoch(10, 0), dry_run="true")
    assert result.returncode == 0, result.stdout
    assert "stage=old_decision_lock status=acquired" in result.stdout
    assert "stage=build status=dry_run" in result.stdout
    assert "stage=full status=dry_run" in result.stdout
    assert result.stdout.index("stage=build") < result.stdout.index("stage=full")
    assert not (tmp_path / "calls.txt").exists()
    manifest = json.loads((tmp_path / "tail/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["roots"]["seed43_run"] == str(tmp_path / "seed43-run")


def test_tail_full_fake_run_verifies_build_seed_metrics_and_process_exit(tmp_path: Path) -> None:
    result = _run_tail(tmp_path, now=_epoch(10, 0))
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").splitlines() == [
        "build",
        "full",
    ]
    manifest = json.loads((tmp_path / "tail/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["seed43_contract"]["status"] == "complete"
    assert manifest["seed43_contract"]["seed"]["effective"] == 43
    assert manifest["seed43_contract"]["metrics"]["val"]["num_samples"] == 200
    assert manifest["seed43_contract"]["metrics"]["test"]["num_samples"] == 200
    for scope in ("base", "lora"):
        for split in ("train", "val", "test"):
            assert manifest["build_contract"][scope]["splits"][split]["status"] == "matched"
    assert [item["audit"]["count"] for item in manifest["accelerate_audits"]] == [
        0,
        0,
        0,
    ]


def test_tail_blocks_after_safe_start_without_touching_wrapper(tmp_path: Path) -> None:
    result = _run_tail(tmp_path, now=_epoch(10, 20, 1))
    assert result.returncode == 0, result.stdout
    assert "status=skipped_deadline" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_tail_blocks_incomplete_canonical_full_contract(tmp_path: Path) -> None:
    env = _base_env(tmp_path, now=_epoch(10, 0))
    _write_full(Path(env["CANONICAL_RUN_ROOT"]), seed=None, num_samples=199)
    result = subprocess.run(
        ["bash", str(TAIL)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "status=skipped_canonical_incomplete" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_tail_blocks_stale_incomplete_seed43_train(tmp_path: Path) -> None:
    env = _base_env(tmp_path, now=_epoch(10, 0))
    stale = Path(env["SEED43_RUN_ROOT"]) / "train/latest_state"
    stale.mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(TAIL)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 4, result.stdout
    assert "status=blocked_stale" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_tail_strictly_skips_existing_complete_seed43(tmp_path: Path) -> None:
    env = _base_env(tmp_path, now=_epoch(10, 0))
    seed_root = Path(env["SEED43_RUN_ROOT"])
    _copy_builds(Path(env["CANONICAL_RUN_ROOT"]), seed_root)
    _write_full(seed_root, seed=43)
    result = subprocess.run(
        ["bash", str(TAIL)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "status=skipped_complete" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_tail_build_sha_mismatch_fails_before_full_launch(tmp_path: Path) -> None:
    result = _run_tail(
        tmp_path,
        now=_epoch(10, 0),
        FAKE_MISMATCH_SPLIT="val",
    )
    assert result.returncode == 6, result.stdout
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").splitlines() == [
        "build"
    ]
    assert "status=build_contract_failed" in result.stdout
    manifest = json.loads((tmp_path / "tail/manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_contract"]["base"]["splits"]["val"]["status"] == "mismatch"


def test_tail_blocks_when_any_accelerate_launcher_is_present(tmp_path: Path) -> None:
    env = _base_env(tmp_path, now=_epoch(10, 0))
    proc = Path(env["RAWFC_SEED43_PROC_ROOT"]) / "123"
    proc.mkdir()
    (proc / "cmdline").write_bytes(
        b"/usr/bin/python\0/opt/env/bin/accelerate\0launch\0--num_processes\04\0"
    )
    result = subprocess.run(
        ["bash", str(TAIL)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 5, result.stdout
    assert "status=blocked_accelerate" in result.stdout
    assert not (tmp_path / "calls.txt").exists()


def test_tail_dry_run_refuses_live_old_decision_lock(tmp_path: Path) -> None:
    env = _base_env(tmp_path, now=_epoch(10, 0), dry_run="true")
    lock_path = Path(env["OLD_DECISION_LOCK"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(TAIL)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    assert result.returncode == 73, result.stdout
    assert "status=blocked_live_lock" in result.stdout
    assert not (tmp_path / "calls.txt").exists()
