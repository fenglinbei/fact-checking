from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "scripts/sentence_trace_method/run_structure_only_reservation_queue.sh"
BEFORE_CUTOFF = "1784240000"
AFTER_CUTOFF = "1784260000"


def _run(tmp_path: Path, tasks: list[str], **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "true",
            "QUEUE_ID": "pytest",
            "QUEUE_RUN_DIR": str(tmp_path / "queue"),
            "QUEUE_NOW_EPOCH": BEFORE_CUTOFF,
            **extra_env,
        }
    )
    return subprocess.run(
        ["bash", str(QUEUE), *tasks],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _write_seed43_training_artifacts(root: Path, *, seed: int = 43) -> None:
    checkpoint = root / "train/checkpoint-800"
    checkpoint.mkdir(parents=True)
    (root / "train/training_complete.json").write_text(
        json.dumps({"completed": True}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"seed43-adapter")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}) + "\n",
        encoding="utf-8",
    )
    (root / "train.resolved.yaml").write_text(
        f"sft_train:\n  seed: {seed}\n",
        encoding="utf-8",
    )


def test_dry_run_preserves_task_order_and_fixed_modes(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        ["rawfc_clean:train", "scifact_clean:eval", "v_r:full"],
    )
    assert result.returncode == 0, result.stdout
    plan_lines = (tmp_path / "queue/plan.tsv").read_text(encoding="utf-8").splitlines()
    assert [line.split("\t")[1:3] for line in plan_lines[1:]] == [
        ["rawfc_clean", "train"],
        ["scifact_clean", "eval"],
        ["v_r", "full"],
    ]
    assert "MODE=train" in result.stdout
    assert "MODE=eval" in result.stdout
    assert "MODE=full" in result.stdout
    assert (tmp_path / "queue/logs/01_rawfc_clean_train.log").is_file()
    assert (tmp_path / "queue/logs/02_scifact_clean_eval.log").is_file()
    assert (tmp_path / "queue/logs/03_v_r_full.log").is_file()


def test_cutoff_blocks_first_incomplete_task_without_invoking_wrapper(tmp_path: Path) -> None:
    result = _run(tmp_path, ["v_r:train"], QUEUE_NOW_EPOCH=AFTER_CUTOFF)
    assert result.returncode == 3
    assert "CUTOFF" in result.stdout
    events = (tmp_path / "queue/events.tsv").read_text(encoding="utf-8")
    assert "cutoff_blocked" in events


def test_valid_completed_training_is_skipped_even_after_cutoff(tmp_path: Path) -> None:
    run_root = tmp_path / "completed-run"
    marker = run_root / "train/training_complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"completed": True}) + "\n", encoding="utf-8")
    result = _run(
        tmp_path,
        ["rawfc_clean:train"],
        QUEUE_NOW_EPOCH=AFTER_CUTOFF,
        QUEUE_RAWFC_CLEAN_RUN_ROOT=str(run_root),
    )
    assert result.returncode == 0, result.stdout
    assert "SKIP complete" in result.stdout
    assert "skipped_complete" in (tmp_path / "queue/events.tsv").read_text(encoding="utf-8")


def test_v_r_and_no_map_are_mutually_exclusive(tmp_path: Path) -> None:
    result = _run(tmp_path, ["v_r:full", "no_map:full"])
    assert result.returncode == 2
    assert "mutually exclusive" in result.stdout


@pytest.mark.parametrize("mode", ["train", "eval", "full"])
def test_seed43_pair_supports_modes_and_best_eval_contract(
    tmp_path: Path,
    mode: str,
) -> None:
    result = _run(tmp_path, [f"seed43_s:{mode}", f"seed43_o:{mode}"])
    assert result.returncode == 0, result.stdout

    plan_lines = (tmp_path / "queue/plan.tsv").read_text(encoding="utf-8").splitlines()
    rows = [line.split("\t") for line in plan_lines[1:]]
    assert [row[1:3] for row in rows] == [
        ["seed43_s", mode],
        ["seed43_o", mode],
    ]
    assert rows[0][3].endswith(
        "learned_marginal_structure_only_fullpool_minmax5_10_seed43.yaml"
    )
    assert rows[1][3].endswith(
        "learned_marginal_structure_only_one_shot_fullpool_minmax5_10_seed43.yaml"
    )
    assert rows[0][4].endswith(
        "run_liar_raw_ministral3_structure_only_seed43.sh"
    )
    assert rows[1][4].endswith(
        "run_liar_raw_ministral3_structure_only_one_shot_seed43.sh"
    )
    assert "seed43_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw" in rows[0][5]
    assert "one_shot_fullpool_minmax5_10_seed43_lora" in rows[1][5]
    if mode == "train":
        assert "/eval/" not in rows[0][6]
        assert "/eval/" not in rows[1][6]
    else:
        assert rows[0][6].endswith("/eval/val/best/label_token/metrics.json")
        assert rows[1][6].endswith("/eval/val/best/label_token/metrics.json")
    for row in rows:
        assert "/train/checkpoint-800/adapter_model.safetensors" in row[6]
        assert "/train/checkpoint-800/adapter_config.json" in row[6]
        assert "/train.resolved.yaml:sft_train.seed=43" in row[6]
    assert result.stdout.count("CHECKPOINTS=best") == 2


@pytest.mark.parametrize("task", ["seed43_s:full", "seed43_o:full"])
def test_seed43_single_task_fails_closed(tmp_path: Path, task: str) -> None:
    result = _run(tmp_path, [task])
    assert result.returncode == 2
    assert "must be paired exactly once" in result.stdout


def test_seed43_reverse_order_fails_closed(tmp_path: Path) -> None:
    result = _run(tmp_path, ["seed43_o:full", "seed43_s:full"])
    assert result.returncode == 2
    assert "adjacent and ordered seed43_s -> seed43_o" in result.stdout


def test_seed43_pair_must_be_adjacent_and_use_same_mode(tmp_path: Path) -> None:
    separated = _run(
        tmp_path / "separated",
        ["seed43_s:full", "rawfc_clean:full", "seed43_o:full"],
    )
    assert separated.returncode == 2
    assert "adjacent and ordered" in separated.stdout

    mixed_mode = _run(
        tmp_path / "mixed",
        ["seed43_s:train", "seed43_o:full"],
    )
    assert mixed_mode.returncode == 2
    assert "must use the same mode" in mixed_mode.stdout


def test_seed43_pair_override_is_explicit_recovery_escape_hatch(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        ["seed43_o:eval"],
        ALLOW_SEED43_PAIR_OVERRIDE="true",
    )
    assert result.returncode == 0, result.stdout
    assert "seed43_o" in result.stdout
    assert "CHECKPOINTS=best" in result.stdout


def test_seed43_full_audits_training_and_best_val_metric(tmp_path: Path) -> None:
    fake_wrapper = tmp_path / "fake-seed43-success.sh"
    fake_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'mkdir -p "$QUEUE_TASK_RUN_ROOT/train/checkpoint-800" '
        '"$QUEUE_TASK_RUN_ROOT/eval/val/best/label_token"\n'
        'printf \'{"completed":true}\\n\' > '
        '"$QUEUE_TASK_RUN_ROOT/train/training_complete.json"\n'
        'printf \'adapter\\n\' > '
        '"$QUEUE_TASK_RUN_ROOT/train/checkpoint-800/adapter_model.safetensors"\n'
        'printf \'{"peft_type":"LORA"}\\n\' > '
        '"$QUEUE_TASK_RUN_ROOT/train/checkpoint-800/adapter_config.json"\n'
        'printf \'sft_train:\\n  seed: 43\\n\' > '
        '"$QUEUE_TASK_RUN_ROOT/train.resolved.yaml"\n'
        'printf \'{"macro_f1":0.4}\\n\' > '
        '"$QUEUE_TASK_RUN_ROOT/eval/val/best/label_token/metrics.json"\n',
        encoding="utf-8",
    )
    fake_wrapper.chmod(0o755)
    result = _run(
        tmp_path,
        ["seed43_s:full", "seed43_o:full"],
        DRY_RUN="false",
        QUEUE_SEED43_S_WRAPPER=str(fake_wrapper),
        QUEUE_SEED43_O_WRAPPER=str(fake_wrapper),
        QUEUE_SEED43_S_RUN_ROOT=str(tmp_path / "seed-s"),
        QUEUE_SEED43_O_RUN_ROOT=str(tmp_path / "seed-o"),
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("DONE task=seed43_") == 2
    events = (tmp_path / "queue/events.tsv").read_text(encoding="utf-8")
    assert events.count("\tcompleted\t0\t") == 2


@pytest.mark.parametrize("invalid_task", ["seed43_s", "seed43_o"])
@pytest.mark.parametrize(
    "missing_name",
    ["adapter_model.safetensors", "adapter_config.json"],
)
def test_seed43_old_training_marker_cannot_be_skipped(
    tmp_path: Path,
    invalid_task: str,
    missing_name: str,
) -> None:
    roots = {"seed43_s": tmp_path / "seed-s", "seed43_o": tmp_path / "seed-o"}
    _write_seed43_training_artifacts(roots["seed43_s"])
    _write_seed43_training_artifacts(roots["seed43_o"])
    invalid_root = roots[invalid_task]
    (invalid_root / "train/checkpoint-800" / missing_name).unlink()

    result = _run(
        tmp_path,
        ["seed43_s:train", "seed43_o:train"],
        QUEUE_NOW_EPOCH=AFTER_CUTOFF,
        QUEUE_SEED43_S_RUN_ROOT=str(roots["seed43_s"]),
        QUEUE_SEED43_O_RUN_ROOT=str(roots["seed43_o"]),
    )
    assert result.returncode == 3, result.stdout
    assert f"CUTOFF task={invalid_task}" in result.stdout
    events = (tmp_path / "queue/events.tsv").read_text(encoding="utf-8")
    invalid_events = [line for line in events.splitlines() if f"\t{invalid_task}\t" in line]
    assert all("skipped_complete" not in line for line in invalid_events)


@pytest.mark.parametrize("seed", [42, 123])
def test_seed43_short_run_config_cannot_be_skipped(tmp_path: Path, seed: int) -> None:
    seed_s = tmp_path / "seed-s"
    seed_o = tmp_path / "seed-o"
    _write_seed43_training_artifacts(seed_s, seed=seed)
    _write_seed43_training_artifacts(seed_o)

    result = _run(
        tmp_path,
        ["seed43_s:train", "seed43_o:train"],
        QUEUE_NOW_EPOCH=AFTER_CUTOFF,
        QUEUE_SEED43_S_RUN_ROOT=str(seed_s),
        QUEUE_SEED43_O_RUN_ROOT=str(seed_o),
    )
    assert result.returncode == 3, result.stdout
    assert "CUTOFF task=seed43_s" in result.stdout
    assert "SKIP complete task=seed43_s" not in result.stdout


def test_seed43_complete_train_is_skipped_only_with_checkpoint_and_seed43(
    tmp_path: Path,
) -> None:
    seed_s = tmp_path / "seed-s"
    seed_o = tmp_path / "seed-o"
    _write_seed43_training_artifacts(seed_s)
    _write_seed43_training_artifacts(seed_o)

    result = _run(
        tmp_path,
        ["seed43_s:train", "seed43_o:train"],
        QUEUE_NOW_EPOCH=AFTER_CUTOFF,
        QUEUE_SEED43_S_RUN_ROOT=str(seed_s),
        QUEUE_SEED43_O_RUN_ROOT=str(seed_o),
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("SKIP complete task=seed43_") == 2


def test_seed43_full_does_not_skip_without_best_val_metric(tmp_path: Path) -> None:
    seed_s = tmp_path / "seed-s"
    seed_o = tmp_path / "seed-o"
    _write_seed43_training_artifacts(seed_s)
    _write_seed43_training_artifacts(seed_o)

    result = _run(
        tmp_path,
        ["seed43_s:full", "seed43_o:full"],
        QUEUE_NOW_EPOCH=AFTER_CUTOFF,
        QUEUE_SEED43_S_RUN_ROOT=str(seed_s),
        QUEUE_SEED43_O_RUN_ROOT=str(seed_o),
    )
    assert result.returncode == 3, result.stdout
    assert "CUTOFF task=seed43_s mode=full" in result.stdout
    assert "SKIP complete task=seed43_s" not in result.stdout


def test_zero_exit_without_completion_marker_fails_closed(tmp_path: Path) -> None:
    fake_wrapper = tmp_path / "fake-success.sh"
    fake_wrapper.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_wrapper.chmod(0o755)
    run_root = tmp_path / "missing-marker-run"
    result = _run(
        tmp_path,
        ["rawfc_clean:train"],
        DRY_RUN="false",
        QUEUE_RAWFC_CLEAN_WRAPPER=str(fake_wrapper),
        QUEUE_RAWFC_CLEAN_RUN_ROOT=str(run_root),
    )
    assert result.returncode == 5, result.stdout
    assert "completion marker audit failed" in result.stdout
    assert "marker_missing" in (tmp_path / "queue/events.tsv").read_text(encoding="utf-8")


def test_nonzero_task_stops_before_next_task(tmp_path: Path) -> None:
    fake_wrapper = tmp_path / "fake-fail.sh"
    fake_wrapper.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    fake_wrapper.chmod(0o755)
    second_wrapper = tmp_path / "must-not-run.sh"
    sentinel = tmp_path / "second-ran"
    second_wrapper.write_text(
        f"#!/usr/bin/env bash\ntouch {sentinel}\nexit 0\n",
        encoding="utf-8",
    )
    second_wrapper.chmod(0o755)
    result = _run(
        tmp_path,
        ["rawfc_clean:train", "scifact_clean:train"],
        DRY_RUN="false",
        QUEUE_RAWFC_CLEAN_WRAPPER=str(fake_wrapper),
        QUEUE_SCIFACT_CLEAN_WRAPPER=str(second_wrapper),
        QUEUE_RAWFC_CLEAN_RUN_ROOT=str(tmp_path / "rawfc"),
        QUEUE_SCIFACT_CLEAN_RUN_ROOT=str(tmp_path / "scifact"),
    )
    assert result.returncode == 7, result.stdout
    assert not sentinel.exists()
    assert "queue stopped" in result.stdout
