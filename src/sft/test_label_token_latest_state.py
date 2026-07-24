from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from sft import label_token_trainer


class _FakeAccelerator:
    is_main_process = True
    num_processes = 1

    def __init__(self) -> None:
        self.barrier_count = 0
        self.save_state_count = 0

    def wait_for_everyone(self) -> None:
        self.barrier_count += 1

    def save_state(self, output_dir: str) -> None:
        self.save_state_count += 1
        model_dir = Path(output_dir) / "pytorch_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"new-{self.save_state_count}.bin").write_bytes(b"new")


class _FakeLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[object, ...]] = []
        self.warnings: list[tuple[object, ...]] = []

    def info(self, *args: object) -> None:
        self.infos.append(args)

    def warning(self, *args: object) -> None:
        self.warnings.append(args)


def _write_old_state(path: Path, *, marker: str = "old.bin") -> None:
    model_dir = path / "pytorch_model"
    model_dir.mkdir(parents=True)
    (model_dir / marker).write_bytes(b"old")


def _save_latest(
    output_dir: Path,
    accelerator: _FakeAccelerator,
    logger: _FakeLogger,
    *,
    global_step: int = 650,
) -> bool:
    return label_token_trainer._save_latest_training_state(
        accelerator=accelerator,
        output_dir=output_dir,
        train_cfg={},
        epoch=2,
        next_batch_index=17,
        global_step=global_step,
        best_score=0.6545,
        no_improve_count=6,
        max_train_steps=1000,
        active_logger=logger,
        enabled=True,
    )


def test_save_latest_state_publishes_complete_replacement(tmp_path: Path) -> None:
    state_dir = tmp_path / "latest_state"
    _write_old_state(state_dir)
    accelerator = _FakeAccelerator()
    logger = _FakeLogger()

    assert _save_latest(tmp_path, accelerator, logger)

    assert accelerator.barrier_count == 4
    assert accelerator.save_state_count == 1
    assert (state_dir / "pytorch_model/new-1.bin").read_bytes() == b"new"
    assert not (state_dir / "pytorch_model/old.bin").exists()
    assert json.loads((state_dir / "trainer_state.json").read_text(encoding="utf-8"))["global_step"] == 650
    assert not [path for path in tmp_path.iterdir() if ".retired" in path.name]


def test_save_reclaims_legacy_uuid_retired_directory_before_allocating_staging(tmp_path: Path) -> None:
    state_dir = tmp_path / "latest_state"
    legacy_retired = tmp_path / ".latest_state.retired-olduuid"
    _write_old_state(state_dir)
    _write_old_state(legacy_retired, marker="legacy.bin")
    accelerator = _FakeAccelerator()
    logger = _FakeLogger()

    assert _save_latest(tmp_path, accelerator, logger)

    assert accelerator.save_state_count == 1
    assert (state_dir / "pytorch_model/new-1.bin").read_bytes() == b"new"
    assert not legacy_retired.exists()
    assert not [path for path in tmp_path.iterdir() if ".retired" in path.name]


def test_consecutive_saves_do_not_accumulate_retired_states_under_persistent_enotempty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "latest_state"
    _write_old_state(state_dir)
    accelerator = _FakeAccelerator()
    logger = _FakeLogger()
    real_rmtree = label_token_trainer.shutil.rmtree

    def nfs_rmtree(path: str | Path, *args: object, **kwargs: object) -> None:
        target = Path(path)
        if ".retired" in target.name:
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target / "pytorch_model"))
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(label_token_trainer.shutil, "rmtree", nfs_rmtree)
    monkeypatch.setattr(label_token_trainer.time, "sleep", lambda _seconds: None)

    first_saved = _save_latest(tmp_path, accelerator, logger, global_step=650)
    second_saved = _save_latest(tmp_path, accelerator, logger, global_step=700)

    assert first_saved is True
    assert second_saved is False
    assert accelerator.barrier_count == 6
    assert accelerator.save_state_count == 1
    assert (state_dir / "pytorch_model/new-1.bin").read_bytes() == b"new"
    assert json.loads((state_dir / "trainer_state.json").read_text(encoding="utf-8"))["global_step"] == 650
    assert (tmp_path / ".latest_state.retired").is_dir()
    assert not (tmp_path / ".latest_state.tmp").exists()
    assert [path.name for path in tmp_path.iterdir() if ".retired" in path.name] == [".latest_state.retired"]


def test_persistent_stale_staging_cleanup_skips_save_without_allocating_another_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "latest_state"
    tmp_state_dir = tmp_path / ".latest_state.tmp"
    _write_old_state(state_dir)
    _write_old_state(tmp_state_dir, marker="stale.bin")
    accelerator = _FakeAccelerator()
    logger = _FakeLogger()

    def nfs_rmtree(path: str | Path, *args: object, **kwargs: object) -> None:
        target = Path(path)
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target / "pytorch_model"))

    monkeypatch.setattr(label_token_trainer.shutil, "rmtree", nfs_rmtree)
    monkeypatch.setattr(label_token_trainer.time, "sleep", lambda _seconds: None)

    assert _save_latest(tmp_path, accelerator, logger) is False

    assert accelerator.barrier_count == 2
    assert accelerator.save_state_count == 0
    assert (state_dir / "pytorch_model/old.bin").read_bytes() == b"old"
    assert not tmp_state_dir.exists()
    assert (tmp_path / ".latest_state.tmp.retired").is_dir()
    assert [path.name for path in tmp_path.iterdir() if ".retired" in path.name] == [
        ".latest_state.tmp.retired"
    ]


def test_retired_cleanup_uses_bounded_retry_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retired = tmp_path / ".latest_state.retired"
    _write_old_state(retired)
    logger = _FakeLogger()
    attempts = 0
    delays: list[float] = []

    def nfs_rmtree(path: str | Path, *args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        target = Path(path)
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target / "pytorch_model"))

    monkeypatch.setattr(label_token_trainer.shutil, "rmtree", nfs_rmtree)
    monkeypatch.setattr(label_token_trainer.time, "sleep", delays.append)

    assert not label_token_trainer._cleanup_retired_directory(retired, active_logger=logger)
    assert attempts == len(label_token_trainer._RETIRED_CLEANUP_BACKOFF_SECONDS) + 1
    assert delays == list(label_token_trainer._RETIRED_CLEANUP_BACKOFF_SECONDS)
    assert retired.is_dir()
    assert len(logger.warnings) == 1


def test_latest_state_publish_rolls_back_when_new_directory_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "latest_state"
    tmp_state_dir = tmp_path / ".latest_state.tmp"
    _write_old_state(state_dir)
    _write_old_state(tmp_state_dir, marker="new.bin")
    logger = _FakeLogger()
    real_rename = Path.rename

    def fail_new_publish(path: Path, target: str | Path) -> Path:
        if path == tmp_state_dir and Path(target) == state_dir:
            raise OSError("simulated publish failure")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_new_publish)

    assert not label_token_trainer._publish_latest_state_directory(
        tmp_state_dir=tmp_state_dir,
        state_dir=state_dir,
        active_logger=logger,
    )

    assert (state_dir / "pytorch_model/old.bin").read_bytes() == b"old"
    assert (tmp_state_dir / "pytorch_model/new.bin").read_bytes() == b"old"
    assert not (tmp_path / ".latest_state.retired").exists()


def test_training_complete_is_not_blocked_by_nfs_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "latest_state"
    tmp_state_dir = tmp_path / ".latest_state.tmp"
    _write_old_state(state_dir)
    _write_old_state(tmp_state_dir, marker="stale.bin")
    accelerator = _FakeAccelerator()
    logger = _FakeLogger()

    def nfs_rmtree(path: str | Path, *args: object, **kwargs: object) -> None:
        target = Path(path)
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target / "pytorch_model"))

    monkeypatch.setattr(label_token_trainer.shutil, "rmtree", nfs_rmtree)
    monkeypatch.setattr(label_token_trainer.time, "sleep", lambda _seconds: None)

    label_token_trainer._write_training_complete(
        accelerator=accelerator,
        output_dir=tmp_path,
        train_cfg={},
        global_step=750,
        best_score=0.6545,
        active_logger=logger,
    )

    assert accelerator.barrier_count == 2
    assert json.loads((tmp_path / "training_complete.json").read_text(encoding="utf-8")) == {
        "completed": True,
        "global_step": 750,
        "best_score": 0.6545,
    }
    assert not state_dir.exists()
    assert not tmp_state_dir.exists()
    assert (tmp_path / ".latest_state.retired").is_dir()
    assert (tmp_path / ".latest_state.tmp.retired").is_dir()
    assert len([path for path in tmp_path.iterdir() if ".retired" in path.name]) == 2
    assert len(logger.warnings) == 2
