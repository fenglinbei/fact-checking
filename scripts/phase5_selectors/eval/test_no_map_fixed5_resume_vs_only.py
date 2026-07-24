from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.phase5_selectors.eval.validate_no_map_fixed5_resume_vs_only import (
    ResumeContractError,
    _sha256,
    _validate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/phase5_selectors/eval/run_no_map_structure_fixed5_resume_vs_only_step800.sh"


def _checkpoint(root: Path, *, completed: bool) -> tuple[Path, str]:
    run_dir = root / "train"
    checkpoint = run_dir / "checkpoint-800"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixed-adapter")
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "config.resolved.yaml").write_text("{}\n", encoding="utf-8")
    if completed:
        (run_dir / "training_complete.json").write_text(
            '{"completed":true,"global_step":2000}\n', encoding="utf-8"
        )
    else:
        (run_dir / "latest_state").mkdir()
        (run_dir / "latest_state/trainer_state.json").write_text(
            '{"global_step":800}\n', encoding="utf-8"
        )
    return run_dir, _sha256(checkpoint / "adapter_model.safetensors")


def test_checkpoint_roles_are_fail_closed(tmp_path: Path) -> None:
    n_run, n_sha = _checkpoint(tmp_path / "n", completed=False)
    s_run, s_sha = _checkpoint(tmp_path / "s", completed=True)
    completion_sha = _sha256(s_run / "training_complete.json")
    assert _validate_checkpoint(n_run, role="V_N", expected_adapter_sha256=n_sha)["progress_step"] == 800
    assert _validate_checkpoint(
        s_run,
        role="V_S",
        expected_adapter_sha256=s_sha,
        expected_completion_sha256=completion_sha,
    )["training_complete"] is True
    (n_run / "training_complete.json").write_text(
        '{"completed":true,"global_step":800}\n', encoding="utf-8"
    )
    with pytest.raises(ResumeContractError, match="without training_complete"):
        _validate_checkpoint(n_run, role="V_N", expected_adapter_sha256=n_sha)


def test_runner_has_exactly_one_vs_only_infer_path() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert text.count("sft.hami_cuda_bootstrap infer") == 1
    assert "gpu_infer=verifier_s_only" in text
    assert '--output-dir "$S_OUTPUT_DIR" --run-dir "$S_RUN_DIR"' in text
    assert "verifier_n.cache" not in text
    assert '--output-dir "$S_OUTPUT_DIR" --unsafe-skip-equivalence-gate' in text
    assert "force replacement is forbidden" in text
    assert "--require-s-complete" in text


def test_dry_run_prints_vs_only_gpu_command_without_launch(tmp_path: Path) -> None:
    helper = tmp_path / "fake_contract.py"
    helper.write_text('print("{\\"status\\":\\"ready\\"}")\n', encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "true",
            "MATRIX_PYTHON_BIN": sys.executable,
            "ACCELERATE_BIN": "/definitely/not/executed/accelerate",
            "CONTRACT_HELPER": str(helper),
            "MATRIX_ROOT": str(tmp_path / "matrix"),
            "MATRIX_MANIFEST": str(tmp_path / "matrix/manifest.json"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "N_RUN_DIR": str(tmp_path / "n/train"),
            "S_RUN_DIR": str(tmp_path / "s/train"),
            "N_CAP_MANIFEST": str(tmp_path / "cap.json"),
        }
    )
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer") == 1
    assert "verifier_s.cache" in result.stdout
    assert "verifier_n.cache" not in result.stdout
    assert "--require-s-complete" not in result.stdout
    assert not (tmp_path / "output").exists()
