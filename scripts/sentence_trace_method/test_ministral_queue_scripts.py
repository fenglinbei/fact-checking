from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "sentence_trace_method"
MINISTRAL_PYTHON = "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"


def test_ministral_wrappers_default_to_gemma4_python() -> None:
    scripts = sorted(SCRIPT_DIR.glob("*ministral*.sh"))
    assert scripts, "expected at least one Ministral wrapper script"

    old_default = 'PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"'
    expected_default = f'PYTHON_BIN="${{PYTHON_BIN:-{MINISTRAL_PYTHON}}}"'

    offenders: list[str] = []
    missing_default: list[str] = []
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        if old_default in text:
            offenders.append(str(script.relative_to(ROOT)))
        if expected_default not in text:
            missing_default.append(str(script.relative_to(ROOT)))

    assert offenders == []
    assert missing_default == []


def test_ministral_fullft_qec_queue_enforces_order_deadline_and_checkpoint_signal() -> None:
    queue_script = SCRIPT_DIR / "run_ministral3_fullft_qec_queue_until_0900.sh"
    text = queue_script.read_text(encoding="utf-8")

    rawfc = "run_rawfc_ministral3_v0_7_adaptive5_10_lr1e5_ep12_fullft_aligned.sh"
    qec = "run_qec_v1_ministral3_prompt_matrix.sh"
    liar = "run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_ep12_fullft_aligned.sh"

    assert text.index(rawfc) < text.index(qec) < text.index(liar)
    assert 'PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"' in text
    assert 'QUEUE_DEADLINE="${QUEUE_DEADLINE:-$(date +%F) 09:00:00}"' in text
    assert "SAVE_LATEST_TRAIN_STATE=true" in text
    assert "RESUME_LATEST_TRAIN_STATE=true" in text
    assert "trainer_pids_for_tree" in text
    assert "kill -TERM $trainer_pids" in text


def test_fullpool_policy_wrapper_moves_vllm_runtime_cache_to_project_outputs() -> None:
    wrapper = (
        SCRIPT_DIR
        / "run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
    )
    text = wrapper.read_text(encoding="utf-8")

    assert (
        'MREC_RUNTIME_CACHE_ROOT="${MREC_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/mrec_fullpool_policy}"'
        in text
    )
    assert 'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MREC_RUNTIME_CACHE_ROOT}/xdg}"' in text
    assert 'export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${MREC_RUNTIME_CACHE_ROOT}/vllm}"' in text
    assert (
        'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${MREC_RUNTIME_CACHE_ROOT}/torchinductor}"'
        in text
    )
    assert 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MREC_RUNTIME_CACHE_ROOT}/triton}"' in text
    assert 'mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"' in text
    assert "RUNTIME_CACHE_ROOT=%s XDG_CACHE_HOME=%s VLLM_CACHE_ROOT=%s TORCHINDUCTOR_CACHE_DIR=%s TRITON_CACHE_DIR=%s" in text


def test_rawfc_ministral3_lora_tuning_queue_waits_active_first_then_runs_ordered() -> None:
    queue_script = SCRIPT_DIR / "run_rawfc_ministral3_lora_tuning_queue.sh"
    text = queue_script.read_text(encoding="utf-8")

    first = "run_rawfc_ministral3_v0_7_adaptive5_10_lora_r32a64_d005_lr1e5_ep12.sh"
    second = "run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d010_lr1e5_ep12.sh"
    third = "run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d005_lr5e6_ep12.sh"

    assert text.index(first) < text.index(second) < text.index(third)
    assert 'PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"' in text
    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in text
    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in text
    assert "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d005_ebs16_lr5em6_ep12_eval50_pat8_rawfc" in text
    assert "eval100_pat8_rawfc" not in text
    assert "active_pids_for_run" in text
    assert "wait_for_active_stage" in text
    assert "training_complete" in text
    assert "queue stop requested while waiting for externally active" in text
    assert "SAVE_LATEST_TRAIN_STATE=true" in text
    assert "RESUME_LATEST_TRAIN_STATE=true" in text


def test_rawfc_ministral3_lora_tuning_queue_dry_run_runs_all_stages(tmp_path: Path) -> None:
    queue_script = SCRIPT_DIR / "run_rawfc_ministral3_lora_tuning_queue.sh"
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "RUN_TAU_EVAL": "false",
        "PREPARE_V0_7_SOURCES": "false",
        "PREPARE_SELECTOR_SOURCES": "false",
        "QUEUE_LOG_ROOT": str(tmp_path),
        "POLL_SECONDS": "1",
    }
    result = subprocess.run(
        ["bash", str(queue_script)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr

    assert "starting rawfc_lora_r32a64_d005_lr1e5" in output
    assert "starting rawfc_lora_r16a32_d010_lr1e5" in output
    assert "starting rawfc_lora_r16a32_d005_lr5e6" in output
    assert output.count("dry-run completed; skipping completed-marker check") == 3
    assert "queue completed successfully" in output
