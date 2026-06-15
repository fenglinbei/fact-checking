from __future__ import annotations

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
