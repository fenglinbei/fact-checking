from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_script(path: str) -> str:
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        "DRY_RUN": "true",
        "PREPARE_V0_7_SOURCES": "false",
        "PREPARE_SELECTOR_SOURCES": "false",
        "RUN_TAU_EVAL": "false",
        "MODE": "build",
    }
    result = subprocess.run(
        ["bash", str(ROOT / path)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_liar_raw_ebs_lr_matrix_dry_run_expands_four_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_v0_7_liar_raw_lora_ebs_lr_matrix.sh")

    for suffix in (
        "_lora_ebs16_lr1em5_ep8_eval100_pat8_liarw",
        "_lora_ebs16_lr2em5_ep8_eval100_pat8_liarw",
        "_lora_ebs32_lr1em5_ep8_eval100_pat8_liarw",
        "_lora_ebs32_lr2em5_ep8_eval100_pat8_liarw",
    ):
        assert suffix in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_LEARNING_RATE=2e-5" in output


def test_rawfc_selector_lr_matrix_dry_run_expands_eight_cases() -> None:
    output = _run_script("scripts/sentence_trace_method/run_rawfc_lora_selector_lr_matrix.sh")

    for suffix in (
        "_lora_ebs16_lr1em5_ep8_eval100_pat8_rawfc",
        "_lora_ebs16_lr5em6_ep8_eval100_pat8_rawfc",
    ):
        assert output.count(suffix) >= 4
    for selector in (
        "sentence_rule_step_adaptive5_10",
        "v0_7_budgeted_marginal_chain_adaptive3_10",
        "v0_7_budgeted_marginal_chain_adaptive5_10",
        "v0_7_budgeted_marginal_chain_adaptive5_12",
    ):
        assert selector in output
    assert "CASE_SUFFIX=__old_adaptive5_10" in output
    assert "CASE_SUFFIX=__v0_7_bm_adaptive5_12" in output
