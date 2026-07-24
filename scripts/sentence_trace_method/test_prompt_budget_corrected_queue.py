from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/experiment/mrec_v0.2"
CONFIG_LOADER = ROOT / "scripts/sentence_trace_method/mrec_policy_config.py"
QUEUE = ROOT / "scripts/sentence_trace_method/run_liar_raw_prompt_budget_corrected_queue.sh"
PYTHON_BIN = "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"


def _config_exports(budget: int) -> dict[str, str]:
    config = CONFIG_ROOT / f"learned_marginal_proxy_fullpool_promptbudget{budget}.yaml"
    result = subprocess.run(
        [
            PYTHON_BIN,
            str(CONFIG_LOADER),
            "--config",
            str(config.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    exports: dict[str, str] = {}
    for line in result.stdout.splitlines():
        tokens = shlex.split(line)
        assert len(tokens) == 2 and tokens[0] == "export"
        key, value = tokens[1].split("=", 1)
        exports[key] = value
    return exports


def test_corrected_prompt_budget_configs_use_final_prompt_contract() -> None:
    suffixes: set[str] = set()
    shared_trace_values: set[tuple[str, str]] = set()
    for budget in (512, 768, 1024):
        exports = _config_exports(budget)
        assert exports["PROMPT_EVIDENCE_POLICY"] == "prompt_budget"
        assert exports["PROMPT_EVIDENCE_MIN_COUNT"] == "1"
        assert exports["PROMPT_EVIDENCE_MAX_COUNT"] == "100"
        assert exports["PROMPT_EVIDENCE_TOKEN_BUDGET"] == ""
        assert exports["PROMPT_EVIDENCE_PROMPT_TOKEN_BUDGET"] == str(budget)
        assert exports["PROMPT_EVIDENCE_MAX_LENGTH_GUARD"] == "error"
        assert exports["TRACE_TOP_K"] == "100"
        assert exports["CASE_SUFFIX"].endswith(f"_fullpool_promptbudget{budget}")
        suffixes.add(exports["CASE_SUFFIX"])
        shared_trace_values.add((exports["TRACE_ROOT"], exports["WEIGHT_FILE"]))

    assert len(suffixes) == 3
    assert shared_trace_values == {
        (
            "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
            "05_mrec_v0_2_learned_marginal_proxy_fullpool",
            "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
            "05_mrec_v0_2_learned_marginal_proxy/weights/weights.json",
        )
    }


def test_corrected_prompt_budget_queue_dry_run_passes_new_cli_only(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "true",
            "MODE": "build",
            "ONLY_STAGE": "promptbudget512",
            "FORCE_BUILD": "true",
            "QUEUE_ID": "test_prompt_budget_corrected",
            "QUEUE_LOG_ROOT": str(tmp_path / "queue"),
            "MREC_RUNTIME_CACHE_ROOT": str(tmp_path / "cache"),
        },
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr

    assert "starting promptbudget512:" in output
    assert "PROMPT_EVIDENCE_POLICY=prompt_budget" in output
    assert "PROMPT_EVIDENCE_TOKEN_BUDGET=" in output
    assert "PROMPT_EVIDENCE_PROMPT_TOKEN_BUDGET=512" in output
    assert "--prompt-evidence-policy prompt_budget" in output
    assert "--prompt-evidence-prompt-token-budget 512" in output
    assert "--prompt-evidence-token-budget" not in output
    assert "promptbudget512 dry-run completed" in output
    assert "queue completed successfully" in output


def test_corrected_prompt_budget_queue_rejects_unknown_stage(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "true",
            "ONLY_STAGE": "promptbudget999",
            "QUEUE_ID": "test_prompt_budget_bad_stage",
            "QUEUE_LOG_ROOT": str(tmp_path / "queue"),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Unknown ONLY_STAGE=promptbudget999" in result.stderr
