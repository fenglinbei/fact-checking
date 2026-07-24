from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/experiment/mrec_v0.2"
CONFIG_LOADER = ROOT / "scripts/sentence_trace_method/mrec_policy_config.py"
QUEUE = ROOT / (
    "scripts/sentence_trace_method/"
    "run_liar_raw_prompt_evidence_policy_remaining_queue.sh"
)
BASE_WRAPPER = ROOT / (
    "scripts/sentence_trace_method/"
    "run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
)
PYTHON_BIN = "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"

CASES = {
    "minmax3_3": {
        "policy": "minmax",
        "min_count": "3",
        "max_count": "3",
        "budget": "",
        "top_k": "3",
    },
    "minmax3_8": {
        "policy": "minmax",
        "min_count": "3",
        "max_count": "8",
        "budget": "",
        "top_k": "8",
    },
    "minmax5_12": {
        "policy": "minmax",
        "min_count": "5",
        "max_count": "12",
        "budget": "",
        "top_k": "12",
    },
    "budget512": {
        "policy": "budget",
        "min_count": "1",
        "max_count": "100",
        "budget": "512",
        "top_k": "100",
    },
    "budget768": {
        "policy": "budget",
        "min_count": "1",
        "max_count": "100",
        "budget": "768",
        "top_k": "100",
    },
}


def _config_exports(slug: str) -> dict[str, str]:
    config = CONFIG_ROOT / f"learned_marginal_proxy_fullpool_{slug}.yaml"
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


def test_remaining_policy_configs_preserve_shared_trace_and_capacity_contract() -> None:
    shared_values: set[tuple[str, str, str]] = set()
    case_suffixes: set[str] = set()

    for slug, expected in CASES.items():
        exports = _config_exports(slug)
        assert exports["PROMPT_EVIDENCE_POLICY"] == expected["policy"]
        assert exports["PROMPT_EVIDENCE_MIN_COUNT"] == expected["min_count"]
        assert exports["PROMPT_EVIDENCE_MAX_COUNT"] == expected["max_count"]
        assert exports["PROMPT_EVIDENCE_TOKEN_BUDGET"] == expected["budget"]
        assert exports["TRACE_TOP_K"] == expected["top_k"]
        assert exports["CASE_SUFFIX"].endswith(f"_fullpool_{slug}")
        case_suffixes.add(exports["CASE_SUFFIX"])
        shared_values.add(
            (
                exports["TRACE_ROOT"],
                exports["WEIGHT_FILE"],
                exports["EXPECTED_SELECTOR_NAME"],
            )
        )

    assert len(case_suffixes) == len(CASES)
    assert shared_values == {
        (
            "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
            "05_mrec_v0_2_learned_marginal_proxy_fullpool",
            "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
            "05_mrec_v0_2_learned_marginal_proxy/weights/weights.json",
            "mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool",
        )
    }


def test_remaining_policy_queue_dry_run_expands_all_five_cells(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "MODE": "full",
        "FORCE_BUILD": "true",
        "QUEUE_ID": "test_prompt_evidence_remaining",
        "QUEUE_LOG_ROOT": str(tmp_path / "queue"),
        "MREC_RUNTIME_CACHE_ROOT": str(tmp_path / "cache"),
    }
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr

    stages = ("fixed3", "minmax3_8", "minmax5_12", "budget512", "budget768")
    marker_indexes: list[int] = []
    for stage, slug in zip(stages, CASES, strict=True):
        marker = f"starting {stage}:"
        marker_indexes.append(output.index(marker))
        expected_run_root = (
            "outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_"
            f"learned_marginal_proxy_fullpool_{slug}_"
            "lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
        )
        assert (
            f"starting {stage}: config=configs/experiment/mrec_v0.2/"
            f"learned_marginal_proxy_fullpool_{slug}.yaml run_root={expected_run_root}"
        ) in output

    assert marker_indexes == sorted(marker_indexes)
    for index, (stage, slug) in enumerate(zip(stages, CASES, strict=True)):
        segment_end = marker_indexes[index + 1] if index + 1 < len(stages) else len(output)
        segment = output[marker_indexes[index] : segment_end]
        expected = CASES[slug]
        assert (
            "CASE_NAME=liar_raw__ministral3_8b__atom_anchor_v0_2_"
            f"learned_marginal_proxy_fullpool_{slug}"
        ) in segment
        assert f"PROMPT_EVIDENCE_POLICY={expected['policy']}" in segment
        assert f"PROMPT_EVIDENCE_MIN_COUNT={expected['min_count']}" in segment
        assert f"PROMPT_EVIDENCE_MAX_COUNT={expected['max_count']}" in segment
        assert f"PROMPT_EVIDENCE_TOKEN_BUDGET={expected['budget']}" in segment
        assert f"--prompt-evidence-policy {expected['policy']}" in segment
        assert f"--prompt-evidence-min-count {expected['min_count']}" in segment
        assert f"--prompt-evidence-max-count {expected['max_count']}" in segment
        assert f"--top-k {expected['top_k']}" in segment
        if expected["budget"]:
            assert f"--prompt-evidence-token-budget {expected['budget']}" in segment
        else:
            assert "--prompt-evidence-token-budget" not in segment
        assert f"{stage} dry-run completed" in segment

    assert output.count("SAVE_LATEST_TRAIN_STATE=true") == len(CASES)
    assert output.count("RESUME_LATEST_TRAIN_STATE=true") == len(CASES)
    assert (
        "shared_trace=outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
        "05_mrec_v0_2_learned_marginal_proxy_fullpool"
    ) in output
    assert "queue completed successfully" in output


def test_remaining_policy_queue_rejects_unknown_stage(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "true",
            "ONLY_STAGE": "unknown",
            "QUEUE_LOG_ROOT": str(tmp_path),
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Unknown ONLY_STAGE=unknown" in result.stderr


def test_remaining_policy_queue_rejects_shared_force_without_single_stage(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "true",
            "FORCE_MREC_BUILD": "true",
            "QUEUE_LOG_ROOT": str(tmp_path),
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "may overwrite the shared trace or weights repeatedly" in result.stderr


def test_base_wrapper_skips_existing_label_token_eval_artifacts(tmp_path: Path) -> None:
    case_root = tmp_path / "completed_eval"
    for split in ("val", "test"):
        metrics = case_root / "eval" / split / "best" / "label_token" / "metrics.json"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(BASE_WRAPPER)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHON_BIN": PYTHON_BIN,
            "DRY_RUN": "true",
            "MODE": "eval",
            "FINETUNE_MODE": "fullft",
            "CASE_ROOT": str(case_root),
            "RUN_TAU_EVAL": "false",
            "EVAL_SPLITS": "val,test",
            "CHECKPOINTS": "best",
        },
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.count("eval exists:") == 2
    assert "sft.label_token_infer" not in result.stdout
