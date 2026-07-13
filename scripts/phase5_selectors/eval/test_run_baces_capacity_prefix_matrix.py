from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/phase5_selectors/eval/run_baces_capacity_prefix_matrix.sh"


def _write_fixture(tmp_path: Path) -> dict[str, str]:
    inference_config = tmp_path / "train.resolved.yaml"
    inference_config.write_text("{}\n", encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "native_command": [sys.executable],
                "checkpoint": {
                    "run_dir": "/fixture/verifier/train",
                    "checkpoint": "best",
                    "adapter_sha256": "a" * 64,
                },
                "artifacts": {
                    "inference_config": {"path": str(inference_config)},
                    "predictions": {"path": "/fixture/native_predictions.jsonl"},
                    "metrics": {"path": "/fixture/native_metrics.json"},
                    "build": {"path": "/fixture/native_build.jsonl"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source_root = tmp_path / "source"
    cells = []
    for selector in ("baces_exact", "learned_marginal"):
        relative_trace = (
            f"{selector}__ordinal_replay_minmax5_10/selection_trace_val.jsonl"
        )
        trace = source_root / relative_trace
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text("{}\n", encoding="utf-8")
        cells.append(
            {
                "cell_id": f"{selector}__ordinal_replay_minmax5_10",
                "selector_level": selector,
                "controller_level": "ordinal_replay_minmax5_10",
                "trace_file": relative_trace,
                "ready": True,
            }
        )
    source_manifest = source_root / "manifest.json"
    source_manifest.write_text(
        json.dumps({"cells": cells}) + "\n",
        encoding="utf-8",
    )
    build_config = tmp_path / "build_config.yaml"
    build_config.write_text("{}\n", encoding="utf-8")

    plan_root = tmp_path / "plans"
    matrix_root = tmp_path / "matrix"
    return {
        **os.environ,
        "REFERENCE_CONTRACT": str(reference),
        "SOURCE_FACTORIAL_MANIFEST": str(source_manifest),
        "BUILD_CONFIG": str(build_config),
        "PYTHON_BIN": sys.executable,
        "CONFIG": str(inference_config),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "DIAGNOSTIC_BUILD_ROOT": str(tmp_path / "diagnostic"),
        "MAX_PLAN_ROOT": str(tmp_path / "maximal"),
        "PLAN_ROOT": str(plan_root),
        "PLAN_MANIFEST": str(plan_root / "manifest.json"),
        "FORMAL_BUILD_ROOT": str(tmp_path / "formal"),
        "MATRIX_ROOT": str(matrix_root),
        "MATRIX_MANIFEST": str(matrix_root / "manifest.json"),
        "OUTPUT_DIR": str(tmp_path / "inference"),
        "ACCELERATE_BIN": "/fixture/bin/accelerate",
        "EVAL_NPROC_PER_NODE": "1",
        "MIN_K": "1",
        "MAX_K": "3",
        "SAMPLE_LIMIT": "7",
        "UNSAFE_SKIP_EQUIVALENCE_GATE": "true",
        "DRY_RUN": "true",
    }


def test_capacity_prefix_wrapper_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], cwd=ROOT, check=True)


def test_capacity_prefix_wrapper_dry_run_materializes_exact_grid(tmp_path: Path) -> None:
    env = _write_fixture(tmp_path)

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    command_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("+")
    ]
    diagnostic = [
        line
        for line in command_lines
        if "build_trace_verifier_data.py" in line
        and "--prompt-evidence-policy fixed_topk" in line
    ]
    formal = [
        line
        for line in command_lines
        if "build_trace_verifier_data.py" in line
        and "--prompt-evidence-policy selected_set" in line
    ]
    assert len(diagnostic) == 2
    assert len(formal) == 6
    assert all("--top-k 3" in line and "--sample-limit 7" in line for line in diagnostic)
    assert all(
        "--val-selection-plan" in line
        and "--forbid-prompt-truncation" in line
        and "--sample-limit 7" in line
        for line in formal
    )
    for selector in ("baces_exact", "learned_marginal"):
        exact_trace = (
            tmp_path
            / "source"
            / f"{selector}__ordinal_replay_minmax5_10"
            / "selection_trace_val.jsonl"
        )
        assert any(f"--val-trace {exact_trace}" in line for line in diagnostic)
        for k in range(1, 4):
            assert any(
                f"{selector}__prefix_k{k:02d}/selection_plan_val.jsonl" in line
                for line in formal
            )

    joined = "\n".join(command_lines)
    assert "project_capacity_prefix_plan.py" in joined
    assert joined.count("project_capacity_prefix_plan.py") == 2
    assert "expand_capacity_prefix_plans.py" in joined
    assert "--selector-plan baces_exact=" in joined
    assert "--selector-plan learned_marginal=" in joined
    assert "materialize_capacity_prefix_matrix.py" in joined
    assert joined.count("materialize_capacity_policy_from_traces.py") == 1
    assert "--policy-id ordinal_replay_minmax5_10" in joined
    assert f"--source-factorial-manifest {tmp_path / 'source' / 'manifest.json'}" in joined
    assert "--trace-order-field selector_available_ordered_indices" in joined
    assert "-m sft.label_token_matrix_infer prepare" in joined
    assert "-m sft.label_token_matrix_infer infer" in joined
    assert "-m sft.label_token_matrix_infer fanout" in joined
    assert "--unsafe-skip-equivalence-gate" in joined
    assert "-m sft.capacity_prefix_analysis" in joined
    assert "--include-fixed-policies" in joined
    assert "--policy ordinal_replay_minmax5_10=" in joined


def test_capacity_prefix_wrapper_dry_run_propagates_force_flags(tmp_path: Path) -> None:
    env = _write_fixture(tmp_path)
    env.update(
        {
            "PHASES": "project,plans,policy,manifest,prepare,infer,fanout,stats",
            "FORCE_PROJECT": "true",
            "FORCE_PLANS": "true",
            "FORCE_POLICY": "true",
            "FORCE_MANIFEST": "true",
            "FORCE_PREPARE": "true",
            "FORCE_INFER": "true",
            "FORCE_FANOUT": "true",
            "FORCE_STATS": "true",
            "CAPACITY_POLICIES": "fixed2=/fixture/fixed2.jsonl,cc=/fixture/cc.jsonl",
        }
    )

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = completed.stdout

    project_lines = [
        line for line in stdout.splitlines() if "project_capacity_prefix_plan.py" in line
    ]
    assert len(project_lines) == 2
    assert all("--overwrite" in line for line in project_lines)
    assert "expand_capacity_prefix_plans.py" in stdout
    assert "materialize_capacity_prefix_matrix.py" in stdout
    assert stdout.count("--overwrite") == 5
    assert "--force-prepare" in stdout
    assert "--force-infer" in stdout
    assert "--force-fanout" in stdout
    assert "--policy fixed2=/fixture/fixed2.jsonl" in stdout
    assert "--policy cc=/fixture/cc.jsonl" in stdout
    assert "--force" in stdout


def test_capacity_prefix_wrapper_full_run_uses_frozen_native_gate(tmp_path: Path) -> None:
    env = _write_fixture(tmp_path)
    env.update(
        {
            "PHASES": "fanout",
            "SAMPLE_LIMIT": "",
            "MAX_K": "10",
            "UNSAFE_SKIP_EQUIVALENCE_GATE": "false",
        }
    )

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--equivalence-gate-cell baces_exact__prefix_k05" in completed.stdout
    assert "--equivalence-gate-reference-contract" in completed.stdout
    assert "--unsafe-skip-equivalence-gate" not in completed.stdout


def test_capacity_prefix_wrapper_rejects_sampled_native_gate(tmp_path: Path) -> None:
    env = _write_fixture(tmp_path)
    env["UNSAFE_SKIP_EQUIVALENCE_GATE"] = "false"

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "SAMPLE_LIMIT cannot use the full native equivalence reference" in completed.stderr


def test_capacity_prefix_wrapper_default_sample_paths_are_isolated(tmp_path: Path) -> None:
    env = _write_fixture(tmp_path)
    for name in (
        "ARTIFACT_ROOT",
        "DIAGNOSTIC_BUILD_ROOT",
        "MAX_PLAN_ROOT",
        "PLAN_ROOT",
        "PLAN_MANIFEST",
        "FORMAL_BUILD_ROOT",
        "MATRIX_ROOT",
        "MATRIX_MANIFEST",
        "OUTPUT_DIR",
    ):
        env.pop(name, None)
    env["PHASES"] = "diagnostic_build,project,plans,formal_build,manifest,prepare"

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "09_baces_capacity_prefix_v0_1/val__sample7" in completed.stdout
    assert "baces_capacity_prefix_diagnostic_v0_1__val__sample7" in completed.stdout
    assert "baces_capacity_prefix_v0_1__val__sample7" in completed.stdout
    assert "baces_capacity_prefix_v0_1/val__sample7/" in completed.stdout


def test_capacity_prefix_wrapper_cascades_upstream_force(tmp_path: Path) -> None:
    env = _write_fixture(tmp_path)
    env.update(
        {
            "PHASES": "project,plans,formal_build,manifest,prepare,infer,fanout,stats",
            "FORCE_PROJECT": "true",
        }
    )

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = completed.stdout
    assert stdout.count("project_capacity_prefix_plan.py") == 2
    assert "expand_capacity_prefix_plans.py" in stdout and "--overwrite" in stdout
    assert stdout.count("--prompt-evidence-policy selected_set") == 6
    assert "materialize_capacity_prefix_matrix.py" in stdout
    assert "--force-prepare" in stdout
    assert "--force-infer" in stdout
    assert "--force-fanout" in stdout
    assert "-m sft.capacity_prefix_analysis" in stdout and "--force" in stdout


def test_capacity_prefix_wrapper_contract_rejects_grid_change_on_same_roots(
    tmp_path: Path,
) -> None:
    env = _write_fixture(tmp_path)
    env.update({"PHASES": "contract", "DRY_RUN": "false"})

    first = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    contract = Path(env["ARTIFACT_ROOT"]) / "prefix_run_contract.json"
    assert contract.is_file()
    assert "froze run contract" in first.stdout

    changed = dict(env)
    changed.update({"SELECTORS": "baces_exact", "MAX_K": "2"})
    second = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=changed,
        text=True,
        capture_output=True,
    )

    assert second.returncode == 2
    assert "run contract mismatch" in second.stderr


def test_capacity_prefix_wrapper_contract_rejects_sample_full_explicit_root_collision(
    tmp_path: Path,
) -> None:
    env = _write_fixture(tmp_path)
    env.update({"PHASES": "contract", "DRY_RUN": "false"})
    subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    full = dict(env)
    full["SAMPLE_LIMIT"] = ""
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=full,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "run contract mismatch" in completed.stderr


def test_capacity_prefix_wrapper_refuses_to_backsign_legacy_outputs(
    tmp_path: Path,
) -> None:
    env = _write_fixture(tmp_path)
    env.update({"PHASES": "contract", "DRY_RUN": "false"})
    legacy = Path(env["FORMAL_BUILD_ROOT"]) / "legacy.txt"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("stale\n", encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "legacy outputs without a contract" in completed.stderr
    assert not (Path(env["ARTIFACT_ROOT"]) / "prefix_run_contract.json").exists()
