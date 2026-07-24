from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment/mrec_v0.2/scifact_atom_union_structure_only_fullpool_minmax9_9.yaml"
WRAPPER = ROOT / "scripts/phase13_scifact/05_train_eval_scifact_structure_only_fullpool_lora.sh"


def test_scifact_structure_only_config_has_isolated_three_split_artifacts() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    trace = payload["trace"]
    assert trace["splits"] == ["train", "val", "test"]
    assert trace["weight_training"]["enabled"] is False
    assert trace["weight_training"]["epochs"] == 30
    assert "structure_only" in trace["output_root"]
    assert "structure_only" in trace["weight_file"]
    assert trace["selector_name"].endswith("structure_only_fullpool")


def test_scifact_structure_only_wrapper_dry_run_uses_strict_trainer() -> None:
    env = dict(os.environ)
    env.update(
        {
            "MODE": "build",
            "DRY_RUN": "true",
            "PYTHON_BIN": env.get(
                "PYTHON_BIN",
                "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python",
            ),
        }
    )
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    output = completed.stdout
    assert "scifact_atom_union_structure_only_fullpool_minmax9_9.yaml" in output
    assert "MODE=build" in output
    assert "train_mrec_learned_marginal_proxy.py" not in output
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    assert "run_liar_raw_mrec_structure_only_weights.sh" in wrapper_text
    assert "MREC_AUTO_TRAIN_WEIGHTS=false" in wrapper_text
