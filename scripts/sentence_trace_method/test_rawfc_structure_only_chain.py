from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    ROOT
    / "scripts/sentence_trace_method/"
    "run_rawfc_ministral3_atom_anchor_v0_2_structure_only_fullpool_minmax5_10_"
    "baseline20_lora_r16a32_d010_lr1e5_ep12_eval50.sh"
)


def test_rawfc_structure_only_chain_dry_run_is_clean_and_preserves_recipe() -> None:
    env = dict(os.environ)
    env.update(
        {
            "MODE": "full",
            "DRY_RUN": "true",
            "PYTHON_BIN": "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python",
        }
    )
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    assert "train_mrec_learned_marginal_structure_only.py" in output
    assert "--map-ablation-mode full" in output
    assert "rawfc_abc_v0_1_baseline20/04_evidence_map" in output
    assert "mrec_greedy_transition_v0_2_learned_marginal_structure_only_fullpool" in output
    assert "DATASET=rawfc LABEL_SCHEMA=rawfc3" in output
    assert "SFT_LEARNING_RATE=1e-5" in output
    assert "SFT_EVAL_STEPS=50" in output
    assert "LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.1" in output
    assert "-m sft.hami_cuda_bootstrap" in output
    assert "_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc" in output
    assert "train_mrec_learned_marginal_proxy.py" not in output
