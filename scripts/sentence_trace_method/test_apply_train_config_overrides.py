from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = Path(__file__).with_name("apply_train_config_overrides.py")
ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    pythonpath = f"{ROOT}:{ROOT / 'src'}"
    if os.environ.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    return {**os.environ, "PYTHONPATH": pythonpath}


def test_apply_train_config_overrides_sets_fullft_alignment_fields(tmp_path: Path) -> None:
    config = tmp_path / "train.resolved.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sft_train": {
                    "gradient_accumulation_steps": 8,
                    "learning_rate": 2e-6,
                    "num_train_epochs": 5,
                    "eval_steps": 50,
                    "save_steps": 50,
                    "early_stopping_patience": 6,
                    "label_token_ce": {
                        "class_weights": {
                            "pants-fire": 1.0,
                            "true": 1.0,
                        },
                    },
                    "lora": {"enabled": False},
                },
                "train": {"deepspeed_config": "configs/deepspeed_zero3_bsz1_ga8_lowpeak.json"},
                "swanlab": {"project": "old", "experiment_name": "old"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--deepspeed-config",
            "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json",
            "--gradient-accumulation-steps",
            "4",
            "--learning-rate",
            "1e-5",
            "--num-train-epochs",
            "12",
            "--eval-steps",
            "100",
            "--save-steps",
            "100",
            "--early-stopping-patience",
            "8",
            "--swanlab-project",
            "fact-checking-sentence-trace-method-rawfc-selector-fullft-lr",
            "--swanlab-experiment-name",
            "rawfc__ministral3_fullft_aligned",
            "--class-weight",
            "pants-fire=1.2",
            "--class-weight",
            "true=1.8",
        ],
        env=_env(),
        check=True,
    )

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert cfg["train"]["deepspeed_config"] == "configs/deepspeed_zero3_bsz1_ga4_lowpeak.json"
    assert cfg["sft_train"]["gradient_accumulation_steps"] == 4
    assert cfg["sft_train"]["learning_rate"] == 1e-5
    assert cfg["sft_train"]["num_train_epochs"] == 12
    assert cfg["sft_train"]["eval_steps"] == 100
    assert cfg["sft_train"]["save_steps"] == 100
    assert cfg["sft_train"]["early_stopping_patience"] == 8
    assert cfg["sft_train"]["label_token_ce"]["class_weights"]["pants-fire"] == 1.2
    assert cfg["sft_train"]["label_token_ce"]["class_weights"]["true"] == 1.8
    assert cfg["sft_train"]["lora"]["enabled"] is False
    assert cfg["swanlab"]["project"] == "fact-checking-sentence-trace-method-rawfc-selector-fullft-lr"
    assert cfg["swanlab"]["experiment_name"] == "rawfc__ministral3_fullft_aligned"
