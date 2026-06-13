from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = Path(__file__).with_name("prepare_lora_config.py")
ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    import os

    pythonpath = f"{ROOT}:{ROOT / 'src'}"
    if os.environ.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    return {**os.environ, "PYTHONPATH": pythonpath}


def test_prepare_lora_config_applies_learning_rate_override(tmp_path: Path) -> None:
    build_dir = tmp_path / "source" / "build"
    build_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (build_dir / f"build_{split}.jsonl").write_text(json.dumps({"id": split}) + "\n", encoding="utf-8")

    source_config = tmp_path / "source" / "train.resolved.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_candidates": str(build_dir / "build_train.jsonl"),
                    "val_candidates": str(build_dir / "build_val.jsonl"),
                    "test_candidates": str(build_dir / "build_test.jsonl"),
                },
                "sft_train": {
                    "learning_rate": 2.0e-6,
                    "gradient_accumulation_steps": 8,
                    "lora": {"enabled": False},
                },
                "train": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "case_lora"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-config",
            str(source_config),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "case_lora",
            "--learning-rate",
            "1e-5",
        ],
        env=_env(),
        check=True,
    )

    cfg = yaml.safe_load((output_root / "train.resolved.yaml").read_text(encoding="utf-8"))
    assert cfg["sft_train"]["learning_rate"] == 1.0e-5


def test_prepare_lora_config_syncs_learning_rate_for_existing_config(tmp_path: Path) -> None:
    output_root = tmp_path / "case_lora"
    output_root.mkdir()
    output_config = output_root / "train.resolved.yaml"
    output_config.write_text(
        yaml.safe_dump(
            {
                "sft_train": {
                    "learning_rate": 2.0e-6,
                    "lora": {"enabled": True},
                },
                "train": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-config",
            str(output_config),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "case_lora",
            "--learning-rate",
            "5e-6",
        ],
        env=_env(),
        check=True,
    )

    cfg = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert cfg["sft_train"]["learning_rate"] == 5.0e-6
