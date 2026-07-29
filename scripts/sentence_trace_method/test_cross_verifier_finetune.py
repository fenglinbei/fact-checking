from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).with_name("cross_verifier_finetune.py")
SPEC = importlib.util.spec_from_file_location("cross_verifier_finetune_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **_kwargs):
        if text == "Label:":
            return {"input_ids": [10, 11]}
        if len(text) == 2 and text.startswith(" ") and text[1] in MODULE.LETTERS:
            return {"input_ids": [20 + MODULE.LETTERS.index(text[1])]}
        raise AssertionError(text)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_config_is_frozen_and_arm_balanced(tmp_path: Path) -> None:
    config = MODULE.build_train_config(
        model_name="qwen3",
        model_path=tmp_path / "model",
        train_path=tmp_path / "assignment_a.jsonl",
        val_path=tmp_path / "paired_val.jsonl",
        run_root=tmp_path / "run",
        seed=20260724,
        smoke=False,
    )
    train = config["sft_train"]
    assert train["learning_rate"] == pytest.approx(2e-5)
    assert train["num_train_epochs"] == pytest.approx(12)
    assert train["gradient_accumulation_steps"] == 4
    assert train["max_length"] == 2048
    assert train["label_token_ce"]["early_stopping_metric"] == "arm_balanced_macro_f1"
    assert train["label_token_ce"]["ordinal_loss"]["alpha"] == pytest.approx(0.2)
    assert train["lora"] == {
        "enabled": True,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "modules_to_save": None,
    }
    assert MODULE.EXPECTED_EVAL_LOGICAL_RESULTS == 23_892


def test_prompt_registry_deduplicates_only_final_token_ids() -> None:
    rows = [
        {
            "logical_id": "1::main::evitrace",
            "event_id": "1.json",
            "comparison_type": "main",
            "evidence_arm": "evitrace",
            "prompt_input_ids": [1, 2, 3],
        },
        {
            "logical_id": "1::order_only::evitrace",
            "event_id": "1.json",
            "comparison_type": "order_only",
            "evidence_arm": "evitrace",
            "prompt_input_ids": [1, 2, 3],
        },
        {
            "logical_id": "1::main::s4",
            "event_id": "1.json",
            "comparison_type": "main",
            "evidence_arm": "s4",
            "prompt_input_ids": [1, 3, 2],
        },
    ]
    logical, unique = MODULE._normalize_prompt_rows(rows, TinyTokenizer())
    assert len(logical) == 3
    assert len(unique) == 2
    assert logical[0]["input_ids_sha256"] == logical[1]["input_ids_sha256"]
    assert logical[0]["token_count"] == 5


def test_prompt_overflow_fails_without_truncation() -> None:
    rows = [
        {
            "logical_id": "too-long",
            "prompt_input_ids": list(range(MODULE.MAX_LENGTH)),
        }
    ]
    with pytest.raises(MODULE.FinetuneExperimentError, match="truncation is forbidden"):
        MODULE._normalize_prompt_rows(rows, TinyTokenizer())


def test_resume_cache_key_locks_base_adapter_and_prompt(tmp_path: Path) -> None:
    input_sha = MODULE.sha256_json([1, 2, 3])
    cache_key = MODULE.sha256_json(
        {
            "base_model_sha256": "base",
            "adapter_sha256": "adapter",
            "input_ids_sha256": input_sha,
        }
    )
    logit_path = tmp_path / "unique_logits.jsonl"
    row = {
        "cache_key": cache_key,
        "input_ids_sha256": input_sha,
        "logits": dict(zip(MODULE.LETTERS, range(6))),
    }
    logit_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    restored = MODULE._load_resume_scores(
        logit_path,
        base_sha="base",
        adapter_sha="adapter",
    )
    assert list(restored) == [input_sha]
    with pytest.raises(MODULE.FinetuneExperimentError, match="cache key mismatch"):
        MODULE._load_resume_scores(
            logit_path,
            base_sha="different",
            adapter_sha="adapter",
        )


def test_best_directory_alone_never_counts_as_complete(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    best = train_dir / "best"
    best.mkdir(parents=True)
    (best / "adapter_model.safetensors").write_bytes(b"adapter")
    assert not MODULE._training_complete(train_dir)
    (best / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert not MODULE._training_complete(train_dir)
    _write_json(train_dir / "training_complete.json", {"training_complete": True})
    assert MODULE._training_complete(train_dir)
    _write_json(train_dir / "training_complete.json", {"training_complete": False})
    assert not MODULE._training_complete(train_dir)


def test_logit_normalization_and_tie_are_deterministic() -> None:
    log_probs, probabilities = MODULE._normalize_logits([0, 0, 0, 0, 0, 0])
    assert set(log_probs) == set(MODULE.LETTERS)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(MODULE.LETTERS, key=lambda letter: log_probs[letter]) == "A"


def test_formal_parser_rejects_unregistered_seed() -> None:
    with pytest.raises(MODULE.FinetuneExperimentError, match="Seed must be"):
        MODULE.main(
            [
                "train",
                "--prepared-manifest",
                "missing.json",
                "--model-name",
                "qwen3",
                "--assignment",
                "a",
                "--seed",
                "7",
                "--dry-run",
            ]
        )


def test_model_facing_manifest_never_opens_gold(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text("{}\n", encoding="utf-8")
    gold = tmp_path / "gold_test.jsonl"
    gold.write_text('{"gold_label":"true"}\n', encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "complete": True,
            "experiment": "evitrace_cross_verifier_finetune_v1",
            "prepared_files": {
                "gold_test": {
                    "path": str(gold),
                    "sha256": MODULE.sha256_file(gold),
                    "bytes": gold.stat().st_size,
                }
            },
            "models": {
                "qwen3": {
                    "files": {
                        "eval_registry": {
                            "path": str(registry),
                            "sha256": MODULE.sha256_file(registry),
                            "bytes": registry.stat().st_size,
                        }
                    }
                }
            },
        },
    )
    original = MODULE.sha256_file

    def guarded(path):
        if Path(path).resolve() == gold.resolve():
            raise AssertionError("model-facing process opened gold")
        return original(path)

    monkeypatch.setattr(MODULE, "sha256_file", guarded)
    _path, _manifest, digest = MODULE.load_prepared_manifest(
        manifest_path,
        model_facing=True,
    )
    assert digest


def test_training_force_is_rejected_before_any_overwrite() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "train",
            "--prepared-manifest",
            "does-not-exist.json",
            "--model-name",
            "qwen3",
            "--assignment",
            "a",
            "--seed",
            "20260724",
            "--force",
        ]
    )
    with pytest.raises(MODULE.FinetuneExperimentError, match="intentionally unsupported"):
        MODULE.train_phase(args)


def test_only_one_truncated_jsonl_tail_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_bytes(b'{"ok":1}\n{"interrupted":')
    assert MODULE.repair_truncated_jsonl_tail(path)
    assert path.read_bytes() == b'{"ok":1}\n'
    assert not MODULE.repair_truncated_jsonl_tail(path)


def test_adapter_fingerprint_covers_config_and_weights(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text('{"r":16}', encoding="utf-8")
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    first, selected = MODULE._adapter_fingerprint(tmp_path)
    assert selected == weights
    (tmp_path / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
    second, _ = MODULE._adapter_fingerprint(tmp_path)
    assert first != second


def test_tokenizer_fingerprint_matches_prepare_and_excludes_weights(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    nested = tmp_path / "remote_code"
    nested.mkdir()
    (nested / "tokenizer.py").write_text("VERSION = 1\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights-v1")

    first = MODULE._tokenizer_fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights-v2")
    assert MODULE._tokenizer_fingerprint(tmp_path) == first
    (nested / "tokenizer.py").write_text("VERSION = 2\n", encoding="utf-8")
    assert MODULE._tokenizer_fingerprint(tmp_path) != first
