from __future__ import annotations

import argparse

import pytest

from scripts.phase5_selectors.build import build_mrec_reward_cache as module


def test_init_scorer_auto_falls_back_to_transformers_when_vllm_rejects_architecture(monkeypatch, capsys) -> None:
    sentinel = object()

    def reject_vllm(args, checkpoint):
        raise ValueError("Model architectures ['Mistral3ForConditionalGeneration'] failed to be inspected.")

    def init_transformers(args, checkpoint):
        assert args.scoring_backend == "auto"
        assert checkpoint["checkpoint_dir"] == "/tmp/teacher/best"
        return sentinel

    monkeypatch.setattr(module, "_init_vllm_scorer", reject_vllm)
    monkeypatch.setattr(module, "_init_transformers_scorer", init_transformers)

    scorer = module._init_scorer(
        argparse.Namespace(scoring_backend="auto"),
        {"checkpoint_dir": "/tmp/teacher/best"},
    )

    captured = capsys.readouterr()
    assert scorer is sentinel
    assert "falling back to transformers" in captured.err


def test_init_scorer_vllm_backend_does_not_fallback(monkeypatch) -> None:
    def reject_vllm(args, checkpoint):
        raise ValueError("vllm unavailable")

    def init_transformers(args, checkpoint):  # pragma: no cover - should not be called.
        raise AssertionError("explicit vllm backend should not fallback")

    monkeypatch.setattr(module, "_init_vllm_scorer", reject_vllm)
    monkeypatch.setattr(module, "_init_transformers_scorer", init_transformers)

    with pytest.raises(ValueError, match="vllm unavailable"):
        module._init_scorer(argparse.Namespace(scoring_backend="vllm"), {})
