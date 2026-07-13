from __future__ import annotations

import json

import pytest

from sft.data.io import load_prebuilt_samples
from sft.label_token_dataset import LabelTokenDataset
from sft.label_token_infer import _save_prediction_only_artifacts


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> dict[str, list[int]]:
        del text, add_special_tokens, truncation
        return {"input_ids": [7]}


def _unlabeled_row() -> dict:
    return {
        "prompt": "Rendered prompt",
        "target": "",
        "prompt_input_ids": [11, 12],
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "gold_id": -1,
        "gold_label": "",
        "gold_explain": "",
        "label_schema": "scifact3",
        "prompt_token_count": 2,
        "target_token_count": 0,
        "evidence_count": 1,
        "claim": "Claim",
    }


def test_unlabeled_samples_require_explicit_loader_and_dataset_opt_in() -> None:
    row = _unlabeled_row()

    assert load_prebuilt_samples([row]) == []
    samples = load_prebuilt_samples([row], include_unlabeled=True)
    assert len(samples) == 1
    assert samples[0].gold_id == -1

    with pytest.raises(ValueError, match="Invalid gold label"):
        LabelTokenDataset(
            samples,
            _FakeTokenizer(),
            max_length=16,
            label_prefix="Label:",
            label_schema="scifact3",
        )

    dataset = LabelTokenDataset(
        samples,
        _FakeTokenizer(),
        max_length=16,
        label_prefix="Label:",
        label_schema="scifact3",
        allow_unlabeled=True,
    )
    assert len(dataset) == 1
    assert dataset[0]["gold_id"] == -1
    assert dataset[0]["input_ids"] == [11, 12, 7]


def test_prediction_only_artifacts_do_not_create_gold_metrics(tmp_path) -> None:
    records = [
        {
            "sample_idx": 0,
            "pred_id": 2,
            "pred_label": "nei",
            "gold_id": -1,
            "gold_label": "",
        }
    ]

    artifacts = _save_prediction_only_artifacts(
        output_dir=tmp_path,
        split="test",
        checkpoint="best",
        label_schema="scifact3",
        prediction_records=records,
        logit_adjust_cfg=None,
    )

    manifest = json.loads((tmp_path / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prediction_only"] is True
    assert manifest["num_samples"] == 1
    assert manifest["num_labeled_samples"] == 0
    assert (tmp_path / "test_predictions.jsonl").exists()
    assert not (tmp_path / "metrics.json").exists()
    assert artifacts["predictions_path"].endswith("test_predictions.jsonl")
