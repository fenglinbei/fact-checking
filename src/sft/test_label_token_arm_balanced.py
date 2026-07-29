from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sft.checkpoint_selection import (
    arm_balanced_metrics_from_records,
    checkpoint_candidate_is_better,
    checkpoint_selection_score,
    select_macro_f1_checkpoint,
)
from sft.data.io import load_prebuilt_samples
from sft.label_token_dataset import LabelTokenCollator, LabelTokenDataset
from sft.label_token_trainer import _evaluate_label_token


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if text == "Label:":
            return {"input_ids": [90]}
        if text == "Coverage:":
            return {"input_ids": [91]}
        return {"input_ids": [11, 12]}


class _FakeAccelerator:
    is_local_main_process = True
    is_main_process = True

    @staticmethod
    def pad_across_processes(tensor: torch.Tensor, **_: object) -> torch.Tensor:
        return tensor

    @staticmethod
    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def gather_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _FakeModel:
    def __init__(self, score_rows: list[list[float]]) -> None:
        self.score_rows = torch.tensor(score_rows, dtype=torch.float32)

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        _ = attention_mask, use_cache
        sample_indices = input_ids[:, 0].to(torch.long)
        scores = self.score_rows.index_select(0, sample_indices)
        return SimpleNamespace(logits=scores.unsqueeze(1))


class _SingleBatchDataloader(list):
    def __init__(self, batch: dict[str, torch.Tensor], samples: list[object]) -> None:
        super().__init__([batch])
        self.dataset = SimpleNamespace(samples=samples)


def _paired_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_idx, (event_id, evidence_arm, gold_id) in enumerate(
        (
            ("event-1", "EviTrace", 0),
            ("event-1", "control", 0),
            ("event-2", "evi", 1),
            ("event-2", "S4", 1),
        )
    ):
        rows.append(
            {
                "prompt": f"prompt-{sample_idx}",
                "prompt_input_ids": [sample_idx],
                "target": f"Label: {'A' if gold_id == 0 else 'B'}",
                "gold_id": gold_id,
                "gold_label": "false" if gold_id == 0 else "true",
                "gold_explain": "",
                "label_schema": "liar6",
                "event_id": event_id,
                "evidence_arm": evidence_arm,
                "assignment_id": "paired_val",
            }
        )
    return rows


def test_arm_metadata_round_trips_through_loader_dataset_and_collator() -> None:
    sample = load_prebuilt_samples(_paired_rows()[:1])[0]

    assert sample.event_id == "event-1"
    assert sample.evidence_arm == "evitrace"
    assert sample.assignment_id == "paired_val"

    tokenizer = _TinyTokenizer()
    dataset = LabelTokenDataset([sample], tokenizer, max_length=32, label_prefix="Label:")
    batch = LabelTokenCollator(tokenizer)([dataset[0]])

    assert dataset[0]["event_id"] == "event-1"
    assert dataset[0]["evidence_arm"] == "evitrace"
    assert dataset[0]["assignment_id"] == "paired_val"
    assert batch["metadata"][0]["event_id"] == "event-1"
    assert batch["metadata"][0]["evidence_arm"] == "evitrace"
    assert batch["metadata"][0]["assignment_id"] == "paired_val"


def test_arm_balanced_metrics_use_canonical_arm_names_and_mean_ce() -> None:
    records = [
        {
            "event_id": "event-1",
            "evidence_arm": "EviTrace",
            "assignment_id": "paired_val",
            "pred_id": 0,
            "gold_id": 0,
            "ce_loss": 0.2,
        },
        {
            "event_id": "event-1",
            "evidence_arm": "control",
            "assignment_id": "paired_val",
            "pred_id": 1,
            "gold_id": 0,
            "ce_loss": 0.8,
        },
        {
            "event_id": "event-2",
            "evidence_arm": "evi",
            "assignment_id": "paired_val",
            "pred_id": 1,
            "gold_id": 1,
            "ce_loss": 0.4,
        },
        {
            "event_id": "event-2",
            "evidence_arm": "S4",
            "assignment_id": "paired_val",
            "pred_id": 1,
            "gold_id": 1,
            "ce_loss": 0.6,
        },
    ]

    metrics = arm_balanced_metrics_from_records(records, labels=["false", "true"])

    assert metrics["arm_balanced_valid"] is True
    assert metrics["arm_balanced_num_events"] == 2
    assert metrics["macro_f1_evitrace"] == pytest.approx(1.0)
    assert metrics["macro_f1_s4"] == pytest.approx(1.0 / 3.0)
    assert metrics["arm_balanced_macro_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["mean_ce_evitrace"] == pytest.approx(0.3)
    assert metrics["mean_ce_s4"] == pytest.approx(0.7)
    assert metrics["arm_balanced_mean_ce"] == pytest.approx(0.5)
    assert checkpoint_selection_score({**metrics, "macro_f1": 0.9}, {}) == pytest.approx(2.0 / 3.0)


def test_invalid_pair_metadata_falls_back_to_full_macro_f1() -> None:
    invalid_metrics = arm_balanced_metrics_from_records(
        [
            {
                "event_id": "event-1",
                "evidence_arm": "evitrace",
                "assignment_id": "paired_val",
                "pred_id": 0,
                "gold_id": 0,
                "ce_loss": 0.2,
            }
        ],
        labels=["false", "true"],
    )

    assert invalid_metrics["arm_balanced_valid"] is False
    assert checkpoint_selection_score({**invalid_metrics, "macro_f1": 0.75}, {}) == 0.75


def test_checkpoint_ties_prefer_lower_mean_ce_then_earlier_step() -> None:
    assert checkpoint_candidate_is_better(
        score=0.6,
        mean_ce=0.4,
        step=200,
        best_score=0.6,
        best_mean_ce=0.5,
        best_step=100,
    )
    assert checkpoint_candidate_is_better(
        score=0.6,
        mean_ce=0.5,
        step=100,
        best_score=0.6,
        best_mean_ce=0.5,
        best_step=200,
    )
    assert not checkpoint_candidate_is_better(
        score=0.6,
        mean_ce=0.5,
        step=300,
        best_score=0.6,
        best_mean_ce=0.5,
        best_step=200,
    )

    selected = select_macro_f1_checkpoint(
        [
            {"checkpoint": "checkpoint-100", "step": 100, "macro_f1": 0.6, "eval_ce_loss": 0.5},
            {"checkpoint": "checkpoint-200", "step": 200, "macro_f1": 0.6, "eval_ce_loss": 0.4},
            {"checkpoint": "checkpoint-300", "step": 300, "macro_f1": 0.6, "eval_ce_loss": 0.4},
        ]
    )
    assert selected["checkpoint"] == "checkpoint-200"


def test_evaluate_label_token_exports_arm_records_and_balanced_metrics() -> None:
    samples = load_prebuilt_samples(_paired_rows())
    batch = {
        "input_ids": torch.tensor([[0], [1], [2], [3]], dtype=torch.long),
        "attention_mask": torch.ones((4, 1), dtype=torch.long),
        "gold_ids": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "sample_indices": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    dataloader = _SingleBatchDataloader(batch, samples)
    model = _FakeModel(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
            [0.0, 4.0],
        ]
    )

    metrics = _evaluate_label_token(
        model=model,
        dataloader=dataloader,
        accelerator=_FakeAccelerator(),
        label_token_ids=torch.tensor([0, 1], dtype=torch.long),
        class_weights=torch.ones(2, dtype=torch.float32),
        train_cfg={"label_token_ce": {"ordinal_loss": {"enabled": False}}},
        label_prefix="Label:",
        labels=["false", "true"],
        letter_order=["A", "B"],
        eval_logger=None,
        log_predictions_limit=0,
    )

    assert metrics["arm_balanced_valid"] is True
    assert metrics["macro_f1_evitrace"] == pytest.approx(1.0)
    assert metrics["macro_f1_s4"] == pytest.approx(1.0 / 3.0)
    records = metrics["prediction_records"]
    assert [record["evidence_arm"] for record in records] == ["evitrace", "s4", "evitrace", "s4"]
    assert {record["assignment_id"] for record in records} == {"paired_val"}
    assert all(float(record["ce_loss"]) > 0.0 for record in records)
