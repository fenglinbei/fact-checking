from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sentence_trace_method.summarize_atom_union_s4_chunking_ablation import (
    compute_three_class_metrics,
    summarize_case,
)


def test_three_class_metrics_collapse_liar_labels() -> None:
    rows = [
        {"gold_label": "pants-fire", "pred_label": "false"},
        {"gold_label": "barely-true", "pred_label": "half-true"},
        {"gold_label": "half-true", "pred_label": "half-true"},
        {"gold_label": "mostly-true", "pred_label": "true"},
        {"gold_label": "true", "pred_label": "false"},
    ]

    metrics = compute_three_class_metrics(rows)

    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert set(metrics["per_class"]) == {"False", "Half-True", "True"}
    assert metrics["per_class"]["False"]["recall"] == pytest.approx(1 / 2)
    assert metrics["per_class"]["Half-True"]["precision"] == pytest.approx(1 / 2)
    assert metrics["per_class"]["True"]["recall"] == pytest.approx(1 / 2)


def test_summarize_case_reads_nested_label_token_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    build_root = output_root / "liar_raw__ministral3_8b__chunk_abc_s4_union_top5_plain"
    eval_root = output_root / "liar_raw__ministral3_8b__chunk_abc_s4_union_top5_plain_lora"
    (build_root / "build").mkdir(parents=True)
    (eval_root / "eval" / "test" / "best" / "label_token").mkdir(parents=True)

    (build_root / "build" / "build_report.json").write_text(
        json.dumps(
            {
                "splits": {
                    "test": {
                        "prompt_token_count": {"mean": 100.0, "p95": 150.0},
                        "evidence_count": {"mean": 5.0},
                        "prompt_truncation_rate": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (eval_root / "eval" / "test" / "best" / "label_token" / "metrics.json").write_text(
        json.dumps({"accuracy": 0.7, "macro_f1": 0.6}),
        encoding="utf-8",
    )
    (eval_root / "eval" / "test" / "best" / "label_token" / "test_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"gold_label": "pants-fire", "pred_label": "false"}),
                json.dumps({"gold_label": "true", "pred_label": "true"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = summarize_case(
        output_root=output_root,
        chunking_case="abc",
        policy="top5",
        split="test",
        checkpoint="best",
        lora_suffix="_lora",
    )

    assert row["case_name"] == "liar_raw__ministral3_8b__chunk_abc_s4_union_top5_plain"
    assert row["metrics_path"].endswith("eval/test/best/label_token/metrics.json")
    assert row["accuracy_6class"] == pytest.approx(0.7)
    assert row["macro_f1_6class"] == pytest.approx(0.6)
    assert row["accuracy_3class"] == pytest.approx(1.0)
    assert row["prompt_token_mean"] == pytest.approx(100.0)
