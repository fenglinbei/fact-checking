from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fact_checking.selectors.verifier_proxy import (
    build_anchor2_delta_rows,
    load_label_token_ids,
    load_score_cache,
    require_verifier_checkpoint,
    score_margin,
    verifier_proxy_cross_encoder_loss,
)


class VerifierProxyTest(unittest.TestCase):
    def test_checkpoint_check_requires_adapter_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            ckpt = run_dir / "best"
            ckpt.mkdir()
            (ckpt / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/data/models/Qwen2.5-7B-Instruct"}),
                encoding="utf-8",
            )
            (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            self._write_label_meta(run_dir)
            with self.assertRaisesRegex(FileNotFoundError, "adapter_model.safetensors"):
                require_verifier_checkpoint(run_dir, "best")

    def test_checkpoint_check_rejects_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "final"):
                require_verifier_checkpoint(tmp_dir, "final")

    def test_label_token_ids_load_from_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            self._write_label_meta(run_dir)
            self.assertEqual(load_label_token_ids(run_dir)["A"], 362)
            self.assertEqual(load_label_token_ids(run_dir)["F"], 434)

    def test_margin_uses_gold_minus_best_wrong(self) -> None:
        result = score_margin({"A": -3.0, "B": -2.0, "C": -1.0, "D": -4.0, "E": -5.0, "F": -6.0}, "false")
        self.assertAlmostEqual(result["margin"], -1.0)
        self.assertEqual(result["pred_label"], "barely-true")
        self.assertFalse(result["is_correct"])

    def test_anchor2_delta_labels_anchor_and_add_one(self) -> None:
        union = {
            "event_id": "e0",
            "claim": "claim",
            "candidates": [
                {"text": "b1", "canonical_text": "b1", "from_baseline": True, "baseline_rank": 1, "union_pool_rank": 1},
                {"text": "b2", "canonical_text": "b2", "from_baseline": True, "baseline_rank": 2, "union_pool_rank": 2},
                {"text": "q1", "canonical_text": "q1", "from_qd": True, "qd_pool_rank": 1, "union_pool_rank": 3},
            ],
        }
        margins = {("b1", "b2"): 1.0, ("b2",): 0.6, ("b1",): 1.2, ("b1", "b2", "q1"): 1.5}
        rows, _raw = build_anchor2_delta_rows(
            split="train",
            union_row=union,
            oracle_row={"event_id": "e0", "gold_label": "true"},
            score_fn=self._fake_score_fn(margins),
        )
        by_key = {row["candidate_key"]: row for row in rows}
        self.assertAlmostEqual(by_key["b1"]["target_utility"], 0.4)
        self.assertAlmostEqual(by_key["b2"]["target_utility"], -0.2)
        self.assertAlmostEqual(by_key["q1"]["target_utility"], 0.5)
        self.assertEqual(by_key["b1"]["evidence_set_policy"], "anchor_leave_one_out")
        self.assertEqual(by_key["q1"]["evidence_set_policy"], "anchor_add_one")
        self.assertTrue(by_key["q1"]["target_positive"])

    def test_all_negative_group_still_has_positive(self) -> None:
        union = {
            "event_id": "e0",
            "claim": "claim",
            "candidates": [
                {"text": "b1", "canonical_text": "b1", "from_baseline": True, "baseline_rank": 1},
                {"text": "q1", "canonical_text": "q1", "from_qd": True, "qd_pool_rank": 1},
            ],
        }
        rows, _raw = build_anchor2_delta_rows(
            split="train",
            union_row=union,
            oracle_row={"event_id": "e0", "gold_label": "true"},
            score_fn=self._fake_score_fn({("b1",): 1.0, tuple(): 1.2, ("b1", "q1"): 0.8}),
        )
        self.assertTrue(any(row["target_positive"] for row in rows))

    def test_cache_uses_last_valid_duplicate_and_counts_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cache.jsonl"
            path.write_text(
                '{"cache_key":"k","status":"completed","margin":1}\n'
                'not-json\n'
                '{"cache_key":"k","status":"completed","margin":2}\n',
                encoding="utf-8",
            )
            cache, invalid, duplicates = load_score_cache(path)
            self.assertEqual(cache["k"]["margin"], 2)
            self.assertEqual(invalid, 1)
            self.assertEqual(duplicates, 1)

    def test_pairwise_loss_decreases_when_positive_score_is_raised(self) -> None:
        low_scores = [torch.tensor([0.0, 0.0])]
        high_scores = [torch.tensor([2.0, -1.0])]
        utilities = [[1.0, 0.0]]
        positives = [[True, False]]
        low_loss, _ = verifier_proxy_cross_encoder_loss(low_scores, utilities, positives)
        high_loss, _ = verifier_proxy_cross_encoder_loss(high_scores, utilities, positives)
        self.assertLess(float(high_loss), float(low_loss))

    def test_baseline_top2_plus_learned_keeps_baseline_top2(self) -> None:
        module = _load_train_script_module()
        groups = [
            {
                "event_id": "e0",
                "claim": "claim",
                "candidates": [
                    {"candidate_key": "b1", "candidate_text": "b1", "from_baseline": True, "baseline_rank": 1},
                    {"candidate_key": "b2", "candidate_text": "b2", "from_baseline": True, "baseline_rank": 2},
                    {"candidate_key": "q1", "candidate_text": "q1", "from_qd": True, "qd_pool_rank": 1},
                ],
            }
        ]
        selected = module.build_selected_rows(groups, [torch.tensor([-10.0, -9.0, 10.0]).numpy()], top_k=3, mode="anchor", baseline_anchor_k=2)
        texts = [row["text"] for row in selected[0]["candidates"]]
        self.assertEqual(texts, ["b1", "b2", "q1"])

    @staticmethod
    def _fake_score_fn(margins: dict[tuple[str, ...], float]):
        def score(candidates):
            keys = tuple(str(candidate.get("canonical_text") or candidate.get("text") or "") for candidate in candidates)
            margin = float(margins[keys])
            return {
                "label_logprobs": {"A": -5.0, "B": -4.0, "C": -3.0, "D": -2.0, "E": -1.0, "F": margin},
                "pred_label": "true",
                "is_correct": True,
                "gold_logprob": margin,
                "best_wrong_logprob": 0.0,
                "margin": margin,
            }
        return score

    @staticmethod
    def _write_label_meta(run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "label_token_ce_meta.json").write_text(
            json.dumps(
                {
                    "label_prefix": "Label:",
                    "label_token_ids": {"A": 362, "B": 425, "C": 356, "D": 422, "E": 468, "F": 434},
                }
            ),
            encoding="utf-8",
        )


def _load_train_script_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "selectors" / "train_verifier_proxy_cross_encoder.py"
    spec = importlib.util.spec_from_file_location("train_verifier_proxy_cross_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
