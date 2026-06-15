from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "phase9_utils" / "collect_result_index.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_result_index", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CollectResultIndexTest(unittest.TestCase):
    def test_collects_metric_scope_status_and_writes_index_files(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            final_dir = (
                workspace
                / "outputs"
                / "sentence_trace_method"
                / "run_a"
                / "eval"
                / "val"
                / "best"
                / "label_token"
            )
            final_dir.mkdir(parents=True)
            (final_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "num_samples": 2,
                        "accuracy": 0.5,
                        "macro_f1": 0.4,
                        "selection_score": 0.7,
                        "parse_error_rate": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            complete_dir = workspace / "outputs" / "sentence_trace_method" / "run_a" / "train"
            complete_dir.mkdir(parents=True)
            (complete_dir / "training_complete.json").write_text('{"global_step": 10}\n', encoding="utf-8")
            (workspace / "outputs" / "sentence_trace_method" / "run_a" / "train.resolved.yaml").write_text(
                "train:\n  seed: 42\n",
                encoding="utf-8",
            )

            step_dir = workspace / "outputs" / "sentence_trace_method" / "run_b" / "eval" / "step-100"
            step_dir.mkdir(parents=True)
            (step_dir / "metrics.json").write_text('{"accuracy": 0.25, "macro_f1": 0.3}\n', encoding="utf-8")
            latest_state = workspace / "outputs" / "sentence_trace_method" / "run_b" / "train" / "latest_state"
            latest_state.mkdir(parents=True)
            (latest_state / "trainer_state.json").write_text('{"completed": false}\n', encoding="utf-8")

            selector_dir = workspace / "outputs" / "selectors" / "selector_a"
            selector_dir.mkdir(parents=True)
            (selector_dir / "selection_metrics.json").write_text('{"recall@5": 0.8}\n', encoding="utf-8")

            cache_dir = workspace / "outputs" / "cache" / "build" / "abc"
            cache_dir.mkdir(parents=True)
            (cache_dir / "manifest.json").write_text('{"status": "cached"}\n', encoding="utf-8")

            rows = module.collect_inventory(workspace / "outputs")
            self.assertFalse(any(row["source_root"] == "cache" for row in rows))

            final_row = next(row for row in rows if row["relative_path"].endswith("label_token/metrics.json"))
            self.assertEqual(final_row["source_root"], "sentence_trace_method")
            self.assertEqual(final_row["artifact_kind"], "metrics_json")
            self.assertEqual(final_row["metric_scope"], "final_named")
            self.assertEqual(final_row["split"], "val")
            self.assertEqual(final_row["checkpoint"], "best")
            self.assertEqual(final_row["eval_kind"], "label_token")
            self.assertEqual(final_row["training_status"], "complete")
            self.assertEqual(final_row["num_samples"], 2)
            self.assertEqual(final_row["accuracy"], 0.5)
            self.assertEqual(final_row["selection_score"], 0.7)

            step_row = next(row for row in rows if row["relative_path"].endswith("step-100/metrics.json"))
            self.assertEqual(step_row["metric_scope"], "step_curve")
            self.assertEqual(step_row["step"], "step-100")
            self.assertEqual(step_row["training_status"], "resume_state_present")

            selector_row = next(row for row in rows if row["relative_path"].endswith("selection_metrics.json"))
            self.assertEqual(selector_row["source_root"], "selectors")
            self.assertEqual(selector_row["artifact_kind"], "metrics_json")
            self.assertEqual(selector_row["metric_scope"], "selector_or_summary")

            paths = module.write_index(rows, workspace / "docs" / "Z-cross-cutting", stamp="20260615")
            with paths.csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            jsonl_rows = [json.loads(line) for line in paths.jsonl_path.read_text(encoding="utf-8").splitlines()]
            summary = paths.md_path.read_text(encoding="utf-8")

            self.assertEqual(len(csv_rows), len(rows))
            self.assertEqual(len(jsonl_rows), len(rows))
            self.assertIn("20260615 Outputs Metric Inventory", summary)
            self.assertIn("sentence_trace_method", summary)


if __name__ == "__main__":
    unittest.main()
