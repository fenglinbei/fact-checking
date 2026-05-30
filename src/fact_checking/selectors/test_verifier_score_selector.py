from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fact_checking.data.constants import LETTER2LABEL, LETTER_ORDER
from fact_checking.selectors.stage2_oracle import Stage2OracleExample
from fact_checking.selectors.verifier_score_selector import (
    EventCheckpointStore,
    GREEDY_STEPWISE_TOP5,
    STATIC_TOP5,
    load_raw_score_cache,
    run_chunked_selector,
)


class VerifierScoreSelectorTest(unittest.TestCase):
    def test_raw_cache_uses_last_completed_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"status": "completed", "cache_key": "a", "value": 1}),
                        "{bad json",
                        json.dumps({"status": "running", "cache_key": "b", "value": 1}),
                        json.dumps({"status": "completed", "cache_key": "a", "value": 2}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cache, invalid, duplicates = load_raw_score_cache(path)
            self.assertEqual(invalid, 2)
            self.assertEqual(duplicates, 1)
            self.assertEqual(cache["a"]["value"], 2)

    def test_event_checkpoint_ignores_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EventCheckpointStore(Path(tmp) / "events")
            payload = {"status": "completed", "run_fingerprint": "fp", "event_id": "evt"}
            store.write_event("evt", payload)
            (Path(tmp) / "events" / "bad.json.tmp").write_text(
                json.dumps({"status": "completed", "run_fingerprint": "fp", "event_id": "bad"}),
                encoding="utf-8",
            )
            completed = store.load_completed("fp")
            self.assertEqual(sorted(completed), ["evt"])

    def test_static_chunked_output_matches_larger_claim_batch(self) -> None:
        examples = [_example("e1"), _example("e2")]
        values = _candidate_values(examples, preferred=[2, 1, 3, 0])
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "batch1"
            out2 = Path(tmp) / "batch4"
            _run(
                examples,
                out1,
                scorer=FakeScorer(values),
                selection_modes=STATIC_TOP5,
                score_modes="pred_margin",
                claim_batch_size=1,
            )
            _run(
                examples,
                out2,
                scorer=FakeScorer(values),
                selection_modes=STATIC_TOP5,
                score_modes="pred_margin",
                claim_batch_size=4,
            )
            self.assertEqual(
                _orders(out1 / "verifier_score_static_top5_pred_margin" / "selection_trace.jsonl"),
                _orders(out2 / "verifier_score_static_top5_pred_margin" / "selection_trace.jsonl"),
            )

    def test_greedy_stepwise_claim_batch_size_is_deterministic(self) -> None:
        examples = [_example("e1"), _example("e2")]
        values = _greedy_values(examples, order=[1, 3, 0])
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "batch1"
            out2 = Path(tmp) / "batch4"
            _run(
                examples,
                out1,
                scorer=FakeScorer(values),
                selection_modes=GREEDY_STEPWISE_TOP5,
                score_modes="base_pred_margin",
                claim_batch_size=1,
            )
            _run(
                examples,
                out2,
                scorer=FakeScorer(values),
                selection_modes=GREEDY_STEPWISE_TOP5,
                score_modes="base_pred_margin",
                claim_batch_size=4,
            )
            self.assertEqual(
                _orders(out1 / "verifier_score_greedy_stepwise_top5_base_pred_margin" / "selection_trace.jsonl"),
                _orders(out2 / "verifier_score_greedy_stepwise_top5_base_pred_margin" / "selection_trace.jsonl"),
            )

    def test_resume_after_scorer_failure_matches_clean_run(self) -> None:
        examples = [_example("e1"), _example("e2")]
        values = _candidate_values(examples, preferred=[3, 0, 2, 1])
        with tempfile.TemporaryDirectory() as tmp:
            resumed = Path(tmp) / "resumed"
            clean = Path(tmp) / "clean"
            with self.assertRaises(RuntimeError):
                _run(
                    examples,
                    resumed,
                    scorer=FakeScorer(values, batch_size=2, fail_after_batches=1),
                    selection_modes=STATIC_TOP5,
                    score_modes="pred_margin",
                    claim_batch_size=2,
                )
            cache, _invalid, _duplicates = load_raw_score_cache(resumed / "_resume" / "raw_verifier_scores.jsonl")
            self.assertGreater(len(cache), 0)
            self.assertFalse(any((resumed / "_resume" / "events").glob("*.json")))

            _run(
                examples,
                resumed,
                scorer=FakeScorer(values, batch_size=2),
                selection_modes=STATIC_TOP5,
                score_modes="pred_margin",
                claim_batch_size=2,
            )
            _run(
                examples,
                clean,
                scorer=FakeScorer(values, batch_size=2),
                selection_modes=STATIC_TOP5,
                score_modes="pred_margin",
                claim_batch_size=2,
            )
            self.assertEqual(
                _orders(resumed / "verifier_score_static_top5_pred_margin" / "selection_trace.jsonl"),
                _orders(clean / "verifier_score_static_top5_pred_margin" / "selection_trace.jsonl"),
            )


class FakeScorer:
    def __init__(
        self,
        values: dict[tuple[str, tuple[int, ...]], float],
        *,
        batch_size: int = 1000,
        fail_after_batches: int | None = None,
    ) -> None:
        self.values = values
        self.batch_size = int(batch_size)
        self.fail_after_batches = fail_after_batches
        self.completed_batches = 0

    def score_batch(self, requests: list[Any], *, on_batch_complete: Any = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start : start + self.batch_size]
            scores = [self._score(request) for request in batch]
            if on_batch_complete is not None:
                on_batch_complete(batch, scores)
            out.extend(scores)
            self.completed_batches += 1
            if self.fail_after_batches is not None and self.completed_batches >= self.fail_after_batches:
                raise RuntimeError("injected scorer failure")
        return out

    def _score(self, request: Any) -> dict[str, Any]:
        evidence_indices = tuple(int(idx) for idx in request.metadata.get("evidence_indices", []))
        value = float(self.values.get((request.event_id, evidence_indices), 0.1))
        label_logprobs = {letter: -8.0 for letter in LETTER_ORDER}
        label_logprobs["A"] = value
        label_logprobs["B"] = 0.0
        return _score_from_logprobs(label_logprobs)


def _run(
    examples: list[Stage2OracleExample],
    output_dir: Path,
    *,
    scorer: FakeScorer,
    selection_modes: str,
    score_modes: str,
    claim_batch_size: int,
) -> dict[str, Any]:
    return run_chunked_selector(
        examples=examples,
        output_dir=output_dir,
        split="val",
        top_k=3,
        selection_modes=selection_modes,
        score_modes=score_modes,
        verifier_fingerprint="verifier-fp",
        prompt_fingerprint="prompt-fp",
        scorer=scorer,
        prompt_builder=_prompt_builder,
        claim_batch_size=claim_batch_size,
        resume=True,
        no_progress=True,
    )


def _prompt_builder(example: Stage2OracleExample, spec: Any) -> dict[str, Any]:
    return {
        "prompt": f"{example.event_id}:{','.join(str(idx) for idx in spec.evidence_indices)}",
        "target": "",
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "gold_id": 0,
        "gold_label": example.gold_label,
        "gold_explain": "",
        "prompt_token_count": 8 + len(spec.evidence_indices),
        "target_token_count": 1,
        "evidence_count": len(spec.evidence_indices),
        "was_truncated": False,
        "claim": example.claim,
    }


def _example(event_id: str) -> Stage2OracleExample:
    candidates = [
        {"text": f"{event_id} evidence {idx}", "candidate_uid": f"{event_id}:{idx}"}
        for idx in range(4)
    ]
    candidate_scores = [
        {"candidate_idx": idx, "hybrid_score": float(4 - idx), "candidate_uid": f"{event_id}:{idx}"}
        for idx in range(4)
    ]
    return Stage2OracleExample(
        event_id=event_id,
        claim=f"claim {event_id}",
        gold_label="pants-fire",
        candidates=candidates,
        candidate_scores=candidate_scores,
        selected_indices=[2, 1, 0],
        fingerprint="fp",
        margin=0.0,
        is_correct=True,
        raw={},
    )


def _candidate_values(
    examples: list[Stage2OracleExample],
    *,
    preferred: list[int],
) -> dict[tuple[str, tuple[int, ...]], float]:
    values: dict[tuple[str, tuple[int, ...]], float] = {}
    for example in examples:
        for rank, idx in enumerate(preferred):
            values[(example.event_id, (idx,))] = float(len(preferred) - rank)
    return values


def _greedy_values(
    examples: list[Stage2OracleExample],
    *,
    order: list[int],
) -> dict[tuple[str, tuple[int, ...]], float]:
    values: dict[tuple[str, tuple[int, ...]], float] = {}
    prefixes = [tuple(order[:step]) for step in range(len(order))]
    for example in examples:
        values[(example.event_id, tuple())] = 0.5
        for step, prefix in enumerate(prefixes):
            target = order[step]
            for idx in range(4):
                if idx in prefix:
                    continue
                evidence = (*prefix, idx)
                values[(example.event_id, evidence)] = 5.0 if idx == target else float(1.0 / (idx + 1))
    return values


def _orders(path: Path) -> list[list[int]]:
    out: list[list[int]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out.append([int(idx) for idx in row["selector_ordered_indices"]])
    return out


def _score_from_logprobs(label_logprobs: dict[str, float]) -> dict[str, Any]:
    ordered = sorted(label_logprobs.items(), key=lambda item: item[1], reverse=True)
    pred_letter, top1 = ordered[0]
    top2 = ordered[1][1]
    gold_logprob = float(label_logprobs["A"])
    best_wrong = max(float(value) for letter, value in label_logprobs.items() if letter != "A")
    return {
        "label_logprobs": dict(label_logprobs),
        "pred_letter": pred_letter,
        "pred_label": LETTER2LABEL[pred_letter],
        "top1_logprob": float(top1),
        "top2_logprob": float(top2),
        "pred_margin": float(top1 - top2),
        "entropy": 1.0 / max(float(top1 - top2), 0.001),
        "entropy_neg": -1.0 / max(float(top1 - top2), 0.001),
        "gold_logprob": gold_logprob,
        "best_wrong_logprob": float(best_wrong),
        "margin": float(gold_logprob - best_wrong),
        "is_correct": pred_letter == "A",
    }


if __name__ == "__main__":
    unittest.main()
