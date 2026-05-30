from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fact_checking.build.candidates import _build_training_row
from fact_checking.data.constants import LETTER_ORDER
from fact_checking.selectors.metrics import (
    build_order_control_trace,
    build_selection_trace,
    ordered_selection_metrics,
    ranked_indices_from_candidate_pool,
    ranked_indices_from_hybrid,
    summarize_ordered_selection,
)
from fact_checking.selectors.stage2_oracle import Stage2OracleExample
from fact_checking.selectors.verifier_proxy import (
    cache_key_for_score,
    candidate_key,
    evidence_set_hash,
    json_safe,
    stable_fingerprint,
)


STATIC_TOP5 = "static_top5"
GREEDY_STEPWISE_TOP5 = "greedy_stepwise_top5"
SELECTION_MODES = (STATIC_TOP5, GREEDY_STEPWISE_TOP5)
SCORE_MODES = (
    "pred_margin",
    "entropy_neg",
    "entropy_reduction",
    "base_pred_margin",
    "gold_margin",
)
DEPLOYABLE_SCORE_MODES = {
    "pred_margin",
    "entropy_neg",
    "entropy_reduction",
    "base_pred_margin",
}
RUN_VERSION = "verifier_score_selector_v0"

PromptBuilder = Callable[[Stage2OracleExample, "EvidenceSetSpec"], dict[str, Any]]


@dataclass(frozen=True)
class EvidenceSetSpec:
    event_id: str
    selection_mode: str
    role: str
    step: int
    candidate_idx: int | None
    evidence_indices: tuple[int, ...]
    scored_candidate_keys: tuple[str, ...]
    evidence_set_hash: str


@dataclass
class ScoreRequest:
    prompt_row: dict[str, Any]
    gold_label: str
    cache_key: str = ""
    event_id: str = ""
    claim: str = ""
    evidence_set_hash: str = ""
    scored_candidate_keys: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.scored_candidate_keys is None:
            self.scored_candidate_keys = []
        if self.metadata is None:
            self.metadata = {}


class ScoreCacheWriter:
    def __init__(self, path: str | Path, *, fsync_cache: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_cache = bool(fsync_cache)
        self._fh = self.path.open("a", encoding="utf-8")

    def append_many(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            self._fh.write(json.dumps(json_safe(dict(row)), ensure_ascii=False) + "\n")
        self._fh.flush()
        if self.fsync_cache:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ScoreCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class EventCheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def event_path(self, event_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(event_id))
        if not safe:
            safe = stable_fingerprint({"event_id": str(event_id)}, length=12)
        return self.path / f"{safe}.json"

    def load_completed(self, run_fingerprint: str) -> dict[str, dict[str, Any]]:
        return {
            str(payload.get("event_id") or ""): payload
            for payload in self.iter_events(run_fingerprint=run_fingerprint)
        }

    def iter_events(self, *, run_fingerprint: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.path.glob("*.json")):
            if path.name.endswith(".tmp"):
                continue
            try:
                with path.open(encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") != "completed":
                continue
            if str(payload.get("run_fingerprint") or "") != str(run_fingerprint):
                continue
            payloads.append(payload)
        return payloads

    def write_event(self, event_id: str, payload: dict[str, Any]) -> None:
        path = self.event_path(event_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)


def normalize_selection_modes(value: str | Sequence[str]) -> list[str]:
    raw = _split_values(value)
    if "both" in raw:
        raw = [STATIC_TOP5, GREEDY_STEPWISE_TOP5]
    out: list[str] = []
    for mode in raw:
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unknown selection mode {mode!r}; expected one of {SELECTION_MODES} or both.")
        if mode not in out:
            out.append(mode)
    return out


def normalize_score_modes(value: str | Sequence[str]) -> list[str]:
    out: list[str] = []
    for mode in _split_values(value):
        mode = "gold_margin" if mode == "gold_margin_diagnostic" else mode
        if mode not in SCORE_MODES:
            raise ValueError(f"Unknown score mode {mode!r}; expected one of {SCORE_MODES}.")
        if mode not in out:
            out.append(mode)
    return out


def build_run_fingerprint(
    *,
    split: str,
    top_k: int,
    selection_modes: Sequence[str],
    score_modes: Sequence[str],
    verifier_fingerprint: str,
    prompt_fingerprint: str,
    run_instance_id: str = "",
) -> str:
    return stable_fingerprint(
        {
            "version": RUN_VERSION,
            "split": str(split),
            "top_k": int(top_k),
            "selection_modes": list(selection_modes),
            "score_modes": list(score_modes),
            "verifier_fingerprint": str(verifier_fingerprint),
            "prompt_fingerprint": str(prompt_fingerprint),
            "run_instance_id": str(run_instance_id or ""),
        },
        length=24,
    )


def load_raw_score_cache(path: str | Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    cache: dict[str, dict[str, Any]] = {}
    invalid = 0
    duplicates = 0
    path = Path(path)
    if not path.exists():
        return cache, invalid, duplicates
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            key = str(row.get("cache_key") or "")
            if not key or row.get("status") != "completed":
                invalid += 1
                continue
            if key in cache:
                duplicates += 1
            cache[key] = row
    return cache, invalid, duplicates


def score_mode_requires_base(score_mode: str) -> bool:
    return score_mode in {"entropy_reduction", "base_pred_margin"}


def selector_name(selection_mode: str, score_mode: str) -> str:
    return f"verifier_score_{selection_mode}_{score_mode}"


def score_for_mode(
    score_row: dict[str, Any],
    score_mode: str,
    *,
    base_score_row: dict[str, Any] | None = None,
) -> float:
    if score_mode == "pred_margin":
        return _safe_float(score_row.get("pred_margin"), default=-1.0e9)
    if score_mode == "entropy_neg":
        return _safe_float(score_row.get("entropy_neg"), default=-1.0e9)
    if score_mode == "gold_margin":
        return _safe_float(score_row.get("margin"), default=-1.0e9)
    if base_score_row is None:
        raise ValueError(f"score_mode={score_mode!r} requires a base score row.")
    if score_mode == "entropy_reduction":
        return _safe_float(base_score_row.get("entropy"), default=0.0) - _safe_float(
            score_row.get("entropy"), default=0.0
        )
    if score_mode == "base_pred_margin":
        base_letter = str(base_score_row.get("pred_letter") or "")
        return margin_for_letter(dict(score_row.get("label_logprobs") or {}), base_letter)
    raise ValueError(f"Unknown score_mode={score_mode!r}")


def margin_for_letter(label_logprobs: dict[str, float], letter: str) -> float:
    if letter not in LETTER_ORDER:
        return -1.0e9
    scores = {item: float(label_logprobs[item]) for item in LETTER_ORDER if item in label_logprobs}
    if letter not in scores:
        return -1.0e9
    wrong = [value for key, value in scores.items() if key != letter]
    best_wrong = max(wrong) if wrong else float("-inf")
    return float(scores[letter] - best_wrong)


def spec_for_indices(
    example: Stage2OracleExample,
    *,
    selection_mode: str,
    role: str,
    step: int,
    candidate_idx: int | None,
    evidence_indices: Sequence[int],
) -> EvidenceSetSpec:
    indices = tuple(int(idx) for idx in evidence_indices)
    keys = tuple(candidate_key(example.candidates[idx]) for idx in indices)
    return EvidenceSetSpec(
        event_id=str(example.event_id),
        selection_mode=str(selection_mode),
        role=str(role),
        step=int(step),
        candidate_idx=int(candidate_idx) if candidate_idx is not None else None,
        evidence_indices=indices,
        scored_candidate_keys=keys,
        evidence_set_hash=evidence_set_hash(keys),
    )


def build_prompt_row_for_spec(
    example: Stage2OracleExample,
    spec: EvidenceSetSpec,
    *,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
) -> dict[str, Any]:
    candidates = [dict(example.candidates[idx]) for idx in spec.evidence_indices]
    retrieval_row = {
        "event_id": example.event_id,
        "claim": example.claim,
        "label": example.gold_label,
        "explain": str(example.raw.get("explain") or example.raw.get("gold_explain") or ""),
        "candidates": candidates,
    }
    return _build_training_row(retrieval_row, tokenizer, prompt_cfg)


def run_chunked_selector(
    *,
    examples: list[Stage2OracleExample],
    output_dir: str | Path,
    split: str,
    top_k: int,
    selection_modes: str | Sequence[str],
    score_modes: str | Sequence[str],
    verifier_fingerprint: str,
    prompt_fingerprint: str,
    scorer: Any | None = None,
    tokenizer: Any | None = None,
    prompt_cfg: dict[str, Any] | None = None,
    prompt_builder: PromptBuilder | None = None,
    claim_batch_size: int = 8,
    resume: bool = True,
    fsync_cache: bool = False,
    finalize_only: bool = False,
    run_instance_id: str = "",
    run_metadata: dict[str, Any] | None = None,
    no_progress: bool = False,
) -> dict[str, Any]:
    selection_mode_list = normalize_selection_modes(selection_modes)
    score_mode_list = normalize_score_modes(score_modes)
    if not resume and not run_instance_id:
        run_instance_id = datetime.now(timezone.utc).isoformat()
    run_fingerprint = build_run_fingerprint(
        split=split,
        top_k=int(top_k),
        selection_modes=selection_mode_list,
        score_modes=score_mode_list,
        verifier_fingerprint=verifier_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        run_instance_id=run_instance_id,
    )

    out_dir = Path(output_dir)
    resume_dir = out_dir / "_resume"
    events_dir = resume_dir / "events"
    resume_dir.mkdir(parents=True, exist_ok=True)
    store = EventCheckpointStore(events_dir)
    cache_path = resume_dir / "raw_verifier_scores.jsonl"
    score_cache, invalid_cache_rows, duplicate_cache_rows = (
        load_raw_score_cache(cache_path) if resume else ({}, 0, 0)
    )
    completed = store.load_completed(run_fingerprint) if resume else {}

    stats: dict[str, Any] = {
        "status": "running",
        "run_fingerprint": run_fingerprint,
        "run_version": RUN_VERSION,
        "split": str(split),
        "top_k": int(top_k),
        "selection_modes": selection_mode_list,
        "score_modes": score_mode_list,
        "n_examples": int(len(examples)),
        "claim_batch_size": int(claim_batch_size),
        "resume": bool(resume),
        "finalize_only": bool(finalize_only),
        "cache_path": str(cache_path),
        "cache_invalid_lines": int(invalid_cache_rows),
        "cache_duplicate_rows": int(duplicate_cache_rows),
        "n_completed_events_at_start": int(len(completed)),
        "n_skipped_completed_events": 0,
        "n_claim_batches": 0,
        "n_score_requests": 0,
        "n_score_cache_hits": 0,
        "n_score_generated": 0,
        "n_event_checkpoints_written": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **dict(run_metadata or {}),
    }
    _write_manifest(resume_dir / "manifest.json", stats)

    if not finalize_only:
        if scorer is None:
            raise ValueError("scorer is required unless --finalize-only is used.")
        if prompt_builder is None and (tokenizer is None or prompt_cfg is None):
            raise ValueError("tokenizer and prompt_cfg are required without a custom prompt_builder.")
        with ScoreCacheWriter(cache_path, fsync_cache=fsync_cache) as writer:
            for batch_start in range(0, len(examples), max(1, int(claim_batch_size))):
                batch = examples[batch_start : batch_start + max(1, int(claim_batch_size))]
                pending = [example for example in batch if example.event_id not in completed]
                stats["n_claim_batches"] += 1
                stats["n_skipped_completed_events"] += len(batch) - len(pending)
                if not pending:
                    _write_manifest(resume_dir / "manifest.json", stats)
                    continue
                if not no_progress:
                    print(
                        f"claim batch {stats['n_claim_batches']} | "
                        f"events {batch_start + 1}-{batch_start + len(batch)} / {len(examples)} | "
                        f"pending={len(pending)}",
                        flush=True,
                    )
                event_payloads = {
                    example.event_id: _empty_event_payload(
                        example,
                        run_fingerprint=run_fingerprint,
                        selection_modes=selection_mode_list,
                        score_modes=score_mode_list,
                        top_k=int(top_k),
                    )
                    for example in pending
                }
                if STATIC_TOP5 in selection_mode_list:
                    _process_static_batch(
                        pending,
                        event_payloads=event_payloads,
                        score_modes=score_mode_list,
                        scorer=scorer,
                        writer=writer,
                        score_cache=score_cache,
                        stats=stats,
                        split=split,
                        top_k=int(top_k),
                        verifier_fingerprint=verifier_fingerprint,
                        prompt_fingerprint=prompt_fingerprint,
                        tokenizer=tokenizer,
                        prompt_cfg=prompt_cfg or {},
                        prompt_builder=prompt_builder,
                    )
                if GREEDY_STEPWISE_TOP5 in selection_mode_list:
                    _process_greedy_batch(
                        pending,
                        event_payloads=event_payloads,
                        score_modes=score_mode_list,
                        scorer=scorer,
                        writer=writer,
                        score_cache=score_cache,
                        stats=stats,
                        split=split,
                        top_k=int(top_k),
                        verifier_fingerprint=verifier_fingerprint,
                        prompt_fingerprint=prompt_fingerprint,
                        tokenizer=tokenizer,
                        prompt_cfg=prompt_cfg or {},
                        prompt_builder=prompt_builder,
                    )
                for event_id, payload in event_payloads.items():
                    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
                    store.write_event(event_id, payload)
                    completed[event_id] = payload
                    stats["n_event_checkpoints_written"] += 1
                stats["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_manifest(resume_dir / "manifest.json", stats)

    final = finalize_outputs(
        output_dir=out_dir,
        store=store,
        examples=examples,
        run_fingerprint=run_fingerprint,
        selection_modes=selection_mode_list,
        score_modes=score_mode_list,
        top_k=int(top_k),
        run_metadata={**stats, "status": "completed"},
    )
    stats.update(
        {
            "status": "completed",
            "n_finalized_events": int(final.get("n_finalized_events", 0)),
            "comparison_table": str(out_dir / "comparison_table.json"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_manifest(resume_dir / "manifest.json", stats)
    return final


def _process_static_batch(
    examples: list[Stage2OracleExample],
    *,
    event_payloads: dict[str, dict[str, Any]],
    score_modes: list[str],
    scorer: Any,
    writer: ScoreCacheWriter,
    score_cache: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    split: str,
    top_k: int,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_builder: PromptBuilder | None,
) -> None:
    include_base = any(score_mode_requires_base(mode) for mode in score_modes)
    specs: list[EvidenceSetSpec] = []
    for example in examples:
        if include_base:
            specs.append(
                spec_for_indices(
                    example,
                    selection_mode=STATIC_TOP5,
                    role="static_base",
                    step=0,
                    candidate_idx=None,
                    evidence_indices=[],
                )
            )
        for idx in range(len(example.candidates)):
            specs.append(
                spec_for_indices(
                    example,
                    selection_mode=STATIC_TOP5,
                    role="static_candidate",
                    step=0,
                    candidate_idx=idx,
                    evidence_indices=[idx],
                )
            )
    _ensure_scores(
        examples,
        specs,
        scorer=scorer,
        writer=writer,
        score_cache=score_cache,
        stats=stats,
        split=split,
        verifier_fingerprint=verifier_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        prompt_builder=prompt_builder,
    )
    for example in examples:
        for score_mode in score_modes:
            trace = _build_static_trace(
                example,
                score_mode=score_mode,
                score_cache=score_cache,
                split=split,
                top_k=top_k,
                verifier_fingerprint=verifier_fingerprint,
                prompt_fingerprint=prompt_fingerprint,
            )
            event_payloads[example.event_id]["traces"][trace["selector_name"]] = trace


def _process_greedy_batch(
    examples: list[Stage2OracleExample],
    *,
    event_payloads: dict[str, dict[str, Any]],
    score_modes: list[str],
    scorer: Any,
    writer: ScoreCacheWriter,
    score_cache: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    split: str,
    top_k: int,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_builder: PromptBuilder | None,
) -> None:
    for score_mode in score_modes:
        selected_by_event = {example.event_id: [] for example in examples}
        per_step_by_event = {example.event_id: [] for example in examples}
        for step in range(int(top_k)):
            active = [
                example
                for example in examples
                if len(selected_by_event[example.event_id]) < min(int(top_k), len(example.candidates))
            ]
            if not active:
                break
            specs: list[EvidenceSetSpec] = []
            for example in active:
                selected = selected_by_event[example.event_id]
                if score_mode_requires_base(score_mode):
                    specs.append(
                        spec_for_indices(
                            example,
                            selection_mode=GREEDY_STEPWISE_TOP5,
                            role="greedy_base",
                            step=step,
                            candidate_idx=None,
                            evidence_indices=selected,
                        )
                    )
                selected_set = set(selected)
                for idx in range(len(example.candidates)):
                    if idx in selected_set:
                        continue
                    specs.append(
                        spec_for_indices(
                            example,
                            selection_mode=GREEDY_STEPWISE_TOP5,
                            role="greedy_candidate",
                            step=step,
                            candidate_idx=idx,
                            evidence_indices=[*selected, idx],
                        )
                    )
            _ensure_scores(
                active,
                specs,
                scorer=scorer,
                writer=writer,
                score_cache=score_cache,
                stats=stats,
                split=split,
                verifier_fingerprint=verifier_fingerprint,
                prompt_fingerprint=prompt_fingerprint,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg,
                prompt_builder=prompt_builder,
            )
            for example in active:
                selected = selected_by_event[example.event_id]
                base_row = None
                if score_mode_requires_base(score_mode):
                    base_spec = spec_for_indices(
                        example,
                        selection_mode=GREEDY_STEPWISE_TOP5,
                        role="greedy_base",
                        step=step,
                        candidate_idx=None,
                        evidence_indices=selected,
                    )
                    base_row = score_cache[_cache_key(base_spec, split, verifier_fingerprint, prompt_fingerprint)]
                step_scores: list[dict[str, Any]] = []
                for idx in range(len(example.candidates)):
                    if idx in selected:
                        continue
                    spec = spec_for_indices(
                        example,
                        selection_mode=GREEDY_STEPWISE_TOP5,
                        role="greedy_candidate",
                        step=step,
                        candidate_idx=idx,
                        evidence_indices=[*selected, idx],
                    )
                    row = score_cache[_cache_key(spec, split, verifier_fingerprint, prompt_fingerprint)]
                    value = score_for_mode(row, score_mode, base_score_row=base_row)
                    step_scores.append(
                        _trace_score_payload(
                            row,
                            spec=spec,
                            selector_score=value,
                            base_score_row=base_row,
                        )
                    )
                best = max(step_scores, key=lambda row: (float(row["selector_score"]), -int(row["candidate_idx"])))
                selected.append(int(best["candidate_idx"]))
                per_step_by_event[example.event_id].append(
                    {
                        "step": int(step),
                        "selected_idx": int(best["candidate_idx"]),
                        "selected_score": float(best["selector_score"]),
                        "base_pred_label": base_row.get("pred_label") if base_row else None,
                        "base_entropy": base_row.get("entropy") if base_row else None,
                        "candidate_scores": step_scores,
                    }
                )
        for example in examples:
            trace = _build_greedy_trace(
                example,
                score_mode=score_mode,
                selected_indices=selected_by_event[example.event_id],
                per_step_action_scores=per_step_by_event[example.event_id],
                top_k=top_k,
            )
            event_payloads[example.event_id]["traces"][trace["selector_name"]] = trace


def _ensure_scores(
    examples: list[Stage2OracleExample],
    specs: list[EvidenceSetSpec],
    *,
    scorer: Any,
    writer: ScoreCacheWriter,
    score_cache: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    split: str,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    prompt_builder: PromptBuilder | None,
) -> None:
    example_by_event = {example.event_id: example for example in examples}
    requests_by_key: dict[str, ScoreRequest] = {}
    for spec in specs:
        key = _cache_key(spec, split, verifier_fingerprint, prompt_fingerprint)
        stats["n_score_requests"] += 1
        if key in score_cache:
            stats["n_score_cache_hits"] += 1
            continue
        if key in requests_by_key:
            continue
        example = example_by_event[spec.event_id]
        prompt_row = (
            prompt_builder(example, spec)
            if prompt_builder is not None
            else build_prompt_row_for_spec(example, spec, tokenizer=tokenizer, prompt_cfg=prompt_cfg)
        )
        requests_by_key[key] = ScoreRequest(
            prompt_row=prompt_row,
            gold_label=example.gold_label,
            cache_key=key,
            event_id=example.event_id,
            claim=example.claim,
            evidence_set_hash=spec.evidence_set_hash,
            scored_candidate_keys=list(spec.scored_candidate_keys),
            metadata={
                "selection_mode": spec.selection_mode,
                "label_policy": _label_policy(spec.selection_mode),
                "role": spec.role,
                "step": spec.step,
                "candidate_idx": spec.candidate_idx,
                "evidence_indices": list(spec.evidence_indices),
            },
        )
    if not requests_by_key:
        return

    def _on_batch_complete(
        batch_requests: list[ScoreRequest],
        batch_scores: list[dict[str, Any]],
    ) -> None:
        rows = [
            _cache_row_from_score(
                request,
                score,
                split=split,
                verifier_fingerprint=verifier_fingerprint,
                prompt_fingerprint=prompt_fingerprint,
            )
            for request, score in zip(batch_requests, batch_scores)
        ]
        writer.append_many(rows)
        for row in rows:
            score_cache[str(row["cache_key"])] = row
        stats["n_score_generated"] += len(rows)

    requests = list(requests_by_key.values())
    try:
        scorer.score_batch(requests, on_batch_complete=_on_batch_complete)
    except TypeError as exc:
        if "on_batch_complete" not in str(exc):
            raise
        scores = scorer.score_batch(requests)
        _on_batch_complete(requests, scores)


def _cache_row_from_score(
    request: ScoreRequest,
    score: dict[str, Any],
    *,
    split: str,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
) -> dict[str, Any]:
    metadata = dict(request.metadata or {})
    return {
        **dict(score),
        "status": "completed",
        "cache_key": request.cache_key,
        "split": str(split),
        "event_id": request.event_id,
        "claim": request.claim,
        "gold_label": request.gold_label,
        "selection_mode": metadata.get("selection_mode"),
        "label_policy": metadata.get("label_policy"),
        "role": metadata.get("role"),
        "step": int(metadata.get("step", -1)),
        "candidate_idx": metadata.get("candidate_idx"),
        "evidence_indices": list(metadata.get("evidence_indices") or []),
        "evidence_set_hash": request.evidence_set_hash,
        "scored_candidate_keys": list(request.scored_candidate_keys),
        "n_scored_candidates": len(request.scored_candidate_keys),
        "prompt_token_count": int(request.prompt_row.get("prompt_token_count") or 0),
        "was_truncated": bool(request.prompt_row.get("was_truncated")),
        "verifier_config_fingerprint": verifier_fingerprint,
        "prompt_config_fingerprint": prompt_fingerprint,
        "scoring_backend": "llm",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_static_trace(
    example: Stage2OracleExample,
    *,
    score_mode: str,
    score_cache: dict[str, dict[str, Any]],
    split: str,
    top_k: int,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
) -> dict[str, Any]:
    base_row = None
    if score_mode_requires_base(score_mode):
        base_spec = spec_for_indices(
            example,
            selection_mode=STATIC_TOP5,
            role="static_base",
            step=0,
            candidate_idx=None,
            evidence_indices=[],
        )
        base_row = score_cache[_cache_key(base_spec, split, verifier_fingerprint, prompt_fingerprint)]

    scores = np.full((len(example.candidates),), -1.0e9, dtype=np.float32)
    per_candidate: list[dict[str, Any]] = []
    for idx in range(len(example.candidates)):
        spec = spec_for_indices(
            example,
            selection_mode=STATIC_TOP5,
            role="static_candidate",
            step=0,
            candidate_idx=idx,
            evidence_indices=[idx],
        )
        row = score_cache[_cache_key(spec, split, verifier_fingerprint, prompt_fingerprint)]
        value = score_for_mode(row, score_mode, base_score_row=base_row)
        scores[idx] = _finite_score(value)
        per_candidate.append(
            _trace_score_payload(row, spec=spec, selector_score=value, base_score_row=base_row)
        )

    name = selector_name(STATIC_TOP5, score_mode)
    trace = build_selection_trace(example, scores, selector_name=name, top_k=int(top_k))
    trace["selection_mode"] = STATIC_TOP5
    trace["score_mode"] = score_mode
    trace["score_mode_is_deployable"] = score_mode in DEPLOYABLE_SCORE_MODES
    trace["per_candidate_verifier_scores"] = per_candidate
    return trace


def _build_greedy_trace(
    example: Stage2OracleExample,
    *,
    score_mode: str,
    selected_indices: list[int],
    per_step_action_scores: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    scores = np.full((len(example.candidates),), -1.0e9, dtype=np.float32)
    for rank, idx in enumerate(selected_indices[:top_k]):
        if 0 <= int(idx) < len(scores):
            scores[int(idx)] = float(top_k - rank)
    name = selector_name(GREEDY_STEPWISE_TOP5, score_mode)
    trace = build_selection_trace(example, scores, selector_name=name, top_k=int(top_k))
    trace["selector_ordered_indices"] = [int(idx) for idx in selected_indices[:top_k]]
    trace.update(ordered_selection_metrics(example.selected_indices, trace["selector_ordered_indices"], top_k=top_k))
    trace["selection_mode"] = GREEDY_STEPWISE_TOP5
    trace["score_mode"] = score_mode
    trace["score_mode_is_deployable"] = score_mode in DEPLOYABLE_SCORE_MODES
    trace["per_step_action_scores"] = per_step_action_scores
    return trace


def finalize_outputs(
    *,
    output_dir: str | Path,
    store: EventCheckpointStore,
    examples: list[Stage2OracleExample],
    run_fingerprint: str,
    selection_modes: Sequence[str],
    score_modes: Sequence[str],
    top_k: int,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_by_event = {example.event_id: example for example in examples}
    payloads = store.iter_events(run_fingerprint=run_fingerprint)
    payload_by_event = {str(payload.get("event_id")): payload for payload in payloads}
    ordered_payloads = [
        payload_by_event[example.event_id]
        for example in examples
        if example.event_id in payload_by_event
    ]

    comparison_rows: list[dict[str, Any]] = []
    for selection_mode in selection_modes:
        for score_mode in score_modes:
            name = selector_name(selection_mode, score_mode)
            traces = [
                payload["traces"][name]
                for payload in ordered_payloads
                if name in dict(payload.get("traces") or {})
            ]
            strategy_dir = out_dir / name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            controls = _control_traces(
                [examples_by_event[str(trace["event_id"])] for trace in traces],
                top_k=int(top_k),
            )
            selector_metrics = summarize_ordered_selection(traces)
            control_metrics = {
                key: summarize_ordered_selection(items)
                for key, items in sorted(controls.items())
            }
            _write_jsonl(strategy_dir / "selection_trace.jsonl", traces)
            _write_jsonl(strategy_dir / "control_hybrid_trace.jsonl", controls["hybrid_score_top5"])
            _write_jsonl(
                strategy_dir / "control_candidate_pool_trace.jsonl",
                controls["candidate_pool_order_top5"],
            )
            metrics_payload = {
                **dict(run_metadata),
                "selector_name": name,
                "selection_mode": selection_mode,
                "score_mode": score_mode,
                "score_mode_is_deployable": score_mode in DEPLOYABLE_SCORE_MODES,
                "n_claims": int(len(traces)),
                "selector_metrics": selector_metrics,
                "controls": control_metrics,
            }
            _write_json(strategy_dir / "selection_metrics.json", metrics_payload)
            _write_analysis(strategy_dir / "analysis.md", metrics_payload)
            hybrid = control_metrics.get("hybrid_score_top5", {})
            row = {
                "selector_name": name,
                "selection_mode": selection_mode,
                "score_mode": score_mode,
                "score_mode_is_deployable": score_mode in DEPLOYABLE_SCORE_MODES,
                "n_claims": int(len(traces)),
                "recall@5": _metric(selector_metrics, "recall@5"),
                "jaccard@5": _metric(selector_metrics, "jaccard@5"),
                "top1_match": _metric(selector_metrics, "top1_match"),
                "oracle_rank_ndcg@5": _metric(selector_metrics, "oracle_rank_ndcg@5"),
                "pairwise_order_acc@5": _metric(selector_metrics, "pairwise_order_acc@5"),
                "delta_vs_hybrid_jaccard@5": _metric(selector_metrics, "jaccard@5")
                - _metric(hybrid, "jaccard@5"),
                "delta_vs_hybrid_recall@5": _metric(selector_metrics, "recall@5")
                - _metric(hybrid, "recall@5"),
            }
            comparison_rows.append(row)

    final = {
        **dict(run_metadata),
        "run_fingerprint": run_fingerprint,
        "n_examples": int(len(examples)),
        "n_finalized_events": int(len(ordered_payloads)),
        "comparison": comparison_rows,
    }
    _write_json(out_dir / "comparison_table.json", final)
    _write_comparison_markdown(out_dir / "analysis.md", final)
    return final


def _control_traces(
    examples: list[Stage2OracleExample],
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    controls = {
        "hybrid_score_top5": [],
        "candidate_pool_order_top5": [],
    }
    for example in examples:
        base = build_selection_trace(
            example,
            np.zeros((len(example.candidates),), dtype=np.float32),
            selector_name="control_stub",
            top_k=int(top_k),
        )
        controls["hybrid_score_top5"].append(
            build_order_control_trace(
                base,
                ranked_indices_from_hybrid(example, top_k=int(top_k)),
                selector_name="hybrid_score_top5",
                top_k=int(top_k),
            )
        )
        controls["candidate_pool_order_top5"].append(
            build_order_control_trace(
                base,
                ranked_indices_from_candidate_pool(example, top_k=int(top_k)),
                selector_name="candidate_pool_order_top5",
                top_k=int(top_k),
            )
        )
    return controls


def _empty_event_payload(
    example: Stage2OracleExample,
    *,
    run_fingerprint: str,
    selection_modes: list[str],
    score_modes: list[str],
    top_k: int,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "run_fingerprint": run_fingerprint,
        "run_version": RUN_VERSION,
        "event_id": example.event_id,
        "gold_label": example.gold_label,
        "fingerprint": example.fingerprint,
        "top_k": int(top_k),
        "selection_modes": list(selection_modes),
        "score_modes": list(score_modes),
        "traces": {},
    }


def _trace_score_payload(
    score_row: dict[str, Any],
    *,
    spec: EvidenceSetSpec,
    selector_score: float,
    base_score_row: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "candidate_idx": spec.candidate_idx,
        "step": int(spec.step),
        "role": spec.role,
        "evidence_indices": list(spec.evidence_indices),
        "evidence_set_hash": spec.evidence_set_hash,
        "selector_score": float(selector_score),
        "pred_label": score_row.get("pred_label"),
        "pred_letter": score_row.get("pred_letter"),
        "pred_margin": _safe_float(score_row.get("pred_margin"), default=math.nan),
        "entropy": _safe_float(score_row.get("entropy"), default=math.nan),
        "entropy_neg": _safe_float(score_row.get("entropy_neg"), default=math.nan),
        "gold_margin_diagnostic": _safe_float(score_row.get("margin"), default=math.nan),
        "label_logprobs": dict(score_row.get("label_logprobs") or {}),
        "base_pred_label": base_score_row.get("pred_label") if base_score_row else None,
        "base_pred_letter": base_score_row.get("pred_letter") if base_score_row else None,
        "base_entropy": base_score_row.get("entropy") if base_score_row else None,
        "prompt_token_count": int(score_row.get("prompt_token_count") or 0),
        "was_truncated": bool(score_row.get("was_truncated")),
    }


def _cache_key(
    spec: EvidenceSetSpec,
    split: str,
    verifier_fingerprint: str,
    prompt_fingerprint: str,
) -> str:
    return cache_key_for_score(
        split=str(split),
        event_id=spec.event_id,
        evidence_set_hash_value=spec.evidence_set_hash,
        verifier_fingerprint=verifier_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        label_policy=_label_policy(spec.selection_mode),
        scoring_backend="llm",
    )


def _label_policy(selection_mode: str) -> str:
    return f"verifier_score_{selection_mode}"


def _split_values(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = []
        for item in value:
            raw.extend(str(item).split(","))
    return [item.strip() for item in raw if item and item.strip()]


def _finite_score(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0e9
    if math.isnan(number) or math.isinf(number):
        return -1.0e9
    return number


def _safe_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number):
        return float(default)
    return number


def _metric(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(json_safe(dict(row)), ensure_ascii=False) + "\n")


def _write_analysis(path: Path, metrics: dict[str, Any]) -> None:
    selector = metrics.get("selector_metrics", {})
    hybrid = dict(metrics.get("controls", {})).get("hybrid_score_top5", {})
    lines = [
        "# Verifier-Score Selector Eval",
        "",
        f"- selector: `{metrics.get('selector_name')}`",
        f"- deployable_score_mode: `{metrics.get('score_mode_is_deployable')}`",
        f"- claims: {metrics.get('n_claims')}",
        f"- recall@5: {_metric(selector, 'recall@5'):.4f}",
        f"- jaccard@5: {_metric(selector, 'jaccard@5'):.4f}",
        f"- top1_match: {_metric(selector, 'top1_match'):.4f}",
        f"- oracle_rank_ndcg@5: {_metric(selector, 'oracle_rank_ndcg@5'):.4f}",
        f"- pairwise_order_acc@5: {_metric(selector, 'pairwise_order_acc@5'):.4f}",
        f"- delta_vs_hybrid_jaccard@5: {_metric(selector, 'jaccard@5') - _metric(hybrid, 'jaccard@5'):.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_markdown(path: Path, final: dict[str, Any]) -> None:
    rows = list(final.get("comparison") or [])
    lines = [
        "# Verifier-Score Selector Comparison",
        "",
        f"- finalized_events: {final.get('n_finalized_events')}",
        f"- run_fingerprint: `{final.get('run_fingerprint')}`",
        "",
        "| selector | deployable | recall@5 | jaccard@5 | top1 | ndcg@5 | pairwise@5 | delta_jaccard_vs_hybrid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {selector} | {deployable} | {recall:.4f} | {jaccard:.4f} | {top1:.4f} | "
            "{ndcg:.4f} | {pairwise:.4f} | {delta:.4f} |".format(
                selector=row.get("selector_name"),
                deployable=str(row.get("score_mode_is_deployable")),
                recall=float(row.get("recall@5", 0.0)),
                jaccard=float(row.get("jaccard@5", 0.0)),
                top1=float(row.get("top1_match", 0.0)),
                ndcg=float(row.get("oracle_rank_ndcg@5", 0.0)),
                pairwise=float(row.get("pairwise_order_acc@5", 0.0)),
                delta=float(row.get("delta_vs_hybrid_jaccard@5", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
