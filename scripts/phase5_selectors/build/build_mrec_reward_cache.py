#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fact_checking.data.constants import letter_order_for_schema  # noqa: E402
from fact_checking.data.io import load_split  # noqa: E402
from fact_checking.selectors.minimal_resolving_chain import (  # noqa: E402
    MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
    MREC_SELECTION_POLICY_TRANSITION_V0_1,
    MRECSelectorParams,
    _atom_by_id,
    _candidate_pool,
    _claim_atoms,
    _copy_step_metadata,
    _evaluate_candidate_transition,
    _fallback_candidate_evaluation,
)
from fact_checking.selectors.mrec_learned_marginal import (  # noqa: E402
    extract_marginal_features,
    hard_state_to_soft_state,
)
from fact_checking.selectors.mrec_schema import build_initial_atom_states, build_mrec_step  # noqa: E402
RUN_VERSION = "mrec_learned_marginal_reward_cache_v0_3"
ROLLIN_POLICY_VERIFIER_UTILITY_GREEDY = "verifier_utility_greedy"
SHARD_ASSIGNMENT = "sha256_event_id_mod"
DEFAULT_TEACHER_RUN_DIR = (
    "outputs/sentence_trace_method/"
    "liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_top5_"
    "lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train"
)
DEFAULT_BASE_MODEL = "/data/models/Ministral-3-8B-Instruct-2512"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload: Mapping[str, Any], *, length: int = 16) -> str:
    blob = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[: int(length)]


def load_label_token_ids(
    run_dir: str | Path,
    *,
    label_prefix: str,
    label_schema: str,
) -> dict[str, int]:
    meta_path = Path(run_dir) / "label_token_ce_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing label token metadata: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(payload.get("label_prefix") or label_prefix) != str(label_prefix):
        raise ValueError(
            f"label_prefix mismatch: expected {label_prefix!r}, got {payload.get('label_prefix')!r}."
        )
    ids = {str(key): int(value) for key, value in dict(payload.get("label_token_ids") or {}).items()}
    letters = letter_order_for_schema(str(payload.get("label_schema") or label_schema))
    missing = [letter for letter in letters if letter not in ids]
    if missing:
        raise ValueError(f"label_token_ce_meta.json is missing label token ids for: {missing}")
    return {letter: ids[letter] for letter in letters}


def _load_prompt_tokenizer(model_name_or_path: str) -> Any:
    from fact_checking.build.candidates import _load_prompt_tokenizer as impl

    return impl(model_name_or_path)


def _build_training_row(retrieval_row: Mapping[str, Any], tokenizer: Any, prompt_cfg: Mapping[str, Any]) -> dict[str, Any]:
    from fact_checking.build.candidates import _build_training_row as impl

    return impl(retrieval_row, tokenizer, prompt_cfg)


def _apply_mrec_prompt_fields(**kwargs: Any) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    from scripts.phase5_selectors.build.build_trace_verifier_data import _apply_mrec_prompt_fields as impl

    return impl(**kwargs)


def _prompt_cfg_for_trace_style(prompt_cfg: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    from scripts.phase5_selectors.build.build_trace_verifier_data import _prompt_cfg_for_trace_style as impl

    return impl(prompt_cfg, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MREC learned-marginal reward cache with a frozen verifier.")
    parser.add_argument("--input", required=True, help="candidate_evidence_map_features_<split>.jsonl")
    parser.add_argument("--raw", required=True, help="Raw split JSON file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--dataset", default="liar_raw")
    parser.add_argument("--label-schema", default="liar6")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--verify-input-only",
        action="store_true",
        help="Validate the utility-cache input contract and estimate scoring work without loading a teacher.",
    )
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument(
        "--rollin-policy",
        default=MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
        choices=[
            MREC_SELECTION_POLICY_TRANSITION_V0_1,
            MREC_SELECTION_POLICY_LEARNED_MARGINAL_PROXY,
            ROLLIN_POLICY_VERIFIER_UTILITY_GREEDY,
        ],
    )
    parser.add_argument("--rollin-weight-file", default="")
    parser.add_argument("--rollin-stop-threshold", type=float, default=0.0)
    parser.add_argument("--target-resolved-rate", type=float, default=1.0)
    parser.add_argument("--cue-policy", default="atom_proposition", choices=["atom_proposition", "atom_query", "legacy_route_prefer"])
    parser.add_argument("--trace-prompt-style", default="mrec_min", choices=["mrec_min"])
    parser.add_argument("--prompt-max-length", type=int, default=1024)
    parser.add_argument("--prompt-model-name-or-path", default="")
    parser.add_argument("--teacher-run-dir", default=DEFAULT_TEACHER_RUN_DIR)
    parser.add_argument("--teacher-checkpoint", default="best")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--label-prefix", default="Label:")
    parser.add_argument("--scoring-backend", default="auto", choices=["auto", "vllm", "transformers"])
    parser.add_argument("--vllm-tokenizer-path", default="")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=4)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-tokenizer-mode", default="auto")
    parser.add_argument("--vllm-config-format", default="")
    parser.add_argument("--vllm-load-format", default="")
    parser.add_argument("--vllm-max-model-len", type=int, default=1032)
    parser.add_argument("--vllm-prompt-batch-size", type=int, default=6000)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-lora-mode", default="dynamic", choices=["dynamic", "merged"])
    parser.add_argument("--vllm-merge-lora-cache-dir", default="outputs/cache/merged_lora")
    parser.add_argument("--vllm-merge-lora-force-rebuild", action="store_true")
    parser.add_argument("--transformers-tokenizer-path", default="")
    parser.add_argument("--transformers-device", default="auto")
    parser.add_argument("--transformers-dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--transformers-prompt-batch-size", type=int, default=24)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fsync-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_execution_args(args)
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_key = _artifact_key(
        str(args.split),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    records_path = out_dir / f"reward_records_{artifact_key}.jsonl"
    raw_scores_path = out_dir / f"raw_teacher_scores_{artifact_key}.jsonl"
    events_path = out_dir / f"reward_event_summaries_{artifact_key}.jsonl"
    manifest_path = out_dir / f"manifest_{artifact_key}.json"
    diagnostics_path = out_dir / f"diagnostics_{artifact_key}.json"
    validation_path = out_dir / f"input_validation_{artifact_key}.json"

    source_rows = _read_jsonl(Path(args.input), sample_limit=0)
    sample_limit = int(args.sample_limit)
    considered_rows = source_rows[:sample_limit] if sample_limit > 0 else source_rows
    raw_by_event = _index_raw_samples(
        load_split(args.raw, dataset=str(args.dataset), label_schema=str(args.label_schema))
    )
    validation = _validate_input_contract(considered_rows, raw_by_event=raw_by_event, args=args)
    rows = _select_shard_rows(
        considered_rows,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    shard_plan = _shard_plan(considered_rows, args=args)
    validation.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_version": RUN_VERSION,
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "raw": str(args.raw),
            "raw_sha256": sha256_file(args.raw),
            "sample_limit": sample_limit,
            "n_source_rows": len(source_rows),
            "n_considered_rows": len(considered_rows),
            "n_shard_rows": len(rows),
            "shard": {
                "assignment": SHARD_ASSIGNMENT,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "event_ids_fingerprint": _event_ids_fingerprint(rows),
                "plan": shard_plan,
            },
        }
    )
    _write_json(validation_path, validation)
    if bool(args.verify_input_only):
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"Wrote input validation report: {validation_path}")
        return 0

    checkpoint = _teacher_checkpoint(args)
    prompt_cfg = _prompt_cfg(args)
    tokenizer = _load_prompt_tokenizer(str(prompt_cfg["model_name_or_path"]))
    scorer = _init_scorer(args, checkpoint)
    checkpoint = dict(checkpoint)
    checkpoint["scoring_fingerprint"] = _scoring_fingerprint(args, scorer)

    run_contract = _run_contract(
        args=args,
        checkpoint=checkpoint,
        prompt_cfg=prompt_cfg,
        validation=validation,
    )
    run_fingerprint = stable_fingerprint(run_contract, length=24)
    checkpoint["run_fingerprint"] = run_fingerprint
    if bool(args.resume) and manifest_path.exists() and _is_strict_utility_mode(args):
        _validate_resume_manifest(manifest_path, expected_run_fingerprint=run_fingerprint)
    score_cache = (
        _load_score_cache(
            raw_scores_path,
            expected_run_fingerprint=run_fingerprint if _is_strict_utility_mode(args) else "",
        )
        if bool(args.resume)
        else {}
    )
    completed_events = (
        _completed_utility_event_ids(
            records_path,
            events_path,
            expected_run_fingerprint=run_fingerprint,
        )
        if bool(args.resume) and _is_strict_utility_mode(args)
        else (_completed_event_ids(records_path) if bool(args.resume) else set())
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_version": RUN_VERSION,
        "run_fingerprint": run_fingerprint,
        "run_contract": run_contract,
        "input": str(args.input),
        "raw": str(args.raw),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "dataset": str(args.dataset),
        "label_schema": str(args.label_schema),
        "params": {
            "candidate_top_n": int(args.candidate_top_n),
            "rollout_steps": int(args.rollout_steps),
            "rollin_policy": str(args.rollin_policy),
            "rollin_weight_file": str(args.rollin_weight_file or ""),
            "rollin_stop_threshold": float(args.rollin_stop_threshold),
            "target_resolved_rate": float(args.target_resolved_rate),
            "cue_policy": str(args.cue_policy),
            "trace_prompt_style": str(args.trace_prompt_style),
            "prompt_max_length": int(args.prompt_max_length),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "teacher": {
            **checkpoint,
            "scoring_backend_requested": str(args.scoring_backend),
            "scoring_backend": str(getattr(scorer, "backend_name", args.scoring_backend)),
            "vllm_lora_mode": str(getattr(scorer, "vllm_lora_mode", args.vllm_lora_mode)),
        },
        "prompt_config": prompt_cfg,
        "n_source_rows": len(source_rows),
        "n_considered_rows": len(considered_rows),
        "n_input_rows": len(rows),
        "n_completed_events_at_start": len(completed_events),
        "input_validation_path": str(validation_path),
        "raw_scores_path": str(raw_scores_path),
        "records_path": str(records_path),
        "event_summaries_path": str(events_path),
        "status": "running",
    }
    _write_json(manifest_path, manifest)

    iterator: Sequence[Mapping[str, Any]] = rows
    if not bool(args.no_progress):
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(rows, desc=f"mrec-reward-cache [{args.split}]", unit="claim", dynamic_ncols=True)
        except Exception:
            pass

    mode = "a" if bool(args.resume) else "w"
    stats = Counter()
    delta_values: list[float] = []
    with (
        records_path.open(mode, encoding="utf-8") as records_fh,
        events_path.open(mode, encoding="utf-8") as events_fh,
        raw_scores_path.open("a" if bool(args.resume) else "w", encoding="utf-8") as raw_scores_fh,
    ):
        for row in iterator:
            event_id = str(row.get("event_id") or "")
            if not event_id:
                stats["missing_event_id"] += 1
                continue
            if event_id in completed_events:
                stats["skipped_completed"] += 1
                continue
            sample = raw_by_event.get(event_id)
            if sample is None:
                stats["missing_raw_sample"] += 1
                continue
            event_rows, event_summary, score_rows = _build_event_reward_rows(
                row,
                sample=sample,
                args=args,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg,
                scorer=scorer,
                score_cache=score_cache,
                checkpoint=checkpoint,
            )
            for score_row in score_rows:
                raw_scores_fh.write(json.dumps(score_row, ensure_ascii=False, sort_keys=True) + "\n")
            raw_scores_fh.flush()
            for reward_row in event_rows:
                records_fh.write(json.dumps(reward_row, ensure_ascii=False, sort_keys=True) + "\n")
                delta_values.append(float(reward_row.get("delta_margin", 0.0)))
            events_fh.write(json.dumps(event_summary, ensure_ascii=False, sort_keys=True) + "\n")
            records_fh.flush()
            events_fh.flush()
            if bool(args.fsync_cache):
                os.fsync(raw_scores_fh.fileno())
                os.fsync(records_fh.fileno())
                os.fsync(events_fh.fileno())
            stats["events"] += 1
            stats["reward_rows"] += len(event_rows)
            stats["score_rows_generated"] += len(score_rows)

    diagnostics = {
        "run_version": RUN_VERSION,
        "run_fingerprint": run_fingerprint,
        "split": str(args.split),
        "shard": {
            "assignment": SHARD_ASSIGNMENT,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "counts": dict(stats),
        "delta_margin": _numeric_summary(delta_values),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    manifest.update({"status": "completed", "diagnostics": diagnostics, "elapsed_seconds": diagnostics["elapsed_seconds"]})
    _write_json(diagnostics_path, diagnostics)
    _write_json(manifest_path, manifest)
    print(f"Wrote MREC reward rows: {records_path}")
    print(f"Wrote raw teacher score cache: {raw_scores_path}")
    print(f"Reward rows: {stats.get('reward_rows', 0)}")
    return 0


def _build_event_reward_rows(
    row: Mapping[str, Any],
    *,
    sample: Any,
    args: argparse.Namespace,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    scorer: LLMVerifierScorer,
    score_cache: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if str(args.rollin_policy) == ROLLIN_POLICY_VERIFIER_UTILITY_GREEDY:
        return _build_verifier_utility_greedy_event_rows(
            row,
            sample=sample,
            args=args,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            scorer=scorer,
            score_cache=score_cache,
            checkpoint=checkpoint,
        )

    rollin_trace = _build_rollin_trace(row, args=args)
    candidates = [dict(candidate) for candidate in rollin_trace.get("candidate_pool") or []]
    claim_atoms = [dict(atom) for atom in rollin_trace.get("claim_atoms") or []]
    atom_by_id = _atom_by_id(claim_atoms)
    initial_atom_states = {str(key): str(value) for key, value in (rollin_trace.get("atom_states_initial") or {}).items()}
    rollin_steps = [dict(step) for step in rollin_trace.get("mrec_steps") or [] if isinstance(step, Mapping)]
    selected_order = [int(step.get("selector_candidate_idx", -1)) for step in rollin_steps]
    selected_order = [idx for idx in selected_order if 0 <= idx < len(candidates)]
    max_prefixes = min(int(args.rollout_steps), len(selected_order)) + 1

    reward_rows: list[dict[str, Any]] = []
    generated_score_rows: list[dict[str, Any]] = []
    event_summary: dict[str, Any] = {
        "event_id": str(row.get("event_id") or ""),
        "gold_label": str(sample.label),
        "n_candidates": len(candidates),
        "rollin_selected_indices": selected_order,
        "step_summaries": [],
    }

    for step in range(max_prefixes):
        prefix_indices = selected_order[:step]
        atom_states = _atom_states_after_prefix(initial_atom_states, rollin_steps[:step])
        soft_state = hard_state_to_soft_state(atom_states)
        prefix_steps = rollin_steps[:step]
        remaining = [idx for idx in range(len(candidates)) if idx not in set(prefix_indices)]
        if not remaining:
            break

        base_request = _score_request_for_indices(
            row,
            sample=sample,
            candidates=candidates,
            claim_atoms=claim_atoms,
            selected_indices=prefix_indices,
            mrec_steps=prefix_steps,
            candidate_idx=None,
            role="base",
            step=step,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            checkpoint=checkpoint,
        )
        after_requests: list[VerifierScoreRequest] = []
        candidate_steps: dict[int, dict[str, Any]] = {}
        candidate_features: dict[int, dict[str, float]] = {}
        for idx in remaining:
            candidate = candidates[idx]
            candidate_step = _candidate_step(
                candidate,
                candidate_idx=idx,
                step=len(prefix_steps) + 1,
                claim_atoms=claim_atoms,
                atom_by_id=atom_by_id,
                atom_states=atom_states,
                cue_policy=str(args.cue_policy),
            )
            candidate_steps[idx] = candidate_step
            candidate_features[idx] = extract_marginal_features(
                candidate,
                selected_steps=prefix_steps,
                soft_state=soft_state,
                token_budget=None,
                pool_max_token_cost=max([int(item.get("mrec_token_cost") or 0) for item in candidates] or [1]),
            )
            after_requests.append(
                _score_request_for_indices(
                    row,
                    sample=sample,
                    candidates=candidates,
                    claim_atoms=claim_atoms,
                    selected_indices=[*prefix_indices, idx],
                    mrec_steps=[*prefix_steps, candidate_step],
                    candidate_idx=idx,
                    role="after",
                    step=step,
                    tokenizer=tokenizer,
                    prompt_cfg=prompt_cfg,
                    checkpoint=checkpoint,
                )
            )

        scored, new_score_rows = _score_requests([base_request, *after_requests], scorer=scorer, score_cache=score_cache)
        generated_score_rows.extend(new_score_rows)
        base_score = scored[base_request.cache_key]
        step_deltas: list[float] = []
        for request in after_requests:
            after_score = scored[request.cache_key]
            idx = int(request.metadata.get("candidate_idx"))
            candidate = candidates[idx]
            delta_margin = float(after_score.get("margin", 0.0) - base_score.get("margin", 0.0))
            step_deltas.append(delta_margin)
            reward_rows.append(
                {
                    "event_id": str(row.get("event_id") or ""),
                    "split": str(args.split),
                    "step": int(step),
                    "prefix_indices": [int(item) for item in prefix_indices],
                    "prefix_size": int(len(prefix_indices)),
                    "candidate_idx": int(idx),
                    "selector_candidate_idx": int(idx),
                    "candidate_uid": str(candidate.get("candidate_uid") or ""),
                    "candidate_key": str(candidate.get("candidate_key") or ""),
                    "evidence_id": str(candidate.get("evidence_id") or ""),
                    "gold_label": str(sample.label),
                    "base_margin": float(base_score.get("margin", 0.0)),
                    "after_margin": float(after_score.get("margin", 0.0)),
                    "delta_margin": float(delta_margin),
                    "base_pred_label": str(base_score.get("pred_label") or ""),
                    "after_pred_label": str(after_score.get("pred_label") or ""),
                    "prediction_changed": bool(base_score.get("pred_label") != after_score.get("pred_label")),
                    "mrec_features": candidate_features[idx],
                    "token_cost": int(candidate.get("mrec_token_cost") or 0),
                    "prefix_token_cost": int(sum(int(candidates[prefix_idx].get("mrec_token_cost") or 0) for prefix_idx in prefix_indices)),
                    "candidate_text_preview": str(candidate.get("text") or candidate.get("evidence_text") or "")[:240],
                    "teacher_model": str(checkpoint.get("base_model_name_or_path") or ""),
                    "teacher_checkpoint": str(checkpoint.get("checkpoint_dir") or ""),
                    "prompt_style": str(args.trace_prompt_style),
                    "reward_source": "ministral3_proxy_top5_delta_margin",
                }
            )
        event_summary["step_summaries"].append(
            {
                "step": int(step),
                "prefix_size": int(len(prefix_indices)),
                "remaining": int(len(remaining)),
                "delta_margin": _numeric_summary(step_deltas),
            }
        )
    event_summary["n_reward_rows"] = len(reward_rows)
    return reward_rows, event_summary, generated_score_rows


def _build_verifier_utility_greedy_event_rows(
    row: Mapping[str, Any],
    *,
    sample: Any,
    args: argparse.Namespace,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    scorer: LLMVerifierScorer,
    score_cache: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if str(args.split) == "test":
        raise ValueError("verifier_utility_greedy is forbidden on test because it requires the gold label.")

    candidates = _candidate_pool(row, candidate_top_n=int(args.candidate_top_n), token_cost_fn=None)
    if not candidates:
        raise ValueError(f"event_id={row.get('event_id')!r} has no candidates for verifier utility rollout.")
    claim_atoms = _claim_atoms(row)
    atom_by_id = _atom_by_id(claim_atoms)
    atom_states = build_initial_atom_states(claim_atoms)
    candidate_pool_fingerprint = _candidate_pool_fingerprint(candidates)
    selected_order: list[int] = []
    selected_steps: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    generated_score_rows: list[dict[str, Any]] = []
    event_summary: dict[str, Any] = {
        "event_id": str(row.get("event_id") or ""),
        "split": str(args.split),
        "gold_label": str(sample.label),
        "n_candidates": len(candidates),
        "rollin_policy": ROLLIN_POLICY_VERIFIER_UTILITY_GREEDY,
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
        "run_fingerprint": str(checkpoint.get("run_fingerprint") or ""),
        "step_summaries": [],
    }

    rollout_steps = min(int(args.rollout_steps), len(candidates))
    for step in range(rollout_steps):
        prefix_indices = list(selected_order)
        prefix_steps = [dict(item) for item in selected_steps]
        soft_state = hard_state_to_soft_state(atom_states)
        selected_set = set(prefix_indices)
        remaining = [idx for idx in range(len(candidates)) if idx not in selected_set]
        if not remaining:
            break

        base_request = _score_request_for_indices(
            row,
            sample=sample,
            candidates=candidates,
            claim_atoms=claim_atoms,
            selected_indices=prefix_indices,
            mrec_steps=prefix_steps,
            candidate_idx=None,
            role="base",
            step=step,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            checkpoint=checkpoint,
        )
        after_requests: list[VerifierScoreRequest] = []
        candidate_steps: dict[int, dict[str, Any]] = {}
        candidate_features: dict[int, dict[str, float]] = {}
        for idx in remaining:
            candidate = candidates[idx]
            candidate_step = _candidate_step(
                candidate,
                candidate_idx=idx,
                step=step + 1,
                claim_atoms=claim_atoms,
                atom_by_id=atom_by_id,
                atom_states=atom_states,
                cue_policy=str(args.cue_policy),
            )
            candidate_steps[idx] = candidate_step
            candidate_features[idx] = extract_marginal_features(
                candidate,
                selected_steps=prefix_steps,
                soft_state=soft_state,
                token_budget=None,
                pool_max_token_cost=max([int(item.get("mrec_token_cost") or 0) for item in candidates] or [1]),
            )
            after_requests.append(
                _score_request_for_indices(
                    row,
                    sample=sample,
                    candidates=candidates,
                    claim_atoms=claim_atoms,
                    selected_indices=[*prefix_indices, idx],
                    mrec_steps=[*prefix_steps, candidate_step],
                    candidate_idx=idx,
                    role="after",
                    step=step,
                    tokenizer=tokenizer,
                    prompt_cfg=prompt_cfg,
                    checkpoint=checkpoint,
                )
            )

        requests = [base_request, *after_requests]
        scored, new_score_rows = _score_requests(
            requests,
            scorer=scorer,
            score_cache=score_cache,
            strict=True,
        )
        generated_score_rows.extend(new_score_rows)
        base_score = scored[base_request.cache_key]
        delta_by_idx: dict[int, float] = {}
        after_score_by_idx: dict[int, dict[str, Any]] = {}
        for request in after_requests:
            idx = int(request.metadata["candidate_idx"])
            after_score = scored[request.cache_key]
            delta = float(after_score["margin"]) - float(base_score["margin"])
            if not math.isfinite(delta):
                raise RuntimeError(
                    f"Non-finite verifier utility for event_id={row.get('event_id')}, step={step}, candidate_idx={idx}."
                )
            delta_by_idx[idx] = delta
            after_score_by_idx[idx] = after_score
        if set(delta_by_idx) != set(remaining):
            raise RuntimeError(
                f"Incomplete verifier utility coverage for event_id={row.get('event_id')}, step={step}: "
                f"expected={remaining}, scored={sorted(delta_by_idx)}."
            )

        winner_idx = min(remaining, key=lambda idx: (-delta_by_idx[idx], idx))
        step_rows: list[dict[str, Any]] = []
        for idx in remaining:
            candidate = candidates[idx]
            after_score = after_score_by_idx[idx]
            step_rows.append(
                {
                    "event_id": str(row.get("event_id") or ""),
                    "split": str(args.split),
                    "step": int(step),
                    "prefix_indices": [int(item) for item in prefix_indices],
                    "prefix_size": int(len(prefix_indices)),
                    "candidate_idx": int(idx),
                    "selector_candidate_idx": int(idx),
                    "candidate_uid": str(candidate.get("candidate_uid") or ""),
                    "candidate_key": str(candidate.get("candidate_key") or ""),
                    "evidence_id": str(candidate.get("evidence_id") or ""),
                    "gold_label": str(sample.label),
                    "base_margin": float(base_score["margin"]),
                    "after_margin": float(after_score["margin"]),
                    "delta_margin": float(delta_by_idx[idx]),
                    "base_pred_label": str(base_score.get("pred_label") or ""),
                    "after_pred_label": str(after_score.get("pred_label") or ""),
                    "prediction_changed": bool(base_score.get("pred_label") != after_score.get("pred_label")),
                    "utility_selected": bool(idx == winner_idx),
                    "mrec_features": candidate_features[idx],
                    "token_cost": int(candidate.get("mrec_token_cost") or 0),
                    "prefix_token_cost": int(
                        sum(int(candidates[prefix_idx].get("mrec_token_cost") or 0) for prefix_idx in prefix_indices)
                    ),
                    "candidate_text_preview": str(candidate.get("text") or candidate.get("evidence_text") or "")[:240],
                    "teacher_model": str(checkpoint.get("base_model_name_or_path") or ""),
                    "teacher_checkpoint": str(checkpoint.get("checkpoint_dir") or ""),
                    "teacher_fingerprint": str(checkpoint.get("teacher_fingerprint") or ""),
                    "scoring_fingerprint": str(checkpoint.get("scoring_fingerprint") or ""),
                    "candidate_pool_fingerprint": candidate_pool_fingerprint,
                    "run_fingerprint": str(checkpoint.get("run_fingerprint") or ""),
                    "prompt_style": str(args.trace_prompt_style),
                    "rollin_policy": ROLLIN_POLICY_VERIFIER_UTILITY_GREEDY,
                    "reward_source": "verifier_utility_delta_margin",
                }
            )
        reward_rows.extend(step_rows)
        selected_order.append(winner_idx)
        winner_step = candidate_steps[winner_idx]
        selected_steps.append(winner_step)
        atom_id = str(winner_step.get("atom_id") or "")
        state_after = str(winner_step.get("state_after") or "")
        if atom_id and state_after:
            atom_states[atom_id] = state_after
        event_summary["step_summaries"].append(
            {
                "step": int(step),
                "prefix_indices": [int(item) for item in prefix_indices],
                "remaining": int(len(remaining)),
                "scored_candidate_indices": [int(item) for item in remaining],
                "coverage_complete": True,
                "selected_candidate_idx": int(winner_idx),
                "selected_delta_margin": float(delta_by_idx[winner_idx]),
                "delta_margin": _numeric_summary(list(delta_by_idx.values())),
            }
        )

    event_summary["rollin_selected_indices"] = [int(item) for item in selected_order]
    event_summary["n_reward_rows"] = len(reward_rows)
    event_summary["expected_reward_rows"] = sum(
        len(candidates) - step for step in range(len(selected_order))
    )
    event_summary["coverage_complete"] = bool(
        event_summary["n_reward_rows"] == event_summary["expected_reward_rows"]
    )
    if not event_summary["coverage_complete"]:
        raise RuntimeError(
            f"Incomplete event coverage for event_id={row.get('event_id')}: "
            f"expected={event_summary['expected_reward_rows']}, actual={event_summary['n_reward_rows']}."
        )
    return reward_rows, event_summary, generated_score_rows


def _build_rollin_trace(row: Mapping[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    params = MRECSelectorParams(
        candidate_top_n=int(args.candidate_top_n),
        max_steps=int(args.rollout_steps),
        min_steps=0,
        target_resolved_rate=float(args.target_resolved_rate),
        cue_policy=str(args.cue_policy),
        selection_policy=str(args.rollin_policy),
        weight_file=str(args.rollin_weight_file or ""),
        stop_threshold=float(args.rollin_stop_threshold),
        selector_name=f"mrec_reward_cache_rollin_{args.rollin_policy}",
    )
    from fact_checking.selectors.minimal_resolving_chain import build_mrec_trace_row

    return build_mrec_trace_row(row, params=params)


def _candidate_step(
    candidate: Mapping[str, Any],
    *,
    candidate_idx: int,
    step: int,
    claim_atoms: list[dict[str, Any]],
    atom_by_id: dict[str, dict[str, Any]],
    atom_states: dict[str, str],
    cue_policy: str,
) -> dict[str, Any]:
    evaluation = _evaluate_candidate_transition(
        candidate,
        atom_states=atom_states,
        atom_by_id=atom_by_id,
        cue_policy=cue_policy,
    )
    if not evaluation:
        evaluation = _fallback_candidate_evaluation(
            candidate,
            claim_atoms=claim_atoms,
            atom_states=atom_states,
            cue_policy=cue_policy,
        )
    step_candidate = evaluation.get("step_candidate")
    if not isinstance(step_candidate, Mapping):
        step_candidate = candidate
    token_cost = int(candidate.get("mrec_token_cost") or 0)
    out = build_mrec_step(
        step=int(step),
        candidate=step_candidate,
        atom_id=str(evaluation.get("atom_id") or ""),
        atom_text=str(evaluation.get("atom_text") or ""),
        state_before=str(evaluation.get("state_before") or ""),
        state_after=str(evaluation.get("state_after") or ""),
        operation=str(evaluation.get("operation") or ""),
        cue_text=str(evaluation.get("cue_text") or ""),
        cue_source=str(evaluation.get("cue_source") or ""),
        transition_reason=str(evaluation.get("transition_reason") or ""),
        token_cost=token_cost,
    )
    out["selector_candidate_idx"] = int(candidate_idx)
    _copy_step_metadata(out, candidate)
    return out


def _score_request_for_indices(
    source_row: Mapping[str, Any],
    *,
    sample: Any,
    candidates: list[dict[str, Any]],
    claim_atoms: list[dict[str, Any]],
    selected_indices: list[int],
    mrec_steps: list[dict[str, Any]],
    candidate_idx: int | None,
    role: str,
    step: int,
    tokenizer: Any,
    prompt_cfg: dict[str, Any],
    checkpoint: dict[str, Any],
) -> VerifierScoreRequest:
    from fact_checking.selectors.verifier_scorer import VerifierScoreRequest

    selected_candidates = [dict(candidates[idx]) for idx in selected_indices if 0 <= int(idx) < len(candidates)]
    candidate_pool_fingerprint = _candidate_pool_fingerprint(candidates)
    prompt_fingerprint = stable_fingerprint(prompt_cfg, length=16)
    trace = {
        "event_id": str(source_row.get("event_id") or ""),
        "claim": str(sample.claim),
        "gold_label": str(sample.label),
        "claim_atoms": claim_atoms,
        "candidate_pool": candidates,
        "selected_indices": [int(idx) for idx in selected_indices],
        "selected_candidates": selected_candidates,
        "mrec_steps": [dict(item) for item in mrec_steps],
    }
    claim, rendered_candidates, _mrec_payload = _apply_mrec_prompt_fields(
        claim=str(sample.claim),
        candidates=selected_candidates,
        trace=trace,
    )
    retrieval_row = {
        "event_id": str(sample.event_id),
        "claim": claim,
        "label": str(sample.label),
        "label_schema": str(prompt_cfg.get("label_schema") or ""),
        "explain": str(getattr(sample, "explain", "") or ""),
        "candidates": rendered_candidates,
    }
    prompt_row = _build_training_row(retrieval_row, tokenizer, prompt_cfg)
    cache_key = _cache_key(
        {
            "run_version": RUN_VERSION,
            "event_id": str(source_row.get("event_id") or ""),
            "selected_indices": [int(idx) for idx in selected_indices],
            "teacher_fingerprint": checkpoint["teacher_fingerprint"],
            "scoring_fingerprint": str(checkpoint.get("scoring_fingerprint") or ""),
            "prompt_fingerprint": prompt_fingerprint,
            "candidate_pool_fingerprint": candidate_pool_fingerprint,
            "run_fingerprint": str(checkpoint.get("run_fingerprint") or ""),
        }
    )
    return VerifierScoreRequest(
        prompt_row=prompt_row,
        gold_label=str(sample.label),
        cache_key=cache_key,
        event_id=str(source_row.get("event_id") or ""),
        claim=str(sample.claim),
        evidence_set_hash=stable_fingerprint({"selected_indices": selected_indices}, length=16),
        scored_candidate_keys=[str(candidates[idx].get("candidate_key") or candidates[idx].get("candidate_uid") or idx) for idx in selected_indices],
        metadata={
            "role": role,
            "step": int(step),
            "candidate_idx": int(candidate_idx) if candidate_idx is not None else None,
            "selected_indices": [int(idx) for idx in selected_indices],
            "teacher_fingerprint": str(checkpoint.get("teacher_fingerprint") or ""),
            "scoring_fingerprint": str(checkpoint.get("scoring_fingerprint") or ""),
            "prompt_fingerprint": prompt_fingerprint,
            "candidate_pool_fingerprint": candidate_pool_fingerprint,
            "run_fingerprint": str(checkpoint.get("run_fingerprint") or ""),
        },
    )


def _score_requests(
    requests: list[VerifierScoreRequest],
    *,
    scorer: LLMVerifierScorer,
    score_cache: dict[str, dict[str, Any]],
    strict: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if strict and len({request.cache_key for request in requests}) != len(requests):
        raise RuntimeError("Verifier score request cache keys are not unique within the scoring step.")
    out: dict[str, dict[str, Any]] = {}
    missing: list[VerifierScoreRequest] = []
    for request in requests:
        cached = score_cache.get(request.cache_key)
        if cached is not None:
            if strict:
                _validate_score_row(cached, request=request)
            out[request.cache_key] = dict(cached)
        else:
            missing.append(request)
    new_rows: list[dict[str, Any]] = []
    if missing:
        scores = scorer.score_batch(missing)
        if strict and len(scores) != len(missing):
            raise RuntimeError(
                f"Verifier scorer returned {len(scores)} scores for {len(missing)} requests."
            )
        for request, score in zip(missing, scores):
            row = {
                "cache_key": request.cache_key,
                "status": "completed",
                "event_id": request.event_id,
                "gold_label": request.gold_label,
                "metadata": dict(request.metadata or {}),
                **dict(score),
            }
            if strict:
                _validate_score_row(row, request=request)
            score_cache[request.cache_key] = row
            out[request.cache_key] = row
            new_rows.append(row)
    if strict:
        missing_keys = [request.cache_key for request in requests if request.cache_key not in out]
        if missing_keys:
            raise RuntimeError(f"Verifier score coverage is incomplete; missing cache keys: {missing_keys}")
    return out, new_rows


def _validate_score_row(row: Mapping[str, Any], *, request: VerifierScoreRequest) -> None:
    if str(row.get("cache_key") or "") != str(request.cache_key):
        raise RuntimeError(f"Verifier score cache key mismatch for event_id={request.event_id}.")
    if str(row.get("status") or "") != "completed":
        raise RuntimeError(f"Verifier score is not completed for event_id={request.event_id}.")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"Verifier score metadata is missing for event_id={request.event_id}.")
    for key in (
        "teacher_fingerprint",
        "scoring_fingerprint",
        "prompt_fingerprint",
        "candidate_pool_fingerprint",
        "run_fingerprint",
    ):
        if str(metadata.get(key) or "") != str(request.metadata.get(key) or ""):
            raise RuntimeError(
                f"Verifier score {key} mismatch for event_id={request.event_id}: "
                f"cached={metadata.get(key)!r}, expected={request.metadata.get(key)!r}."
            )
    margin = row.get("margin")
    if not isinstance(margin, (int, float)) or not math.isfinite(float(margin)):
        raise RuntimeError(f"Verifier score margin is missing or non-finite for event_id={request.event_id}.")
    if not str(row.get("pred_label") or ""):
        raise RuntimeError(f"Verifier predicted label is missing for event_id={request.event_id}.")
    label_logprobs = row.get("label_logprobs")
    if not isinstance(label_logprobs, Mapping) or not label_logprobs:
        raise RuntimeError(f"Verifier label logprobs are missing for event_id={request.event_id}.")


def _atom_states_after_prefix(initial: Mapping[str, str], prefix_steps: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    atom_states = {str(key): str(value) for key, value in initial.items()}
    for step in prefix_steps:
        atom_id = str(step.get("atom_id") or "")
        state_after = str(step.get("state_after") or "")
        if atom_id and state_after:
            atom_states[atom_id] = state_after
    return atom_states


def _teacher_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.teacher_run_dir)
    checkpoint_dir = run_dir / str(args.teacher_checkpoint)
    adapter_config = checkpoint_dir / "adapter_config.json"
    adapter_model = checkpoint_dir / "adapter_model.safetensors"
    missing = [str(path) for path in (adapter_config, adapter_model) if not path.exists()]
    if missing:
        raise FileNotFoundError("Teacher checkpoint is incomplete:\n" + "\n".join(f"- {item}" for item in missing))
    label_token_ids = load_label_token_ids(
        run_dir,
        label_prefix=str(args.label_prefix),
        label_schema=str(args.label_schema),
    )
    with adapter_config.open(encoding="utf-8") as handle:
        adapter_cfg = json.load(handle)
    base_model = str(args.base_model or adapter_cfg.get("base_model_name_or_path") or DEFAULT_BASE_MODEL)
    teacher_fingerprint = stable_fingerprint(
        {
            "run_dir": str(run_dir),
            "checkpoint": str(args.teacher_checkpoint),
            "adapter_sha256": sha256_file(adapter_model),
            "base_model": base_model,
            "label_prefix": str(args.label_prefix),
            "label_token_ids": label_token_ids,
        },
        length=24,
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(args.teacher_checkpoint),
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config_path": str(adapter_config),
        "adapter_model_path": str(adapter_model),
        "adapter_sha256": sha256_file(adapter_model),
        "base_model_name_or_path": base_model,
        "label_prefix": str(args.label_prefix),
        "label_token_ids": label_token_ids,
        "teacher_fingerprint": teacher_fingerprint,
    }


class TransformersVerifierScorer:
    backend_name = "transformers"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        label_token_ids: Mapping[str, int],
        label_prefix: str,
        prompt_batch_size: int,
        label_schema: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._label_token_ids = {str(key): int(value) for key, value in dict(label_token_ids).items()}
        self._label_prefix = str(label_prefix)
        self._prompt_batch_size = max(int(prompt_batch_size), 1)
        self._label_schema = str(label_schema)
        self._prefix_ids = tokenizer(self._label_prefix, add_special_tokens=False, truncation=False)["input_ids"]
        if not self._prefix_ids:
            raise ValueError(f"label_prefix={self._label_prefix!r} produced no tokens.")
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    def score_batch(self, requests: list[VerifierScoreRequest]) -> list[dict[str, Any]]:
        if not requests:
            return []

        import torch
        from fact_checking.selectors.verifier_scorer import compute_score_from_logprobs

        label_inputs: list[tuple[int, str, list[int], int]] = []
        for request_idx, request in enumerate(requests):
            for letter, input_ids in self._label_input_ids(request):
                label_inputs.append((request_idx, letter, input_ids, int(self._label_token_ids[letter])))

        label_logprobs_by_request: list[dict[str, float]] = [{} for _ in requests]
        batch_size = max(int(self._prompt_batch_size), 1)
        total_batches = (len(label_inputs) + batch_size - 1) // batch_size
        input_device = _first_parameter_device(self._model)
        self._model.eval()
        with torch.inference_mode():
            for batch_start in range(0, len(label_inputs), batch_size):
                batch_num = batch_start // batch_size + 1
                batch_items = label_inputs[batch_start : batch_start + batch_size]
                input_ids, attention_mask, lengths = _pad_label_inputs(
                    [item[2] for item in batch_items],
                    pad_token_id=int(self._tokenizer.pad_token_id),
                    device=input_device,
                )
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                for row_idx, (request_idx, letter, _ids, label_token_id) in enumerate(batch_items):
                    length = int(lengths[row_idx])
                    if length < 2:
                        raise RuntimeError("Verifier scoring prompt must contain at least two tokens.")
                    log_probs = torch.log_softmax(logits[row_idx, length - 2, :].float(), dim=-1)
                    label_logprobs_by_request[request_idx][letter] = float(log_probs[int(label_token_id)].item())
                print(
                    f"Transformers batch {batch_num}/{total_batches} | "
                    f"{min(batch_start + len(batch_items), len(label_inputs))}/{len(label_inputs)} label prompts",
                    flush=True,
                )

        results: list[dict[str, Any]] = []
        for request, label_logprobs in zip(requests, label_logprobs_by_request):
            missing = [letter for letter in self._label_token_ids if letter not in label_logprobs]
            if missing:
                raise RuntimeError(f"Missing label logprobs for event_id={request.event_id}: {missing}")
            score = compute_score_from_logprobs(
                label_logprobs,
                request.gold_label,
                label_schema=self._label_schema,
            )
            score["scoring_backend"] = self.backend_name
            results.append(score)
        return results

    def _label_input_ids(self, request: VerifierScoreRequest) -> list[tuple[str, list[int]]]:
        prompt_row = request.prompt_row
        raw_prompt_ids = prompt_row.get("prompt_input_ids")
        if isinstance(raw_prompt_ids, list):
            prompt_ids = [int(token_id) for token_id in raw_prompt_ids]
        else:
            prompt_text = str(prompt_row.get("prompt") or "").rstrip()
            if bool(prompt_row.get("prompt_add_special_tokens", False)):
                prompt_text += " "
            prompt_ids = [
                int(token_id)
                for token_id in self._tokenizer(
                    prompt_text,
                    add_special_tokens=bool(prompt_row.get("prompt_add_special_tokens", False)),
                    truncation=False,
                )["input_ids"]
            ]
        out: list[tuple[str, list[int]]] = []
        for letter in self._label_token_ids:
            out.append((letter, [*prompt_ids, *[int(token_id) for token_id in self._prefix_ids], int(self._label_token_ids[letter])]))
        return out


def _pad_label_inputs(
    rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any, list[int]]:
    import torch

    lengths = [len(row) for row in rows]
    max_len = max(lengths)
    input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        row_tensor = torch.tensor([int(token_id) for token_id in row], dtype=torch.long, device=device)
        input_ids[row_idx, : len(row)] = row_tensor
        attention_mask[row_idx, : len(row)] = 1
    return input_ids, attention_mask, lengths


def _first_parameter_device(model: Any) -> Any:
    import torch

    for parameter in model.parameters():
        return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype(raw: str) -> Any:
    import torch

    value = str(raw or "auto").strip().lower()
    if value == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported transformers dtype: {raw}")


def _init_scorer(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> Any:
    backend = str(args.scoring_backend)
    if backend == "transformers":
        return _init_transformers_scorer(args, checkpoint)
    if backend == "vllm":
        return _init_vllm_scorer(args, checkpoint)
    try:
        return _init_vllm_scorer(args, checkpoint)
    except Exception as exc:  # noqa: BLE001 - auto mode intentionally catches backend incompatibilities.
        print(
            f"[mrec-reward] vLLM scorer init failed with {type(exc).__name__}: {exc}; "
            "falling back to transformers.",
            file=sys.stderr,
            flush=True,
        )
        return _init_transformers_scorer(args, checkpoint)


def _scoring_fingerprint(args: argparse.Namespace, scorer: Any) -> str:
    backend = str(getattr(scorer, "backend_name", args.scoring_backend))
    payload: dict[str, Any] = {
        "backend": backend,
        "label_schema": str(args.label_schema),
    }
    if backend == "vllm":
        payload["vllm_lora_mode"] = str(
            getattr(scorer, "vllm_lora_mode", getattr(args, "vllm_lora_mode", "dynamic"))
        )
        payload["vllm_dtype"] = str(getattr(args, "vllm_dtype", "auto") or "auto")
        payload["vllm_tokenizer_path"] = str(getattr(args, "vllm_tokenizer_path", "") or "")
        payload["vllm_tokenizer_mode"] = str(getattr(args, "vllm_tokenizer_mode", "auto") or "auto")
        payload["vllm_config_format"] = str(getattr(args, "vllm_config_format", "") or "")
        payload["vllm_load_format"] = str(getattr(args, "vllm_load_format", "") or "")
        payload["vllm_max_model_len"] = int(getattr(args, "vllm_max_model_len", 0) or 0)
    elif backend == "transformers":
        payload["transformers_dtype"] = str(getattr(args, "transformers_dtype", "auto") or "auto")
        payload["transformers_tokenizer_path"] = str(
            getattr(args, "transformers_tokenizer_path", "") or ""
        )
    return stable_fingerprint(payload, length=16)


def _init_transformers_scorer(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> TransformersVerifierScorer:
    import torch

    from peft import PeftModel
    from sft.runtime.model_loading import load_causal_lm_compatible_model, load_compatible_tokenizer

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_model = str(checkpoint["base_model_name_or_path"])
    tokenizer_path = str(args.transformers_tokenizer_path or args.vllm_tokenizer_path or base_model)
    tokenizer = load_compatible_tokenizer(tokenizer_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(str(args.transformers_dtype)),
    }
    device = str(args.transformers_device or "auto").strip().lower()
    if device == "auto" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = load_causal_lm_compatible_model(
        base_model,
        use_mistral3_text_only=False,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(model, str(checkpoint["checkpoint_dir"]))
    if device != "auto":
        model.to(torch.device(device))
    model.eval()
    scorer = TransformersVerifierScorer(
        model=model,
        tokenizer=tokenizer,
        label_token_ids=dict(checkpoint["label_token_ids"]),
        label_prefix=str(checkpoint["label_prefix"]),
        prompt_batch_size=int(args.transformers_prompt_batch_size),
        label_schema=str(args.label_schema),
    )
    return scorer


def _merge_lora_for_vllm_scorer(
    *,
    base_model: str,
    adapter_dir: str | Path,
    tokenizer_dir: str | Path,
    dtype: str,
    cache_dir: str | Path,
    force_rebuild: bool,
) -> Path:
    from fact_checking.infer.api import _merge_lora_to_cache

    return _merge_lora_to_cache(
        base_model=base_model,
        adapter_dir=adapter_dir,
        tokenizer_dir=tokenizer_dir,
        dtype=dtype,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )


def _should_disable_vllm_image_inputs_for_text_scoring(model_path: str | Path) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    architectures = [str(item) for item in config.get("architectures") or []]
    model_type = str(config.get("model_type") or "").lower()
    architecture_blob = " ".join(architectures).lower()
    return (
        "mistral3" in architecture_blob
        or "pixtral" in architecture_blob
        or model_type in {"mistral3", "pixtral"}
    )


def _ensure_vllm_mistral3_text_architecture(model_path: str | Path) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    architectures = [str(item) for item in config.get("architectures") or []]
    if not any("Mistral3ForConditionalGeneration" == item for item in architectures):
        return False

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return False
    if text_config.get("architectures") is not None:
        return False
    text_model_type = str(text_config.get("model_type") or "").lower()
    if text_model_type not in {"mistral", "ministral", "ministral3"}:
        return False

    text_config["architectures"] = ["MistralForCausalLM"]
    try:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def _model_has_fp8_scale_weights(model_path: str | Path) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        return False

    for shard_path in Path(model_path).glob("*.safetensors"):
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                if any(
                    key.endswith(".weight_scale")
                    or key.endswith(".weight_scale_inv")
                    or key.endswith(".input_scale")
                    for key in handle.keys()
                ):
                    return True
        except Exception:  # noqa: BLE001 - best-effort compatibility metadata probe.
            continue
    return False


def _ensure_vllm_fp8_quantization_config(model_path: str | Path) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists() or not _model_has_fp8_scale_weights(model_path):
        return False
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    quant_config = {
        "activation_scheme": "dynamic",
        "quant_method": "fp8",
    }
    changed = False
    if config.get("quantization_config") is None:
        config["quantization_config"] = dict(quant_config)
        changed = True
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config.get("quantization_config") is None:
        text_config["quantization_config"] = dict(quant_config)
        changed = True
    if not changed:
        return False
    try:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def _patch_vllm_mistral_common_tokenizer_kwargs() -> None:
    from fact_checking.utils.vllm_mistral_common_patch import apply_vllm_mistral_common_tokenizer_patch

    apply_vllm_mistral_common_tokenizer_patch()


def _init_vllm_scorer(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> LLMVerifierScorer:
    from fact_checking.selectors.verifier_scorer import LLMVerifierScorer

    for lib in ("vllm", "vllm.engine", "vllm.executor", "vllm.worker"):
        import logging

        logging.getLogger(lib).setLevel(logging.WARNING)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("MREC_PATCH_VLLM_MISTRAL_COMMON_TOKENIZER", "1")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required to build the reward cache with --scoring-backend vllm.") from exc
    _patch_vllm_mistral_common_tokenizer_kwargs()

    base_model = str(checkpoint["base_model_name_or_path"])
    tokenizer_path = str(args.vllm_tokenizer_path or base_model)
    lora_mode = str(getattr(args, "vllm_lora_mode", "dynamic") or "dynamic").strip().lower()
    if lora_mode not in {"dynamic", "merged"}:
        raise ValueError(f"Unsupported vllm_lora_mode={lora_mode!r}; expected dynamic or merged.")

    lora_request = None
    llm_model_path = base_model
    llm_tokenizer_path = tokenizer_path
    dynamic_lora_kwargs: dict[str, Any] = {}
    if lora_mode == "merged":
        merged_dir = _merge_lora_for_vllm_scorer(
            base_model=base_model,
            adapter_dir=str(checkpoint["checkpoint_dir"]),
            tokenizer_dir=tokenizer_path,
            dtype=str(args.vllm_dtype),
            cache_dir=str(getattr(args, "vllm_merge_lora_cache_dir", "outputs/cache/merged_lora")),
            force_rebuild=bool(getattr(args, "vllm_merge_lora_force_rebuild", False)),
        )
        llm_model_path = str(merged_dir)
        llm_tokenizer_path = str(merged_dir)
    else:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError(
                "vLLM dynamic LoRA support is required for --vllm-lora-mode dynamic. "
                "Use --vllm-lora-mode merged to score a merged teacher checkpoint instead."
            ) from exc
        with Path(checkpoint["adapter_config_path"]).open(encoding="utf-8") as handle:
            adapter_cfg = json.load(handle)
        dynamic_lora_kwargs = {
            "enable_lora": True,
            "max_lora_rank": int(adapter_cfg.get("r", 16)),
        }
        lora_request = LoRARequest("mrec-reward-teacher", 1, str(checkpoint["checkpoint_dir"]))

    _ensure_vllm_mistral3_text_architecture(llm_model_path)
    _ensure_vllm_fp8_quantization_config(llm_model_path)
    llm_kwargs: dict[str, Any] = {
        "model": llm_model_path,
        "tokenizer": llm_tokenizer_path,
        "tensor_parallel_size": int(args.vllm_tensor_parallel_size),
        "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "dtype": str(args.vllm_dtype),
        "tokenizer_mode": str(getattr(args, "vllm_tokenizer_mode", "auto") or "auto"),
        "trust_remote_code": True,
    }
    if _should_disable_vllm_image_inputs_for_text_scoring(llm_model_path):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
    if _model_has_fp8_scale_weights(llm_model_path):
        llm_kwargs["quantization"] = "fp8"
    config_format = str(getattr(args, "vllm_config_format", "") or "").strip()
    if config_format:
        llm_kwargs["config_format"] = config_format
    load_format = str(getattr(args, "vllm_load_format", "") or "").strip()
    if load_format:
        llm_kwargs["load_format"] = load_format
    llm_kwargs.update(dynamic_lora_kwargs)
    if int(args.vllm_max_model_len) > 0:
        llm_kwargs["max_model_len"] = int(args.vllm_max_model_len)
    if bool(args.vllm_enforce_eager):
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    scorer = LLMVerifierScorer(
        llm=llm,
        sampling_params=sampling_params,
        label_token_ids=dict(checkpoint["label_token_ids"]),
        label_prefix=str(checkpoint["label_prefix"]),
        lora_request=lora_request,
        prompt_batch_size=int(args.vllm_prompt_batch_size),
        label_schema=str(args.label_schema),
    )
    scorer.backend_name = "vllm"
    scorer.vllm_lora_mode = lora_mode
    return scorer



def _prompt_cfg(args: argparse.Namespace) -> dict[str, Any]:
    prompt_cfg = {
        "model_name_or_path": str(args.prompt_model_name_or_path or args.base_model or DEFAULT_BASE_MODEL),
        "auto_length": True,
        "max_length": int(args.prompt_max_length),
        "output_mode": "label_only",
        "label_format": "letter",
        "system_prompt": None,
        "label_schema": str(args.label_schema),
    }
    return _prompt_cfg_for_trace_style(
        prompt_cfg,
        trace_prompt_style=str(args.trace_prompt_style),
        label_schema=str(args.label_schema),
    )


def _is_strict_utility_mode(args: argparse.Namespace) -> bool:
    return str(args.rollin_policy) == ROLLIN_POLICY_VERIFIER_UTILITY_GREEDY


def _validate_execution_args(args: argparse.Namespace) -> None:
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}.")
    if int(args.rollout_steps) < 1 and _is_strict_utility_mode(args):
        raise ValueError("rollout_steps must be >= 1 for verifier_utility_greedy.")
    if str(args.split) == "test" and _is_strict_utility_mode(args):
        raise ValueError("verifier_utility_greedy is forbidden on test because it requires the gold label.")


def _artifact_key(split: str, *, num_shards: int, shard_index: int) -> str:
    if int(num_shards) == 1:
        return str(split)
    return f"{split}.shard-{int(shard_index):05d}-of-{int(num_shards):05d}"


def _index_raw_samples(samples: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sample in samples:
        event_id = str(getattr(sample, "event_id", "") or "")
        if not event_id:
            raise ValueError("Raw split contains a sample without event_id.")
        if event_id in out:
            raise ValueError(f"Raw split contains duplicate event_id={event_id!r}.")
        out[event_id] = sample
    return out


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_uid", "candidate_key", "canonical_text", "text", "evidence_text"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    raise ValueError("Candidate has no stable identity field.")


def _candidate_pool_fingerprint(candidates: Sequence[Mapping[str, Any]]) -> str:
    return stable_fingerprint(
        {"candidates": [dict(candidate) for candidate in candidates]},
        length=24,
    )


def _event_ids_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return stable_fingerprint(
        {"event_ids": [str(row.get("event_id") or "") for row in rows]},
        length=24,
    )


def _shard_for_event_id(event_id: str, *, num_shards: int) -> int:
    digest = hashlib.sha256(str(event_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(num_shards)


def _select_shard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if _shard_for_event_id(str(row.get("event_id") or ""), num_shards=int(num_shards)) == int(shard_index)
    ]
    if not selected:
        raise ValueError(
            f"Shard {shard_index}/{num_shards} contains no rows; reduce num_shards or choose a populated shard."
        )
    return selected


def _event_score_counts(candidate_count: int, *, rollout_steps: int) -> dict[str, int]:
    steps = min(max(int(rollout_steps), 0), max(int(candidate_count), 0))
    reward_rows = sum(int(candidate_count) - step for step in range(steps))
    return {
        "rollout_steps": steps,
        "reward_rows": reward_rows,
        "logical_score_requests": reward_rows + steps,
        "unique_teacher_scores": reward_rows + (1 if steps > 0 else 0),
    }


def _validate_input_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_by_event: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Utility-cache input contains no rows after sample_limit.")
    seen_event_ids: set[str] = set()
    candidate_counts: list[int] = []
    total_reward_rows = 0
    total_logical_requests = 0
    total_unique_scores = 0
    for row_index, row in enumerate(rows):
        event_id = str(row.get("event_id") or "")
        if not event_id:
            raise ValueError(f"Input row {row_index} has no event_id.")
        if event_id in seen_event_ids:
            raise ValueError(f"Input contains duplicate event_id={event_id!r}.")
        seen_event_ids.add(event_id)
        sample = raw_by_event.get(event_id)
        if sample is None:
            raise ValueError(f"Input event_id={event_id!r} is missing from the raw split.")
        raw_label = str(getattr(sample, "label", "") or "").strip().lower()
        if not raw_label:
            raise ValueError(f"Raw sample event_id={event_id!r} has no gold label.")
        for label_key in ("gold_label", "label"):
            feature_label = str(row.get(label_key) or "").strip().lower()
            if feature_label and feature_label != raw_label:
                raise ValueError(
                    f"Label mismatch for event_id={event_id!r}: raw={raw_label!r}, "
                    f"{label_key}={feature_label!r}."
                )
        candidates = _candidate_pool(
            row,
            candidate_top_n=int(args.candidate_top_n),
            token_cost_fn=None,
        )
        if not candidates:
            raise ValueError(f"Input event_id={event_id!r} has no candidates after candidate_top_n.")
        identities = [_candidate_identity(candidate) for candidate in candidates]
        duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"Input event_id={event_id!r} has duplicate candidates after candidate_top_n: {duplicates[:3]}."
            )
        candidate_counts.append(len(candidates))
        counts = _event_score_counts(len(candidates), rollout_steps=int(args.rollout_steps))
        total_reward_rows += counts["reward_rows"]
        total_logical_requests += counts["logical_score_requests"]
        total_unique_scores += counts["unique_teacher_scores"]
    return {
        "status": "validated",
        "split": str(args.split),
        "dataset": str(args.dataset),
        "label_schema": str(args.label_schema),
        "rollin_policy": str(args.rollin_policy),
        "candidate_top_n": int(args.candidate_top_n),
        "rollout_steps": int(args.rollout_steps),
        "n_validated_rows": len(rows),
        "event_ids_fingerprint": _event_ids_fingerprint(rows),
        "candidate_count": {
            "min": min(candidate_counts),
            "max": max(candidate_counts),
            "mean": float(sum(candidate_counts) / len(candidate_counts)),
            "total": int(sum(candidate_counts)),
        },
        "estimated_reward_rows": int(total_reward_rows),
        "estimated_logical_score_requests": int(total_logical_requests),
        "estimated_unique_teacher_scores": int(total_unique_scores),
        "checks": {
            "event_ids_nonempty_unique": True,
            "raw_rows_complete": True,
            "gold_labels_nonempty_consistent": True,
            "candidate_pools_nonempty": True,
            "candidate_identities_nonempty_unique": True,
            "test_gold_utility_forbidden": bool(str(args.split) != "test"),
        },
    }


def _shard_plan(rows: Sequence[Mapping[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for shard_index in range(int(args.num_shards)):
        shard_rows = [
            row
            for row in rows
            if _shard_for_event_id(
                str(row.get("event_id") or ""),
                num_shards=int(args.num_shards),
            )
            == shard_index
        ]
        reward_rows = 0
        logical_requests = 0
        unique_scores = 0
        for row in shard_rows:
            candidate_count = len(
                _candidate_pool(row, candidate_top_n=int(args.candidate_top_n), token_cost_fn=None)
            )
            counts = _event_score_counts(candidate_count, rollout_steps=int(args.rollout_steps))
            reward_rows += counts["reward_rows"]
            logical_requests += counts["logical_score_requests"]
            unique_scores += counts["unique_teacher_scores"]
        plan.append(
            {
                "shard_index": int(shard_index),
                "n_events": len(shard_rows),
                "event_ids_fingerprint": _event_ids_fingerprint(shard_rows),
                "estimated_reward_rows": int(reward_rows),
                "estimated_logical_score_requests": int(logical_requests),
                "estimated_unique_teacher_scores": int(unique_scores),
            }
        )
    return plan


def _run_contract(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    prompt_cfg: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    shard = dict(validation.get("shard") or {})
    return {
        "run_version": RUN_VERSION,
        "input_sha256": str(validation.get("input_sha256") or ""),
        "raw_sha256": str(validation.get("raw_sha256") or ""),
        "split": str(args.split),
        "dataset": str(args.dataset),
        "label_schema": str(args.label_schema),
        "candidate_top_n": int(args.candidate_top_n),
        "rollout_steps": int(args.rollout_steps),
        "rollin_policy": str(args.rollin_policy),
        "cue_policy": str(args.cue_policy),
        "trace_prompt_style": str(args.trace_prompt_style),
        "prompt_max_length": int(args.prompt_max_length),
        "sample_limit": int(args.sample_limit),
        "shard": {
            "assignment": SHARD_ASSIGNMENT,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "event_ids_fingerprint": str(shard.get("event_ids_fingerprint") or ""),
        },
        "teacher_fingerprint": str(checkpoint.get("teacher_fingerprint") or ""),
        "scoring_fingerprint": str(checkpoint.get("scoring_fingerprint") or ""),
        "prompt_fingerprint": stable_fingerprint(dict(prompt_cfg), length=16),
    }


def _validate_resume_manifest(path: Path, *, expected_run_fingerprint: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read resume manifest {path}: {exc}") from exc
    actual = str(payload.get("run_fingerprint") or "")
    if not actual or actual != str(expected_run_fingerprint):
        raise RuntimeError(
            f"Resume manifest fingerprint mismatch for {path}: actual={actual!r}, "
            f"expected={expected_run_fingerprint!r}. Use a new output directory or --no-resume."
        )


def _load_score_cache(
    path: Path,
    *,
    expected_run_fingerprint: str = "",
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if expected_run_fingerprint:
                    raise RuntimeError(f"Invalid JSON in strict score cache {path}:{line_number}.") from exc
                continue
            key = str(row.get("cache_key") or "")
            if expected_run_fingerprint:
                metadata = row.get("metadata")
                actual = str(metadata.get("run_fingerprint") or "") if isinstance(metadata, Mapping) else ""
                if actual != str(expected_run_fingerprint):
                    raise RuntimeError(
                        f"Score-cache fingerprint mismatch at {path}:{line_number}: "
                        f"actual={actual!r}, expected={expected_run_fingerprint!r}."
                    )
                if not key or row.get("status") != "completed":
                    raise RuntimeError(f"Incomplete strict score-cache row at {path}:{line_number}.")
                if key in cache:
                    raise RuntimeError(f"Duplicate cache_key={key!r} in strict score cache {path}.")
            if key and row.get("status") == "completed":
                cache[key] = row
    return cache


def _completed_event_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = str(row.get("event_id") or "")
            if event_id:
                out.add(event_id)
    return out


def _completed_utility_event_ids(
    records_path: Path,
    events_path: Path,
    *,
    expected_run_fingerprint: str,
) -> set[str]:
    if not records_path.exists() and not events_path.exists():
        return set()
    if not records_path.exists() or not events_path.exists():
        raise RuntimeError(
            "Strict verifier-utility resume requires both reward records and event summaries; "
            f"records_exists={records_path.exists()}, summaries_exists={events_path.exists()}."
        )

    summaries: dict[str, dict[str, Any]] = {}
    with events_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in event summary {events_path}:{line_number}.") from exc
            event_id = str(row.get("event_id") or "")
            if not event_id:
                raise RuntimeError(f"Missing event_id in event summary {events_path}:{line_number}.")
            if event_id in summaries:
                raise RuntimeError(f"Duplicate event summary for event_id={event_id!r} in {events_path}.")
            if str(row.get("run_fingerprint") or "") != str(expected_run_fingerprint):
                raise RuntimeError(f"Event-summary fingerprint mismatch for event_id={event_id!r}.")
            if row.get("coverage_complete") is not True:
                raise RuntimeError(f"Event summary is not coverage-complete for event_id={event_id!r}.")
            summaries[event_id] = row

    record_counts: Counter[str] = Counter()
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in reward records {records_path}:{line_number}.") from exc
            event_id = str(row.get("event_id") or "")
            if not event_id:
                raise RuntimeError(f"Missing event_id in reward record {records_path}:{line_number}.")
            if str(row.get("run_fingerprint") or "") != str(expected_run_fingerprint):
                raise RuntimeError(f"Reward-record fingerprint mismatch for event_id={event_id!r}.")
            record_counts[event_id] += 1

    incomplete_events = sorted(set(record_counts) - set(summaries))
    if incomplete_events:
        raise RuntimeError(
            "Reward records contain events without a completed summary; resume is fail-closed for: "
            f"{incomplete_events[:5]}. Use a new output directory or --no-resume."
        )
    for event_id, summary in summaries.items():
        expected = int(summary.get("n_reward_rows") or 0)
        actual = int(record_counts.get(event_id, 0))
        if actual != expected:
            raise RuntimeError(
                f"Reward-record coverage mismatch for event_id={event_id!r}: expected={expected}, actual={actual}."
            )
    return set(summaries)


def _read_jsonl(path: Path, *, sample_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if sample_limit > 0 and len(rows) >= int(sample_limit):
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows read from {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cache_key(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    return {
        "count": float(count),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
    }


if __name__ == "__main__":
    raise SystemExit(main())
