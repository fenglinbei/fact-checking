#!/usr/bin/env python3
"""Build post-hoc strict same-visible-set evidence-order controls.

This builder deliberately starts *after* prompt realization.  For every source
build row it freezes ``candidates[:evidence_count]`` byte-for-byte by stable
``candidate_uid`` and changes only their display order.  It refuses source rows
whose sole surviving evidence text was already truncated, and it never invokes
the prompt builder's automatic truncation path.

Output layout::

    OUTPUT_DIR/<arm>/build_<split>.jsonl
    OUTPUT_DIR/<arm>/strict_same_set_control_<split>.jsonl
    OUTPUT_DIR/<arm>/summary_<split>.json

The sidecar is the canonical home for order-dependent BACES display replay.
Candidate-local ``solver_role`` is assigned once in the exact frozen-set order
and then carried unchanged into every arm; display marginals and states are
recomputed for each arm independently.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence, TextIO

from fact_checking.build.prompts import build_training_row, load_prompt_tokenizer
from fact_checking.config import load_yaml
from fact_checking.selectors.baces_exact import solve_fixed_set_order
from fact_checking.selectors.baces_objective import (
    BacesEvaluation,
    BacesProblem,
    compile_feature_problem,
    evaluate_display,
    padded_auc,
)


SCHEMA_VERSION = "strict-same-set-control-v0.1"
DEFAULT_RANDOM_SEEDS = (0, 1, 2, 3, 4)
FIXED_ARMS = (
    "original",
    "baces_exact",
    "retrieval_score",
    "candidate_pool",
    "reverse",
)
RANDOM_SEED_HASH_POLICY = "sha256(seed\\0event_id)-first64be-python-random-v1"
RETRIEVAL_SCORE_POLICY = (
    "feature.hybrid_score_desc_missing_last_then_candidate_uid_asc_v1"
)
CANDIDATE_POOL_POLICY = "feature.candidates_array_order_v1"
REVERSE_POLICY = "reverse_source_final_visible_order_v1"


class StrictControlError(ValueError):
    """Raised when an input or rebuilt row violates the strict-control contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-build", required=True, help="Source build_<split>.jsonl")
    parser.add_argument("--features", required=True, help="Evidence-map feature JSONL")
    parser.add_argument("--audit", required=True, help="BACES reference audit JSONL")
    parser.add_argument("--config", required=True, help="Original experiment YAML")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANDOM_SEEDS),
        help="Event-specific random-order arms (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Build only the first N source rows; intended for contract smoke tests.",
    )
    args = parser.parse_args()
    if args.sample_limit is not None and args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("--random-seeds must not contain duplicates")
    return args


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = _load_experiment_config(config_path)
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    model_name_or_path = str(prompt_cfg.get("model_name_or_path") or "").strip()
    if not model_name_or_path:
        raise StrictControlError("config build.prompt.model_name_or_path is required")
    tokenizer = load_prompt_tokenizer(model_name_or_path)

    manifest = build_strict_same_set_controls(
        source_build_path=Path(args.source_build),
        features_path=Path(args.features),
        audit_path=Path(args.audit),
        config_path=config_path,
        split=str(args.split),
        output_dir=Path(args.output_dir),
        random_seeds=tuple(int(seed) for seed in args.random_seeds),
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        sample_limit=args.sample_limit,
    )
    print(
        f"Wrote {manifest['n_events']} events x {len(manifest['arms'])} arms "
        f"to {manifest['output_dir']}"
    )
    return 0


def build_strict_same_set_controls(
    *,
    source_build_path: Path,
    features_path: Path,
    audit_path: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    random_seeds: Sequence[int],
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Validate inputs and materialize every strict order-only arm.

    Large feature/build artifacts are handled without retaining their decoded
    candidate payloads in memory.  Feature and audit files receive lightweight
    event-id-to-byte-offset indexes, while the source build is streamed once.
    Final files are atomically promoted only after every processed row passes.
    """

    source_build_path = Path(source_build_path)
    features_path = Path(features_path)
    audit_path = Path(audit_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    seeds = tuple(int(seed) for seed in random_seeds)
    if len(set(seeds)) != len(seeds):
        raise StrictControlError("random seeds must be unique")
    if sample_limit is not None and sample_limit < 0:
        raise StrictControlError("sample_limit must be non-negative")

    feature_offsets = _index_jsonl(features_path, artifact="features")
    audit_offsets = _index_jsonl(audit_path, artifact="audit")
    arm_specs = _arm_specs(seeds)
    prompt_cfg_no_truncation = dict(prompt_cfg)
    prompt_cfg_no_truncation["auto_length"] = False
    max_length = _positive_int(
        prompt_cfg_no_truncation.get("max_length", 2048),
        context="build.prompt.max_length",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths: dict[str, dict[str, Path]] = {}
    temp_paths: list[Path] = []
    stats: dict[str, dict[str, list[int]]] = {}
    for spec in arm_specs:
        arm = str(spec["name"])
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        final_paths[arm] = {
            "build": arm_dir / f"build_{split}.jsonl",
            "sidecar": arm_dir / f"strict_same_set_control_{split}.jsonl",
            "summary": arm_dir / f"summary_{split}.json",
        }
        stats[arm] = {
            "K": [],
            "prompt_tokens": [],
            "target_tokens": [],
            "total_tokens": [],
            "display_utility": [],
            "display_T": [],
            "display_AUC": [],
            "order_regret": [],
        }

    seen_source_ids: set[str] = set()
    n_events = 0
    promoted = False
    try:
        with ExitStack() as stack:
            feature_handle = stack.enter_context(features_path.open("rb"))
            audit_handle = stack.enter_context(audit_path.open("rb"))
            writers: dict[str, dict[str, TextIO]] = {}
            for spec in arm_specs:
                arm = str(spec["name"])
                writers[arm] = {}
                for kind in ("build", "sidecar"):
                    final_path = final_paths[arm][kind]
                    temp_path = final_path.with_name(
                        f".{final_path.name}.tmp.{os.getpid()}"
                    )
                    temp_paths.append(temp_path)
                    writers[arm][kind] = stack.enter_context(
                        temp_path.open("w", encoding="utf-8")
                    )

            for line_number, source_row in _iter_jsonl(
                source_build_path, artifact="source_build"
            ):
                if sample_limit is not None and n_events >= sample_limit:
                    break
                event_id = _event_id(source_row, f"source_build:{line_number}")
                if event_id in seen_source_ids:
                    raise StrictControlError(
                        f"duplicate event_id {event_id!r} in source build"
                    )
                seen_source_ids.add(event_id)
                if event_id not in feature_offsets:
                    raise StrictControlError(
                        f"source event {event_id!r} is missing from features"
                    )
                if event_id not in audit_offsets:
                    raise StrictControlError(
                        f"source event {event_id!r} is missing from audit"
                    )
                feature_row = _read_row_at(
                    feature_handle,
                    feature_offsets[event_id],
                    artifact=f"features:{event_id}",
                )
                audit_row = _read_row_at(
                    audit_handle,
                    audit_offsets[event_id],
                    artifact=f"audit:{event_id}",
                )
                built_by_arm = _build_event_controls(
                    source_row=source_row,
                    feature_row=feature_row,
                    audit_row=audit_row,
                    split=split,
                    random_seeds=seeds,
                    tokenizer=tokenizer,
                    prompt_cfg=prompt_cfg_no_truncation,
                    max_length=max_length,
                )
                for arm, payload in built_by_arm.items():
                    build_row = payload["build_row"]
                    sidecar = payload["sidecar"]
                    writers[arm]["build"].write(_json_line(build_row))
                    writers[arm]["sidecar"].write(_json_line(sidecar))
                    _append_stats(stats[arm], build_row=build_row, sidecar=sidecar)
                n_events += 1

        if sample_limit is None:
            feature_ids = set(feature_offsets)
            audit_ids = set(audit_offsets)
            if seen_source_ids != feature_ids:
                raise StrictControlError(
                    "source/features event-id sets differ: "
                    f"source_only={sorted(seen_source_ids - feature_ids)[:10]}, "
                    f"features_only={sorted(feature_ids - seen_source_ids)[:10]}"
                )
            if seen_source_ids != audit_ids:
                raise StrictControlError(
                    "source/audit event-id sets differ: "
                    f"source_only={sorted(seen_source_ids - audit_ids)[:10]}, "
                    f"audit_only={sorted(audit_ids - seen_source_ids)[:10]}"
                )

        for spec in arm_specs:
            arm = str(spec["name"])
            for kind in ("build", "sidecar"):
                final_path = final_paths[arm][kind]
                temp_path = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
                temp_path.replace(final_path)

            summary = _arm_summary(
                arm_spec=spec,
                split=split,
                n_events=n_events,
                max_length=max_length,
                stats=stats[arm],
                source_build_path=source_build_path,
                features_path=features_path,
                audit_path=audit_path,
                config_path=config_path,
                build_path=final_paths[arm]["build"],
                sidecar_path=final_paths[arm]["sidecar"],
                sample_limit=sample_limit,
            )
            _write_json_atomic(final_paths[arm]["summary"], summary)
        promoted = True
    finally:
        if not promoted:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "n_events": n_events,
        "output_dir": str(output_dir),
        "source_build": str(source_build_path),
        "features": str(features_path),
        "audit": str(audit_path),
        "config": str(config_path),
        "sample_limit": sample_limit,
        "arms": {
            str(spec["name"]): {
                "random_seed": spec.get("random_seed"),
                **{kind: str(path) for kind, path in final_paths[str(spec["name"])].items()},
            }
            for spec in arm_specs
        },
        "contract": {
            "frozen_source_slice": "source.candidates[:source.evidence_count]",
            "auto_length": False,
            "set_text_block_fingerprints": "order_invariant_sha256_canonical_json_v1",
            "display_order_fingerprint": (
                "order_sensitive_sha256_canonical_json_event_id_and_ordered_candidate_uids_v1"
            ),
            "random_seed_hash_policy": RANDOM_SEED_HASH_POLICY,
            "retrieval_score_policy": RETRIEVAL_SCORE_POLICY,
            "candidate_pool_policy": CANDIDATE_POOL_POLICY,
            "reverse_policy": REVERSE_POLICY,
        },
    }
    _write_json_atomic(output_dir / f"manifest_{split}.json", manifest)
    return manifest


def _build_event_controls(
    *,
    source_row: Mapping[str, Any],
    feature_row: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    split: str,
    random_seeds: Sequence[int],
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
    max_length: int,
) -> dict[str, dict[str, Any]]:
    """Pure, single-event implementation used by the CLI and unit tests."""

    event_id = _event_id(source_row, "source build row")
    if _event_id(feature_row, "feature row") != event_id:
        raise StrictControlError(f"{event_id}: feature event_id mismatch")
    if _event_id(audit_row, "audit row") != event_id:
        raise StrictControlError(f"{event_id}: audit event_id mismatch")

    frozen_candidates = _freeze_source_candidates(source_row)
    original_uids = tuple(_candidate_uid(candidate) for candidate in frozen_candidates)
    _validate_audit_alignment(audit_row, original_uids=original_uids, event_id=event_id)
    problem = _compile_frozen_problem(
        feature_row=feature_row,
        audit_row=audit_row,
        frozen_candidates=frozen_candidates,
        event_id=event_id,
    )
    local_exact = solve_fixed_set_order(problem, original_uids)
    audit_exact_uids = tuple(_string_sequence(
        audit_row.get("final_same_set_optimal_keys"),
        context=f"{event_id}: audit final_same_set_optimal_keys",
    ))
    if tuple(local_exact.keys) != audit_exact_uids:
        raise StrictControlError(
            f"{event_id}: local fixed-set exact order differs from audit: "
            f"local={list(local_exact.keys)}, audit={list(audit_exact_uids)}"
        )
    audit_exact_T = audit_row.get("final_same_set_T_opt")
    if audit_exact_T is not None and _nonnegative_int(
        audit_exact_T, context=f"{event_id}: final_same_set_T_opt"
    ) != int(local_exact.acquisition_time):
        raise StrictControlError(
            f"{event_id}: local exact T={local_exact.acquisition_time} differs "
            f"from audit T={audit_exact_T}"
        )

    orders = _build_arm_orders(
        event_id=event_id,
        original_uids=original_uids,
        exact_uids=audit_exact_uids,
        feature_row=feature_row,
        random_seeds=random_seeds,
    )
    candidate_by_uid = {
        _candidate_uid(candidate): copy.deepcopy(dict(candidate))
        for candidate in frozen_candidates
    }
    frozen_fingerprints = _candidate_fingerprints(frozen_candidates)
    solver_roles = _solver_roles(local_exact)

    output: dict[str, dict[str, Any]] = {}
    original_rebuilt: dict[str, Any] | None = None
    for spec in _arm_specs(random_seeds):
        arm = str(spec["name"])
        ordered_uids = orders[arm]
        ordered_candidates = [copy.deepcopy(candidate_by_uid[uid]) for uid in ordered_uids]
        rebuilt = _rebuild_arm_row(
            source_row=source_row,
            ordered_candidates=ordered_candidates,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
        )
        rebuilt_fingerprints = _candidate_fingerprints(rebuilt.get("candidates") or [])
        if rebuilt_fingerprints != frozen_fingerprints:
            raise StrictControlError(
                f"{event_id}:{arm}: candidate UID/text/block fingerprints changed"
            )
        if arm == "original":
            _validate_original_rebuild(source_row=source_row, rebuilt=rebuilt)
            original_rebuilt = rebuilt
        _validate_no_retruncation(
            rebuilt,
            event_id=event_id,
            arm=arm,
            expected_K=len(original_uids),
            max_length=max_length,
        )

        display_eval = evaluate_display(problem, ordered_uids)
        sidecar = _display_sidecar(
            event_id=event_id,
            split=split,
            arm_spec=spec,
            ordered_uids=ordered_uids,
            problem=problem,
            display_eval=display_eval,
            exact_eval=local_exact,
            solver_roles=solver_roles,
            fingerprints=frozen_fingerprints,
        )
        compact_control = {
            "schema_version": SCHEMA_VERSION,
            "arm": arm,
            "random_seed": spec.get("random_seed"),
            "K": len(ordered_uids),
            "ordered_candidate_uids": list(ordered_uids),
            **frozen_fingerprints,
            "display_order_fingerprint": sidecar["display_order_fingerprint"],
            "display_terminal_state": list(display_eval.state),
            "display_utility": int(display_eval.utility),
            "display_T": int(display_eval.acquisition_time),
            "display_AUC": int(sidecar["display_AUC"]),
            "display_AUC_horizon": int(sidecar["display_AUC_horizon"]),
            "order_regret_to_baces_exact": int(sidecar["order_regret_to_baces_exact"]),
        }
        rebuilt["strict_same_set_control"] = compact_control
        rebuilt["evidence_count_before"] = len(ordered_uids)
        rebuilt["evidence_text_truncated"] = False
        rebuilt["was_truncated"] = False
        output[arm] = {"build_row": rebuilt, "sidecar": sidecar}

    if original_rebuilt is None:
        raise AssertionError("original arm was not built")
    reference_prompt_fields = (
        original_rebuilt.get("target"),
        original_rebuilt.get("target_token_count"),
    )
    for arm, payload in output.items():
        rebuilt = payload["build_row"]
        if (rebuilt.get("target"), rebuilt.get("target_token_count")) != reference_prompt_fields:
            raise StrictControlError(f"{event_id}:{arm}: target changed across order arms")
    return output


def _freeze_source_candidates(source_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_id = _event_id(source_row, "source build row")
    if source_row.get("evidence_text_truncated") is not False:
        raise StrictControlError(
            f"{event_id}: source evidence_text_truncated must be exactly false"
        )
    evidence_count = _nonnegative_int(
        source_row.get("evidence_count"), context=f"{event_id}: evidence_count"
    )
    raw_candidates = source_row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise StrictControlError(f"{event_id}: source candidates must be a list")
    if evidence_count > len(raw_candidates):
        raise StrictControlError(
            f"{event_id}: evidence_count={evidence_count} exceeds "
            f"len(candidates)={len(raw_candidates)}"
        )
    frozen: list[dict[str, Any]] = []
    for position, candidate in enumerate(raw_candidates[:evidence_count]):
        if not isinstance(candidate, Mapping):
            raise StrictControlError(
                f"{event_id}: candidates[{position}] must be a mapping"
            )
        copied = copy.deepcopy(dict(candidate))
        _candidate_uid(copied)
        if "text" not in copied or not isinstance(copied.get("text"), str):
            raise StrictControlError(
                f"{event_id}: candidate {copied.get('candidate_uid')!r} has no string text block"
            )
        frozen.append(copied)
    uids = [_candidate_uid(candidate) for candidate in frozen]
    if len(uids) != len(set(uids)):
        raise StrictControlError(f"{event_id}: frozen candidate UIDs are not unique")
    return frozen


def _validate_audit_alignment(
    audit_row: Mapping[str, Any], *, original_uids: Sequence[str], event_id: str
) -> None:
    if str(audit_row.get("status") or "") != "ok":
        raise StrictControlError(
            f"{event_id}: audit.status must be 'ok', got {audit_row.get('status')!r}"
        )
    final_keys = tuple(_string_sequence(
        audit_row.get("final_keys"), context=f"{event_id}: audit final_keys"
    ))
    if final_keys != tuple(original_uids):
        raise StrictControlError(
            f"{event_id}: audit final_keys do not equal source visible order: "
            f"audit={list(final_keys)}, source={list(original_uids)}"
        )
    if audit_row.get("K_final") is not None and _nonnegative_int(
        audit_row.get("K_final"), context=f"{event_id}: audit K_final"
    ) != len(original_uids):
        raise StrictControlError(f"{event_id}: audit K_final differs from source evidence_count")
    if audit_row.get("build_evidence_text_truncated") is True:
        raise StrictControlError(f"{event_id}: audit reports truncated evidence text")


def _compile_frozen_problem(
    *,
    feature_row: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    frozen_candidates: Sequence[Mapping[str, Any]],
    event_id: str,
) -> BacesProblem:
    if str(audit_row.get("weight_policy") or "unit") != "unit":
        raise StrictControlError(
            f"{event_id}: strict builder currently requires audit weight_policy='unit'"
        )
    K = len(frozen_candidates)
    k_max = _nonnegative_int(
        audit_row.get("k_max"), context=f"{event_id}: audit k_max"
    )
    if K > k_max:
        raise StrictControlError(f"{event_id}: frozen K={K} exceeds audit k_max={k_max}")
    token_budget_raw = audit_row.get("token_budget")
    token_budget = (
        None
        if token_budget_raw is None
        else _nonnegative_int(token_budget_raw, context=f"{event_id}: token_budget")
    )
    cost_overrides: dict[str, int] = {}
    for candidate in frozen_candidates:
        uid = _candidate_uid(candidate)
        if candidate.get("mrec_token_cost") is not None:
            cost_overrides[uid] = _nonnegative_int(
                candidate.get("mrec_token_cost"),
                context=f"{event_id}:{uid}: mrec_token_cost",
            )
    problem = compile_feature_problem(
        feature_row,
        k_max=k_max,
        token_budget=token_budget,
        weights=[1] * _feature_atom_count(feature_row),
        cost_overrides=cost_overrides,
    )
    problem_keys = {candidate.key for candidate in problem.candidates}
    frozen_uids = {_candidate_uid(candidate) for candidate in frozen_candidates}
    missing = sorted(frozen_uids - problem_keys)
    if missing:
        raise StrictControlError(
            f"{event_id}: frozen candidate UIDs missing from compiled features: {missing}"
        )
    return problem


def _build_arm_orders(
    *,
    event_id: str,
    original_uids: Sequence[str],
    exact_uids: Sequence[str],
    feature_row: Mapping[str, Any],
    random_seeds: Sequence[int],
) -> dict[str, tuple[str, ...]]:
    """Return deterministic strict-set orders without touching candidate text."""

    original = tuple(str(uid) for uid in original_uids)
    exact = tuple(str(uid) for uid in exact_uids)
    _assert_same_uid_set(original, exact, context=f"{event_id}: baces_exact")
    frozen_set = set(original)

    raw_feature_candidates = feature_row.get("candidates")
    if not isinstance(raw_feature_candidates, (list, tuple)):
        raise StrictControlError(f"{event_id}: feature candidates must be an array")
    feature_by_uid: dict[str, Mapping[str, Any]] = {}
    candidate_pool: list[str] = []
    for position, raw_candidate in enumerate(raw_feature_candidates):
        if not isinstance(raw_candidate, Mapping):
            raise StrictControlError(
                f"{event_id}: feature candidates[{position}] must be a mapping"
            )
        uid = _candidate_uid(raw_candidate)
        if uid in feature_by_uid:
            raise StrictControlError(f"{event_id}: duplicate feature UID {uid!r}")
        feature_by_uid[uid] = raw_candidate
        if uid in frozen_set:
            candidate_pool.append(uid)
    missing = sorted(frozen_set - set(feature_by_uid))
    if missing:
        raise StrictControlError(f"{event_id}: frozen UIDs missing from features: {missing}")

    retrieval = tuple(
        sorted(
            original,
            key=lambda uid: _retrieval_score_sort_key(feature_by_uid[uid], uid=uid),
        )
    )
    orders: dict[str, tuple[str, ...]] = {
        "original": original,
        "baces_exact": exact,
        "retrieval_score": retrieval,
        "candidate_pool": tuple(candidate_pool),
        "reverse": tuple(reversed(original)),
    }
    for seed in random_seeds:
        name = _random_arm_name(int(seed))
        shuffled = list(original)
        random.Random(_event_random_seed(int(seed), event_id)).shuffle(shuffled)
        orders[name] = tuple(shuffled)
    for arm, order in orders.items():
        _assert_same_uid_set(original, order, context=f"{event_id}:{arm}")
    return orders


def _retrieval_score_sort_key(
    candidate: Mapping[str, Any], *, uid: str
) -> tuple[int, float, str]:
    value = candidate.get("hybrid_score")
    score: float | None
    if value is None or isinstance(value, bool):
        score = None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = math.nan
        score = parsed if math.isfinite(parsed) else None
    return (1, 0.0, uid) if score is None else (0, -score, uid)


def _event_random_seed(seed: int, event_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{event_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _random_arm_name(seed: int) -> str:
    return f"random_seed{int(seed)}"


def _arm_specs(random_seeds: Sequence[int]) -> list[dict[str, Any]]:
    specs = [{"name": arm, "random_seed": None} for arm in FIXED_ARMS]
    specs.extend(
        {"name": _random_arm_name(int(seed)), "random_seed": int(seed)}
        for seed in random_seeds
    )
    return specs


def _rebuild_arm_row(
    *,
    source_row: Mapping[str, Any],
    ordered_candidates: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild one prompt with ``auto_length=False`` from frozen text blocks."""

    retrieval_row = copy.deepcopy(dict(source_row))
    retrieval_row["candidates"] = [copy.deepcopy(dict(item)) for item in ordered_candidates]
    cfg = dict(prompt_cfg)
    cfg["auto_length"] = False
    rebuilt = build_training_row(
        retrieval_row,
        tokenizer,
        cfg,
        allow_unlabeled=True,
    )
    return rebuilt


def _validate_original_rebuild(
    *, source_row: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> None:
    event_id = _event_id(source_row, "source build row")
    for field in ("prompt", "prompt_input_ids"):
        if rebuilt.get(field) != source_row.get(field):
            raise StrictControlError(
                f"{event_id}: original rebuild {field} differs from source byte-for-byte"
            )
    for field in ("target", "prompt_token_count", "target_token_count"):
        if rebuilt.get(field) != source_row.get(field):
            raise StrictControlError(
                f"{event_id}: original rebuild {field} differs from source"
            )


def _validate_no_retruncation(
    rebuilt: Mapping[str, Any],
    *,
    event_id: str,
    arm: str,
    expected_K: int,
    max_length: int,
) -> None:
    if bool(rebuilt.get("was_truncated")):
        raise StrictControlError(f"{event_id}:{arm}: rebuilt prompt was truncated")
    if bool(rebuilt.get("evidence_text_truncated")):
        raise StrictControlError(f"{event_id}:{arm}: rebuilt evidence text was truncated")
    evidence_count = _nonnegative_int(
        rebuilt.get("evidence_count"), context=f"{event_id}:{arm}: evidence_count"
    )
    if evidence_count != expected_K:
        raise StrictControlError(
            f"{event_id}:{arm}: rebuilt K={evidence_count}, expected {expected_K}"
        )
    prompt_tokens = _nonnegative_int(
        rebuilt.get("prompt_token_count"),
        context=f"{event_id}:{arm}: prompt_token_count",
    )
    target_tokens = _effective_target_token_count(rebuilt)
    if prompt_tokens + target_tokens > max_length:
        raise StrictControlError(
            f"{event_id}:{arm}: prompt+target={prompt_tokens + target_tokens} "
            f"exceeds max_length={max_length}; strict controls forbid truncation"
        )


def _solver_roles(exact_eval: BacesEvaluation) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    core_position = 0
    seen_fill = False
    for exact_position, step in enumerate(exact_eval.steps, start=1):
        if int(step.delta) > 0:
            if seen_fill:
                raise StrictControlError("exact frozen-set order has CORE after FILL")
            core_position += 1
            solver_role = "CORE"
            solver_core_position: int | None = core_position
            operation = "COVER"
        else:
            seen_fill = True
            solver_role = "FILL"
            solver_core_position = None
            operation = "ZERO_GAIN_FILL"
        roles[step.key] = {
            "solver_role": solver_role,
            "solver_core_position": solver_core_position,
            "solver_exact_position": exact_position,
            "operation": operation,
        }
    return roles


def _display_sidecar(
    *,
    event_id: str,
    split: str,
    arm_spec: Mapping[str, Any],
    ordered_uids: Sequence[str],
    problem: BacesProblem,
    display_eval: BacesEvaluation,
    exact_eval: BacesEvaluation,
    solver_roles: Mapping[str, Mapping[str, Any]],
    fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    atom_ids = tuple(problem.atom_ids) or tuple(
        f"A{idx + 1}" for idx in range(len(problem.weights))
    )
    candidate_by_key = {candidate.key: candidate for candidate in problem.candidates}
    terminal_state = tuple(exact_eval.state)
    steps: list[dict[str, Any]] = []
    for step in display_eval.steps:
        role = dict(solver_roles[step.key])
        candidate = candidate_by_key[step.key]
        steps.append(
            {
                "step": int(step.position),
                "candidate_uid": step.key,
                "candidate_stable_key": step.key,
                **role,
                "display_operation": (
                    "ORDINAL_UPGRADE" if int(step.delta) > 0 else "DISPLAY_ZERO_GAIN"
                ),
                "pair_coverage_levels": {
                    atom_id: int(level) for atom_id, level in zip(atom_ids, candidate.q)
                },
                "display_coverage_levels_before": {
                    atom_id: int(level)
                    for atom_id, level in zip(atom_ids, step.state_before)
                },
                "display_coverage_levels_after": {
                    atom_id: int(level)
                    for atom_id, level in zip(atom_ids, step.state_after)
                },
                "display_marginal_coverage_units": int(step.delta),
                "display_cumulative_coverage_units": int(step.cumulative_utility),
                "display_weighted_acquisition_time_so_far": int(
                    step.acquisition_time_so_far
                ),
                "candidate_token_cost": int(step.candidate_cost),
                "display_cumulative_token_cost": int(step.cumulative_cost),
                "target_coverage_reached": tuple(step.state_after) == terminal_state,
            }
        )
    horizon = len(ordered_uids)
    display_auc = int(padded_auc(display_eval, horizon))
    order_regret = int(display_eval.acquisition_time) - int(exact_eval.acquisition_time)
    if order_regret < 0:
        raise StrictControlError(
            f"{event_id}:{arm_spec['name']}: negative exact-order regret {order_regret}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "split": split,
        "arm": str(arm_spec["name"]),
        "random_seed": arm_spec.get("random_seed"),
        "K": horizon,
        "ordered_candidate_uids": list(ordered_uids),
        **dict(fingerprints),
        "display_order_fingerprint": _display_order_fingerprint(
            event_id=event_id, ordered_uids=ordered_uids
        ),
        "solver_role_reference_order": list(exact_eval.keys),
        "solver_core_keys": [
            key for key in exact_eval.keys if solver_roles[key]["solver_role"] == "CORE"
        ],
        "solver_fill_keys": [
            key for key in exact_eval.keys if solver_roles[key]["solver_role"] == "FILL"
        ],
        "display_terminal_state": {
            atom_id: int(level) for atom_id, level in zip(atom_ids, display_eval.state)
        },
        "display_utility": int(display_eval.utility),
        "display_T": int(display_eval.acquisition_time),
        "display_AUC": display_auc,
        "display_AUC_horizon": horizon,
        "display_token_cost": int(display_eval.token_cost),
        "baces_exact_T": int(exact_eval.acquisition_time),
        "baces_exact_AUC": int(padded_auc(exact_eval, horizon)),
        "order_regret_to_baces_exact": order_regret,
        "steps": steps,
        "order_contract": {
            "retrieval_score_policy": RETRIEVAL_SCORE_POLICY,
            "candidate_pool_policy": CANDIDATE_POOL_POLICY,
            "reverse_policy": REVERSE_POLICY,
            "random_seed_hash_policy": RANDOM_SEED_HASH_POLICY,
        },
    }


def _candidate_fingerprints(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    by_uid: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise StrictControlError("candidate fingerprint input must contain mappings")
        uid = _candidate_uid(candidate)
        if uid in by_uid:
            raise StrictControlError(f"duplicate candidate UID {uid!r} in fingerprint")
        by_uid[uid] = candidate
    uids = sorted(by_uid)
    uid_text = [
        {"candidate_uid": uid, "text": by_uid[uid].get("text")}
        for uid in uids
    ]
    uid_blocks = [
        {"candidate_uid": uid, "candidate": by_uid[uid]}
        for uid in uids
    ]
    return {
        "uid_set_fingerprint": _canonical_fingerprint(uids),
        "uid_text_fingerprint": _canonical_fingerprint(uid_text),
        "candidate_block_fingerprint": _canonical_fingerprint(uid_blocks),
    }


def _display_order_fingerprint(
    *, event_id: str, ordered_uids: Sequence[str]
) -> str:
    """Hash the event-scoped, order-sensitive visible UID sequence."""

    return _canonical_fingerprint(
        {
            "event_id": str(event_id),
            "ordered_candidate_uids": [str(uid) for uid in ordered_uids],
        }
    )


def _canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_same_uid_set(
    reference: Sequence[str], candidate: Sequence[str], *, context: str
) -> None:
    if len(reference) != len(set(reference)):
        raise StrictControlError(f"{context}: reference UIDs contain duplicates")
    if len(candidate) != len(set(candidate)):
        raise StrictControlError(f"{context}: order UIDs contain duplicates")
    if len(reference) != len(candidate) or set(reference) != set(candidate):
        raise StrictControlError(
            f"{context}: not a strict permutation; "
            f"missing={sorted(set(reference) - set(candidate))}, "
            f"extra={sorted(set(candidate) - set(reference))}"
        )


def _feature_atom_count(feature_row: Mapping[str, Any]) -> int:
    evidence_map = feature_row.get("evidence_map")
    atoms = evidence_map.get("claim_atoms") if isinstance(evidence_map, Mapping) else None
    if not atoms:
        atoms = feature_row.get("claim_atoms")
    if not isinstance(atoms, (list, tuple)) or not atoms:
        raise StrictControlError("feature row has no claim atoms")
    return len(atoms)


def _effective_target_token_count(row: Mapping[str, Any]) -> int:
    target_count = _nonnegative_int(
        row.get("target_token_count", 0), context="target_token_count"
    )
    reserve_raw = row.get("inference_target_token_reserve")
    if reserve_raw is None:
        return target_count
    reserve = _nonnegative_int(reserve_raw, context="inference_target_token_reserve")
    return max(target_count, reserve)


def _append_stats(
    stats: dict[str, list[int]],
    *,
    build_row: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    prompt_tokens = int(build_row["prompt_token_count"])
    target_tokens = _effective_target_token_count(build_row)
    stats["K"].append(int(sidecar["K"]))
    stats["prompt_tokens"].append(prompt_tokens)
    stats["target_tokens"].append(target_tokens)
    stats["total_tokens"].append(prompt_tokens + target_tokens)
    stats["display_utility"].append(int(sidecar["display_utility"]))
    stats["display_T"].append(int(sidecar["display_T"]))
    stats["display_AUC"].append(int(sidecar["display_AUC"]))
    stats["order_regret"].append(int(sidecar["order_regret_to_baces_exact"]))


def _arm_summary(
    *,
    arm_spec: Mapping[str, Any],
    split: str,
    n_events: int,
    max_length: int,
    stats: Mapping[str, Sequence[int]],
    source_build_path: Path,
    features_path: Path,
    audit_path: Path,
    config_path: Path,
    build_path: Path,
    sidecar_path: Path,
    sample_limit: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "arm": str(arm_spec["name"]),
        "random_seed": arm_spec.get("random_seed"),
        "n_events": n_events,
        "sample_limit": sample_limit,
        "max_length": max_length,
        "source_build": str(source_build_path),
        "features": str(features_path),
        "audit": str(audit_path),
        "config": str(config_path),
        "build_output": str(build_path),
        "strict_same_set_control_sidecar": str(sidecar_path),
        "metrics": {name: _numeric_summary(values) for name, values in stats.items()},
        "contract_checks": {
            "audit_status_ok": True,
            "audit_final_keys_equal_source_visible_keys": True,
            "source_evidence_text_truncated_false": True,
            "original_prompt_exact_rebuild": True,
            "original_prompt_input_ids_exact_rebuild": True,
            "all_arms_same_K_uid_set_and_text_blocks": True,
            "all_arms_auto_length_false": True,
            "all_arms_prompt_plus_target_within_max_length": True,
            "local_fixed_set_solver_equals_audit": True,
            "solver_role_frozen_display_marginal_replayed": True,
        },
        "order_contract": {
            "set_text_block_fingerprints": "order_invariant_sha256_canonical_json_v1",
            "display_order_fingerprint": (
                "order_sensitive_sha256_canonical_json_event_id_and_ordered_candidate_uids_v1"
            ),
            "retrieval_score_policy": RETRIEVAL_SCORE_POLICY,
            "candidate_pool_policy": CANDIDATE_POOL_POLICY,
            "reverse_policy": REVERSE_POLICY,
            "random_seed_hash_policy": RANDOM_SEED_HASH_POLICY,
        },
    }


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _index_jsonl(path: Path, *, artifact: str) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with Path(path).open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise StrictControlError(
                    f"invalid JSON in {artifact}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise StrictControlError(f"{artifact}:{line_number} is not an object")
            event_id = _event_id(row, f"{artifact}:{line_number}")
            if event_id in offsets:
                raise StrictControlError(
                    f"duplicate event_id {event_id!r} in {artifact}"
                )
            offsets[event_id] = offset
    return offsets


def _read_row_at(handle: BinaryIO, offset: int, *, artifact: str) -> dict[str, Any]:
    handle.seek(offset)
    raw_line = handle.readline()
    try:
        row = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise StrictControlError(f"invalid indexed JSON in {artifact}: {exc}") from exc
    if not isinstance(row, dict):
        raise StrictControlError(f"{artifact} is not a JSON object")
    return row


def _iter_jsonl(
    path: Path, *, artifact: str
) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StrictControlError(
                    f"invalid JSON in {artifact}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise StrictControlError(f"{artifact}:{line_number} is not an object")
            yield line_number, row


def _event_id(row: Mapping[str, Any], context: str) -> str:
    value = str(row.get("event_id") or "").strip()
    if not value:
        raise StrictControlError(f"{context} has no event_id")
    return value


def _candidate_uid(candidate: Mapping[str, Any]) -> str:
    uid = str(candidate.get("candidate_uid") or "").strip()
    if not uid:
        raise StrictControlError("candidate has no non-empty candidate_uid")
    return uid


def _string_sequence(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise StrictControlError(f"{context} must be an array")
    out: list[str] = []
    for position, item in enumerate(value):
        token = str(item or "").strip()
        if not token:
            raise StrictControlError(f"{context}[{position}] is empty")
        out.append(token)
    if len(out) != len(set(out)):
        raise StrictControlError(f"{context} contains duplicates")
    return out


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise StrictControlError(f"{context} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StrictControlError(f"{context} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise StrictControlError(f"{context} must be an integer")
    if parsed < 0:
        raise StrictControlError(f"{context} must be non-negative")
    return parsed


def _positive_int(value: Any, *, context: str) -> int:
    parsed = _nonnegative_int(value, context=context)
    if parsed <= 0:
        raise StrictControlError(f"{context} must be positive")
    return parsed


def _json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _load_experiment_config(path: Path) -> dict[str, Any]:
    """Load the repository's lightweight recursive ``extends`` convention."""

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    payload = load_yaml(resolved)
    if not isinstance(payload, dict):
        raise StrictControlError(f"config {resolved} is not a mapping")
    payload = dict(payload)
    parent = payload.pop("extends", None)
    if not parent:
        return payload
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge(_load_experiment_config(parent_path), payload)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
