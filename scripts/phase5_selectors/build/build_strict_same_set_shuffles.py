#!/usr/bin/env python3
"""Build deterministic post-prefix same-visible-set shuffle controls.

The input is an already rendered ``build_<split>.jsonl`` artifact.  For each
event this script freezes exactly ``candidates[:evidence_count]`` and rebuilds
the prompt with those candidate blocks in a deterministic event-and-seed
permutation.  It is deliberately selector-agnostic: no selector trace is read,
replayed, or rewritten.

The contract is fail-closed.  Source rows with text truncation, duplicate or
missing candidate UIDs, prompt/config drift, overflow, target/label drift, or a
change to the order-invariant prompt-token multiset are rejected.  Automatic
length adjustment is disabled for every rebuilt row.

Output layout::

    OUTPUT_DIR/shuffle_seed0/build_<split>.jsonl
    OUTPUT_DIR/shuffle_seed0/strict_same_set_shuffle_<split>.jsonl
    OUTPUT_DIR/shuffle_seed0/summary_<split>.json
    ...
    OUTPUT_DIR/manifest_<split>.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from fact_checking.build.prompts import build_training_row, load_prompt_tokenizer
from fact_checking.config import load_yaml


SCHEMA_VERSION = "strict-same-visible-set-shuffle-v0.1"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
RANDOM_SEED_HASH_POLICY = "sha256(seed\\0event_id)-first64be-python-random-v1"
IDENTITY_FALLBACK_POLICY = "left-rotate-one-if-random-shuffle-is-identity-v1"
ORDER_SAMPLING_POLICY = (
    "event-seeded-random-unique-permutations-then-lexicographic-fallback-v1"
)
MAX_SHUFFLE_ORDER_ATTEMPTS = 256


class StrictShuffleError(ValueError):
    """Raised when an input or rebuilt row violates the shuffle contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-build", required=True, help="Rendered build_<split>.jsonl")
    parser.add_argument("--config", required=True, help="Experiment YAML used for the source build")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Deterministic event-specific shuffle seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Build only the first N rows for a contract smoke test.",
    )
    args = parser.parse_args()
    try:
        _normalize_seeds(args.seeds)
    except StrictShuffleError as exc:
        parser.error(str(exc))
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    return args


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = _load_experiment_config(config_path)
    prompt_cfg = dict((cfg.get("build", {}) or {}).get("prompt", {}) or {})
    model_name_or_path = str(prompt_cfg.get("model_name_or_path") or "").strip()
    if not model_name_or_path:
        raise StrictShuffleError("config build.prompt.model_name_or_path is required")
    tokenizer = load_prompt_tokenizer(model_name_or_path)
    manifest = build_strict_same_set_shuffles(
        source_build_path=Path(args.source_build),
        config_path=config_path,
        split=str(args.split),
        output_dir=Path(args.output_dir),
        seeds=tuple(args.seeds),
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
        sample_limit=args.sample_limit,
    )
    print(
        f"Wrote {manifest['n_events']} events x {len(manifest['arms'])} shuffle arms "
        f"to {manifest['output_dir']}"
    )
    return 0


def build_strict_same_set_shuffles(
    *,
    source_build_path: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    seeds: Sequence[int],
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Stream a source build once and materialize all requested seed arms."""

    source_build_path = Path(source_build_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    normalized_seeds = _normalize_seeds(seeds)
    if sample_limit is not None and sample_limit <= 0:
        raise StrictShuffleError("sample_limit must be positive")
    if not source_build_path.is_file():
        raise StrictShuffleError(f"source build does not exist: {source_build_path}")
    if not config_path.is_file():
        raise StrictShuffleError(f"config does not exist: {config_path}")

    cfg = dict(prompt_cfg)
    cfg["auto_length"] = False
    max_length = _positive_int(cfg.get("max_length", 2048), context="build.prompt.max_length")
    arm_specs = [_arm_spec(seed) for seed in normalized_seeds]
    output_dir.mkdir(parents=True, exist_ok=True)

    final_paths: dict[str, dict[str, Path]] = {}
    temp_paths: list[Path] = []
    stats: dict[str, dict[str, int | list[int]]] = {}
    for spec in arm_specs:
        arm = str(spec["name"])
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        final_paths[arm] = {
            "build": arm_dir / f"build_{split}.jsonl",
            "sidecar": arm_dir / f"strict_same_set_shuffle_{split}.jsonl",
            "summary": arm_dir / f"summary_{split}.json",
        }
        stats[arm] = {
            "K": [],
            "prompt_tokens": [],
            "shuffle_order_attempt": [],
            "shuffle_rejected_order_count": [],
            "order_eligible": 0,
            "order_changed": 0,
            "verifier_prompt_changed": 0,
        }

    seen_event_ids: set[str] = set()
    n_events = 0
    promoted = False
    try:
        with ExitStack() as stack:
            writers: dict[str, dict[str, TextIO]] = {}
            for spec in arm_specs:
                arm = str(spec["name"])
                writers[arm] = {}
                for kind in ("build", "sidecar"):
                    final_path = final_paths[arm][kind]
                    temp_path = _temp_path(final_path)
                    temp_paths.append(temp_path)
                    writers[arm][kind] = stack.enter_context(temp_path.open("w", encoding="utf-8"))

            for line_number, source_row in _iter_jsonl(source_build_path):
                if sample_limit is not None and n_events >= sample_limit:
                    break
                event_id = _event_id(source_row, context=f"source:{line_number}")
                if event_id in seen_event_ids:
                    raise StrictShuffleError(f"duplicate event_id {event_id!r} in source build")
                seen_event_ids.add(event_id)
                payloads = _build_event_shuffles(
                    source_row=source_row,
                    split=split,
                    seeds=normalized_seeds,
                    tokenizer=tokenizer,
                    prompt_cfg=cfg,
                    max_length=max_length,
                )
                for arm, payload in payloads.items():
                    writers[arm]["build"].write(_json_line(payload["build_row"]))
                    writers[arm]["sidecar"].write(_json_line(payload["sidecar"]))
                    _append_stats(stats[arm], payload["build_row"], payload["sidecar"])
                n_events += 1

        if n_events == 0:
            raise StrictShuffleError("source build produced zero events")

        for spec in arm_specs:
            arm = str(spec["name"])
            for kind in ("build", "sidecar"):
                _temp_path(final_paths[arm][kind]).replace(final_paths[arm][kind])
            summary = _arm_summary(
                arm_spec=spec,
                split=split,
                n_events=n_events,
                max_length=max_length,
                source_build_path=source_build_path,
                config_path=config_path,
                build_path=final_paths[arm]["build"],
                sidecar_path=final_paths[arm]["sidecar"],
                stats=stats[arm],
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
        "sample_limit": sample_limit,
        "output_dir": str(output_dir),
        "source_build": str(source_build_path),
        "source_build_sha256": _sha256_file(source_build_path),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "arms": {
            str(spec["name"]): {
                "seed": int(spec["seed"]),
                **{kind: str(path) for kind, path in final_paths[str(spec["name"])].items()},
            }
            for spec in arm_specs
        },
        "contract": _contract_manifest(max_length=max_length),
    }
    _write_json_atomic(output_dir / f"manifest_{split}.json", manifest)
    return manifest


def _build_event_shuffles(
    *,
    source_row: Mapping[str, Any],
    split: str,
    seeds: Sequence[int],
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
    max_length: int,
) -> dict[str, dict[str, Any]]:
    """Pure single-event implementation used by the CLI and tests."""

    event_id = _event_id(source_row, context="source row")
    frozen = _freeze_visible_candidates(source_row)
    original_uids = tuple(_candidate_uid(candidate) for candidate in frozen)
    original_rebuilt = _rebuild_row(
        source_row=source_row,
        ordered_candidates=frozen,
        tokenizer=tokenizer,
        prompt_cfg=prompt_cfg,
    )
    _validate_original_rebuild(source_row=source_row, rebuilt=original_rebuilt)
    _validate_no_retruncation(
        original_rebuilt,
        event_id=event_id,
        arm="source_rebuild",
        expected_K=len(frozen),
        max_length=max_length,
    )

    frozen_fingerprints = _candidate_fingerprints(frozen, tokenizer=tokenizer)
    source_prompt_tokens = _prompt_token_ids(original_rebuilt, tokenizer=tokenizer)
    source_prompt_multiset_fingerprint = _token_multiset_fingerprint(source_prompt_tokens)
    source_prompt_sequence_fingerprint = _canonical_fingerprint(source_prompt_tokens)
    target_label_fingerprint = _target_label_fingerprint(source_row)
    candidate_by_uid = {
        _candidate_uid(candidate): copy.deepcopy(dict(candidate)) for candidate in frozen
    }

    output: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        spec = _arm_spec(seed)
        arm = str(spec["name"])
        shuffled_uids, rebuilt, prompt_tokens, rejected_orders = _find_contract_valid_shuffle(
            source_row=source_row,
            event_id=event_id,
            arm=arm,
            seed=int(seed),
            original_uids=original_uids,
            candidate_by_uid=candidate_by_uid,
            frozen_fingerprints=frozen_fingerprints,
            source_prompt_token_count=int(original_rebuilt["prompt_token_count"]),
            source_prompt_multiset_fingerprint=source_prompt_multiset_fingerprint,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            max_length=max_length,
        )
        prompt_multiset_fingerprint = _token_multiset_fingerprint(prompt_tokens)
        if _target_label_fingerprint(rebuilt) != target_label_fingerprint:
            raise StrictShuffleError(f"{event_id}:{arm}: target/label fingerprint changed")

        order_changed = shuffled_uids != original_uids
        if len(original_uids) > 1 and not order_changed:
            raise StrictShuffleError(f"{event_id}:{arm}: eligible shuffle remained identity")
        prompt_sequence_fingerprint = _canonical_fingerprint(prompt_tokens)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "split": split,
            "arm": arm,
            "seed": int(seed),
            "K": len(frozen),
            "source_ordered_candidate_uids": list(original_uids),
            "ordered_candidate_uids": list(shuffled_uids),
            **frozen_fingerprints,
            "source_order_fingerprint": _order_fingerprint(event_id, original_uids),
            "display_order_fingerprint": _order_fingerprint(event_id, shuffled_uids),
            "target_label_fingerprint": target_label_fingerprint,
            "prompt_token_multiset_fingerprint": prompt_multiset_fingerprint,
            "source_prompt_token_sequence_fingerprint": source_prompt_sequence_fingerprint,
            "prompt_token_sequence_fingerprint": prompt_sequence_fingerprint,
            "prompt_token_count": int(rebuilt["prompt_token_count"]),
            "shuffle_order_attempt": rejected_orders + 1,
            "shuffle_rejected_order_count": rejected_orders,
            "order_eligible": len(original_uids) > 1,
            "order_changed": order_changed,
            "verifier_prompt_changed": prompt_sequence_fingerprint
            != source_prompt_sequence_fingerprint,
            "source_was_truncated": bool(source_row.get("was_truncated")),
            "source_evidence_count_before": source_row.get("evidence_count_before"),
            "contract": {
                "presentation_view_only": True,
                "auto_length": False,
                "random_seed_hash_policy": RANDOM_SEED_HASH_POLICY,
                "identity_fallback_policy": IDENTITY_FALLBACK_POLICY,
                "order_sampling_policy": ORDER_SAMPLING_POLICY,
                "max_shuffle_order_attempts": MAX_SHUFFLE_ORDER_ATTEMPTS,
            },
        }
        rebuilt["strict_same_set_shuffle"] = {
            key: sidecar[key]
            for key in (
                "schema_version",
                "arm",
                "seed",
                "K",
                "ordered_candidate_uids",
                "uid_set_fingerprint",
                "uid_text_fingerprint",
                "candidate_block_fingerprint",
                "evidence_token_content_fingerprint",
                "source_order_fingerprint",
                "display_order_fingerprint",
                "target_label_fingerprint",
                "prompt_token_multiset_fingerprint",
                "prompt_token_sequence_fingerprint",
                "shuffle_order_attempt",
                "shuffle_rejected_order_count",
                "order_changed",
                "verifier_prompt_changed",
            )
        }
        output[arm] = {"build_row": rebuilt, "sidecar": sidecar}
    return output


def _find_contract_valid_shuffle(
    *,
    source_row: Mapping[str, Any],
    event_id: str,
    arm: str,
    seed: int,
    original_uids: Sequence[str],
    candidate_by_uid: Mapping[str, Mapping[str, Any]],
    frozen_fingerprints: Mapping[str, str],
    source_prompt_token_count: int,
    source_prompt_multiset_fingerprint: str,
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
    max_length: int,
) -> tuple[tuple[str, ...], dict[str, Any], list[int], int]:
    """Return the first deterministic permutation satisfying the token contract.

    Whole-prompt tokenization can be boundary-sensitive: a uniformly sampled
    permutation can occasionally change token count or token IDs even though
    every evidence text block is unchanged.  We therefore rejection-sample
    deterministic event-seeded permutations.  This keeps the strict token
    contract instead of silently accepting drift or dropping the event.
    """

    if len(original_uids) <= 1:
        candidate_orders: Iterable[tuple[str, ...]] = [tuple(original_uids)]
    else:
        candidate_orders = _candidate_shuffle_orders(
            original_uids,
            event_id=event_id,
            seed=seed,
            max_attempts=MAX_SHUFFLE_ORDER_ATTEMPTS,
        )

    rejected = 0
    last_reason = "no candidate permutation was generated"
    for shuffled_uids in candidate_orders:
        ordered_candidates = [
            copy.deepcopy(dict(candidate_by_uid[uid])) for uid in shuffled_uids
        ]
        rebuilt = _rebuild_row(
            source_row=source_row,
            ordered_candidates=ordered_candidates,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
        )
        rebuilt["evidence_count_before"] = len(ordered_candidates)
        rebuilt["was_truncated"] = False
        rebuilt["evidence_text_truncated"] = False

        rebuilt_fingerprints = _candidate_fingerprints(
            rebuilt.get("candidates") or [], tokenizer=tokenizer
        )
        if rebuilt_fingerprints != frozen_fingerprints:
            raise StrictShuffleError(
                f"{event_id}:{arm}: UID/text/block/token-content fingerprints changed"
            )
        _validate_target_and_label(source_row=source_row, rebuilt=rebuilt, arm=arm)
        _validate_rebuilt_shape(
            rebuilt,
            event_id=event_id,
            arm=arm,
            expected_K=len(original_uids),
        )
        prompt_tokens = _prompt_token_ids(rebuilt, tokenizer=tokenizer)
        if int(rebuilt.get("prompt_token_count", -1)) != source_prompt_token_count:
            rejected += 1
            last_reason = "prompt_token_count changed"
            continue
        prompt_multiset_fingerprint = _token_multiset_fingerprint(prompt_tokens)
        if prompt_multiset_fingerprint != source_prompt_multiset_fingerprint:
            rejected += 1
            last_reason = "order-invariant prompt token content changed"
            continue
        _validate_no_retruncation(
            rebuilt,
            event_id=event_id,
            arm=arm,
            expected_K=len(original_uids),
            max_length=max_length,
        )
        return tuple(shuffled_uids), rebuilt, prompt_tokens, rejected

    raise StrictShuffleError(
        f"{event_id}:{arm}: no non-identity permutation satisfied the strict token "
        f"contract after {rejected} rejected orders; last_reason={last_reason}"
    )


def _freeze_visible_candidates(source_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_id = _event_id(source_row, context="source row")
    if source_row.get("evidence_text_truncated") is not False:
        raise StrictShuffleError(
            f"{event_id}: source evidence_text_truncated must be exactly false"
        )
    evidence_count = _nonnegative_int(
        source_row.get("evidence_count"), context=f"{event_id}: evidence_count"
    )
    candidates = source_row.get("candidates")
    if not isinstance(candidates, list):
        raise StrictShuffleError(f"{event_id}: candidates must be a list")
    if evidence_count > len(candidates):
        raise StrictShuffleError(
            f"{event_id}: evidence_count={evidence_count} exceeds len(candidates)={len(candidates)}"
        )
    frozen: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates[:evidence_count]):
        if not isinstance(candidate, Mapping):
            raise StrictShuffleError(f"{event_id}: candidates[{position}] must be a mapping")
        copied = copy.deepcopy(dict(candidate))
        uid = _candidate_uid(copied)
        if not isinstance(copied.get("text"), str):
            raise StrictShuffleError(f"{event_id}:{uid}: text must be a string")
        frozen.append(copied)
    uids = [_candidate_uid(candidate) for candidate in frozen]
    if len(uids) != len(set(uids)):
        raise StrictShuffleError(f"{event_id}: visible candidate UIDs are not unique")
    return frozen


def _rebuild_row(
    *,
    source_row: Mapping[str, Any],
    ordered_candidates: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    retrieval_row = copy.deepcopy(dict(source_row))
    retrieval_row["candidates"] = [copy.deepcopy(dict(candidate)) for candidate in ordered_candidates]
    cfg = dict(prompt_cfg)
    cfg["auto_length"] = False
    return build_training_row(retrieval_row, tokenizer, cfg, allow_unlabeled=True)


def _validate_original_rebuild(
    *, source_row: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> None:
    event_id = _event_id(source_row, context="source row")
    for field in (
        "prompt",
        "prompt_input_ids",
        "prompt_token_count",
        "target",
        "target_token_count",
        "label",
        "label_schema",
        "gold_label",
        "gold_id",
    ):
        if rebuilt.get(field) != source_row.get(field):
            raise StrictShuffleError(
                f"{event_id}: source visible-prefix rebuild changed {field!r}; "
                "config/tokenizer/source artifact do not match"
            )
    _validate_optional_identity_fields(source_row=source_row, rebuilt=rebuilt, arm="source_rebuild")


def _validate_target_and_label(
    *, source_row: Mapping[str, Any], rebuilt: Mapping[str, Any], arm: str
) -> None:
    event_id = _event_id(source_row, context="source row")
    for field in (
        "event_id",
        "claim",
        "label",
        "label_schema",
        "explain",
        "target",
        "target_token_count",
        "gold_label",
        "gold_id",
        "gold_explain",
        "prompt_add_special_tokens",
        "preserve_prompt_prefix",
    ):
        if rebuilt.get(field) != source_row.get(field):
            raise StrictShuffleError(f"{event_id}:{arm}: field {field!r} changed")
    _validate_optional_identity_fields(source_row=source_row, rebuilt=rebuilt, arm=arm)


def _validate_optional_identity_fields(
    *, source_row: Mapping[str, Any], rebuilt: Mapping[str, Any], arm: str
) -> None:
    event_id = _event_id(source_row, context="source row")
    for field in (
        "coverage_label",
        "unlabeled_inference",
        "inference_target_token_reserve",
    ):
        if field in source_row or field in rebuilt:
            if rebuilt.get(field) != source_row.get(field):
                raise StrictShuffleError(f"{event_id}:{arm}: optional field {field!r} changed")


def _validate_no_retruncation(
    rebuilt: Mapping[str, Any],
    *,
    event_id: str,
    arm: str,
    expected_K: int,
    max_length: int,
) -> None:
    _validate_rebuilt_shape(
        rebuilt,
        event_id=event_id,
        arm=arm,
        expected_K=expected_K,
    )
    prompt_tokens = _nonnegative_int(
        rebuilt.get("prompt_token_count"), context=f"{event_id}:{arm}: prompt_token_count"
    )
    total_tokens = prompt_tokens + _effective_target_token_count(rebuilt)
    if total_tokens > max_length:
        raise StrictShuffleError(
            f"{event_id}:{arm}: prompt+target={total_tokens} exceeds max_length={max_length}; "
            "strict shuffle forbids retruncation"
        )


def _validate_rebuilt_shape(
    rebuilt: Mapping[str, Any],
    *,
    event_id: str,
    arm: str,
    expected_K: int,
) -> None:
    if bool(rebuilt.get("was_truncated")):
        raise StrictShuffleError(f"{event_id}:{arm}: rebuilt prompt was truncated")
    if bool(rebuilt.get("evidence_text_truncated")):
        raise StrictShuffleError(f"{event_id}:{arm}: rebuilt evidence text was truncated")
    evidence_count = _nonnegative_int(
        rebuilt.get("evidence_count"), context=f"{event_id}:{arm}: evidence_count"
    )
    if evidence_count != expected_K:
        raise StrictShuffleError(
            f"{event_id}:{arm}: rebuilt evidence_count={evidence_count}, expected {expected_K}"
        )
    candidates = rebuilt.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_K:
        raise StrictShuffleError(f"{event_id}:{arm}: rebuilt candidate count differs from K")
    _nonnegative_int(
        rebuilt.get("prompt_token_count"), context=f"{event_id}:{arm}: prompt_token_count"
    )


def _deterministic_nonidentity_shuffle(
    original_uids: Sequence[str], *, event_id: str, seed: int
) -> tuple[str, ...]:
    original = tuple(str(uid) for uid in original_uids)
    shuffled = list(original)
    random.Random(_event_random_seed(seed, event_id)).shuffle(shuffled)
    if len(shuffled) > 1 and tuple(shuffled) == original:
        shuffled = shuffled[1:] + shuffled[:1]
    if len(shuffled) != len(original) or set(shuffled) != set(original):
        raise StrictShuffleError(f"{event_id}: shuffle seed {seed} is not a strict permutation")
    return tuple(shuffled)


def _candidate_shuffle_orders(
    original_uids: Sequence[str],
    *,
    event_id: str,
    seed: int,
    max_attempts: int,
) -> Iterable[tuple[str, ...]]:
    """Yield unique deterministic non-identity orders up to ``max_attempts``."""

    original = tuple(str(uid) for uid in original_uids)
    if len(original) <= 1:
        return
    seen: set[tuple[str, ...]] = {original}
    rng = random.Random(_event_random_seed(seed, event_id))

    first = list(original)
    rng.shuffle(first)
    if tuple(first) == original:
        first = first[1:] + first[:1]
    first_order = tuple(first)
    seen.add(first_order)
    yield first_order
    yielded = 1

    draw_budget = max_attempts * 20
    for _ in range(draw_budget):
        if yielded >= max_attempts:
            return
        candidate = list(original)
        rng.shuffle(candidate)
        order = tuple(candidate)
        if order in seen:
            continue
        seen.add(order)
        yielded += 1
        yield order

    # Small K can exhaust the random draw budget through duplicates.  Finish
    # deterministically so an available valid order is not missed by chance.
    if len(original) <= 7:
        for order in itertools.permutations(original):
            if yielded >= max_attempts:
                return
            if order in seen:
                continue
            seen.add(order)
            yielded += 1
            yield order


def _event_random_seed(seed: int, event_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{event_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _arm_spec(seed: int) -> dict[str, int | str]:
    return {"name": f"shuffle_seed{int(seed)}", "seed": int(seed)}


def _normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise StrictShuffleError("at least one seed is required")
    if any(seed < 0 for seed in normalized):
        raise StrictShuffleError("seeds must be non-negative")
    if len(set(normalized)) != len(normalized):
        raise StrictShuffleError("seeds must not contain duplicates")
    return normalized


def _candidate_fingerprints(
    candidates: Iterable[Mapping[str, Any]], *, tokenizer: Any
) -> dict[str, str]:
    by_uid: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise StrictShuffleError("candidate fingerprint input must contain mappings")
        uid = _candidate_uid(candidate)
        if uid in by_uid:
            raise StrictShuffleError(f"duplicate candidate UID {uid!r} in fingerprint input")
        by_uid[uid] = candidate
    uids = sorted(by_uid)
    uid_text = [{"candidate_uid": uid, "text": by_uid[uid].get("text")} for uid in uids]
    uid_blocks = [{"candidate_uid": uid, "candidate": by_uid[uid]} for uid in uids]
    token_content = [
        {
            "candidate_uid": uid,
            "text_token_ids": _tokenize_text(
                str(by_uid[uid].get("text") or "").strip(), tokenizer=tokenizer
            ),
        }
        for uid in uids
    ]
    return {
        "uid_set_fingerprint": _canonical_fingerprint(uids),
        "uid_text_fingerprint": _canonical_fingerprint(uid_text),
        "candidate_block_fingerprint": _canonical_fingerprint(uid_blocks),
        "evidence_token_content_fingerprint": _canonical_fingerprint(token_content),
    }


def _prompt_token_ids(row: Mapping[str, Any], *, tokenizer: Any) -> list[int]:
    prompt_input_ids = row.get("prompt_input_ids")
    if prompt_input_ids is not None:
        return _as_token_ids(prompt_input_ids, context="prompt_input_ids")
    return _tokenize_text(str(row.get("prompt") or ""), tokenizer=tokenizer)


def _tokenize_text(text: str, *, tokenizer: Any) -> list[int]:
    encoded = tokenizer(text, truncation=False, add_special_tokens=False)
    value = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    return _as_token_ids(value, context="tokenizer input_ids")


def _as_token_ids(value: Any, *, context: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise StrictShuffleError(f"{context} contains multiple sequences")
        value = value[0]
    if not isinstance(value, list):
        raise StrictShuffleError(f"{context} must be a token-id list")
    output: list[int] = []
    for position, token_id in enumerate(value):
        if isinstance(token_id, bool):
            raise StrictShuffleError(f"{context}[{position}] is bool, not int")
        try:
            output.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise StrictShuffleError(f"{context}[{position}] is not an int") from exc
    return output


def _token_multiset_fingerprint(token_ids: Sequence[int]) -> str:
    counts: dict[int, int] = {}
    for token_id in token_ids:
        key = int(token_id)
        counts[key] = counts.get(key, 0) + 1
    return _canonical_fingerprint([[token_id, counts[token_id]] for token_id in sorted(counts)])


def _target_label_fingerprint(row: Mapping[str, Any]) -> str:
    return _canonical_fingerprint(
        {
            field: row.get(field)
            for field in (
                "event_id",
                "claim",
                "label",
                "label_schema",
                "explain",
                "target",
                "target_token_count",
                "gold_label",
                "gold_id",
                "gold_explain",
                "coverage_label",
                "unlabeled_inference",
                "inference_target_token_reserve",
            )
            if field in row
        }
    )


def _order_fingerprint(event_id: str, ordered_uids: Sequence[str]) -> str:
    return _canonical_fingerprint(
        {"event_id": event_id, "ordered_candidate_uids": [str(uid) for uid in ordered_uids]}
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


def _effective_target_token_count(row: Mapping[str, Any]) -> int:
    target_count = _nonnegative_int(
        row.get("target_token_count", 0), context="target_token_count"
    )
    reserve = row.get("inference_target_token_reserve")
    if reserve is None:
        return target_count
    return max(
        target_count,
        _nonnegative_int(reserve, context="inference_target_token_reserve"),
    )


def _candidate_uid(candidate: Mapping[str, Any]) -> str:
    uid = str(candidate.get("candidate_uid") or "").strip()
    if not uid:
        raise StrictShuffleError("candidate has no non-empty candidate_uid")
    return uid


def _event_id(row: Mapping[str, Any], *, context: str) -> str:
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        raise StrictShuffleError(f"{context} has no event_id")
    return event_id


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise StrictShuffleError(f"{context} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StrictShuffleError(f"{context} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise StrictShuffleError(f"{context} must be an integer")
    if parsed < 0:
        raise StrictShuffleError(f"{context} must be non-negative")
    return parsed


def _positive_int(value: Any, *, context: str) -> int:
    parsed = _nonnegative_int(value, context=context)
    if parsed <= 0:
        raise StrictShuffleError(f"{context} must be positive")
    return parsed


def _append_stats(
    stats: dict[str, int | list[int]],
    build_row: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    for name, value in (
        ("K", int(sidecar["K"])),
        ("prompt_tokens", int(build_row["prompt_token_count"])),
        ("shuffle_order_attempt", int(sidecar["shuffle_order_attempt"])),
        (
            "shuffle_rejected_order_count",
            int(sidecar["shuffle_rejected_order_count"]),
        ),
    ):
        target = stats[name]
        if not isinstance(target, list):
            raise AssertionError(f"stats[{name!r}] is not a list")
        target.append(value)
    for name in ("order_eligible", "order_changed", "verifier_prompt_changed"):
        stats[name] = int(stats[name]) + int(bool(sidecar[name]))


def _arm_summary(
    *,
    arm_spec: Mapping[str, Any],
    split: str,
    n_events: int,
    max_length: int,
    source_build_path: Path,
    config_path: Path,
    build_path: Path,
    sidecar_path: Path,
    stats: Mapping[str, int | Sequence[int]],
    sample_limit: int | None,
) -> dict[str, Any]:
    eligible = int(stats["order_eligible"])
    changed = int(stats["order_changed"])
    prompt_changed = int(stats["verifier_prompt_changed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "arm": str(arm_spec["name"]),
        "seed": int(arm_spec["seed"]),
        "n_events": n_events,
        "sample_limit": sample_limit,
        "max_length": max_length,
        "source_build": str(source_build_path),
        "source_build_sha256": _sha256_file(source_build_path),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "build_output": str(build_path),
        "build_output_sha256": _sha256_file(build_path),
        "sidecar_output": str(sidecar_path),
        "sidecar_output_sha256": _sha256_file(sidecar_path),
        "metrics": {
            "K": _numeric_summary(_as_int_sequence(stats["K"])),
            "prompt_tokens": _numeric_summary(_as_int_sequence(stats["prompt_tokens"])),
            "shuffle_order_attempt": _numeric_summary(
                _as_int_sequence(stats["shuffle_order_attempt"])
            ),
            "shuffle_rejected_order_count": _numeric_summary(
                _as_int_sequence(stats["shuffle_rejected_order_count"])
            ),
            "order_eligible": eligible,
            "order_changed": changed,
            "order_changed_rate_among_eligible": changed / eligible if eligible else None,
            "verifier_prompt_changed": prompt_changed,
            "verifier_prompt_changed_rate": prompt_changed / n_events,
        },
        "contract_checks": {
            "source_visible_prefix_frozen": True,
            "candidate_uid_text_block_multiset_equal": True,
            "evidence_token_content_fingerprint_equal": True,
            "target_and_label_equal": True,
            "prompt_token_multiset_equal": True,
            "prompt_token_count_equal": True,
            "auto_length_false": True,
            "no_retruncation": True,
            "eligible_orders_changed": changed == eligible,
            "presentation_view_only": True,
        },
        "contract": _contract_manifest(max_length=max_length),
    }


def _contract_manifest(*, max_length: int) -> dict[str, Any]:
    return {
        "frozen_source_slice": "source.candidates[:source.evidence_count]",
        "auto_length": False,
        "max_length": max_length,
        "source_evidence_text_truncated_required": False,
        "candidate_fingerprints": "order_invariant_sha256_canonical_json_v1",
        "evidence_token_content_fingerprint": "uid_to_standalone_text_token_ids_v1",
        "prompt_token_content_fingerprint": "order_invariant_token_id_multiset_v1",
        "display_order_fingerprint": "event_scoped_order_sensitive_uid_sequence_v1",
        "random_seed_hash_policy": RANDOM_SEED_HASH_POLICY,
        "identity_fallback_policy": IDENTITY_FALLBACK_POLICY,
        "order_sampling_policy": ORDER_SAMPLING_POLICY,
        "max_shuffle_order_attempts": MAX_SHUFFLE_ORDER_ATTEMPTS,
        "presentation_view_only": True,
        "selector_transition_metadata_replayed": False,
    }


def _as_int_sequence(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, int):
        raise AssertionError("expected a sequence, got scalar")
    return [int(item) for item in value]


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StrictShuffleError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise StrictShuffleError(f"{path}:{line_number} is not a JSON object")
            yield line_number, row


def _json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _temp_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path(path)
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_experiment_config(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    payload = load_yaml(resolved)
    if not isinstance(payload, dict):
        raise StrictShuffleError(f"config {resolved} is not a mapping")
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
