#!/usr/bin/env python3
"""Frozen cross-verifier evaluation for EviTrace main and order-only pairs.

The command has three deliberately separate phases:

* ``prepare`` audits the canonical artifacts and writes gold-free comparisons;
* ``infer`` renders model-specific prompts and scores all six LIAR labels;
* ``analyze`` joins the frozen gold file after inference and runs paired tests.

The implementation is experiment-specific on purpose.  Counts, exclusions,
artifact hashes, prompt schema, and model-facing fields are fail-closed so that
reruns cannot silently drift from the preregistered quick evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
EXPORTER_PATH = (
    PROJECT_ROOT
    / "docs/paper/aaai/annotation_project/scripts/export_exp3_trace_alignment.py"
)

DEFAULT_BUILD_TEST = (
    PROJECT_ROOT
    / "outputs/sentence_trace_method/"
    "liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10/"
    "build/build_test.jsonl"
)
DEFAULT_EVITRACE_TEST = (
    PROJECT_ROOT
    / "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
    "05_mrec_v0_2_learned_marginal_proxy_fullpool/selection_trace_test.jsonl"
)
DEFAULT_S4_TEST = (
    PROJECT_ROOT
    / "outputs/selectors/selector_mechanism_ablation_chunking/"
    "liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_test/"
    "selection_trace_test.jsonl"
)
DEFAULT_BUILD_VAL = DEFAULT_BUILD_TEST.with_name("build_val.jsonl")
DEFAULT_EVITRACE_VAL = DEFAULT_EVITRACE_TEST.with_name("selection_trace_val.jsonl")
DEFAULT_S4_VAL = (
    PROJECT_ROOT
    / "outputs/selectors/selector_mechanism_ablation_chunking/"
    "liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_val/"
    "selection_trace_val.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/analysis/evitrace_cross_verifier_quick_v1"
)

DEFAULT_SEED = 20260724
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_RANDOMIZATION = 100_000
EXPECTED_TEST_EVENTS = 1_251
EXPECTED_VAL_EVENTS = 1_274
EXPECTED_MAIN_COMPARISONS = 1_250
EXPECTED_ORDER_COMPARISONS = 1_152
EXPECTED_ORDER_IDENTICAL = 98
EXPECTED_COMPARISONS = 2_402
EXPECTED_LOGICAL_RESULTS_PER_MODEL = 4_804
EXPECTED_UNIQUE_PROMPTS_PER_MODEL = 3_412
EXCLUDED_TEST_EVENT_IDS = frozenset({"7845.json"})

LIAR6_LABELS = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
LETTERS = ("A", "B", "C", "D", "E", "F")
LETTER_TO_LABEL = dict(zip(LETTERS, LIAR6_LABELS))
LABEL_TO_LETTER = {label: letter for letter, label in LETTER_TO_LABEL.items()}
LABEL_TO_ID = {label: idx for idx, label in enumerate(LIAR6_LABELS)}

FROZEN_ARTIFACT_SHA256 = {
    "build_test": "7499006f339a54b84220174a8fd392f6c5e7afa1aa35b94d648762b7f6860a7f",
    "evitrace_test": "aaa8574d915026100c435b7adfb99cb52e890ab98e21783aff69abb3627f79fa",
    "s4_test": "67d740002d0543724707281a34cc1adce6ad2628efda25ac83b84d9ab58a6705",
    "build_val": "060055f5087b1de8da20711e2416789bde5477a8c6a4b18e42331a74b435c174",
    "evitrace_val": "01d66e246ba5fa79fb6fd3dad8ea7a03f39320f9a8a6e95086ba9a735f3deddf",
    "s4_val": "6869f41bd7fc1ded26fde4581a97be08343fe42e0560990c8d3d6d18cb4734a1",
}

ORDER_LABEL_COUNTS = {
    "pants-fire": 76,
    "false": 230,
    "barely-true": 193,
    "half-true": 243,
    "mostly-true": 217,
    "true": 193,
}
ORDER_COMPLEXITY_COUNTS = {"single": 752, "multi": 400}

FORBIDDEN_PROMPT_MARKERS = (
    "candidate_uid",
    "gold_label",
    "claim_atoms",
    "atom_id",
    "state_before",
    "state_after",
    "map_relation",
    "map_directness",
    "map_confidence",
    "source_score",
    "selector_score",
    "mrec_",
    "evitrace",
    "s4_source",
)
CHECK_CUE = re.compile(r"(?im)^\s*Check:\s*")


class QuickEvalError(RuntimeError):
    """Raised when a frozen experiment contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def preserve_created_at_if_unchanged(
    path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Keep a manifest byte-stable across semantically identical reruns."""

    destination = Path(path)
    if not destination.exists():
        return manifest
    try:
        existing = load_json(destination)
    except (OSError, json.JSONDecodeError, QuickEvalError):
        return manifest
    existing_without_time = dict(existing)
    candidate_without_time = dict(manifest)
    existing_without_time.pop("created_at", None)
    candidate_without_time.pop("created_at", None)
    if existing_without_time == candidate_without_time and existing.get("created_at"):
        manifest = dict(manifest)
        manifest["created_at"] = existing["created_at"]
    return manifest


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return count


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QuickEvalError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuickEvalError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise QuickEvalError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _load_exporter():
    if not EXPORTER_PATH.exists():
        raise QuickEvalError(f"Missing canonical exporter: {EXPORTER_PATH}")
    module_name = "_evitrace_exp3_exporter_for_cross_verifier"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise QuickEvalError(f"Cannot import exporter: {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def recompute_s4_source_score_order(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the canonical exporter's exact standard-library rank mirror."""

    return _load_exporter().recompute_s4_source_score_order(candidates)


def _assert_frozen_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.exists():
            raise QuickEvalError(f"Missing source artifact: {resolved}")
        digest = sha256_file(resolved)
        expected = FROZEN_ARTIFACT_SHA256[name]
        if digest != expected:
            raise QuickEvalError(
                f"Frozen SHA mismatch for {name}: expected {expected}, found {digest}"
            )
        metadata[name] = {
            "path": str(resolved),
            "sha256": digest,
            "bytes": resolved.stat().st_size,
        }
    return metadata


def _candidate_arm(event: Any, uids: Sequence[str], method: str) -> dict[str, Any]:
    candidates = [event.candidates_by_uid[uid] for uid in uids]
    texts = [candidate.text for candidate in candidates]
    if len(candidates) != event.k_visible:
        raise QuickEvalError(
            f"{event.event_id}: {method} has {len(candidates)} candidates, "
            f"expected K_visible={event.k_visible}"
        )
    for text in texts:
        if not text.strip():
            raise QuickEvalError(f"{event.event_id}: empty clean evidence text")
        if CHECK_CUE.search(text):
            raise QuickEvalError(f"{event.event_id}: Check: cue leaked into clean evidence")
    return {
        "method": method,
        "candidate_uids": list(uids),
        "evidence_texts": texts,
        "evidence_multiset_sha256": sha256_text(
            canonical_json(sorted(Counter(texts).items()))
        ),
        "character_count": sum(len(text) for text in texts),
    }


def _comparison_row(event: Any, comparison_type: str) -> dict[str, Any]:
    evitrace = _candidate_arm(
        event,
        event.evi_visible_uids,
        "evitrace_visible_selection"
        if comparison_type == "main"
        else "evitrace_order",
    )
    if comparison_type == "main":
        control_uids = event.s4_order_uids[: event.k_visible]
        control_method = "s4_source_score_top_k"
    elif comparison_type == "order_only":
        control_uids = event.s4_reordered_evi_uids
        control_method = "s4_source_score_reorder_same_set"
    else:
        raise QuickEvalError(f"Unknown comparison type: {comparison_type}")
    control = _candidate_arm(event, control_uids, control_method)
    row = {
        "comparison_id": f"{event.event_id}::{comparison_type}",
        "event_id": event.event_id,
        "split": event.split,
        "comparison_type": comparison_type,
        "claim": event.claim,
        "complexity": event.complexity,
        "atom_count": len(event.atoms),
        "k_visible": event.k_visible,
        "k_selected": event.k_selected,
        "arms": {"evitrace": evitrace, "control": control},
    }
    if comparison_type == "order_only":
        validate_order_only_pair(row)
    return row


def validate_order_only_pair(row: Mapping[str, Any]) -> None:
    if row.get("comparison_type") != "order_only":
        raise QuickEvalError("validate_order_only_pair requires comparison_type=order_only")
    arms = row.get("arms")
    if not isinstance(arms, Mapping):
        raise QuickEvalError("Order-only row has no arms object")
    evitrace = arms.get("evitrace")
    control = arms.get("control")
    if not isinstance(evitrace, Mapping) or not isinstance(control, Mapping):
        raise QuickEvalError("Order-only row is missing an arm")
    evi_uids = list(evitrace.get("candidate_uids") or [])
    ctl_uids = list(control.get("candidate_uids") or [])
    evi_texts = list(evitrace.get("evidence_texts") or [])
    ctl_texts = list(control.get("evidence_texts") or [])
    if len(evi_uids) != len(ctl_uids) or not evi_uids:
        raise QuickEvalError(f"{row.get('event_id')}: order-only count mismatch")
    if set(evi_uids) != set(ctl_uids):
        raise QuickEvalError(f"{row.get('event_id')}: order-only UID set mismatch")
    if evi_uids == ctl_uids:
        raise QuickEvalError(f"{row.get('event_id')}: identical order was not excluded")
    if Counter(evi_texts) != Counter(ctl_texts):
        raise QuickEvalError(f"{row.get('event_id')}: order-only text multiset mismatch")
    if sum(map(len, evi_texts)) != sum(map(len, ctl_texts)):
        raise QuickEvalError(f"{row.get('event_id')}: order-only character mismatch")


def _gold_row(event: Any) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "gold_label": event.gold_label,
        "complexity": event.complexity,
        "atom_count": len(event.atoms),
    }


def _deterministic_val_smoke(events: Sequence[Any], seed: int) -> list[Any]:
    selected: list[Any] = []
    for label in LIAR6_LABELS:
        pool = [
            event
            for event in events
            if event.gold_label == label and not event.order_is_identical
        ]
        pool.sort(
            key=lambda event: (
                sha256_text(f"{seed}:val-smoke:{label}:{event.event_id}"),
                event.event_id,
            )
        )
        if len(pool) < 2:
            raise QuickEvalError(f"Val smoke has fewer than two eligible {label} events")
        selected.extend(pool[:2])
    if len(selected) != 12 or len({event.event_id for event in selected}) != 12:
        raise QuickEvalError("Val smoke selection must contain 12 unique events")
    return selected


def _prepared_file_metadata(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def prepare_experiment(args: argparse.Namespace) -> dict[str, Any]:
    source_paths = {
        "build_test": Path(args.build),
        "evitrace_test": Path(args.evitrace),
        "s4_test": Path(args.s4),
        "build_val": Path(args.build_val),
        "evitrace_val": Path(args.evitrace_val),
        "s4_val": Path(args.s4_val),
    }
    source_metadata = _assert_frozen_artifacts(source_paths)
    exporter = _load_exporter()
    current_ranker = exporter._load_current_s4_ranker()

    test_events, test_audit = exporter.load_aligned_events(
        source_paths["build_test"],
        source_paths["evitrace_test"],
        source_paths["s4_test"],
        split="test",
        expected_event_count=EXPECTED_TEST_EVENTS,
        current_s4_ranker=current_ranker,
    )
    val_events, val_audit = exporter.load_aligned_events(
        source_paths["build_val"],
        source_paths["evitrace_val"],
        source_paths["s4_val"],
        split="val",
        expected_event_count=EXPECTED_VAL_EVENTS,
        current_s4_ranker=current_ranker,
    )

    excluded_found = {
        event.event_id for event in test_events if event.event_id in EXCLUDED_TEST_EVENT_IDS
    }
    if excluded_found != set(EXCLUDED_TEST_EVENT_IDS):
        raise QuickEvalError(
            f"Expected excluded events {sorted(EXCLUDED_TEST_EVENT_IDS)}, "
            f"found {sorted(excluded_found)}"
        )
    eligible = [
        event for event in test_events if event.event_id not in EXCLUDED_TEST_EVENT_IDS
    ]
    if len(eligible) != EXPECTED_MAIN_COMPARISONS:
        raise QuickEvalError(
            f"Expected {EXPECTED_MAIN_COMPARISONS} eligible test events, found {len(eligible)}"
        )

    main_rows = [_comparison_row(event, "main") for event in eligible]
    identical_order_events = [event for event in eligible if event.order_is_identical]
    order_events = [event for event in eligible if not event.order_is_identical]
    if len(identical_order_events) != EXPECTED_ORDER_IDENTICAL:
        raise QuickEvalError(
            f"Expected {EXPECTED_ORDER_IDENTICAL} identical order events, "
            f"found {len(identical_order_events)}"
        )
    if len(order_events) != EXPECTED_ORDER_COMPARISONS:
        raise QuickEvalError(
            f"Expected {EXPECTED_ORDER_COMPARISONS} order comparisons, "
            f"found {len(order_events)}"
        )
    order_rows = [_comparison_row(event, "order_only") for event in order_events]
    comparisons = main_rows + order_rows
    if len(comparisons) != EXPECTED_COMPARISONS:
        raise QuickEvalError(
            f"Expected {EXPECTED_COMPARISONS} comparisons, found {len(comparisons)}"
        )
    comparison_ids = [row["comparison_id"] for row in comparisons]
    if len(comparison_ids) != len(set(comparison_ids)):
        raise QuickEvalError("Duplicate comparison_id in prepared test data")

    order_label_counts = Counter(event.gold_label for event in order_events)
    order_complexity_counts = Counter(event.complexity for event in order_events)
    if dict(order_label_counts) != ORDER_LABEL_COUNTS:
        raise QuickEvalError(
            f"Order label distribution drift: {dict(order_label_counts)}"
        )
    if dict(order_complexity_counts) != ORDER_COMPLEXITY_COUNTS:
        raise QuickEvalError(
            f"Order complexity distribution drift: {dict(order_complexity_counts)}"
        )

    exact_sequence = 0
    same_set_different_order = 0
    different_set = 0
    for row in main_rows:
        evi_uids = row["arms"]["evitrace"]["candidate_uids"]
        ctl_uids = row["arms"]["control"]["candidate_uids"]
        if evi_uids == ctl_uids:
            exact_sequence += 1
        elif set(evi_uids) == set(ctl_uids):
            same_set_different_order += 1
        else:
            different_set += 1
    main_relation_counts = {
        "exact_sequence": exact_sequence,
        "same_set_different_order": same_set_different_order,
        "different_set": different_set,
    }
    if main_relation_counts != {
        "exact_sequence": 33,
        "same_set_different_order": 207,
        "different_set": 1_010,
    }:
        raise QuickEvalError(f"Main pair relation drift: {main_relation_counts}")

    val_selected = _deterministic_val_smoke(val_events, int(args.seed))
    val_comparisons = [
        row
        for event in val_selected
        for row in (
            _comparison_row(event, "main"),
            _comparison_row(event, "order_only"),
        )
    ]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_path = output_dir / "comparisons_test.jsonl"
    gold_path = output_dir / "gold_test.jsonl"
    val_path = output_dir / "comparisons_val_smoke.jsonl"
    excluded_path = output_dir / "excluded_order_same.jsonl"

    comparison_count = write_jsonl(comparisons_path, comparisons)
    gold_count = write_jsonl(gold_path, (_gold_row(event) for event in eligible))
    val_count = write_jsonl(val_path, val_comparisons)
    excluded_count = write_jsonl(
        excluded_path,
        (
            {
                "event_id": event.event_id,
                "reason": "evitrace_and_s4_reordered_visible_set_have_identical_order",
                "k_visible": event.k_visible,
                "candidate_uids": list(event.evi_visible_uids),
            }
            for event in identical_order_events
        ),
    )
    if (
        comparison_count != EXPECTED_COMPARISONS
        or gold_count != EXPECTED_MAIN_COMPARISONS
        or val_count != 24
        or excluded_count != EXPECTED_ORDER_IDENTICAL
    ):
        raise QuickEvalError("Prepared output row count mismatch")

    prepared_files = {
        "comparisons_test": _prepared_file_metadata(comparisons_path, comparison_count),
        "gold_test": _prepared_file_metadata(gold_path, gold_count),
        "comparisons_val_smoke": _prepared_file_metadata(val_path, val_count),
        "excluded_order_same": _prepared_file_metadata(excluded_path, excluded_count),
    }
    manifest = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_quick_v1",
        "created_at": utc_now(),
        "seed": int(args.seed),
        "complete": True,
        "source_artifacts": source_metadata,
        "source_audits": {"test": test_audit, "val": val_audit},
        "prepared_files": prepared_files,
        "code": {
            "cross_verifier_quick": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "canonical_exporter": {
                "path": str(EXPORTER_PATH.resolve()),
                "sha256": sha256_file(EXPORTER_PATH),
            },
        },
        "contracts": {
            "excluded_test_event_ids": sorted(EXCLUDED_TEST_EVENT_IDS),
            "k_field": "evidence_count",
            "forbidden_k_field": "evidence_count_before",
            "test_event_count_aligned": len(test_events),
            "main_comparison_count": len(main_rows),
            "order_identical_preexcluded_count": len(identical_order_events),
            "order_comparison_count": len(order_rows),
            "comparison_count": len(comparisons),
            "val_smoke_claim_count": len(val_selected),
            "val_smoke_comparison_count": len(val_comparisons),
            "main_pair_relation_counts": main_relation_counts,
            "order_label_counts": {
                label: order_label_counts[label] for label in LIAR6_LABELS
            },
            "order_complexity_counts": {
                key: order_complexity_counts[key] for key in ("single", "multi")
            },
            "gold_is_separate_from_comparisons": True,
            "prompt_fields": ["claim", "arms.*.evidence_texts"],
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = preserve_created_at_if_unchanged(manifest_path, manifest)
    write_json(manifest_path, manifest)
    print(
        f"Prepared {len(main_rows)} main and {len(order_rows)} order-only comparisons "
        f"at {output_dir}"
    )
    return manifest


def validate_prompt_text(text: str) -> None:
    if not text.strip():
        raise QuickEvalError("Rendered prompt is empty")
    if CHECK_CUE.search(text):
        raise QuickEvalError("Rendered prompt contains a Check: atom cue")
    lowered = text.lower()
    leaked = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in lowered]
    if leaked:
        raise QuickEvalError(f"Rendered prompt contains forbidden markers: {leaked}")


def normalize_label_logprobs(
    label_logprobs: Mapping[str, float],
) -> dict[str, float]:
    if set(label_logprobs) != set(LETTERS):
        raise QuickEvalError(
            f"Expected exactly A-F label scores, found {sorted(label_logprobs)}"
        )
    values = {letter: float(label_logprobs[letter]) for letter in LETTERS}
    if any(not math.isfinite(value) for value in values.values()):
        raise QuickEvalError(f"Non-finite label log-probabilities: {values}")
    maximum = max(values.values())
    log_denom = maximum + math.log(
        sum(math.exp(value - maximum) for value in values.values())
    )
    return {letter: values[letter] - log_denom for letter in LETTERS}


def _model_prompt_rows(
    comparisons: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from fact_checking.build.prompts import (
        build_chat_prompt,
        build_system_message,
        build_user_content,
    )

    chat_template: dict[str, Any] = {
        "template_kwargs": {"enable_thinking": False}
    }
    system_message = build_system_message(None, "liar6")
    logical_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        arms = comparison.get("arms")
        if not isinstance(arms, Mapping):
            raise QuickEvalError(
                f"{comparison.get('comparison_id')}: missing arms object"
            )
        for arm_name in ("evitrace", "control"):
            arm = arms.get(arm_name)
            if not isinstance(arm, Mapping):
                raise QuickEvalError(
                    f"{comparison.get('comparison_id')}: missing {arm_name} arm"
                )
            evidence_texts = list(arm.get("evidence_texts") or [])
            user_content = build_user_content(
                str(comparison["claim"]),
                evidence_texts,
                "label",
                "letter",
                "liar6",
            )
            rendered_chat = build_chat_prompt(
                tokenizer,
                system_message,
                user_content,
                chat_template=chat_template,
            )
            # Preserve the assistant-role newline emitted by the tokenizer
            # template.  Stripping it would produce malformed suffixes such as
            # ``assistantLabel:`` for Qwen and ``end_header_id|>Label:`` for
            # Llama.
            prompt = rendered_chat + "Label:"
            validate_prompt_text(prompt)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ]
            chat_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(chat_ids, "tolist"):
                chat_ids = chat_ids.tolist()
            if chat_ids and isinstance(chat_ids[0], list):
                if len(chat_ids) != 1:
                    raise QuickEvalError("Expected one tokenized chat prompt")
                chat_ids = chat_ids[0]
            prefix_ids = tokenizer(
                "Label:",
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            input_ids = [int(token_id) for token_id in chat_ids] + [
                int(token_id) for token_id in prefix_ids
            ]
            roundtrip_ids = [
                int(token_id)
                for token_id in tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            ]
            if roundtrip_ids != input_ids:
                raise QuickEvalError(
                    f"{comparison['comparison_id']}::{arm_name}: tokenized chat "
                    "template differs from rendered prompt round-trip"
                )
            if len(input_ids) + 2 > 2_048:
                raise QuickEvalError(
                    f"{comparison['comparison_id']}::{arm_name}: "
                    f"prompt plus canonical label/generation has {len(input_ids) + 2} "
                    "tokens (>2048); truncation is forbidden"
                )
            logical_id = f"{comparison['comparison_id']}::{arm_name}"
            logical_rows.append(
                {
                    "logical_id": logical_id,
                    "comparison_id": comparison["comparison_id"],
                    "event_id": comparison["event_id"],
                    "comparison_type": comparison["comparison_type"],
                    "complexity": comparison["complexity"],
                    "arm": arm_name,
                    "method": arm["method"],
                    "k_visible": comparison["k_visible"],
                    "candidate_uids": list(arm["candidate_uids"]),
                    "evidence_multiset_sha256": arm["evidence_multiset_sha256"],
                    "character_count": arm["character_count"],
                    "prompt_text": prompt,
                    "prompt_text_sha256": sha256_text(prompt),
                    "input_ids": input_ids,
                    "input_ids_sha256": sha256_text(canonical_json(input_ids)),
                    "prompt_token_count": len(input_ids),
                }
            )
    return logical_rows


def make_prompt_registry(
    logical_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate fully rendered prompts and return registry plus logical refs."""

    registry_by_hash: dict[str, dict[str, Any]] = {}
    refs: list[dict[str, Any]] = []
    seen_logical: set[str] = set()
    for raw in logical_rows:
        row = dict(raw)
        logical_id = str(row.get("logical_id") or "")
        digest = str(row.get("input_ids_sha256") or "")
        input_ids = [int(value) for value in row.get("input_ids") or []]
        if not logical_id or logical_id in seen_logical:
            raise QuickEvalError(f"Missing/duplicate logical_id: {logical_id!r}")
        if not digest or not input_ids:
            raise QuickEvalError(f"{logical_id}: missing prompt hash/input IDs")
        expected_digest = sha256_text(canonical_json(input_ids))
        if digest != expected_digest:
            raise QuickEvalError(f"{logical_id}: input_ids SHA mismatch")
        seen_logical.add(logical_id)
        existing = registry_by_hash.get(digest)
        if existing is None:
            existing = {
                "input_ids_sha256": digest,
                "prompt_text_sha256": row.get("prompt_text_sha256"),
                "prompt_text": row.get("prompt_text"),
                "input_ids": input_ids,
                "prompt_token_count": len(input_ids),
                "logical_ids": [],
            }
            registry_by_hash[digest] = existing
        else:
            if existing["input_ids"] != input_ids:
                raise QuickEvalError("SHA-256 collision across distinct input IDs")
            if existing["prompt_text"] != row.get("prompt_text"):
                raise QuickEvalError(
                    "Identical input IDs arose from different rendered prompt text"
                )
        existing["logical_ids"].append(logical_id)
        refs.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"prompt_text", "input_ids"}
            }
        )
    registry = list(registry_by_hash.values())
    registry.sort(key=lambda row: row["input_ids_sha256"])
    refs.sort(key=lambda row: row["logical_id"])
    return registry, refs


def expand_logical_results(
    logical_rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(scores, Mapping):
        score_by_hash = {str(key): dict(value) for key, value in scores.items()}
    else:
        score_by_hash = {
            str(row["input_ids_sha256"]): dict(row) for row in scores
        }
    expanded: list[dict[str, Any]] = []
    for logical in logical_rows:
        digest = str(logical["input_ids_sha256"])
        score = score_by_hash.get(digest)
        if score is None:
            raise QuickEvalError(
                f"Missing score for {logical.get('logical_id')} ({digest})"
            )
        merged = dict(logical)
        for key, value in score.items():
            if key in {
                "input_ids_sha256",
                "prompt_text_sha256",
                "prompt_token_count",
            }:
                if key in merged and merged[key] != value:
                    raise QuickEvalError(
                        f"{logical.get('logical_id')}: score metadata mismatch for {key}"
                    )
            else:
                merged[key] = value
        expanded.append(merged)
    expanded.sort(key=lambda row: row["logical_id"])
    return expanded


def validate_resume_row(
    row: Mapping[str, Any],
    *,
    model_sha: str,
    scoring_config_sha: str | None = None,
) -> None:
    if row.get("model_sha256") != model_sha:
        raise QuickEvalError("Resume score model SHA mismatch")
    if scoring_config_sha is not None and row.get("scoring_config_sha256") != scoring_config_sha:
        raise QuickEvalError("Resume score configuration SHA mismatch")
    digest = str(row.get("input_ids_sha256") or "")
    if len(digest) != 64:
        raise QuickEvalError("Resume score has invalid input_ids SHA")
    normalized = row.get("label_logprobs")
    if not isinstance(normalized, Mapping):
        raise QuickEvalError("Resume score has no label_logprobs")
    checked = normalize_label_logprobs(
        {letter: float(normalized[letter]) for letter in normalized}
    )
    if max(abs(float(normalized[key]) - checked[key]) for key in LETTERS) > 1e-6:
        raise QuickEvalError("Resume label log-probabilities are not normalized")
    probabilities = row.get("label_probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(LETTERS):
        raise QuickEvalError("Resume score has invalid label probabilities")
    if abs(sum(float(probabilities[key]) for key in LETTERS) - 1.0) > 1e-6:
        raise QuickEvalError("Resume label probabilities do not sum to one")


def load_resume_scores(
    path: str | Path,
    *,
    model_sha: str,
    scoring_config_sha: str | None = None,
) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return {}
    scores: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(source):
        validate_resume_row(
            row,
            model_sha=model_sha,
            scoring_config_sha=scoring_config_sha,
        )
        digest = str(row["input_ids_sha256"])
        if digest in scores:
            raise QuickEvalError(f"Duplicate resume score for {digest}")
        scores[digest] = row
    return scores


def _directory_sha256(
    path: str | Path,
    *,
    exclude_weight_files: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Hash a model/tokenizer directory without trusting names or mtimes."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise QuickEvalError(f"Model/tokenizer path is not a directory: {root}")
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix()
        if exclude_weight_files and file_path.suffix.lower() in weight_suffixes:
            continue
        file_digest = sha256_file(file_path)
        entry = {
            "path": relative,
            "bytes": file_path.stat().st_size,
            "sha256": file_digest,
        }
        entries.append(entry)
        digest.update(canonical_json(entry).encode("utf-8"))
        digest.update(b"\n")
    if not entries:
        raise QuickEvalError(f"No files found while hashing directory: {root}")
    return digest.hexdigest(), entries


def _resolve_gpu_id(gpu_id: str) -> str:
    requested = str(gpu_id).strip().lower()
    if requested != "auto":
        if not requested.isdigit():
            raise QuickEvalError(f"--gpu-id must be a physical integer or auto: {gpu_id}")
        return requested
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates: list[tuple[int, int]] = []
    for line in process.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
            raise QuickEvalError(f"Cannot parse nvidia-smi row: {line!r}")
        candidates.append((int(pieces[1]), int(pieces[0])))
    if not candidates:
        raise QuickEvalError("nvidia-smi returned no GPUs")
    _free_mib, selected = max(candidates, key=lambda item: (item[0], -item[1]))
    return str(selected)


def _label_token_ids(tokenizer: Any) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    for letter in LETTERS:
        ids = tokenizer(
            f" {letter}",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(ids) != 1:
            raise QuickEvalError(
                f"Label choice {letter!r} must be one token, found IDs {ids}"
            )
        token_ids[letter] = int(ids[0])
    if len(set(token_ids.values())) != len(LETTERS):
        raise QuickEvalError(f"Label token IDs are not unique: {token_ids}")
    return token_ids


def _logprob_value(entry: Any) -> float:
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, Mapping):
        for key in ("logprob", "token_logprob"):
            if key in entry:
                return float(entry[key])
    return float(entry)


def _lookup_token_logprob(values: Any, token_id: int) -> float:
    if isinstance(values, Mapping):
        for key in (token_id, str(token_id)):
            if key in values:
                return _logprob_value(values[key])
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for entry in values:
            if isinstance(entry, Mapping):
                found_id = entry.get("token_id", entry.get("id"))
                if found_id is not None and int(found_id) == int(token_id):
                    return _logprob_value(entry)
    raise QuickEvalError(f"Token log-probability missing for token_id={token_id}")


def extract_canonical_prompt_logprob(output: Any, token_id: int) -> float:
    prompt_token_ids = list(getattr(output, "prompt_token_ids", None) or [])
    if not prompt_token_ids or int(prompt_token_ids[-1]) != int(token_id):
        raise QuickEvalError(
            f"Canonical output does not end in expected label token {token_id}"
        )
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if not isinstance(prompt_logprobs, list) or not prompt_logprobs:
        raise QuickEvalError("Canonical output did not include prompt_logprobs")
    for item in reversed(prompt_logprobs):
        if item:
            value = _lookup_token_logprob(item, token_id)
            if not math.isfinite(value):
                raise QuickEvalError("Canonical label log-probability is non-finite")
            return value
    raise QuickEvalError("Canonical output has no non-empty prompt log-probability")


def extract_direct_label_logprobs(
    output: Any,
    label_token_ids: Mapping[str, int],
) -> dict[str, float]:
    completions = list(getattr(output, "outputs", None) or [])
    if len(completions) != 1:
        raise QuickEvalError("Direct scoring expected exactly one completion")
    steps = getattr(completions[0], "logprobs", None)
    if not isinstance(steps, list) or len(steps) != 1 or not steps[0]:
        raise QuickEvalError("Direct scoring did not return one token logprob map")
    raw = {
        letter: _lookup_token_logprob(steps[0], token_id)
        for letter, token_id in label_token_ids.items()
    }
    return normalize_label_logprobs(raw)


def _score_record(
    registry_row: Mapping[str, Any],
    label_logprobs: Mapping[str, float],
    *,
    model_name: str,
    model_sha: str,
    tokenizer_sha: str,
    scoring_config_sha: str,
    scoring_method: str,
) -> dict[str, Any]:
    normalized = normalize_label_logprobs(label_logprobs)
    probabilities = {
        letter: math.exp(normalized[letter]) for letter in LETTERS
    }
    pred_letter = max(LETTERS, key=lambda letter: normalized[letter])
    return {
        "input_ids_sha256": registry_row["input_ids_sha256"],
        "prompt_text_sha256": registry_row["prompt_text_sha256"],
        "prompt_token_count": registry_row["prompt_token_count"],
        "model_name": model_name,
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "scoring_config_sha256": scoring_config_sha,
        "scoring_method": scoring_method,
        "label_logprobs": normalized,
        "label_probabilities": probabilities,
        "pred_letter": pred_letter,
        "pred_label": LETTER_TO_LABEL[pred_letter],
    }


def _score_direct_batch(
    llm: Any,
    registry: Sequence[Mapping[str, Any]],
    label_token_ids: Mapping[str, int],
) -> list[dict[str, float]]:
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        allowed_token_ids=list(label_token_ids.values()),
        logprobs=len(LETTERS),
        detokenize=False,
    )
    outputs = llm.generate(
        prompt_token_ids=[list(row["input_ids"]) for row in registry],
        sampling_params=params,
        use_tqdm=False,
    )
    if len(outputs) != len(registry):
        raise QuickEvalError("Direct scoring output count mismatch")
    return [
        extract_direct_label_logprobs(output, label_token_ids)
        for output in outputs
    ]


def _score_canonical_batch(
    llm: Any,
    registry: Sequence[Mapping[str, Any]],
    label_token_ids: Mapping[str, int],
) -> list[dict[str, float]]:
    from vllm import SamplingParams

    expanded: list[list[int]] = []
    expected: list[tuple[int, str, int]] = []
    for row_index, row in enumerate(registry):
        base_ids = [int(value) for value in row["input_ids"]]
        for letter in LETTERS:
            token_id = int(label_token_ids[letter])
            expanded.append(base_ids + [token_id])
            expected.append((row_index, letter, token_id))
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        prompt_logprobs=0,
        detokenize=False,
    )
    outputs = llm.generate(
        prompt_token_ids=expanded,
        sampling_params=params,
        use_tqdm=False,
    )
    if len(outputs) != len(expected):
        raise QuickEvalError("Canonical scoring output count mismatch")
    raw_by_row: list[dict[str, float]] = [dict() for _ in registry]
    for output, (row_index, letter, token_id) in zip(outputs, expected):
        if letter in raw_by_row[row_index]:
            raise QuickEvalError("Duplicate canonical label score")
        raw_by_row[row_index][letter] = extract_canonical_prompt_logprob(
            output, token_id
        )
    return [normalize_label_logprobs(scores) for scores in raw_by_row]


def _score_method_batch(
    method: str,
    llm: Any,
    registry: Sequence[Mapping[str, Any]],
    label_token_ids: Mapping[str, int],
) -> list[dict[str, float]]:
    if method == "allowed_token":
        return _score_direct_batch(llm, registry, label_token_ids)
    if method == "canonical_six_continuation":
        return _score_canonical_batch(llm, registry, label_token_ids)
    raise QuickEvalError(f"Unknown scoring method: {method}")


def _smoke_parity(
    llm: Any,
    smoke_registry: Sequence[Mapping[str, Any]],
    label_token_ids: Mapping[str, int],
) -> tuple[str, dict[str, Any]]:
    canonical = _score_canonical_batch(llm, smoke_registry, label_token_ids)
    report: dict[str, Any] = {
        "prompt_count": len(smoke_registry),
        "canonical_complete": True,
        "threshold": 1.0e-3,
    }
    try:
        direct = _score_direct_batch(llm, smoke_registry, label_token_ids)
    except (QuickEvalError, KeyError, RuntimeError, ValueError) as exc:
        report.update(
            {
                "direct_complete": False,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
                "selected_method": "canonical_six_continuation",
            }
        )
        return "canonical_six_continuation", report

    max_difference = 0.0
    prediction_mismatches = 0
    for direct_scores, canonical_scores in zip(direct, canonical):
        max_difference = max(
            max_difference,
            max(
                abs(direct_scores[letter] - canonical_scores[letter])
                for letter in LETTERS
            ),
        )
        direct_pred = max(LETTERS, key=lambda letter: direct_scores[letter])
        canonical_pred = max(LETTERS, key=lambda letter: canonical_scores[letter])
        prediction_mismatches += int(direct_pred != canonical_pred)
    parity_passed = prediction_mismatches == 0 and max_difference <= 1.0e-3
    report.update(
        {
            "direct_complete": True,
            "prediction_mismatch_count": prediction_mismatches,
            "max_abs_normalized_logprob_difference": max_difference,
            "parity_passed": parity_passed,
            "selected_method": (
                "allowed_token" if parity_passed else "canonical_six_continuation"
            ),
        }
    )
    if not parity_passed:
        report["fallback_reason"] = "allowed-token/canonical parity check failed"
    return str(report["selected_method"]), report


def _verify_prepared_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if not manifest.get("complete"):
        raise QuickEvalError("Prepared manifest is not complete")
    if manifest.get("experiment") != "evitrace_cross_verifier_quick_v1":
        raise QuickEvalError("Prepared manifest belongs to another experiment")
    for metadata in manifest.get("prepared_files", {}).values():
        path = Path(str(metadata["path"]))
        if sha256_file(path) != metadata["sha256"]:
            raise QuickEvalError(f"Prepared file SHA mismatch: {path}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise QuickEvalError(f"Prepared file size mismatch: {path}")
    return manifest


def _logical_prompt_audit(
    logical_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in logical_rows:
        key = (str(row["event_id"]), str(row["comparison_type"]))
        grouped[key][str(row["arm"])] = row
    order_nonzero_tokens: list[dict[str, Any]] = []
    order_character_mismatch = 0
    main_within64 = 0
    for (event_id, comparison_type), arms in grouped.items():
        if set(arms) != {"evitrace", "control"}:
            raise QuickEvalError(f"{event_id}/{comparison_type}: incomplete prompt arms")
        evi = arms["evitrace"]
        ctl = arms["control"]
        token_difference = int(evi["prompt_token_count"]) - int(
            ctl["prompt_token_count"]
        )
        if comparison_type == "main":
            main_within64 += int(abs(token_difference) <= 64)
        else:
            if int(evi["character_count"]) != int(ctl["character_count"]):
                order_character_mismatch += 1
            if token_difference != 0:
                order_nonzero_tokens.append(
                    {"event_id": event_id, "difference_evi_minus_control": token_difference}
                )
    if order_character_mismatch:
        raise QuickEvalError("Order-only prompt character counts differ")
    if order_nonzero_tokens != [
        {"event_id": "1290.json", "difference_evi_minus_control": -1}
    ]:
        raise QuickEvalError(
            f"Unexpected order-only token differences: {order_nonzero_tokens}"
        )
    return {
        "main_within64_count": main_within64,
        "order_nonzero_token_differences": order_nonzero_tokens,
        "max_prompt_token_count": max(
            int(row["prompt_token_count"]) for row in logical_rows
        ),
    }


def _completed_inference_is_reusable(
    runtime_manifest_path: Path,
    *,
    prepared_manifest_sha: str,
    model_sha: str,
    tokenizer_sha: str,
) -> bool:
    if not runtime_manifest_path.exists():
        return False
    runtime = load_json(runtime_manifest_path)
    if not runtime.get("complete"):
        return False
    if (
        runtime.get("prepared_manifest_sha256") != prepared_manifest_sha
        or runtime.get("model_sha256") != model_sha
        or runtime.get("tokenizer_sha256") != tokenizer_sha
    ):
        return False
    for name in ("unique_scores", "logical_results", "prompt_registry", "logical_refs"):
        metadata = runtime.get("files", {}).get(name)
        if not isinstance(metadata, Mapping):
            return False
        path = Path(str(metadata.get("path") or ""))
        if not path.exists() or sha256_file(path) != metadata.get("sha256"):
            return False
    return True


def infer_model(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.prepared_manifest).resolve()
    prepared = _verify_prepared_manifest(manifest_path)
    prepared_manifest_sha = sha256_file(manifest_path)
    selected_gpu = _resolve_gpu_id(str(args.gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from sft.runtime.model_loading import load_compatible_tokenizer

    model_path = Path(args.model_path).resolve()
    tokenizer = load_compatible_tokenizer(str(model_path), trust_remote_code=True)
    label_token_ids = _label_token_ids(tokenizer)
    model_sha, model_files = _directory_sha256(model_path)
    tokenizer_sha, tokenizer_files = _directory_sha256(
        model_path, exclude_weight_files=True
    )

    comparisons_path = Path(
        prepared["prepared_files"]["comparisons_test"]["path"]
    )
    smoke_path = Path(
        prepared["prepared_files"]["comparisons_val_smoke"]["path"]
    )
    comparisons = load_jsonl(comparisons_path)
    smoke_comparisons = load_jsonl(smoke_path)
    logical_full = _model_prompt_rows(
        comparisons, tokenizer, model_name=str(args.model_name)
    )
    logical_smoke = _model_prompt_rows(
        smoke_comparisons, tokenizer, model_name=str(args.model_name)
    )
    registry, logical_refs = make_prompt_registry(logical_full)
    smoke_registry, _smoke_refs = make_prompt_registry(logical_smoke)
    if len(logical_refs) != EXPECTED_LOGICAL_RESULTS_PER_MODEL:
        raise QuickEvalError(
            f"Expected {EXPECTED_LOGICAL_RESULTS_PER_MODEL} logical prompts, "
            f"found {len(logical_refs)}"
        )
    if len(registry) != EXPECTED_UNIQUE_PROMPTS_PER_MODEL:
        raise QuickEvalError(
            f"Expected {EXPECTED_UNIQUE_PROMPTS_PER_MODEL} unique prompts, "
            f"found {len(registry)}"
        )
    prompt_audit = _logical_prompt_audit(logical_refs)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_registry_path = output_dir / "prompt_registry.jsonl"
    logical_refs_path = output_dir / "logical_refs.jsonl"
    unique_scores_path = output_dir / "unique_scores.jsonl"
    logical_results_path = output_dir / "logical_results.jsonl"
    smoke_report_path = output_dir / "smoke_parity.json"
    runtime_manifest_path = output_dir / "runtime_manifest.json"
    write_jsonl(prompt_registry_path, registry)
    write_jsonl(logical_refs_path, logical_refs)

    if _completed_inference_is_reusable(
        runtime_manifest_path,
        prepared_manifest_sha=prepared_manifest_sha,
        model_sha=model_sha,
        tokenizer_sha=tokenizer_sha,
    ):
        print(f"Reusing complete inference at {output_dir}")
        return load_json(runtime_manifest_path)

    # Import vLLM only after CUDA_VISIBLE_DEVICES has been set.
    from vllm import LLM
    import transformers
    import vllm

    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        task="generate",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=2_048,
        gpu_memory_utilization=0.80,
        swap_space=0,
        seed=int(args.seed),
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    selected_method, smoke_report = _smoke_parity(
        llm, smoke_registry, label_token_ids
    )
    smoke_report.update(
        {
            "model_name": args.model_name,
            "model_sha256": model_sha,
            "tokenizer_sha256": tokenizer_sha,
            "label_token_ids": label_token_ids,
            "thinking_enabled": False,
        }
    )
    write_json(smoke_report_path, smoke_report)

    scoring_config = {
        "schema_version": 1,
        "prepared_manifest_sha256": prepared_manifest_sha,
        "model_name": str(args.model_name),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "label_token_ids": label_token_ids,
        "selected_method": selected_method,
        "max_model_len": 2_048,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.80,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "seed": int(args.seed),
        "thinking_enabled": False,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    scoring_config_sha = sha256_text(canonical_json(scoring_config))
    existing = load_resume_scores(
        unique_scores_path,
        model_sha=model_sha,
        scoring_config_sha=scoring_config_sha,
    )
    registry_by_hash = {
        str(row["input_ids_sha256"]): row for row in registry
    }
    unknown_resume = set(existing) - set(registry_by_hash)
    if unknown_resume:
        raise QuickEvalError(
            f"Resume file contains {len(unknown_resume)} prompts outside registry"
        )

    pending = [
        row for row in registry if row["input_ids_sha256"] not in existing
    ]
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        label_scores = _score_method_batch(
            selected_method, llm, batch, label_token_ids
        )
        records = [
            _score_record(
                registry_row,
                scores,
                model_name=str(args.model_name),
                model_sha=model_sha,
                tokenizer_sha=tokenizer_sha,
                scoring_config_sha=scoring_config_sha,
                scoring_method=selected_method,
            )
            for registry_row, scores in zip(batch, label_scores)
        ]
        append_jsonl(unique_scores_path, records)
        for record in records:
            existing[str(record["input_ids_sha256"])] = record
        print(
            f"{args.model_name}: scored {min(start + len(batch), len(pending))}/"
            f"{len(pending)} pending unique prompts "
            f"({len(existing)}/{len(registry)} total)"
        )
    if set(existing) != set(registry_by_hash):
        raise QuickEvalError("Unique scoring did not complete the prompt registry")

    # Canonically rewrite the append-only resume file after completion.
    ordered_scores = [existing[row["input_ids_sha256"]] for row in registry]
    write_jsonl(unique_scores_path, ordered_scores)
    logical_results = expand_logical_results(logical_refs, existing)
    if len(logical_results) != EXPECTED_LOGICAL_RESULTS_PER_MODEL:
        raise QuickEvalError("Logical result expansion count mismatch")
    write_jsonl(logical_results_path, logical_results)

    files = {
        "prompt_registry": _prepared_file_metadata(
            prompt_registry_path, len(registry)
        ),
        "logical_refs": _prepared_file_metadata(
            logical_refs_path, len(logical_refs)
        ),
        "unique_scores": _prepared_file_metadata(
            unique_scores_path, len(ordered_scores)
        ),
        "logical_results": _prepared_file_metadata(
            logical_results_path, len(logical_results)
        ),
        "smoke_parity": {
            "path": str(smoke_report_path),
            "sha256": sha256_file(smoke_report_path),
            "bytes": smoke_report_path.stat().st_size,
        },
    }
    runtime_manifest = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_quick_v1",
        "created_at": utc_now(),
        "complete": True,
        "model_name": str(args.model_name),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": prepared_manifest_sha,
        "scoring_config": scoring_config,
        "scoring_config_sha256": scoring_config_sha,
        "selected_gpu_physical_index": selected_gpu,
        "smoke_parity": smoke_report,
        "prompt_audit": prompt_audit,
        "counts": {
            "logical_results": len(logical_results),
            "unique_prompts": len(registry),
            "new_scores_this_run": len(pending),
            "resumed_scores": len(registry) - len(pending),
        },
        "files": files,
        "model_file_manifest": model_files,
        "tokenizer_file_manifest": tokenizer_files,
    }
    write_json(runtime_manifest_path, runtime_manifest)
    print(
        f"Completed {args.model_name}: {len(registry)} unique prompts, "
        f"{len(logical_results)} logical results at {output_dir}"
    )
    return runtime_manifest


def exact_mcnemar_pvalue(wins: int, losses: int) -> float:
    wins = int(wins)
    losses = int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("McNemar discordant counts must be non-negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(min(wins, losses), discordant, 0.5).pvalue)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = [float(value) for value in pvalues]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Holm adjustment requires p-values in [0,1]: {values}")
    count = len(values)
    order = sorted(range(count), key=lambda idx: values[idx])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("gold/predicted length mismatch")
    scores: list[float] = []
    for label in LIAR6_LABELS:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(gold, predicted)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    return statistics.fmean(scores)


def _conditional_win_interval(wins: int, losses: int) -> tuple[float, float]:
    discordant = wins + losses
    if discordant == 0:
        return 0.0, 1.0
    from scipy.stats import binomtest

    interval = binomtest(wins, discordant, 0.5).proportion_ci(
        confidence_level=0.95,
        method="exact",
    )
    return float(interval.low), float(interval.high)


def _core_comparison_metrics(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not pair_rows:
        raise QuickEvalError("Cannot compute metrics on an empty paired sample")
    gold = [str(row["gold_label"]) for row in pair_rows]
    evi_pred = [str(row["evitrace_pred_label"]) for row in pair_rows]
    ctl_pred = [str(row["control_pred_label"]) for row in pair_rows]
    if any(label not in LABEL_TO_ID for label in gold + evi_pred + ctl_pred):
        raise QuickEvalError("Paired result contains an invalid LIAR6 label")
    evi_correct = [actual == guess for actual, guess in zip(gold, evi_pred)]
    ctl_correct = [actual == guess for actual, guess in zip(gold, ctl_pred)]
    wins = sum(evi and not ctl for evi, ctl in zip(evi_correct, ctl_correct))
    losses = sum(ctl and not evi for evi, ctl in zip(evi_correct, ctl_correct))
    both_correct = sum(evi and ctl for evi, ctl in zip(evi_correct, ctl_correct))
    both_wrong = sum(not evi and not ctl for evi, ctl in zip(evi_correct, ctl_correct))
    ties = both_correct + both_wrong
    discordant = wins + losses
    conditional_rate = wins / discordant if discordant else 0.5
    conditional_ci = _conditional_win_interval(wins, losses)
    logprob_deltas = [
        float(row["evitrace_gold_logprob"]) - float(row["control_gold_logprob"])
        for row in pair_rows
    ]
    if any(not math.isfinite(value) for value in logprob_deltas):
        raise QuickEvalError("Non-finite gold-label log-probability delta")
    tolerance = 1.0e-12
    return {
        "n": len(pair_rows),
        "evitrace": {
            "accuracy": statistics.fmean(evi_correct),
            "macro_f1": _macro_f1(gold, evi_pred),
        },
        "control": {
            "accuracy": statistics.fmean(ctl_correct),
            "macro_f1": _macro_f1(gold, ctl_pred),
        },
        "delta": {
            "accuracy": statistics.fmean(evi_correct)
            - statistics.fmean(ctl_correct),
            "macro_f1": _macro_f1(gold, evi_pred) - _macro_f1(gold, ctl_pred),
            "gold_logprob_mean": statistics.fmean(logprob_deltas),
        },
        "correctness_pairs": {
            "evitrace_only_correct_wins": wins,
            "control_only_correct_wins": losses,
            "ties": ties,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "conditional_evitrace_win_rate": conditional_rate,
            "conditional_win_rate_ci95": list(conditional_ci),
            "exact_mcnemar_pvalue": exact_mcnemar_pvalue(wins, losses),
        },
        "gold_logprob_delta": {
            "mean": statistics.fmean(logprob_deltas),
            "median": statistics.median(logprob_deltas),
            "positive": sum(value > tolerance for value in logprob_deltas),
            "negative": sum(value < -tolerance for value in logprob_deltas),
            "tie": sum(abs(value) <= tolerance for value in logprob_deltas),
        },
    }


def compute_comparison_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute paired metrics for one model/comparison population."""

    metrics = _core_comparison_metrics(pair_rows)
    by_label: dict[str, Any] = {}
    for label in LIAR6_LABELS:
        subset = [row for row in pair_rows if row["gold_label"] == label]
        if subset:
            by_label[label] = _core_comparison_metrics(subset)
    by_complexity: dict[str, Any] = {}
    for complexity in ("single", "multi"):
        subset = [row for row in pair_rows if row["complexity"] == complexity]
        if subset:
            by_complexity[complexity] = _core_comparison_metrics(subset)
    metrics["by_label"] = by_label
    metrics["by_complexity"] = by_complexity
    return metrics


def _panel_point(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_model[str(row["model_name"])].append(row)
    if not by_model:
        raise QuickEvalError("Panel has no models")
    per_model = {
        model: _core_comparison_metrics(rows)
        for model, rows in by_model.items()
    }
    return {
        "accuracy_delta": statistics.fmean(
            value["delta"]["accuracy"] for value in per_model.values()
        ),
        "macro_f1_delta": statistics.fmean(
            value["delta"]["macro_f1"] for value in per_model.values()
        ),
        "gold_logprob_delta": statistics.fmean(
            value["delta"]["gold_logprob_mean"] for value in per_model.values()
        ),
    }


def _percentile_ci(values: Sequence[float]) -> list[float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return [
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    ]


def stratified_cluster_bootstrap(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    claims: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    labels_by_claim: dict[str, str] = {}
    for row in pair_rows:
        event_id = str(row["event_id"])
        label = str(row["gold_label"])
        previous = labels_by_claim.setdefault(event_id, label)
        if previous != label:
            raise QuickEvalError(f"{event_id}: inconsistent gold label in panel")
        claims[event_id].append(row)
    model_counts = {len(rows) for rows in claims.values()}
    if len(model_counts) != 1:
        raise QuickEvalError("Each bootstrap claim must carry the same model count")
    by_label: dict[str, list[str]] = {
        label: sorted(
            event_id
            for event_id, gold_label in labels_by_claim.items()
            if gold_label == label
        )
        for label in LIAR6_LABELS
    }
    if any(not event_ids for event_ids in by_label.values()):
        raise QuickEvalError("Every LIAR6 label must occur in bootstrap data")

    rng = random.Random(int(seed))
    distributions = {
        "accuracy_delta": [],
        "macro_f1_delta": [],
        "gold_logprob_delta": [],
    }
    for _ in range(iterations):
        sampled_rows: list[Mapping[str, Any]] = []
        for label in LIAR6_LABELS:
            pool = by_label[label]
            for _sample in range(len(pool)):
                sampled_rows.extend(claims[rng.choice(pool)])
        point = _panel_point(sampled_rows)
        for name in distributions:
            distributions[name].append(point[name])
    point = _panel_point(pair_rows)
    return {
        "iterations": iterations,
        "seed": int(seed),
        "claim_count": len(claims),
        "model_count_per_claim": next(iter(model_counts)),
        "point": point,
        "ci95": {
            name: _percentile_ci(values)
            for name, values in distributions.items()
        },
    }


def claim_swap_randomization(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = DEFAULT_RANDOMIZATION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("randomization iterations must be positive")
    deltas_by_claim: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        gold = str(row["gold_label"])
        delta = float(str(row["evitrace_pred_label"]) == gold) - float(
            str(row["control_pred_label"]) == gold
        )
        deltas_by_claim[str(row["event_id"])].append(delta)
    model_counts = {len(values) for values in deltas_by_claim.values()}
    if len(model_counts) != 1:
        raise QuickEvalError("Each randomized claim must carry the same model count")
    import numpy as np

    claim_deltas = np.asarray(
        [
            statistics.fmean(deltas_by_claim[event_id])
            for event_id in sorted(deltas_by_claim)
        ],
        dtype=np.float64,
    )
    observed = float(claim_deltas.mean())
    rng = np.random.default_rng(int(seed))
    exceedances = 0
    completed = 0
    chunk_size = 2_000
    while completed < iterations:
        current = min(chunk_size, iterations - completed)
        signs = rng.integers(
            0,
            2,
            size=(current, len(claim_deltas)),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        permuted = signs @ claim_deltas / len(claim_deltas)
        exceedances += int(
            np.count_nonzero(np.abs(permuted) >= abs(observed) - 1.0e-15)
        )
        completed += current
    pvalue = (exceedances + 1) / (iterations + 1)
    return {
        "iterations": iterations,
        "seed": int(seed),
        "claim_count": len(claim_deltas),
        "model_count_per_claim": next(iter(model_counts)),
        "observed_accuracy_delta": observed,
        "two_sided_pvalue": pvalue,
        "exceedances": exceedances,
        "same_swap_bit_across_models": True,
    }


def _validate_result_file(
    result_path: Path,
    *,
    prepared_manifest_sha: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    runtime_path = result_path.with_name("runtime_manifest.json")
    if not runtime_path.exists():
        raise QuickEvalError(f"Missing runtime manifest beside {result_path}")
    runtime = load_json(runtime_path)
    if not runtime.get("complete"):
        raise QuickEvalError(f"Inference runtime is incomplete: {runtime_path}")
    if runtime.get("prepared_manifest_sha256") != prepared_manifest_sha:
        raise QuickEvalError(f"{result_path}: prepared manifest SHA mismatch")
    counts = runtime.get("counts") or {}
    if (
        int(counts.get("logical_results", -1))
        != EXPECTED_LOGICAL_RESULTS_PER_MODEL
        or int(counts.get("unique_prompts", -1))
        != EXPECTED_UNIQUE_PROMPTS_PER_MODEL
    ):
        raise QuickEvalError(f"{runtime_path}: inference count contract mismatch")
    smoke = runtime.get("smoke_parity") or {}
    selected_method = runtime.get("scoring_config", {}).get("selected_method")
    if selected_method not in {
        "allowed_token",
        "canonical_six_continuation",
    } or smoke.get("selected_method") != selected_method:
        raise QuickEvalError(f"{runtime_path}: invalid smoke/scoring method contract")
    prompt_audit = runtime.get("prompt_audit") or {}
    if prompt_audit.get("order_nonzero_token_differences") != [
        {"event_id": "1290.json", "difference_evi_minus_control": -1}
    ]:
        raise QuickEvalError(f"{runtime_path}: order token audit mismatch")
    expected_file_rows = {
        "prompt_registry": EXPECTED_UNIQUE_PROMPTS_PER_MODEL,
        "logical_refs": EXPECTED_LOGICAL_RESULTS_PER_MODEL,
        "unique_scores": EXPECTED_UNIQUE_PROMPTS_PER_MODEL,
        "logical_results": EXPECTED_LOGICAL_RESULTS_PER_MODEL,
    }
    for file_name, expected_rows in expected_file_rows.items():
        file_metadata = runtime.get("files", {}).get(file_name)
        if not isinstance(file_metadata, Mapping):
            raise QuickEvalError(f"{runtime_path}: missing {file_name} metadata")
        file_path = Path(str(file_metadata.get("path") or ""))
        if (
            not file_path.exists()
            or sha256_file(file_path) != file_metadata.get("sha256")
            or int(file_metadata.get("rows", -1)) != expected_rows
        ):
            raise QuickEvalError(
                f"{runtime_path}: {file_name} hash/count contract mismatch"
            )
    metadata = runtime.get("files", {}).get("logical_results")
    if not isinstance(metadata, Mapping):
        raise QuickEvalError(f"{runtime_path}: missing logical_results metadata")
    if Path(str(metadata["path"])).resolve() != result_path.resolve():
        raise QuickEvalError(f"{runtime_path}: logical result path mismatch")
    if sha256_file(result_path) != metadata.get("sha256"):
        raise QuickEvalError(f"{result_path}: logical result SHA mismatch")
    rows = load_jsonl(result_path)
    if len(rows) != EXPECTED_LOGICAL_RESULTS_PER_MODEL:
        raise QuickEvalError(
            f"{result_path}: expected {EXPECTED_LOGICAL_RESULTS_PER_MODEL} rows, "
            f"found {len(rows)}"
        )
    model_names = {str(row.get("model_name") or "") for row in rows}
    if len(model_names) != 1 or "" in model_names:
        raise QuickEvalError(f"{result_path}: inconsistent model names")
    model_name = next(iter(model_names))
    if model_name != runtime.get("model_name"):
        raise QuickEvalError(f"{result_path}: model name differs from runtime")
    model_shas = {str(row.get("model_sha256") or "") for row in rows}
    if model_shas != {str(runtime.get("model_sha256") or "")}:
        raise QuickEvalError(f"{result_path}: model SHA differs from runtime")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("comparison_id") or ""), str(row.get("arm") or ""))
        if not key[0] or key[1] not in {"evitrace", "control"} or key in seen:
            raise QuickEvalError(f"{result_path}: duplicate/invalid logical key {key}")
        seen.add(key)
        normalized = normalize_label_logprobs(row.get("label_logprobs") or {})
        if (
            max(
                abs(float(row["label_logprobs"][letter]) - normalized[letter])
                for letter in LETTERS
            )
            > 1.0e-6
        ):
            raise QuickEvalError(f"{result_path}: unnormalized label scores")
        probabilities = row.get("label_probabilities") or {}
        if set(probabilities) != set(LETTERS):
            raise QuickEvalError(f"{result_path}: invalid label probabilities")
        if abs(sum(float(probabilities[letter]) for letter in LETTERS) - 1.0) > 1e-6:
            raise QuickEvalError(f"{result_path}: label probabilities do not sum to one")
    return model_name, rows, runtime


def _build_pair_rows(
    *,
    model_name: str,
    result_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    gold_by_event: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result_by_key = {
        (str(row["comparison_id"]), str(row["arm"])): row for row in result_rows
    }
    expected_keys = {
        (str(comparison["comparison_id"]), arm)
        for comparison in comparisons
        for arm in ("evitrace", "control")
    }
    if set(result_by_key) != expected_keys:
        missing = expected_keys - set(result_by_key)
        extra = set(result_by_key) - expected_keys
        raise QuickEvalError(
            f"{model_name}: logical result contract mismatch "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    pair_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        comparison_id = str(comparison["comparison_id"])
        event_id = str(comparison["event_id"])
        gold_row = gold_by_event.get(event_id)
        if gold_row is None:
            raise QuickEvalError(f"{event_id}: no frozen gold row")
        gold_label = str(gold_row["gold_label"])
        gold_letter = LABEL_TO_LETTER[gold_label]
        evi = result_by_key[(comparison_id, "evitrace")]
        control = result_by_key[(comparison_id, "control")]
        pair_rows.append(
            {
                "model_name": model_name,
                "comparison_id": comparison_id,
                "event_id": event_id,
                "comparison_type": comparison["comparison_type"],
                "gold_label": gold_label,
                "complexity": comparison["complexity"],
                "evitrace_pred_label": evi["pred_label"],
                "control_pred_label": control["pred_label"],
                "evitrace_gold_logprob": float(
                    evi["label_logprobs"][gold_letter]
                ),
                "control_gold_logprob": float(
                    control["label_logprobs"][gold_letter]
                ),
                "evitrace_prompt_token_count": int(evi["prompt_token_count"]),
                "control_prompt_token_count": int(control["prompt_token_count"]),
                "token_difference_evi_minus_control": int(
                    evi["prompt_token_count"]
                )
                - int(control["prompt_token_count"]),
                "evitrace_input_ids_sha256": evi["input_ids_sha256"],
                "control_input_ids_sha256": control["input_ids_sha256"],
            }
        )
    return pair_rows


def _model_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for comparison_type in ("main", "order_only"):
        subset = [
            row for row in pair_rows if row["comparison_type"] == comparison_type
        ]
        expected = (
            EXPECTED_MAIN_COMPARISONS
            if comparison_type == "main"
            else EXPECTED_ORDER_COMPARISONS
        )
        if len(subset) != expected:
            raise QuickEvalError(
                f"Expected {expected} {comparison_type} pairs, found {len(subset)}"
            )
        output[comparison_type] = compute_comparison_metrics(subset)
    main_within64 = [
        row
        for row in pair_rows
        if row["comparison_type"] == "main"
        and abs(int(row["token_difference_evi_minus_control"])) <= 64
    ]
    output["main_token_sensitivity_abs_le_64"] = compute_comparison_metrics(
        main_within64
    )
    order_rows = [
        row for row in pair_rows if row["comparison_type"] == "order_only"
    ]
    nonzero = [
        {
            "event_id": row["event_id"],
            "difference_evi_minus_control": row[
                "token_difference_evi_minus_control"
            ],
        }
        for row in order_rows
        if row["token_difference_evi_minus_control"] != 0
    ]
    if nonzero != [
        {"event_id": "1290.json", "difference_evi_minus_control": -1}
    ]:
        raise QuickEvalError(
            f"Unexpected analyzed order-only token differences: {nonzero}"
        )
    output["order_token_audit"] = {
        "equal_token_count": len(order_rows) - len(nonzero),
        "nonzero_token_count": len(nonzero),
        "nonzero": nonzero,
        "no_posthoc_exclusion": True,
    }
    return output


def _interpret_results(
    per_model: Mapping[str, Mapping[str, Any]],
    panel: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    def decision_positive(comparison_type: str) -> bool:
        panel_result = panel[comparison_type]
        ci_low = panel_result["bootstrap"]["ci95"]["accuracy_delta"][0]
        adjusted_p = panel_result["accuracy_randomization"]["holm_pvalue"]
        model_direction = all(
            metrics[comparison_type]["correctness_pairs"][
                "evitrace_only_correct_wins"
            ]
            >= metrics[comparison_type]["correctness_pairs"][
                "control_only_correct_wins"
            ]
            for metrics in per_model.values()
        )
        return (
            panel_result["point"]["accuracy_delta"] > 0.0
            and ci_low > 0.0
            and adjusted_p < 0.05
            and model_direction
        )

    main_positive = decision_positive("main")
    if main_positive:
        intersection_delta = panel["main_token_sensitivity_intersection"][
            "point"
        ]["accuracy_delta"]
        main_positive = intersection_delta >= 0.0
    order_positive = decision_positive("order_only")
    order_score_positive = (
        not order_positive
        and panel["order_only"]["bootstrap"]["ci95"]["gold_logprob_delta"][0]
        > 0.0
        and all(
            metrics["order_only"]["delta"]["gold_logprob_mean"] > 0.0
            for metrics in per_model.values()
        )
    )

    if main_positive and order_positive:
        main_text = (
            "The evaluated external-verifier panel supports improved evidence "
            "selection/organization and a decision-level ordering benefit."
        )
        order_text = (
            "improves decision-relevant evidence ordering across external verifiers"
        )
        category = "main_and_order_decision_level_positive"
    elif main_positive:
        main_text = (
            "improves evidence selection and overall organization for external verifiers"
        )
        if order_score_positive:
            order_text = "shows a score-level preference for EviTrace ordering"
            category = "main_positive_order_score_level_only"
        else:
            order_text = "order-only evidence is mixed or uncertain"
            category = "main_positive_order_inconclusive"
    else:
        main_text = "main-comparison evidence is mixed or uncertain"
        if order_positive:
            order_text = (
                "order-only results are decision-level positive, but the overall "
                "main comparison is not robustly positive"
            )
            category = "order_only_positive_main_inconclusive"
        elif order_score_positive:
            order_text = "shows a score-level preference for EviTrace ordering"
            category = "score_level_order_only"
        else:
            order_text = "order-only evidence is mixed or uncertain"
            category = "mixed_or_uncertain"
    return {
        "category": category,
        "main_wording": main_text,
        "order_wording": order_text,
        "prohibited_claims": (
            "Do not claim human alignment, human fact-checking accuracy, causal "
            "explanation, or access to latent reasoning."
        ),
    }


def _markdown_report(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# EviTrace Cross-Verifier Quick Evaluation",
        "",
        "This report evaluates gold-label utility for two frozen external base models. "
        "It is not a human-alignment evaluation.",
        "",
        "## Results",
        "",
        "| Model | Comparison | Evi Acc. | Control Acc. | ΔAcc. | ΔMacro-F1 | W/L/T | McNemar p | ΔGold log p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model_metrics in metrics["per_model"].items():
        for comparison_type in ("main", "order_only"):
            item = model_metrics[comparison_type]
            pairs = item["correctness_pairs"]
            lines.append(
                "| {model} | {comparison} | {evi:.4f} | {ctl:.4f} | {delta:.4f} "
                "| {f1:.4f} | {w}/{l}/{t} | {p:.4g} | {logp:.4f} |".format(
                    model=model_name,
                    comparison=comparison_type,
                    evi=item["evitrace"]["accuracy"],
                    ctl=item["control"]["accuracy"],
                    delta=item["delta"]["accuracy"],
                    f1=item["delta"]["macro_f1"],
                    w=pairs["evitrace_only_correct_wins"],
                    l=pairs["control_only_correct_wins"],
                    t=pairs["ties"],
                    p=pairs["exact_mcnemar_pvalue"],
                    logp=item["delta"]["gold_logprob_mean"],
                )
            )
    lines.extend(["", "## Equal-weight model panel", ""])
    for comparison_type in ("main", "order_only"):
        item = metrics["panel"][comparison_type]
        lines.extend(
            [
                f"### {comparison_type}",
                "",
                f"- Accuracy delta: {item['point']['accuracy_delta']:.6f}",
                f"- Accuracy bootstrap 95% CI: "
                f"[{item['bootstrap']['ci95']['accuracy_delta'][0]:.6f}, "
                f"{item['bootstrap']['ci95']['accuracy_delta'][1]:.6f}]",
                f"- Macro-F1 delta: {item['point']['macro_f1_delta']:.6f}",
                f"- Gold-label log-probability delta: "
                f"{item['point']['gold_logprob_delta']:.6f}",
                f"- Claim-swap p: "
                f"{item['accuracy_randomization']['two_sided_pvalue']:.6g}",
                f"- Holm-adjusted p: "
                f"{item['accuracy_randomization']['holm_pvalue']:.6g}",
                "",
            ]
        )
    lines.extend(
        [
            "## Token robustness",
            "",
            f"- Main |Δtokens|≤64 intersection: "
            f"{metrics['panel']['main_token_sensitivity_intersection']['claim_count']} claims.",
            "- Order-only uses identical evidence-text multisets and character counts. "
            "All but `1290.json` have equal tokenizer length; that event is retained "
            "with a one-token difference.",
            "",
            "## Locked interpretation",
            "",
            f"- Category: `{metrics['interpretation']['category']}`",
            f"- Main: {metrics['interpretation']['main_wording']}.",
            f"- Order: {metrics['interpretation']['order_wording']}.",
            f"- {metrics['interpretation']['prohibited_claims']}",
            "",
        ]
    )
    return "\n".join(lines)


def _latex_table(metrics: Mapping[str, Any]) -> str:
    lines = [
        "% Auto-generated by cross_verifier_quick.py; do not hand edit.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Verifier & Comparison & Evi Acc. & Ctrl. Acc. & $\\Delta$Acc. & W/L \\\\",
        "\\midrule",
    ]
    for model_name, model_metrics in metrics["per_model"].items():
        safe_model = model_name.replace("_", "\\_")
        for comparison_type in ("main", "order_only"):
            item = model_metrics[comparison_type]
            pairs = item["correctness_pairs"]
            safe_comparison = comparison_type.replace("_", "\\_")
            lines.append(
                f"{safe_model} & {safe_comparison} & "
                f"{item['evitrace']['accuracy']:.3f} & "
                f"{item['control']['accuracy']:.3f} & "
                f"{item['delta']['accuracy']:+.3f} & "
                f"{pairs['evitrace_only_correct_wins']}/"
                f"{pairs['control_only_correct_wins']} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def analyze_results(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.prepared_manifest).resolve()
    prepared = _verify_prepared_manifest(manifest_path)
    prepared_manifest_sha = sha256_file(manifest_path)
    comparisons = load_jsonl(
        Path(prepared["prepared_files"]["comparisons_test"]["path"])
    )
    gold_rows = load_jsonl(Path(prepared["prepared_files"]["gold_test"]["path"]))
    gold_by_event = {str(row["event_id"]): row for row in gold_rows}
    if len(gold_by_event) != EXPECTED_MAIN_COMPARISONS:
        raise QuickEvalError("Frozen gold file has duplicate/missing events")

    result_paths = [Path(path).resolve() for path in args.result]
    if len(result_paths) != 2 or len(set(result_paths)) != 2:
        raise QuickEvalError("Analyze requires exactly two distinct --result files")
    runtimes: dict[str, dict[str, Any]] = {}
    all_pairs: list[dict[str, Any]] = []
    per_model: dict[str, dict[str, Any]] = {}
    result_metadata: dict[str, Any] = {}
    for result_path in result_paths:
        model_name, rows, runtime = _validate_result_file(
            result_path,
            prepared_manifest_sha=prepared_manifest_sha,
        )
        if model_name in runtimes:
            raise QuickEvalError(f"Duplicate model in panel: {model_name}")
        runtimes[model_name] = runtime
        pair_rows = _build_pair_rows(
            model_name=model_name,
            result_rows=rows,
            comparisons=comparisons,
            gold_by_event=gold_by_event,
        )
        all_pairs.extend(pair_rows)
        per_model[model_name] = _model_metrics(pair_rows)
        result_metadata[model_name] = {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
            "rows": len(rows),
            "runtime_manifest_path": str(
                result_path.with_name("runtime_manifest.json")
            ),
            "runtime_manifest_sha256": sha256_file(
                result_path.with_name("runtime_manifest.json")
            ),
            "model_sha256": runtime["model_sha256"],
            "tokenizer_sha256": runtime["tokenizer_sha256"],
            "scoring_method": runtime["scoring_config"]["selected_method"],
        }
    if len(runtimes) != 2 or len(all_pairs) != 2 * EXPECTED_COMPARISONS:
        raise QuickEvalError("Panel result count mismatch")

    panel: dict[str, Any] = {}
    randomization_pvalues: list[float] = []
    for offset, comparison_type in enumerate(("main", "order_only")):
        subset = [
            row for row in all_pairs if row["comparison_type"] == comparison_type
        ]
        point = _panel_point(subset)
        bootstrap = stratified_cluster_bootstrap(
            subset,
            iterations=int(args.bootstrap),
            seed=int(args.seed) + 101 * offset,
        )
        randomization = claim_swap_randomization(
            subset,
            iterations=int(args.randomization),
            seed=int(args.seed) + 211 * offset,
        )
        randomization_pvalues.append(randomization["two_sided_pvalue"])
        panel[comparison_type] = {
            "point": point,
            "bootstrap": bootstrap,
            "accuracy_randomization": randomization,
        }
    adjusted = holm_adjust(randomization_pvalues)
    for comparison_type, adjusted_p in zip(("main", "order_only"), adjusted):
        panel[comparison_type]["accuracy_randomization"][
            "holm_pvalue"
        ] = adjusted_p

    main_rows = [row for row in all_pairs if row["comparison_type"] == "main"]
    eligible_by_model: dict[str, set[str]] = defaultdict(set)
    for row in main_rows:
        if abs(int(row["token_difference_evi_minus_control"])) <= 64:
            eligible_by_model[str(row["model_name"])].add(str(row["event_id"]))
    if len(eligible_by_model) != 2:
        raise QuickEvalError("Token sensitivity did not cover both models")
    intersection = set.intersection(*eligible_by_model.values())
    intersection_rows = [
        row for row in main_rows if str(row["event_id"]) in intersection
    ]
    expected_intersection_rows = len(intersection) * 2
    if len(intersection_rows) != expected_intersection_rows:
        raise QuickEvalError("Token intersection did not retain both model rows")
    panel["main_token_sensitivity_intersection"] = {
        "claim_count": len(intersection),
        "model_specific_claim_counts": {
            model: len(event_ids)
            for model, event_ids in sorted(eligible_by_model.items())
        },
        "point": _panel_point(intersection_rows),
        "definition": "|T_Evi-T_S4|<=64 for both tokenizers",
    }

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_quick_v1",
        "created_at": utc_now(),
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": prepared_manifest_sha,
        "seed": int(args.seed),
        "bootstrap_iterations": int(args.bootstrap),
        "randomization_iterations": int(args.randomization),
        "result_files": result_metadata,
        "per_model": per_model,
        "panel": panel,
    }
    metrics["interpretation"] = _interpret_results(per_model, panel)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    latex_path = output_dir / "paper_table.tex"
    write_json(metrics_path, metrics)
    report_path.write_text(_markdown_report(metrics), encoding="utf-8")
    latex_path.write_text(_latex_table(metrics), encoding="utf-8")

    complete_manifest = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_quick_v1",
        "created_at": utc_now(),
        "complete": True,
        "completion_is_effect_independent": True,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": prepared_manifest_sha,
        "models": sorted(runtimes),
        "counts": {
            "models": len(runtimes),
            "comparisons": len(comparisons),
            "logical_results_per_model": EXPECTED_LOGICAL_RESULTS_PER_MODEL,
            "logical_results_total": 2 * EXPECTED_LOGICAL_RESULTS_PER_MODEL,
            "unique_prompts_per_model": EXPECTED_UNIQUE_PROMPTS_PER_MODEL,
            "unique_prompts_total": 2 * EXPECTED_UNIQUE_PROMPTS_PER_MODEL,
            "main_claims": EXPECTED_MAIN_COMPARISONS,
            "order_only_claims": EXPECTED_ORDER_COMPARISONS,
        },
        "invariants": {
            "no_missing_or_duplicate_logical_results": True,
            "all_label_scores_finite_and_normalized": True,
            "same_prepared_manifest_for_both_models": True,
            "claim_clustered_panel_analysis": True,
            "order_same_set_contract_prepared": True,
            "order_token_boundary_event_retained": "1290.json",
        },
        "files": {
            "metrics": {
                "path": str(metrics_path),
                "sha256": sha256_file(metrics_path),
                "bytes": metrics_path.stat().st_size,
            },
            "report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "bytes": report_path.stat().st_size,
            },
            "paper_table": {
                "path": str(latex_path),
                "sha256": sha256_file(latex_path),
                "bytes": latex_path.stat().st_size,
            },
        },
        "interpretation_category": metrics["interpretation"]["category"],
    }
    complete_path = output_dir / "complete_manifest.json"
    write_json(complete_path, complete_manifest)
    print(f"Analysis complete: {report_path}")
    return complete_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen EviTrace external-verifier quick evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Audit artifacts and export gold-free comparisons"
    )
    prepare.add_argument("--build", default=str(DEFAULT_BUILD_TEST))
    prepare.add_argument("--evitrace", default=str(DEFAULT_EVITRACE_TEST))
    prepare.add_argument("--s4", default=str(DEFAULT_S4_TEST))
    prepare.add_argument("--build-val", default=str(DEFAULT_BUILD_VAL))
    prepare.add_argument("--evitrace-val", default=str(DEFAULT_EVITRACE_VAL))
    prepare.add_argument("--s4-val", default=str(DEFAULT_S4_VAL))
    prepare.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "prepared")
    )
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.set_defaults(handler=prepare_experiment)

    infer = subparsers.add_parser(
        "infer", help="Render one model's prompts and score A-F labels"
    )
    infer.add_argument("--prepared-manifest", required=True)
    infer.add_argument("--model-name", required=True)
    infer.add_argument("--model-path", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--gpu-id", default="auto")
    infer.add_argument("--seed", type=int, default=DEFAULT_SEED)
    infer.add_argument("--batch-size", type=int, default=128)
    infer.set_defaults(handler=infer_model)

    analyze = subparsers.add_parser(
        "analyze", help="Join gold and run paired model/panel analyses"
    )
    analyze.add_argument("--prepared-manifest", required=True)
    analyze.add_argument("--result", action="append", required=True)
    analyze.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "analysis")
    )
    analyze.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    analyze.add_argument(
        "--randomization", type=int, default=DEFAULT_RANDOMIZATION
    )
    analyze.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze.set_defaults(handler=analyze_results)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "seed", DEFAULT_SEED) != DEFAULT_SEED:
        raise QuickEvalError(
            f"This frozen experiment requires seed {DEFAULT_SEED}; "
            f"received {args.seed}"
        )
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise QuickEvalError("--batch-size must be positive")
    if hasattr(args, "bootstrap") and args.bootstrap <= 0:
        raise QuickEvalError("--bootstrap must be positive")
    if hasattr(args, "randomization") and args.randomization <= 0:
        raise QuickEvalError("--randomization must be positive")
    args.handler(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QuickEvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
