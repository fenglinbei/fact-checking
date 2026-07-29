#!/usr/bin/env python3
"""Prepare leakage-controlled cross-verifier fine-tuning/evaluation datasets.

This module owns data preparation only.  It deliberately reuses the canonical
Exp3 alignment exporter for event alignment, candidate identity mapping, and
S4 order verification.  Training and inference launchers may import
``prepare_dataset`` without depending on a particular command-line wrapper.

The prepared evaluation registry is gold-free and stores chat-only
``prompt_input_ids``.  Runtime inference must append the tokenizer-specific
``Label:`` prefix before scoring the six one-token choices ``" A"``--``" F"``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
EXPORTER_PATH = (
    PROJECT_ROOT
    / "docs/paper/aaai/annotation_project/scripts/export_exp3_trace_alignment.py"
)

DEFAULT_BUILD_ROOT = (
    PROJECT_ROOT
    / "outputs/sentence_trace_method/"
    "liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10/"
    "build"
)
DEFAULT_EVITRACE_ROOT = (
    PROJECT_ROOT
    / "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
    "05_mrec_v0_2_learned_marginal_proxy_fullpool"
)
DEFAULT_S4_ROOT = (
    PROJECT_ROOT
    / "outputs/selectors/selector_mechanism_ablation_chunking"
)
DEFAULT_ARTIFACT_PATHS = {
    "build_train": DEFAULT_BUILD_ROOT / "build_train.jsonl",
    "build_val": DEFAULT_BUILD_ROOT / "build_val.jsonl",
    "build_test": DEFAULT_BUILD_ROOT / "build_test.jsonl",
    "evitrace_train": DEFAULT_EVITRACE_ROOT / "selection_trace_train.jsonl",
    "evitrace_val": DEFAULT_EVITRACE_ROOT / "selection_trace_val.jsonl",
    "evitrace_test": DEFAULT_EVITRACE_ROOT / "selection_trace_test.jsonl",
    "s4_train": (
        DEFAULT_S4_ROOT
        / "liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_train/"
        "selection_trace_train.jsonl"
    ),
    "s4_val": (
        DEFAULT_S4_ROOT
        / "liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_val/"
        "selection_trace_val.jsonl"
    ),
    "s4_test": (
        DEFAULT_S4_ROOT
        / "liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_test/"
        "selection_trace_test.jsonl"
    ),
}
DEFAULT_MODEL_PATHS = {
    "qwen3": Path("/data/models/Qwen3-4B-Instruct-2507"),
    "llama31": Path("/data/models/Meta-Llama-3.1-8B-Instruct"),
}
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/analysis/evitrace_cross_verifier_finetune_v1/prepared"
)

DEFAULT_SEED = 20260724
MAX_MODEL_LEN = 2048
LABEL_PREFIX = "Label:"
LABELS = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
LETTERS = ("A", "B", "C", "D", "E", "F")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
LABEL_TO_LETTER = dict(zip(LABELS, LETTERS))

EXPECTED_RAW_COUNTS = {"train": 10_065, "val": 1_274, "test": 1_251}
EXPECTED_COUNTS = {"train": 10_050, "val": 1_274, "test": 1_250}
EXPECTED_MAIN = 1_250
EXPECTED_ORDER = 1_152
EXPECTED_ORDER_IDENTICAL = 98
EXPECTED_COMPARISONS = EXPECTED_MAIN + EXPECTED_ORDER
EXPECTED_PREFIX = 6_996
EXPECTED_PREFIX_RELATIONS = {
    "same_order": 1_138,
    "different_set": 3_838,
    "same_set_different_order": 2_020,
}
PREAUDIT_PLANNED_PREFIX = 7_448
IDENTICAL_ORDER_EXCLUDED_PREFIX_POSITIONS = 452
EXPECTED_TEST_LOGICAL_ROWS = 2 * (
    EXPECTED_MAIN + EXPECTED_ORDER + EXPECTED_PREFIX
)
EXPECTED_VAL_LOGICAL_ROWS = 2 * EXPECTED_COUNTS["val"] + 2 * EXPECTED_COUNTS["val"]
EXPECTED_EVAL_LOGICAL_ROWS = EXPECTED_TEST_LOGICAL_ROWS + EXPECTED_VAL_LOGICAL_ROWS

TRAIN_ATOM_PARSE_EXCLUSIONS = frozenset(
    {
        "1606.json",
        "242.json",
        "8753.json",
        "7196.json",
        "9505.json",
        "2987.json",
    }
)
TRAIN_CROSS_SPLIT_CLAIM_EXCLUSIONS = frozenset(
    {
        "12055.json",
        "2519.json",
        "1211.json",
        "2517.json",
        "3263.json",
        "601.json",
        "10987.json",
        "6622.json",
        "6918.json",
    }
)
TEST_ATOM_PARSE_EXCLUSIONS = frozenset({"7845.json"})
SPLIT_EXCLUSIONS = {
    "train": TRAIN_ATOM_PARSE_EXCLUSIONS | TRAIN_CROSS_SPLIT_CLAIM_EXCLUSIONS,
    "val": frozenset(),
    "test": TEST_ATOM_PARSE_EXCLUSIONS,
}

FROZEN_ARTIFACT_SHA256 = {
    "build_train": "ed0544ab175e72dd929ce64d4f02842397584d2e4972bb8a7101c72d75d8e3df",
    "build_val": "060055f5087b1de8da20711e2416789bde5477a8c6a4b18e42331a74b435c174",
    "build_test": "7499006f339a54b84220174a8fd392f6c5e7afa1aa35b94d648762b7f6860a7f",
    "evitrace_train": "b6130b3909419ee43e2b7c34e2a561bc6c22ac5aa57c670c8ab415efeabb9b32",
    "evitrace_val": "01d66e246ba5fa79fb6fd3dad8ea7a03f39320f9a8a6e95086ba9a735f3deddf",
    "evitrace_test": "aaa8574d915026100c435b7adfb99cb52e890ab98e21783aff69abb3627f79fa",
    "s4_train": "1fcf191f08ec1123418e2bdf3f7bad85c4bd90abd11246f6147be83610662926",
    "s4_val": "6869f41bd7fc1ded26fde4581a97be08343fe42e0560990c8d3d6d18cb4734a1",
    "s4_test": "67d740002d0543724707281a34cc1adce6ad2628efda25ac83b84d9ab58a6705",
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

CHECK_CUE = re.compile(r"(?im)^\s*Check:\s*")
_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".pt",
    ".pth",
    ".safetensors",
}
_GOLD_KEYS = {"gold_label", "gold_id", "gold_explain", "target"}


class FinetunePrepareError(RuntimeError):
    """Raised when a frozen preparation contract is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for raw in rows:
            handle.write(canonical_json(dict(raw)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def _file_metadata(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(rows),
    }


def _load_exporter() -> Any:
    if not EXPORTER_PATH.exists():
        raise FinetunePrepareError(f"Missing canonical exporter: {EXPORTER_PATH}")
    module_name = "_evitrace_exp3_exporter_for_finetune_prepare"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise FinetunePrepareError(f"Cannot import canonical exporter: {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _argument_path(args: argparse.Namespace, name: str, default: Path) -> Path:
    raw = getattr(args, name, None)
    return Path(raw if raw not in (None, "") else default).resolve()


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        name: _argument_path(args, name, default)
        for name, default in DEFAULT_ARTIFACT_PATHS.items()
    }


def _assert_frozen_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if set(paths) != set(FROZEN_ARTIFACT_SHA256):
        raise FinetunePrepareError(
            f"Artifact keys differ from frozen contract: {sorted(paths)}"
        )
    for name, path in paths.items():
        if not path.is_file():
            raise FinetunePrepareError(f"Missing source artifact: {path}")
        digest = sha256_file(path)
        expected = FROZEN_ARTIFACT_SHA256[name]
        if digest != expected:
            raise FinetunePrepareError(
                f"Frozen SHA mismatch for {name}: expected {expected}, found {digest}"
            )
        metadata[name] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    return metadata


def _normalize_claim(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _filter_events(
    events: Sequence[Any],
    *,
    split: str,
) -> list[Any]:
    exclusions = SPLIT_EXCLUSIONS[split]
    present = {event.event_id for event in events if event.event_id in exclusions}
    if present != set(exclusions):
        raise FinetunePrepareError(
            f"{split}: expected exclusions {sorted(exclusions)}, found {sorted(present)}"
        )
    eligible = [event for event in events if event.event_id not in exclusions]
    if len(eligible) != EXPECTED_COUNTS[split]:
        raise FinetunePrepareError(
            f"{split}: expected {EXPECTED_COUNTS[split]} eligible events, "
            f"found {len(eligible)}"
        )
    return eligible


def _assert_no_cross_split_claim_overlap(
    train_events: Sequence[Any],
    val_events: Sequence[Any],
    test_events: Sequence[Any],
) -> None:
    train_claims = {_normalize_claim(event.claim) for event in train_events}
    val_claims = {_normalize_claim(event.claim) for event in val_events}
    test_claims = {_normalize_claim(event.claim) for event in test_events}
    overlaps = {
        "train_val": sorted(train_claims & val_claims),
        "train_test": sorted(train_claims & test_claims),
    }
    if any(overlaps.values()):
        summary = {key: len(value) for key, value in overlaps.items()}
        raise FinetunePrepareError(
            f"Cross-split normalized claim leakage remains after purge: {summary}"
        )


def _duplicate_claim_audit(events: Sequence[Any]) -> dict[str, Any]:
    clusters: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        clusters[_normalize_claim(event.claim)].append(event)
    duplicates = [
        (claim, sorted(values, key=lambda event: event.event_id))
        for claim, values in clusters.items()
        if len(values) > 1
    ]
    duplicates.sort(key=lambda item: sha256_text(item[0]))
    size_counts = Counter(len(values) for _claim, values in duplicates)
    details = [
        {
            "normalized_claim_sha256": sha256_text(claim),
            "size": len(values),
            "event_ids": [event.event_id for event in values],
            "labels": [event.gold_label for event in values],
            "complexities": [event.complexity for event in values],
            "label_conflict": len({event.gold_label for event in values}) > 1,
        }
        for claim, values in duplicates
    ]
    audit = {
        "normalization": "unicode_casefold_whitespace_collapse",
        "cluster_count": len(details),
        "event_count": sum(item["size"] for item in details),
        "extra_event_count": sum(item["size"] - 1 for item in details),
        "cluster_size_counts": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "label_conflict_cluster_count": sum(
            int(item["label_conflict"]) for item in details
        ),
        "clusters": details,
        "retained": True,
        "assignment_unit": "event_id",
        "assignment_is_not_claim_cluster_grouped": True,
        "limitation": (
            "normalized duplicate claims are retained and independently assigned "
            "by event_id to preserve the frozen label x complexity balance"
        ),
    }
    expected = {
        "cluster_count": 15,
        "event_count": 31,
        "extra_event_count": 16,
        "cluster_size_counts": {"2": 14, "3": 1},
    }
    observed = {key: audit[key] for key in expected}
    if observed != expected:
        raise FinetunePrepareError(
            f"Eligible-train duplicate claim audit drift: {observed}"
        )
    return audit


def _candidate_arm(
    event: Any,
    uids: Sequence[str],
    *,
    evidence_arm: str,
    method: str,
) -> dict[str, Any]:
    uid_list = [str(uid) for uid in uids]
    try:
        texts = [event.candidates_by_uid[uid].text for uid in uid_list]
    except KeyError as exc:
        raise FinetunePrepareError(
            f"{event.event_id}: missing candidate UID {exc.args[0]!r}"
        ) from exc
    if len(uid_list) != len(texts) or not uid_list:
        raise FinetunePrepareError(f"{event.event_id}: empty/misaligned arm")
    for text in texts:
        if not text.strip():
            raise FinetunePrepareError(f"{event.event_id}: empty clean evidence")
        if CHECK_CUE.search(text):
            raise FinetunePrepareError(
                f"{event.event_id}: Check: cue leaked into clean evidence"
            )
    return {
        "evidence_arm": evidence_arm,
        "method": method,
        "candidate_uids": uid_list,
        "evidence_texts": texts,
        "evidence_sequence_sha256": _json_sha(texts),
        "evidence_snippet_sha256s": [sha256_text(text) for text in texts],
        "evidence_multiset_sha256": _json_sha(sorted(Counter(texts).items())),
        "character_count": sum(len(text) for text in texts),
    }


def _main_comparison(event: Any) -> dict[str, Any]:
    evi = _candidate_arm(
        event,
        event.evi_visible_uids,
        evidence_arm="evitrace",
        method="evitrace_visible_selection",
    )
    s4 = _candidate_arm(
        event,
        event.s4_order_uids[: event.k_visible],
        evidence_arm="s4",
        method="s4_source_score_top_k",
    )
    return {
        "comparison_id": f"{event.event_id}::main",
        "event_id": event.event_id,
        "split": event.split,
        "comparison_type": "main",
        "claim": event.claim,
        "complexity": event.complexity,
        "k": event.k_visible,
        "k_visible": event.k_visible,
        "arms": {"evitrace": evi, "s4": s4},
    }


def _order_comparison(event: Any) -> dict[str, Any]:
    evi = _candidate_arm(
        event,
        event.evi_visible_uids,
        evidence_arm="evitrace",
        method="evitrace_order",
    )
    s4 = _candidate_arm(
        event,
        event.s4_reordered_evi_uids,
        evidence_arm="s4",
        method="s4_source_score_reorder_same_set",
    )
    if evi["candidate_uids"] == s4["candidate_uids"]:
        raise FinetunePrepareError(
            f"{event.event_id}: identical full order entered Order-only"
        )
    if set(evi["candidate_uids"]) != set(s4["candidate_uids"]):
        raise FinetunePrepareError(
            f"{event.event_id}: final Order-only UID set mismatch"
        )
    if Counter(evi["evidence_texts"]) != Counter(s4["evidence_texts"]):
        raise FinetunePrepareError(
            f"{event.event_id}: final Order-only text multiset mismatch"
        )
    return {
        "comparison_id": f"{event.event_id}::order_only",
        "event_id": event.event_id,
        "split": event.split,
        "comparison_type": "order_only",
        "claim": event.claim,
        "complexity": event.complexity,
        "k": event.k_visible,
        "k_visible": event.k_visible,
        "prefix_relation": "same_set_different_order",
        "arms": {"evitrace": evi, "s4": s4},
    }


def _prefix_relation(evi_uids: Sequence[str], s4_uids: Sequence[str]) -> str:
    if list(evi_uids) == list(s4_uids):
        return "same_order"
    if set(evi_uids) == set(s4_uids):
        return "same_set_different_order"
    return "different_set"


def build_prefix_comparisons(order_events: Sequence[Any]) -> list[dict[str, Any]]:
    """Build every k=1..K prefix pair for final-K Order-only events."""

    rows: list[dict[str, Any]] = []
    for event in order_events:
        if event.order_is_identical:
            raise FinetunePrepareError(
                f"{event.event_id}: identical-order event entered prefix evaluation"
            )
        for k in range(1, event.k_visible + 1):
            evi_uids = event.evi_visible_uids[:k]
            s4_uids = event.s4_reordered_evi_uids[:k]
            relation = _prefix_relation(evi_uids, s4_uids)
            rows.append(
                {
                    "comparison_id": f"{event.event_id}::prefix::{k}",
                    "parent_comparison_id": f"{event.event_id}::order_only",
                    "event_id": event.event_id,
                    "split": event.split,
                    "comparison_type": "prefix",
                    "claim": event.claim,
                    "complexity": event.complexity,
                    "k": k,
                    "k_visible": event.k_visible,
                    "prefix_relation": relation,
                    "positional_only": (
                        relation == "same_set_different_order"
                    ),
                    "arms": {
                        "evitrace": _candidate_arm(
                            event,
                            evi_uids,
                            evidence_arm="evitrace",
                            method="evitrace_prefix",
                        ),
                        "s4": _candidate_arm(
                            event,
                            s4_uids,
                            evidence_arm="s4",
                            method="s4_reordered_prefix",
                        ),
                    },
                }
            )
    return rows


def _gold_row(event: Any) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "gold_label": event.gold_label,
        "gold_id": LABEL_TO_ID[event.gold_label],
        "complexity": event.complexity,
    }


def _assert_gold_free(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        leaked = sorted(_GOLD_KEYS & set(value))
        if leaked:
            raise FinetunePrepareError(f"{context}: gold keys leaked: {leaked}")
        for key, nested in value.items():
            _assert_gold_free(nested, context=f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _assert_gold_free(nested, context=f"{context}[{index}]")


def complementary_assignments(
    events: Sequence[Any],
    *,
    seed: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Assign one arm per event with exact global and per-stratum balance.

    Within each label x complexity cell events are ordered by a seeded SHA-256
    score.  Odd cells receive their extra EviTrace/S4 item in a second seeded
    balancing pass so assignment A is exactly 50/50 globally.  Assignment B is
    the pointwise complement.
    """

    cells: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for event in events:
        cell = (str(event.gold_label), str(event.complexity))
        if cell[0] not in LABEL_TO_ID or cell[1] not in {"single", "multi"}:
            raise FinetunePrepareError(f"Invalid assignment stratum: {cell}")
        cells[cell].append(event)
    expected_cells = {(label, complexity) for label in LABELS for complexity in ("single", "multi")}
    if set(cells) != expected_cells:
        raise FinetunePrepareError(
            f"Expected all 12 assignment strata, found {sorted(cells)}"
        )
    if len({event.event_id for event in events}) != len(events):
        raise FinetunePrepareError("Duplicate event_id in assignment input")
    if len(events) % 2:
        raise FinetunePrepareError("Complementary assignment requires even event count")

    odd_cells = [cell for cell, values in cells.items() if len(values) % 2]
    if len(odd_cells) % 2:
        raise FinetunePrepareError("Odd strata cannot be globally balanced")
    odd_cells.sort(
        key=lambda cell: (
            sha256_text(f"{seed}:odd-cell:{cell[0]}:{cell[1]}"),
            cell,
        )
    )
    evi_extra_cells = set(odd_cells[: len(odd_cells) // 2])

    assignment_a: dict[str, str] = {}
    cell_stats: dict[str, Any] = {}
    for cell in sorted(cells, key=lambda value: (LABEL_TO_ID[value[0]], value[1])):
        values = sorted(
            cells[cell],
            key=lambda event: (
                sha256_text(
                    f"{seed}:assignment-a:{cell[0]}:{cell[1]}:{event.event_id}"
                ),
                event.event_id,
            ),
        )
        evi_count = len(values) // 2 + int(cell in evi_extra_cells)
        for index, event in enumerate(values):
            assignment_a[event.event_id] = "evitrace" if index < evi_count else "s4"
        key = f"{cell[0]}::{cell[1]}"
        cell_stats[key] = {
            "events": len(values),
            "assignment_a": {
                "evitrace": evi_count,
                "s4": len(values) - evi_count,
            },
        }

    assignment_b = {
        event_id: ("s4" if arm == "evitrace" else "evitrace")
        for event_id, arm in assignment_a.items()
    }
    a_counts = Counter(assignment_a.values())
    b_counts = Counter(assignment_b.values())
    expected_half = len(events) // 2
    if a_counts != {"evitrace": expected_half, "s4": expected_half}:
        raise FinetunePrepareError(f"Assignment A is not exactly balanced: {a_counts}")
    if b_counts != {"evitrace": expected_half, "s4": expected_half}:
        raise FinetunePrepareError(f"Assignment B is not exactly balanced: {b_counts}")
    for event_id in assignment_a:
        if assignment_a[event_id] == assignment_b[event_id]:
            raise FinetunePrepareError(f"{event_id}: assignments are not complementary")
    for key, stats in cell_stats.items():
        if abs(
            stats["assignment_a"]["evitrace"] - stats["assignment_a"]["s4"]
        ) > 1:
            raise FinetunePrepareError(f"{key}: within-cell imbalance exceeds one")
        stats["assignment_b"] = {
            "evitrace": stats["assignment_a"]["s4"],
            "s4": stats["assignment_a"]["evitrace"],
        }
    return assignment_a, assignment_b, {
        "seed": int(seed),
        "algorithm": "sha256_rank_with_globally_balanced_odd_strata",
        "assignment_a": dict(a_counts),
        "assignment_b": dict(b_counts),
        "cells": cell_stats,
        "pointwise_complementary": True,
    }


def _coerce_ids(value: Any, *, context: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if value and isinstance(value, list) and isinstance(value[0], list):
        if len(value) != 1:
            raise FinetunePrepareError(f"{context}: expected one tokenized prompt")
        value = value[0]
    if not isinstance(value, list) or not value:
        raise FinetunePrepareError(f"{context}: missing/non-list input IDs")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise FinetunePrepareError(f"{context}: non-integer input IDs") from exc


class PromptRenderer:
    """Render and cache the frozen LIAR6 chat prompt for one tokenizer."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_key: str,
        max_model_len: int = MAX_MODEL_LEN,
    ) -> None:
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))
        from fact_checking.build.prompts import (
            build_system_message,
            build_user_content,
        )

        self.tokenizer = tokenizer
        self.model_key = str(model_key)
        self.max_model_len = int(max_model_len)
        self.system_message = build_system_message(None, "liar6")
        self._build_user_content = build_user_content
        self.cache: dict[str, dict[str, Any]] = {}
        self.label_prefix_ids = _coerce_ids(
            tokenizer(
                LABEL_PREFIX,
                add_special_tokens=False,
                truncation=False,
            ),
            context=f"{model_key}:Label prefix",
        )
        self.label_token_ids: dict[str, int] = {}
        for letter in LETTERS:
            ids = _coerce_ids(
                tokenizer(
                    f" {letter}",
                    add_special_tokens=False,
                    truncation=False,
                ),
                context=f"{model_key}:label {letter}",
            )
            if len(ids) != 1:
                raise FinetunePrepareError(
                    f"{model_key}: label choice {letter!r} is not one token: {ids}"
                )
            self.label_token_ids[letter] = ids[0]
        if len(set(self.label_token_ids.values())) != len(LETTERS):
            raise FinetunePrepareError(
                f"{model_key}: A-F token IDs are not unique: {self.label_token_ids}"
            )

    def render(
        self,
        claim: str,
        evidence_texts: Sequence[str],
    ) -> dict[str, Any]:
        clean_claim = str(claim).strip()
        texts = [str(text).strip() for text in evidence_texts]
        if not clean_claim:
            raise FinetunePrepareError(f"{self.model_key}: empty claim")
        if any(not text for text in texts):
            raise FinetunePrepareError(f"{self.model_key}: empty evidence text")
        if any(CHECK_CUE.search(text) for text in texts):
            raise FinetunePrepareError(f"{self.model_key}: Check: cue in evidence")
        cache_key = _json_sha([clean_claim, texts])
        existing = self.cache.get(cache_key)
        if existing is not None:
            return existing

        user_content = self._build_user_content(
            clean_claim,
            texts,
            "label",
            "letter",
            "liar6",
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_content},
        ]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if not isinstance(prompt, str) or not prompt:
            raise FinetunePrepareError(
                f"{self.model_key}: chat template did not return text"
            )
        prompt_ids = _coerce_ids(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            context=f"{self.model_key}:chat template",
        )
        roundtrip_ids = _coerce_ids(
            self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
            ),
            context=f"{self.model_key}:chat roundtrip",
        )
        if roundtrip_ids != prompt_ids:
            raise FinetunePrepareError(
                f"{self.model_key}: rendered chat differs from tokenized chat template"
            )
        prefixed_roundtrip = _coerce_ids(
            self.tokenizer(
                prompt + LABEL_PREFIX,
                add_special_tokens=False,
                truncation=False,
            ),
            context=f"{self.model_key}:Label-prefixed roundtrip",
        )
        if prefixed_roundtrip != prompt_ids + self.label_prefix_ids:
            raise FinetunePrepareError(
                f"{self.model_key}: appending Label: changes chat token boundary"
            )
        if len(prompt_ids) + len(self.label_prefix_ids) + 2 > self.max_model_len:
            raise FinetunePrepareError(
                f"{self.model_key}: prompt plus Label:/reserved tokens has "
                f"{len(prompt_ids) + len(self.label_prefix_ids) + 2} tokens "
                f"(>{self.max_model_len}); truncation is forbidden"
            )
        payload = {
            "prompt": prompt,
            "prompt_text_sha256": sha256_text(prompt),
            "prompt_input_ids": prompt_ids,
            "prompt_input_ids_sha256": _json_sha(prompt_ids),
            "prompt_token_count": len(prompt_ids),
        }
        self.cache[cache_key] = payload
        return payload

    def supervised_target(self, gold_label: str) -> tuple[str, int]:
        if gold_label not in LABEL_TO_LETTER:
            raise FinetunePrepareError(
                f"{self.model_key}: invalid gold label {gold_label!r}"
            )
        target = f"{LABEL_PREFIX} {LABEL_TO_LETTER[gold_label]}"
        target_ids = _coerce_ids(
            self.tokenizer(
                target,
                add_special_tokens=False,
                truncation=False,
            ),
            context=f"{self.model_key}:target",
        )
        target_count = len(target_ids) + int(
            getattr(self.tokenizer, "eos_token_id", None) is not None
        )
        return target, target_count

    def validate_supervised_length(
        self,
        prompt_token_count: int,
        target_token_count: int,
        *,
        context: str,
    ) -> None:
        total = int(prompt_token_count) + int(target_token_count)
        if total > self.max_model_len:
            raise FinetunePrepareError(
                f"{context}: supervised sequence has {total} tokens "
                f"(>{self.max_model_len}); truncation is forbidden"
            )


def _arm_for_event(event: Any, evidence_arm: str) -> tuple[list[str], list[str]]:
    if evidence_arm == "evitrace":
        uids = list(event.evi_visible_uids)
    elif evidence_arm == "s4":
        uids = list(event.s4_order_uids[: event.k_visible])
    else:
        raise FinetunePrepareError(f"Unsupported training evidence_arm={evidence_arm!r}")
    texts = [event.candidates_by_uid[uid].text for uid in uids]
    return uids, texts


def _supervised_row(
    event: Any,
    renderer: PromptRenderer,
    *,
    assignment_id: str,
    evidence_arm: str,
    candidate_uids: Sequence[str] | None = None,
    evidence_texts: Sequence[str] | None = None,
    comparison_type: str = "main",
    donor_event_id: str | None = None,
) -> dict[str, Any]:
    if candidate_uids is None or evidence_texts is None:
        candidate_uids, evidence_texts = _arm_for_event(event, evidence_arm)
    uid_list = [str(uid) for uid in candidate_uids]
    text_list = [str(text) for text in evidence_texts]
    payload = renderer.render(event.claim, text_list)
    target, target_count = renderer.supervised_target(event.gold_label)
    renderer.validate_supervised_length(
        payload["prompt_token_count"],
        target_count,
        context=f"{renderer.model_key}:{assignment_id}:{event.event_id}",
    )
    row = {
        "event_id": event.event_id,
        "pair_id": f"{event.event_id}::{comparison_type}",
        "split": event.split,
        "assignment_id": assignment_id,
        "comparison_type": comparison_type,
        "evidence_arm": evidence_arm,
        "claim": event.claim,
        "complexity": event.complexity,
        "k_visible": event.k_visible,
        "candidate_uids": uid_list,
        "evidence_sequence_sha256": _json_sha(text_list),
        "evidence_snippet_sha256s": [sha256_text(text) for text in text_list],
        "prompt": payload["prompt"],
        "prompt_input_ids": list(payload["prompt_input_ids"]),
        "prompt_input_ids_sha256": payload["prompt_input_ids_sha256"],
        "prompt_text_sha256": payload["prompt_text_sha256"],
        "prompt_token_count": payload["prompt_token_count"],
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "target": target,
        "target_token_count": target_count,
        "gold_label": event.gold_label,
        "gold_id": LABEL_TO_ID[event.gold_label],
        "gold_explain": "",
        "label_schema": "liar6",
        "evidence_count": len(text_list),
        "was_truncated": False,
        "evidence_text_truncated": False,
    }
    if donor_event_id is not None:
        row["donor_event_id"] = donor_event_id
        row["donor_evidence_arm"] = "evitrace"
    return row


def _event_visible_character_count(event: Any) -> int:
    return sum(
        len(event.candidates_by_uid[uid].text)
        for uid in event.evi_visible_uids
    )


def _distribution_summary(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0, "max": 0}
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    p95_index = max(
        0,
        min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1),
    )
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _mismatch_donors(
    events: Sequence[Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct a within-stratum, one-to-one, approximate-length derangement."""

    strata: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for event in events:
        strata[(event.gold_label, event.complexity)].append(event)
    mapping: dict[str, Any] = {}
    stratum_audit: dict[str, Any] = {}
    for stratum in sorted(
        strata,
        key=lambda value: (LABEL_TO_ID[value[0]], value[1]),
    ):
        values = sorted(
            strata[stratum],
            key=lambda event: (
                event.k_visible,
                _event_visible_character_count(event),
                sha256_text(
                    f"{seed}:mismatched-val:{stratum[0]}:"
                    f"{stratum[1]}:{event.event_id}"
                ),
                event.event_id,
            ),
        )
        if len(values) < 2:
            raise FinetunePrepareError(
                f"Val mismatch stratum {stratum} has fewer than two events"
            )
        paired_until = len(values)
        if len(values) % 2:
            paired_until -= 3
        for index in range(0, paired_until, 2):
            left, right = values[index : index + 2]
            mapping[left.event_id] = right
            mapping[right.event_id] = left
        if paired_until < len(values):
            first, second, third = values[-3:]
            mapping[first.event_id] = second
            mapping[second.event_id] = third
            mapping[third.event_id] = first
        stratum_audit[f"{stratum[0]}::{stratum[1]}"] = {
            "events": len(values),
            "construction": (
                "adjacent_swap"
                if len(values) % 2 == 0
                else "adjacent_swap_plus_final_3_cycle"
            ),
        }

    if set(mapping) != {event.event_id for event in events}:
        raise FinetunePrepareError("Mismatched val derangement is incomplete")
    if len({donor.event_id for donor in mapping.values()}) != len(events):
        raise FinetunePrepareError("Mismatched val donors are not one-to-one")
    k_differences: list[int] = []
    character_differences: list[int] = []
    event_by_id = {event.event_id: event for event in events}
    for event_id, donor in mapping.items():
        event = event_by_id[event_id]
        if donor.event_id == event_id:
            raise FinetunePrepareError(f"{event_id}: mismatched val self-loop")
        if (
            donor.gold_label != event.gold_label
            or donor.complexity != event.complexity
        ):
            raise FinetunePrepareError(
                f"{event_id}: mismatched donor left label x complexity stratum"
            )
        k_differences.append(abs(event.k_visible - donor.k_visible))
        character_differences.append(
            abs(
                _event_visible_character_count(event)
                - _event_visible_character_count(donor)
            )
        )
    return mapping, {
        "algorithm": (
            "within_gold_label_x_complexity_sorted_by_K_character_length_"
            "then_adjacent_swap_or_final_3_cycle"
        ),
        "one_to_one": True,
        "no_self_loop": True,
        "same_gold_label": True,
        "same_complexity": True,
        "strata": stratum_audit,
        "absolute_k_difference": _distribution_summary(k_differences),
        "absolute_visible_character_difference": _distribution_summary(
            character_differences
        ),
    }


def _training_rows(
    events: Sequence[Any],
    assignment: Mapping[str, str],
    renderer: PromptRenderer,
    *,
    assignment_id: str,
) -> list[dict[str, Any]]:
    rows = [
        _supervised_row(
            event,
            renderer,
            assignment_id=assignment_id,
            evidence_arm=assignment[event.event_id],
        )
        for event in sorted(events, key=lambda value: value.event_id)
    ]
    arms = Counter(row["evidence_arm"] for row in rows)
    if arms != {"evitrace": len(events) // 2, "s4": len(events) // 2}:
        raise FinetunePrepareError(f"{assignment_id}: arm count drift: {arms}")
    if any(row["evidence_arm"] not in {"evitrace", "s4"} for row in rows):
        raise FinetunePrepareError(f"{assignment_id}: invalid training evidence_arm")
    return rows


def _paired_val_rows(
    events: Sequence[Any],
    renderer: PromptRenderer,
) -> list[dict[str, Any]]:
    return [
        _supervised_row(
            event,
            renderer,
            assignment_id="paired_val",
            evidence_arm=evidence_arm,
            comparison_type="val_paired",
        )
        for event in sorted(events, key=lambda value: value.event_id)
        for evidence_arm in ("evitrace", "s4")
    ]


def _claim_only_val_rows(
    events: Sequence[Any],
    renderer: PromptRenderer,
) -> list[dict[str, Any]]:
    return [
        _supervised_row(
            event,
            renderer,
            assignment_id="claim_only_val",
            evidence_arm="claim_only",
            candidate_uids=[],
            evidence_texts=[],
            comparison_type="val_claim_only",
        )
        for event in sorted(events, key=lambda value: value.event_id)
    ]


def _mismatched_val_rows(
    events: Sequence[Any],
    renderer: PromptRenderer,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donors, mismatch_audit = _mismatch_donors(events, seed=seed)
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda value: value.event_id):
        donor = donors[event.event_id]
        donor_uids, donor_texts = _arm_for_event(donor, "evitrace")
        row = _supervised_row(
            event,
            renderer,
            assignment_id="mismatched_val",
            evidence_arm="mismatched",
            candidate_uids=donor_uids,
            evidence_texts=donor_texts,
            comparison_type="val_mismatched",
            donor_event_id=donor.event_id,
        )
        row["donor_k_visible"] = donor.k_visible
        row["mismatch_absolute_k_difference"] = abs(
            event.k_visible - donor.k_visible
        )
        row["mismatch_absolute_character_difference"] = abs(
            _event_visible_character_count(event)
            - _event_visible_character_count(donor)
        )
        rows.append(row)
    if len({row["donor_event_id"] for row in rows}) != len(events):
        raise FinetunePrepareError("Mismatched val donors are not a permutation")
    return rows, mismatch_audit


def build_eval_registry(
    comparisons: Sequence[Mapping[str, Any]],
    prefixes: Sequence[Mapping[str, Any]],
    renderer: PromptRenderer,
) -> list[dict[str, Any]]:
    """Materialize one gold-free chat-prompt row per logical model arm."""

    logical_sources = list(comparisons) + list(prefixes)
    rows: list[dict[str, Any]] = []
    for comparison in logical_sources:
        arms = comparison.get("arms")
        if not isinstance(arms, Mapping):
            raise FinetunePrepareError(
                f"{comparison.get('comparison_id')}: missing arms"
            )
        for evidence_arm in ("evitrace", "s4"):
            arm = arms.get(evidence_arm)
            if not isinstance(arm, Mapping):
                raise FinetunePrepareError(
                    f"{comparison.get('comparison_id')}: missing {evidence_arm} arm"
                )
            payload = renderer.render(
                str(comparison["claim"]),
                list(arm.get("evidence_texts") or []),
            )
            row = {
                "logical_id": f"{comparison['comparison_id']}::{evidence_arm}",
                "comparison_id": comparison["comparison_id"],
                "event_id": comparison["event_id"],
                "split": "test",
                "comparison_type": comparison["comparison_type"],
                "evidence_arm": evidence_arm,
                "k": int(comparison["k"]),
                "k_visible": int(comparison["k_visible"]),
                "prefix_relation": str(
                    comparison.get("prefix_relation") or "not_applicable"
                ),
                "candidate_uids": list(arm.get("candidate_uids") or []),
                "evidence_sequence_sha256": arm["evidence_sequence_sha256"],
                "evidence_snippet_sha256s": list(
                    arm.get("evidence_snippet_sha256s") or []
                ),
                "prompt_input_ids": list(payload["prompt_input_ids"]),
                "prompt_input_ids_sha256": payload["prompt_input_ids_sha256"],
                "prompt_text_sha256": payload["prompt_text_sha256"],
                "prompt_token_count": payload["prompt_token_count"],
                "prompt_add_special_tokens": False,
                "preserve_prompt_prefix": True,
                "label_prefix": LABEL_PREFIX,
                "label_prefix_in_prompt_input_ids": False,
            }
            _assert_gold_free(row, context=row["logical_id"])
            rows.append(row)
    logical_ids = [row["logical_id"] for row in rows]
    if len(logical_ids) != len(set(logical_ids)):
        raise FinetunePrepareError("Duplicate logical_id in eval registry")
    return rows


def build_val_eval_registry(
    val_paired: Sequence[Mapping[str, Any]],
    val_claim_only: Sequence[Mapping[str, Any]],
    val_mismatched: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project supervised val rows into gold-free logical inference rows."""

    groups = (
        ("val_paired", val_paired),
        ("val_claim_only", val_claim_only),
        ("val_mismatched", val_mismatched),
    )
    registry: list[dict[str, Any]] = []
    for comparison_type, rows in groups:
        for source in rows:
            evidence_arm = str(source["evidence_arm"])
            logical_id = (
                f"{source['event_id']}::{comparison_type}::{evidence_arm}"
            )
            row = {
                "logical_id": logical_id,
                "comparison_id": f"{source['event_id']}::{comparison_type}",
                "event_id": source["event_id"],
                "split": "val",
                "comparison_type": comparison_type,
                "evidence_arm": evidence_arm,
                "k": int(source["evidence_count"]),
                "k_visible": int(source["k_visible"]),
                "prefix_relation": "not_applicable",
                "candidate_uids": list(source.get("candidate_uids") or []),
                "evidence_sequence_sha256": source[
                    "evidence_sequence_sha256"
                ],
                "evidence_snippet_sha256s": list(
                    source.get("evidence_snippet_sha256s") or []
                ),
                "prompt_input_ids": list(source["prompt_input_ids"]),
                "prompt_input_ids_sha256": source[
                    "prompt_input_ids_sha256"
                ],
                "prompt_text_sha256": source["prompt_text_sha256"],
                "prompt_token_count": int(source["prompt_token_count"]),
                "prompt_add_special_tokens": False,
                "preserve_prompt_prefix": True,
                "label_prefix": LABEL_PREFIX,
                "label_prefix_in_prompt_input_ids": False,
            }
            if comparison_type == "val_mismatched":
                row.update(
                    {
                        "donor_event_id": source["donor_event_id"],
                        "donor_k_visible": int(source["donor_k_visible"]),
                        "mismatch_absolute_k_difference": int(
                            source["mismatch_absolute_k_difference"]
                        ),
                        "mismatch_absolute_character_difference": int(
                            source[
                                "mismatch_absolute_character_difference"
                            ]
                        ),
                    }
                )
            _assert_gold_free(row, context=logical_id)
            registry.append(row)
    logical_ids = [row["logical_id"] for row in registry]
    if len(logical_ids) != len(set(logical_ids)):
        raise FinetunePrepareError("Duplicate logical_id in val eval registry")
    return registry


def _length_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(int(row["prompt_token_count"]) for row in rows)
    if not values:
        return {"rows": 0, "max": 0, "p95": 0}
    p95_index = max(0, min(len(values) - 1, int(0.95 * len(values) + 0.999999) - 1))
    return {
        "rows": len(values),
        "max": values[-1],
        "p95_nearest_rank": values[p95_index],
    }


def _tokenizer_directory_metadata(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FinetunePrepareError(f"Model/tokenizer directory does not exist: {path}")
    files: list[dict[str, Any]] = []
    for candidate in sorted(value for value in path.rglob("*") if value.is_file()):
        if candidate.suffix.lower() in _WEIGHT_SUFFIXES:
            continue
        files.append(
            {
                "path": str(candidate.relative_to(path)),
                "sha256": sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
        )
    if not files:
        raise FinetunePrepareError(f"No tokenizer/config files found under {path}")
    return {
        "sha256": _json_sha(files),
        "files": files,
    }


def _default_tokenizer_loader(model_path: Path) -> Any:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from sft.runtime.model_loading import load_compatible_tokenizer

    return load_compatible_tokenizer(str(model_path), trust_remote_code=True)


def _prepare_model_files(
    *,
    model_key: str,
    model_path: Path,
    output_dir: Path,
    train_events: Sequence[Any],
    val_events: Sequence[Any],
    comparisons: Sequence[Mapping[str, Any]],
    prefixes: Sequence[Mapping[str, Any]],
    assignment_a: Mapping[str, str],
    assignment_b: Mapping[str, str],
    seed: int,
    tokenizer_loader: Callable[[Path], Any],
) -> dict[str, Any]:
    tokenizer = tokenizer_loader(model_path)
    renderer = PromptRenderer(tokenizer, model_key=model_key)
    train_a = _training_rows(
        train_events,
        assignment_a,
        renderer,
        assignment_id="assignment_a",
    )
    train_b = _training_rows(
        train_events,
        assignment_b,
        renderer,
        assignment_id="assignment_b",
    )
    val_paired = _paired_val_rows(val_events, renderer)
    val_claim_only = _claim_only_val_rows(val_events, renderer)
    val_mismatched, mismatch_audit = _mismatched_val_rows(
        val_events,
        renderer,
        seed=seed,
    )
    test_eval_registry = build_eval_registry(comparisons, prefixes, renderer)
    val_eval_registry = build_val_eval_registry(
        val_paired,
        val_claim_only,
        val_mismatched,
    )
    if len(test_eval_registry) != EXPECTED_TEST_LOGICAL_ROWS:
        raise FinetunePrepareError(
            f"{model_key}: expected {EXPECTED_TEST_LOGICAL_ROWS} test logical "
            f"rows, found {len(test_eval_registry)}"
        )
    if len(val_eval_registry) != EXPECTED_VAL_LOGICAL_ROWS:
        raise FinetunePrepareError(
            f"{model_key}: expected {EXPECTED_VAL_LOGICAL_ROWS} val logical "
            f"rows, found {len(val_eval_registry)}"
        )
    eval_registry = test_eval_registry + val_eval_registry

    expected_rows = {
        "train_assignment_a": EXPECTED_COUNTS["train"],
        "train_assignment_b": EXPECTED_COUNTS["train"],
        "val_paired": 2 * EXPECTED_COUNTS["val"],
        "val_claim_only": EXPECTED_COUNTS["val"],
        "val_mismatched": EXPECTED_COUNTS["val"],
        "eval_registry": EXPECTED_EVAL_LOGICAL_ROWS,
    }
    row_sets = {
        "train_assignment_a": train_a,
        "train_assignment_b": train_b,
        "val_paired": val_paired,
        "val_claim_only": val_claim_only,
        "val_mismatched": val_mismatched,
        "eval_registry": eval_registry,
    }
    for name, rows in row_sets.items():
        if len(rows) != expected_rows[name]:
            raise FinetunePrepareError(
                f"{model_key}/{name}: expected {expected_rows[name]} rows, "
                f"found {len(rows)}"
            )
    a_by_event = {row["event_id"]: row for row in train_a}
    b_by_event = {row["event_id"]: row for row in train_b}
    if set(a_by_event) != set(b_by_event):
        raise FinetunePrepareError(f"{model_key}: assignment event sets differ")
    for event_id in a_by_event:
        if a_by_event[event_id]["evidence_arm"] == b_by_event[event_id]["evidence_arm"]:
            raise FinetunePrepareError(
                f"{model_key}/{event_id}: assignment files are not complementary"
            )
    if any(row.get("assignment_id") != "paired_val" for row in val_paired):
        raise FinetunePrepareError(f"{model_key}: paired val assignment_id drift")

    model_output = output_dir / "models" / model_key
    file_metadata: dict[str, Any] = {}
    for name, rows in row_sets.items():
        path = model_output / f"{name}.jsonl"
        count = _write_jsonl(path, rows)
        file_metadata[name] = _file_metadata(path, count)

    length_stats = {
        name: _length_stats(rows) for name, rows in row_sets.items()
    }
    if max(stats["max"] for stats in length_stats.values()) >= MAX_MODEL_LEN:
        raise FinetunePrepareError(
            f"{model_key}: a stored chat prompt reaches max_model_len"
        )
    tokenizer_metadata = _tokenizer_directory_metadata(model_path)
    return {
        "model_path": str(model_path),
        "tokenizer_sha256": tokenizer_metadata["sha256"],
        "tokenizer_files": tokenizer_metadata["files"],
        "chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "label_prefix": LABEL_PREFIX,
        "label_prefix_in_prompt_input_ids": False,
        "label_token_ids": renderer.label_token_ids,
        "mismatched_val_audit": mismatch_audit,
        "prompt_length_stats": length_stats,
        "files": file_metadata,
    }


def prepare_dataset(
    args: argparse.Namespace,
    *,
    exporter_module: Any | None = None,
    tokenizer_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Prepare all shared and tokenizer-specific files and return the manifest."""

    seed = int(getattr(args, "seed", DEFAULT_SEED))
    if seed != DEFAULT_SEED:
        raise FinetunePrepareError(
            f"Frozen preparation requires seed={DEFAULT_SEED}, found {seed}"
        )
    output_dir = _argument_path(args, "output_dir", DEFAULT_OUTPUT_DIR)
    source_paths = _source_paths(args)
    source_metadata = _assert_frozen_artifacts(source_paths)
    exporter = exporter_module or _load_exporter()
    current_ranker = exporter._load_current_s4_ranker()

    raw_events: dict[str, list[Any]] = {}
    source_audits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        events, audit = exporter.load_aligned_events(
            source_paths[f"build_{split}"],
            source_paths[f"evitrace_{split}"],
            source_paths[f"s4_{split}"],
            split=split,
            expected_event_count=EXPECTED_RAW_COUNTS[split],
            current_s4_ranker=current_ranker,
        )
        raw_events[split] = events
        source_audits[split] = audit
    events = {
        split: _filter_events(raw_events[split], split=split)
        for split in ("train", "val", "test")
    }
    _assert_no_cross_split_claim_overlap(
        events["train"],
        events["val"],
        events["test"],
    )

    test_main = [_main_comparison(event) for event in events["test"]]
    identical_order = [event for event in events["test"] if event.order_is_identical]
    order_events = [event for event in events["test"] if not event.order_is_identical]
    if len(identical_order) != EXPECTED_ORDER_IDENTICAL:
        raise FinetunePrepareError(
            f"Expected {EXPECTED_ORDER_IDENTICAL} identical-order test events, "
            f"found {len(identical_order)}"
        )
    if len(order_events) != EXPECTED_ORDER:
        raise FinetunePrepareError(
            f"Expected {EXPECTED_ORDER} final-K Order-only events, found {len(order_events)}"
        )
    order_labels = Counter(event.gold_label for event in order_events)
    order_complexities = Counter(event.complexity for event in order_events)
    if dict(order_labels) != ORDER_LABEL_COUNTS:
        raise FinetunePrepareError(f"Order label-count drift: {order_labels}")
    if dict(order_complexities) != ORDER_COMPLEXITY_COUNTS:
        raise FinetunePrepareError(
            f"Order complexity-count drift: {order_complexities}"
        )
    test_order = [_order_comparison(event) for event in order_events]
    comparisons = test_main + test_order
    prefixes = build_prefix_comparisons(order_events)
    prefix_relations = Counter(row["prefix_relation"] for row in prefixes)
    if len(comparisons) != EXPECTED_COMPARISONS:
        raise FinetunePrepareError(
            f"Expected {EXPECTED_COMPARISONS} comparisons, found {len(comparisons)}"
        )
    if len(prefixes) != EXPECTED_PREFIX:
        raise FinetunePrepareError(
            f"Expected {EXPECTED_PREFIX} prefix pairs, found {len(prefixes)}"
        )
    if dict(prefix_relations) != EXPECTED_PREFIX_RELATIONS:
        raise FinetunePrepareError(
            f"Prefix relation inventory drift: {dict(prefix_relations)}"
        )
    for row in comparisons:
        _assert_gold_free(row, context=row["comparison_id"])
    for row in prefixes:
        _assert_gold_free(row, context=row["comparison_id"])

    assignment_a, assignment_b, assignment_audit = complementary_assignments(
        events["train"],
        seed=seed,
    )
    duplicate_claim_audit = _duplicate_claim_audit(events["train"])
    assignment_audit["assignment_unit"] = "event_id"
    assignment_audit["retained_duplicate_claim_audit"] = duplicate_claim_audit

    shared_dir = output_dir / "shared"
    shared_rows = {
        "gold_test": [_gold_row(event) for event in events["test"]],
        "gold_val": [_gold_row(event) for event in events["val"]],
        "comparisons": comparisons,
        "prefix": prefixes,
    }
    shared_files: dict[str, Any] = {}
    for name, rows in shared_rows.items():
        path = shared_dir / f"{name}.jsonl"
        count = _write_jsonl(path, rows)
        shared_files[name] = _file_metadata(path, count)

    loader = tokenizer_loader or _default_tokenizer_loader
    model_paths = {
        "qwen3": _argument_path(
            args, "qwen_model_path", DEFAULT_MODEL_PATHS["qwen3"]
        ),
        "llama31": _argument_path(
            args, "llama_model_path", DEFAULT_MODEL_PATHS["llama31"]
        ),
    }
    models: dict[str, Any] = {}
    for model_key in ("qwen3", "llama31"):
        models[model_key] = _prepare_model_files(
            model_key=model_key,
            model_path=model_paths[model_key],
            output_dir=output_dir,
            train_events=events["train"],
            val_events=events["val"],
            comparisons=comparisons,
            prefixes=prefixes,
            assignment_a=assignment_a,
            assignment_b=assignment_b,
            seed=seed,
            tokenizer_loader=loader,
        )

    manifest = {
        "schema_version": 1,
        "experiment": "evitrace_cross_verifier_finetune_v1",
        "complete": True,
        "seed": seed,
        "max_model_len": MAX_MODEL_LEN,
        "source_artifacts": source_metadata,
        "source_audits": source_audits,
        "exclusions": {
            split: sorted(SPLIT_EXCLUSIONS[split])
            for split in ("train", "val", "test")
        },
        "contracts": {
            "raw_event_counts": EXPECTED_RAW_COUNTS,
            "eligible_event_counts": EXPECTED_COUNTS,
            "k_field": "evidence_count",
            "forbidden_k_field": "evidence_count_before",
            "main_comparisons": len(test_main),
            "order_comparisons": len(test_order),
            "comparisons_total": len(comparisons),
            "prefix_pairs": len(prefixes),
            "prefix_relations": dict(prefix_relations),
            "positional_only_prefix_pairs": prefix_relations[
                "same_set_different_order"
            ],
            "planned_count_7448_corrected_after_artifact_audit": True,
            "prefix_count_correction": {
                "preaudit_planned_count": PREAUDIT_PLANNED_PREFIX,
                "artifact_audited_count": EXPECTED_PREFIX,
                "scope": "1152_final_order_eligible_events_only",
                "excluded_final_order_identical_events": EXPECTED_ORDER_IDENTICAL,
                "excluded_prefix_positions": IDENTICAL_ORDER_EXCLUDED_PREFIX_POSITIONS,
                "identity": (
                    f"{PREAUDIT_PLANNED_PREFIX}-"
                    f"{IDENTICAL_ORDER_EXCLUDED_PREFIX_POSITIONS}="
                    f"{EXPECTED_PREFIX}"
                ),
            },
            "test_eval_logical_rows_per_model": EXPECTED_TEST_LOGICAL_ROWS,
            "val_eval_logical_rows_per_model": EXPECTED_VAL_LOGICAL_ROWS,
            "eval_logical_rows_per_model": EXPECTED_EVAL_LOGICAL_ROWS,
            "gold_test_physically_separate": True,
            "eval_registry_gold_free": True,
            "eval_prompt_input_ids": "chat_only_without_Label_prefix",
            "eligible_train_duplicate_claims": duplicate_claim_audit,
        },
        "assignment": assignment_audit,
        "shared_files": shared_files,
        "prepared_files": shared_files,
        "models": models,
        "code": {
            "prepare_module": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "canonical_exporter": {
                "path": str(EXPORTER_PATH.resolve()),
                "sha256": sha256_file(EXPORTER_PATH),
            },
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def add_prepare_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for name, default in DEFAULT_ARTIFACT_PATHS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            default=str(default),
        )
    parser.add_argument(
        "--qwen-model-path",
        default=str(DEFAULT_MODEL_PATHS["qwen3"]),
    )
    parser.add_argument(
        "--llama-model-path",
        default=str(DEFAULT_MODEL_PATHS["llama31"]),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_prepare_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_dataset(args)
    print(
        canonical_json(
            {
                "complete": manifest["complete"],
                "manifest": str(
                    _argument_path(args, "output_dir", DEFAULT_OUTPUT_DIR)
                    / "artifact_manifest.json"
                ),
                "eligible_event_counts": manifest["contracts"][
                    "eligible_event_counts"
                ],
                "prefix_pairs": manifest["contracts"]["prefix_pairs"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
