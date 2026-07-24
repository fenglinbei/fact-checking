#!/usr/bin/env python3
"""Prepare the blinded EviTrace human-alignment experiment.

The exporter intentionally has no network code.  English is authoritative;
Chinese text is read from an optional, content-addressed translation cache and
all cache misses are written to ``translation_requests.jsonl``.

The frozen test artifacts are audited before sampling:

* all three JSONL files have the same, unique event sequence;
* the EviTrace candidate identity is unique and maps into the S4 Atom-Union
  pool;
* the stored S4 order exactly matches the current source-score formula;
* the verifier-visible EviTrace prefix is the first ``evidence_count``
  candidates, never ``evidence_count_before`` or the full 20-step trace.

Public task JSONL files contain only the documented Label Studio import fields.
All event IDs, labels, candidate IDs, token counts, strata, transition
operations, and A/B mappings are kept in private key files.  The public task
manifest contains SHA-256 commitments to those private files.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ANNOTATION_ROOT = Path(__file__).resolve().parents[1]

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
DEFAULT_OUTPUT_DIR = ANNOTATION_ROOT / "results/exp3_trace_alignment_v1"
DEFAULT_ATOM_PARSE_TEST = (
    PROJECT_ROOT
    / "outputs/selectors/atom_anchor/liar_raw_abc_v0_1/"
    "01_claim_atoms/claim_atoms_test.jsonl"
)
DEFAULT_PREFERENCE_CONFIG = ANNOTATION_ROOT / "config/exp3_trace_preference.xml"
DEFAULT_TRANSITION_CONFIG = ANNOTATION_ROOT / "config/exp4_transition_audit.xml"

DEFAULT_SAMPLE_SEED = 20260724
DEFAULT_N_MAIN = 120
DEFAULT_N_ORDER = 80
DEFAULT_N_TRANSITION = 100
DEFAULT_N_PILOT_MAIN = 20
DEFAULT_N_PILOT_ORDER = 10
DEFAULT_N_PILOT_TRANSITION = 15

EXPECTED_TEST_EVENTS = 1251
EXPECTED_VAL_EVENTS = 1274
EXCLUDED_TEST_EVENT_IDS = frozenset({"7845.json"})

LIAR6_LABELS = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
COMPLEXITIES = ("single", "multi")
TRANSITION_OPERATIONS = ("OPEN", "CONTRAST", "BRIDGE", "CORROBORATE", "FALLBACK")
TRANSITION_WEIGHTS = {
    "OPEN": 40,
    "CONTRAST": 20,
    "BRIDGE": 20,
    "CORROBORATE": 10,
    "FALLBACK": 10,
}
CHANGE_OPERATIONS = frozenset({"OPEN", "CONTRAST"})
SELF_TRANSITION_OPERATIONS = frozenset({"BRIDGE", "CORROBORATE", "FALLBACK"})
VALID_STATES = frozenset({"U", "S", "R", "Q", "C"})

PREFERENCE_PUBLIC_FIELDS = frozenset(
    {
        "blind_task_id",
        "claim_en",
        "claim_zh",
        "sequence_a_html",
        "sequence_b_html",
    }
)
TRANSITION_PUBLIC_FIELDS = frozenset(
    {
        "blind_task_id",
        "claim_en",
        "claim_zh",
        "focal_atom_en",
        "focal_atom_zh",
        "state_legend_html",
        "prior_evidence_html",
        "current_evidence_html",
        "proposed_transition",
    }
)

_CHECK_CUE_LINE = re.compile(r"(?im)^\s*Check:\s*[^\n]*(?:\n+|$)")
_FORBIDDEN_PUBLIC_MARKERS = (
    "candidate_uid",
    "gold_label",
    "mrec_",
    "evitrace",
    "source_score",
    "selector_score",
    "map_relation",
    "map_directness",
    "map_confidence",
)


class ExportError(RuntimeError):
    """Raised when a frozen-data or blinding contract is violated."""


@dataclass(frozen=True)
class CandidateView:
    uid: str
    key: str
    text: str
    source_id: str
    domain: str
    artifact_token_count: int


@dataclass(frozen=True)
class EventView:
    event_id: str
    split: str
    claim: str
    gold_label: str
    atoms: tuple[dict[str, Any], ...]
    complexity: str
    k_selected: int
    k_visible: int
    candidates_by_uid: Mapping[str, CandidateView]
    uid_by_key: Mapping[str, str]
    evi_visible_uids: tuple[str, ...]
    s4_order_uids: tuple[str, ...]
    s4_reordered_evi_uids: tuple[str, ...]
    visible_steps: tuple[dict[str, Any], ...]

    @property
    def order_is_identical(self) -> bool:
        return self.evi_visible_uids == self.s4_reordered_evi_uids


@dataclass(frozen=True)
class SplitSelection:
    main: tuple[EventView, ...]
    order_only: tuple[EventView, ...]
    transitions: tuple[tuple[EventView, dict[str, Any]], ...]
    stats: dict[str, Any]


class EvidenceTokenCounter:
    """Interface used to count evidence-only tokenizer tokens."""

    def count(self, candidate: CandidateView) -> int:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class ArtifactTokenCounter(EvidenceTokenCounter):
    """Use the frozen selector's per-candidate token-cost field.

    This mode is useful for synthetic tests and emergency audits.  The formal
    export defaults to ``TokenizerTokenCounter``.
    """

    def count(self, candidate: CandidateView) -> int:
        return int(candidate.artifact_token_count)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "artifact_mrec_token_cost",
            "definition": "sum of frozen per-candidate mrec_token_cost values",
        }


class WhitespaceTokenCounter(EvidenceTokenCounter):
    """Small deterministic counter for unit tests."""

    def count(self, candidate: CandidateView) -> int:
        return len(candidate.text.split())

    def metadata(self) -> dict[str, Any]:
        return {"kind": "whitespace_test_counter"}


class TokenizerTokenCounter(EvidenceTokenCounter):
    """Count each clean evidence text with a local Hugging Face tokenizer."""

    def __init__(self, tokenizer_path: Path):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise ExportError(
                "transformers is required for formal tokenizer counts; "
                "run with the pipeline Python environment"
            ) from exc
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                fix_mistral_regex=True,
                **kwargs,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), **kwargs)
        self._path = tokenizer_path.resolve()
        self._cache: dict[str, int] = {}

    def count(self, candidate: CandidateView) -> int:
        text = candidate.text
        if text not in self._cache:
            self._cache[text] = len(
                self._tokenizer.encode(text, add_special_tokens=False)
            )
        return self._cache[text]

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "huggingface_tokenizer",
            "tokenizer_path": str(self._path),
            "definition": (
                "sum over clean evidence items of encode(text, "
                "add_special_tokens=False); order-invariant by construction"
            ),
        }


class TranslationCache:
    """Content cache that guarantees identical English gets identical Chinese."""

    def __init__(
        self,
        translations: Mapping[str, str] | None = None,
        *,
        source_file_sha256: str | None = None,
    ):
        self._translations: dict[str, str] = {}
        self.source_file_sha256 = source_file_sha256
        for source, target in (translations or {}).items():
            self.add(str(source), str(target))

    def add(self, source: str, target: str) -> None:
        source = source.strip()
        target = target.strip()
        if not source or not target:
            return
        previous = self._translations.get(source)
        if previous is not None and previous != target:
            raise ExportError(
                "translation cache contains conflicting translations for "
                f"{source[:80]!r}"
            )
        self._translations[source] = target

    def get(self, source: str) -> str:
        return self._translations.get(source.strip(), "")

    @property
    def entry_count(self) -> int:
        return len(self._translations)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            self._translations,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return _sha256_text(payload)

    @classmethod
    def load(cls, path: Path | None) -> "TranslationCache":
        cache = cls()
        if path is None:
            return cache
        if not path.exists():
            raise ExportError(f"translation cache does not exist: {path}")
        raw_bytes = path.read_bytes()
        raw = raw_bytes.decode("utf-8")
        cache.source_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if path.suffix.lower() == ".jsonl":
            values: Any = [
                json.loads(line)
                for line in raw.splitlines()
                if line.strip()
            ]
        else:
            values = json.loads(raw)
        _collect_translation_pairs(values, cache)
        return cache


class TranslationResolver:
    def __init__(self, cache: TranslationCache):
        self.cache = cache
        self.field_types: dict[str, set[str]] = defaultdict(set)
        self.missing: dict[str, set[str]] = defaultdict(set)

    def resolve(self, text: str, field_type: str) -> str:
        text = text.strip()
        if not text:
            return ""
        self.field_types[text].add(field_type)
        translated = self.cache.get(text)
        if not translated:
            self.missing[text].add(field_type)
        return translated

    def request_rows(self) -> list[dict[str, Any]]:
        rows = []
        for text in sorted(self.missing, key=lambda value: _sha256_text(value)):
            rows.append(
                {
                    "translation_key": f"sha256:{_sha256_text(text)}",
                    "text_en": text,
                    "field_types": sorted(self.field_types[text]),
                }
            )
        return rows

    def inventory_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "translation_key": f"sha256:{_sha256_text(text)}",
                "text_en": text,
                "field_types": sorted(self.field_types[text]),
            }
            for text in sorted(self.field_types, key=_sha256_text)
        ]


def _collect_translation_pairs(value: Any, cache: TranslationCache) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_translation_pairs(item, cache)
        return
    if not isinstance(value, dict):
        return
    source = value.get("text_en") or value.get("source_text") or value.get("english")
    target = value.get("text_zh") or value.get("translation") or value.get("chinese")
    if isinstance(source, str) and isinstance(target, str):
        cache.add(source, target)
    if value and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        for key, translated in value.items():
            cache.add(key, translated)
    else:
        for nested in value.values():
            _collect_translation_pairs(nested, cache)


def _clean_evidence_text(value: Any) -> str:
    text = str(value or "").strip()
    text = _CHECK_CUE_LINE.sub("", text).strip()
    if text.startswith("Check:") and "Evidence:" in text:
        text = text.split("Evidence:", 1)[1].strip()
    return text


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("canonical_text") or candidate.get("candidate_key") or "").strip()


def _stable_candidate_uid(event_id: str, key: str) -> str:
    return hashlib.sha1(f"{event_id}|{key}".encode("utf-8")).hexdigest()[:12]


def _source_projection(candidate: Mapping[str, Any], uid: str) -> tuple[str, str]:
    report = candidate.get("source_report")
    report = report if isinstance(report, dict) else {}
    report_id = candidate.get("report_id") or report.get("report_id")
    source_id = f"R{report_id}" if report_id not in (None, "") else f"R{uid[:8]}"
    raw_domain = str(
        candidate.get("source_domain")
        or report.get("domain")
        or report.get("link")
        or candidate.get("source_link")
        or ""
    ).strip()
    parsed = urlparse(raw_domain if "://" in raw_domain else f"//{raw_domain}")
    domain = (parsed.hostname or raw_domain or "unknown-source").lower()
    return source_id, domain


def _candidate_view(event_id: str, candidate: Mapping[str, Any]) -> CandidateView:
    key = _candidate_key(candidate)
    if not key:
        raise ExportError(f"{event_id}: candidate has no canonical identity")
    uid = str(candidate.get("candidate_uid") or _stable_candidate_uid(event_id, key)).strip()
    text = _clean_evidence_text(candidate.get("text") or candidate.get("evidence_text"))
    if not text:
        raise ExportError(f"{event_id}: candidate {uid} has empty clean evidence text")
    source_id, domain = _source_projection(candidate, uid)
    token_cost = candidate.get("mrec_token_cost")
    if token_cost is None:
        token_cost = candidate.get("token_cost")
    if token_cost is None:
        token_cost = max(1, len(text.split()))
    return CandidateView(
        uid=uid,
        key=key,
        text=text,
        source_id=source_id,
        domain=domain,
        artifact_token_count=max(0, int(token_cost)),
    )


def recompute_s4_source_score_order(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Exact standard-library mirror of rank_atom_union_source_score_candidates.

    Constants and tie-breaks are kept identical to
    ``src/fact_checking/selectors/atom_retrieval_union.py``.  Avoiding an import
    here keeps this audit runnable without importing torch.
    """

    scored: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = dict(raw)
        baseline_rank = candidate.get("baseline_rank")
        baseline_component = 0.0
        if candidate.get("from_baseline"):
            baseline_component += 0.04
            if baseline_rank is not None:
                baseline_component += 0.01 / max(float(baseline_rank), 1.0)
        atom_component = float(candidate.get("atom_rrf_score") or 0.0)
        atom_component += 0.004 * float(candidate.get("atom_route_hit_count") or 0.0)
        atom_component += 0.01 * float(candidate.get("atom_max_route_hybrid") or 0.0)
        candidate["atom_union_source_score"] = float(
            baseline_component + atom_component
        )
        scored.append(candidate)
    scored.sort(
        key=lambda candidate: (
            -float(candidate.get("atom_union_source_score") or 0.0),
            int(candidate.get("baseline_rank") or 10**9),
            int(candidate.get("atom_pool_rank") or 10**9),
            int(candidate.get("union_pool_rank") or 10**9),
        )
    )
    for rank, candidate in enumerate(scored, start=1):
        candidate["source_score_rank"] = rank
        candidate["selection_rank"] = rank
    return scored


def _load_current_s4_ranker():
    """Load the repository implementation used to produce the frozen S4 file."""

    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from fact_checking.selectors.atom_retrieval_union import (
            AtomUnionSelectionParams,
            rank_atom_union_source_score_candidates,
        )
    except Exception as exc:  # pragma: no cover - formal environment diagnostic
        raise ExportError(
            "canonical export must import the current "
            "rank_atom_union_source_score_candidates(); run with the pipeline "
            "Python environment (PYTHON_BIN)"
        ) from exc
    return rank_atom_union_source_score_candidates, AtomUnionSelectionParams


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise ExportError(f"artifact does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ExportError(f"{path}:{line_number}: expected JSON object")
            yield value


def load_aligned_events(
    build_path: Path,
    evitrace_path: Path,
    s4_path: Path,
    *,
    split: str,
    expected_event_count: int | None = None,
    current_s4_ranker: tuple[Any, Any] | None = None,
) -> tuple[list[EventView], dict[str, Any]]:
    """Stream, audit, and compact one aligned artifact triple."""

    events: list[EventView] = []
    seen: set[str] = set()
    candidate_count = 0
    s4_candidate_count = 0
    s4_pool_extra_events = 0
    s4_order_differences = 0
    selected_counts: list[int] = []
    visible_counts: list[int] = []
    same_order_count = 0

    rows = itertools.zip_longest(
        _jsonl_rows(build_path),
        _jsonl_rows(evitrace_path),
        _jsonl_rows(s4_path),
        fillvalue=None,
    )
    for row_number, triple in enumerate(rows, start=1):
        build_row, evi_row, s4_row = triple
        if build_row is None or evi_row is None or s4_row is None:
            raise ExportError(
                f"{split}: artifact row counts differ at logical row {row_number}"
            )
        event_ids = (
            str(build_row.get("event_id") or ""),
            str(evi_row.get("event_id") or ""),
            str(s4_row.get("event_id") or ""),
        )
        if not event_ids[0] or len(set(event_ids)) != 1:
            raise ExportError(
                f"{split}: event alignment mismatch at row {row_number}: {event_ids}"
            )
        event_id = event_ids[0]
        if event_id in seen:
            raise ExportError(f"{split}: duplicate event_id {event_id!r}")
        seen.add(event_id)

        claims = (
            str(build_row.get("claim") or "").strip(),
            str(evi_row.get("claim") or "").strip(),
            str(s4_row.get("claim") or "").strip(),
        )
        if not claims[0] or len(set(claims)) != 1:
            raise ExportError(f"{event_id}: claim mismatch across artifacts")
        labels = (
            str(build_row.get("gold_label") or build_row.get("label") or "").strip(),
            str(evi_row.get("gold_label") or "").strip(),
            str(s4_row.get("gold_label") or s4_row.get("label") or "").strip(),
        )
        if len(set(labels)) != 1 or labels[0] not in LIAR6_LABELS:
            raise ExportError(f"{event_id}: label mismatch/invalid label: {labels}")

        raw_evi_pool = list(evi_row.get("candidate_pool") or [])
        raw_s4_pool = list(s4_row.get("candidate_pool") or [])
        if not raw_evi_pool or not raw_s4_pool:
            raise ExportError(f"{event_id}: empty candidate pool")
        evi_views = [_candidate_view(event_id, candidate) for candidate in raw_evi_pool]
        evi_uids = [candidate.uid for candidate in evi_views]
        evi_keys = [candidate.key for candidate in evi_views]
        s4_keys = [_candidate_key(candidate) for candidate in raw_s4_pool]
        if any(not value for value in evi_uids + evi_keys + s4_keys):
            raise ExportError(f"{event_id}: empty candidate identity")
        if len(evi_uids) != len(set(evi_uids)):
            raise ExportError(f"{event_id}: duplicate EviTrace candidate_uid")
        if len(evi_keys) != len(set(evi_keys)):
            raise ExportError(f"{event_id}: duplicate EviTrace canonical candidate")
        if len(s4_keys) != len(set(s4_keys)):
            raise ExportError(f"{event_id}: duplicate S4 canonical candidate")
        if not set(evi_keys).issubset(set(s4_keys)):
            raise ExportError(f"{event_id}: EviTrace pool is not a subset of S4 pool")
        if len(s4_keys) > len(evi_keys):
            s4_pool_extra_events += 1

        recomputed = recompute_s4_source_score_order(raw_s4_pool)
        recomputed_keys = [_candidate_key(candidate) for candidate in recomputed]
        stored_s4 = list(s4_row.get("selected_candidates") or [])
        stored_keys = [_candidate_key(candidate) for candidate in stored_s4]
        if recomputed_keys != stored_keys:
            s4_order_differences += 1
            raise ExportError(
                f"{event_id}: stored S4 order differs from "
                "rank_atom_union_source_score_candidates()"
            )
        if current_s4_ranker is not None:
            ranker, params_type = current_s4_ranker
            current_ranked = ranker(
                raw_s4_pool,
                params=params_type(selector_top_k=len(raw_s4_pool)),
            )
            current_keys = [_candidate_key(candidate) for candidate in current_ranked]
            if current_keys != stored_keys:
                raise ExportError(
                    f"{event_id}: stored S4 order differs from the current "
                    "rank_atom_union_source_score_candidates() implementation"
                )
        for expected, stored in zip(recomputed, stored_s4):
            stored_score = stored.get("atom_union_source_score")
            if stored_score is None:
                raise ExportError(f"{event_id}: S4 candidate missing stored source score")
            if not math.isclose(
                float(expected["atom_union_source_score"]),
                float(stored_score),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ExportError(f"{event_id}: stored S4 source score differs")

        candidates_by_uid = {candidate.uid: candidate for candidate in evi_views}
        uid_by_key = {candidate.key: candidate.uid for candidate in evi_views}
        k_visible = int(build_row.get("evidence_count", -1))
        k_selected = int(build_row.get("evidence_count_before", -1))
        build_candidates = list(build_row.get("candidates") or [])
        if not (0 < k_visible <= k_selected <= len(raw_evi_pool)):
            raise ExportError(
                f"{event_id}: invalid K_visible/K_selected "
                f"({k_visible}/{k_selected})"
            )
        if len(build_candidates) != k_selected:
            raise ExportError(
                f"{event_id}: build candidates must represent K_selected "
                f"({len(build_candidates)} != {k_selected})"
            )
        build_visible_uids = tuple(
            str(candidate.get("candidate_uid") or "").strip()
            for candidate in build_candidates[:k_visible]
        )
        if any(not uid for uid in build_visible_uids):
            raise ExportError(f"{event_id}: build visible prefix missing candidate_uid")

        raw_steps = list(evi_row.get("mrec_steps") or [])
        if len(raw_steps) < k_visible:
            raise ExportError(f"{event_id}: trace shorter than K_visible")
        trace_visible_uids = tuple(
            str(step.get("candidate_uid") or "").strip()
            for step in raw_steps[:k_visible]
        )
        if build_visible_uids != trace_visible_uids:
            raise ExportError(
                f"{event_id}: build final K prefix differs from fullpool trace prefix"
            )
        if any(uid not in candidates_by_uid for uid in build_visible_uids):
            raise ExportError(f"{event_id}: visible candidate UID not in fullpool")

        if any(key not in uid_by_key for key in stored_keys[:k_visible]):
            raise ExportError(
                f"{event_id}: an actual S4 top-K candidate cannot be mapped "
                "to the EviTrace fullpool"
            )
        s4_order_uids = tuple(
            uid_by_key[key] for key in stored_keys if key in uid_by_key
        )
        if len(s4_order_uids) < k_visible:
            raise ExportError(f"{event_id}: S4 top-K cannot map to fullpool identities")
        s4_top_k = s4_order_uids[:k_visible]
        if len(s4_top_k) != k_visible:
            raise ExportError(f"{event_id}: S4 matched-count control is too short")

        evi_uid_set = set(build_visible_uids)
        s4_reordered = tuple(uid for uid in s4_order_uids if uid in evi_uid_set)
        if len(s4_reordered) != k_visible or set(s4_reordered) != evi_uid_set:
            raise ExportError(f"{event_id}: order-only set identity failure")

        atoms = tuple(
            dict(atom)
            for atom in (evi_row.get("claim_atoms") or [])
            if isinstance(atom, dict)
        )
        complexity = "single" if len(atoms) == 1 else "multi"
        visible_steps: list[dict[str, Any]] = []
        for position, raw_step in enumerate(raw_steps[:k_visible], start=1):
            step = dict(raw_step)
            uid = str(step.get("candidate_uid") or "")
            if uid not in candidates_by_uid:
                raise ExportError(f"{event_id}: transition candidate not in fullpool")
            step["step"] = int(step.get("step") or position)
            step["candidate_uid"] = uid
            step["atom_id"] = str(step.get("atom_id") or "")
            step["operation"] = str(step.get("operation") or "").upper()
            step["state_before"] = str(step.get("state_before") or "").upper()
            step["state_after"] = str(step.get("state_after") or "").upper()
            visible_steps.append(step)

        event = EventView(
            event_id=event_id,
            split=split,
            claim=claims[0],
            gold_label=labels[0],
            atoms=atoms,
            complexity=complexity,
            k_selected=k_selected,
            k_visible=k_visible,
            candidates_by_uid=candidates_by_uid,
            uid_by_key=uid_by_key,
            evi_visible_uids=build_visible_uids,
            s4_order_uids=s4_order_uids,
            s4_reordered_evi_uids=s4_reordered,
            visible_steps=tuple(visible_steps),
        )
        events.append(event)
        candidate_count += len(evi_views)
        s4_candidate_count += len(raw_s4_pool)
        selected_counts.append(k_selected)
        visible_counts.append(k_visible)
        same_order_count += int(event.order_is_identical)

    if expected_event_count is not None and len(events) != expected_event_count:
        raise ExportError(
            f"{split}: expected {expected_event_count} events, found {len(events)}"
        )
    audit = {
        "split": split,
        "event_count": len(events),
        "unique_event_count": len(seen),
        "candidate_identity_unique": True,
        "evitrace_candidate_count": candidate_count,
        "s4_candidate_count": s4_candidate_count,
        "s4_pool_extra_events": s4_pool_extra_events,
        "s4_order_difference_count": s4_order_differences,
        "s4_order_exact_match": s4_order_differences == 0,
        "compared_with_current_rank_function": current_s4_ranker is not None,
        "same_order_count": same_order_count,
        "k_selected_mean": _mean(selected_counts),
        "k_visible_mean": _mean(visible_counts),
        "k_selected_field": "evidence_count_before",
        "k_visible_field": "evidence_count",
    }
    return events, audit


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _seed_score(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


def _deterministic_sample(
    values: Sequence[Any],
    n: int,
    *,
    seed: int,
    namespace: str,
    identity,
) -> list[Any]:
    if n < 0 or n > len(values):
        raise ExportError(
            f"cannot sample {n} from pool of {len(values)} for {namespace}"
        )
    ordered = sorted(
        values,
        key=lambda value: (
            _seed_score(seed, namespace, str(identity(value))),
            str(identity(value)),
        ),
    )
    return ordered[:n]


def _largest_remainder(
    weights: Mapping[str, int | float],
    total: int,
    *,
    caps: Mapping[str, int] | None = None,
    order: Sequence[str] | None = None,
) -> dict[str, int]:
    keys = list(order or weights.keys())
    if total < 0:
        raise ExportError("allocation total cannot be negative")
    caps = dict(caps or {key: total for key in keys})
    if sum(max(0, int(caps.get(key, 0))) for key in keys) < total:
        raise ExportError("allocation caps cannot satisfy requested total")
    allocations = {key: 0 for key in keys}
    active = {key for key in keys if caps.get(key, 0) > 0}
    remaining = total
    while remaining and active:
        weight_sum = sum(max(0.0, float(weights.get(key, 0.0))) for key in active)
        if weight_sum <= 0:
            weight_sum = float(len(active))
            active_weights = {key: 1.0 for key in active}
        else:
            active_weights = {
                key: max(0.0, float(weights.get(key, 0.0))) for key in active
            }
        raw = {
            key: remaining * active_weights[key] / weight_sum for key in active
        }
        progress = 0
        for key in active:
            room = int(caps[key]) - allocations[key]
            add = min(room, int(math.floor(raw[key])))
            if add:
                allocations[key] += add
                remaining -= add
                progress += add
        if not remaining:
            break
        ranked = sorted(
            active,
            key=lambda key: (
                -(raw[key] - math.floor(raw[key])),
                keys.index(key),
            ),
        )
        for key in ranked:
            if remaining <= 0:
                break
            if allocations[key] < int(caps[key]):
                allocations[key] += 1
                remaining -= 1
                progress += 1
        active = {
            key for key in active if allocations[key] < int(caps.get(key, 0))
        }
        if progress == 0 and remaining:
            raise ExportError("largest-remainder allocation made no progress")
    if remaining:
        raise ExportError("largest-remainder allocation left unassigned units")
    return allocations


def _valid_transition_step(step: Mapping[str, Any]) -> bool:
    operation = str(step.get("operation") or "").upper()
    before = str(step.get("state_before") or "").upper()
    after = str(step.get("state_after") or "").upper()
    if operation not in TRANSITION_OPERATIONS:
        return False
    if before not in VALID_STATES or after not in VALID_STATES:
        return False
    if not str(step.get("atom_id") or ""):
        return False
    if operation in CHANGE_OPERATIONS:
        return before != after
    if operation in SELF_TRANSITION_OPERATIONS:
        return before == after
    return False


def sample_split(
    events: Sequence[EventView],
    *,
    n_main: int,
    n_order: int,
    n_transition: int,
    seed: int,
    namespace: str,
    excluded_event_ids: Iterable[str] = (),
) -> SplitSelection:
    excluded_ids = set(excluded_event_ids)
    parse_failed_ids = {event.event_id for event in events if not event.atoms}
    eligible = [
        event
        for event in events
        if event.event_id not in excluded_ids and event.event_id not in parse_failed_ids
    ]
    if len(eligible) < n_main + n_order + n_transition:
        raise ExportError(
            f"{namespace}: insufficient mutually exclusive claims for requested sample"
        )

    strata_order = [
        f"{label}|{complexity}"
        for label in LIAR6_LABELS
        for complexity in COMPLEXITIES
    ]
    strata_pools: dict[str, list[EventView]] = {}
    for stratum in strata_order:
        label, complexity = stratum.split("|", 1)
        strata_pools[stratum] = [
            event
            for event in eligible
            if event.gold_label == label and event.complexity == complexity
        ]
    main_alloc = _largest_remainder(
        {stratum: 1 for stratum in strata_order},
        n_main,
        caps={stratum: len(strata_pools[stratum]) for stratum in strata_order},
        order=strata_order,
    )
    main: list[EventView] = []
    main_strata_stats = []
    total_main_pool = sum(len(pool) for pool in strata_pools.values())
    for stratum in strata_order:
        selected = _deterministic_sample(
            strata_pools[stratum],
            main_alloc[stratum],
            seed=seed,
            namespace=f"{namespace}:main:{stratum}",
            identity=lambda event: event.event_id,
        )
        main.extend(selected)
        label, complexity = stratum.split("|", 1)
        pool_size = len(strata_pools[stratum])
        main_strata_stats.append(
            {
                "stratum": stratum,
                "label": label,
                "complexity": complexity,
                "pool_size": pool_size,
                "sampled_count": len(selected),
                "design_weight": (
                    float(pool_size / total_main_pool) if total_main_pool else 0.0
                ),
            }
        )
    main_ids = {event.event_id for event in main}

    same_order_all = sum(event.order_is_identical for event in events)
    same_order_eligible = sum(event.order_is_identical for event in eligible)
    order_pool = [
        event
        for event in eligible
        if event.event_id not in main_ids and not event.order_is_identical
    ]
    complexity_pools = {
        complexity: [
            event for event in order_pool if event.complexity == complexity
        ]
        for complexity in COMPLEXITIES
    }
    complexity_alloc = _largest_remainder(
        {complexity: 1 for complexity in COMPLEXITIES},
        n_order,
        caps={
            complexity: len(complexity_pools[complexity])
            for complexity in COMPLEXITIES
        },
        order=COMPLEXITIES,
    )
    order_only: list[EventView] = []
    order_alloc_stats: list[dict[str, Any]] = []
    for complexity in COMPLEXITIES:
        pool = complexity_pools[complexity]
        label_pools = {
            label: [event for event in pool if event.gold_label == label]
            for label in LIAR6_LABELS
        }
        label_alloc = _largest_remainder(
            {label: len(label_pools[label]) for label in LIAR6_LABELS},
            complexity_alloc[complexity],
            caps={label: len(label_pools[label]) for label in LIAR6_LABELS},
            order=LIAR6_LABELS,
        )
        for label in LIAR6_LABELS:
            selected = _deterministic_sample(
                label_pools[label],
                label_alloc[label],
                seed=seed,
                namespace=f"{namespace}:order:{complexity}:{label}",
                identity=lambda event: event.event_id,
            )
            order_only.extend(selected)
            order_alloc_stats.append(
                {
                    "complexity": complexity,
                    "label": label,
                    "pool_size": len(label_pools[label]),
                    "sampled_count": len(selected),
                }
            )
    order_ids = {event.event_id for event in order_only}

    remaining_events = [
        event
        for event in eligible
        if event.event_id not in main_ids and event.event_id not in order_ids
    ]
    transition_alloc = _largest_remainder(
        TRANSITION_WEIGHTS,
        n_transition,
        caps={
            operation: sum(
                any(
                    _valid_transition_step(step)
                    and step["operation"] == operation
                    for step in event.visible_steps
                )
                for event in remaining_events
            )
            for operation in TRANSITION_OPERATIONS
        },
        order=TRANSITION_OPERATIONS,
    )
    transition_pool_sizes = {}
    transitions_by_operation: dict[str, list[tuple[EventView, dict[str, Any]]]] = {
        operation: [] for operation in TRANSITION_OPERATIONS
    }
    used_transition_ids: set[str] = set()
    operation_processing_order = sorted(
        TRANSITION_OPERATIONS,
        key=lambda operation: (
            sum(
                any(
                    _valid_transition_step(step)
                    and step["operation"] == operation
                    for step in event.visible_steps
                )
                for event in remaining_events
            ),
            TRANSITION_OPERATIONS.index(operation),
        ),
    )
    for operation in operation_processing_order:
        per_event: list[tuple[EventView, dict[str, Any]]] = []
        for event in remaining_events:
            if event.event_id in used_transition_ids:
                continue
            steps = [
                dict(step)
                for step in event.visible_steps
                if _valid_transition_step(step) and step["operation"] == operation
            ]
            if not steps:
                continue
            chosen = _deterministic_sample(
                steps,
                1,
                seed=seed,
                namespace=f"{namespace}:transition-step:{operation}:{event.event_id}",
                identity=lambda step: f"{step['step']}|{step['candidate_uid']}",
            )[0]
            per_event.append((event, chosen))
        transition_pool_sizes[operation] = len(per_event)
        selected = _deterministic_sample(
            per_event,
            transition_alloc[operation],
            seed=seed,
            namespace=f"{namespace}:transition:{operation}",
            identity=lambda pair: f"{pair[0].event_id}|{pair[1]['step']}",
        )
        transitions_by_operation[operation].extend(selected)
        used_transition_ids.update(event.event_id for event, _ in selected)
    transitions = [
        pair
        for operation in TRANSITION_OPERATIONS
        for pair in transitions_by_operation[operation]
    ]

    if len(main) != n_main or len(order_only) != n_order or len(transitions) != n_transition:
        raise ExportError(f"{namespace}: sampled counts do not match requested counts")
    transition_ids = {event.event_id for event, _ in transitions}
    if (
        main_ids & order_ids
        or main_ids & transition_ids
        or order_ids & transition_ids
        or len(transition_ids) != len(transitions)
    ):
        raise ExportError(f"{namespace}: sample groups are not claim-disjoint")

    operation_counts = Counter(step["operation"] for _, step in transitions)
    for operation in TRANSITION_OPERATIONS:
        if operation_counts[operation] != transition_alloc[operation]:
            raise ExportError(f"{namespace}: transition operation quota mismatch")
    for _, step in transitions:
        is_change = step["state_before"] != step["state_after"]
        if step["operation"] in CHANGE_OPERATIONS and not is_change:
            raise ExportError("change operation sampled a self-transition")
        if step["operation"] in SELF_TRANSITION_OPERATIONS and is_change:
            raise ExportError("self operation sampled a state change")

    stats = {
        "namespace": namespace,
        "aligned_event_count": len(events),
        "eligible_event_count": len(eligible),
        "configured_excluded_event_count": len(
            {event.event_id for event in events} & excluded_ids
        ),
        "configured_excluded_event_ids": sorted(
            {event.event_id for event in events} & excluded_ids
        ),
        "atom_parse_failed_event_count": len(parse_failed_ids),
        "atom_parse_failed_event_ids": sorted(parse_failed_ids),
        "main": {
            "sampled_count": len(main),
            "strata": main_strata_stats,
        },
        "order_only": {
            "sampled_count": len(order_only),
            "complexity_counts": dict(
                sorted(Counter(event.complexity for event in order_only).items())
            ),
            "allocations": order_alloc_stats,
            "same_order_preexcluded_count_all_aligned": same_order_all,
            "same_order_preexcluded_count_after_event_exclusions": same_order_eligible,
        },
        "transition": {
            "sampled_count": len(transitions),
            "operation_quotas": {
                operation: transition_alloc[operation]
                for operation in TRANSITION_OPERATIONS
            },
            "operation_counts": {
                operation: operation_counts[operation]
                for operation in TRANSITION_OPERATIONS
            },
            "operation_eligible_claim_counts": {
                operation: transition_pool_sizes.get(operation, 0)
                for operation in TRANSITION_OPERATIONS
            },
            "change_operations": sorted(CHANGE_OPERATIONS),
            "self_transition_operations": sorted(SELF_TRANSITION_OPERATIONS),
            "one_step_per_claim": len(transition_ids) == len(transitions),
        },
        "claim_disjoint": True,
        "overlap_counts": {
            "main_order": len(main_ids & order_ids),
            "main_transition": len(main_ids & transition_ids),
            "order_transition": len(order_ids & transition_ids),
        },
    }
    return SplitSelection(
        main=tuple(main),
        order_only=tuple(order_only),
        transitions=tuple(transitions),
        stats=stats,
    )


def _sequence_candidates(
    event: EventView,
    comparison_type: str,
) -> tuple[list[CandidateView], list[CandidateView]]:
    evi = [event.candidates_by_uid[uid] for uid in event.evi_visible_uids]
    if comparison_type == "main":
        control_uids = event.s4_order_uids[: event.k_visible]
    elif comparison_type == "order_only":
        control_uids = event.s4_reordered_evi_uids
    else:
        raise ExportError(f"unknown preference comparison: {comparison_type}")
    control = [event.candidates_by_uid[uid] for uid in control_uids]
    if len(evi) != event.k_visible or len(control) != event.k_visible:
        raise ExportError(f"{event.event_id}: evidence-count matching failed")
    if comparison_type == "order_only":
        if {candidate.uid for candidate in evi} != {
            candidate.uid for candidate in control
        }:
            raise ExportError(f"{event.event_id}: order-only text set differs")
        if [candidate.uid for candidate in evi] == [
            candidate.uid for candidate in control
        ]:
            raise ExportError(f"{event.event_id}: identical order was not excluded")
    return evi, control


def _render_sequence(
    candidates: Sequence[CandidateView],
    translations: TranslationResolver,
) -> str:
    pieces = ['<ol class="evidence-sequence">']
    for candidate in candidates:
        source_id = html.escape(candidate.source_id, quote=True)
        domain = html.escape(candidate.domain, quote=True)
        evidence_en = html.escape(candidate.text, quote=True)
        evidence_zh = translations.resolve(candidate.text, "evidence")
        pieces.append("<li>")
        pieces.append(
            '<div class="evidence-source"><strong>'
            f"Source {source_id}</strong><span> · {domain}</span></div>"
        )
        pieces.append(f'<p class="evidence-en">{evidence_en}</p>')
        if evidence_zh:
            pieces.append(
                '<p class="evidence-zh"><strong>中文辅助：</strong>'
                f"{html.escape(evidence_zh, quote=True)}</p>"
            )
        pieces.append("</li>")
    pieces.append("</ol>")
    return "".join(pieces)


def _render_single_evidence(
    candidate: CandidateView,
    translations: TranslationResolver,
) -> str:
    return _render_sequence([candidate], translations)


STATE_LEGEND_HTML = (
    '<div class="state-legend">'
    "<span><strong>U</strong> · Unresolved / 未决</span><br>"
    "<span><strong>S</strong> · Supported / 支持</span><br>"
    "<span><strong>R</strong> · Refuted / 反驳</span><br>"
    "<span><strong>Q</strong> · Qualified or mixed / 限定或混合</span><br>"
    "<span><strong>C</strong> · Conflict / 冲突</span>"
    "</div>"
)


def _opaque_id(secret: bytes, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()[:20]


def _balanced_evi_sides(
    items: Sequence[tuple[str, str]],
    *,
    secret: bytes,
    namespace: str,
) -> dict[str, str]:
    ordered = sorted(
        items,
        key=lambda pair: (
            _opaque_id(secret, namespace, pair[0], pair[1]),
            pair,
        ),
    )
    n_a = len(ordered) // 2
    return {
        event_id: ("A" if index < n_a else "B")
        for index, (event_id, _) in enumerate(ordered)
    }


def _public_fingerprint(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _artifact_hashes_for_split(
    source_artifacts: Mapping[str, Mapping[str, Any]],
    split: str,
) -> dict[str, str]:
    prefix = f"{split}_"
    return {
        name[len(prefix) :]: str(metadata["sha256"])
        for name, metadata in source_artifacts.items()
        if name.startswith(prefix)
    }


def build_preference_tasks(
    formal: SplitSelection,
    pilot: SplitSelection,
    *,
    blind_secret: bytes,
    translations: TranslationResolver,
    token_counter: EvidenceTokenCounter,
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    specs: list[tuple[str, str, EventView]] = []
    for phase, selection in (("formal", formal), ("pilot", pilot)):
        specs.extend((phase, "main", event) for event in selection.main)
        specs.extend(
            (phase, "order_only", event) for event in selection.order_only
        )

    side_by_identity: dict[tuple[str, str], dict[str, str]] = {}
    for phase in ("formal", "pilot"):
        for comparison_type in ("main", "order_only"):
            group = [
                (event.event_id, comparison_type)
                for item_phase, item_type, event in specs
                if item_phase == phase and item_type == comparison_type
            ]
            mapping = _balanced_evi_sides(
                group,
                secret=blind_secret,
                namespace=f"side:{phase}:{comparison_type}",
            )
            side_by_identity[(phase, comparison_type)] = mapping

    public_by_phase: dict[str, list[dict[str, Any]]] = {
        "formal": [],
        "pilot": [],
    }
    private_rows: list[dict[str, Any]] = []
    seen_blind_ids: set[str] = set()
    for phase, comparison_type, event in specs:
        evi_candidates, control_candidates = _sequence_candidates(
            event, comparison_type
        )
        evi_side = side_by_identity[(phase, comparison_type)][event.event_id]
        control_side = "B" if evi_side == "A" else "A"
        evi_html = _render_sequence(evi_candidates, translations)
        control_html = _render_sequence(control_candidates, translations)
        blind_task_id = "P-" + _opaque_id(
            blind_secret,
            "preference",
            phase,
            comparison_type,
            event.event_id,
        )
        if blind_task_id in seen_blind_ids:
            raise ExportError("preference blind_task_id collision")
        seen_blind_ids.add(blind_task_id)
        public = {
            "blind_task_id": blind_task_id,
            "claim_en": event.claim,
            "claim_zh": translations.resolve(event.claim, "claim"),
            "sequence_a_html": evi_html if evi_side == "A" else control_html,
            "sequence_b_html": control_html if evi_side == "A" else evi_html,
        }
        _validate_preference_public(public)
        public_by_phase[phase].append(public)

        evi_tokens = sum(token_counter.count(candidate) for candidate in evi_candidates)
        control_tokens = sum(
            token_counter.count(candidate) for candidate in control_candidates
        )
        if comparison_type == "order_only" and evi_tokens != control_tokens:
            raise ExportError(
                f"{event.event_id}: order-only token totals must be identical"
            )
        method_names = (
            {
                "evitrace": "evitrace_visible_selection",
                "control": "s4_source_score_top_k",
            }
            if comparison_type == "main"
            else {
                "evitrace": "evitrace_order",
                "control": "s4_source_score_reorder_same_set",
            }
        )
        candidate_uids = {
            "evitrace": [candidate.uid for candidate in evi_candidates],
            "control": [candidate.uid for candidate in control_candidates],
        }
        method_to_side = {"evitrace": evi_side, "control": control_side}
        private_rows.append(
            {
                "blind_task_id": blind_task_id,
                "phase": phase,
                "task_type": "preference",
                "comparison_type": comparison_type,
                "event_id": event.event_id,
                "claim_cluster": event.event_id,
                "split": event.split,
                "gold_label": event.gold_label,
                "complexity": event.complexity,
                "atom_count": len(event.atoms),
                "stratum": (
                    f"{event.gold_label}|{event.complexity}"
                    if comparison_type == "main"
                    else f"{event.complexity}|{event.gold_label}"
                ),
                "method_names": method_names,
                "method_to_side": method_to_side,
                "side_a_method": (
                    "evitrace" if evi_side == "A" else "control"
                ),
                "side_b_method": (
                    "control" if evi_side == "A" else "evitrace"
                ),
                "candidate_uids": candidate_uids,
                "candidate_uids_by_side": {
                    evi_side: candidate_uids["evitrace"],
                    control_side: candidate_uids["control"],
                },
                "evi_candidate_uids": candidate_uids["evitrace"],
                "control_candidate_uids": candidate_uids["control"],
                "k_i": event.k_visible,
                "k_visible": event.k_visible,
                "k_selected": event.k_selected,
                "evi_token_count": evi_tokens,
                "control_token_count": control_tokens,
                "token_count_difference_evi_minus_control": (
                    evi_tokens - control_tokens
                ),
                "token_counts_by_side": {
                    evi_side: evi_tokens,
                    control_side: control_tokens,
                },
                "same_evidence_set": comparison_type == "order_only",
                "artifact_sha256": _artifact_hashes_for_split(
                    source_artifacts, event.split
                ),
                "public_task_sha256": _public_fingerprint(public),
            }
        )

    for _phase, rows in public_by_phase.items():
        rows.sort(key=lambda row: row["blind_task_id"])
    private_rows.sort(key=lambda row: row["blind_task_id"])
    side_balance = {}
    for phase in ("formal", "pilot"):
        side_balance[phase] = {}
        for comparison_type in ("main", "order_only"):
            group = [
                row
                for row in private_rows
                if row["phase"] == phase
                and row["comparison_type"] == comparison_type
            ]
            side_balance[phase][comparison_type] = {
                "evitrace_on_a": sum(
                    row["method_to_side"]["evitrace"] == "A" for row in group
                ),
                "evitrace_on_b": sum(
                    row["method_to_side"]["evitrace"] == "B" for row in group
                ),
            }
    return (
        public_by_phase["formal"],
        public_by_phase["pilot"],
        private_rows,
        side_balance,
    )


def _atom_text(event: EventView, atom_id: str) -> str:
    for atom in event.atoms:
        if str(atom.get("atom_id") or "") == atom_id:
            return str(
                atom.get("proposition") or atom.get("text") or ""
            ).strip()
    return ""


def build_transition_tasks(
    formal: SplitSelection,
    pilot: SplitSelection,
    *,
    blind_secret: bytes,
    translations: TranslationResolver,
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    public_by_phase: dict[str, list[dict[str, Any]]] = {
        "formal": [],
        "pilot": [],
    }
    private_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for phase, selection in (("formal", formal), ("pilot", pilot)):
        for event, step in selection.transitions:
            operation = str(step["operation"])
            before = str(step["state_before"])
            after = str(step["state_after"])
            is_change = before != after
            if operation in CHANGE_OPERATIONS and not is_change:
                raise ExportError("invalid change transition")
            if operation in SELF_TRANSITION_OPERATIONS and is_change:
                raise ExportError("invalid self-transition")
            atom_id = str(step["atom_id"])
            focal_atom = _atom_text(event, atom_id)
            if not focal_atom:
                raise ExportError(
                    f"{event.event_id}: focal atom {atom_id!r} is unavailable"
                )
            current_uid = str(step["candidate_uid"])
            current = event.candidates_by_uid[current_uid]
            previous_steps = [
                earlier
                for earlier in event.visible_steps
                if int(earlier["step"]) < int(step["step"])
                and str(earlier["atom_id"]) == atom_id
            ]
            prior_candidates = [
                event.candidates_by_uid[str(earlier["candidate_uid"])]
                for earlier in previous_steps
            ]
            prior_html = (
                _render_sequence(prior_candidates, translations)
                if prior_candidates
                else (
                    '<p class="empty-prefix">'
                    "No earlier evidence for this atom. / 该原子此前无证据。"
                    "</p>"
                )
            )
            blind_task_id = "T-" + _opaque_id(
                blind_secret,
                "transition",
                phase,
                event.event_id,
                str(step["step"]),
            )
            if blind_task_id in seen_ids:
                raise ExportError("transition blind_task_id collision")
            seen_ids.add(blind_task_id)
            public = {
                "blind_task_id": blind_task_id,
                "claim_en": event.claim,
                "claim_zh": translations.resolve(event.claim, "claim"),
                "focal_atom_en": focal_atom,
                "focal_atom_zh": translations.resolve(focal_atom, "atom"),
                "state_legend_html": STATE_LEGEND_HTML,
                "prior_evidence_html": prior_html,
                "current_evidence_html": _render_single_evidence(
                    current, translations
                ),
                "proposed_transition": f"{before} → {after}",
            }
            _validate_transition_public(public)
            public_by_phase[phase].append(public)
            private_rows.append(
                {
                    "blind_task_id": blind_task_id,
                    "phase": phase,
                    "task_type": "transition",
                    "comparison_type": "transition",
                    "event_id": event.event_id,
                    "claim_cluster": event.event_id,
                    "split": event.split,
                    "gold_label": event.gold_label,
                    "complexity": event.complexity,
                    "atom_count": len(event.atoms),
                    "operation": operation,
                    "atom_id": atom_id,
                    "state_before": before,
                    "state_after": after,
                    "transition_kind": "change" if is_change else "self",
                    "is_state_change": is_change,
                    "step": int(step["step"]),
                    "candidate_uid": current_uid,
                    "prior_same_atom_candidate_uids": [
                        candidate.uid for candidate in prior_candidates
                    ],
                    "hidden_relation": step.get("relation"),
                    "hidden_directness": step.get("directness"),
                    "hidden_confidence": step.get("map_confidence"),
                    "artifact_sha256": _artifact_hashes_for_split(
                        source_artifacts, event.split
                    ),
                    "public_task_sha256": _public_fingerprint(public),
                }
            )

    for _phase, rows in public_by_phase.items():
        rows.sort(key=lambda row: row["blind_task_id"])
    private_rows.sort(key=lambda row: row["blind_task_id"])
    return (
        public_by_phase["formal"],
        public_by_phase["pilot"],
        private_rows,
    )


def _validate_preference_public(row: Mapping[str, Any]) -> None:
    if frozenset(row) != PREFERENCE_PUBLIC_FIELDS:
        raise ExportError("preference public schema contains missing/forbidden fields")
    combined_html = f"{row['sequence_a_html']}\n{row['sequence_b_html']}"
    if "Check:" in combined_html:
        raise ExportError("preference HTML leaked a Check: cue")
    lowered = combined_html.lower()
    if any(marker in lowered for marker in _FORBIDDEN_PUBLIC_MARKERS):
        raise ExportError("preference HTML leaked a method-specific marker")
    if "<script" in lowered or "javascript:" in lowered:
        raise ExportError("unsafe preference HTML")


def _validate_transition_public(row: Mapping[str, Any]) -> None:
    if frozenset(row) != TRANSITION_PUBLIC_FIELDS:
        raise ExportError("transition public schema contains missing/forbidden fields")
    combined = "\n".join(str(value) for value in row.values()).lower()
    for marker in (
        "gold_label",
        "candidate_uid",
        "mrec_",
        "evitrace",
        "source_score",
        "map_relation",
        "map_directness",
        "map_confidence",
    ):
        if marker in combined:
            raise ExportError("transition task leaked a hidden field")
    if "<script" in combined or "javascript:" in combined:
        raise ExportError("unsafe transition HTML")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_for_manifest(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_artifact_metadata(
    paths: Mapping[str, Path],
    row_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": _path_for_manifest(path),
            "sha256": sha256_file(path),
            "rows": int(row_counts[name.split("_", 1)[0]]),
        }
        for name, path in paths.items()
    }


def audit_test_atom_parse_status(
    path: Path,
    *,
    aligned_event_ids: set[str],
) -> dict[str, Any]:
    """Assert the sole canonical test parse failure is ``7845.json``."""

    rows = list(_jsonl_rows(path))
    event_ids = [str(row.get("event_id") or "") for row in rows]
    if any(not event_id for event_id in event_ids):
        raise ExportError(f"{path}: atom-cache row missing event_id")
    if len(event_ids) != len(set(event_ids)):
        raise ExportError(f"{path}: duplicate atom-cache event_id")
    if set(event_ids) != aligned_event_ids:
        raise ExportError(
            f"{path}: atom-cache event universe differs from the aligned test artifacts"
        )
    failed_ids = {
        str(row["event_id"])
        for row in rows
        if str(row.get("parse_status") or "") != "ok"
    }
    if failed_ids != set(EXCLUDED_TEST_EVENT_IDS):
        raise ExportError(
            "canonical atom parse-failed IDs must be exactly "
            f"{sorted(EXCLUDED_TEST_EVENT_IDS)}, found {sorted(failed_ids)}"
        )
    return {
        "path": _path_for_manifest(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "parse_failed_event_ids": sorted(failed_ids),
        "exact_expected_failure_set": True,
    }


def _infer_tokenizer_path(build_path: Path) -> Path:
    report_path = build_path.with_name("build_report.json")
    if not report_path.exists():
        raise ExportError(
            "cannot infer tokenizer path: pass --tokenizer-path or use "
            "--token-counter artifact"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    value = report.get("prompt_model_name_or_path")
    if not value:
        raise ExportError(f"{report_path}: prompt_model_name_or_path is missing")
    path = Path(str(value))
    if not path.exists():
        raise ExportError(f"inferred tokenizer path does not exist: {path}")
    return path


def export_experiment(
    *,
    build_test_path: Path,
    evitrace_test_path: Path,
    s4_test_path: Path,
    build_val_path: Path,
    evitrace_val_path: Path,
    s4_val_path: Path,
    output_dir: Path,
    blind_secret: bytes,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    n_main: int = DEFAULT_N_MAIN,
    n_order: int = DEFAULT_N_ORDER,
    n_transition: int = DEFAULT_N_TRANSITION,
    n_pilot_main: int = DEFAULT_N_PILOT_MAIN,
    n_pilot_order: int = DEFAULT_N_PILOT_ORDER,
    n_pilot_transition: int = DEFAULT_N_PILOT_TRANSITION,
    translation_cache: TranslationCache | None = None,
    translation_cache_path: Path | None = None,
    token_counter: EvidenceTokenCounter | None = None,
    expected_test_events: int | None = EXPECTED_TEST_EVENTS,
    expected_val_events: int | None = EXPECTED_VAL_EVENTS,
    test_atom_parse_path: Path | None = DEFAULT_ATOM_PARSE_TEST,
    preference_config_path: Path = DEFAULT_PREFERENCE_CONFIG,
    transition_config_path: Path = DEFAULT_TRANSITION_CONFIG,
) -> dict[str, Any]:
    if not blind_secret:
        raise ExportError("blind seed file must contain non-empty secret bytes")
    for name, value in {
        "n_main": n_main,
        "n_order": n_order,
        "n_transition": n_transition,
        "n_pilot_main": n_pilot_main,
        "n_pilot_order": n_pilot_order,
        "n_pilot_transition": n_pilot_transition,
    }.items():
        if value < 0:
            raise ExportError(f"{name} cannot be negative")

    current_s4_ranker = (
        _load_current_s4_ranker()
        if expected_test_events == EXPECTED_TEST_EVENTS
        else None
    )
    test_events, test_audit = load_aligned_events(
        build_test_path,
        evitrace_test_path,
        s4_test_path,
        split="test",
        expected_event_count=expected_test_events,
        current_s4_ranker=current_s4_ranker,
    )
    val_events, val_audit = load_aligned_events(
        build_val_path,
        evitrace_val_path,
        s4_val_path,
        split="val",
        expected_event_count=expected_val_events,
    )
    if expected_test_events == EXPECTED_TEST_EVENTS and len(test_events) != 1251:
        raise ExportError("canonical test export requires exactly 1,251 events")
    atom_parse_audit: dict[str, Any] | None = None
    if expected_test_events == EXPECTED_TEST_EVENTS:
        if test_atom_parse_path is None:
            raise ExportError(
                "canonical export requires the claim-atom parse-status artifact"
            )
        atom_parse_audit = audit_test_atom_parse_status(
            test_atom_parse_path,
            aligned_event_ids={event.event_id for event in test_events},
        )
    for config_path in (preference_config_path, transition_config_path):
        if not config_path.exists():
            raise ExportError(f"Label Studio config does not exist: {config_path}")
    config_sha256 = {
        "preference": sha256_file(preference_config_path),
        "transition": sha256_file(transition_config_path),
    }

    formal = sample_split(
        test_events,
        n_main=n_main,
        n_order=n_order,
        n_transition=n_transition,
        seed=sample_seed,
        namespace="formal-test",
        excluded_event_ids=EXCLUDED_TEST_EVENT_IDS,
    )
    pilot = sample_split(
        val_events,
        n_main=n_pilot_main,
        n_order=n_pilot_order,
        n_transition=n_pilot_transition,
        seed=sample_seed,
        namespace="pilot-val",
    )

    source_paths = {
        "test_build": build_test_path,
        "test_evitrace": evitrace_test_path,
        "test_s4": s4_test_path,
        "val_build": build_val_path,
        "val_evitrace": evitrace_val_path,
        "val_s4": s4_val_path,
    }
    source_artifacts = _source_artifact_metadata(
        source_paths,
        {"test": len(test_events), "val": len(val_events)},
    )
    resolved_translation_cache = translation_cache or TranslationCache()
    translations = TranslationResolver(resolved_translation_cache)
    counter = token_counter or ArtifactTokenCounter()

    (
        preference_tasks,
        pilot_preference_tasks,
        blinding_key,
        side_balance,
    ) = build_preference_tasks(
        formal,
        pilot,
        blind_secret=blind_secret,
        translations=translations,
        token_counter=counter,
        source_artifacts=source_artifacts,
    )
    (
        transition_tasks,
        pilot_transition_tasks,
        transition_key,
    ) = build_transition_tasks(
        formal,
        pilot,
        blind_secret=blind_secret,
        translations=translations,
        source_artifacts=source_artifacts,
    )

    expected_counts = {
        "formal_preference": n_main + n_order,
        "formal_transition": n_transition,
        "pilot_preference": n_pilot_main + n_pilot_order,
        "pilot_transition": n_pilot_transition,
    }
    actual_counts = {
        "formal_preference": len(preference_tasks),
        "formal_transition": len(transition_tasks),
        "pilot_preference": len(pilot_preference_tasks),
        "pilot_transition": len(pilot_transition_tasks),
    }
    if actual_counts != expected_counts:
        raise ExportError(
            f"task counts differ from requested counts: {actual_counts} != {expected_counts}"
        )
    if n_main == 120:
        sampled_strata = {
            row["stratum"]: row["sampled_count"]
            for row in formal.stats["main"]["strata"]
        }
        if set(sampled_strata.values()) != {10}:
            raise ExportError("formal 120-main sample must have 10 tasks per stratum")
    if n_order == 80 and formal.stats["order_only"]["complexity_counts"] != {
        "multi": 40,
        "single": 40,
    }:
        raise ExportError("formal order-only sample must be 40 single / 40 multi")
    if n_transition == 100 and formal.stats["transition"]["operation_counts"] != {
        "OPEN": 40,
        "CONTRAST": 20,
        "BRIDGE": 20,
        "CORROBORATE": 10,
        "FALLBACK": 10,
    }:
        raise ExportError("formal transition quotas differ from preregistration")
    if n_main == 120 and side_balance["formal"]["main"] != {
        "evitrace_on_a": 60,
        "evitrace_on_b": 60,
    }:
        raise ExportError("main A/B allocation is not exactly 60/60")
    if n_order == 80 and side_balance["formal"]["order_only"] != {
        "evitrace_on_a": 40,
        "evitrace_on_b": 40,
    }:
        raise ExportError("order-only A/B allocation is not exactly 40/40")

    sampling_stats = {
        "schema_version": "evitrace_human_alignment_sampling_v1",
        "sample_seed": sample_seed,
        "formal": formal.stats,
        "pilot": pilot.stats,
        "pilot_excluded_from_paper_statistics": True,
        "k_summary": {
            "test_k_selected_mean": test_audit["k_selected_mean"],
            "test_k_visible_mean": test_audit["k_visible_mean"],
            "k_selected_field": "evidence_count_before",
            "k_visible_field": "evidence_count",
        },
    }
    translation_inventory = translations.inventory_rows()
    translation_requests = translations.request_rows()
    translation_complete = len(translation_requests) == 0
    if translation_cache_path is not None:
        if not translation_cache_path.is_file():
            raise ExportError(
                f"translation cache file does not exist: {translation_cache_path}"
            )
        cache_file_sha256 = sha256_file(translation_cache_path)
        if (
            resolved_translation_cache.source_file_sha256 is not None
            and cache_file_sha256
            != resolved_translation_cache.source_file_sha256
        ):
            raise ExportError("translation cache changed after it was loaded")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "preference_tasks": output_dir / "preference_tasks.jsonl",
        "transition_tasks": output_dir / "transition_tasks.jsonl",
        "pilot_preference_tasks": output_dir / "pilot_preference_tasks.jsonl",
        "pilot_transition_tasks": output_dir / "pilot_transition_tasks.jsonl",
        "blinding_key": output_dir / "private/blinding_key.jsonl",
        "transition_key": output_dir / "private/transition_key.jsonl",
        "translation_inventory": output_dir / "translation_inventory.jsonl",
        "translation_requests": output_dir / "translation_requests.jsonl",
        "sampling_stats": output_dir / "sampling_stats.json",
    }
    rows_by_name: dict[str, Sequence[Mapping[str, Any]]] = {
        "preference_tasks": preference_tasks,
        "transition_tasks": transition_tasks,
        "pilot_preference_tasks": pilot_preference_tasks,
        "pilot_transition_tasks": pilot_transition_tasks,
        "blinding_key": blinding_key,
        "transition_key": transition_key,
        "translation_inventory": translation_inventory,
        "translation_requests": translation_requests,
    }
    for name, rows in rows_by_name.items():
        _write_jsonl(output_paths[name], rows)
    _write_json(output_paths["sampling_stats"], sampling_stats)

    artifact_entries: dict[str, dict[str, Any]] = {}
    outputs_by_path: dict[str, dict[str, Any]] = {}
    for name, path in output_paths.items():
        relative_path = str(path.relative_to(output_dir))
        rows = (
            len(rows_by_name[name])
            if name in rows_by_name
            else None
        )
        entry = {
            "path": relative_path,
            "sha256": sha256_file(path),
            "visibility": "private" if relative_path.startswith("private/") else "public",
        }
        if rows is not None:
            entry["rows"] = rows
        artifact_entries[name] = entry
        outputs_by_path[relative_path] = {
            key: value for key, value in entry.items() if key != "path"
        }
    if translation_cache_path is not None:
        cache_manifest_path = _path_for_manifest(translation_cache_path)
        cache_entry = {
            "path": cache_manifest_path,
            "sha256": sha256_file(translation_cache_path),
            "visibility": "public",
            "rows": resolved_translation_cache.entry_count,
        }
        artifact_entries["translation_cache"] = cache_entry
        outputs_by_path[cache_manifest_path] = {
            key: value for key, value in cache_entry.items() if key != "path"
        }

    manifest = {
        "schema_version": "evitrace_human_alignment_task_manifest_v1",
        "complete": translation_complete,
        "annotation_complete": False,
        "status": (
            "prepared_and_frozen"
            if translation_complete
            else "translation_cache_incomplete"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_seed": sample_seed,
        "requested_counts": {
            "n_main": n_main,
            "n_order": n_order,
            "n_transition": n_transition,
            "n_pilot_main": n_pilot_main,
            "n_pilot_order": n_pilot_order,
            "n_pilot_transition": n_pilot_transition,
        },
        "counts": {
            **actual_counts,
            "main": n_main,
            "order_only": n_order,
            "transition": n_transition,
            "blinding_key": len(blinding_key),
            "transition_key": len(transition_key),
            "translation_requests": len(translation_requests),
            "translation_inventory": len(translation_inventory),
        },
        "source_artifacts": source_artifacts,
        "artifact_sha256": {
            name: metadata["sha256"]
            for name, metadata in source_artifacts.items()
        },
        "config_sha256": config_sha256,
        "config_paths": {
            "preference": _path_for_manifest(preference_config_path),
            "transition": _path_for_manifest(transition_config_path),
        },
        "artifacts": artifact_entries,
        "outputs": outputs_by_path,
        "private_key_commitments": {
            "blinding_key_sha256": artifact_entries["blinding_key"]["sha256"],
            "transition_key_sha256": artifact_entries["transition_key"]["sha256"],
            "blind_seed_sha256": hashlib.sha256(blind_secret).hexdigest(),
        },
        "sampling": sampling_stats,
        "side_balance": side_balance,
        "token_count": counter.metadata(),
        "translation_cache": {
            "path": (
                _path_for_manifest(translation_cache_path)
                if translation_cache_path is not None
                else None
            ),
            "content_sha256": resolved_translation_cache.content_sha256,
            "source_file_sha256": resolved_translation_cache.source_file_sha256,
            "entry_count": resolved_translation_cache.entry_count,
            "miss_count": len(translation_requests),
            "complete": translation_complete,
            "network_used": False,
        },
        "artifact_audit": {
            "test": test_audit,
            "val": val_audit,
            "test_atom_parse": atom_parse_audit,
            "canonical_test_event_alignment_1251": (
                len(test_events) == EXPECTED_TEST_EVENTS
            ),
            "excluded_test_event_ids": sorted(EXCLUDED_TEST_EVENT_IDS),
        },
        "blinding": {
            "system_blind": True,
            "double_annotated": True,
            "annotator_order_keys": ["annotator_1", "annotator_2"],
            "method_side_mapping_public": False,
            "private_key_paths": {
                "preference": artifact_entries["blinding_key"]["path"],
                "transition": artifact_entries["transition_key"]["path"],
            },
        },
        "public_schema": {
            "preference": sorted(PREFERENCE_PUBLIC_FIELDS),
            "transition": sorted(TRANSITION_PUBLIC_FIELDS),
        },
        "leakage_checks": {
            "preference_public_exact_field_whitelist": True,
            "transition_public_exact_field_whitelist": True,
            "html_escaped": True,
            "check_cue_absent": True,
            "method_specific_projection_absent": True,
            "translation_cache_reused_by_exact_english_text": True,
        },
        "analysis_boundaries": {
            "matched_evidence_count_only": True,
            "matched_token_budget_claimed": False,
            "order_only_same_text_and_length": True,
            "earliest_sufficient_prefix_collected": False,
            "pilot_included_in_paper_statistics": False,
        },
    }
    manifest_path = output_dir / "task_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parse_expected_count(value: int) -> int | None:
    return None if value <= 0 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export blinded EviTrace preference and transition-audit tasks"
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--n-main", type=int, default=DEFAULT_N_MAIN)
    parser.add_argument("--n-order", type=int, default=DEFAULT_N_ORDER)
    parser.add_argument("--n-transition", type=int, default=DEFAULT_N_TRANSITION)
    parser.add_argument("--n-pilot-main", type=int, default=DEFAULT_N_PILOT_MAIN)
    parser.add_argument("--n-pilot-order", type=int, default=DEFAULT_N_PILOT_ORDER)
    parser.add_argument(
        "--n-pilot-transition", type=int, default=DEFAULT_N_PILOT_TRANSITION
    )
    parser.add_argument("--blind-seed-file", type=Path, required=True)
    parser.add_argument(
        "--build-artifact",
        "--build-path",
        dest="build_test_path",
        type=Path,
        default=DEFAULT_BUILD_TEST,
    )
    parser.add_argument(
        "--evitrace-artifact",
        "--evitrace-path",
        dest="evitrace_test_path",
        type=Path,
        default=DEFAULT_EVITRACE_TEST,
    )
    parser.add_argument(
        "--s4-artifact",
        "--s4-path",
        dest="s4_test_path",
        type=Path,
        default=DEFAULT_S4_TEST,
    )
    parser.add_argument(
        "--pilot-build-artifact",
        dest="build_val_path",
        type=Path,
        default=DEFAULT_BUILD_VAL,
    )
    parser.add_argument(
        "--pilot-evitrace-artifact",
        dest="evitrace_val_path",
        type=Path,
        default=DEFAULT_EVITRACE_VAL,
    )
    parser.add_argument(
        "--pilot-s4-artifact",
        dest="s4_val_path",
        type=Path,
        default=DEFAULT_S4_VAL,
    )
    parser.add_argument(
        "--test-atom-parse-artifact",
        type=Path,
        default=DEFAULT_ATOM_PARSE_TEST,
    )
    parser.add_argument(
        "--preference-config",
        type=Path,
        default=DEFAULT_PREFERENCE_CONFIG,
    )
    parser.add_argument(
        "--transition-config",
        type=Path,
        default=DEFAULT_TRANSITION_CONFIG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--translation-cache", type=Path)
    parser.add_argument(
        "--token-counter",
        choices=("tokenizer", "artifact"),
        default="tokenizer",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--expected-test-events", type=int, default=EXPECTED_TEST_EVENTS
    )
    parser.add_argument(
        "--expected-val-events", type=int, default=EXPECTED_VAL_EVENTS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.blind_seed_file.exists():
        raise ExportError(f"blind seed file does not exist: {args.blind_seed_file}")
    blind_secret = args.blind_seed_file.read_bytes().strip()
    cache = TranslationCache.load(args.translation_cache)
    if args.token_counter == "tokenizer":
        tokenizer_path = args.tokenizer_path or _infer_tokenizer_path(
            args.build_test_path
        )
        counter: EvidenceTokenCounter = TokenizerTokenCounter(tokenizer_path)
    else:
        counter = ArtifactTokenCounter()
    manifest = export_experiment(
        build_test_path=args.build_test_path,
        evitrace_test_path=args.evitrace_test_path,
        s4_test_path=args.s4_test_path,
        build_val_path=args.build_val_path,
        evitrace_val_path=args.evitrace_val_path,
        s4_val_path=args.s4_val_path,
        output_dir=args.output_dir,
        blind_secret=blind_secret,
        sample_seed=args.sample_seed,
        n_main=args.n_main,
        n_order=args.n_order,
        n_transition=args.n_transition,
        n_pilot_main=args.n_pilot_main,
        n_pilot_order=args.n_pilot_order,
        n_pilot_transition=args.n_pilot_transition,
        translation_cache=cache,
        translation_cache_path=args.translation_cache,
        token_counter=counter,
        expected_test_events=_parse_expected_count(args.expected_test_events),
        expected_val_events=_parse_expected_count(args.expected_val_events),
        test_atom_parse_path=args.test_atom_parse_artifact,
        preference_config_path=args.preference_config,
        transition_config_path=args.transition_config,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "counts": manifest["counts"],
                "manifest": str((args.output_dir / "task_manifest.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
