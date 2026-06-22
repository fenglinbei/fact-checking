from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fact_checking.selectors.stage2_oracle import write_json, write_jsonl


CLAIM_ATOM_PROMPT_VERSION = "claim_atomization_v0_1"
CLAIM_ATOM_SCHEMA_VERSION = "claim_atoms_v1"
MIN_ATOMS = 1
MAX_ATOMS = 6

SYSTEM_PROMPT = """You create canonical atomic verification units for fact-checking.
Use only the given claim. Do not use outside knowledge.
Return strictly valid JSON."""

USER_PROMPT_TEMPLATE = """Claim:
{claim}

Decompose the claim into 1 to 6 atomic verification units.

Rules:
- Each atom must be a complete proposition with clear truth conditions.
- Do not create standalone entity, date, number, or topic fragments.
- Keep dates, quantities, negation, modality, attribution, comparison targets, offices, locations, and scope inside the proposition.
- Split only when the claim contains multiple separately verifiable propositions.
- A single-sentence claim making one factual assertion should usually have one atom.
- Preserve the claim's meaning; do not add facts not present in the claim.
- The query_rendering is only a retrieval view of the same atom, not an independent question.

Return JSON exactly in this schema:
{{
  "complexity": "simple|moderate|complex",
  "claim_atoms": [
    {{
      "atom_id": "A1",
      "proposition": "...",
      "importance": 1.0,
      "type": "entity|quantity|date|comparison|causal|attribution|policy|other",
      "keywords": ["..."],
      "query_rendering": "...?"
    }}
  ]
}}
"""

ALLOWED_COMPLEXITIES = {"simple", "moderate", "complex"}
ALLOWED_TYPES = {"entity", "quantity", "date", "comparison", "causal", "attribution", "policy", "other"}


@dataclass(frozen=True)
class AtomInputExample:
    event_id: str
    claim: str
    gold_label: str = ""


@dataclass(frozen=True)
class AtomGenerationSettings:
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: int = 20260526
    guided_json: bool = True
    thinking_type: str | None = "disabled"
    prompt_version: str = CLAIM_ATOM_PROMPT_VERSION
    schema_version: str = CLAIM_ATOM_SCHEMA_VERSION
    min_atoms: int = MIN_ATOMS
    max_atoms: int = MAX_ATOMS


@dataclass(frozen=True)
class AtomCachePaths:
    atom_cache_path: Path
    raw_cache_path: Path
    cache_manifest_path: Path
    lock_path: Path


@dataclass
class AtomCacheIndex:
    rows_by_key: dict[str, dict[str, Any]]
    invalid_lines: int = 0
    duplicate_rows: int = 0
    total_valid_rows: int = 0


@dataclass
class AtomGenerationResult:
    rows: list[dict[str, Any]]
    raw_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


class AtomAPITransientError(RuntimeError):
    pass


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.last_response_metadata: dict[str, Any] = {}

    def generate(self, *, system_prompt: str, user_prompt: str, settings: AtomGenerationSettings) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(settings.max_tokens),
            "temperature": float(settings.temperature),
            "top_p": float(settings.top_p),
            "stream": False,
        }
        if bool(settings.guided_json):
            payload["response_format"] = {"type": "json_object"}
        if settings.thinking_type:
            payload["thinking"] = {"type": str(settings.thinking_type)}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if _is_transient_http_status(exc.code):
                raise AtomAPITransientError(f"Transient API HTTP {exc.code}: {exc.reason}") from exc
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise AtomAPITransientError(f"Transient API connection error: {exc}") from exc

        self.last_response_metadata = _chat_response_metadata(data)
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")


def make_openai_chat_client_factory(
    *,
    base_url: str,
    model: str,
    api_key_env: str | None,
    timeout: float,
) -> Callable[[], OpenAIChatClient]:
    def _factory() -> OpenAIChatClient:
        api_key = os.environ.get(api_key_env or "") if api_key_env else None
        return OpenAIChatClient(base_url=base_url, model=model, api_key=api_key, timeout=timeout)

    return _factory


def atom_config_payload(settings: AtomGenerationSettings) -> dict[str, Any]:
    return {
        "prompt_version": str(settings.prompt_version),
        "system_prompt_sha256": _sha256_text(SYSTEM_PROMPT),
        "user_prompt_template_sha256": _sha256_text(USER_PROMPT_TEMPLATE),
        "schema_version": str(settings.schema_version),
        "model": str(settings.model),
        "temperature": float(settings.temperature),
        "top_p": float(settings.top_p),
        "max_tokens": int(settings.max_tokens),
        "seed": int(settings.seed),
        "min_atoms": int(settings.min_atoms),
        "max_atoms": int(settings.max_atoms),
        "guided_json": bool(settings.guided_json),
        "thinking_type": settings.thinking_type,
    }


def atom_config_fingerprint(settings: AtomGenerationSettings) -> str:
    payload = atom_config_payload(settings)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def atom_cache_paths(cache_dir: str | Path, split: str, cache_fingerprint: str) -> AtomCachePaths:
    cache_dir = Path(cache_dir)
    return AtomCachePaths(
        atom_cache_path=cache_dir / f"claim_atom_cache_{split}_{cache_fingerprint}.jsonl",
        raw_cache_path=cache_dir / f"raw_claim_atom_cache_{split}_{cache_fingerprint}.jsonl",
        cache_manifest_path=cache_dir / f"claim_atom_cache_manifest_{split}_{cache_fingerprint}.json",
        lock_path=cache_dir / f"claim_atom_cache_{split}_{cache_fingerprint}.lock",
    )


def claim_sha256(claim: str) -> str:
    return _sha256_text(str(claim).strip())


def atom_cache_key(event_id: str, claim_hash: str, cache_fingerprint: str) -> str:
    return f"{event_id}\t{claim_hash}\t{cache_fingerprint}"


def read_atom_cache(path: str | Path, *, cache_fingerprint: str | None = None) -> AtomCacheIndex:
    rows_by_key: dict[str, dict[str, Any]] = {}
    invalid_lines = 0
    duplicate_rows = 0
    total_valid_rows = 0
    path = Path(path)
    if not path.exists():
        return AtomCacheIndex(rows_by_key=rows_by_key)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not _is_valid_atom_cache_row(row, cache_fingerprint=cache_fingerprint):
                invalid_lines += 1
                continue
            key = atom_cache_key(
                str(row["event_id"]),
                str(row["claim_sha256"]),
                str(row["atom_config_fingerprint"]),
            )
            if key in rows_by_key:
                duplicate_rows += 1
            rows_by_key[key] = row
            total_valid_rows += 1
    return AtomCacheIndex(
        rows_by_key=rows_by_key,
        invalid_lines=invalid_lines,
        duplicate_rows=duplicate_rows,
        total_valid_rows=total_valid_rows,
    )


def read_raw_atom_cache(path: str | Path, *, cache_fingerprint: str | None = None) -> dict[str, dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    path = Path(path)
    if not path.exists():
        return rows_by_key
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            event_id = str(row.get("event_id") or "")
            claim_hash = str(row.get("claim_sha256") or "")
            fp = str(row.get("atom_config_fingerprint") or "")
            if not event_id or not claim_hash or not fp:
                continue
            if cache_fingerprint is not None and fp != cache_fingerprint:
                continue
            rows_by_key[atom_cache_key(event_id, claim_hash, fp)] = row
    return rows_by_key


def parse_claim_atoms_from_generation(raw_text: str, *, claim: str) -> tuple[list[dict[str, Any]], str, str | None, str]:
    text = str(raw_text or "").strip()
    if not text:
        return fallback_atoms_for_claim(claim), "empty_generation", "empty generation", "moderate"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        if extracted is None:
            return fallback_atoms_for_claim(claim), "parse_failed", "could not parse JSON object", "moderate"
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as exc:
            return fallback_atoms_for_claim(claim), "parse_failed", f"invalid JSON: {exc}", "moderate"
    if not isinstance(payload, dict):
        return fallback_atoms_for_claim(claim), "parse_failed", "top-level JSON value is not an object", "moderate"
    complexity = str(payload.get("complexity") or "moderate").strip().lower()
    if complexity not in ALLOWED_COMPLEXITIES:
        complexity = "moderate"
    raw_atoms = payload.get("claim_atoms")
    if not isinstance(raw_atoms, list) or not (MIN_ATOMS <= len(raw_atoms) <= MAX_ATOMS):
        return fallback_atoms_for_claim(claim), "parse_failed", f"claim_atoms count must be {MIN_ATOMS}-{MAX_ATOMS}", complexity

    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_atoms, start=1):
        if not isinstance(raw, dict):
            return fallback_atoms_for_claim(claim), "parse_failed", f"atom {idx} is not an object", complexity
        proposition = _compact(raw.get("proposition") or raw.get("text") or "")
        if not proposition:
            return fallback_atoms_for_claim(claim), "parse_failed", f"atom {idx} proposition is empty", complexity
        key = proposition.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords = _keywords(raw.get("keywords"))
        query = _clean_query(raw.get("query_rendering") or proposition)
        atom_type = str(raw.get("type") or "other").strip().lower()
        if atom_type not in ALLOWED_TYPES:
            atom_type = "other"
        atoms.append(
            {
                "atom_id": f"A{len(atoms) + 1}",
                "proposition": proposition,
                "text": proposition,
                "importance": _importance(raw.get("importance")),
                "type": atom_type,
                "keywords": keywords,
                "query_rendering": query,
            }
        )
    if not atoms:
        return fallback_atoms_for_claim(claim), "parse_failed", "no valid deduplicated atoms", complexity
    return atoms, "ok", None, complexity


def fallback_atoms_for_claim(claim: str) -> list[dict[str, Any]]:
    text = _compact(claim) or "Full claim"
    return [
        {
            "atom_id": "A1",
            "proposition": text,
            "text": text,
            "importance": 1.0,
            "type": "other",
            "keywords": _fallback_keywords(text),
            "query_rendering": f"What evidence verifies whether this claim is true: {text}?",
        }
    ]


def format_user_prompt(claim: str) -> str:
    return USER_PROMPT_TEMPLATE.format(claim=str(claim).strip())


def generate_or_load_claim_atoms(
    *,
    examples: Sequence[Any],
    split: str,
    output_dir: str | Path,
    atom_cache_dir: str | Path,
    settings: AtomGenerationSettings,
    client_factory: Callable[[], Any] | None,
    cache_id: str | None = None,
    resume_atoms: bool = True,
    api_max_retries: int = 5,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    api_parse_max_retries: int = 2,
    api_concurrency: int = 1,
    run_metadata: dict[str, Any] | None = None,
    no_progress: bool = False,
) -> AtomGenerationResult:
    started_at = time.time()
    output_dir = Path(output_dir)
    atom_cache_dir = Path(atom_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atom_cache_dir.mkdir(parents=True, exist_ok=True)

    generation_fingerprint = atom_config_fingerprint(settings)
    cache_fingerprint = str(cache_id or generation_fingerprint)
    paths = atom_cache_paths(atom_cache_dir, split, cache_fingerprint)
    config_payload = atom_config_payload(settings)

    normalized_examples = [_normalize_example(example) for example in examples]
    with _cache_lock(paths.lock_path):
        cache_index = (
            read_atom_cache(paths.atom_cache_path, cache_fingerprint=cache_fingerprint)
            if bool(resume_atoms)
            else AtomCacheIndex(rows_by_key={})
        )
        pending = [
            ex
            for ex in normalized_examples
            if _example_cache_key(ex, cache_fingerprint) not in cache_index.rows_by_key
        ]
        n_loaded_initial = len(normalized_examples) - len(pending)
        n_api_generated = 0
        n_parse_failures = 0
        if pending:
            if client_factory is None:
                raise RuntimeError("Missing API client factory for pending atom generations.")
            with paths.atom_cache_path.open("a", encoding="utf-8") as cache_fh, paths.raw_cache_path.open(
                "a", encoding="utf-8"
            ) as raw_fh:
                worker_kwargs = {
                    "split": str(split),
                    "settings": settings,
                    "cache_fingerprint": cache_fingerprint,
                    "generation_fingerprint": generation_fingerprint,
                    "atom_config_payload": config_payload,
                    "client_factory": client_factory,
                    "api_max_retries": int(api_max_retries),
                    "retry_initial_delay": float(retry_initial_delay),
                    "retry_max_delay": float(retry_max_delay),
                    "api_parse_max_retries": int(api_parse_max_retries),
                }
                concurrency = max(int(api_concurrency), 1)
                if concurrency == 1:
                    generated_iter = (
                        _generate_atoms_for_example(ex, **worker_kwargs)
                        for ex in _iter_progress(
                            pending,
                            desc="atom API",
                            unit="claim",
                            disable=bool(no_progress),
                            total=len(pending),
                        )
                    )
                    for row, raw_row, parse_failed in generated_iter:
                        _append_generated_atom_row(
                            cache_fh,
                            raw_fh,
                            cache_index=cache_index,
                            cache_fingerprint=cache_fingerprint,
                            row=row,
                            raw_row=raw_row,
                        )
                        n_api_generated += 1
                        if parse_failed:
                            n_parse_failures += 1
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [
                            executor.submit(_generate_atoms_for_example, ex, **worker_kwargs)
                            for ex in pending
                        ]
                        for future in _iter_progress(
                            as_completed(futures),
                            desc=f"atom API x{concurrency}",
                            unit="claim",
                            disable=bool(no_progress),
                            total=len(futures),
                        ):
                            row, raw_row, parse_failed = future.result()
                            _append_generated_atom_row(
                                cache_fh,
                                raw_fh,
                                cache_index=cache_index,
                                cache_fingerprint=cache_fingerprint,
                                row=row,
                                raw_row=raw_row,
                            )
                            n_api_generated += 1
                            if parse_failed:
                                n_parse_failures += 1

        raw_cache = read_raw_atom_cache(paths.raw_cache_path, cache_fingerprint=cache_fingerprint)
        ordered_rows: list[dict[str, Any]] = []
        ordered_raw_rows: list[dict[str, Any]] = []
        missing_after_generation: list[str] = []
        for ex in _iter_progress(
            normalized_examples,
            desc="atom cache export",
            unit="claim",
            disable=bool(no_progress),
            total=len(normalized_examples),
        ):
            key = _example_cache_key(ex, cache_fingerprint)
            row = cache_index.rows_by_key.get(key)
            if row is None:
                missing_after_generation.append(ex.event_id)
                continue
            ordered_rows.append(row)
            raw_row = raw_cache.get(key)
            if raw_row is not None:
                ordered_raw_rows.append(raw_row)
        if missing_after_generation:
            sample = missing_after_generation[:5]
            raise RuntimeError(f"Missing atom rows after generation for {len(missing_after_generation)} examples: {sample}")

        atoms_path = output_dir / f"claim_atoms_{split}.jsonl"
        raw_generations_path = output_dir / f"raw_claim_atom_generations_{split}.jsonl"
        write_jsonl(atoms_path, ordered_rows)
        write_jsonl(raw_generations_path, ordered_raw_rows)

        manifest = {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": str(split),
            "output_dir": str(output_dir),
            "claim_atoms_path": str(atoms_path),
            "raw_generations_path": str(raw_generations_path),
            "atom_cache_dir": str(atom_cache_dir),
            "atom_cache_path": str(paths.atom_cache_path),
            "raw_atom_cache_path": str(paths.raw_cache_path),
            "atom_cache_manifest_path": str(paths.cache_manifest_path),
            "atom_config_fingerprint": str(cache_fingerprint),
            "generation_config_fingerprint": str(generation_fingerprint),
            "atom_cache_id": str(cache_id) if cache_id else None,
            "atom_config_payload": config_payload,
            "resume_atoms": bool(resume_atoms),
            "n_examples_requested": len(normalized_examples),
            "n_loaded_from_cache": int(n_loaded_initial),
            "n_api_generated": int(n_api_generated),
            "n_pending_after_cache": int(len(pending)),
            "cache_hit_rate": float(n_loaded_initial / len(normalized_examples)) if normalized_examples else 0.0,
            "cache_invalid_lines": int(cache_index.invalid_lines),
            "cache_duplicate_rows": int(cache_index.duplicate_rows),
            "parse_failures": int(n_parse_failures),
            "api_max_retries": int(api_max_retries),
            "api_parse_max_retries": int(api_parse_max_retries),
            "api_concurrency": max(int(api_concurrency), 1),
            "no_progress": bool(no_progress),
            "run_metadata": dict(run_metadata or {}),
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        write_json(output_dir / "claim_atom_manifest.json", manifest)
        write_json(output_dir / f"claim_atom_manifest_{split}.json", manifest)
        write_json(paths.cache_manifest_path, manifest)
        _write_analysis(output_dir / "analysis.md", manifest)
    return AtomGenerationResult(rows=ordered_rows, raw_rows=ordered_raw_rows, manifest=manifest)


def _normalize_example(example: Any) -> AtomInputExample:
    if isinstance(example, AtomInputExample):
        return example
    if isinstance(example, dict):
        return AtomInputExample(
            event_id=str(example.get("event_id") or ""),
            claim=str(example.get("claim") or ""),
            gold_label=str(example.get("gold_label") or ""),
        )
    return AtomInputExample(
        event_id=str(getattr(example, "event_id")),
        claim=str(getattr(example, "claim")),
        gold_label=str(getattr(example, "gold_label", "")),
    )


def _generate_atoms_for_example(
    example: AtomInputExample,
    *,
    split: str,
    settings: AtomGenerationSettings,
    cache_fingerprint: str,
    generation_fingerprint: str,
    atom_config_payload: dict[str, Any],
    client_factory: Callable[[], Any],
    api_max_retries: int,
    retry_initial_delay: float,
    retry_max_delay: float,
    api_parse_max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    client = client_factory()
    prompt = format_user_prompt(example.claim)
    raw_text = ""
    api_metadata: dict[str, Any] = {}
    atoms: list[dict[str, Any]] = []
    parse_status = "empty_generation"
    parse_error: str | None = "empty generation"
    complexity = "moderate"
    parse_attempts = 0
    max_parse_retries = max(int(api_parse_max_retries), 0)
    for parse_attempt in range(max_parse_retries + 1):
        parse_attempts = parse_attempt + 1
        raw_text = _generate_with_retries(
            client,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            settings=settings,
            max_retries=int(api_max_retries),
            initial_delay=float(retry_initial_delay),
            max_delay=float(retry_max_delay),
        )
        api_metadata = dict(getattr(client, "last_response_metadata", {}) or {})
        atoms, parse_status, parse_error, complexity = parse_claim_atoms_from_generation(raw_text, claim=example.claim)
        if parse_status == "ok" or not _should_retry_parse_failure(
            parse_status=parse_status,
            parse_error=parse_error,
            api_metadata=api_metadata,
        ):
            break
        if parse_attempt < max_parse_retries:
            time.sleep(min(1.0, max(float(retry_initial_delay), 0.0)))
    atom_source = "api"
    parse_failed = parse_status != "ok"
    if parse_failed:
        atom_source = "fallback_parse_failed"
    row = _build_atom_row(
        example,
        split=split,
        settings=settings,
        cache_fingerprint=cache_fingerprint,
        generation_fingerprint=generation_fingerprint,
        atom_config_payload=atom_config_payload,
        atoms=atoms,
        complexity=complexity,
        parse_status=parse_status,
        parse_error=parse_error,
        atom_source=atom_source,
    )
    raw_row = _build_raw_row(
        row,
        raw_text=raw_text,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        api_metadata=api_metadata,
        parse_attempts=parse_attempts,
    )
    return row, raw_row, parse_failed


def _build_atom_row(
    example: AtomInputExample,
    *,
    split: str,
    settings: AtomGenerationSettings,
    cache_fingerprint: str,
    generation_fingerprint: str,
    atom_config_payload: dict[str, Any],
    atoms: list[dict[str, Any]],
    complexity: str,
    parse_status: str,
    parse_error: str | None,
    atom_source: str,
) -> dict[str, Any]:
    return {
        "event_id": str(example.event_id),
        "claim": str(example.claim),
        "claim_sha256": claim_sha256(example.claim),
        "gold_label": str(example.gold_label),
        "split": str(split),
        "atom_config_fingerprint": str(cache_fingerprint),
        "generation_config_fingerprint": str(generation_fingerprint),
        "atom_config_payload": atom_config_payload,
        "model": str(settings.model),
        "parse_status": str(parse_status),
        "parse_error": parse_error,
        "atom_source": str(atom_source),
        "complexity": str(complexity),
        "claim_atoms": atoms,
        "n_atoms": int(len(atoms)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_raw_row(
    atom_row: dict[str, Any],
    *,
    raw_text: str,
    system_prompt: str,
    user_prompt: str,
    api_metadata: dict[str, Any] | None = None,
    parse_attempts: int = 1,
) -> dict[str, Any]:
    return {
        "event_id": atom_row.get("event_id"),
        "claim": atom_row.get("claim"),
        "claim_sha256": atom_row.get("claim_sha256"),
        "gold_label": atom_row.get("gold_label"),
        "split": atom_row.get("split"),
        "atom_config_fingerprint": atom_row.get("atom_config_fingerprint"),
        "generation_config_fingerprint": atom_row.get("generation_config_fingerprint"),
        "model": atom_row.get("model"),
        "parse_status": atom_row.get("parse_status"),
        "parse_error": atom_row.get("parse_error"),
        "atom_source": atom_row.get("atom_source"),
        "complexity": atom_row.get("complexity"),
        "claim_atoms": atom_row.get("claim_atoms"),
        "raw_text": str(raw_text or ""),
        "api_metadata": dict(api_metadata or {}),
        "parse_attempts": int(parse_attempts),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "created_at": atom_row.get("created_at"),
    }


def _append_generated_atom_row(
    cache_fh: Any,
    raw_fh: Any,
    *,
    cache_index: AtomCacheIndex,
    cache_fingerprint: str,
    row: dict[str, Any],
    raw_row: dict[str, Any],
) -> None:
    cache_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    cache_fh.flush()
    raw_fh.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
    raw_fh.flush()
    key = atom_cache_key(str(row["event_id"]), str(row["claim_sha256"]), cache_fingerprint)
    cache_index.rows_by_key[key] = row
    cache_index.total_valid_rows += 1


def _generate_with_retries(
    client: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    settings: AtomGenerationSettings,
    max_retries: int,
    initial_delay: float,
    max_delay: float,
) -> str:
    attempts = max(int(max_retries), 0) + 1
    delay = max(float(initial_delay), 0.0)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return str(client.generate(system_prompt=system_prompt, user_prompt=user_prompt, settings=settings))
        except AtomAPITransientError as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break
            sleep_for = min(float(max_delay), delay * (2 ** attempt))
            if sleep_for > 0:
                time.sleep(sleep_for + random.uniform(0.0, min(0.25, sleep_for)))
    assert last_exc is not None
    raise last_exc


def _should_retry_parse_failure(
    *,
    parse_status: str,
    parse_error: str | None,
    api_metadata: dict[str, Any],
) -> bool:
    finish_reason = str(api_metadata.get("finish_reason") or "")
    if finish_reason == "length":
        return True
    if parse_status == "empty_generation":
        return True
    if parse_error and "could not parse JSON object" in parse_error:
        return True
    return False


def _example_cache_key(example: AtomInputExample, cache_fingerprint: str) -> str:
    return atom_cache_key(example.event_id, claim_sha256(example.claim), cache_fingerprint)


def _is_valid_atom_cache_row(row: Any, *, cache_fingerprint: str | None) -> bool:
    if not isinstance(row, dict):
        return False
    event_id = str(row.get("event_id") or "")
    claim_hash = str(row.get("claim_sha256") or "")
    fp = str(row.get("atom_config_fingerprint") or "")
    atoms = row.get("claim_atoms")
    if not event_id or not claim_hash or not fp:
        return False
    if cache_fingerprint is not None and fp != cache_fingerprint:
        return False
    if not isinstance(atoms, list) or not atoms:
        return False
    return True


def _keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple):
        values = list(value)
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _compact(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 12:
            break
    return out


def _fallback_keywords(text: str) -> list[str]:
    words = [word.strip(".,;:!?()[]{}\"'") for word in str(text).split()]
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) < 4:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(word)
        if len(out) >= 8:
            break
    return out


def _clean_query(value: Any) -> str:
    text = _compact(value)
    if not text:
        return ""
    if not text.endswith("?"):
        text = f"{text}?"
    return text


def _importance(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 1.0
    if raw > 1.0:
        raw = 1.0
    return float(max(0.05, min(raw, 1.0)))


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _is_transient_http_status(status: int) -> bool:
    return int(status) in {408, 409, 425, 429} or int(status) >= 500


def _chat_response_metadata(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    return {
        "id": data.get("id"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "content_length": len(str(content or "")),
        "has_reasoning_content": bool(reasoning),
        "reasoning_content_length": len(str(reasoning or "")),
        "usage": usage,
    }


@contextlib.contextmanager
def _cache_lock(lock_path: Path) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as fh:
        try:
            import fcntl
        except ImportError:
            yield
            return
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _iter_progress(
    items: Iterable[Any],
    *,
    desc: str,
    unit: str,
    disable: bool,
    total: int | None = None,
) -> Iterable[Any]:
    if disable:
        return items
    try:
        from tqdm.auto import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc, unit=unit, total=total, dynamic_ncols=True)


def _write_analysis(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Claim Atomization Cache",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- split: `{manifest.get('split')}`",
        f"- atom_config_fingerprint: `{manifest.get('atom_config_fingerprint')}`",
        f"- generation_config_fingerprint: `{manifest.get('generation_config_fingerprint')}`",
        f"- n_examples_requested: {manifest.get('n_examples_requested')}",
        f"- n_loaded_from_cache: {manifest.get('n_loaded_from_cache')}",
        f"- n_api_generated: {manifest.get('n_api_generated')}",
        f"- cache_hit_rate: {manifest.get('cache_hit_rate')}",
        f"- parse_failures: {manifest.get('parse_failures')}",
        f"- atom_cache_path: `{manifest.get('atom_cache_path')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
