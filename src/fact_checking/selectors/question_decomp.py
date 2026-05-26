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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fact_checking.selectors.stage2_oracle import write_json, write_jsonl


QUESTION_DECOMP_PROMPT_VERSION = "question_decomp_retrieval_v0"
QUESTION_DECOMP_SCHEMA_VERSION = "question_decomp_questions_v1"
MIN_QUESTIONS = 1
MAX_QUESTIONS = 5

SYSTEM_PROMPT = """You generate retrieval questions for fact-checking. Use only the given claim.
Do not answer the claim. Do not add external facts.
Return only valid JSON."""

USER_PROMPT_TEMPLATE = """Claim:
{claim}

Generate 1 to 5 self-contained retrieval questions needed to verify this claim.

Rules:
- Use 1 question if the claim is simple.
- Use 2-3 questions if the claim has multiple entities, dates, quantities, comparisons, or attribution.
- Use 4-5 questions only if the claim is multi-part, causal, conditional, or requires separate evidence facets.
- Question q1 must be a broad overall verification question for the original claim.
- Every question must include the necessary named entities, dates, numbers, comparison target, and attribution if present.
- Do not generate vague topics.
- Do not answer the questions.
- Avoid redundant questions.

Return JSON exactly in this schema:
{{
  "complexity": "simple|moderate|complex",
  "questions": [
    {{
      "id": "q1",
      "question": "...?",
      "focus": "overall|entity|quantity|time|comparison|causal|attribution|policy|other",
      "priority": 1
    }}
  ]
}}
"""

ALLOWED_COMPLEXITIES = {"simple", "moderate", "complex"}
ALLOWED_FOCI = {
    "overall",
    "entity",
    "quantity",
    "time",
    "comparison",
    "causal",
    "attribution",
    "policy",
    "other",
}


@dataclass(frozen=True)
class QuestionInputExample:
    event_id: str
    claim: str
    gold_label: str = ""


@dataclass(frozen=True)
class QuestionGenerationSettings:
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int = 20260526
    guided_json: bool = True
    thinking_type: str | None = None
    prompt_version: str = QUESTION_DECOMP_PROMPT_VERSION
    schema_version: str = QUESTION_DECOMP_SCHEMA_VERSION
    min_questions: int = MIN_QUESTIONS
    max_questions: int = MAX_QUESTIONS


@dataclass(frozen=True)
class QuestionCachePaths:
    question_cache_path: Path
    raw_cache_path: Path
    cache_manifest_path: Path
    lock_path: Path


@dataclass
class QuestionCacheIndex:
    rows_by_key: dict[str, dict[str, Any]]
    invalid_lines: int = 0
    duplicate_rows: int = 0
    total_valid_rows: int = 0


@dataclass
class QuestionGenerationResult:
    rows: list[dict[str, Any]]
    raw_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


class QuestionAPITransientError(RuntimeError):
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

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        settings: QuestionGenerationSettings,
    ) -> str:
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
                raise QuestionAPITransientError(f"Transient API HTTP {exc.code}: {exc.reason}") from exc
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise QuestionAPITransientError(f"Transient API connection error: {exc}") from exc

        self.last_response_metadata = _chat_response_metadata(data)
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")


def question_config_payload(settings: QuestionGenerationSettings) -> dict[str, Any]:
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
        "min_questions": int(settings.min_questions),
        "max_questions": int(settings.max_questions),
        "guided_json": bool(settings.guided_json),
        "thinking_type": settings.thinking_type,
    }


def question_config_fingerprint(settings: QuestionGenerationSettings) -> str:
    payload = question_config_payload(settings)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def question_cache_paths(cache_dir: str | Path, split: str, cache_fingerprint: str) -> QuestionCachePaths:
    cache_dir = Path(cache_dir)
    return QuestionCachePaths(
        question_cache_path=cache_dir / f"question_cache_{split}_{cache_fingerprint}.jsonl",
        raw_cache_path=cache_dir / f"raw_question_cache_{split}_{cache_fingerprint}.jsonl",
        cache_manifest_path=cache_dir / f"question_cache_manifest_{split}_{cache_fingerprint}.json",
        lock_path=cache_dir / f"question_cache_{split}_{cache_fingerprint}.lock",
    )


def claim_sha256(claim: str) -> str:
    return _sha256_text(str(claim).strip())


def question_cache_key(event_id: str, claim_hash: str, cache_fingerprint: str) -> str:
    return f"{event_id}\t{claim_hash}\t{cache_fingerprint}"


def read_question_cache(path: str | Path, *, cache_fingerprint: str | None = None) -> QuestionCacheIndex:
    rows_by_key: dict[str, dict[str, Any]] = {}
    invalid_lines = 0
    duplicate_rows = 0
    total_valid_rows = 0
    path = Path(path)
    if not path.exists():
        return QuestionCacheIndex(rows_by_key=rows_by_key)
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
            if not _is_valid_question_cache_row(row, cache_fingerprint=cache_fingerprint):
                invalid_lines += 1
                continue
            key = question_cache_key(
                str(row["event_id"]),
                str(row["claim_sha256"]),
                str(row["question_config_fingerprint"]),
            )
            if key in rows_by_key:
                duplicate_rows += 1
            rows_by_key[key] = row
            total_valid_rows += 1
    return QuestionCacheIndex(
        rows_by_key=rows_by_key,
        invalid_lines=invalid_lines,
        duplicate_rows=duplicate_rows,
        total_valid_rows=total_valid_rows,
    )


def read_raw_question_cache(path: str | Path, *, cache_fingerprint: str | None = None) -> dict[str, dict[str, Any]]:
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
            fp = str(row.get("question_config_fingerprint") or "")
            if not event_id or not claim_hash or not fp:
                continue
            if cache_fingerprint is not None and fp != cache_fingerprint:
                continue
            rows_by_key[question_cache_key(event_id, claim_hash, fp)] = row
    return rows_by_key


def parse_questions_from_generation(raw_text: str) -> tuple[list[dict[str, Any]], str, str | None, str]:
    text = str(raw_text or "").strip()
    if not text:
        return [], "empty_generation", "empty generation", "moderate"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        if extracted is None:
            return [], "parse_failed", "could not parse JSON object", "moderate"
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as exc:
            return [], "parse_failed", str(exc), "moderate"
    if not isinstance(payload, dict):
        return [], "parse_failed", "top-level JSON value is not an object", "moderate"
    complexity = str(payload.get("complexity") or "moderate").strip().lower()
    if complexity not in ALLOWED_COMPLEXITIES:
        complexity = "moderate"
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        return [], "parse_failed", "questions must be a list", complexity
    if not (MIN_QUESTIONS <= len(raw_questions) <= MAX_QUESTIONS):
        return [], "parse_failed", f"questions count must be {MIN_QUESTIONS}-{MAX_QUESTIONS}", complexity

    questions: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for idx, raw in enumerate(raw_questions, start=1):
        if not isinstance(raw, dict):
            return [], "parse_failed", f"question {idx} is not an object", complexity
        text_value = _clean_question_text(raw.get("question"))
        if not text_value:
            return [], "parse_failed", f"question {idx} is empty", complexity
        text_key = " ".join(text_value.lower().split())
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        focus = str(raw.get("focus") or ("overall" if idx == 1 else "other")).strip().lower()
        if focus not in ALLOWED_FOCI:
            focus = "other"
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "question": text_value,
                "focus": focus,
                "priority": len(questions) + 1,
            }
        )
    if not (MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS):
        return [], "parse_failed", f"valid deduplicated question count must be {MIN_QUESTIONS}-{MAX_QUESTIONS}", complexity
    questions[0]["focus"] = "overall"
    return questions, "ok", None, complexity


def fallback_questions_for_claim(claim: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "q1",
            "question": f"What evidence verifies whether this claim is true: {str(claim).strip()}?",
            "focus": "overall",
            "priority": 1,
        }
    ]


def format_user_prompt(claim: str) -> str:
    return USER_PROMPT_TEMPLATE.format(claim=str(claim).strip())


def generate_or_load_questions(
    *,
    examples: Sequence[Any],
    split: str,
    output_dir: str | Path,
    question_cache_dir: str | Path,
    settings: QuestionGenerationSettings,
    client_factory: Callable[[], Any] | None,
    cache_id: str | None = None,
    resume_questions: bool = True,
    api_max_retries: int = 5,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    api_parse_max_retries: int = 2,
    run_metadata: dict[str, Any] | None = None,
    no_progress: bool = False,
) -> QuestionGenerationResult:
    started_at = time.time()
    output_dir = Path(output_dir)
    question_cache_dir = Path(question_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    question_cache_dir.mkdir(parents=True, exist_ok=True)

    generation_fingerprint = question_config_fingerprint(settings)
    cache_fingerprint = str(cache_id or generation_fingerprint)
    paths = question_cache_paths(question_cache_dir, split, cache_fingerprint)
    config_payload = question_config_payload(settings)

    normalized_examples = [_normalize_example(example) for example in examples]
    with _cache_lock(paths.lock_path):
        cache_index = (
            read_question_cache(paths.question_cache_path, cache_fingerprint=cache_fingerprint)
            if bool(resume_questions)
            else QuestionCacheIndex(rows_by_key={})
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
                raise RuntimeError("Missing API client factory for pending question generations.")
            client = client_factory()
            with paths.question_cache_path.open("a", encoding="utf-8") as cache_fh, paths.raw_cache_path.open(
                "a", encoding="utf-8"
            ) as raw_fh:
                for ex in _iter_progress(
                    pending,
                    desc="question API",
                    unit="claim",
                    disable=bool(no_progress),
                ):
                    prompt = format_user_prompt(ex.claim)
                    raw_text = ""
                    api_metadata: dict[str, Any] = {}
                    questions: list[dict[str, Any]] = []
                    parse_status = "empty_generation"
                    parse_error: str | None = "empty generation"
                    complexity = "moderate"
                    parse_attempts = 0
                    for parse_attempt in range(max(int(api_parse_max_retries), 0) + 1):
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
                        questions, parse_status, parse_error, complexity = parse_questions_from_generation(raw_text)
                        if parse_status == "ok" or not _should_retry_parse_failure(
                            parse_status=parse_status,
                            parse_error=parse_error,
                            api_metadata=api_metadata,
                        ):
                            break
                        if parse_attempt < max(int(api_parse_max_retries), 0):
                            time.sleep(min(1.0, max(float(retry_initial_delay), 0.0)))
                    question_source = "api"
                    if parse_status != "ok":
                        questions = fallback_questions_for_claim(ex.claim)
                        question_source = "fallback_parse_failed"
                        n_parse_failures += 1
                    row = _build_question_row(
                        ex,
                        split=split,
                        settings=settings,
                        cache_fingerprint=cache_fingerprint,
                        generation_fingerprint=generation_fingerprint,
                        question_config_payload=config_payload,
                        questions=questions,
                        complexity=complexity,
                        parse_status=parse_status,
                        parse_error=parse_error,
                        question_source=question_source,
                    )
                    raw_row = _build_raw_row(
                        row,
                        raw_text=raw_text,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=prompt,
                        api_metadata=api_metadata,
                        parse_attempts=parse_attempts,
                    )
                    cache_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    cache_fh.flush()
                    raw_fh.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                    raw_fh.flush()
                    cache_index.rows_by_key[_example_cache_key(ex, cache_fingerprint)] = row
                    cache_index.total_valid_rows += 1
                    n_api_generated += 1

        raw_cache = read_raw_question_cache(paths.raw_cache_path, cache_fingerprint=cache_fingerprint)
        ordered_rows: list[dict[str, Any]] = []
        ordered_raw_rows: list[dict[str, Any]] = []
        missing_after_generation: list[str] = []
        for ex in _iter_progress(
            normalized_examples,
            desc="question cache export",
            unit="claim",
            disable=bool(no_progress),
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
            raise RuntimeError(f"Missing question rows after generation for {len(missing_after_generation)} examples: {sample}")

        questions_path = output_dir / f"questions_{split}.jsonl"
        raw_generations_path = output_dir / f"raw_question_generations_{split}.jsonl"
        write_jsonl(questions_path, ordered_rows)
        write_jsonl(raw_generations_path, ordered_raw_rows)

        manifest = {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": str(split),
            "output_dir": str(output_dir),
            "questions_path": str(questions_path),
            "raw_generations_path": str(raw_generations_path),
            "question_cache_dir": str(question_cache_dir),
            "question_cache_path": str(paths.question_cache_path),
            "raw_question_cache_path": str(paths.raw_cache_path),
            "question_cache_manifest_path": str(paths.cache_manifest_path),
            "question_config_fingerprint": str(cache_fingerprint),
            "generation_config_fingerprint": str(generation_fingerprint),
            "question_cache_id": str(cache_id) if cache_id else None,
            "question_config_payload": config_payload,
            "resume_questions": bool(resume_questions),
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
            "no_progress": bool(no_progress),
            "run_metadata": dict(run_metadata or {}),
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        write_json(output_dir / "question_manifest.json", manifest)
        write_json(paths.cache_manifest_path, manifest)
        _write_analysis(output_dir / "analysis.md", manifest)
    return QuestionGenerationResult(rows=ordered_rows, raw_rows=ordered_raw_rows, manifest=manifest)


def make_openai_chat_client_factory(
    *,
    base_url: str,
    model: str,
    api_key_env: str | None,
    timeout: float,
    thinking_type: str | None = None,
) -> Callable[[], OpenAIChatClient]:
    def _factory() -> OpenAIChatClient:
        api_key = os.environ.get(api_key_env or "") if api_key_env else None
        return OpenAIChatClient(base_url=base_url, model=model, api_key=api_key, timeout=timeout)

    return _factory


def _normalize_example(example: Any) -> QuestionInputExample:
    if isinstance(example, QuestionInputExample):
        return example
    if isinstance(example, dict):
        return QuestionInputExample(
            event_id=str(example.get("event_id") or ""),
            claim=str(example.get("claim") or ""),
            gold_label=str(example.get("gold_label") or ""),
        )
    return QuestionInputExample(
        event_id=str(getattr(example, "event_id")),
        claim=str(getattr(example, "claim")),
        gold_label=str(getattr(example, "gold_label", "")),
    )


def _build_question_row(
    example: QuestionInputExample,
    *,
    split: str,
    settings: QuestionGenerationSettings,
    cache_fingerprint: str,
    generation_fingerprint: str,
    question_config_payload: dict[str, Any],
    questions: list[dict[str, Any]],
    complexity: str,
    parse_status: str,
    parse_error: str | None,
    question_source: str,
) -> dict[str, Any]:
    return {
        "event_id": str(example.event_id),
        "claim": str(example.claim),
        "claim_sha256": claim_sha256(example.claim),
        "gold_label": str(example.gold_label),
        "split": str(split),
        "question_config_fingerprint": str(cache_fingerprint),
        "generation_config_fingerprint": str(generation_fingerprint),
        "question_config_payload": question_config_payload,
        "model": str(settings.model),
        "parse_status": str(parse_status),
        "parse_error": parse_error,
        "question_source": str(question_source),
        "complexity": str(complexity),
        "questions": questions,
        "n_questions": int(len(questions)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_raw_row(
    question_row: dict[str, Any],
    *,
    raw_text: str,
    system_prompt: str,
    user_prompt: str,
    api_metadata: dict[str, Any] | None = None,
    parse_attempts: int = 1,
) -> dict[str, Any]:
    return {
        "event_id": question_row.get("event_id"),
        "claim": question_row.get("claim"),
        "claim_sha256": question_row.get("claim_sha256"),
        "gold_label": question_row.get("gold_label"),
        "split": question_row.get("split"),
        "question_config_fingerprint": question_row.get("question_config_fingerprint"),
        "generation_config_fingerprint": question_row.get("generation_config_fingerprint"),
        "model": question_row.get("model"),
        "parse_status": question_row.get("parse_status"),
        "parse_error": question_row.get("parse_error"),
        "question_source": question_row.get("question_source"),
        "complexity": question_row.get("complexity"),
        "questions": question_row.get("questions"),
        "raw_text": str(raw_text or ""),
        "api_metadata": dict(api_metadata or {}),
        "parse_attempts": int(parse_attempts),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "created_at": question_row.get("created_at"),
    }


def _generate_with_retries(
    client: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    settings: QuestionGenerationSettings,
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
        except QuestionAPITransientError as exc:
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
    if parse_error == "could not parse JSON object":
        return True
    return False


def _iter_progress(items: Sequence[Any], *, desc: str, unit: str, disable: bool) -> Iterable[Any]:
    if disable:
        return items
    try:
        from tqdm.auto import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)


def _example_cache_key(example: QuestionInputExample, cache_fingerprint: str) -> str:
    return question_cache_key(example.event_id, claim_sha256(example.claim), cache_fingerprint)


def _is_valid_question_cache_row(row: Any, *, cache_fingerprint: str | None) -> bool:
    if not isinstance(row, dict):
        return False
    event_id = str(row.get("event_id") or "")
    claim_hash = str(row.get("claim_sha256") or "")
    fp = str(row.get("question_config_fingerprint") or "")
    questions = row.get("questions")
    if not event_id or not claim_hash or not fp:
        return False
    if cache_fingerprint is not None and fp != cache_fingerprint:
        return False
    if not isinstance(questions, list) or not questions:
        return False
    return True


def _clean_question_text(value: Any) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if not text:
        return ""
    if not text.endswith("?"):
        text = f"{text}?"
    return text


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


def _write_analysis(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Question Decomposition Retrieval Cache",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- split: `{manifest.get('split')}`",
        f"- question_config_fingerprint: `{manifest.get('question_config_fingerprint')}`",
        f"- generation_config_fingerprint: `{manifest.get('generation_config_fingerprint')}`",
        f"- n_examples_requested: {manifest.get('n_examples_requested')}",
        f"- n_loaded_from_cache: {manifest.get('n_loaded_from_cache')}",
        f"- n_api_generated: {manifest.get('n_api_generated')}",
        f"- cache_hit_rate: {manifest.get('cache_hit_rate')}",
        f"- parse_failures: {manifest.get('parse_failures')}",
        f"- question_cache_path: `{manifest.get('question_cache_path')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
