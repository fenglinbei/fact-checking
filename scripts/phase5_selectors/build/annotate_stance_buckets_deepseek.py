#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from fact_checking.selectors.stance_buckets import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TeacherAnnotationError,
    annotation_key,
    format_user_prompt,
    parse_teacher_content,
    teacher_annotation_payload,
)
from fact_checking.utils.io import read_jsonl, save_json


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class AnnotationJob:
    event_id: str
    claim: str
    candidate_uid: str
    candidate_key: str
    evidence_text: str
    annotation_key: str


class APITransientError(RuntimeError):
    pass


class APINonRetryableError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, *, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = max(int(requests_per_minute), 1)
        self.tokens_per_minute = max(int(tokens_per_minute), 1)
        self._request_times: list[float] = []
        self._token_events: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def wait(self, estimated_tokens: int) -> None:
        estimated_tokens = max(int(estimated_tokens), 1)
        while True:
            with self._lock:
                now = time.time()
                cutoff = now - 60.0
                self._request_times = [stamp for stamp in self._request_times if stamp >= cutoff]
                self._token_events = [(stamp, tokens) for stamp, tokens in self._token_events if stamp >= cutoff]
                request_ok = len(self._request_times) < self.requests_per_minute
                token_ok = sum(tokens for _, tokens in self._token_events) + estimated_tokens <= self.tokens_per_minute
                if request_ok and token_ok:
                    self._request_times.append(now)
                    self._token_events.append((now, estimated_tokens))
                    return
                sleeps: list[float] = [0.25]
                if self._request_times:
                    sleeps.append(max(0.25, 60.0 - (now - min(self._request_times))))
                if self._token_events:
                    sleeps.append(max(0.25, 60.0 - (now - min(stamp for stamp, _ in self._token_events))))
            time.sleep(min(sleeps))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate claim-evidence stance buckets with a DeepSeek-compatible chat API.")
    p.add_argument("--candidate-pool", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--model", default=os.environ.get("TEACHER_MODEL", DEFAULT_MODEL))
    p.add_argument("--api-key-env", default=os.environ.get("TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--top-logprobs", type=int, default=20)
    p.add_argument("--fallback-top-logprobs", type=int, default=5)
    p.add_argument("--thinking-type", default="disabled", choices=["disabled", "enabled", "none"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests-per-minute", type=int, default=120)
    p.add_argument("--tokens-per-minute", type=int, default=200000)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-base-sleep", type=float, default=2.0)
    p.add_argument("--retry-max-sleep", type=float, default=60.0)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / f"deepseek_teacher_annotations_{args.split}.jsonl"
    raw_path = out_dir / f"deepseek_teacher_raw_responses_{args.split}.jsonl"
    errors_path = out_dir / f"deepseek_teacher_errors_{args.split}.jsonl"
    progress_path = out_dir / "deepseek_teacher_progress.json"

    rows = read_jsonl(args.candidate_pool)
    jobs = build_jobs(rows, model=str(args.model), sample_limit=args.sample_limit)
    completed_keys = _load_completed_keys(annotations_path) if args.resume else set()
    pending = [job for job in jobs if job.annotation_key not in completed_keys]
    api_key = os.environ.get(str(args.api_key_env) or "")
    limiter = RateLimiter(
        requests_per_minute=int(args.requests_per_minute),
        tokens_per_minute=int(args.tokens_per_minute),
    )

    n_completed = len(completed_keys)
    n_written = 0
    n_errors = 0
    usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    top_logprobs_used = CounterLike()

    mode = "a" if args.resume else "w"
    with annotations_path.open(mode, encoding="utf-8") as ann_fh, raw_path.open(mode, encoding="utf-8") as raw_fh, errors_path.open(mode, encoding="utf-8") as err_fh:
        with ThreadPoolExecutor(max_workers=max(int(args.concurrency), 1)) as executor:
            futures = [
                executor.submit(_run_job, job, args=args, api_key=api_key, limiter=limiter)
                for job in pending
            ]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"deepseek stance [{args.split}]",
                unit="candidate",
                dynamic_ncols=True,
                disable=bool(args.no_progress),
            )
            for future in iterator:
                result = future.result()
                raw = result.get("raw")
                if raw:
                    raw_fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    raw_fh.flush()
                if result.get("ok"):
                    ann_fh.write(json.dumps(result["annotation"], ensure_ascii=False) + "\n")
                    ann_fh.flush()
                    n_written += 1
                    usage = result["annotation"].get("api_usage") or {}
                    for key in usage_totals:
                        usage_totals[key] += int(usage.get(key) or 0)
                    top_logprobs_used.add(str(result["annotation"].get("top_logprobs_used")))
                else:
                    err_fh.write(json.dumps(result["error"], ensure_ascii=False) + "\n")
                    err_fh.flush()
                    n_errors += 1
                _write_progress(
                    progress_path,
                    args=args,
                    started_at=started_at,
                    n_jobs=len(jobs),
                    n_pending=len(pending),
                    n_completed_initial=n_completed,
                    n_written=n_written,
                    n_errors=n_errors,
                    usage_totals=usage_totals,
                    top_logprobs_used=top_logprobs_used.payload(),
                )

    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "candidate_pool": str(args.candidate_pool),
        "split": str(args.split),
        "base_url": str(args.base_url),
        "model": str(args.model),
        "api_key_env": str(args.api_key_env),
        "prompt_version": PROMPT_VERSION,
        "top_logprobs_requested": int(args.top_logprobs),
        "fallback_top_logprobs": int(args.fallback_top_logprobs),
        "thinking_type": str(args.thinking_type),
        "top_logprobs_used": top_logprobs_used.payload(),
        "concurrency": max(int(args.concurrency), 1),
        "requests_per_minute": int(args.requests_per_minute),
        "tokens_per_minute": int(args.tokens_per_minute),
        "resume": bool(args.resume),
        "n_jobs": len(jobs),
        "n_completed_initial": n_completed,
        "n_pending": len(pending),
        "n_written": n_written,
        "n_errors": n_errors,
        "usage_totals": usage_totals,
        "outputs": {
            "annotations": str(annotations_path),
            "raw_responses": str(raw_path),
            "errors": str(errors_path),
            "progress": str(progress_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "teacher_api_audit_manifest.json")
    save_json(usage_totals, out_dir / "teacher_api_usage_summary.json")
    save_json({"note": "No price table is encoded; totals are token usage only.", "usage_totals": usage_totals}, out_dir / "teacher_api_cost_estimate.json")
    print(f"Wrote annotations: {annotations_path}")
    print(f"new={n_written} errors={n_errors} skipped_resume={n_completed} total_jobs={len(jobs)}")


def build_jobs(rows: list[dict[str, Any]], *, model: str, sample_limit: int | None) -> list[AnnotationJob]:
    jobs: list[AnnotationJob] = []
    seen: set[str] = set()
    for row in rows:
        event_id = str(row.get("event_id") or "")
        claim = str(row.get("claim") or "")
        for candidate in row.get("candidates") or []:
            candidate_uid = str(candidate.get("candidate_uid") or "")
            if not event_id or not candidate_uid:
                continue
            key = annotation_key(
                event_id=event_id,
                candidate_uid=candidate_uid,
                prompt_version=PROMPT_VERSION,
                model=model,
            )
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                AnnotationJob(
                    event_id=event_id,
                    claim=claim,
                    candidate_uid=candidate_uid,
                    candidate_key=str(candidate.get("candidate_key") or ""),
                    evidence_text=str(candidate.get("text") or ""),
                    annotation_key=key,
                )
            )
            if sample_limit is not None and len(jobs) >= int(sample_limit):
                return jobs
    return jobs


def _run_job(job: AnnotationJob, *, args: argparse.Namespace, api_key: str | None, limiter: RateLimiter) -> dict[str, Any]:
    system_prompt = SYSTEM_PROMPT
    user_prompt = format_user_prompt(claim=job.claim, evidence_text=job.evidence_text)
    estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4 + int(args.max_tokens)
    attempts = max(int(args.max_retries), 0) + 1
    last_error = ""
    last_raw: dict[str, Any] | None = None
    attempts_made = 0
    top_logprobs = int(args.top_logprobs)
    for attempt in range(1, attempts + 1):
        attempts_made = attempt
        try:
            limiter.wait(estimated_tokens)
            data, top_used = _chat_completion(
                args=args,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                top_logprobs=top_logprobs,
            )
            created_at = datetime.now(timezone.utc).isoformat()
            raw = _raw_row(
                job,
                args=args,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=data,
                top_logprobs_used=top_used,
                created_at=created_at,
            )
            content = ""
            try:
                content = _response_content(data)
                annotation = parse_teacher_content(content)
            except TeacherAnnotationError as exc:
                raw["parse_status"] = "schema_validation_failed"
                raw["parse_error"] = str(exc)
                raw["response_content_preview"] = content[:1000]
                last_raw = raw
                raise
            raw["parse_status"] = "ok"
            raw["response_content_preview"] = content[:1000]
            usage = dict(data.get("usage") or {})
            row = {
                "annotation_key": job.annotation_key,
                "event_id": job.event_id,
                "candidate_uid": job.candidate_uid,
                "candidate_key": job.candidate_key,
                "prompt_version": PROMPT_VERSION,
                "model": str(args.model),
                "teacher_annotation": teacher_annotation_payload(annotation),
                "api_usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                    "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens") or 0),
                    "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens") or 0),
                },
                "finish_reason": _finish_reason(data),
                "logprobs_saved": True,
                "top_logprobs_used": int(top_used),
                "created_at": created_at,
            }
            return {"ok": True, "annotation": row, "raw": raw}
        except TeacherAnnotationError as exc:
            last_error = f"schema_validation: {exc}"
            if attempt >= min(2, attempts):
                break
        except APITransientError as exc:
            last_error = f"transient: {exc}"
        except APINonRetryableError as exc:
            message = str(exc)
            if int(top_logprobs) > int(args.fallback_top_logprobs) and ("top_logprobs" in message or "logprobs" in message):
                top_logprobs = int(args.fallback_top_logprobs)
                last_error = f"retrying_with_top_logprobs_{top_logprobs}: {message}"
            else:
                return {"ok": False, "error": _error_row(job, "terminal_api_error", message, attempt, raw=last_raw), "raw": last_raw}
        except Exception as exc:
            last_error = f"unexpected: {type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(min(float(args.retry_max_sleep), float(args.retry_base_sleep) * (2 ** (attempt - 1))))
    return {"ok": False, "error": _error_row(job, "retry_exhausted", last_error, attempts_made, raw=last_raw), "raw": last_raw}


def _chat_completion(
    *,
    args: argparse.Namespace,
    api_key: str | None,
    system_prompt: str,
    user_prompt: str,
    top_logprobs: int,
) -> tuple[dict[str, Any], int]:
    payload = {
        "model": str(args.model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "temperature": 0,
        "max_tokens": int(args.max_tokens),
        "user": "stance_bucket_v0",
        "stream": False,
    }
    thinking_type = str(getattr(args, "thinking_type", "disabled") or "disabled")
    if thinking_type != "none":
        payload["thinking"] = {"type": thinking_type}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url=f"{str(args.base_url).rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(args.timeout)) as response:
            return json.loads(response.read().decode("utf-8")), int(top_logprobs)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise APITransientError(f"HTTP {exc.code}: {body[:500]}") from exc
        raise APINonRetryableError(f"HTTP {exc.code}: {body[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise APITransientError(str(exc)) from exc


def _raw_row(
    job: AnnotationJob,
    *,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    response: dict[str, Any],
    top_logprobs_used: int,
    created_at: str,
) -> dict[str, Any]:
    return {
        "annotation_key": job.annotation_key,
        "event_id": job.event_id,
        "candidate_uid": job.candidate_uid,
        "candidate_key": job.candidate_key,
        "model": str(args.model),
        "prompt_version": PROMPT_VERSION,
        "top_logprobs_used": int(top_logprobs_used),
        "thinking_type": str(getattr(args, "thinking_type", "disabled") or "disabled"),
        "finish_reason": _finish_reason(response),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "created_at": created_at,
    }


def _response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise TeacherAnnotationError("API response has no choices.")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def _error_row(job: AnnotationJob, error_type: str, message: str, attempts: int, *, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "annotation_key": job.annotation_key,
        "event_id": job.event_id,
        "candidate_uid": job.candidate_uid,
        "candidate_key": job.candidate_key,
        "error_type": error_type,
        "message": str(message),
        "attempts": int(attempts),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw:
        response = raw.get("response") if isinstance(raw.get("response"), dict) else {}
        usage = response.get("usage") or {}
        row.update(
            {
                "finish_reason": raw.get("finish_reason", ""),
                "parse_status": raw.get("parse_status", ""),
                "parse_error": raw.get("parse_error", ""),
                "response_content_preview": raw.get("response_content_preview", ""),
                "top_logprobs_used": raw.get("top_logprobs_used"),
                "thinking_type": raw.get("thinking_type", ""),
                "api_usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
            }
        )
    return row


def _load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("annotation_key") or "") for row in read_jsonl(path) if row.get("annotation_key")}


def _write_progress(
    path: Path,
    *,
    args: argparse.Namespace,
    started_at: float,
    n_jobs: int,
    n_pending: int,
    n_completed_initial: int,
    n_written: int,
    n_errors: int,
    usage_totals: dict[str, int],
    top_logprobs_used: dict[str, int],
) -> None:
    save_json(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "split": str(args.split),
            "model": str(args.model),
            "n_jobs": int(n_jobs),
            "n_pending_at_start": int(n_pending),
            "n_completed_initial": int(n_completed_initial),
            "n_written": int(n_written),
            "n_errors": int(n_errors),
            "usage_totals": usage_totals,
            "top_logprobs_used": top_logprobs_used,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        path,
    )


class CounterLike:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}

    def add(self, key: str) -> None:
        self._values[str(key)] = self._values.get(str(key), 0) + 1

    def payload(self) -> dict[str, int]:
        return dict(sorted(self._values.items()))


if __name__ == "__main__":
    main()
