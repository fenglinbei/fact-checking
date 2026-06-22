#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from fact_checking.selectors.evidence_map_selector import (
    ATOM_EVIDENCE_PROMPT_VERSION,
    ATOM_FACTS_ABC_PROMPT_VERSION,
    COMPACT_PROMPT_VERSION,
    DEFAULT_MAX_EVIDENCE_CHARS,
    PROMPT_VERSION,
    EvidenceMapSchemaError,
    build_teacher_messages,
    evidence_map_annotation_key,
    evidence_items_fingerprint,
    mock_evidence_map_for_row,
    parse_evidence_map_content,
)
from fact_checking.utils.io import read_jsonl, save_json


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class EvidenceMapJob:
    event_id: str
    annotation_key: str
    row: dict[str, Any]


class APITransientError(RuntimeError):
    pass


class APINonRetryableError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate event-level claim-atom evidence maps with DeepSeek-compatible API.")
    p.add_argument("--candidate-pool", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--prompt-version", default=PROMPT_VERSION)
    p.add_argument("--max-evidence-chars", type=int, default=None)
    p.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--model", default=os.environ.get("TEACHER_MODEL", DEFAULT_MODEL))
    p.add_argument("--api-key-env", default=os.environ.get("TEACHER_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("TEACHER_MAX_TOKENS", "2048")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TEACHER_TOP_P", "1.0")))
    p.add_argument("--thinking-type", default=os.environ.get("THINKING_TYPE", "disabled"), choices=["disabled", "enabled", "none"])
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("TEACHER_CONCURRENCY", "128")))
    p.add_argument("--requests-per-minute", type=int, default=int(os.environ.get("TEACHER_REQUESTS_PER_MINUTE", "2048")))
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-base-sleep", type=float, default=2.0)
    p.add_argument("--retry-max-sleep", type=float, default=60.0)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--mock-maps", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_evidence_chars is None and str(args.prompt_version) == COMPACT_PROMPT_VERSION:
        args.max_evidence_chars = DEFAULT_MAX_EVIDENCE_CHARS
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / f"deepseek_evidence_map_annotations_{args.split}.jsonl"
    raw_path = out_dir / f"deepseek_evidence_map_raw_responses_{args.split}.jsonl"
    errors_path = out_dir / f"deepseek_evidence_map_errors_{args.split}.jsonl"
    progress_path = out_dir / "deepseek_evidence_map_progress.json"

    rows = read_jsonl(args.candidate_pool)
    if args.sample_limit is not None:
        rows = rows[: int(args.sample_limit)]
    jobs = _build_jobs(rows, model=str(args.model), prompt_version=str(args.prompt_version))
    completed = _completed_keys(annotations_path) if args.resume else set()
    pending = [job for job in jobs if job.annotation_key not in completed]

    if args.mock_maps:
        n_written, n_errors, usage_totals = _write_mock_maps(
            pending,
            args=args,
            annotations_path=annotations_path,
            raw_path=raw_path,
            errors_path=errors_path,
        )
    else:
        api_key = os.environ.get(str(args.api_key_env) or "")
        n_written, n_errors, usage_totals = _run_api_jobs(
            pending,
            args=args,
            api_key=api_key,
            annotations_path=annotations_path,
            raw_path=raw_path,
            errors_path=errors_path,
            progress_path=progress_path,
            started_at=started_at,
            n_jobs=len(jobs),
            n_completed_initial=len(completed),
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
        "prompt_version": str(args.prompt_version),
        "max_evidence_chars": args.max_evidence_chars,
        "timeout": float(args.timeout),
        "max_tokens": int(args.max_tokens),
        "top_p": float(args.top_p),
        "mock_maps": bool(args.mock_maps),
        "thinking_type": str(args.thinking_type),
        "concurrency": max(int(args.concurrency), 1),
        "requests_per_minute": int(args.requests_per_minute),
        "resume": bool(args.resume),
        "n_jobs": len(jobs),
        "n_completed_initial": len(completed),
        "n_pending": len(pending),
        "n_written": int(n_written),
        "n_errors": int(n_errors),
        "usage_totals": usage_totals,
        "outputs": {
            "annotations": str(annotations_path),
            "raw_responses": str(raw_path),
            "errors": str(errors_path),
            "progress": str(progress_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(manifest, out_dir / "evidence_map_teacher_manifest.json")
    save_json(manifest, out_dir / f"evidence_map_teacher_manifest_{args.split}.json")
    save_json(usage_totals, out_dir / "evidence_map_teacher_usage_summary.json")
    save_json(usage_totals, out_dir / f"evidence_map_teacher_usage_summary_{args.split}.json")
    _write_progress(progress_path, manifest)
    print(f"Wrote evidence-map annotations: {annotations_path}")
    print(f"new={n_written} errors={n_errors} skipped_resume={len(completed)} total_jobs={len(jobs)}")


def _build_jobs(rows: list[dict[str, Any]], *, model: str, prompt_version: str) -> list[EvidenceMapJob]:
    jobs: list[EvidenceMapJob] = []
    seen: set[str] = set()
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        key = evidence_map_annotation_key(
            event_id=event_id,
            prompt_version=prompt_version,
            model=model,
            evidence_fingerprint=_annotation_fingerprint(row, prompt_version=prompt_version),
        )
        if key in seen:
            continue
        seen.add(key)
        jobs.append(EvidenceMapJob(event_id=event_id, annotation_key=key, row=dict(row)))
    return jobs


def _annotation_fingerprint(row: dict[str, Any], *, prompt_version: str) -> str:
    if prompt_version not in {COMPACT_PROMPT_VERSION, ATOM_FACTS_ABC_PROMPT_VERSION, ATOM_EVIDENCE_PROMPT_VERSION}:
        return ""
    evidence_fp = evidence_items_fingerprint(row.get("evidence_items") or [])
    if prompt_version != ATOM_EVIDENCE_PROMPT_VERSION:
        return evidence_fp
    atom_fp = _claim_atoms_fingerprint(row.get("claim_atoms") or [])
    return f"{evidence_fp}:{atom_fp}"


def _claim_atoms_fingerprint(atoms: list[dict[str, Any]]) -> str:
    payload = [
        [
            str(atom.get("atom_id") or ""),
            str(atom.get("proposition") or atom.get("text") or ""),
        ]
        for atom in atoms
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _write_mock_maps(
    jobs: list[EvidenceMapJob],
    *,
    args: argparse.Namespace,
    annotations_path: Path,
    raw_path: Path,
    errors_path: Path,
) -> tuple[int, int, dict[str, int]]:
    mode = "a" if args.resume else "w"
    n_written = 0
    n_errors = 0
    with annotations_path.open(mode, encoding="utf-8") as ann_fh, raw_path.open(mode, encoding="utf-8") as raw_fh, errors_path.open(mode, encoding="utf-8") as err_fh:
        for job in jobs:
            created_at = datetime.now(timezone.utc).isoformat()
            try:
                system_prompt, user_prompt = build_teacher_messages(
                    job.row,
                    prompt_version=str(args.prompt_version),
                    max_evidence_chars=args.max_evidence_chars,
                )
                evidence_map = mock_evidence_map_for_row(job.row)
                row = _annotation_row(job, args=args, evidence_map=evidence_map, created_at=created_at, usage={})
                ann_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_fh.write(
                    json.dumps(
                        {
                            "annotation_key": job.annotation_key,
                            "event_id": job.event_id,
                            "model": str(args.model),
                            "prompt_version": str(args.prompt_version),
                            "max_evidence_chars": args.max_evidence_chars,
                            "evidence_items_fingerprint": str(job.row.get("evidence_items_fingerprint") or ""),
                            "mock_maps": True,
                            "thinking_type": str(args.thinking_type),
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "response_content_preview": json.dumps(evidence_map, ensure_ascii=False)[:1000],
                            "created_at": created_at,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1
            except Exception as exc:
                err_fh.write(json.dumps(_error_row(job, "mock_generation_error", str(exc), 1), ensure_ascii=False) + "\n")
                n_errors += 1
    return n_written, n_errors, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _run_api_jobs(
    jobs: list[EvidenceMapJob],
    *,
    args: argparse.Namespace,
    api_key: str | None,
    annotations_path: Path,
    raw_path: Path,
    errors_path: Path,
    progress_path: Path,
    started_at: float,
    n_jobs: int,
    n_completed_initial: int,
) -> tuple[int, int, dict[str, int]]:
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    n_written = 0
    n_errors = 0
    limiter = RateLimiter(requests_per_minute=int(args.requests_per_minute))
    mode = "a" if args.resume else "w"
    with annotations_path.open(mode, encoding="utf-8") as ann_fh, raw_path.open(mode, encoding="utf-8") as raw_fh, errors_path.open(mode, encoding="utf-8") as err_fh:
        with ThreadPoolExecutor(max_workers=max(int(args.concurrency), 1)) as executor:
            futures = [executor.submit(_run_job, job, args=args, api_key=api_key, limiter=limiter) for job in jobs]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"evidence-map teacher [{args.split}]",
                unit="claim",
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
                else:
                    err_fh.write(json.dumps(result["error"], ensure_ascii=False) + "\n")
                    err_fh.flush()
                    n_errors += 1
                _write_progress(
                    progress_path,
                    {
                        "status": "running",
                        "started_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
                        "n_jobs": n_jobs,
                        "n_completed_initial": n_completed_initial,
                        "n_pending": len(jobs),
                        "n_written": n_written,
                        "n_errors": n_errors,
                        "usage_totals": usage_totals,
                    },
                )
    return n_written, n_errors, usage_totals


def _run_job(job: EvidenceMapJob, *, args: argparse.Namespace, api_key: str | None, limiter: "RateLimiter") -> dict[str, Any]:
    system_prompt, user_prompt = build_teacher_messages(
        job.row,
        prompt_version=str(args.prompt_version),
        max_evidence_chars=args.max_evidence_chars,
    )
    attempts = max(int(args.max_retries), 0) + 1
    last_error = ""
    last_raw: dict[str, Any] | None = None
    attempts_made = 0
    for attempt in range(1, attempts + 1):
        attempts_made = attempt
        try:
            limiter.wait()
            data = _chat_completion(args=args, api_key=api_key, system_prompt=system_prompt, user_prompt=user_prompt)
            created_at = datetime.now(timezone.utc).isoformat()
            content = _response_content(data)
            valid_ids = [str(item.get("evidence_id") or "") for item in job.row.get("evidence_items") or []]
            raw = _raw_row(job, args=args, system_prompt=system_prompt, user_prompt=user_prompt, response=data, content=content, created_at=created_at)
            last_raw = raw
            try:
                evidence_map = parse_evidence_map_content(
                    content,
                    valid_evidence_ids=valid_ids,
                    claim_atoms=job.row.get("claim_atoms") or None,
                )
            except EvidenceMapSchemaError as exc:
                raw["parse_status"] = "schema_validation_failed"
                raw["parse_error"] = str(exc)
                raise
            raw["parse_status"] = "ok"
            usage = dict(data.get("usage") or {})
            return {
                "ok": True,
                "annotation": _annotation_row(job, args=args, evidence_map=evidence_map, created_at=created_at, usage=usage),
                "raw": raw,
            }
        except EvidenceMapSchemaError as exc:
            last_error = f"schema_validation: {exc}"
            if attempt >= min(2, attempts):
                break
        except APITransientError as exc:
            last_error = f"transient: {exc}"
        except APINonRetryableError as exc:
            return {"ok": False, "error": _error_row(job, "terminal_api_error", str(exc), attempt, raw=last_raw), "raw": last_raw}
        except Exception as exc:
            last_error = f"unexpected: {type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(min(float(args.retry_max_sleep), float(args.retry_base_sleep) * (2 ** (attempt - 1))))
    return {"ok": False, "error": _error_row(job, "retry_exhausted", last_error, attempts_made, raw=last_raw), "raw": last_raw}


def _chat_completion(*, args: argparse.Namespace, api_key: str | None, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    payload = {
        "model": str(args.model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "user": str(args.prompt_version),
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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {429, 500, 502, 503, 504}:
            raise APITransientError(f"HTTP {exc.code}: {body[:500]}") from exc
        raise APINonRetryableError(f"HTTP {exc.code}: {body[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise APITransientError(str(exc)) from exc


class RateLimiter:
    def __init__(self, *, requests_per_minute: int) -> None:
        self.requests_per_minute = max(int(requests_per_minute), 1)
        self._request_times: list[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                cutoff = now - 60.0
                self._request_times = [stamp for stamp in self._request_times if stamp >= cutoff]
                if len(self._request_times) < self.requests_per_minute:
                    self._request_times.append(now)
                    return
                sleep_for = max(0.25, 60.0 - (now - min(self._request_times)))
            time.sleep(sleep_for)


def _annotation_row(job: EvidenceMapJob, *, args: argparse.Namespace, evidence_map: dict[str, Any], created_at: str, usage: dict[str, Any]) -> dict[str, Any]:
    row = {
        "annotation_key": job.annotation_key,
        "event_id": job.event_id,
        "prompt_version": str(args.prompt_version),
        "max_evidence_chars": args.max_evidence_chars,
        "evidence_items_fingerprint": str(job.row.get("evidence_items_fingerprint") or ""),
        "model": str(args.model),
        "evidence_map": evidence_map,
        "api_usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        "mock_maps": bool(args.mock_maps),
        "created_at": created_at,
    }
    if "candidate_atom_alignments" in evidence_map:
        row["candidate_atom_alignments"] = evidence_map.get("candidate_atom_alignments") or []
    if "candidate_alignments" in evidence_map:
        row["candidate_alignments"] = evidence_map.get("candidate_alignments") or []
    return row


def _raw_row(job: EvidenceMapJob, *, args: argparse.Namespace, system_prompt: str, user_prompt: str, response: dict[str, Any], content: str, created_at: str) -> dict[str, Any]:
    return {
        "annotation_key": job.annotation_key,
        "event_id": job.event_id,
        "model": str(args.model),
        "prompt_version": str(args.prompt_version),
        "max_evidence_chars": args.max_evidence_chars,
        "evidence_items_fingerprint": str(job.row.get("evidence_items_fingerprint") or ""),
        "thinking_type": str(args.thinking_type),
        "finish_reason": _finish_reason(response),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "response_content_preview": content[:1000],
        "created_at": created_at,
    }


def _error_row(job: EvidenceMapJob, error_type: str, message: str, attempts: int, *, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "annotation_key": job.annotation_key,
        "event_id": job.event_id,
        "error_type": error_type,
        "message": str(message),
        "attempts": int(attempts),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw:
        row.update(
            {
                "finish_reason": raw.get("finish_reason", ""),
                "parse_status": raw.get("parse_status", ""),
                "parse_error": raw.get("parse_error", ""),
                "response_content_preview": raw.get("response_content_preview", ""),
            }
        )
    return row


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("annotation_key") or "") for row in read_jsonl(path) if row.get("annotation_key")}


def _response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise EvidenceMapSchemaError("API response has no choices.")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    save_json(payload, path)


if __name__ == "__main__":
    main()
