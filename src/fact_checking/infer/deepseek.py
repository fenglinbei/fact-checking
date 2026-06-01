from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

from fact_checking.config import load_yaml
from fact_checking.data.constants import LABEL2ID, LABELS
from fact_checking.utils.io import read_jsonl, save_json
from sft.data.types import PreparedSample
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics
from sft.parser import _parse_label_id

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
PROVIDER_VERSION = 1

_QWEN_MESSAGE_PATTERN = re.compile(r"<\|im_start\|>([a-zA-Z_]+)\n(.*?)<\|im_end\|>", re.DOTALL)


class APITransientError(RuntimeError):
    pass


class APINonRetryableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeepSeekJob:
    sample_idx: int
    event_id: str
    row: dict[str, Any]
    sample: PreparedSample
    messages: list[dict[str, str]]
    prompt_hash: str
    request_key: str


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


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _stable_request_key(
    *,
    model: str,
    thinking_type: str,
    reasoning_effort: str,
    event_id: str,
    prompt_hash: str,
) -> str:
    payload = {
        "model": model,
        "thinking_type": thinking_type,
        "reasoning_effort": reasoning_effort,
        "event_id": event_id,
        "prompt_hash": prompt_hash,
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_qwen_chat_prompt(prompt: str) -> list[dict[str, str]]:
    """Convert the saved Qwen chat-template prompt back to Chat API messages."""
    messages: list[dict[str, str]] = []
    for match in _QWEN_MESSAGE_PATTERN.finditer(prompt):
        role = match.group(1).strip().lower()
        if role == "assistant":
            break
        if role in {"system", "user"}:
            messages.append({"role": role, "content": match.group(2).strip()})
    if messages:
        return messages

    cleaned = re.sub(r"<\|im_start\|>assistant\s*$", "", str(prompt).strip())
    cleaned = cleaned.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return [{"role": "user", "content": cleaned.strip()}]


def build_deepseek_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    thinking_type: str,
    reasoning_effort: str | None,
    max_tokens: int,
    temperature: float | None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thinking_type = str(thinking_type or "disabled").strip().lower()
    if thinking_type not in {"enabled", "disabled", "none"}:
        raise ValueError("infer.thinking.type must be 'enabled', 'disabled', or 'none'.")

    payload: dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if thinking_type != "none":
        payload["thinking"] = {"type": thinking_type}
    if thinking_type == "enabled":
        if reasoning_effort:
            payload["reasoning_effort"] = str(reasoning_effort)
    elif temperature is not None:
        payload["temperature"] = float(temperature)
    if extra_body:
        payload.update(extra_body)
    return payload


class DeepSeekChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 120.0,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.urlopen = urlopen or urllib.request.urlopen

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        thinking_type: str,
        reasoning_effort: str | None,
        max_tokens: int,
        temperature: float | None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = build_deepseek_payload(
            model=self.model,
            messages=messages,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
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
            with self.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                raise APITransientError(f"HTTP {exc.code}: {body[:500]}") from exc
            raise APINonRetryableError(f"HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise APITransientError(str(exc)) from exc


def run_deepseek_inference(
    *,
    run_dir: str | Path,
    checkpoint: str,
    split: str,
    config_path: str | Path,
    infer_cfg: dict[str, Any],
    eval_dir: str | Path,
    log_dir: str | Path,
) -> dict[str, str]:
    del run_dir, checkpoint
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    train_cfg = load_yaml(config_path)
    rows = _load_split_rows(train_cfg, split)
    sample_limit = infer_cfg.get("sample_limit")
    if sample_limit is not None:
        rows = rows[: int(sample_limit)]

    model = str(infer_cfg.get("model", DEFAULT_MODEL))
    base_url = str(infer_cfg.get("base_url") or DEFAULT_BASE_URL)
    thinking_cfg = dict(infer_cfg.get("thinking", {}) or {})
    thinking_type = str(thinking_cfg.get("type", infer_cfg.get("thinking_type", "disabled"))).strip().lower()
    reasoning_effort = str(infer_cfg.get("reasoning_effort") or "").strip()
    max_tokens = int(infer_cfg.get("max_tokens", infer_cfg.get("max_new_tokens", 16)))
    temperature_value = infer_cfg.get("temperature", 0.0)
    temperature = None if temperature_value is None else float(temperature_value)
    timeout = float(infer_cfg.get("request_timeout_seconds", infer_cfg.get("timeout", 120.0)))
    max_retries = int(infer_cfg.get("max_retries", 4))
    retry_base_sleep = float(infer_cfg.get("retry_base_sleep", 2.0))
    retry_max_sleep = float(infer_cfg.get("retry_max_sleep", 60.0))
    concurrency = max(int(infer_cfg.get("concurrency", 1)), 1)
    requests_per_minute = int(infer_cfg.get("requests_per_minute", 60))
    resume = bool(infer_cfg.get("resume", True))
    force = bool(infer_cfg.get("force", False))
    retry_failed = bool(infer_cfg.get("retry_failed", True))
    api_key_env = str(infer_cfg.get("api_key_env", "DEEPSEEK_API_KEY"))
    mode_label = str(infer_cfg.get("mode_label") or _mode_label(thinking_type, reasoning_effort))

    jobs = _build_jobs(rows, model=model, thinking_type=thinking_type, reasoning_effort=reasoning_effort)
    predictions_path = eval_path / f"{split}_predictions.jsonl"
    latest_predictions_path = eval_path / f"{split}_predictions_latest.jsonl"
    raw_path = eval_path / "raw_responses.jsonl"
    errors_path = eval_path / "errors.jsonl"
    usage_summary_path = eval_path / "usage_summary.json"
    manifest_path = eval_path / "deepseek_infer_manifest.json"

    existing = _latest_predictions_by_key(predictions_path) if resume and not force else {}
    pending = _select_pending_jobs(jobs, existing, resume=resume, force=force, retry_failed=retry_failed)
    if pending:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"DeepSeek API key is missing. Set ${api_key_env} before running infer.provider=deepseek_chat.")
        client = DeepSeekChatClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        _run_pending_jobs(
            pending,
            client=client,
            predictions_path=predictions_path,
            raw_path=raw_path,
            errors_path=errors_path,
            model=model,
            base_url=base_url,
            mode_label=mode_label,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
            retry_max_sleep=retry_max_sleep,
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
        )
    else:
        logger.info("DeepSeek inference has no pending jobs for split=%s mode=%s.", split, mode_label)

    latest_by_key = _latest_predictions_by_key(predictions_path)
    latest_records = [latest_by_key[job.request_key] for job in jobs if job.request_key in latest_by_key]
    latest_records = sorted(latest_records, key=lambda row: int(row.get("sample_idx", 0)))
    _write_jsonl(latest_predictions_path, latest_records, mode="w")

    eval_metrics = _summarize_prediction_records(latest_records)
    metrics = _build_serializable_metrics(eval_metrics, n_expected=len(jobs))
    artifacts = _save_eval_artifacts(
        eval_dir=eval_path,
        metrics=metrics,
        confusion_matrix=eval_metrics["confusion_matrix"],
        confusion_labels=eval_metrics["confusion_labels"],
        predictions_path=predictions_path,
        latest_predictions_path=latest_predictions_path,
        title=f"Confusion Matrix ({split}, DeepSeek {mode_label})",
    )
    usage_summary = _build_usage_summary(
        latest_records,
        n_expected=len(jobs),
        n_error_rows=_count_jsonl(errors_path),
    )
    save_json(usage_summary, usage_summary_path)

    parse_errors = int(usage_summary["n_parse_errors"])
    status = "completed"
    if len(latest_records) < len(jobs):
        status = "incomplete"
    elif parse_errors > 0:
        status = "completed_with_parse_errors"
    manifest = {
        "status": status,
        "provider": "deepseek_chat",
        "provider_version": PROVIDER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "split": split,
        "mode_label": mode_label,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "thinking_type": thinking_type,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "resume": resume,
        "force": force,
        "retry_failed": retry_failed,
        "n_expected": len(jobs),
        "n_predictions_latest": len(latest_records),
        "n_pending_started": len(pending),
        "artifacts": {**artifacts, "usage_summary_path": str(usage_summary_path), "manifest_path": str(manifest_path)},
    }
    save_json(manifest, manifest_path)

    return {**artifacts, "usage_summary_path": str(usage_summary_path), "manifest_path": str(manifest_path)}


def _load_split_rows(train_cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    data_cfg = dict(train_cfg.get("data", {}) or {})
    split_map = {
        "train": data_cfg.get("train_candidates"),
        "val": data_cfg.get("val_candidates"),
        "test": data_cfg.get("test_candidates"),
    }
    if split not in split_map or not split_map[split]:
        raise ValueError(f"Unsupported split={split!r}; expected one of {sorted(k for k, v in split_map.items() if v)}.")
    return read_jsonl(str(split_map[split]))


def _prepared_sample_from_row(row: dict[str, Any]) -> PreparedSample | None:
    gold_label = str(row.get("gold_label", ""))
    if not gold_label:
        return None
    return PreparedSample(
        prompt=str(row["prompt"]),
        target=str(row["target"]),
        prompt_add_special_tokens=bool(row.get("prompt_add_special_tokens", False)),
        preserve_prompt_prefix=bool(row.get("preserve_prompt_prefix", True)),
        gold_id=int(row.get("gold_id", LABEL2ID.get(gold_label, -1))),
        gold_label=gold_label,
        gold_explain=str(row.get("gold_explain", "")),
        prompt_token_count=int(row.get("prompt_token_count", 0)),
        target_token_count=int(row.get("target_token_count", 0)),
        evidence_count=int(row.get("evidence_count", 0)),
        was_truncated=bool(row.get("was_truncated", False)),
        claim=str(row.get("claim", "")),
        no_evidence=int(row.get("evidence_count", 0)) == 0,
        long_claim=len(str(row.get("claim", "")).split()) > 64,
    )


def _build_jobs(
    rows: list[dict[str, Any]],
    *,
    model: str,
    thinking_type: str,
    reasoning_effort: str,
) -> list[DeepSeekJob]:
    jobs: list[DeepSeekJob] = []
    for row_idx, row in enumerate(rows):
        sample = _prepared_sample_from_row(row)
        if sample is None:
            continue
        event_id = str(row.get("event_id") or row_idx)
        prompt_hash = _sha1_text(sample.prompt)
        request_key = _stable_request_key(
            model=model,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            event_id=event_id,
            prompt_hash=prompt_hash,
        )
        jobs.append(
            DeepSeekJob(
                sample_idx=len(jobs),
                event_id=event_id,
                row=row,
                sample=sample,
                messages=parse_qwen_chat_prompt(sample.prompt),
                prompt_hash=prompt_hash,
                request_key=request_key,
            )
        )
    return jobs


def _mode_label(thinking_type: str, reasoning_effort: str) -> str:
    if thinking_type == "enabled":
        return f"thinking_{reasoning_effort or 'high'}"
    if thinking_type == "disabled":
        return "no_thinking"
    return "thinking_none"


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


def _select_pending_jobs(
    jobs: list[DeepSeekJob],
    existing: dict[str, dict[str, Any]],
    *,
    resume: bool,
    force: bool,
    retry_failed: bool,
) -> list[DeepSeekJob]:
    if force or not resume:
        return list(jobs)
    pending: list[DeepSeekJob] = []
    for job in jobs:
        row = existing.get(job.request_key)
        if row is None:
            pending.append(job)
            continue
        if str(row.get("parse_status") or "") == "ok":
            continue
        if retry_failed:
            pending.append(job)
    return pending


def _run_pending_jobs(
    jobs: list[DeepSeekJob],
    *,
    client: DeepSeekChatClient,
    predictions_path: Path,
    raw_path: Path,
    errors_path: Path,
    model: str,
    base_url: str,
    mode_label: str,
    thinking_type: str,
    reasoning_effort: str,
    max_tokens: int,
    temperature: float | None,
    max_retries: int,
    retry_base_sleep: float,
    retry_max_sleep: float,
    concurrency: int,
    requests_per_minute: int,
) -> None:
    limiter = RateLimiter(requests_per_minute=requests_per_minute)
    run_kwargs = {
        "client": client,
        "limiter": limiter,
        "model": model,
        "base_url": base_url,
        "mode_label": mode_label,
        "thinking_type": thinking_type,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_retries": max_retries,
        "retry_base_sleep": retry_base_sleep,
        "retry_max_sleep": retry_max_sleep,
    }

    with predictions_path.open("a", encoding="utf-8") as pred_fh, raw_path.open("a", encoding="utf-8") as raw_fh, errors_path.open("a", encoding="utf-8") as err_fh:
        if concurrency <= 1:
            iterator = tqdm(jobs, desc=f"deepseek[{mode_label}]", unit="sample", dynamic_ncols=True)
            for job in iterator:
                _write_result(_run_job(job, **run_kwargs), pred_fh=pred_fh, raw_fh=raw_fh, err_fh=err_fh)
            return

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_run_job, job, **run_kwargs) for job in jobs]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"deepseek[{mode_label}]",
                unit="sample",
                dynamic_ncols=True,
            )
            for future in iterator:
                _write_result(future.result(), pred_fh=pred_fh, raw_fh=raw_fh, err_fh=err_fh)


def _run_job(
    job: DeepSeekJob,
    *,
    client: DeepSeekChatClient,
    limiter: RateLimiter,
    model: str,
    base_url: str,
    mode_label: str,
    thinking_type: str,
    reasoning_effort: str,
    max_tokens: int,
    temperature: float | None,
    max_retries: int,
    retry_base_sleep: float,
    retry_max_sleep: float,
) -> dict[str, Any]:
    attempts = max(max_retries, 0) + 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            limiter.wait()
            response = client.chat(
                messages=job.messages,
                thinking_type=thinking_type,
                reasoning_effort=reasoning_effort or None,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            created_at = datetime.now(timezone.utc).isoformat()
            content = _response_content(response)
            reasoning_content = _response_reasoning_content(response)
            usage = dict(response.get("usage") or {})
            pred_id = _parse_label_id(content)
            parse_status = "ok" if pred_id >= 0 else "parse_error"
            prediction = {
                "sample_idx": job.sample_idx,
                "event_id": job.event_id,
                "request_key": job.request_key,
                "prompt_hash": job.prompt_hash,
                "provider": "deepseek_chat",
                "model": model,
                "mode_label": mode_label,
                "thinking_type": thinking_type,
                "reasoning_effort": reasoning_effort,
                "max_tokens": max_tokens,
                "prompt": job.sample.prompt,
                "target": job.sample.target,
                "raw_output": content,
                "raw_completion": content,
                "reasoning_content": reasoning_content,
                "pred_id": int(pred_id),
                "pred_label": _label_name_from_id(int(pred_id)),
                "gold_id": int(job.sample.gold_id),
                "gold_label": job.sample.gold_label,
                "gold_explain": job.sample.gold_explain,
                "parse_status": parse_status,
                "finish_reason": _finish_reason(response),
                "api_usage": usage,
                "attempts": attempt,
                "created_at": created_at,
            }
            raw = {
                "sample_idx": job.sample_idx,
                "event_id": job.event_id,
                "request_key": job.request_key,
                "provider": "deepseek_chat",
                "model": model,
                "base_url": base_url,
                "mode_label": mode_label,
                "thinking_type": thinking_type,
                "reasoning_effort": reasoning_effort,
                "finish_reason": _finish_reason(response),
                "parse_status": parse_status,
                "response": response,
                "response_content_preview": content[:1000],
                "reasoning_content_preview": reasoning_content[:1000],
                "created_at": created_at,
            }
            return {"ok": True, "prediction": prediction, "raw": raw}
        except APITransientError as exc:
            last_error = f"transient: {exc}"
        except APINonRetryableError as exc:
            return {"ok": False, "error": _error_row(job, "terminal_api_error", str(exc), attempt)}
        except Exception as exc:
            last_error = f"unexpected: {type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(min(retry_max_sleep, retry_base_sleep * (2 ** (attempt - 1))))
    return {"ok": False, "error": _error_row(job, "retry_exhausted", last_error, attempts)}


def _write_result(result: dict[str, Any], *, pred_fh, raw_fh, err_fh) -> None:
    if result.get("ok"):
        pred_fh.write(json.dumps(result["prediction"], ensure_ascii=False) + "\n")
        pred_fh.flush()
        raw_fh.write(json.dumps(result["raw"], ensure_ascii=False) + "\n")
        raw_fh.flush()
        return
    err_fh.write(json.dumps(result["error"], ensure_ascii=False) + "\n")
    err_fh.flush()


def _error_row(job: DeepSeekJob, error_type: str, message: str, attempts: int) -> dict[str, Any]:
    return {
        "sample_idx": job.sample_idx,
        "event_id": job.event_id,
        "request_key": job.request_key,
        "error_type": error_type,
        "message": str(message),
        "attempts": int(attempts),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _response_reasoning_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("reasoning_content") or "")


def _finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def _latest_predictions_by_key(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        key = str(row.get("request_key") or "")
        if key:
            out[key] = row
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _summarize_prediction_records(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not prediction_records:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "parse_error_rate": 0.0,
            "per_class": {},
            "confusion_matrix": np.zeros((len(LABELS), len(LABELS) + 1), dtype=np.int64),
            "confusion_labels": LABELS + ["parse_error"],
            "prediction_records": [],
        }
    pred_ids = np.asarray([int(record["pred_id"]) for record in prediction_records], dtype=np.int64)
    gold_ids = np.asarray([int(record["gold_id"]) for record in prediction_records], dtype=np.int64)
    metrics = _compute_classification_metrics(pred_ids, gold_ids)
    confusion_matrix, confusion_labels = _build_confusion_matrix(pred_ids, gold_ids)
    metrics["confusion_matrix"] = confusion_matrix
    metrics["confusion_labels"] = confusion_labels
    metrics["prediction_records"] = prediction_records
    return metrics


def _build_serializable_metrics(eval_metrics: dict[str, Any], *, n_expected: int) -> dict[str, Any]:
    prediction_records = eval_metrics.get("prediction_records", [])
    return {
        "num_samples": len(prediction_records) if isinstance(prediction_records, list) else 0,
        "num_expected": int(n_expected),
        "accuracy": float(eval_metrics["accuracy"]),
        "macro_precision": float(eval_metrics["macro_precision"]),
        "macro_recall": float(eval_metrics["macro_recall"]),
        "macro_f1": float(eval_metrics["macro_f1"]),
        "parse_error_rate": float(eval_metrics["parse_error_rate"]),
        "per_class": eval_metrics.get("per_class", {}),
    }


def _save_eval_artifacts(
    *,
    eval_dir: Path,
    metrics: dict[str, Any],
    confusion_matrix: np.ndarray,
    confusion_labels: list[str],
    predictions_path: Path,
    latest_predictions_path: Path,
    title: str,
) -> dict[str, str]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    confusion_data_path = eval_dir / "confusion_matrix.json"
    confusion_png_path = eval_dir / "confusion_matrix.png"
    save_json(metrics, metrics_path)
    save_json(
        {
            "gold_labels": LABELS,
            "pred_labels": confusion_labels,
            "matrix": confusion_matrix.tolist(),
        },
        confusion_data_path,
    )

    artifacts = {
        "metrics_path": str(metrics_path),
        "confusion_data_path": str(confusion_data_path),
        "predictions_path": str(predictions_path),
        "latest_predictions_path": str(latest_predictions_path),
    }
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return artifacts

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(confusion_labels)))
    ax.set_xticklabels(confusion_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(confusion_png_path, dpi=200)
    plt.close(fig)
    artifacts["confusion_png_path"] = str(confusion_png_path)
    return artifacts


def _flatten_numeric_usage(value: Any, *, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            item_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric_usage(item, prefix=item_key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _build_usage_summary(
    prediction_records: list[dict[str, Any]],
    *,
    n_expected: int,
    n_error_rows: int,
) -> dict[str, Any]:
    totals: dict[str, float] = {}
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    total_tokens: list[float] = []
    for record in prediction_records:
        usage = dict(record.get("api_usage") or {})
        flat = _flatten_numeric_usage(usage)
        for key, value in flat.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        prompt_tokens.append(float(usage.get("prompt_tokens") or 0))
        completion_tokens.append(float(usage.get("completion_tokens") or 0))
        total_tokens.append(float(usage.get("total_tokens") or 0))

    reasoning_tokens = totals.get("completion_tokens_details.reasoning_tokens", totals.get("reasoning_tokens", 0.0))
    cache_token_totals = {
        key: value
        for key, value in totals.items()
        if "cache" in key.lower() or "cached" in key.lower()
    }
    n_parse_errors = sum(1 for row in prediction_records if str(row.get("parse_status") or "") != "ok")
    return {
        "n_expected": int(n_expected),
        "n_predictions": len(prediction_records),
        "n_success": sum(1 for row in prediction_records if str(row.get("parse_status") or "") == "ok"),
        "n_parse_errors": int(n_parse_errors),
        "n_missing_predictions": max(int(n_expected) - len(prediction_records), 0),
        "n_error_rows": int(n_error_rows),
        "prompt_tokens": _stats(prompt_tokens),
        "completion_tokens": _stats(completion_tokens),
        "total_tokens": _stats(total_tokens),
        "reasoning_tokens_total": float(reasoning_tokens),
        "cache_token_totals": cache_token_totals,
        "usage_detail_totals": totals,
    }
