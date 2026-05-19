from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.data.constants import LABELS, LETTER_ORDER
from sft.infer_common import build_inference_context, build_label_decoding_prompt
from sft.logit_adjust import build_logit_bias, build_logit_adjust_cfg_from_train_config, load_logit_adjust_cfg
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics
from sft.parser import _parse_label_id
from sft.runtime.adapters import checkpoint_has_hf_artifacts, checkpoint_has_peft_adapter

logger = logging.getLogger(__name__)


@dataclass
class _VLLMServerHandle:
    process: subprocess.Popen | None = None
    merged_model_dir: Path | None = None
    cleanup_merged_model_dir: bool = False
    managed_pid: int | None = None
    pid_file: Path | None = None


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


_LETTER_LABEL_ONLY_TARGET = re.compile(r"(?is)^\s*label\s*:\s*[A-F]\s*$")


def _is_letter_label_only_task(samples: list[Any]) -> bool:
    if not samples:
        return False
    checked = samples[: min(len(samples), 32)]
    return all(_LETTER_LABEL_ONLY_TARGET.match(str(sample.target)) is not None for sample in checked)


def _optional_float(cfg: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = cfg.get(key, default)
    if value is None:
        return None
    return float(value)


class OpenAICompletionsClient:
    def __init__(self, *, base_url: str, model: str, timeout: float = 120.0, logit_bias: dict[str, float] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.logit_bias = logit_bias

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias
        if extra_body:
            payload.update(extra_body)
        req = urllib.request.Request(
            url=f"{self.base_url}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("text", ""))


def run_api_inference(
    *,
    run_dir: str | Path,
    checkpoint: str,
    split: str,
    config_path: str | Path,
    infer_cfg: dict[str, Any],
    eval_dir: str | Path,
    log_dir: str | Path,
) -> dict[str, str]:
    provider = str(infer_cfg.get("provider", "vllm_openai")).strip().lower()
    if provider != "vllm_openai":
        raise ValueError("infer.provider currently supports 'vllm_openai'.")

    context = build_inference_context(
        run_dir=run_dir,
        checkpoint=checkpoint,
        split=split,
        config_path=str(config_path),
    )
    eval_path = Path(eval_dir)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logit_adjust_cfg = load_logit_adjust_cfg(Path(run_dir))
    if logit_adjust_cfg is None:
        logit_adjust_cfg = build_logit_adjust_cfg_from_train_config(context.cfg, context.tokenizer)
    logit_bias = build_logit_bias(logit_adjust_cfg) if logit_adjust_cfg else None

    server_handle = _ensure_vllm_server(context=context, infer_cfg=infer_cfg, log_path=log_path / "vllm_server.log")
    try:
        client = OpenAICompletionsClient(
            base_url=_base_url(infer_cfg),
            model=str(infer_cfg.get("served_model_name", "fact-checking-sft")),
            logit_bias=logit_bias,
            timeout=float(infer_cfg.get("request_timeout_seconds", 120)),
        )
        max_tokens_value = infer_cfg.get("max_new_tokens")
        if max_tokens_value is None:
            max_tokens_value = context.baseline_cfg.get("max_new_tokens", 24)
        temperature_value = infer_cfg.get("temperature")
        if temperature_value is None:
            temperature_value = context.baseline_cfg.get("temperature", 0.0)
        max_tokens = int(max_tokens_value)
        temperature = float(temperature_value)
        top_p = _optional_float(infer_cfg, "top_p", 1.0)
        presence_penalty = _optional_float(infer_cfg, "presence_penalty", 0.0)
        frequency_penalty = _optional_float(infer_cfg, "frequency_penalty", 0.0)
        repetition_penalty = _optional_float(infer_cfg, "repetition_penalty", 1.0)
        decoding_extra_body = {
            key: value
            for key, value in {
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repetition_penalty": repetition_penalty,
            }.items()
            if value is not None
        }
        label_decoding_cfg = dict(infer_cfg.get("label_decoding", {}) or {})
        use_label_decoding = bool(label_decoding_cfg.get("enabled", True)) and _is_letter_label_only_task(
            context.samples
        )
        label_prefix = str(label_decoding_cfg.get("prefix", "Label:"))
        label_choices = [f" {letter}" for letter in LETTER_ORDER]
        label_extra_body: dict[str, Any] | None = dict(decoding_extra_body) if decoding_extra_body else None
        use_guided_choice = use_label_decoding and bool(label_decoding_cfg.get("guided_choice", True))
        if use_guided_choice:
            label_extra_body = dict(label_extra_body or {})
            label_extra_body["guided_choice"] = label_choices
        label_max_tokens = int(label_decoding_cfg.get("max_tokens", 1))
        logger.info(
            "API inference decoding: label_decoding=%s guided_choice=%s max_tokens=%d logit_bias_tokens=%d "
            "top_p=%s presence_penalty=%s frequency_penalty=%s repetition_penalty=%s",
            use_label_decoding,
            use_guided_choice,
            label_max_tokens if use_label_decoding else max_tokens,
            len(logit_bias or {}),
            top_p,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
        )

        prediction_records: list[dict[str, object]] = []
        correct = 0
        parse_errors = 0
        progress = tqdm(
            context.samples,
            total=len(context.samples),
            desc=f"infer[{split}/{checkpoint}]",
            unit="sample",
            dynamic_ncols=True,
        )
        for sample_idx, sample in enumerate(progress):
            request_prompt = sample.prompt
            request_max_tokens = max_tokens
            extra_body = dict(decoding_extra_body) if decoding_extra_body else None
            if use_label_decoding:
                request_prompt = build_label_decoding_prompt(sample, label_prefix)
                request_max_tokens = label_max_tokens
                extra_body = label_extra_body
            raw_completion = client.generate(
                request_prompt,
                max_tokens=request_max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            raw_output = f"{label_prefix}{raw_completion}" if use_label_decoding else raw_completion
            pred_id = _parse_label_id(raw_output)
            if pred_id == int(sample.gold_id):
                correct += 1
            if pred_id < 0:
                parse_errors += 1
            processed = sample_idx + 1
            progress.set_postfix(
                acc=f"{correct / processed:.3f}",
                parse_err=f"{parse_errors / processed:.3f}",
            )
            prediction_records.append(
                {
                    "sample_idx": sample_idx,
                    "prompt": sample.prompt,
                    "target": sample.target,
                    "raw_output": raw_output,
                    "raw_completion": raw_completion,
                    "pred_id": int(pred_id),
                    "pred_label": _label_name_from_id(int(pred_id)),
                    "gold_id": int(sample.gold_id),
                    "gold_label": sample.gold_label,
                    "gold_explain": sample.gold_explain,
                }
            )
        progress.close()

        eval_metrics = _summarize_prediction_records(prediction_records)
        artifacts = _save_eval_artifacts(
            eval_dir=eval_path,
            metrics=_build_serializable_metrics(eval_metrics),
            confusion_matrix=eval_metrics["confusion_matrix"],
            confusion_labels=eval_metrics["confusion_labels"],
            prediction_records=eval_metrics.get("prediction_records", []),
            predictions_filename=f"{split}_predictions.jsonl",
            title=f"Confusion Matrix ({split}/{checkpoint}, API)",
        )
        return artifacts
    finally:
        _cleanup_vllm_server(server_handle, infer_cfg)


def _base_url(infer_cfg: dict[str, Any]) -> str:
    host = str(infer_cfg.get("host", "127.0.0.1"))
    port = int(infer_cfg.get("port", 8000))
    return str(infer_cfg.get("base_url") or f"http://{host}:{port}/v1")


def _summarize_prediction_records(prediction_records: list[dict[str, object]]) -> dict[str, object]:
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
    metrics["prediction_records"] = sorted(prediction_records, key=lambda record: int(record["sample_idx"]))
    return metrics


def _build_serializable_metrics(eval_metrics: dict[str, object]) -> dict[str, object]:
    prediction_records = eval_metrics.get("prediction_records", [])
    return {
        "num_samples": len(prediction_records) if isinstance(prediction_records, list) else 0,
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
    metrics: dict[str, object],
    confusion_matrix: np.ndarray,
    confusion_labels: list[str],
    prediction_records: list[dict[str, object]],
    predictions_filename: str,
    title: str,
) -> dict[str, str]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    confusion_data_path = eval_dir / "confusion_matrix.json"
    confusion_png_path = eval_dir / "confusion_matrix.png"
    predictions_path = eval_dir / predictions_filename

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    confusion_data_path.write_text(
        json.dumps(
            {
                "gold_labels": LABELS,
                "pred_labels": confusion_labels,
                "matrix": confusion_matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in prediction_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "metrics_path": str(metrics_path),
            "confusion_data_path": str(confusion_data_path),
            "predictions_path": str(predictions_path),
        }

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
    return {
        "metrics_path": str(metrics_path),
        "confusion_data_path": str(confusion_data_path),
        "confusion_png_path": str(confusion_png_path),
        "predictions_path": str(predictions_path),
    }


def _server_ready(infer_cfg: dict[str, Any]) -> bool:
    req = urllib.request.Request(url=f"{_base_url(infer_cfg)}/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def _wait_for_server(infer_cfg: dict[str, Any]) -> None:
    deadline = time.time() + int(infer_cfg.get("wait_seconds", 180))
    while time.time() < deadline:
        if _server_ready(infer_cfg):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for inference server at {_base_url(infer_cfg)}.")


def _adapter_weight_path(adapter_dir: Path) -> Path:
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        return adapter_path
    adapter_path = adapter_dir / "adapter_model.bin"
    if adapter_path.exists():
        return adapter_path
    raise FileNotFoundError(f"LoRA adapter weights not found in {adapter_dir}")


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _merged_lora_cache_key(
    *,
    base_model: str,
    adapter_dir: str | Path,
    dtype: str,
) -> str:
    adapter_dir = Path(adapter_dir)
    adapter_path = _adapter_weight_path(adapter_dir)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"LoRA adapter config not found in {adapter_dir}")
    payload = {
        "version": 1,
        "base_model": str(base_model),
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_weights": _file_signature(adapter_path),
        "adapter_config": _file_signature(adapter_config_path),
        "dtype": str(dtype),
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def _merged_lora_cache_complete(path: Path) -> bool:
    return checkpoint_has_hf_artifacts(path) and (path / "tokenizer_config.json").exists()


def _write_merge_cache_metadata(
    *,
    path: Path,
    base_model: str,
    adapter_dir: str | Path,
    dtype: str,
    cache_key: str,
) -> None:
    metadata = {
        "cache_key": cache_key,
        "base_model": str(base_model),
        "adapter_dir": str(Path(adapter_dir).resolve()),
        "dtype": str(dtype),
        "created_at": int(time.time()),
    }
    (path / "merge_cache.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_lora_to_dir(
    *,
    base_model: str,
    adapter_dir: str | Path,
    tokenizer_dir: str | Path,
    dtype: str,
    output_dir: Path,
) -> Path:
    import safetensors.torch
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    tokenizer_dir = Path(tokenizer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, trust_remote_code=True)

    adapter_path = _adapter_weight_path(adapter_dir)
    adapter_config = LoraConfig.from_pretrained(adapter_dir)
    if adapter_path.suffix == ".safetensors":
        adapter_state = safetensors.torch.load_file(str(adapter_path))
    else:
        adapter_state = torch.load(str(adapter_path), map_location="cpu")
        if isinstance(adapter_state, dict) and isinstance(adapter_state.get("state_dict"), dict):
            adapter_state = adapter_state["state_dict"]
    saved_lora_keys = _filter_lora_state_keys(adapter_state.keys())
    if not saved_lora_keys:
        sample_keys = sorted(str(key) for key in adapter_state.keys())[:10]
        raise RuntimeError(
            f"No LoRA tensors found in adapter checkpoint: {adapter_path}. "
            f"First saved tensor keys: {sample_keys}"
        )

    peft_model = get_peft_model(model, adapter_config)
    common_prefix = _find_adapter_key_prefix(
        adapter_state.keys(),
        (name for name, _ in peft_model.named_parameters()),
    )
    if common_prefix:
        adapter_state = {k.removeprefix(common_prefix): v for k, v in adapter_state.items()}
        saved_lora_keys = _filter_lora_state_keys(adapter_state.keys())
        logger.info("Stripped adapter key prefix %r for %d tensors", common_prefix, len(adapter_state))

    load_result = set_peft_model_state_dict(peft_model, adapter_state)
    missing, unexpected = _unpack_incompatible_keys(load_result)
    missing_lora = _filter_lora_state_keys(missing)
    unexpected_lora = _filter_lora_state_keys(unexpected)
    model_lora_keys = _filter_lora_state_keys(name for name, _ in peft_model.named_parameters())
    logger.info(
        "LoRA adapter load check: checkpoint_lora_tensors=%d model_lora_tensors=%d "
        "missing_lora=%d unexpected_lora=%d non_lora_missing=%d",
        len(saved_lora_keys),
        len(model_lora_keys),
        len(missing_lora),
        len(unexpected_lora),
        len(missing) - len(missing_lora),
    )
    if missing_lora or unexpected_lora:
        raise RuntimeError(
            "Failed to load LoRA adapter tensors: "
            f"missing_lora={missing_lora[:5]} unexpected_lora={unexpected_lora[:5]}"
        )

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir)
    del merged, peft_model

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _merge_lora_to_tmp(
    base_model: str,
    adapter_dir: str | Path,
    tokenizer_dir: str | Path,
    dtype: str = "bfloat16",
) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="merged_lora_"))
    result = _merge_lora_to_dir(
        base_model=base_model,
        adapter_dir=adapter_dir,
        tokenizer_dir=tokenizer_dir,
        dtype=dtype,
        output_dir=tmp,
    )
    logger.info("Merged LoRA adapter %s into temporary HF model %s", adapter_dir, result)
    return result


def _merge_lora_to_cache(
    *,
    base_model: str,
    adapter_dir: str | Path,
    tokenizer_dir: str | Path,
    dtype: str,
    cache_dir: str | Path,
    force_rebuild: bool = False,
) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = _merged_lora_cache_key(base_model=base_model, adapter_dir=adapter_dir, dtype=dtype)
    target = cache_root / cache_key
    if not force_rebuild and _merged_lora_cache_complete(target):
        logger.info("Reusing cached merged LoRA model: %s", target)
        return target

    tmp = Path(tempfile.mkdtemp(prefix=f"{cache_key}.tmp.", dir=cache_root))
    try:
        _merge_lora_to_dir(
            base_model=base_model,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            dtype=dtype,
            output_dir=tmp,
        )
        _write_merge_cache_metadata(
            path=tmp,
            base_model=base_model,
            adapter_dir=adapter_dir,
            dtype=dtype,
            cache_key=cache_key,
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info("Merged LoRA adapter %s into cached HF model %s", adapter_dir, target)
    return target


def _merge_lora_for_inference(
    *,
    base_model: str,
    adapter_dir: str | Path,
    tokenizer_dir: str | Path,
    infer_cfg: dict[str, Any],
) -> tuple[Path, bool]:
    dtype = str(infer_cfg.get("dtype", "bfloat16"))
    cache_cfg = dict(infer_cfg.get("merge_lora_cache", {}) or {})
    if bool(cache_cfg.get("enabled", False)):
        cache_dir = str(cache_cfg.get("dir", "outputs/cache/merged_lora"))
        force_rebuild = bool(cache_cfg.get("force_rebuild", False))
        return (
            _merge_lora_to_cache(
                base_model=base_model,
                adapter_dir=adapter_dir,
                tokenizer_dir=tokenizer_dir,
                dtype=dtype,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
            ),
            False,
        )
    return (
        _merge_lora_to_tmp(
            base_model=base_model,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            dtype=dtype,
        ),
        True,
    )


def _is_lora_state_key(key: str) -> bool:
    return any(marker in key for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."))


def _filter_lora_state_keys(keys) -> list[str]:
    return sorted(str(key) for key in keys if _is_lora_state_key(str(key)))


def _find_adapter_key_prefix(saved_keys, expected_keys) -> str:
    expected: list[str] = []
    for key in expected_keys:
        key = str(key)
        expected.append(key)
        expected.append(_strip_default_adapter_name(key))
    for saved_key in sorted(str(key) for key in saved_keys):
        for expected_key in expected:
            if saved_key.endswith(expected_key):
                return saved_key[: -len(expected_key)]
    return ""


def _strip_default_adapter_name(key: str) -> str:
    for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        key = key.replace(f".{marker}.default.", f".{marker}.")
    return key


def _unpack_incompatible_keys(load_result) -> tuple[list[str], list[str]]:
    if load_result is None:
        return [], []
    missing = getattr(load_result, "missing_keys", None)
    unexpected = getattr(load_result, "unexpected_keys", None)
    if missing is not None and unexpected is not None:
        return list(missing), list(unexpected)
    missing, unexpected = load_result
    return list(missing), list(unexpected)


def _server_pid_file(server_cfg: dict[str, Any]) -> Path | None:
    value = str(server_cfg.get("pid_file", "") or "").strip()
    return Path(value) if value else None


def _write_server_pid_file(pid_file: Path | None, *, pid: int, infer_cfg: dict[str, Any], command: list[str]) -> None:
    if pid_file is None:
        return
    payload = {
        "pid": int(pid),
        "base_url": _base_url(infer_cfg),
        "command": command,
        "created_at": int(time.time()),
    }
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_server_pid_file(pid_file: Path | None) -> int | None:
    if pid_file is None or not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
        if not text:
            return None
        if text.startswith("{"):
            return int(json.loads(text)["pid"])
        return int(text)
    except Exception:
        logger.warning("Could not read vLLM pid file: %s", pid_file)
        return None


def _pid_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
    except OSError:
        return ""


def _pid_matches_vllm_server(pid: int, infer_cfg: dict[str, Any]) -> bool:
    cmdline = _pid_cmdline(pid)
    if "vllm.entrypoints.openai.api_server" not in cmdline:
        return False
    return "--port" in cmdline and str(int(infer_cfg.get("port", 8000))) in cmdline


def _terminate_managed_server(pid: int, infer_cfg: dict[str, Any], pid_file: Path | None) -> None:
    if not _pid_matches_vllm_server(pid, infer_cfg):
        logger.warning("Refusing to terminate pid=%s because it does not look like the managed vLLM server.", pid)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        if pid_file is not None:
            pid_file.unlink(missing_ok=True)
        return
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if pid_file is not None:
                pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if pid_file is not None:
        pid_file.unlink(missing_ok=True)


def _ensure_vllm_server(
    *, context, infer_cfg: dict[str, Any], log_path: Path
) -> _VLLMServerHandle:
    server_cfg = dict(infer_cfg.get("server", {}) or {})
    manage = bool(server_cfg.get("manage", True))
    pid_file = _server_pid_file(server_cfg)
    if _server_ready(infer_cfg):
        return _VLLMServerHandle(managed_pid=_read_server_pid_file(pid_file), pid_file=pid_file)
    if not manage:
        raise RuntimeError(f"Inference server is not reachable at {_base_url(infer_cfg)} and infer.server.manage=false.")

    merged_dir: Path | None = None
    cleanup_merged_dir = False
    if checkpoint_has_peft_adapter(context.checkpoint_dir):
        merged_dir, cleanup_merged_dir = _merge_lora_for_inference(
            base_model=context.model_name_or_path,
            adapter_dir=context.checkpoint_dir,
            tokenizer_dir=context.checkpoint_dir,
            infer_cfg=infer_cfg,
        )

    command = _build_vllm_command(context=context, infer_cfg=infer_cfg, merged_model_dir=merged_dir)
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cuda_devices = str(infer_cfg.get("cuda_visible_devices", "") or "").strip()
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("$ " + " ".join(command) + "\n\n")
    log_file.flush()
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    _write_server_pid_file(pid_file, pid=process.pid, infer_cfg=infer_cfg, command=command)
    try:
        _wait_for_server(infer_cfg)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        if pid_file is not None:
            pid_file.unlink(missing_ok=True)
        if merged_dir is not None and cleanup_merged_dir:
            shutil.rmtree(str(merged_dir), ignore_errors=True)
        raise
    return _VLLMServerHandle(
        process=process,
        merged_model_dir=merged_dir,
        cleanup_merged_model_dir=cleanup_merged_dir,
        pid_file=pid_file,
    )


def _cleanup_vllm_server(handle: _VLLMServerHandle, infer_cfg: dict[str, Any]) -> None:
    stop_after_infer = bool(infer_cfg.get("server", {}).get("stop_after_infer", True))
    stopped_process = False
    if stop_after_infer:
        if handle.process is not None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                handle.process.kill()
            stopped_process = True
            if handle.pid_file is not None:
                handle.pid_file.unlink(missing_ok=True)
        elif handle.managed_pid is not None:
            _terminate_managed_server(handle.managed_pid, infer_cfg, handle.pid_file)
            stopped_process = True

    if handle.merged_model_dir is not None and handle.cleanup_merged_model_dir:
        if stop_after_infer and stopped_process:
            shutil.rmtree(str(handle.merged_model_dir), ignore_errors=True)
        elif not stop_after_infer:
            logger.warning(
                "Keeping temporary merged LoRA model because vLLM server was left running: %s",
                handle.merged_model_dir,
            )


def _build_vllm_command(
    *, context, infer_cfg: dict[str, Any], merged_model_dir: Path | None = None
) -> list[str]:
    checkpoint_dir = context.checkpoint_dir
    served_model_name = str(infer_cfg.get("served_model_name", "fact-checking-sft"))
    if merged_model_dir is not None:
        model_path = str(merged_model_dir)
        tokenizer_path = str(merged_model_dir)
    else:
        model_path = str(checkpoint_dir)
        tokenizer_path = str(checkpoint_dir)
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tokenizer",
        tokenizer_path,
        "--host",
        str(infer_cfg.get("host", "127.0.0.1")),
        "--port",
        str(int(infer_cfg.get("port", 8000))),
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(int(infer_cfg.get("tensor_parallel_size", 1))),
        "--gpu-memory-utilization",
        str(float(infer_cfg.get("gpu_memory_utilization", 0.9))),
        "--dtype",
        str(infer_cfg.get("dtype", "auto")),
    ]
    max_model_len = infer_cfg.get("max_model_len")
    if max_model_len is None:
        max_new_tokens = infer_cfg.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = context.baseline_cfg.get("max_new_tokens", 24)
        max_model_len = int(context.max_length) + int(max_new_tokens)
    command.extend(["--max-model-len", str(int(max_model_len))])

    if merged_model_dir is None and checkpoint_has_peft_adapter(checkpoint_dir):
        model_path = context.model_name_or_path
        tokenizer_path = model_path
        lora_cfg = context.train_cfg.get("lora", {}) or {}
        max_lora_rank = int(lora_cfg.get("r", 16)) if isinstance(lora_cfg, dict) else 16
        command[command.index("--model") + 1] = model_path
        command[command.index("--tokenizer") + 1] = tokenizer_path
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-modules",
                f"{served_model_name}={checkpoint_dir}",
            ]
        )

    extra_args = infer_cfg.get("server", {}).get("extra_args", []) or []
    command.extend(str(arg) for arg in extra_args)
    return command
