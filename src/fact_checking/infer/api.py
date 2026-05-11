from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from fact_checking.data.constants import LABELS
from sft.infer_common import build_inference_context
from sft.logit_adjust import build_logit_bias, build_logit_adjust_cfg_from_train_config, load_logit_adjust_cfg
from sft.metrics import _build_confusion_matrix, _compute_classification_metrics
from sft.parser import _parse_label_id
from sft.runtime.adapters import checkpoint_has_peft_adapter


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


class OpenAICompletionsClient:
    def __init__(self, *, base_url: str, model: str, timeout: float = 120.0, logit_bias: dict[str, float] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.logit_bias = logit_bias

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias
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

    process = _ensure_vllm_server(context=context, infer_cfg=infer_cfg, log_path=log_path / "vllm_server.log")
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
            raw_output = client.generate(sample.prompt, max_tokens=max_tokens, temperature=temperature)
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
        if process is not None and bool(infer_cfg.get("server", {}).get("stop_after_infer", True)):
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


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


def _ensure_vllm_server(*, context, infer_cfg: dict[str, Any], log_path: Path) -> subprocess.Popen | None:
    server_cfg = dict(infer_cfg.get("server", {}) or {})
    manage = bool(server_cfg.get("manage", True))
    if _server_ready(infer_cfg):
        return None
    if not manage:
        raise RuntimeError(f"Inference server is not reachable at {_base_url(infer_cfg)} and infer.server.manage=false.")

    command = _build_vllm_command(context=context, infer_cfg=infer_cfg)
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
    try:
        _wait_for_server(infer_cfg)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return process


def _build_vllm_command(*, context, infer_cfg: dict[str, Any]) -> list[str]:
    checkpoint_dir = context.checkpoint_dir
    served_model_name = str(infer_cfg.get("served_model_name", "fact-checking-sft"))
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

    if checkpoint_has_peft_adapter(checkpoint_dir):
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
