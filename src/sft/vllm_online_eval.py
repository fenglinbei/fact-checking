from __future__ import annotations

import contextlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Iterable

import torch
from transformers import AutoModelForCausalLM

from fact_checking.data.constants import LABELS
from fact_checking.utils.logging import init_logger
from sft.data.types import PreparedSample
from sft.eval import summarize_prediction_records
from sft.parser import _parse_label_id
from sft.runtime.adapters import lora_enabled

module_logger = init_logger(__name__)


def online_vllm_eval_enabled(train_cfg: dict) -> bool:
    cfg = train_cfg.get("online_vllm_eval", {}) or {}
    return bool(cfg.get("enabled", False))


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).split(".")[-1]


def _parse_cuda_device_index(device: str) -> int | None:
    match = re.fullmatch(r"\s*cuda(?::(\d+))?\s*", device)
    if not match:
        return None
    value = match.group(1)
    return 0 if value is None else int(value)


@dataclass
class OnlineVLLMEvalConfig:
    enabled: bool
    device: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    backend: str
    load_format: str | None
    enforce_eager: bool
    packed: bool
    sleep_before_sync: bool
    sleep_after_eval: bool
    use_tqdm: bool
    max_num_seqs: int | None

    @classmethod
    def from_train_cfg(cls, train_cfg: dict) -> "OnlineVLLMEvalConfig":
        cfg = train_cfg.get("online_vllm_eval", {}) or {}
        load_format = cfg.get("load_format", "dummy")
        if load_format is not None:
            load_format = str(load_format).strip() or None
        max_num_seqs = cfg.get("max_num_seqs")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            device=str(cfg.get("device", "cuda:3")),
            tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)),
            dtype=str(cfg.get("dtype", "bfloat16")),
            backend=str(cfg.get("backend", "nccl")),
            load_format=load_format,
            enforce_eager=bool(cfg.get("enforce_eager", True)),
            packed=bool(cfg.get("packed", True)),
            sleep_before_sync=bool(cfg.get("sleep_before_sync", True)),
            sleep_after_eval=bool(cfg.get("sleep_after_eval", True)),
            use_tqdm=bool(cfg.get("use_tqdm", True)),
            max_num_seqs=None if max_num_seqs is None else int(max_num_seqs),
        )


class OnlineVLLMEvaluator:
    """Rank-0 vLLM evaluator with NCCL weight sync from a ZeRO-2 trainer."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        samples: list[PreparedSample],
        max_length: int,
        baseline_cfg: dict,
        train_cfg: dict,
        logger: Logger | None = None,
    ) -> None:
        self.cfg = OnlineVLLMEvalConfig.from_train_cfg(train_cfg)
        if self.cfg.backend != "nccl":
            raise ValueError("sft_train.online_vllm_eval.backend currently supports only 'nccl'.")
        if lora_enabled(train_cfg):
            raise ValueError(
                "Online vLLM eval currently supports full fine-tuning only. "
                "Disable sft_train.lora.enabled or use offline vLLM LoRA eval."
            )

        self.samples = samples
        self.baseline_cfg = baseline_cfg
        self.logger = logger or module_logger
        self.max_length = int(max_length)
        self._sleeping = False

        try:
            from vllm import LLM, SamplingParams
            from vllm.config import WeightTransferConfig
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLTrainerSendWeightsArgs,
                NCCLWeightTransferEngine,
            )
            from vllm.utils.network_utils import get_ip, get_open_port
        except ImportError as exc:
            raise RuntimeError(
                "sft_train.online_vllm_eval.enabled=true requires vLLM with NCCL weight transfer support."
            ) from exc

        self._SamplingParams = SamplingParams
        self._NCCLTrainerSendWeightsArgs = NCCLTrainerSendWeightsArgs
        self._NCCLWeightTransferEngine = NCCLWeightTransferEngine
        self._get_ip = get_ip
        self._get_open_port = get_open_port

        llm_kwargs = {
            "model": model_name_or_path,
            "tokenizer": tokenizer_name_or_path,
            "trust_remote_code": True,
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "dtype": self.cfg.dtype,
            "max_model_len": self.max_length,
            "enforce_eager": self.cfg.enforce_eager,
            "weight_transfer_config": WeightTransferConfig(backend=self.cfg.backend),
        }
        if self.cfg.sleep_before_sync or self.cfg.sleep_after_eval:
            llm_kwargs["enable_sleep_mode"] = True
        if self.cfg.load_format is not None:
            llm_kwargs["load_format"] = self.cfg.load_format
        if self.cfg.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = self.cfg.max_num_seqs

        self.logger.info(
            "[INFO] initializing online vLLM evaluator on %s (tp=%d, backend=%s, load_format=%s)",
            self.cfg.device,
            self.cfg.tensor_parallel_size,
            self.cfg.backend,
            self.cfg.load_format,
        )
        self.llm = self._build_llm(LLM, llm_kwargs)
        self.model_update_group = None
        self._init_weight_transfer_group()

    def _build_llm(self, llm_cls: type, llm_kwargs: dict) -> object:
        kwargs_with_device = dict(llm_kwargs)
        kwargs_with_device["device"] = self.cfg.device
        try:
            return llm_cls(**kwargs_with_device)
        except TypeError as exc:
            if "device" not in str(exc):
                raise

        device_index = _parse_cuda_device_index(self.cfg.device)
        if device_index is None:
            raise RuntimeError(
                f"vLLM in this environment does not accept device=..., and {self.cfg.device!r} "
                "is not a CUDA device string that can be selected via torch.cuda.set_device()."
            )

        previous_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(device_index)
            self.logger.warning(
                "[WARN] vLLM LLM(...) did not accept device=; falling back to torch.cuda.set_device(%d) during init.",
                device_index,
            )
            return llm_cls(**llm_kwargs)
        finally:
            if previous_device is not None:
                torch.cuda.set_device(previous_device)

    def _run_weight_transfer_pair(
        self,
        inference_fn: Callable[[], object],
        trainer_fn: Callable[[], object],
    ) -> tuple[object, object]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            inference_future = executor.submit(inference_fn)
            trainer_future = executor.submit(trainer_fn)
            trainer_result = trainer_future.result()
            inference_result = inference_future.result()
        return trainer_result, inference_result

    def _init_weight_transfer_group(self) -> None:
        master_address = self._get_ip()
        master_port = self._get_open_port()
        world_size = int(self.llm.get_world_size()) + 1
        init_request = {
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": world_size,
            }
        }
        trainer_init = {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
        }
        self.logger.info(
            "[INFO] initializing online vLLM NCCL weight-transfer group at %s:%s (world_size=%d)",
            master_address,
            master_port,
            world_size,
        )
        trainer_result, _ = self._run_weight_transfer_pair(
            inference_fn=lambda: self.llm.init_weight_transfer_engine(init_request),
            trainer_fn=lambda: self._NCCLWeightTransferEngine.trainer_init(trainer_init),
        )
        self.model_update_group = trainer_result

    def _maybe_sleep(self) -> None:
        if self._sleeping or not hasattr(self.llm, "sleep"):
            return
        self.llm.sleep(level=0)
        self._sleeping = True

    def _maybe_wake(self) -> None:
        if not self._sleeping or not hasattr(self.llm, "wake_up"):
            return
        try:
            self.llm.wake_up(tags=["scheduling"])
        except TypeError:
            self.llm.wake_up()
        self._sleeping = False

    def _named_parameters(self, model: AutoModelForCausalLM) -> Iterable[tuple[str, torch.Tensor]]:
        for name, param in model.named_parameters():
            yield name, param.detach()

    def _weight_metadata(self, model: AutoModelForCausalLM) -> tuple[list[str], list[str], list[list[int]]]:
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, param in self._named_parameters(model):
            names.append(name)
            dtype_names.append(_dtype_name(param))
            shapes.append(list(param.shape))
        return names, dtype_names, shapes

    def sync_weights(self, model: AutoModelForCausalLM) -> None:
        if self.model_update_group is None:
            raise RuntimeError("Online vLLM weight-transfer group has not been initialized.")

        if self.cfg.sleep_before_sync:
            self._maybe_sleep()

        names, dtype_names, shapes = self._weight_metadata(model)
        update_request = {
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": self.cfg.packed,
            }
        }
        trainer_args = self._NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=self.cfg.packed,
        )
        self.logger.info("[INFO] syncing %d tensors to online vLLM evaluator", len(names))
        self._run_weight_transfer_pair(
            inference_fn=lambda: self.llm.update_weights(update_request),
            trainer_fn=lambda: self._NCCLWeightTransferEngine.trainer_send_weights(
                iterator=self._named_parameters(model),
                trainer_args=trainer_args,
            ),
        )
        self._maybe_wake()

    def evaluate(
        self,
        *,
        model: AutoModelForCausalLM,
        max_new_tokens: int,
        log_predictions_limit: int,
    ) -> dict[str, object]:
        self.sync_weights(model)
        sampling_params = self._SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(self.baseline_cfg.get("temperature", 0.0)),
        )
        outputs = self.llm.generate(
            prompts=[sample.prompt for sample in self.samples],
            sampling_params=sampling_params,
            use_tqdm=self.cfg.use_tqdm,
        )

        prediction_records: list[dict[str, object]] = []
        for sample_idx, (sample, output) in enumerate(zip(self.samples, outputs)):
            raw_output = output.outputs[0].text if output.outputs else ""
            pred_id = _parse_label_id(raw_output)
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

        if self.cfg.sleep_after_eval:
            self._maybe_sleep()

        return summarize_prediction_records(
            prediction_records,
            eval_logger=self.logger,
            log_predictions_limit=int(log_predictions_limit),
        )

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            if hasattr(self.llm, "shutdown"):
                self.llm.shutdown()
