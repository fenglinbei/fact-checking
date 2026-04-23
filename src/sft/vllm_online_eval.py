from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from logging import Logger
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from fact_checking.data.constants import LABELS
from fact_checking.utils.logging import init_logger
from sft.data.types import PreparedSample
from sft.eval import summarize_prediction_records
from sft.parser import _parse_label_id
from sft.runtime.adapters import is_peft_model

module_logger = init_logger(__name__)


def online_vllm_eval_enabled(train_cfg: dict) -> bool:
    cfg = train_cfg.get("online_vllm_eval", {}) or {}
    return bool(cfg.get("enabled", False))


def _label_name_from_id(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return LABELS[int(label_id)]
    return "parse_error"


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
            backend=str(cfg.get("backend", "direct_load")),
            load_format=load_format,
            enforce_eager=bool(cfg.get("enforce_eager", True)),
            sleep_after_eval=bool(cfg.get("sleep_after_eval", False)),
            use_tqdm=bool(cfg.get("use_tqdm", True)),
            max_num_seqs=None if max_num_seqs is None else int(max_num_seqs),
        )


class OnlineVLLMEvaluator:
    """Rank-0 vLLM evaluator that reloads in-memory ZeRO-2 weights CPPO-style."""

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
        if self.cfg.backend not in {"direct_load", "load_weights", "cppo"}:
            raise ValueError(
                "sft_train.online_vllm_eval.backend supports 'direct_load', 'load_weights', or 'cppo'."
            )
        self.samples = samples
        self.baseline_cfg = baseline_cfg
        self.logger = logger or module_logger
        self.max_length = int(max_length)
        self._sleeping = False

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "sft_train.online_vllm_eval.enabled=true requires vLLM in this environment."
            ) from exc

        self._SamplingParams = SamplingParams

        llm_kwargs = {
            "model": model_name_or_path,
            "tokenizer": tokenizer_name_or_path,
            "trust_remote_code": True,
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "dtype": self.cfg.dtype,
            "max_model_len": self.max_length,
            "enforce_eager": self.cfg.enforce_eager,
        }
        if self.cfg.sleep_after_eval:
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

    def _build_llm(self, llm_cls: type, llm_kwargs: dict) -> object:
        def build_with_accelerate_patches(kwargs: dict) -> object:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("torch.distributed.get_world_size", return_value=1))
                try:
                    stack.enter_context(
                        patch(
                            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                            return_value=None,
                        )
                    )
                except (AttributeError, ImportError, ModuleNotFoundError):
                    pass
                return llm_cls(**kwargs)

        kwargs_with_device = dict(llm_kwargs)
        kwargs_with_device["device"] = self.cfg.device
        try:
            return build_with_accelerate_patches(kwargs_with_device)
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
            return build_with_accelerate_patches(llm_kwargs)
        finally:
            if previous_device is not None:
                torch.cuda.set_device(previous_device)

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

    def _get_vllm_model(self) -> object:
        llm_engine = getattr(self.llm, "llm_engine", None)
        model_executor = getattr(llm_engine, "model_executor", None)
        if model_executor is None:
            raise RuntimeError("Cannot find vLLM model_executor on the embedded LLM engine.")

        candidates = (
            ("driver_worker", "model_runner", "model"),
            ("driver_worker", "worker", "model_runner", "model"),
        )
        for path in candidates:
            value = model_executor
            for attr in path:
                value = getattr(value, attr, None)
                if value is None:
                    break
            if value is not None and hasattr(value, "load_weights"):
                return value

        raise RuntimeError(
            "Cannot locate vLLM's internal model.load_weights() path. "
            "This CPPO-style evaluator expects vLLM 0.7.x/0.8.x style executors."
        )

    def _iter_state_dict_items(self, state_dict: dict[str, object]):
        for name, tensor in state_dict.items():
            if torch.is_tensor(tensor):
                yield name, tensor.detach()
            else:
                yield name, tensor

    def _prepare_state_dict_for_vllm(self, model: AutoModelForCausalLM) -> tuple[dict[str, object], bool]:
        merged_adapter = False
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if is_peft_model(model):
            if not hasattr(model, "merge_adapter") or not hasattr(model, "unmerge_adapter"):
                raise RuntimeError(
                    "Online vLLM eval with LoRA requires a PEFT model that supports "
                    "merge_adapter() and unmerge_adapter()."
                )
            self.logger.info("[INFO] merging LoRA adapter before loading weights into online vLLM evaluator")
            model.merge_adapter()
            merged_adapter = True
            try:
                adapter_prefix = str(getattr(model, "prefix", "lora_"))
                state_dict = {
                    name.removeprefix("base_model.model.").replace(".base_layer", ""): tensor
                    for name, tensor in model.state_dict().items()
                }
                if adapter_prefix:
                    state_dict = {name: tensor for name, tensor in state_dict.items() if adapter_prefix not in name}
                state_dict = {
                    name.replace("modules_to_save.default.", ""): tensor
                    for name, tensor in state_dict.items()
                    if "original_module" not in name
                }
                return state_dict, merged_adapter
            except Exception:
                model.unmerge_adapter()
                raise

        return dict(model.state_dict()), merged_adapter

    def _unmerge_adapter_if_needed(self, model: AutoModelForCausalLM, merged_adapter: bool) -> None:
        if not merged_adapter:
            return
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.unmerge_adapter()

    def sync_weights(self, model: AutoModelForCausalLM) -> None:
        self._maybe_wake()
        llm_model = self._get_vllm_model()
        state_dict, merged_adapter = self._prepare_state_dict_for_vllm(model)
        try:
            self.logger.info("[INFO] loading %d tensors into online vLLM evaluator", len(state_dict))
            loaded_params = llm_model.load_weights(self._iter_state_dict_items(state_dict))
            with contextlib.suppress(TypeError):
                self.logger.info("[INFO] vLLM load_weights loaded %d tensors", len(loaded_params))
        finally:
            self._unmerge_adapter_if_needed(model, merged_adapter)

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
