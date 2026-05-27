from __future__ import annotations

import time
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fact_checking.data.constants import LETTER_ORDER
from fact_checking.infer.api import (
    OpenAICompletionsClient,
    _choice_payload_prompt_logprobs,
    _extract_final_prompt_logprob,
)
from fact_checking.selectors.verifier_proxy import score_margin
from sft.data.types import PreparedSample
from sft.infer_common import build_label_decoding_prompt, label_choice_text


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60}m"


@dataclass
class VerifierScoreRequest:
    prompt_row: dict[str, Any]
    gold_label: str
    cache_key: str = ""
    event_id: str = ""
    claim: str = ""
    evidence_set_hash: str = ""
    scored_candidate_keys: list[str] = None

    def __post_init__(self) -> None:
        if self.scored_candidate_keys is None:
            self.scored_candidate_keys = []


def build_verifier_scoring_prompts(prompt_row: dict[str, Any], label_prefix: str) -> list[str]:
    sample = PreparedSample(
        prompt=str(prompt_row["prompt"]),
        target=str(prompt_row.get("target") or ""),
        prompt_add_special_tokens=bool(prompt_row.get("prompt_add_special_tokens", False)),
        preserve_prompt_prefix=bool(prompt_row.get("preserve_prompt_prefix", True)),
        gold_id=int(prompt_row.get("gold_id", -1)),
        gold_label=str(prompt_row.get("gold_label") or ""),
        gold_explain=str(prompt_row.get("gold_explain") or ""),
        prompt_token_count=int(prompt_row.get("prompt_token_count") or 0),
        target_token_count=int(prompt_row.get("target_token_count") or 0),
        evidence_count=int(prompt_row.get("evidence_count") or 0),
        was_truncated=bool(prompt_row.get("was_truncated")),
        claim=str(prompt_row.get("claim") or ""),
        no_evidence=int(prompt_row.get("evidence_count") or 0) == 0,
        long_claim=len(str(prompt_row.get("claim") or "").split()) > 64,
    )
    return [
        build_label_decoding_prompt(sample, label_prefix) + label_choice_text(label_prefix, letter)
        for letter in LETTER_ORDER
    ]


def extract_label_logprobs_from_prompt_logprobs_list(
    prompt_logprobs_list: list[Any],
    label_token_ids: dict[str, int],
) -> dict[str, float]:
    if len(prompt_logprobs_list) != len(LETTER_ORDER):
        raise RuntimeError(
            f"Expected {len(LETTER_ORDER)} prompt_logprobs entries, got {len(prompt_logprobs_list)}."
        )
    label_logprobs: dict[str, float] = {}
    for idx, letter in enumerate(LETTER_ORDER):
        label_logprobs[letter] = _extract_final_prompt_logprob(
            prompt_logprobs_list[idx], int(label_token_ids[letter])
        )
    return label_logprobs


def compute_score_from_logprobs(
    label_logprobs: dict[str, float], gold_label: str
) -> dict[str, Any]:
    margin = score_margin(label_logprobs, gold_label)
    return {"label_logprobs": label_logprobs, **margin}


class BaseVerifierScorer(ABC):
    @abstractmethod
    def score_batch(self, requests: list[VerifierScoreRequest]) -> list[dict[str, Any]]:
        ...

    def score_one(self, request: VerifierScoreRequest) -> dict[str, Any]:
        return self.score_batch([request])[0]


class APIVerifierScorer(BaseVerifierScorer):
    def __init__(
        self,
        client: OpenAICompletionsClient,
        *,
        label_token_ids: dict[str, int],
        label_prefix: str = "Label:",
        prompt_logprobs: int = 0,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self._client = client
        self._label_token_ids = label_token_ids
        self._label_prefix = label_prefix
        self._prompt_logprobs = prompt_logprobs
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._max_delay = max_delay

    def score_batch(self, requests: list[VerifierScoreRequest]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for request in requests:
            results.append(self._score_one_with_retries(request))
        return results

    def _score_one_with_retries(self, request: VerifierScoreRequest) -> dict[str, Any]:
        delay = float(self._initial_delay)
        last_error: BaseException | None = None
        for attempt in range(max(int(self._max_retries), 1)):
            try:
                return self._score_one(request)
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                last_error = exc
                if attempt >= max(int(self._max_retries), 1) - 1:
                    break
                time.sleep(delay)
                delay = min(delay * 2.0, float(self._max_delay))
        raise RuntimeError(
            f"Verifier API scoring failed after {self._max_retries} retries: {last_error}"
        ) from last_error

    def _score_one(self, request: VerifierScoreRequest) -> dict[str, Any]:
        prompts = build_verifier_scoring_prompts(request.prompt_row, self._label_prefix)
        data = self._client.complete(
            prompts,
            max_tokens=1,
            temperature=0.0,
            extra_body={"prompt_logprobs": int(self._prompt_logprobs)},
        )
        choices = data.get("choices", [])
        if len(choices) != len(LETTER_ORDER):
            raise RuntimeError(
                f"Verifier API returned {len(choices)} choices for {len(LETTER_ORDER)} labels."
            )
        by_index = {int(choice.get("index", idx)): choice for idx, choice in enumerate(choices)}
        prompt_logprobs_list = [
            _choice_payload_prompt_logprobs(by_index[idx]) for idx in range(len(LETTER_ORDER))
        ]
        label_logprobs = extract_label_logprobs_from_prompt_logprobs_list(
            prompt_logprobs_list, self._label_token_ids
        )
        return compute_score_from_logprobs(label_logprobs, request.gold_label)


class LLMVerifierScorer(BaseVerifierScorer):
    def __init__(
        self,
        llm,
        sampling_params,
        *,
        label_token_ids: dict[str, int],
        label_prefix: str = "Label:",
        lora_request=None,
        prompt_batch_size: int = 6000,
    ) -> None:
        self._llm = llm
        self._sampling_params = sampling_params
        self._label_token_ids = label_token_ids
        self._label_prefix = label_prefix
        self._lora_request = lora_request
        self._prompt_batch_size = int(prompt_batch_size)

    def score_batch(
        self,
        requests: list[VerifierScoreRequest],
        *,
        on_batch_complete: object = None,
    ) -> list[dict[str, Any]]:
        if not requests:
            return []

        n_labels = len(LETTER_ORDER)
        all_prompts: list[str] = []
        for request in requests:
            prompts = build_verifier_scoring_prompts(request.prompt_row, self._label_prefix)
            all_prompts.extend(prompts)

        batch_size = max(self._prompt_batch_size, n_labels)
        batch_size = batch_size - (batch_size % n_labels)
        total_prompt_batches = (len(all_prompts) + batch_size - 1) // batch_size
        reqs_per_batch = batch_size // n_labels

        all_results: list[dict[str, Any]] = []
        total_requests = len(requests)
        t_start = time.monotonic()
        for batch_idx in range(0, len(all_prompts), batch_size):
            batch_num = batch_idx // batch_size + 1
            batch_prompts = all_prompts[batch_idx : batch_idx + batch_size]
            req_start = batch_idx // n_labels
            req_end = min(req_start + reqs_per_batch, total_requests)
            batch_requests = requests[req_start:req_end]
            completed_reqs = req_end

            elapsed = time.monotonic() - t_start
            if batch_num > 1:
                avg_per_batch = elapsed / (batch_num - 1)
                eta = avg_per_batch * (total_prompt_batches - batch_num + 1)
                eta_str = _format_duration(int(eta))
            else:
                eta_str = "..."

            progress_pct = 100.0 * completed_reqs / max(total_requests, 1)
            print(
                f"vLLM batch {batch_num}/{total_prompt_batches} | "
                f"{completed_reqs}/{total_requests} sets ({progress_pct:.1f}%) | "
                f"elapsed {_format_duration(int(elapsed))} | ETA {eta_str}",
                flush=True,
            )
            generate_kwargs: dict[str, Any] = {
                "prompts": batch_prompts,
                "sampling_params": self._sampling_params,
                "use_tqdm": True,
            }
            if self._lora_request is not None:
                generate_kwargs["lora_request"] = self._lora_request
            batch_outputs = self._llm.generate(**generate_kwargs)
            if len(batch_outputs) != len(batch_prompts):
                raise RuntimeError(
                    f"LLM.generate returned {len(batch_outputs)} outputs for {len(batch_prompts)} prompts."
                )

            batch_results: list[dict[str, Any]] = []
            for req_idx, request in enumerate(batch_requests):
                chunk = batch_outputs[req_idx * n_labels : (req_idx + 1) * n_labels]
                prompt_logprobs_list = [output.prompt_logprobs for output in chunk]
                label_logprobs = extract_label_logprobs_from_prompt_logprobs_list(
                    prompt_logprobs_list, self._label_token_ids
                )
                batch_results.append(compute_score_from_logprobs(label_logprobs, request.gold_label))

            if on_batch_complete is not None:
                on_batch_complete(batch_requests, batch_results)
            all_results.extend(batch_results)

        return all_results
