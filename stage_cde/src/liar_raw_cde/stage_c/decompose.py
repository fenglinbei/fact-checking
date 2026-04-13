from __future__ import annotations

import json
import re
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer

from liar_raw_cde.utils.text import clean_text, jaccard, word_tokens


_SPLIT_RE = re.compile(
    r"\s+(?:and|but|while|although|because|if|when|since|after|before|whereas|however)\s+",
    flags=re.IGNORECASE,
)


@dataclass
class DecompositionResult:
    subclaims: list[str]
    method: str


class HeuristicClaimDecomposer:
    def __init__(self, max_subclaims: int = 4, min_subclaim_tokens: int = 4) -> None:
        self.max_subclaims = max_subclaims
        self.min_subclaim_tokens = min_subclaim_tokens

    def _filter_parts(self, parts: list[str], claim: str) -> list[str]:
        kept: list[str] = []
        for part in parts:
            part = clean_text(part.strip(" ,;:-"))
            if len(word_tokens(part)) < self.min_subclaim_tokens:
                continue
            if jaccard(part, claim) < 0.15:
                continue
            if all(jaccard(part, old) < 0.85 for old in kept):
                kept.append(part)

        if not kept:
            kept = [claim]
        return kept[: self.max_subclaims]

    def decompose(self, claim: str) -> DecompositionResult:
        claim = clean_text(claim)
        if not claim:
            return DecompositionResult([], "heuristic")

        parts = []
        for chunk in re.split(r"[;:]", claim):
            subparts = _SPLIT_RE.split(chunk)
            parts.extend(subparts)

        parts = self._filter_parts(parts, claim)
        return DecompositionResult(parts, "heuristic")


class HFLocalClaimDecomposer:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 192,
        max_subclaims: int = 4,
        device_map: str = "auto",
        use_vllm: bool = False,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.vllm_llm = None

        if self.use_vllm:
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError(
                    "use_vllm=True but vLLM is not installed. Install with `pip install vllm`."
                ) from e

            self.vllm_llm = LLM(
                model=model_name,
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
        self.max_new_tokens = max_new_tokens
        self.max_subclaims = max_subclaims

    def decompose(self, claim: str) -> DecompositionResult:
        prompt = f"""Decompose the following factual claim into the minimum number of atomic subclaims.
Rules:
- preserve entities, numbers, and time expressions
- do not add new facts
- return a JSON array of strings only
Claim: {claim}
JSON:"""
        if self.use_vllm:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.max_new_tokens,
            )
            outputs = self.vllm_llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            text = prompt + generated_text
        else:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **enc,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        json_start = text.rfind("[")
        json_text = text[json_start:] if json_start >= 0 else "[]"
        try:
            items = json.loads(json_text)
            subclaims = [clean_text(str(x)) for x in items if clean_text(str(x))]
        except Exception:
            subclaims = [clean_text(claim)]
        if not subclaims:
            subclaims = [clean_text(claim)]
        subclaims = subclaims[: self.max_subclaims]
        return DecompositionResult(subclaims, "hf_local")
