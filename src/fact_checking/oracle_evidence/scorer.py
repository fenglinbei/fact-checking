from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fact_checking.build.candidates import (
    _build_chat_prompt,
    _build_system_message,
    _build_user_content,
)
from fact_checking.data.constants import LABEL_LETTERS, LETTER_ORDER

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def build_label_token_ids(tokenizer: AutoTokenizer, label_prefix: str = "Label:") -> dict[str, int]:
    """Map each label letter (A-F) to its token ID when following ``label_prefix``.

    Constructs ``{label_prefix} {letter}``, tokenizes, and returns the *last*
    token ID.  This is the token the model must predict immediately after
    the prompt ending with ``label_prefix``.
    """
    letter_token_ids: dict[str, int] = {}
    for letter in LETTER_ORDER:
        continuation = f"{label_prefix} {letter}"
        ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if not ids:
            raise RuntimeError(f"Empty tokenization for continuation {continuation!r}")
        letter_token_ids[letter] = int(ids[-1])
    return letter_token_ids


def _extract_prompt_token_logprob(output, token_id: int) -> float:
    """Extract logprob of *token_id* from the last position of prompt_logprobs."""
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if not prompt_logprobs:
        raise RuntimeError("vLLM output missing prompt_logprobs")
    last_logprobs = prompt_logprobs[-1]
    if not last_logprobs:
        raise RuntimeError("Empty prompt logprobs at last position")
    entry = last_logprobs.get(token_id)
    if entry is None:
        entry = last_logprobs.get(str(token_id))
    if entry is None:
        raise RuntimeError(
            f"Token {token_id} not found in last position logprobs. "
            f"Available: {list(last_logprobs.keys())[:20]}"
        )
    if isinstance(entry, float):
        return entry
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    return float(entry)


def _parse_label_from_text(text: str) -> str | None:
    """Extract a single label letter (A-F) from generated text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("LABEL:"):
            parts = stripped.split(":", 1)
            if len(parts) > 1:
                token = parts[1].strip().upper()
                if len(token) >= 1 and token[0] in LETTER_ORDER:
                    return token[0]
    # Fallback: standalone single letter A-F on its own line
    for line in text.splitlines():
        token = line.strip().upper()
        if len(token) == 1 and token in LETTER_ORDER:
            return token
    return None


@dataclass
class VerifierScorer:
    """Scores evidence sets using vLLM offline inference.

    Uses *prompt_logprobs* to extract the log-probability of the correct
    label token, exactly matching the approach in ``compute_oracle_lambda.py``.
    """

    llm: LLM
    tokenizer: AutoTokenizer
    label_token_ids: dict[str, int] = field(default_factory=dict)
    label_prefix: str = "Label:"
    system_prompt: str | None = None
    output_mode: str = "label_only"
    label_format: str = "letter"
    max_prompt_length: int = 1024

    score_sampling_params: SamplingParams | None = None
    gen_sampling_params: SamplingParams | None = None

    def __post_init__(self) -> None:
        if not self.label_token_ids:
            self.label_token_ids = build_label_token_ids(self.tokenizer, self.label_prefix)
        if self.score_sampling_params is None:
            from vllm import SamplingParams

            self.score_sampling_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                prompt_logprobs=0,
                detokenize=False,
            )
        if self.gen_sampling_params is None:
            from vllm import SamplingParams

            self.gen_sampling_params = SamplingParams(
                max_tokens=8,
                temperature=0.0,
                detokenize=True,
            )

    # ------------------------------------------------------------------
    # Prompt construction (matches b3 pipeline exactly)
    # ------------------------------------------------------------------

    def _build_prompt(self, claim: str, evidence_texts: list[str]) -> str:
        """Construct a chat prompt identical to the training pipeline.

        If the prompt exceeds *max_prompt_length* tokens, evidence items are
        popped from the tail until it fits (matching ``_auto_truncate_evidence``
        behaviour but without single-item binary-search truncation).
        """
        system_msg = _build_system_message(self.system_prompt)
        user_content = _build_user_content(
            claim=claim,
            evidence_texts=evidence_texts,
            output_mode=self.output_mode,
            label_format=self.label_format,
        )
        prompt = _build_chat_prompt(self.tokenizer, system_msg, user_content)
        token_count = len(
            self.tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"]
        )

        if token_count <= self.max_prompt_length or not evidence_texts:
            return prompt

        # Pop from tail until prompt fits
        kept = list(evidence_texts)
        while token_count > self.max_prompt_length and len(kept) > 1:
            kept.pop()
            user_content = _build_user_content(
                claim=claim,
                evidence_texts=kept,
                output_mode=self.output_mode,
                label_format=self.label_format,
            )
            prompt = _build_chat_prompt(self.tokenizer, system_msg, user_content)
            token_count = len(
                self.tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"]
            )

        if token_count > self.max_prompt_length:
            logger.warning(
                "Prompt still over budget (%d > %d) with %d evidence items; vLLM may fail",
                token_count, self.max_prompt_length, len(kept),
            )
        return prompt

    def _build_scoring_token_ids(
        self, prompt: str, gold_letter: str
    ) -> list[int]:
        """Tokenize ``prompt + label_prefix + gold_letter``.

        The last token is the gold label token, whose logprob we extract from
        the model's prompt_logprobs at that position.
        """
        continuation = f"{self.label_prefix} {gold_letter}"
        full_text = prompt + continuation
        ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        expected = self.label_token_ids[gold_letter]
        if int(ids[-1]) != int(expected):
            raise RuntimeError(
                f"Last token id {ids[-1]} != expected {expected} for letter {gold_letter}. "
                f"Continuation: {continuation!r}"
            )
        return ids

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_evidence_sets(
        self,
        claims: list[str],
        current_sets: list[list[str]],
        candidate_texts: list[str],
        gold_label_letters: list[str],
    ) -> np.ndarray:
        """Score many (current_set + one candidate) combinations in one batch.

        Args:
            claims: one claim per entry (same length as everything else).
            current_sets: already-selected evidence texts for each entry.
            candidate_texts: one new candidate text per entry.
            gold_label_letters: gold label letter (A-F) per entry.

        Returns:
            Float array of shape [N] with log-probability of correct label.
        """
        n = len(claims)
        assert len(current_sets) == n
        assert len(candidate_texts) == n
        assert len(gold_label_letters) == n

        # Build a prompt for each (claim, current_set + candidate) pair
        prompts: list[str] = []
        for i in range(n):
            evidence = list(current_sets[i]) + [candidate_texts[i]]
            prompts.append(self._build_prompt(claims[i], evidence))

        # Build scoring token_ids: prompt + "Label: X"
        batch_token_ids: list[list[int]] = []
        for i in range(n):
            batch_token_ids.append(
                self._build_scoring_token_ids(prompts[i], gold_label_letters[i])
            )

        # Single vLLM call
        outputs = self.llm.generate(
            prompt_token_ids=batch_token_ids,
            sampling_params=self.score_sampling_params,
        )
        if len(outputs) != n:
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs, expected {n}")

        logprobs = np.empty(n, dtype=np.float32)
        for i, output in enumerate(outputs):
            token_id = self.label_token_ids[gold_label_letters[i]]
            logprobs[i] = _extract_prompt_token_logprob(output, token_id)
        return logprobs

    def score_complete_sets(
        self,
        claims: list[str],
        evidence_sets: list[list[str]],
        gold_label_letters: list[str],
    ) -> np.ndarray:
        """Score arbitrary (already-built) evidence sets in one batch.

        Unlike ``score_evidence_sets`` which appends one candidate to a
        current set, this method scores each evidence set as-is.
        """
        n = len(claims)
        assert len(evidence_sets) == n
        assert len(gold_label_letters) == n

        prompts = [
            self._build_prompt(claims[i], evidence_sets[i])
            for i in range(n)
        ]
        batch_token_ids = [
            self._build_scoring_token_ids(prompts[i], gold_label_letters[i])
            for i in range(n)
        ]
        outputs = self.llm.generate(
            prompt_token_ids=batch_token_ids,
            sampling_params=self.score_sampling_params,
        )
        if len(outputs) != n:
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs, expected {n}")

        logprobs = np.empty(n, dtype=np.float32)
        for i, output in enumerate(outputs):
            token_id = self.label_token_ids[gold_label_letters[i]]
            logprobs[i] = _extract_prompt_token_logprob(output, token_id)
        return logprobs

    # ------------------------------------------------------------------
    # Label prediction (for final evaluation)
    # ------------------------------------------------------------------

    def predict_labels(self, prompts: list[str]) -> list[int]:
        """Generate a label prediction for each prompt. Returns label id (0-5)
        or -1 on parse failure."""
        from fact_checking.data.constants import LETTER2LABEL, LABEL2ID

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.gen_sampling_params,
        )
        label_ids: list[int] = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            letter = _parse_label_from_text(text)
            if letter is None:
                label_ids.append(-1)
                continue
            label_name = LETTER2LABEL.get(letter, "")
            label_ids.append(LABEL2ID.get(label_name, -1))
        return label_ids
