from __future__ import annotations

from sft.data.types import PreparedSample
from sft.infer_common import build_label_decoding_prompt, build_label_scoring_prompt


def _sample(prompt: str, *, prompt_add_special_tokens: bool = False) -> PreparedSample:
    return PreparedSample(
        prompt=prompt,
        target="Label: A",
        prompt_add_special_tokens=prompt_add_special_tokens,
        preserve_prompt_prefix=True,
        gold_id=0,
        gold_label="pants-fire",
        gold_explain="",
    )


def test_build_label_decoding_prompt_strips_trailing_chat_newline() -> None:
    sample = _sample("<|im_start|>assistant\n")

    assert build_label_decoding_prompt(sample, "Label:") == "<|im_start|>assistantLabel:"


def test_build_label_decoding_prompt_preserves_special_token_spacing() -> None:
    sample = _sample("Plain prompt\n", prompt_add_special_tokens=True)

    assert build_label_decoding_prompt(sample, "Label:") == "Plain prompt Label:"


def test_build_label_scoring_prompt_appends_space_prefixed_choice() -> None:
    sample = _sample("<|im_start|>assistant\n")

    assert build_label_scoring_prompt(sample, "Label:", "C") == "<|im_start|>assistantLabel: C"
