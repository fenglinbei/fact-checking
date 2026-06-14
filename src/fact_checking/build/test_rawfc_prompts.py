from __future__ import annotations

from fact_checking.build.prompts import build_chat_prompt, build_target, build_training_row, build_user_content


class _FakeTokenizer:
    last_kwargs: dict[str, object]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: object,
    ) -> str:
        del tokenize
        self.last_kwargs = dict(kwargs)
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            prompt += "\nassistant:"
        return prompt


class _FakeMistralCommonTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.seen_tokenize_values: list[bool] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        **kwargs: object,
    ) -> dict[str, list[int]]:
        del messages
        if kwargs:
            raise ValueError(f"Unsupported kwargs: {sorted(kwargs)}")
        self.seen_tokenize_values.append(bool(tokenize))
        if not tokenize:
            raise AssertionError("MistralCommon chat templates must be tokenized directly.")
        return {"input_ids": [1, 17, 101, 18, 3, 102, 4]}

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        add_special_tokens: bool = False,
        **kwargs: object,
    ) -> dict[str, list[int]]:
        del truncation, add_special_tokens, kwargs
        return {"input_ids": [1000 + idx for idx, _ in enumerate(str(text).split(), start=1)]}


_FakeMistralCommonTokenizer.__module__ = "transformers.tokenization_mistral_common"


def test_rawfc_prompt_lists_only_three_rawfc_labels() -> None:
    prompt = build_user_content(
        "A test claim.",
        ["Closed evidence."],
        output_mode="label_only",
        label_format="letter",
        label_schema="rawfc3",
    )

    assert "RAWFC label" in prompt
    assert "- A (false):" in prompt
    assert "- B (half):" in prompt
    assert "- C (true):" in prompt
    assert "A-C" in prompt
    assert "pants-fire" not in prompt
    assert "barely-true" not in prompt
    assert "half-true" not in prompt
    assert "mostly-true" not in prompt


def test_rawfc_target_uses_schema_letters() -> None:
    assert build_target({"label_schema": "rawfc3"}, "false", "label_only", "letter") == "Label: A"
    assert build_target({"label_schema": "rawfc3"}, "half", "label_only", "letter") == "Label: B"
    assert build_target({"label_schema": "rawfc3"}, "true", "label_only", "letter") == "Label: C"


def test_chat_prompt_passes_qwen3_enable_thinking_false_to_template() -> None:
    tokenizer = _FakeTokenizer()
    prompt = build_chat_prompt(
        tokenizer,
        "System",
        "User task",
        chat_template={
            "mode": "tokenizer_default",
            "template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
        },
    )

    assert "user: User task" in prompt
    assert "/no_think" not in prompt
    assert tokenizer.last_kwargs == {"enable_thinking": False}
    assert prompt.endswith("assistant:")


def test_mistral_common_build_row_stores_tokenized_chat_prompt() -> None:
    tokenizer = _FakeMistralCommonTokenizer()
    row = build_training_row(
        {
            "event_id": "e1",
            "claim": "A test claim.",
            "label": "false",
            "label_schema": "rawfc3",
            "candidates": [{"text": "Evidence."}],
        },
        tokenizer,
        {
            "auto_length": False,
            "output_mode": "label_only",
            "label_format": "letter",
            "label_schema": "rawfc3",
        },
    )

    assert tokenizer.seen_tokenize_values == [True]
    assert row["prompt_input_ids"] == [1, 17, 101, 18, 3, 102, 4]
    assert row["prompt_token_count"] == 7
