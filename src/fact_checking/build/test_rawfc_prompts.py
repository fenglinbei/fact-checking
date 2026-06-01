from __future__ import annotations

from fact_checking.build.prompts import build_target, build_user_content


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
