from __future__ import annotations

import json

from fact_checking.infer.deepseek import (
    DeepSeekChatClient,
    _build_jobs,
    _build_usage_summary,
    _select_pending_jobs,
    build_deepseek_payload,
    parse_qwen_chat_prompt,
)


QWEN_PROMPT = (
    "<|im_start|>system\n"
    "System text.<|im_end|>\n"
    "<|im_start|>user\n"
    "User text.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _row(event_id: str = "e1") -> dict:
    return {
        "event_id": event_id,
        "prompt": QWEN_PROMPT,
        "target": "Label: A",
        "gold_label": "pants-fire",
        "gold_id": 0,
        "gold_explain": "gold",
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "prompt_token_count": 10,
        "target_token_count": 2,
        "evidence_count": 1,
        "claim": "claim",
    }


def test_parse_qwen_chat_prompt_strips_template_tokens() -> None:
    messages = parse_qwen_chat_prompt(QWEN_PROMPT)
    assert messages == [
        {"role": "system", "content": "System text."},
        {"role": "user", "content": "User text."},
    ]


def test_build_deepseek_payload_no_thinking_includes_temperature() -> None:
    payload = build_deepseek_payload(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "x"}],
        thinking_type="disabled",
        reasoning_effort=None,
        max_tokens=16,
        temperature=0.0,
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 16
    assert "reasoning_effort" not in payload


def test_build_deepseek_payload_thinking_high_omits_temperature() -> None:
    payload = build_deepseek_payload(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "x"}],
        thinking_type="enabled",
        reasoning_effort="high",
        max_tokens=1024,
        temperature=0.0,
    )
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 1024
    assert "temperature" not in payload


def test_client_sends_authorization_and_payload() -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "Label: A"}}]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["authorization"] = req.get_header("Authorization")
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    client = DeepSeekChatClient(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key="secret",
        timeout=7,
        urlopen=fake_urlopen,
    )
    client.chat(
        messages=[{"role": "user", "content": "x"}],
        thinking_type="disabled",
        reasoning_effort=None,
        max_tokens=16,
        temperature=0.0,
    )
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["timeout"] == 7
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["thinking"] == {"type": "disabled"}


def test_resume_does_not_reuse_other_thinking_mode() -> None:
    disabled_jobs = _build_jobs([_row()], model="deepseek-v4-flash", thinking_type="disabled", reasoning_effort="")
    thinking_jobs = _build_jobs([_row()], model="deepseek-v4-flash", thinking_type="enabled", reasoning_effort="high")
    existing = {disabled_jobs[0].request_key: {"request_key": disabled_jobs[0].request_key, "parse_status": "ok"}}

    assert _select_pending_jobs(disabled_jobs, existing, resume=True, force=False, retry_failed=True) == []
    assert _select_pending_jobs(thinking_jobs, existing, resume=True, force=False, retry_failed=True) == thinking_jobs


def test_usage_summary_aggregates_reasoning_and_cache_tokens() -> None:
    records = [
        {
            "parse_status": "ok",
            "api_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        },
        {
            "parse_status": "parse_error",
            "api_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        },
    ]
    summary = _build_usage_summary(records, n_expected=3, n_error_rows=1)
    assert summary["n_expected"] == 3
    assert summary["n_predictions"] == 2
    assert summary["n_success"] == 1
    assert summary["n_parse_errors"] == 1
    assert summary["n_missing_predictions"] == 1
    assert summary["n_error_rows"] == 1
    assert summary["prompt_tokens"]["total"] == 18.0
    assert summary["completion_tokens"]["total"] == 7.0
    assert summary["total_tokens"]["total"] == 25.0
    assert summary["reasoning_tokens_total"] == 3.0
    assert summary["cache_token_totals"]["prompt_tokens_details.cached_tokens"] == 4.0
