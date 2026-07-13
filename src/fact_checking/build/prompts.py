"""Prompt construction helpers for the build pipeline."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from fact_checking.data.constants import (
    COVERAGE_LABEL_DEFINITIONS,
    COVERAGE_LABEL_LETTERS,
    label2id_for_schema,
    label_definitions_for_schema,
    label_letters_for_schema,
    labels_for_schema,
    task_name_for_schema,
)
from sft.runtime.model_loading import is_mistral_common_tokenizer, load_compatible_tokenizer
from sft.data.labels import normalize_gold_label

def default_system_prompt(label_schema: str | None = None) -> str:
    task_name = task_name_for_schema(label_schema)
    return (
        f"You are a careful fact-checking assistant for {task_name} claims. "
        "Classify claims using only the claim and retrieved evidence supplied by the user."
    )


def load_prompt_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    return load_compatible_tokenizer(model_name_or_path, trust_remote_code=True)


def count_tokens(text: str, tokenizer: AutoTokenizer, *, add_special_tokens: bool = False) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=add_special_tokens)["input_ids"])


def count_target_tokens(target: str, tokenizer: AutoTokenizer) -> int:
    ids = tokenizer(target.strip(), truncation=False, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return len(ids)


def build_system_message(system_prompt: str | None, label_schema: str | None = None) -> str:
    if system_prompt and str(system_prompt).strip():
        return str(system_prompt).strip()
    return default_system_prompt(label_schema)


def format_evidence_block(evidence_texts: list[str]) -> str:
    lines = [f"[{i}] {text}" for i, text in enumerate(evidence_texts, start=1)]
    return "\n".join(lines)


def label_definitions_text(label_format: str = "name", label_schema: str | None = None) -> str:
    label_definitions = label_definitions_for_schema(label_schema)
    label_letters = label_letters_for_schema(label_schema)
    if label_format == "letter":
        return "\n".join(
            f"- {label_letters[label]} ({label}): {label_definitions[label]}"
            for label in label_definitions
        )
    return "\n".join(f"- {label}: {label_definitions[label]}" for label in label_definitions)


def coverage_label_definitions_text(label_format: str = "name") -> str:
    if label_format == "letter":
        return "\n".join(
            f"- {COVERAGE_LABEL_LETTERS[label]} ({label}): {COVERAGE_LABEL_DEFINITIONS[label]}"
            for label in COVERAGE_LABEL_DEFINITIONS
        )
    return "\n".join(f"- {label}: {COVERAGE_LABEL_DEFINITIONS[label]}" for label in COVERAGE_LABEL_DEFINITIONS)


def build_user_content(
    claim: str,
    evidence_texts: list[str],
    output_mode: str,
    label_format: str = "name",
    label_schema: str | None = None,
) -> str:
    evidence_block = format_evidence_block(evidence_texts)
    evidence_display = evidence_block.strip() if evidence_block.strip() else "(no evidence available)"

    label_letters = list(label_letters_for_schema(label_schema).values())
    label_range = f"{label_letters[0]}-{label_letters[-1]}" if len(label_letters) > 1 else label_letters[0]
    label_placeholder = f"<a single letter from {label_range}>" if label_format == "letter" else "<label>"
    task_name = task_name_for_schema(label_schema)

    if output_mode == "explanation_label":
        return (
            f"Classify the claim into exactly one {task_name} label and provide a concise evidence-grounded explanation.\n\n"
            "Labels:\n"
            f"{label_definitions_text(label_format, label_schema)}\n\n"
            "Rules:\n"
            "- Use the retrieved evidence as the primary source.\n"
            "- Do not invent facts not supported by the evidence.\n"
            "- Keep the explanation brief and evidence-grounded.\n"
            "- Respond with exactly two lines in this format:\n"
            "Explanation: <brief explanation>\n"
            f"Label: {label_placeholder}\n\n"
            f"Claim:\n{claim.strip()}\n\n"
            f"Evidence:\n{evidence_display}"
        )
    if output_mode == "label_with_coverage":
        coverage_letters = list(COVERAGE_LABEL_LETTERS.values())
        coverage_range = f"{coverage_letters[0]}-{coverage_letters[-1]}"
        coverage_placeholder = (
            f"<a single letter from {coverage_range}>" if label_format == "letter" else "<coverage label>"
        )
        return (
            f"Classify the claim into exactly one {task_name} label and one evidence coverage label.\n\n"
            "Labels:\n"
            f"{label_definitions_text(label_format, label_schema)}\n\n"
            "Evidence coverage labels:\n"
            f"{coverage_label_definitions_text(label_format)}\n\n"
            "Rules:\n"
            "- Use the retrieved evidence as the primary source.\n"
            "- Do not invent facts not supported by the evidence.\n"
            "- Coverage means whether the retrieved evidence is sufficient to judge the claim, not whether the claim is true.\n"
            "- Respond with exactly two lines in this format:\n"
            f"Label: {label_placeholder}\n"
            f"Coverage: {coverage_placeholder}\n\n"
            f"Claim:\n{claim.strip()}\n\n"
            f"Evidence:\n{evidence_display}"
        )
    return (
        f"Classify the claim into exactly one {task_name} label.\n\n"
        "Labels:\n"
        f"{label_definitions_text(label_format, label_schema)}\n\n"
        "Rules:\n"
        "- Use the retrieved evidence as the primary source.\n"
        "- Do not invent facts not supported by the evidence.\n"
        f"- Respond with exactly one line: Label: {label_placeholder}\n\n"
        f"Claim:\n{claim.strip()}\n\n"
        f"Evidence:\n{evidence_display}"
    )


def _chat_template_cfg(chat_template: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(chat_template or {})
    mode = str(cfg.get("mode") or "tokenizer_default").strip().lower()
    if mode != "tokenizer_default":
        raise ValueError(f"Unsupported build.prompt.chat_template.mode={mode!r}; use tokenizer_default.")
    return cfg


def _append_user_suffix(user_content: str, suffix: str) -> str:
    suffix = str(suffix or "").strip()
    if not suffix:
        return user_content
    stripped = user_content.rstrip()
    if stripped.endswith(suffix):
        return stripped
    return f"{stripped}\n\n{suffix}"


def _uses_tokenized_chat_template(tokenizer: AutoTokenizer) -> bool:
    return is_mistral_common_tokenizer(tokenizer)


_MISTRAL_COMMON_CHAT_TEMPLATE_KWARGS = {
    "tools",
    "continue_final_message",
    "padding",
    "truncation",
    "max_length",
    "return_tensors",
    "return_dict",
}


def _mistral_common_template_kwargs(template_kwargs: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(set(template_kwargs) - _MISTRAL_COMMON_CHAT_TEMPLATE_KWARGS)
    if unsupported:
        raise ValueError(
            "Unsupported MistralCommon chat template kwargs: "
            f"{unsupported}. Supported: {sorted(_MISTRAL_COMMON_CHAT_TEMPLATE_KWARGS)}"
        )
    return dict(template_kwargs)


def _as_input_ids(encoded: Any) -> list[int]:
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids, list) and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError(f"Expected a single chat prompt, got {len(input_ids)} tokenized prompts.")
        input_ids = input_ids[0]
    if not isinstance(input_ids, list):
        raise TypeError(f"Expected tokenized chat template input_ids as a list, got {type(input_ids).__name__}.")
    return [int(token_id) for token_id in input_ids]


def _readable_chat_prompt(system_msg: str, user_content: str, *, add_generation_prompt: bool) -> str:
    prompt = f"System:\n{system_msg}\n\nUser:\n{user_content}"
    if add_generation_prompt:
        prompt += "\n\nAssistant:\n"
    return prompt


def build_chat_prompt_with_input_ids(
    tokenizer: AutoTokenizer,
    system_msg: str,
    user_content: str,
    *,
    chat_template: dict[str, Any] | None = None,
) -> tuple[str, list[int] | None]:
    chat_cfg = _chat_template_cfg(chat_template)
    user_content = _append_user_suffix(user_content, str(chat_cfg.get("user_suffix") or ""))
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    template_kwargs = dict(chat_cfg.get("template_kwargs") or {})
    add_generation_prompt = bool(chat_cfg.get("add_generation_prompt", True))
    if _uses_tokenized_chat_template(tokenizer):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            **_mistral_common_template_kwargs(template_kwargs),
        )
        return (
            _readable_chat_prompt(system_msg, user_content, add_generation_prompt=add_generation_prompt),
            _as_input_ids(encoded),
        )

    return (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        ),
        None,
    )


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    system_msg: str,
    user_content: str,
    *,
    chat_template: dict[str, Any] | None = None,
) -> str:
    chat_cfg = _chat_template_cfg(chat_template)
    user_content = _append_user_suffix(user_content, str(chat_cfg.get("user_suffix") or ""))
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    template_kwargs = dict(chat_cfg.get("template_kwargs") or {})
    if _uses_tokenized_chat_template(tokenizer):
        return _readable_chat_prompt(
            system_msg,
            user_content,
            add_generation_prompt=bool(chat_cfg.get("add_generation_prompt", True)),
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(chat_cfg.get("add_generation_prompt", True)),
        **template_kwargs,
    )


def render_prompt(
    *,
    claim: str,
    evidence_texts: list[str],
    tokenizer: AutoTokenizer,
    system_msg: str,
    output_mode: str,
    label_format: str,
    label_schema: str | None = None,
    chat_template: dict[str, Any] | None = None,
) -> tuple[str, int]:
    payload = render_prompt_payload(
        claim=claim,
        evidence_texts=evidence_texts,
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
        label_schema=label_schema,
        chat_template=chat_template,
    )
    return str(payload["prompt"]), int(payload["prompt_token_count"])


def render_prompt_payload(
    *,
    claim: str,
    evidence_texts: list[str],
    tokenizer: AutoTokenizer,
    system_msg: str,
    output_mode: str,
    label_format: str,
    label_schema: str | None = None,
    chat_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_content = build_user_content(claim, evidence_texts, output_mode, label_format, label_schema)
    prompt, prompt_input_ids = build_chat_prompt_with_input_ids(
        tokenizer,
        system_msg,
        user_content,
        chat_template=chat_template,
    )
    prompt_token_count = (
        len(prompt_input_ids)
        if prompt_input_ids is not None
        else count_tokens(prompt, tokenizer, add_special_tokens=False)
    )
    return {
        "prompt": prompt,
        "prompt_input_ids": prompt_input_ids,
        "prompt_token_count": prompt_token_count,
    }


def decode_token_prefix(tokenizer: AutoTokenizer, token_ids: list[int], length: int) -> str:
    if length <= 0:
        return ""
    return tokenizer.decode(token_ids[:length], skip_special_tokens=True).strip()


def truncate_single_evidence_to_budget(
    *,
    claim: str,
    evidence_text: str,
    tokenizer: AutoTokenizer,
    system_msg: str,
    output_mode: str,
    label_format: str,
    label_schema: str | None,
    budget: int,
    chat_template: dict[str, Any] | None = None,
) -> tuple[list[str], str, int, list[int] | None, bool]:
    """Shorten one evidence item until the full chat prompt fits the prompt budget."""
    token_ids = tokenizer(
        evidence_text,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]

    best_text: str | None = None
    best_prompt: str | None = None
    best_tokens: int | None = None
    best_input_ids: list[int] | None = None
    left = 0
    right = len(token_ids)
    while left <= right:
        mid = (left + right) // 2
        candidate_text = decode_token_prefix(tokenizer, token_ids, mid)
        payload = render_prompt_payload(
            claim=claim,
            evidence_texts=[candidate_text],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            label_schema=label_schema,
            chat_template=chat_template,
        )
        prompt = str(payload["prompt"])
        prompt_tokens = int(payload["prompt_token_count"])
        if prompt_tokens <= budget:
            best_text = candidate_text
            best_prompt = prompt
            best_tokens = prompt_tokens
            best_input_ids = payload.get("prompt_input_ids")
            left = mid + 1
        else:
            right = mid - 1

    if best_text is not None and best_prompt is not None and best_tokens is not None:
        return [best_text], best_prompt, best_tokens, best_input_ids, best_text.strip() != evidence_text.strip()

    no_evidence_payload = render_prompt_payload(
        claim=claim,
        evidence_texts=[],
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
        label_schema=label_schema,
        chat_template=chat_template,
    )
    return (
        [],
        str(no_evidence_payload["prompt"]),
        int(no_evidence_payload["prompt_token_count"]),
        no_evidence_payload.get("prompt_input_ids"),
        True,
    )


def build_target(row: dict, gold_label: str, output_mode: str, label_format: str = "name") -> str:
    label_schema = str(row.get("label_schema") or "").strip() or None
    target_label = label_letters_for_schema(label_schema)[gold_label] if label_format == "letter" else gold_label
    if output_mode == "explanation_label":
        explanation = str(row.get("explain", "")).strip() or "The available evidence supports this label."
        return f"Explanation: {explanation}\nLabel: {target_label}"
    if output_mode == "label_with_coverage":
        coverage_label = str(row.get("coverage_label") or "").strip()
        if coverage_label not in COVERAGE_LABEL_LETTERS:
            raise ValueError(
                "output_mode=label_with_coverage requires coverage_label to be one of "
                f"{sorted(COVERAGE_LABEL_LETTERS)}; got {coverage_label!r}."
            )
        target_coverage = COVERAGE_LABEL_LETTERS[coverage_label] if label_format == "letter" else coverage_label
        return f"Label: {target_label}\nCoverage: {target_coverage}"
    return f"Label: {target_label}"


OPTIONAL_BUILD_ROW_KEYS = (
    "coverage_label",
    "coverage_score",
    "coverage_version",
    "coverage",
    "selection_method",
    "raw_top_method",
    "raw_candidate_count",
    "raw_positive_count",
    "raw_selected_positive_count",
    "prompt_budget_reference_path",
    "prompt_budget_target_tokens",
    "prompt_budget_selected_tokens",
    "prompt_budget_delta_tokens",
    "prompt_budget_min_k",
    "prompt_budget_max_k",
    "prompt_budget_candidate_pool_k",
    "prompt_budget_raw_candidate_count",
    "prompt_budget_ranked_count",
    "prompt_budget_overshoot_tolerance_tokens",
    "prompt_budget_missing_policy",
)


def copy_optional_build_row_metadata(output: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    for key in OPTIONAL_BUILD_ROW_KEYS:
        if key in row:
            output[key] = row[key]
    return output


def auto_truncate_evidence(
    *,
    claim: str,
    evidence_texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    output_mode: str,
    system_prompt: str | None,
    row: dict,
    gold_label: str,
    label_format: str = "name",
    label_schema: str | None = None,
    chat_template: dict[str, Any] | None = None,
) -> dict:
    """Remove evidence items from the tail until the prompt fits within max_length."""
    system_msg = build_system_message(system_prompt, label_schema)
    row_for_target = {**row, "label_schema": label_schema or row.get("label_schema", "")}
    target = build_target(row_for_target, gold_label, output_mode, label_format)
    target_token_count = count_target_tokens(target, tokenizer)
    budget = max(0, int(max_length) - target_token_count)

    evidence_count_before = len(evidence_texts)
    kept = list(evidence_texts)

    prompt_payload = render_prompt_payload(
        claim=claim,
        evidence_texts=kept,
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
        label_schema=label_schema,
        chat_template=chat_template,
    )
    prompt = str(prompt_payload["prompt"])
    prompt_tokens = int(prompt_payload["prompt_token_count"])
    prompt_input_ids = prompt_payload.get("prompt_input_ids")

    was_truncated = False
    evidence_text_truncated = False
    while prompt_tokens > budget and len(kept) > 1:
        kept.pop()
        was_truncated = True
        prompt_payload = render_prompt_payload(
            claim=claim,
            evidence_texts=kept,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            label_schema=label_schema,
            chat_template=chat_template,
        )
        prompt = str(prompt_payload["prompt"])
        prompt_tokens = int(prompt_payload["prompt_token_count"])
        prompt_input_ids = prompt_payload.get("prompt_input_ids")

    if prompt_tokens > budget and len(kept) == 1:
        kept, prompt, prompt_tokens, prompt_input_ids, evidence_text_truncated = truncate_single_evidence_to_budget(
            claim=claim,
            evidence_text=kept[0],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            label_schema=label_schema,
            budget=budget,
            chat_template=chat_template,
        )
        was_truncated = True

    return {
        "prompt": prompt,
        "target": target,
        "prompt_input_ids": prompt_input_ids,
        "prompt_token_count": prompt_tokens,
        "target_token_count": target_token_count,
        "evidence_count": len(kept),
        "evidence_count_before": evidence_count_before,
        "was_truncated": was_truncated,
        "evidence_text_truncated": evidence_text_truncated,
        "overflow_after": prompt_tokens > budget,
    }


def build_training_row(
    retrieval_result: dict,
    tokenizer: AutoTokenizer,
    prompt_cfg: dict,
    *,
    allow_unlabeled: bool = False,
) -> dict:
    row = retrieval_result
    prompt_cfg = dict(prompt_cfg or {})
    label_schema = str(prompt_cfg.get("label_schema") or row.get("label_schema") or "").strip() or None
    gold_label = normalize_gold_label(row, label_schema=label_schema)
    is_unlabeled = not gold_label
    if is_unlabeled and not allow_unlabeled:
        return copy_optional_build_row_metadata(
            {**row, "gold_label": "", "gold_id": -1, "gold_explain": "",
             "prompt": "", "target": "", "prompt_add_special_tokens": False,
             "preserve_prompt_prefix": True, "prompt_token_count": 0,
             "target_token_count": 0, "evidence_count": 0, "was_truncated": False},
            row,
        )
    budget_label = gold_label or labels_for_schema(label_schema)[0]

    candidates = row.get("candidates", [])
    evidence_texts = [str(c.get("text", "")).strip() for c in candidates if isinstance(c, dict)]

    auto_length = bool(prompt_cfg.get("auto_length", True))
    max_length = int(prompt_cfg.get("max_length", 2048))
    output_mode = str(prompt_cfg.get("output_mode", "label_only")).strip().lower()
    label_format = str(prompt_cfg.get("label_format", "name")).strip().lower()
    system_prompt = prompt_cfg.get("system_prompt") or None
    chat_template = prompt_cfg.get("chat_template") if isinstance(prompt_cfg.get("chat_template"), dict) else None
    label2id = label2id_for_schema(label_schema)

    if auto_length and evidence_texts:
        result = auto_truncate_evidence(
            claim=str(row.get("claim", "")),
            evidence_texts=evidence_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            output_mode=output_mode,
            system_prompt=system_prompt,
            row=row,
            gold_label=budget_label,
            label_format=label_format,
            label_schema=label_schema,
            chat_template=chat_template,
        )
        output = {
            "event_id": row.get("event_id", ""),
            "claim": row.get("claim", ""),
            "label": row.get("label", ""),
            "label_schema": label_schema or "liar6",
            "explain": row.get("explain", ""),
            "candidates": candidates,
            "prompt": result["prompt"],
            "target": "" if is_unlabeled else result["target"],
            "prompt_input_ids": result.get("prompt_input_ids"),
            "gold_label": gold_label,
            "gold_id": label2id.get(gold_label, -1),
            "gold_explain": str(row.get("explain", "")).strip(),
            "prompt_add_special_tokens": False,
            "preserve_prompt_prefix": True,
            "prompt_token_count": result["prompt_token_count"],
            "target_token_count": 0 if is_unlabeled else result["target_token_count"],
            "evidence_count": result["evidence_count"],
            "evidence_count_before": result["evidence_count_before"],
            "was_truncated": result["was_truncated"],
            "evidence_text_truncated": result["evidence_text_truncated"],
        }
        if is_unlabeled:
            output["unlabeled_inference"] = True
            output["inference_target_token_reserve"] = result["target_token_count"]
        return copy_optional_build_row_metadata(output, row)

    system_msg = build_system_message(system_prompt, label_schema)
    row_for_target = {**row, "label_schema": label_schema or "liar6"}
    budget_target = build_target(row_for_target, budget_label, output_mode, label_format)
    budget_target_token_count = count_target_tokens(budget_target, tokenizer)
    target = "" if is_unlabeled else budget_target
    target_token_count = 0 if is_unlabeled else budget_target_token_count
    user_content = build_user_content(str(row.get("claim", "")), evidence_texts, output_mode, label_format, label_schema)
    prompt, prompt_input_ids = build_chat_prompt_with_input_ids(
        tokenizer,
        system_msg,
        user_content,
        chat_template=chat_template,
    )
    prompt_token_count = (
        len(prompt_input_ids)
        if prompt_input_ids is not None
        else count_tokens(prompt, tokenizer, add_special_tokens=False)
    )

    output = {
        "event_id": row.get("event_id", ""),
        "claim": row.get("claim", ""),
        "label": row.get("label", ""),
        "label_schema": label_schema or "liar6",
        "explain": row.get("explain", ""),
        "candidates": candidates,
        "prompt": prompt,
        "target": target,
        "prompt_input_ids": prompt_input_ids,
        "gold_label": gold_label,
        "gold_id": label2id.get(gold_label, -1),
        "gold_explain": str(row.get("explain", "")).strip(),
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "evidence_count": len(evidence_texts),
        "was_truncated": False,
    }
    if is_unlabeled:
        output["unlabeled_inference"] = True
        output["inference_target_token_reserve"] = budget_target_token_count
    return copy_optional_build_row_metadata(output, row)
