"""Prompt construction helpers for the build pipeline."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from fact_checking.data.constants import LABEL_DEFINITIONS, LABEL_LETTERS, LABEL2ID
from sft.data.labels import normalize_gold_label

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant for LIAR-RAW claims. "
    "Classify claims using only the claim and retrieved evidence supplied by the user."
)


def load_prompt_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(text: str, tokenizer: AutoTokenizer, *, add_special_tokens: bool = False) -> int:
    return len(tokenizer(text, truncation=False, add_special_tokens=add_special_tokens)["input_ids"])


def count_target_tokens(target: str, tokenizer: AutoTokenizer) -> int:
    ids = tokenizer(target.strip(), truncation=False, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return len(ids)


def build_system_message(system_prompt: str | None) -> str:
    if system_prompt and str(system_prompt).strip():
        return str(system_prompt).strip()
    return DEFAULT_SYSTEM_PROMPT


def format_evidence_block(evidence_texts: list[str]) -> str:
    lines = [f"[{i}] {text}" for i, text in enumerate(evidence_texts, start=1)]
    return "\n".join(lines)


def label_definitions_text(label_format: str = "name") -> str:
    if label_format == "letter":
        return "\n".join(
            f"- {LABEL_LETTERS[label]} ({label}): {LABEL_DEFINITIONS[label]}"
            for label in LABEL_DEFINITIONS
        )
    return "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABEL_DEFINITIONS)


def build_user_content(
    claim: str, evidence_texts: list[str], output_mode: str, label_format: str = "name"
) -> str:
    evidence_block = format_evidence_block(evidence_texts)
    evidence_display = evidence_block.strip() if evidence_block.strip() else "(no evidence available)"

    label_placeholder = "<a single letter from A-F>" if label_format == "letter" else "<label>"

    if output_mode == "explanation_label":
        return (
            "Classify the claim into exactly one LIAR-RAW label and provide a concise evidence-grounded explanation.\n\n"
            "Labels:\n"
            f"{label_definitions_text(label_format)}\n\n"
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
    return (
        "Classify the claim into exactly one LIAR-RAW label.\n\n"
        "Labels:\n"
        f"{label_definitions_text(label_format)}\n\n"
        "Rules:\n"
        "- Use the retrieved evidence as the primary source.\n"
        "- Do not invent facts not supported by the evidence.\n"
        f"- Respond with exactly one line: Label: {label_placeholder}\n\n"
        f"Claim:\n{claim.strip()}\n\n"
        f"Evidence:\n{evidence_display}"
    )


def build_chat_prompt(tokenizer: AutoTokenizer, system_msg: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def render_prompt(
    *,
    claim: str,
    evidence_texts: list[str],
    tokenizer: AutoTokenizer,
    system_msg: str,
    output_mode: str,
    label_format: str,
) -> tuple[str, int]:
    user_content = build_user_content(claim, evidence_texts, output_mode, label_format)
    prompt = build_chat_prompt(tokenizer, system_msg, user_content)
    return prompt, count_tokens(prompt, tokenizer, add_special_tokens=False)


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
    budget: int,
) -> tuple[list[str], str, int, bool]:
    """Shorten one evidence item until the full chat prompt fits the prompt budget."""
    token_ids = tokenizer(
        evidence_text,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]

    best_text: str | None = None
    best_prompt: str | None = None
    best_tokens: int | None = None
    left = 0
    right = len(token_ids)
    while left <= right:
        mid = (left + right) // 2
        candidate_text = decode_token_prefix(tokenizer, token_ids, mid)
        prompt, prompt_tokens = render_prompt(
            claim=claim,
            evidence_texts=[candidate_text],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
        )
        if prompt_tokens <= budget:
            best_text = candidate_text
            best_prompt = prompt
            best_tokens = prompt_tokens
            left = mid + 1
        else:
            right = mid - 1

    if best_text is not None and best_prompt is not None and best_tokens is not None:
        return [best_text], best_prompt, best_tokens, best_text.strip() != evidence_text.strip()

    no_evidence_prompt, no_evidence_tokens = render_prompt(
        claim=claim,
        evidence_texts=[],
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
    )
    return [], no_evidence_prompt, no_evidence_tokens, True


def build_target(row: dict, gold_label: str, output_mode: str, label_format: str = "name") -> str:
    target_label = LABEL_LETTERS[gold_label] if label_format == "letter" else gold_label
    if output_mode == "explanation_label":
        explanation = str(row.get("explain", "")).strip() or "The available evidence supports this label."
        return f"Explanation: {explanation}\nLabel: {target_label}"
    return f"Label: {target_label}"


OPTIONAL_BUILD_ROW_KEYS = (
    "selection_method",
    "raw_top_method",
    "raw_candidate_count",
    "raw_positive_count",
    "raw_selected_positive_count",
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
) -> dict:
    """Remove evidence items from the tail until the prompt fits within max_length."""
    system_msg = build_system_message(system_prompt)
    target = build_target(row, gold_label, output_mode, label_format)
    target_token_count = count_target_tokens(target, tokenizer)
    budget = max(0, int(max_length) - target_token_count)

    evidence_count_before = len(evidence_texts)
    kept = list(evidence_texts)

    prompt, prompt_tokens = render_prompt(
        claim=claim,
        evidence_texts=kept,
        tokenizer=tokenizer,
        system_msg=system_msg,
        output_mode=output_mode,
        label_format=label_format,
    )

    was_truncated = False
    evidence_text_truncated = False
    while prompt_tokens > budget and len(kept) > 1:
        kept.pop()
        was_truncated = True
        prompt, prompt_tokens = render_prompt(
            claim=claim,
            evidence_texts=kept,
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
        )

    if prompt_tokens > budget and len(kept) == 1:
        kept, prompt, prompt_tokens, evidence_text_truncated = truncate_single_evidence_to_budget(
            claim=claim,
            evidence_text=kept[0],
            tokenizer=tokenizer,
            system_msg=system_msg,
            output_mode=output_mode,
            label_format=label_format,
            budget=budget,
        )
        was_truncated = True

    return {
        "prompt": prompt,
        "target": target,
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
) -> dict:
    row = retrieval_result
    gold_label = normalize_gold_label(row)
    if not gold_label:
        return copy_optional_build_row_metadata(
            {**row, "gold_label": "", "gold_id": -1, "gold_explain": "",
             "prompt": "", "target": "", "prompt_add_special_tokens": False,
             "preserve_prompt_prefix": True, "prompt_token_count": 0,
             "target_token_count": 0, "evidence_count": 0, "was_truncated": False},
            row,
        )

    candidates = row.get("candidates", [])
    evidence_texts = [str(c.get("text", "")).strip() for c in candidates if isinstance(c, dict)]

    auto_length = bool(prompt_cfg.get("auto_length", True))
    max_length = int(prompt_cfg.get("max_length", 2048))
    output_mode = str(prompt_cfg.get("output_mode", "label_only")).strip().lower()
    label_format = str(prompt_cfg.get("label_format", "name")).strip().lower()
    system_prompt = prompt_cfg.get("system_prompt") or None

    if auto_length and evidence_texts:
        result = auto_truncate_evidence(
            claim=str(row.get("claim", "")),
            evidence_texts=evidence_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            output_mode=output_mode,
            system_prompt=system_prompt,
            row=row,
            gold_label=gold_label,
            label_format=label_format,
        )
        return copy_optional_build_row_metadata({
            "event_id": row.get("event_id", ""),
            "claim": row.get("claim", ""),
            "label": row.get("label", ""),
            "explain": row.get("explain", ""),
            "candidates": candidates,
            "prompt": result["prompt"],
            "target": result["target"],
            "gold_label": gold_label,
            "gold_id": LABEL2ID.get(gold_label, -1),
            "gold_explain": str(row.get("explain", "")).strip(),
            "prompt_add_special_tokens": False,
            "preserve_prompt_prefix": True,
            "prompt_token_count": result["prompt_token_count"],
            "target_token_count": result["target_token_count"],
            "evidence_count": result["evidence_count"],
            "evidence_count_before": result["evidence_count_before"],
            "was_truncated": result["was_truncated"],
            "evidence_text_truncated": result["evidence_text_truncated"],
        }, row)

    system_msg = build_system_message(system_prompt)
    target = build_target(row, gold_label, output_mode, label_format)
    target_token_count = count_target_tokens(target, tokenizer)
    user_content = build_user_content(str(row.get("claim", "")), evidence_texts, output_mode, label_format)
    prompt = build_chat_prompt(tokenizer, system_msg, user_content)
    prompt_token_count = count_tokens(prompt, tokenizer, add_special_tokens=False)

    return copy_optional_build_row_metadata({
        "event_id": row.get("event_id", ""),
        "claim": row.get("claim", ""),
        "label": row.get("label", ""),
        "explain": row.get("explain", ""),
        "candidates": candidates,
        "prompt": prompt,
        "target": target,
        "gold_label": gold_label,
        "gold_id": LABEL2ID.get(gold_label, -1),
        "gold_explain": str(row.get("explain", "")).strip(),
        "prompt_add_special_tokens": False,
        "preserve_prompt_prefix": True,
        "prompt_token_count": prompt_token_count,
        "target_token_count": target_token_count,
        "evidence_count": len(evidence_texts),
        "was_truncated": False,
    }, row)
