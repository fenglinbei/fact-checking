from __future__ import annotations

from transformers import AutoTokenizer

from fact_checking import LABELS
from fact_checking.baselines.llm_baseline import build_zero_shot_prompt

LABEL_DEFINITIONS = {
    "pants-fire": "completely false and implausible",
    "false": "false based on the available evidence",
    "barely-true": "mostly false, with only a small element of truth",
    "half-true": "partly true and partly false",
    "mostly-true": "mostly true, with minor missing context or caveats",
    "true": "accurate based on the available evidence",
}


class OutputStrategy:
    name = "label_only"
    prompt_version = "v1"
    prompt_add_special_tokens = True
    preserve_prompt_prefix = False

    def build_prompt(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer | None = None,
    ) -> str:
        raise NotImplementedError

    def build_target(self, row: dict, gold_label: str) -> str:
        raise NotImplementedError


class LabelOnlyOutputStrategy(OutputStrategy):
    name = "label_only"

    def build_prompt(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer | None = None,
    ) -> str:
        del tokenizer
        return build_zero_shot_prompt(claim=claim, evidence_block=evidence_block)

    def build_target(self, row: dict, gold_label: str) -> str:
        del row
        return gold_label


class ExplanationLabelOutputStrategy(OutputStrategy):
    name = "explanation_label"

    def build_prompt(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer | None = None,
    ) -> str:
        del tokenizer
        evidence_text = evidence_block.strip() if evidence_block.strip() else "(no evidence available)"
        label_list = ", ".join(LABELS)
        return (
            "You are a fact-checking assistant. Read the claim and retrieved evidence, "
            "then provide a concise evidence-grounded explanation followed by the final label.\n\n"
            f"Valid labels: {label_list}\n\n"
            f"Claim:\n{claim.strip()}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Respond with exactly the following format:\n"
            "Explanation: <brief evidence-grounded explanation>\n"
            "Label: <one valid label>\n\n"
            "Explanation: "
        )

    def build_target(self, row: dict, gold_label: str) -> str:
        explanation = str(row.get("explain", "")).strip() or "The available evidence supports this label."
        return f"{explanation}\nLabel: {gold_label}"


class ChatTemplateOutputStrategy(OutputStrategy):
    prompt_version = "v2"
    prompt_add_special_tokens = False
    preserve_prompt_prefix = True

    def _render_chat_prompt(self, tokenizer: AutoTokenizer | None, user_content: str) -> str:
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("prompt_version=v2 requires a tokenizer with apply_chat_template support.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful fact-checking assistant for LIAR-RAW claims. "
                    "Classify claims using only the claim and retrieved evidence supplied by the user."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _label_definitions_text() -> str:
        return "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABELS)

    @staticmethod
    def _evidence_text(evidence_block: str) -> str:
        evidence_text = evidence_block.strip()
        return evidence_text if evidence_text else "(no evidence available)"


class ChatLabelOnlyOutputStrategy(ChatTemplateOutputStrategy):
    name = "label_only"

    def build_prompt(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer | None = None,
    ) -> str:
        user_content = (
            "Classify the claim into exactly one LIAR-RAW label.\n\n"
            "Labels:\n"
            f"{self._label_definitions_text()}\n\n"
            "Rules:\n"
            "- Use the retrieved evidence as the primary source.\n"
            "- Do not invent facts not supported by the evidence.\n"
            "- Respond with exactly one line: Label: <label>\n\n"
            f"Claim:\n{claim.strip()}\n\n"
            f"Evidence:\n{self._evidence_text(evidence_block)}"
        )
        return self._render_chat_prompt(tokenizer, user_content)

    def build_target(self, row: dict, gold_label: str) -> str:
        del row
        return f"Label: {gold_label}"


class ChatExplanationLabelOutputStrategy(ChatTemplateOutputStrategy):
    name = "explanation_label"

    def build_prompt(
        self,
        claim: str,
        evidence_block: str,
        tokenizer: AutoTokenizer | None = None,
    ) -> str:
        user_content = (
            "Classify the claim into exactly one LIAR-RAW label and provide a concise evidence-grounded explanation.\n\n"
            "Labels:\n"
            f"{self._label_definitions_text()}\n\n"
            "Rules:\n"
            "- Use the retrieved evidence as the primary source.\n"
            "- Do not invent facts not supported by the evidence.\n"
            "- Keep the explanation brief and evidence-grounded.\n"
            "- Respond with exactly two lines in this format:\n"
            "Explanation: <brief explanation>\n"
            "Label: <label>\n\n"
            f"Claim:\n{claim.strip()}\n\n"
            f"Evidence:\n{self._evidence_text(evidence_block)}"
        )
        return self._render_chat_prompt(tokenizer, user_content)

    def build_target(self, row: dict, gold_label: str) -> str:
        explanation = str(row.get("explain", "")).strip() or "The available evidence supports this label."
        return f"Explanation: {explanation}\nLabel: {gold_label}"


def _infer_output_mode(baseline_cfg: dict) -> str:
    explicit_mode = str(baseline_cfg.get("output_mode", "")).strip().lower()
    if explicit_mode:
        return explicit_mode

    return "label_only"


def _infer_prompt_version(baseline_cfg: dict) -> str:
    return str(baseline_cfg.get("prompt_version", "v1")).strip().lower() or "v1"


def build_output_strategy(baseline_cfg: dict) -> OutputStrategy:
    output_mode = _infer_output_mode(baseline_cfg)
    prompt_version = _infer_prompt_version(baseline_cfg)
    if prompt_version == "v1":
        if output_mode == "label_only":
            return LabelOnlyOutputStrategy()
        if output_mode == "explanation_label":
            return ExplanationLabelOutputStrategy()
    elif prompt_version == "v2":
        if output_mode == "label_only":
            return ChatLabelOnlyOutputStrategy()
        if output_mode == "explanation_label":
            return ChatExplanationLabelOutputStrategy()
    else:
        raise ValueError("Unsupported baseline.prompt_version=%s. Use 'v1' or 'v2'." % prompt_version)

    raise ValueError(
        f"Unsupported baseline.output_mode={output_mode}. "
        "Use 'label_only' or 'explanation_label'."
    )
