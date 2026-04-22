from __future__ import annotations

from fact_checking import LABELS
from fact_checking.baselines.llm_baseline import build_zero_shot_prompt


class OutputStrategy:
    name = "label_only"

    def build_prompt(self, claim: str, evidence_block: str) -> str:
        raise NotImplementedError

    def build_target(self, row: dict, gold_label: str) -> str:
        raise NotImplementedError


class LabelOnlyOutputStrategy(OutputStrategy):
    name = "label_only"

    def build_prompt(self, claim: str, evidence_block: str) -> str:
        return build_zero_shot_prompt(claim=claim, evidence_block=evidence_block)

    def build_target(self, row: dict, gold_label: str) -> str:
        return gold_label


class ExplanationLabelOutputStrategy(OutputStrategy):
    name = "explanation_label"

    def build_prompt(self, claim: str, evidence_block: str) -> str:
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
            "Explanation:"
        )

    def build_target(self, row: dict, gold_label: str) -> str:
        explanation = str(row.get("explanation", "")).strip() or "The available evidence supports this label."
        return f"Explanation: {explanation}\nLabel: {gold_label}"


def _infer_output_mode(baseline_cfg: dict) -> str:
    explicit_mode = str(baseline_cfg.get("output_mode", "")).strip().lower()
    if explicit_mode:
        return explicit_mode

    return "label_only"


def build_output_strategy(baseline_cfg: dict) -> OutputStrategy:
    output_mode = _infer_output_mode(baseline_cfg)
    if output_mode == "label_only":
        return LabelOnlyOutputStrategy()
    if output_mode == "explanation_label":
        return ExplanationLabelOutputStrategy()
    raise ValueError(
        f"Unsupported baseline.output_mode={output_mode}. "
        "Use 'label_only' or 'explanation_label'."
    )
