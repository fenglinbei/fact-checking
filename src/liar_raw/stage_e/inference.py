from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from liar_raw.stage_e.dataset import build_structured_input
from liar_raw.stage_e.faithfulness import FaithfulnessFilter
from liar_raw.stage_e.templater import TemplateExplainer


def generate_template_records(pred_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templater = TemplateExplainer()
    outputs = []
    for item in pred_items:
        explanation = templater.build(
            claim=item["claim"],
            pred_label=item["pred_label"],
            support_evidence=item.get("support_evidence", []),
            refute_evidence=item.get("refute_evidence", []),
        )
        outputs.append(
            {
                "event_id": item["event_id"],
                "claim": item["claim"],
                "pred_label": item["pred_label"],
                "gold_label": item.get("gold_label"),
                "explanation": explanation,
            }
        )
    return outputs


def generate_seq2seq_records(
    pred_items: list[dict[str, Any]],
    model_dir: str | Path,
    max_input_length: int,
    max_output_length: int,
    faithfulness_filter: FaithfulnessFilter | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    templater = TemplateExplainer()
    outputs = []
    for item in tqdm(pred_items, desc="Generate explanations"):
        source = build_structured_input(
            claim=item["claim"],
            label=item["pred_label"],
            selected_subclaims=item.get("selected_subclaims", []),
            support_evidence=item.get("support_evidence", []),
            refute_evidence=item.get("refute_evidence", []),
        )
        enc = tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        out = model.generate(
            **enc,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_output_length,
        )
        explanation = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        fallback = templater.build(
            claim=item["claim"],
            pred_label=item["pred_label"],
            support_evidence=item.get("support_evidence", []),
            refute_evidence=item.get("refute_evidence", []),
        )

        if faithfulness_filter is not None:
            evidence_texts = [
                *(x.get("text") or x.get("sentence") or "" for x in item.get("support_evidence", [])),
                *(x.get("text") or x.get("sentence") or "" for x in item.get("refute_evidence", [])),
            ]
            explanation = faithfulness_filter.filter_or_fallback(explanation, evidence_texts, fallback)

        outputs.append(
            {
                "event_id": item["event_id"],
                "claim": item["claim"],
                "pred_label": item["pred_label"],
                "gold_label": item.get("gold_label"),
                "explanation": explanation,
            }
        )
    return outputs
