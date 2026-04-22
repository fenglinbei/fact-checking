from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import LABELS
from data.io import robust_sentence_split
from retrieval.embedder import EmbedderConfig, TextEmbedder
from retrieval.mmr import maximal_marginal_relevance


@dataclass(slots=True)
class BaselineConfig:
    model_name_or_path: str
    top_k: int = 8
    use_context: bool = False
    context_k: int = 1
    prompt_mode: str = "few_shot"  # zero_shot | few_shot
    few_shot_k: int = 10
    few_shot_mmr_lambda: float = 0.7
    retrieval_model: str = "/home/fenglin/project/models/bge-base-en-v1.5/"
    retrieval_batch_size: int = 64
    retrieval_max_length: int = 256
    max_new_tokens: int = 24
    temperature: float = 0.0
    do_sample: bool = False


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _context_window(content: str, sent_idx: int, k: int) -> str:
    sents = robust_sentence_split(content)
    if not sents:
        return ""
    idx = min(max(int(sent_idx), 0), len(sents) - 1)
    left = max(0, idx - k)
    right = min(len(sents), idx + k + 1)
    snippet = [sents[pos] for pos in range(left, right)]
    return " ".join(snippet)


def build_evidence_block(row: dict[str, Any], top_k: int, use_context: bool, context_k: int) -> str:
    candidates = sorted(
        row.get("candidates", []),
        key=lambda x: float(x.get("hybrid_score", 0.0)),
        reverse=True,
    )[:top_k]
    lines: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        sent = str(cand.get("text", "")).strip()
        if use_context:
            report = cand.get("source_report", {}) if isinstance(cand.get("source_report", {}), dict) else {}
            content = str(report.get("content", ""))
            ctx = _context_window(content=content, sent_idx=int(cand.get("sent_idx", 0)), k=context_k)
            evidence_text = ctx or sent
        else:
            evidence_text = sent
        lines.append(f"[{i}] {evidence_text}")
    return "\n".join(lines)


def _format_label_space() -> str:
    return ", ".join(LABELS)


def _extract_label(raw_text: str) -> str:
    lower = raw_text.strip().lower()
    lower = re.sub(r"[^a-z\- ]", " ", lower)
    for label in LABELS:
        if label in lower:
            return label
    # fallback to nearest by last token
    tokens = [t for t in lower.split() if t]
    if not tokens:
        return "half-true"
    token = tokens[-1]
    for label in LABELS:
        if token in label:
            return label
    return "half-true"


def build_zero_shot_prompt(claim: str, evidence_block: str) -> str:
    return (
        "You are a fact-checking classifier for the LIAR-RAW 6-way labels.\n"
        f"Label set: {_format_label_space()}.\n"
        "Given a claim and retrieved evidence, output exactly one label from the label set.\n"
        "Do not explain. Only output the label token.\n\n"
        f"Claim: {claim}\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Label:"
    )


def retrieve_few_shot_indices(
    query_claim: str,
    train_rows: list[dict[str, Any]],
    train_emb: np.ndarray,
    embedder: TextEmbedder,
    few_shot_k: int,
    mmr_lambda: float,
) -> list[int]:
    if not train_rows:
        return []
    q_emb = embedder.encode([query_claim], is_query=True)[0]
    query_scores = train_emb @ q_emb
    keep = maximal_marginal_relevance(
        query_scores=query_scores,
        sentence_vectors=train_emb,
        top_k=min(few_shot_k, len(train_rows)),
        lambda_weight=mmr_lambda,
    )
    return keep


def build_few_shot_prompt(
    claim: str,
    evidence_block: str,
    few_shot_examples: list[dict[str, Any]],
    top_k: int,
    use_context: bool,
    context_k: int,
) -> str:
    head = [
        "You are a fact-checking classifier for the LIAR-RAW 6-way labels.",
        f"Label set: {_format_label_space()}.",
        "Read examples then output exactly one label for the query claim.",
        "Do not explain. Only output the label token.",
        "",
    ]
    body: list[str] = []
    for i, ex in enumerate(few_shot_examples, start=1):
        ex_ev = build_evidence_block(ex, top_k=top_k, use_context=use_context, context_k=context_k)
        body.extend(
            [
                f"Example {i} Claim: {ex['claim']}",
                f"Example {i} Evidence:\n{ex_ev}",
                f"Example {i} Label: {ex['label']}",
                "",
            ]
        )
    tail = [
        f"Query Claim: {claim}",
        f"Query Evidence:\n{evidence_block}",
        "Query Label:",
    ]
    return "\n".join(head + body + tail)


def _build_model_and_tokenizer(model_name_or_path: str) -> tuple[Any, Any, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def run_inference(
    *,
    cfg: BaselineConfig,
    input_path: str | Path,
    output_path: str | Path,
    train_path_for_few_shot: str | Path | None = None,
) -> None:
    rows = load_jsonl(input_path)
    few_shot_rows: list[dict[str, Any]] = []
    train_emb: np.ndarray | None = None
    embedder: TextEmbedder | None = None

    if cfg.prompt_mode == "few_shot":
        if train_path_for_few_shot is None:
            raise ValueError("few_shot mode requires train_path_for_few_shot")
        few_shot_rows = load_jsonl(train_path_for_few_shot)
        embedder = TextEmbedder(
            EmbedderConfig(
                model_name=cfg.retrieval_model,
                device="cuda",
                max_length=cfg.retrieval_max_length,
                batch_size=cfg.retrieval_batch_size,
            )
        )
        train_claims = [str(x.get("claim", "")) for x in few_shot_rows]
        train_emb = embedder.encode(train_claims, is_query=False)

    model, tokenizer, device = _build_model_and_tokenizer(cfg.model_name_or_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as writer:
        for row in tqdm(rows, desc="LLM baseline infer"):
            evidence_block = build_evidence_block(
                row,
                top_k=cfg.top_k,
                use_context=cfg.use_context,
                context_k=cfg.context_k,
            )
            if cfg.prompt_mode == "few_shot":
                assert train_emb is not None and embedder is not None
                idxs = retrieve_few_shot_indices(
                    query_claim=str(row["claim"]),
                    train_rows=few_shot_rows,
                    train_emb=train_emb,
                    embedder=embedder,
                    few_shot_k=cfg.few_shot_k,
                    mmr_lambda=cfg.few_shot_mmr_lambda,
                )
                examples = [few_shot_rows[i] for i in idxs]
                prompt = build_few_shot_prompt(
                    claim=str(row["claim"]),
                    evidence_block=evidence_block,
                    few_shot_examples=examples,
                    top_k=cfg.top_k,
                    use_context=cfg.use_context,
                    context_k=cfg.context_k,
                )
            else:
                prompt = build_zero_shot_prompt(claim=str(row["claim"]), evidence_block=evidence_block)

            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_ids = out[0, enc["input_ids"].shape[1] :]
            raw_pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred_label = _extract_label(raw_pred)
            gold = str(row.get("label", ""))
            writer.write(
                json.dumps(
                    {
                        "event_id": row.get("event_id"),
                        "claim": row.get("claim"),
                        "gold_label": gold,
                        "pred_label": pred_label,
                        "prompt_mode": cfg.prompt_mode,
                        "baseline": "B1" if cfg.use_context else "B0",
                        "raw_generation": raw_pred.strip(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_sft_instances(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    use_context: bool,
    context_k: int,
) -> list[dict[str, str]]:
    instances: list[dict[str, str]] = []
    for row in rows:
        evidence_block = build_evidence_block(row, top_k=top_k, use_context=use_context, context_k=context_k)
        prompt = build_zero_shot_prompt(claim=str(row.get("claim", "")), evidence_block=evidence_block)
        target = str(row.get("label", "")).strip().lower()
        instances.append({"prompt": prompt, "target": target})
    return instances
