#!/usr/bin/env python3
"""Inspect evidence shape produced by build.retrieval.chunking strategies.

This helper reuses the production ChunkingStrategy implementations. It selects
readable anchor sentences with cheap lexical/BM25 scoring for visualization; it
does not rerun the full dense retrieval + MMR build phase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fact_checking.build.chunking import ChunkingStrategy, build_chunking_strategy
from fact_checking.data.io import iter_sentences, load_split
from fact_checking.retrieval.text_utils import (
    bm25_like_score_from_counters,
    content_tokens_counter,
    lexical_overlap_f1_from_counters,
)
from fact_checking.utils.text import clean_text, robust_sentence_split, word_tokens


DEFAULT_STRATEGIES = "sentence,ctx_window,raw,semantic,ctx_semantic"


@dataclass(frozen=True)
class Anchor:
    sample_index: int
    event_id: str
    claim: str
    label: str
    report_id: int | str
    sent_idx: int
    source_sentence: str
    report_content: str
    link: str | None
    domain: str | None
    score: float


@dataclass(frozen=True)
class StrategyRender:
    strategy: str
    status: str
    text: str
    span: str
    sentence_count: int
    word_count: int
    char_count: int
    zh: str


@dataclass(frozen=True)
class PackedEvidence:
    rank: int
    report_id: int | str
    sent_idx: int
    score: float
    text: str
    span: str
    token_count: int
    sentence_count: int
    word_count: int
    zh: str


@dataclass(frozen=True)
class StrategyPack:
    strategy: str
    status: str
    budget_tokens: int
    used_tokens: int
    item_count: int
    skipped_duplicates: int
    skipped_too_large: int
    items: list[PackedEvidence]


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class WordTokenCounter:
    name = "word_tokens fallback"

    def count(self, text: str) -> int:
        return len(word_tokens(text))


class HFTokenCounter:
    def __init__(self, model_name_or_path: str) -> None:
        from transformers import AutoTokenizer

        self.name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how build.retrieval.chunking.strategy changes the "
            "evidence text produced from the same selected sentence."
        )
    )
    parser.add_argument("--experiment", default="b0", help="Hydra experiment name, e.g. b0/b1/b3/b4.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--anchors-per-example", type=int, default=1)
    parser.add_argument("--scan-limit", type=int, default=200)
    parser.add_argument(
        "--min-source-sentences",
        type=int,
        default=2,
        help="Skip source reports shorter than this many split sentences.",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        default=[],
        help="Limit examples to one or more event_id values. Accepts either 11269 or 11269.json.",
    )
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--context-k", type=int, default=None, help="Override chunking.context_k for visualization.")
    parser.add_argument("--theta", type=float, default=None, help="Override chunking.theta for semantic strategies.")
    parser.add_argument(
        "--theta-sweep",
        default="",
        help="Comma/space separated theta values for horizontal semantic comparison, e.g. '0.3,0.5,0.7'.",
    )
    parser.add_argument(
        "--max-display-chars",
        type=int,
        default=900,
        help="Maximum evidence preview length per strategy in Markdown.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be passed multiple times.",
    )
    parser.add_argument(
        "--translations",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping from exact/normalized English evidence text "
            "to Chinese translation."
        ),
    )
    parser.add_argument(
        "--translation-template",
        type=Path,
        default=None,
        help="Optional path to write an empty English-to-Chinese translation template.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/chunking_evidence_examples.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--json-output", type=Path, default=None, help="Optional structured JSON output path.")
    parser.add_argument(
        "--case-study-budget",
        type=int,
        default=0,
        help="If >0, add a case study packing evidence chunks under this token budget.",
    )
    parser.add_argument(
        "--case-study-event-id",
        default="",
        help="Optional event_id for the 512-context packing case study. Defaults to the first selected example.",
    )
    parser.add_argument(
        "--case-study-max-candidates",
        type=int,
        default=64,
        help="Maximum ranked anchor sentences considered for the packing case study.",
    )
    return parser.parse_args()


def load_pipeline_cfg(experiment: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="pipeline/default",
            overrides=[f"experiment={experiment}", *overrides],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def split_strategy_names(value: str) -> list[str]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--strategies must contain at least one strategy")
    return names


def parse_theta_values(value: str) -> list[float]:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    return [float(part) for part in parts]


def normalized_key(text: str) -> str:
    return " ".join(clean_text(text).split())


def canonical_event_id(value: str) -> str:
    return str(value).strip().removesuffix(".json")


def load_translations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Translation file must be a JSON object: {path}")
    translations: dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        translations[str(key)] = text
        translations[normalized_key(str(key))] = text
    return translations


def score_anchor(claim: str, sentence: str) -> float:
    q_ctr, q_len = content_tokens_counter(claim)
    s_ctr, s_len = content_tokens_counter(sentence)
    lexical = lexical_overlap_f1_from_counters(q_ctr, s_ctr, q_len, s_len)
    bm25 = bm25_like_score_from_counters(q_ctr, s_ctr, s_len)
    return float(lexical + 0.03 * bm25)


def select_anchors(
    cfg: dict[str, Any],
    *,
    split: str,
    num_examples: int,
    anchors_per_example: int,
    scan_limit: int,
    event_ids: list[str],
    min_source_sentences: int,
) -> list[Anchor]:
    data_cfg = cfg["build"]["data"]
    split_path = PROJECT_ROOT / str(data_cfg[f"{split}_path"])
    samples = load_split(split_path)
    event_filter = {canonical_event_id(event_id) for event_id in event_ids}
    event_order = {canonical_event_id(event_id): idx for idx, event_id in enumerate(event_ids)}

    grouped: list[tuple[float, list[Anchor]]] = []
    scan_samples = samples if event_filter else samples[: max(scan_limit, 1)]
    for sample_index, sample in enumerate(scan_samples):
        if event_filter and canonical_event_id(sample.event_id) not in event_filter:
            continue
        anchors: list[Anchor] = []
        for sent in iter_sentences(sample):
            content = clean_text(str(sent.raw.get("content", ""))) if isinstance(sent.raw, dict) else ""
            if not content:
                continue
            source_sents = robust_sentence_split(content)
            if len(source_sents) < max(min_source_sentences, 1):
                continue
            score = score_anchor(sample.claim, sent.text)
            if score <= 0:
                continue
            anchors.append(
                Anchor(
                    sample_index=sample_index,
                    event_id=sample.event_id,
                    claim=sample.claim,
                    label=sample.label,
                    report_id=sent.report_id,
                    sent_idx=sent.sent_idx,
                    source_sentence=sent.text,
                    report_content=content,
                    link=sent.link,
                    domain=sent.domain,
                    score=score,
                )
            )
        anchors.sort(key=lambda item: item.score, reverse=True)
        if anchors:
            grouped.append((anchors[0].score, anchors[: max(anchors_per_example, 1)]))

    if event_filter:
        grouped.sort(key=lambda item: event_order.get(canonical_event_id(item[1][0].event_id), len(event_order)))
    else:
        grouped.sort(key=lambda item: item[0], reverse=True)
    selected: list[Anchor] = []
    for _, anchors in grouped[: max(num_examples, 1)]:
        selected.extend(anchors)
    if not selected:
        raise RuntimeError(f"No usable anchors found in split={split!r} from {split_path}")
    return selected


def strategy_config(
    base_chunking_cfg: dict[str, Any],
    strategy: str,
    args: argparse.Namespace,
    *,
    theta: float | None = None,
) -> dict[str, Any]:
    cfg = dict(base_chunking_cfg)
    cfg["strategy"] = strategy
    if args.context_k is not None:
        cfg["context_k"] = args.context_k
    if args.theta is not None:
        cfg["theta"] = args.theta
    if theta is not None:
        cfg["theta"] = theta
    return cfg


def build_strategies(
    retrieval_cfg: dict[str, Any],
    strategy_names: list[str],
    args: argparse.Namespace,
) -> dict[str, tuple[ChunkingStrategy | None, str | None]]:
    base_chunking_cfg = dict(retrieval_cfg.get("chunking") or {})
    result: dict[str, tuple[ChunkingStrategy | None, str | None]] = {}
    for name in strategy_names:
        cfg = strategy_config(base_chunking_cfg, name, args)
        try:
            result[name] = (build_chunking_strategy(cfg, retrieval_cfg), None)
        except Exception as exc:
            result[name] = (None, f"{type(exc).__name__}: {exc}")
    return result


def build_theta_sweep_strategies(
    retrieval_cfg: dict[str, Any],
    theta_values: list[float],
    args: argparse.Namespace,
) -> dict[str, dict[float, tuple[ChunkingStrategy | None, str | None]]]:
    base_chunking_cfg = dict(retrieval_cfg.get("chunking") or {})
    result: dict[str, dict[float, tuple[ChunkingStrategy | None, str | None]]] = {}
    for name in ("semantic", "ctx_semantic"):
        result[name] = {}
        for theta in theta_values:
            cfg = strategy_config(base_chunking_cfg, name, args, theta=theta)
            try:
                result[name][theta] = (build_chunking_strategy(cfg, retrieval_cfg), None)
            except Exception as exc:
                result[name][theta] = (None, f"{type(exc).__name__}: {exc}")
    return result


def build_token_counter(cfg: dict[str, Any]) -> TokenCounter:
    model_name = str(cfg.get("build", {}).get("prompt", {}).get("model_name_or_path", "") or "").strip()
    if not model_name:
        return WordTokenCounter()
    try:
        return HFTokenCounter(model_name)
    except Exception as exc:
        print(f"Warning: failed to load tokenizer {model_name!r}: {exc}. Falling back to word tokens.", file=sys.stderr)
        return WordTokenCounter()


def locate_span(sents: list[str], evidence: str) -> list[int]:
    target = normalized_key(evidence)
    if not target:
        return []
    n = len(sents)
    for start in range(n):
        for end in range(start, n):
            if normalized_key(" ".join(sents[start : end + 1])) == target:
                return list(range(start, end + 1))
    contained = [idx for idx, sent in enumerate(sents) if normalized_key(sent) in target]
    return contained


def format_span(indices: list[int], anchor_idx: int) -> str:
    if not indices:
        return "unknown"
    if len(indices) == 1:
        span = f"S{indices[0]:02d}"
    else:
        span = f"S{min(indices):02d}-S{max(indices):02d}"
    if anchor_idx in indices:
        span += " (anchor included)"
    return span


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f" ... [truncated, {omitted} chars omitted]"


def html_preview(text: str, max_chars: int) -> str:
    import html

    return html.escape(truncate(text, max_chars)).replace("\n", "<br>")


def render_strategy(
    strategy_name: str,
    strategy: ChunkingStrategy | None,
    init_error: str | None,
    anchor: Anchor,
    translations: dict[str, str],
) -> StrategyRender:
    if init_error:
        return StrategyRender(strategy_name, f"unavailable: {init_error}", "", "unknown", 0, 0, 0, "")
    assert strategy is not None
    sents = robust_sentence_split(anchor.report_content)
    try:
        evidence = clean_text(strategy.chunk_from_presplit(sents, anchor.sent_idx))
    except Exception as exc:
        return StrategyRender(strategy_name, f"error: {type(exc).__name__}: {exc}", "", "unknown", 0, 0, 0, "")

    indices = locate_span(sents, evidence)
    zh = translations.get(evidence) or translations.get(normalized_key(evidence), "")
    return StrategyRender(
        strategy=strategy_name,
        status="ok",
        text=evidence,
        span=format_span(indices, anchor.sent_idx),
        sentence_count=len(indices),
        word_count=len(word_tokens(evidence)),
        char_count=len(evidence),
        zh=zh,
    )


def render_text_for_strategy(
    strategy: ChunkingStrategy,
    content: str,
    sent_idx: int,
    translations: dict[str, str],
) -> StrategyRender:
    sents = robust_sentence_split(content)
    evidence = clean_text(strategy.chunk_from_presplit(sents, sent_idx))
    indices = locate_span(sents, evidence)
    zh = translations.get(evidence) or translations.get(normalized_key(evidence), "")
    return StrategyRender(
        strategy="",
        status="ok",
        text=evidence,
        span=format_span(indices, sent_idx),
        sentence_count=len(indices),
        word_count=len(word_tokens(evidence)),
        char_count=len(evidence),
        zh=zh,
    )


def render_theta_sweep(
    theta_sweep_strategies: dict[str, dict[float, tuple[ChunkingStrategy | None, str | None]]],
    anchor: Anchor,
    translations: dict[str, str],
) -> dict[str, dict[float, StrategyRender]]:
    rendered: dict[str, dict[float, StrategyRender]] = {}
    for strategy_name, by_theta in theta_sweep_strategies.items():
        rendered[strategy_name] = {}
        for theta, (strategy, init_error) in by_theta.items():
            rendered[strategy_name][theta] = render_strategy(
                f"{strategy_name}@theta={theta:g}",
                strategy,
                init_error,
                anchor,
                translations,
            )
    return rendered


def find_sample(cfg: dict[str, Any], split: str, event_id: str):
    data_cfg = cfg["build"]["data"]
    split_path = PROJECT_ROOT / str(data_cfg[f"{split}_path"])
    target = canonical_event_id(event_id)
    for sample in load_split(split_path):
        if canonical_event_id(sample.event_id) == target:
            return sample
    raise RuntimeError(f"event_id={event_id!r} not found in split={split!r}")


def ranked_case_study_anchors(
    cfg: dict[str, Any],
    split: str,
    event_id: str,
    max_candidates: int,
    min_source_sentences: int,
) -> list[Anchor]:
    sample = find_sample(cfg, split, event_id)
    anchors: list[Anchor] = []
    for sent in iter_sentences(sample):
        content = clean_text(str(sent.raw.get("content", ""))) if isinstance(sent.raw, dict) else ""
        if not content:
            continue
        if len(robust_sentence_split(content)) < max(min_source_sentences, 1):
            continue
        score = score_anchor(sample.claim, sent.text)
        if score <= 0:
            continue
        anchors.append(
            Anchor(
                sample_index=-1,
                event_id=sample.event_id,
                claim=sample.claim,
                label=sample.label,
                report_id=sent.report_id,
                sent_idx=sent.sent_idx,
                source_sentence=sent.text,
                report_content=content,
                link=sent.link,
                domain=sent.domain,
                score=score,
            )
        )
    anchors.sort(key=lambda item: item.score, reverse=True)
    return anchors[: max(max_candidates, 1)]


def pack_strategy_context(
    strategy_name: str,
    strategy: ChunkingStrategy | None,
    init_error: str | None,
    anchors: list[Anchor],
    token_counter: TokenCounter,
    budget_tokens: int,
    translations: dict[str, str],
) -> StrategyPack:
    if init_error:
        return StrategyPack(strategy_name, f"unavailable: {init_error}", budget_tokens, 0, 0, 0, 0, [])
    assert strategy is not None
    used_tokens = 0
    items: list[PackedEvidence] = []
    seen: set[str] = set()
    skipped_duplicates = 0
    skipped_too_large = 0
    for anchor in anchors:
        try:
            rendered = render_text_for_strategy(strategy, anchor.report_content, anchor.sent_idx, translations)
        except Exception:
            continue
        text_key = normalized_key(rendered.text)
        if not text_key:
            continue
        if text_key in seen:
            skipped_duplicates += 1
            continue
        seen.add(text_key)
        item_prefix = f"[{len(items) + 1}] "
        item_tokens = token_counter.count(item_prefix + rendered.text + "\n")
        if item_tokens > budget_tokens or used_tokens + item_tokens > budget_tokens:
            skipped_too_large += 1
            continue
        items.append(
            PackedEvidence(
                rank=len(items) + 1,
                report_id=anchor.report_id,
                sent_idx=anchor.sent_idx,
                score=anchor.score,
                text=rendered.text,
                span=rendered.span,
                token_count=item_tokens,
                sentence_count=rendered.sentence_count,
                word_count=rendered.word_count,
                zh=rendered.zh,
            )
        )
        used_tokens += item_tokens
    return StrategyPack(
        strategy=strategy_name,
        status="ok",
        budget_tokens=budget_tokens,
        used_tokens=used_tokens,
        item_count=len(items),
        skipped_duplicates=skipped_duplicates,
        skipped_too_large=skipped_too_large,
        items=items,
    )


def build_context_case_study(
    cfg: dict[str, Any],
    split: str,
    event_id: str,
    strategies: dict[str, tuple[ChunkingStrategy | None, str | None]],
    token_counter: TokenCounter,
    budget_tokens: int,
    max_candidates: int,
    min_source_sentences: int,
    translations: dict[str, str],
) -> dict[str, Any]:
    anchors = ranked_case_study_anchors(
        cfg,
        split=split,
        event_id=event_id,
        max_candidates=max_candidates,
        min_source_sentences=min_source_sentences,
    )
    if not anchors:
        raise RuntimeError(f"No ranked anchors for case study event_id={event_id!r}")
    packs = [
        pack_strategy_context(
            strategy_name,
            strategy,
            init_error,
            anchors,
            token_counter,
            budget_tokens,
            translations,
        )
        for strategy_name, (strategy, init_error) in strategies.items()
    ]
    sample = find_sample(cfg, split, event_id)
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "tokenizer": token_counter.name,
        "budget_tokens": budget_tokens,
        "candidate_count": len(anchors),
        "packs": packs,
    }


def source_context(anchor: Anchor, radius: int = 2) -> list[tuple[int, str, bool]]:
    sents = robust_sentence_split(anchor.report_content)
    if not sents:
        return []
    idx = min(max(anchor.sent_idx, 0), len(sents) - 1)
    left = max(0, idx - radius)
    right = min(len(sents), idx + radius + 1)
    return [(pos, sents[pos], pos == idx) for pos in range(left, right)]


def strategy_note(name: str, retrieval_cfg: dict[str, Any], args: argparse.Namespace) -> str:
    chunk_cfg = strategy_config(dict(retrieval_cfg.get("chunking") or {}), name, args)
    if name == "sentence":
        return "single selected sentence"
    if name == "ctx_window":
        return f"selected sentence +/- context_k={int(chunk_cfg.get('context_k', 1))}"
    if name == "raw":
        return "full source report"
    if name == "semantic":
        return f"merge adjacent sentences when cosine similarity > theta={float(chunk_cfg.get('theta', 0.7))}"
    if name == "ctx_semantic":
        return (
            f"partition into context_k={int(chunk_cfg.get('context_k', 1))} windows, "
            f"then merge adjacent windows when cosine similarity > theta={float(chunk_cfg.get('theta', 0.7))}"
        )
    return "custom strategy"


def append_theta_sweep_markdown(
    lines: list[str],
    theta_rendered: dict[str, dict[float, StrategyRender]],
    theta_values: list[float],
    max_display_chars: int,
) -> None:
    if not theta_values:
        return
    lines.append("### Theta sweep")
    lines.append("")
    for strategy_name in ("semantic", "ctx_semantic"):
        if strategy_name not in theta_rendered:
            continue
        lines.append(f"#### {strategy_name}")
        lines.append("")
        header = "| field | " + " | ".join(f"theta={theta:g}" for theta in theta_values) + " |"
        sep = "|---|" + "|".join("---" for _ in theta_values) + "|"
        lines.append(header)
        lines.append(sep)
        fields = [
            ("span", lambda item: item.span),
            ("sentences", lambda item: str(item.sentence_count)),
            ("words", lambda item: str(item.word_count)),
            ("English evidence", lambda item: html_preview(item.text, max_display_chars)),
            ("Chinese translation", lambda item: html_preview(item.zh, max_display_chars) if item.zh else ""),
        ]
        for field_name, getter in fields:
            cells: list[str] = []
            for theta in theta_values:
                item = theta_rendered[strategy_name][theta]
                if item.status != "ok":
                    cells.append(f"`{item.status}`")
                else:
                    cells.append(getter(item))
            lines.append(f"| {field_name} | " + " | ".join(cells) + " |")
        lines.append("")


def strategy_pack_to_dict(pack: StrategyPack) -> dict[str, Any]:
    return {
        "strategy": pack.strategy,
        "status": pack.status,
        "budget_tokens": pack.budget_tokens,
        "used_tokens": pack.used_tokens,
        "item_count": pack.item_count,
        "skipped_duplicates": pack.skipped_duplicates,
        "skipped_too_large": pack.skipped_too_large,
        "items": [item.__dict__ for item in pack.items],
    }


def append_context_case_study_markdown(
    lines: list[str],
    case_study: dict[str, Any] | None,
    max_display_chars: int,
) -> None:
    if not case_study:
        return
    packs: list[StrategyPack] = case_study["packs"]
    budget = int(case_study["budget_tokens"])
    lines.append("## 512-Token Context Packing Case Study")
    lines.append("")
    lines.append("Evidence chunks are greedily packed under the same evidence-only token budget.")
    lines.append("")
    lines.append(f"- event_id: `{case_study['event_id']}`")
    lines.append(f"- label: `{case_study['label']}`")
    lines.append(f"- claim: {case_study['claim']}")
    lines.append(f"- budget: `{budget}` evidence tokens")
    lines.append(f"- tokenizer: `{case_study['tokenizer']}`")
    lines.append(f"- ranked candidate anchors considered: `{case_study['candidate_count']}`")
    lines.append("")
    lines.append("| strategy | status | packed items | used tokens | utilization | skipped duplicate chunks | skipped over budget |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for pack in packs:
        utilization = (pack.used_tokens / budget) if budget else 0.0
        lines.append(
            f"| `{pack.strategy}` | {pack.status} | {pack.item_count} | {pack.used_tokens}/{budget} | "
            f"{utilization:.1%} | {pack.skipped_duplicates} | {pack.skipped_too_large} |"
        )
    lines.append("")

    for pack in packs:
        lines.append(f"### Context pack: {pack.strategy}")
        lines.append("")
        if pack.status != "ok":
            lines.append(f"`{pack.status}`")
            lines.append("")
            continue
        lines.append("| # | tokens | span | source | score | evidence preview | Chinese translation |")
        lines.append("|---:|---:|---|---|---:|---|---|")
        for item in pack.items:
            source = f"report={item.report_id}, sent={item.sent_idx}"
            lines.append(
                f"| {item.rank} | {item.token_count} | {item.span} | {source} | {item.score:.4f} | "
                f"{html_preview(item.text, max_display_chars)} | {html_preview(item.zh, max_display_chars) if item.zh else ''} |"
            )
        lines.append("")


def markdown_for(
    cfg: dict[str, Any],
    anchors: list[Anchor],
    strategies: dict[str, tuple[ChunkingStrategy | None, str | None]],
    theta_sweep_strategies: dict[str, dict[float, tuple[ChunkingStrategy | None, str | None]]],
    theta_values: list[float],
    context_case_study: dict[str, Any] | None,
    translations: dict[str, str],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    retrieval_cfg = cfg["build"]["retrieval"]
    lines: list[str] = []
    rows: list[dict[str, Any]] = []

    lines.append("# Chunking Evidence Visualization")
    lines.append("")
    lines.append(f"- experiment: `{args.experiment}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- strategies: `{', '.join(strategies.keys())}`")
    if theta_values:
        lines.append(f"- theta sweep: `{', '.join(f'{theta:g}' for theta in theta_values)}`")
    lines.append(f"- chunking config: `{json.dumps(retrieval_cfg.get('chunking', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Strategy notes")
    lines.append("")
    for name in strategies:
        lines.append(f"- `{name}`: {strategy_note(name, retrieval_cfg, args)}")
    lines.append("")

    for example_idx, anchor in enumerate(anchors, start=1):
        sents = robust_sentence_split(anchor.report_content)
        rendered = [
            render_strategy(name, strategy, init_error, anchor, translations)
            for name, (strategy, init_error) in strategies.items()
        ]
        theta_rendered = render_theta_sweep(theta_sweep_strategies, anchor, translations)
        lines.append(f"## Example {example_idx}: `{anchor.event_id}`")
        lines.append("")
        lines.append(f"- label: `{anchor.label}`")
        lines.append(f"- claim: {anchor.claim}")
        lines.append(
            f"- anchor: report_id=`{anchor.report_id}`, sent_idx=`{anchor.sent_idx}`, "
            f"score=`{anchor.score:.4f}`, source_sentences=`{len(sents)}`"
        )
        if anchor.domain or anchor.link:
            lines.append(f"- source: {anchor.domain or ''} {anchor.link or ''}".rstrip())
        lines.append("")
        lines.append("### Source context")
        lines.append("")
        for pos, sent, is_anchor in source_context(anchor):
            marker = "**anchor** " if is_anchor else ""
            lines.append(f"- S{pos:02d} {marker}{sent}")
        lines.append("")
        lines.append("### Shape summary")
        lines.append("")
        lines.append("| strategy | status | span | sentences | words | chars |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for item in rendered:
            lines.append(
                f"| `{item.strategy}` | {item.status} | {item.span} | "
                f"{item.sentence_count} | {item.word_count} | {item.char_count} |"
            )
        lines.append("")
        for item in rendered:
            lines.append(f"### {item.strategy}")
            lines.append("")
            if item.status != "ok":
                lines.append(f"`{item.status}`")
                lines.append("")
                continue
            lines.append("English evidence:")
            lines.append("")
            lines.append("> " + truncate(item.text, args.max_display_chars).replace("\n", "\n> "))
            lines.append("")
            if item.zh:
                lines.append("Chinese translation:")
                lines.append("")
                lines.append("> " + truncate(item.zh, args.max_display_chars).replace("\n", "\n> "))
                lines.append("")
            else:
                lines.append("Chinese translation: *(not provided; use --translations or --translation-template)*")
                lines.append("")

        append_theta_sweep_markdown(lines, theta_rendered, theta_values, args.max_display_chars)

        rows.append(
            {
                "event_id": anchor.event_id,
                "claim": anchor.claim,
                "label": anchor.label,
                "report_id": anchor.report_id,
                "sent_idx": anchor.sent_idx,
                "score": anchor.score,
                "strategies": [item.__dict__ for item in rendered],
                "theta_sweep": {
                    strategy_name: {f"{theta:g}": item.__dict__ for theta, item in by_theta.items()}
                    for strategy_name, by_theta in theta_rendered.items()
                },
            }
        )

    append_context_case_study_markdown(lines, context_case_study, args.max_display_chars)

    payload: dict[str, Any] = {
        "examples": rows,
        "context_case_study": (
            {
                **{k: v for k, v in context_case_study.items() if k != "packs"},
                "packs": [strategy_pack_to_dict(pack) for pack in context_case_study["packs"]],
            }
            if context_case_study
            else None
        ),
    }
    return "\n".join(lines).rstrip() + "\n", payload


def write_translation_template(path: Path, payload: dict[str, Any]) -> None:
    template: dict[str, str] = {}
    for row in payload["examples"]:
        for item in row["strategies"]:
            text = str(item.get("text", "")).strip()
            if text:
                template.setdefault(normalized_key(text), "")
        for by_theta in row.get("theta_sweep", {}).values():
            for item in by_theta.values():
                text = str(item.get("text", "")).strip()
                if text:
                    template.setdefault(normalized_key(text), "")
    case_study = payload.get("context_case_study")
    if case_study:
        for pack in case_study.get("packs", []):
            for item in pack.get("items", []):
                text = str(item.get("text", "")).strip()
                if text:
                    template.setdefault(normalized_key(text), "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_cfg(args.experiment, args.override)
    strategy_names = split_strategy_names(args.strategies)
    theta_values = parse_theta_values(args.theta_sweep)
    anchors = select_anchors(
        cfg,
        split=args.split,
        num_examples=args.num_examples,
        anchors_per_example=args.anchors_per_example,
        scan_limit=args.scan_limit,
        event_ids=args.event_id,
        min_source_sentences=args.min_source_sentences,
    )
    translations = load_translations(args.translations)
    strategies = build_strategies(cfg["build"]["retrieval"], strategy_names, args)
    theta_sweep_strategies = build_theta_sweep_strategies(cfg["build"]["retrieval"], theta_values, args)
    context_case_study = None
    if args.case_study_budget > 0:
        case_study_event_id = args.case_study_event_id or anchors[0].event_id
        context_case_study = build_context_case_study(
            cfg,
            split=args.split,
            event_id=case_study_event_id,
            strategies=strategies,
            token_counter=build_token_counter(cfg),
            budget_tokens=args.case_study_budget,
            max_candidates=args.case_study_max_candidates,
            min_source_sentences=args.min_source_sentences,
            translations=translations,
        )
    markdown, payload = markdown_for(
        cfg,
        anchors,
        strategies,
        theta_sweep_strategies,
        theta_values,
        context_case_study,
        translations,
        args,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote Markdown: {args.output}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote JSON: {args.json_output}")

    if args.translation_template is not None:
        write_translation_template(args.translation_template, payload)
        print(f"Wrote translation template: {args.translation_template}")


if __name__ == "__main__":
    main()
