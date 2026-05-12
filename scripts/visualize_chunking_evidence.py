#!/usr/bin/env python3
"""Inspect evidence shape produced by build.retrieval.chunking strategies.

This helper reuses the production ChunkingStrategy implementations. It selects
readable anchor sentences with cheap lexical/BM25 scoring for visualization; it
does not rerun the full dense retrieval + MMR build phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
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


def strategy_config(base_chunking_cfg: dict[str, Any], strategy: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(base_chunking_cfg)
    cfg["strategy"] = strategy
    if args.context_k is not None:
        cfg["context_k"] = args.context_k
    if args.theta is not None:
        cfg["theta"] = args.theta
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


def markdown_for(
    cfg: dict[str, Any],
    anchors: list[Anchor],
    strategies: dict[str, tuple[ChunkingStrategy | None, str | None]],
    translations: dict[str, str],
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]]]:
    retrieval_cfg = cfg["build"]["retrieval"]
    lines: list[str] = []
    rows: list[dict[str, Any]] = []

    lines.append("# Chunking Evidence Visualization")
    lines.append("")
    lines.append(f"- experiment: `{args.experiment}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- strategies: `{', '.join(strategies.keys())}`")
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

        rows.append(
            {
                "event_id": anchor.event_id,
                "claim": anchor.claim,
                "label": anchor.label,
                "report_id": anchor.report_id,
                "sent_idx": anchor.sent_idx,
                "score": anchor.score,
                "strategies": [item.__dict__ for item in rendered],
            }
        )

    return "\n".join(lines).rstrip() + "\n", rows


def write_translation_template(path: Path, rows: list[dict[str, Any]]) -> None:
    template: dict[str, str] = {}
    for row in rows:
        for item in row["strategies"]:
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
    markdown, rows = markdown_for(cfg, anchors, strategies, translations, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote Markdown: {args.output}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote JSON: {args.json_output}")

    if args.translation_template is not None:
        write_translation_template(args.translation_template, rows)
        print(f"Wrote translation template: {args.translation_template}")


if __name__ == "__main__":
    main()
