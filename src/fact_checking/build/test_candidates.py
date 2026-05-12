from __future__ import annotations

from copy import deepcopy

import numpy as np

from fact_checking.build.chunking import ChunkRecord, ChunkingStrategy
from fact_checking.build.candidates import (
    PreMMRSample,
    _auto_truncate_evidence,
    _chunk_mmr_config_fingerprint,
    _compute_chunk_mmr_batch,
    _premmr_config_fingerprint,
    _select_candidates_from_chunk_sample,
)


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        del truncation
        ids: list[int] = []
        if add_special_tokens:
            ids.append(1)
        for token in str(text).split():
            ids.append(self._token_id(token))
        return {"input_ids": ids}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {0, 1}:
                continue
            tokens.append(self._id_to_token[token_id])
        return " ".join(tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        del tokenize
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def _token_id(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 2
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


def _base_build_cfg() -> dict:
    return {
        "data": {
            "train_path": "data/raw/LIAR-RAW/train.json",
            "val_path": "data/raw/LIAR-RAW/val.json",
            "test_path": "data/raw/LIAR-RAW/test.json",
        },
        "retrieval": {
            "embedder_model": "/data/models/bge-base-en-v1.5/",
            "device": "cuda",
            "max_length": 256,
            "batch_size": 64,
            "top_k": 16,
            "alpha_dense": 0.70,
            "alpha_lexical": 0.20,
            "alpha_bm25": 0.10,
            "mmr_lambda": 0.70,
            "precision": "bf16",
            "num_gpus": 4,
            "prefetch_size": 200,
            "cpu_workers": 4,
            "chunking": {
                "strategy": "sentence",
                "context_k": 1,
            },
        },
        "prompt": {
            "model_name_or_path": "/data/models/Qwen2.5-7B-Instruct",
            "auto_length": True,
            "max_length": 2048,
            "output_mode": "label_only",
            "label_format": "letter",
        },
    }


def test_premmr_fingerprint_ignores_non_embedding_settings() -> None:
    base = _base_build_cfg()
    changed = deepcopy(base)
    changed["retrieval"]["top_k"] = 24
    changed["retrieval"]["mmr_lambda"] = 0.1
    changed["retrieval"]["alpha_dense"] = 0.5
    changed["retrieval"]["alpha_lexical"] = 0.4
    changed["retrieval"]["alpha_bm25"] = 0.1
    changed["retrieval"]["num_gpus"] = 1
    changed["retrieval"]["prefetch_size"] = 512
    changed["retrieval"]["cpu_workers"] = 8
    changed["retrieval"]["chunking"] = {
        "strategy": "semantic",
        "theta": 0.5,
        "device": "cuda",
        "precision": "bf16",
    }
    changed["prompt"]["max_length"] = 1024

    assert _premmr_config_fingerprint(base) == _premmr_config_fingerprint(changed)


def test_premmr_fingerprint_keeps_embedding_settings() -> None:
    base = _base_build_cfg()
    changed = deepcopy(base)
    changed["retrieval"]["max_length"] = 512

    assert _premmr_config_fingerprint(base) != _premmr_config_fingerprint(changed)


def test_chunk_mmr_fingerprint_tracks_chunking_not_topk() -> None:
    base = _base_build_cfg()
    changed_topk = deepcopy(base)
    changed_topk["retrieval"]["top_k"] = 24
    changed_topk["retrieval"]["mmr_lambda"] = 0.1

    changed_chunking = deepcopy(base)
    changed_chunking["retrieval"]["chunking"] = {
        "strategy": "semantic",
        "theta": 0.5,
    }

    assert _chunk_mmr_config_fingerprint(base) == _chunk_mmr_config_fingerprint(changed_topk)
    assert _chunk_mmr_config_fingerprint(base) != _chunk_mmr_config_fingerprint(changed_chunking)


class _PairFirstTwoChunking(ChunkingStrategy):
    def chunk(self, content: str, sent_idx: int) -> str:
        del content, sent_idx
        return ""

    def chunks_from_presplit(self, sents: list[str]) -> list[ChunkRecord]:
        return [
            ChunkRecord(text=" ".join(sents[:2]), sent_indices=(0, 1)),
            ChunkRecord(text=sents[2], sent_indices=(2,)),
        ]


class _ChunkEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        self.calls.append((tuple(texts), is_query))
        mapping = {
            "Alpha. Beta.": np.array([1.0, 0.0], dtype=np.float32),
            "Gamma.": np.array([0.8, 0.2], dtype=np.float32),
        }
        return np.stack([mapping[text] for text in texts], axis=0)


def test_mmr_selects_over_reembedded_chunks_not_pre_chunk_sentences() -> None:
    content = "Alpha. Beta. Gamma."
    pre = PreMMRSample(
        event_id="e1",
        claim="alpha claim",
        label="true",
        explain="",
        sentences=[
            {"event_id": "e1", "report_id": "r1", "sent_idx": 0, "text": "Alpha.", "raw": {"content": content}},
            {"event_id": "e1", "report_id": "r1", "sent_idx": 1, "text": "Beta.", "raw": {"content": content}},
            {"event_id": "e1", "report_id": "r1", "sent_idx": 2, "text": "Gamma.", "raw": {"content": content}},
        ],
        sent_emb=np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        claim_emb=np.array([1.0, 0.0], dtype=np.float32),
    )
    embedder = _ChunkEmbedder()

    chunk_sample = _compute_chunk_mmr_batch(
        [pre],
        embedder=embedder,
        strategy=_PairFirstTwoChunking(),
    )[0]
    row = _select_candidates_from_chunk_sample(
        chunk_sample,
        top_k=2,
        alpha_dense=1.0,
        alpha_lexical=0.0,
        alpha_bm25=0.0,
        mmr_lambda=1.0,
    )

    assert embedder.calls == [(("Alpha. Beta.", "Gamma."), False)]
    assert [candidate["text"] for candidate in row["candidates"]] == ["Alpha. Beta.", "Gamma."]


def test_auto_truncate_evidence_trims_single_long_evidence_item() -> None:
    tokenizer = _FakeTokenizer()
    evidence = " ".join(f"evidence_token_{idx}" for idx in range(200))

    result = _auto_truncate_evidence(
        claim="short claim",
        evidence_texts=[evidence],
        tokenizer=tokenizer,
        max_length=120,
        output_mode="label_only",
        system_prompt="system prompt",
        row={"label": "true"},
        gold_label="true",
        label_format="letter",
    )

    assert result["evidence_count"] == 1
    assert result["evidence_count_before"] == 1
    assert result["was_truncated"] is True
    assert result["evidence_text_truncated"] is True
    assert result["overflow_after"] is False
    assert result["prompt_token_count"] <= 120 - result["target_token_count"]
    assert "evidence_token_199" not in result["prompt"]
