from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_strict_same_set_shuffles.py")
SPEC = importlib.util.spec_from_file_location("build_strict_same_set_shuffles", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
strict_shuffle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = strict_shuffle
SPEC.loader.exec_module(strict_shuffle)


class MistralCommonFakeTokenizer:
    eos_token_id = 0

    @staticmethod
    def _ids(text: str) -> list[int]:
        output = []
        for token in str(text).split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            output.append(int.from_bytes(digest[:4], byteorder="big") + 1)
        return output

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        del truncation
        ids = self._ids(text)
        if add_special_tokens:
            ids.insert(0, self.eos_token_id)
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        **_: object,
    ) -> str | list[int]:
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return self._ids(rendered) if tokenize else rendered


class MistralCommonOrderContextTokenizer(MistralCommonFakeTokenizer):
    """Inject an order-dependent token while preserving sequence length."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        **kwargs: object,
    ) -> str | list[int]:
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if not tokenize:
            return rendered
        ids = self._ids(rendered)
        alpha_first = rendered.index("Alpha evidence") < rendered.index("Beta evidence")
        return ids + ([900_001] if alpha_first else [900_002])


PROMPT_CFG = {
    "auto_length": False,
    "max_length": 512,
    "output_mode": "label_only",
    "label_format": "letter",
    "label_schema": "liar6",
    "chat_template": {
        "mode": "tokenizer_default",
        "add_generation_prompt": True,
        "template_kwargs": {},
    },
}


def test_default_seeds_are_deterministic_nonidentity_strict_permutations() -> None:
    original = ("a", "b", "c", "d")
    for seed in range(5):
        first = strict_shuffle._deterministic_nonidentity_shuffle(
            original, event_id="event-1", seed=seed
        )
        second = strict_shuffle._deterministic_nonidentity_shuffle(
            original, event_id="event-1", seed=seed
        )
        assert first == second
        assert first != original
        assert len(first) == len(original)
        assert set(first) == set(original)

    assert strict_shuffle._event_random_seed(0, "event-1") != (
        strict_shuffle._event_random_seed(1, "event-1")
    )


def test_event_shuffle_preserves_visible_set_content_labels_and_token_multiset() -> None:
    tokenizer = MistralCommonFakeTokenizer()
    source = _source_build_row(tokenizer, include_invisible_tail=True)
    source["mrec_steps"] = [{"stale": "must not be replayed"}]

    payloads = strict_shuffle._build_event_shuffles(
        source_row=source,
        split="val",
        seeds=(0, 1, 2, 3, 4),
        tokenizer=tokenizer,
        prompt_cfg=PROMPT_CFG,
        max_length=512,
    )

    assert set(payloads) == {f"shuffle_seed{seed}" for seed in range(5)}
    source_visible = source["candidates"][: source["evidence_count"]]
    source_by_uid = {candidate["candidate_uid"]: candidate for candidate in source_visible}
    fingerprints = None
    for payload in payloads.values():
        row = payload["build_row"]
        sidecar = payload["sidecar"]
        assert row["evidence_count"] == 3
        assert row["evidence_count_before"] == 3
        assert len(row["candidates"]) == 3
        assert row["was_truncated"] is False
        assert row["evidence_text_truncated"] is False
        assert row["target"] == source["target"]
        assert row["label"] == source["label"]
        assert row["gold_label"] == source["gold_label"]
        assert row["prompt_token_count"] == source["prompt_token_count"]
        assert sidecar["order_changed"] is True
        assert sidecar["verifier_prompt_changed"] is True
        assert sidecar["ordered_candidate_uids"] != sidecar["source_ordered_candidate_uids"]
        assert set(sidecar["ordered_candidate_uids"]) == set(source_by_uid)
        assert "mrec_steps" not in row
        assert row["strict_same_set_shuffle"]["prompt_token_multiset_fingerprint"] == (
            sidecar["prompt_token_multiset_fingerprint"]
        )
        for candidate in row["candidates"]:
            assert candidate == source_by_uid[candidate["candidate_uid"]]

        current = {
            key: sidecar[key]
            for key in (
                "uid_set_fingerprint",
                "uid_text_fingerprint",
                "candidate_block_fingerprint",
                "evidence_token_content_fingerprint",
                "target_label_fingerprint",
                "prompt_token_multiset_fingerprint",
            )
        }
        fingerprints = fingerprints or current
        assert current == fingerprints


def test_source_evidence_text_truncation_fails_closed() -> None:
    tokenizer = MistralCommonFakeTokenizer()
    source = _source_build_row(tokenizer)
    source["evidence_text_truncated"] = True

    with pytest.raises(
        strict_shuffle.StrictShuffleError,
        match="evidence_text_truncated must be exactly false",
    ):
        strict_shuffle._build_event_shuffles(
            source_row=source,
            split="val",
            seeds=(0,),
            tokenizer=tokenizer,
            prompt_cfg=PROMPT_CFG,
            max_length=512,
        )


def test_overflow_fails_instead_of_retruncating() -> None:
    tokenizer = MistralCommonFakeTokenizer()
    source = _source_build_row(tokenizer)

    with pytest.raises(strict_shuffle.StrictShuffleError, match="exceeds max_length"):
        strict_shuffle._build_event_shuffles(
            source_row=source,
            split="val",
            seeds=(0,),
            tokenizer=tokenizer,
            prompt_cfg={**PROMPT_CFG, "max_length": 10},
            max_length=10,
        )


def test_order_dependent_prompt_token_content_drift_fails_closed() -> None:
    tokenizer = MistralCommonOrderContextTokenizer()
    source = _source_build_row(tokenizer, n_visible=2)

    with pytest.raises(
        strict_shuffle.StrictShuffleError,
        match="no non-identity permutation satisfied the strict token contract",
    ):
        strict_shuffle._build_event_shuffles(
            source_row=source,
            split="val",
            seeds=(0,),
            tokenizer=tokenizer,
            prompt_cfg=PROMPT_CFG,
            max_length=512,
        )


def test_rejection_sampling_finds_a_token_contract_preserving_order() -> None:
    tokenizer = MistralCommonOrderContextTokenizer()
    source = _source_build_row(tokenizer, n_visible=3)

    payloads = strict_shuffle._build_event_shuffles(
        source_row=source,
        split="val",
        seeds=(0, 1, 2, 3, 4),
        tokenizer=tokenizer,
        prompt_cfg=PROMPT_CFG,
        max_length=512,
    )

    assert any(
        payload["sidecar"]["shuffle_rejected_order_count"] > 0
        for payload in payloads.values()
    )
    for payload in payloads.values():
        sidecar = payload["sidecar"]
        assert sidecar["order_changed"] is True
        assert sidecar["shuffle_order_attempt"] == (
            sidecar["shuffle_rejected_order_count"] + 1
        )


def test_stream_builder_writes_five_auditable_arms(tmp_path: Path) -> None:
    tokenizer = MistralCommonFakeTokenizer()
    rows = [
        _source_build_row(tokenizer, event_id="event-1"),
        _source_build_row(tokenizer, event_id="event-2"),
    ]
    source_path = tmp_path / "build_val.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("build:\n  prompt:\n    max_length: 512\n", encoding="utf-8")
    output_dir = tmp_path / "controls"

    manifest = strict_shuffle.build_strict_same_set_shuffles(
        source_build_path=source_path,
        config_path=config_path,
        split="val",
        output_dir=output_dir,
        seeds=(0, 1, 2, 3, 4),
        tokenizer=tokenizer,
        prompt_cfg=PROMPT_CFG,
    )

    assert manifest["n_events"] == 2
    assert set(manifest["arms"]) == {f"shuffle_seed{seed}" for seed in range(5)}
    assert manifest["contract"]["auto_length"] is False
    for seed in range(5):
        arm = f"shuffle_seed{seed}"
        build_path = output_dir / arm / "build_val.jsonl"
        sidecar_path = output_dir / arm / "strict_same_set_shuffle_val.jsonl"
        summary_path = output_dir / arm / "summary_val.json"
        assert len(build_path.read_text(encoding="utf-8").splitlines()) == 2
        assert len(sidecar_path.read_text(encoding="utf-8").splitlines()) == 2
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["contract_checks"]["eligible_orders_changed"] is True
        assert summary["contract_checks"]["prompt_token_multiset_equal"] is True
        assert summary["metrics"]["order_changed_rate_among_eligible"] == 1.0


def test_invalid_seed_contract_is_rejected() -> None:
    with pytest.raises(strict_shuffle.StrictShuffleError, match="duplicates"):
        strict_shuffle._normalize_seeds((0, 0))
    with pytest.raises(strict_shuffle.StrictShuffleError, match="non-negative"):
        strict_shuffle._normalize_seeds((-1,))
    with pytest.raises(strict_shuffle.StrictShuffleError, match="at least one"):
        strict_shuffle._normalize_seeds(())


def _source_build_row(
    tokenizer: MistralCommonFakeTokenizer,
    *,
    event_id: str = "event-1",
    n_visible: int = 3,
    include_invisible_tail: bool = False,
) -> dict:
    candidates = [
        _candidate("a", "Alpha evidence supports the first detail."),
        _candidate("b", "Beta evidence qualifies the second detail."),
        _candidate("c", "Gamma evidence supplies final context."),
    ][:n_visible]
    row = strict_shuffle.build_training_row(
        {
            "event_id": event_id,
            "claim": "A compact factual claim.",
            "label": "false",
            "label_schema": "liar6",
            "explain": "",
            "candidates": candidates,
        },
        tokenizer,
        PROMPT_CFG,
    )
    row["evidence_count_before"] = n_visible
    row["evidence_text_truncated"] = False
    if include_invisible_tail:
        row["candidates"].append(_candidate("tail", "This tail is not verifier visible."))
        row["evidence_count_before"] = n_visible + 1
        row["was_truncated"] = True
    return row


def _candidate(uid: str, text: str) -> dict:
    return {
        "candidate_uid": uid,
        "candidate_key": f"key-{uid}",
        "evidence_id": f"E-{uid}",
        "text": text,
        "nested_metadata": {"uid": uid, "values": [1, 2, 3]},
    }
