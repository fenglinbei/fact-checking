import numpy as np

from fact_checking.oracle_evidence.scorer import build_label_token_ids
from fact_checking.oracle_evidence.search import _records_from_label_logprobs


class _DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, **_kwargs):
        return {"input_ids": [ord(text[-1])]}


def test_build_label_token_ids_uses_rawfc3_letters_only():
    token_ids = build_label_token_ids(_DummyTokenizer(), label_schema="rawfc3")

    assert token_ids == {"A": ord("A"), "B": ord("B"), "C": ord("C")}


def test_label_logprob_records_use_rawfc3_label_ids():
    records = _records_from_label_logprobs(
        np.asarray([[-2.0, -1.5, -0.1]], dtype=np.float32),
        ["C"],
        objective="margin",
        label_schema="rawfc3",
    )

    record = records[0]
    assert set(record["label_logprobs"]) == {"A", "B", "C"}
    assert record["pred_letter"] == "C"
    assert record["pred_id"] == 2
    assert abs(record["margin"] - 1.4) < 1e-5
