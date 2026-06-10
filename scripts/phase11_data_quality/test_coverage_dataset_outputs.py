from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


materialize = load_script("materialize_coverage_datasets", "scripts/phase11_data_quality/materialize_coverage_datasets.py")
tagger = load_script("tag_source_coverage", "scripts/phase11_data_quality/tag_source_coverage.py")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def coverage_row(event_id: str, label: str, *, split: str = "val") -> dict:
    return {
        "event_id": event_id,
        "split": split,
        "coverage_label": label,
        "rule_coverage_label": label,
        "coverage_score": 0.8 if label == "covered" else 0.4,
        "critical_missing": [],
        "retrieval": {
            "best_embedding": 0.71,
            "best_bm25": 2.5,
            "best_lexical": 0.33,
        },
        "llm_judgment": {"status": "not_requested"},
    }


def test_materialize_liar_raw_policies(tmp_path: Path):
    raw_path = tmp_path / "raw" / "val.json"
    coverage_path = tmp_path / "coverage" / "source_coverage_val.jsonl"
    output_root = tmp_path / "out"
    write_json(
        raw_path,
        [
            {"event_id": "a.json", "claim": "a", "label": "true", "explain": "ea", "reports": []},
            {"event_id": "b.json", "claim": "b", "label": "false", "explain": "eb", "reports": []},
            {"event_id": "c.json", "claim": "c", "label": "half-true", "explain": "ec", "reports": []},
        ],
    )
    write_jsonl(
        coverage_path,
        [
            coverage_row("a.json", "covered"),
            coverage_row("b.json", "weak_covered"),
            coverage_row("c.json", "uncovered"),
        ],
    )

    summary = materialize.materialize_split(
        spec=materialize.DATASET_SPECS["liar_raw"],
        split="val",
        raw_path=raw_path,
        coverage_path=coverage_path,
        output_root=output_root,
        coverage_version="source_coverage_v2",
        policies=["all", "covered", "covered_weak"],
        strict=True,
        sample_limit=None,
        event_ids=None,
        indent=2,
    )

    assert summary["raw_rows"] == 3
    assert summary["coverage_counts"] == {"covered": 1, "uncovered": 1, "weak_covered": 1}
    all_rows = json.loads((output_root / "liar_raw" / "all" / "val.json").read_text(encoding="utf-8"))
    covered_rows = json.loads((output_root / "liar_raw" / "covered" / "val.json").read_text(encoding="utf-8"))
    covered_weak_rows = json.loads((output_root / "liar_raw" / "covered_weak" / "val.json").read_text(encoding="utf-8"))
    assert [row["event_id"] for row in all_rows] == ["a.json", "b.json", "c.json"]
    assert [row["event_id"] for row in covered_rows] == ["a.json"]
    assert [row["event_id"] for row in covered_weak_rows] == ["a.json", "b.json"]
    assert all_rows[0]["coverage_label"] == "covered"
    assert all_rows[0]["coverage"]["bm25_score"] == 2.5
    assert "reports" in all_rows[0]


def test_materialize_rawfc_preserves_schema(tmp_path: Path):
    raw_path = tmp_path / "rawfc" / "val.json"
    coverage_path = tmp_path / "coverage" / "source_coverage_val.jsonl"
    output_root = tmp_path / "out"
    write_json(
        raw_path,
        [
            {"id": 1, "claim": "a", "label": 1, "explanation": "ea", "evidence": ["x"]},
            {"id": 2, "claim": "b", "label": 0, "explanation": "eb", "evidence": ["y"]},
        ],
    )
    write_jsonl(coverage_path, [coverage_row("1", "covered"), coverage_row("2", "uncovered")])

    materialize.materialize_split(
        spec=materialize.DATASET_SPECS["rawfc"],
        split="val",
        raw_path=raw_path,
        coverage_path=coverage_path,
        output_root=output_root,
        coverage_version="source_coverage_v2",
        policies=["all", "covered_weak"],
        strict=True,
        sample_limit=None,
        event_ids=None,
        indent=2,
    )

    all_rows = json.loads((output_root / "rawfc" / "all" / "val.json").read_text(encoding="utf-8"))
    covered_weak_rows = json.loads((output_root / "rawfc" / "covered_weak" / "val.json").read_text(encoding="utf-8"))
    assert all_rows[0]["id"] == 1
    assert all_rows[0]["explanation"] == "ea"
    assert all_rows[0]["evidence"] == ["x"]
    assert [row["id"] for row in covered_weak_rows] == [1]


def test_materialize_strict_missing_and_duplicate_fail(tmp_path: Path):
    raw_path = tmp_path / "raw" / "val.json"
    coverage_path = tmp_path / "coverage" / "source_coverage_val.jsonl"
    write_json(raw_path, [{"event_id": "a.json", "claim": "a", "label": "true", "explain": "ea", "reports": []}])
    write_jsonl(coverage_path, [])
    with pytest.raises(ValueError, match="Missing coverage rows"):
        materialize.materialize_split(
            spec=materialize.DATASET_SPECS["liar_raw"],
            split="val",
            raw_path=raw_path,
            coverage_path=coverage_path,
            output_root=tmp_path / "out",
            coverage_version="source_coverage_v2",
            policies=["all"],
            strict=True,
            sample_limit=None,
            event_ids=None,
            indent=2,
        )

    write_jsonl(coverage_path, [coverage_row("a.json", "covered"), coverage_row("a.json", "covered")])
    with pytest.raises(ValueError, match="Duplicate coverage row"):
        materialize.load_coverage_rows(coverage_path, split="val")


def llm_args(**overrides):
    values = {
        "llm_model_policy": "auto",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_api_key_env": "DEEPSEEK_API_KEY",
        "llm_model": None,
        "llm_pro_model": "deepseek-v4-pro",
        "llm_flash_model": "deepseek-v4-flash",
        "llm_pro_max_reviews": 500,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_llm_preflight_model_policy(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    assert tagger.resolve_llm_run_plan(llm_args(), review_count=0).status == "skipped_no_review_candidates"
    assert tagger.resolve_llm_run_plan(llm_args(), review_count=500).selected_model == "deepseek-v4-pro"
    assert tagger.resolve_llm_run_plan(llm_args(), review_count=501).selected_model == "deepseek-v4-flash"
    assert tagger.resolve_llm_run_plan(llm_args(llm_model="manual-model"), review_count=501).selected_model == "manual-model"


def review_args(**overrides):
    values = {
        "covered_threshold": 0.72,
        "weak_threshold": 0.38,
        "llm_boundary_margin": 0.08,
        "llm_embedding_threshold": 0.75,
        "llm_critical_weak_threshold": 0.60,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_embedding_review_reason_is_boundary_only():
    ranked = [(object(), {"embedding": 0.99, "embedding_scaled": 1.0})]
    far_uncovered = {
        "coverage_label": "uncovered",
        "coverage_score": 0.20,
        "weak_score": 0.20,
        "critical_missing": [],
    }
    assert tagger.llm_review_reasons(rule=far_uncovered, ranked=ranked, args=review_args()) == []

    near_uncovered = {
        "coverage_label": "uncovered",
        "coverage_score": 0.20,
        "weak_score": 0.31,
        "critical_missing": [],
    }
    reasons = tagger.llm_review_reasons(rule=near_uncovered, ranked=ranked, args=review_args())
    assert "near_weak_threshold" in reasons
    assert "embedding_rule_disagreement" in reasons


def test_env_file_key_loading_and_manifest_redaction(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=secret-value\n", encoding="utf-8")
    assert tagger.load_env_file(env_path)
    plan = tagger.resolve_llm_run_plan(llm_args(), review_count=1)
    manifest = tagger.llm_plan_to_manifest(plan, {"review_candidates": 1})
    serialized = json.dumps(manifest)
    assert plan.enabled
    assert manifest["api_key_source"] == "environment"
    assert "secret-value" not in serialized
