from __future__ import annotations

import json
from pathlib import Path

from scripts.phase5_selectors.analyze.summarize_structure_only_clean_results import (
    render_markdown,
    summarize_clean_results,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path) -> Path:
    roots = {
        "s": tmp_path / "runs/liar-structure-only-stateful",
        "o": tmp_path / "runs/liar-structure-only-one-shot",
        "r": tmp_path / "runs/liar-retrieval-order-matched",
        "rawfc": tmp_path / "runs/rawfc-structure-only-clean",
        "scifact": tmp_path / "runs/scifact-structure-only-clean",
        "no_map": tmp_path / "runs/liar-structure-only-no-map",
        "so": tmp_path / "crossovers/matched-verifier-so",
        "rs": tmp_path / "crossovers/matched-verifier-rs",
    }
    payload = {
        "schema_version": "structure-only-clean-results-audit-input-v0.1",
        "input_policy": {
            "fallback_allowed": False,
            "search_allowed": False,
            "require_within_repo": True,
            "forbidden_root_fragments": ["learned_marginal_proxy"],
        },
        "sections": {
            "verifier_crossover_s_o": {
                "kind": "crossover",
                "summary_root": str(roots["so"]),
                "required_summary_root_fragments": ["matched-verifier-so"],
                "summary_schema_version": "structure-only-matched-verifier-crossover-summary-v0.1",
                "checkpoint": "checkpoint-800",
                "split": "val",
                "prompt_cells": {"O": "one_shot__fixed5", "S": "stateful__fixed5"},
                "verifiers": {
                    "V_S": {
                        "run_root": str(roots["s"]),
                        "summary_output_dir": "verifier_s",
                        "required_root_fragments": ["structure-only-stateful"],
                    },
                    "V_O": {
                        "run_root": str(roots["o"]),
                        "summary_output_dir": "verifier_o",
                        "required_root_fragments": ["structure-only-one-shot"],
                    },
                },
            },
            "liar_main": {
                "kind": "standard",
                "run_root": str(roots["s"]),
                "required_root_fragments": ["structure-only-stateful"],
                "checkpoint": "best",
                "label_schema": "liar6",
                "splits": ["test"],
                "expected_num_samples": {"test": 3},
            },
            "rawfc_clean": {
                "kind": "standard",
                "run_root": str(roots["rawfc"]),
                "required_root_fragments": ["rawfc-structure-only-clean"],
                "checkpoint": "best",
                "label_schema": "rawfc3",
                "splits": ["val", "test"],
            },
            "scifact_clean": {
                "kind": "scifact",
                "run_root": str(roots["scifact"]),
                "required_root_fragments": ["scifact-structure-only-clean"],
                "checkpoint": "best",
                "label_schema": "scifact3",
                "expected_trace": str(tmp_path / "traces/structure-only/selection_trace_val.jsonl"),
            },
            "verifier_crossover_r_s": {
                "kind": "crossover",
                "summary_root": str(roots["rs"]),
                "required_summary_root_fragments": ["matched-verifier-rs"],
                "summary_schema_version": "retrieval-stateful-matched-verifier-crossover-summary-v0.1",
                "checkpoint": "checkpoint-800",
                "split": "val",
                "prompt_cells": {"R": "retrieval__fixed5", "S": "stateful__fixed5"},
                "verifiers": {
                    "V_R": {
                        "run_root": str(roots["r"]),
                        "summary_output_dir": "verifier_r",
                        "required_root_fragments": ["retrieval-order-matched"],
                    },
                    "V_S": {
                        "run_root": str(roots["s"]),
                        "summary_output_dir": "verifier_s",
                        "required_root_fragments": ["structure-only-stateful"],
                    },
                },
            },
            "liar_no_map": {
                "kind": "standard",
                "run_root": str(roots["no_map"]),
                "required_root_fragments": ["structure-only-no-map"],
                "checkpoint": "best",
                "label_schema": "liar6",
                "splits": ["val", "test"],
            },
        },
    }
    path = tmp_path / "audit_manifest.json"
    _write_json(path, payload)
    return path


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _marker(root: Path) -> None:
    _write_json(
        root / "train/training_complete.json",
        {"completed": True, "global_step": 1000, "best_score": 0.4},
    )


def _metrics(root: Path, split: str, label_schema: str) -> None:
    _write_json(
        root / f"eval/{split}/best/label_token/metrics.json",
        {
            "num_samples": 3,
            "accuracy": 0.5,
            "macro_precision": 0.51,
            "macro_recall": 0.52,
            "macro_f1": 0.53,
            "parse_error_rate": 0.0,
            "label_schema": label_schema,
            "eval_backend": "label_token_logits",
            "checkpoint": "best",
            "split": split,
        },
    )


def _crossover(
    root: Path,
    *,
    schema: str,
    prompt_cells: dict[str, str],
    verifiers: dict[str, tuple[Path, str]],
) -> None:
    prompt_ids = list(prompt_cells)
    _write_json(
        root / "summary.json",
        {
            "schema_version": schema,
            "status": "complete",
            "split": "val",
            "checkpoint": "checkpoint-800",
            "event_count": 3,
            "prompt_cells": prompt_cells,
            "verifiers": {
                verifier_id: {
                    "run_dir": str(run_root / "train"),
                    "root": str(root / output_dir),
                    "adapter_sha256": str(index + 1) * 64,
                }
                for index, (verifier_id, (run_root, output_dir)) in enumerate(
                    verifiers.items()
                )
            },
            "macro_f1_matrix": {
                verifier_id: {
                    prompt_id: 0.4 + 0.01 * prompt_index
                    for prompt_index, prompt_id in enumerate(prompt_ids)
                }
                for verifier_id in verifiers
            },
            "contrasts": {"matched_mean_minus_crossed_mean": 0.01},
        },
    )


def test_missing_clean_roots_are_pending_without_proxy_fallback(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    old_proxy = tmp_path / "runs/rawfc-learned_marginal_proxy"
    _marker(old_proxy)
    _metrics(old_proxy, "val", "rawfc3")
    _metrics(old_proxy, "test", "rawfc3")
    old_scifact = tmp_path / "runs/scifact-atom-union-fullpool"
    _marker(old_scifact)
    _metrics(old_scifact, "val", "scifact3")

    summary = summarize_clean_results(manifest, repo_root=tmp_path)
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["status"] == "pending"
    assert summary["coverage"] == {
        "total": 6,
        "complete": 0,
        "pending": 6,
        "invalid": 0,
    }
    assert "rawfc-learned_marginal_proxy" not in rendered
    assert "scifact-atom-union-fullpool" not in rendered
    assert "training_complete.json" in render_markdown(summary)
    assert "**PENDING**" in render_markdown(summary)


def test_standard_runs_use_one_normalized_metric_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = _load_manifest(manifest)
    for section_name, label_schema in (("rawfc_clean", "rawfc3"), ("liar_no_map", "liar6")):
        root = Path(payload["sections"][section_name]["run_root"])
        _marker(root)
        _metrics(root, "val", label_schema)
        _metrics(root, "test", label_schema)

    summary = summarize_clean_results(manifest, repo_root=tmp_path)

    assert summary["sections"]["rawfc_clean"]["status"] == "complete"
    assert summary["sections"]["liar_no_map"]["status"] == "complete"
    assert summary["sections"]["rawfc_clean"]["metrics"]["test"]["macro_f1"] == 0.53
    assert summary["status"] == "partial"


def test_liar_main_requires_canonical_test_metrics_not_step_or_best_score(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = _load_manifest(manifest)
    root = Path(payload["sections"]["liar_main"]["run_root"])
    _marker(root)
    _write_json(
        root / "eval/step-800/metrics.json",
        {
            "split": "test",
            "checkpoint": "best",
            "label_schema": "liar6",
            "macro_f1": 0.99,
        },
    )

    pending = summarize_clean_results(manifest, repo_root=tmp_path)

    liar_main = pending["sections"]["liar_main"]
    assert liar_main["status"] == "pending"
    assert liar_main["training"]["best_score"] == 0.4
    assert liar_main["metrics"] == {}
    assert liar_main["missing_artifacts"] == [
        "runs/liar-structure-only-stateful/eval/test/best/label_token/metrics.json"
    ]

    _metrics(root, "test", "liar6")
    complete = summarize_clean_results(manifest, repo_root=tmp_path)
    liar_main = complete["sections"]["liar_main"]
    assert liar_main["status"] == "complete"
    assert liar_main["metrics"]["test"]["num_samples"] == 3
    assert liar_main["metrics"]["test"]["macro_f1"] == 0.53
    assert liar_main["artifacts"]["metrics_test"]["sha256"]
    assert "| liar_main | test | available |" in render_markdown(complete)


def test_scifact_requires_clean_provenance_and_both_exports(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = _load_manifest(manifest)
    spec = payload["sections"]["scifact_clean"]
    root = Path(spec["run_root"])
    _marker(root)
    _metrics(root, "val", "scifact3")
    _write_json(
        root / "eval/test/best/label_token/prediction_manifest.json",
        {
            "prediction_only": True,
            "num_samples": 3,
            "num_labeled_samples": 0,
            "label_schema": "scifact3",
            "checkpoint": "best",
            "split": "test",
        },
    )
    official = {
        "output": str(root / "submission/scifact_submission_val.jsonl"),
        "predictions": str(root / "eval/val/best/label_token/val_predictions.jsonl"),
        "build_jsonl": str(root / "build/build_val.jsonl"),
        "trace": spec["expected_trace"],
        "claim_label": {"accuracy": 0.5, "macro_f1": 0.53, "n": 3},
        "abstract": {"precision": 0.4, "recall": 0.5, "f1": 0.44},
        "abstract_label_only": {"precision": 0.4, "recall": 0.5, "f1": 0.44},
        "sentence": {"precision": 0.4, "recall": 0.5, "f1": 0.44},
        "sentence_selection_only": {"precision": 0.4, "recall": 0.5, "f1": 0.44},
    }
    _write_json(root / "submission/scifact_official_style_metrics_val.json", official)
    for split in ("val", "test"):
        path = root / f"submission/scifact_submission_{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"id": i, "evidence": {}}) + "\n" for i in range(3)), encoding="utf-8")

    complete = summarize_clean_results(manifest, repo_root=tmp_path)
    assert complete["sections"]["scifact_clean"]["status"] == "complete"

    (root / "submission/scifact_submission_test.jsonl").unlink()
    pending = summarize_clean_results(manifest, repo_root=tmp_path)
    assert pending["sections"]["scifact_clean"]["status"] == "pending"
    assert pending["sections"]["scifact_clean"]["missing_artifacts"] == [
        "runs/scifact-structure-only-clean/submission/scifact_submission_test.jsonl"
    ]


def test_crossover_summary_must_point_back_to_exact_clean_runs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = _load_manifest(manifest)
    for section_name in ("verifier_crossover_s_o", "verifier_crossover_r_s"):
        spec = payload["sections"][section_name]
        verifier_roots = {
            verifier_id: (Path(verifier_spec["run_root"]), verifier_spec["summary_output_dir"])
            for verifier_id, verifier_spec in spec["verifiers"].items()
        }
        for run_root, _ in verifier_roots.values():
            _marker(run_root)
        _crossover(
            Path(spec["summary_root"]),
            schema=spec["summary_schema_version"],
            prompt_cells=spec["prompt_cells"],
            verifiers=verifier_roots,
        )

    complete = summarize_clean_results(manifest, repo_root=tmp_path)
    assert complete["sections"]["verifier_crossover_s_o"]["status"] == "complete"
    assert complete["sections"]["verifier_crossover_r_s"]["status"] == "complete"

    so_spec = payload["sections"]["verifier_crossover_s_o"]
    summary_path = Path(so_spec["summary_root"]) / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["verifiers"]["V_O"]["run_dir"] = str(
        tmp_path / "runs/old-proxy/train"
    )
    _write_json(summary_path, summary_payload)
    invalid = summarize_clean_results(manifest, repo_root=tmp_path)
    assert invalid["sections"]["verifier_crossover_s_o"]["status"] == "invalid"
    assert invalid["status"] == "invalid"


def test_forbidden_proxy_root_is_invalid_even_when_manifest_lists_it(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = _load_manifest(manifest)
    proxy_root = tmp_path / "runs/rawfc-learned_marginal_proxy"
    payload["sections"]["rawfc_clean"]["run_root"] = str(proxy_root)
    payload["sections"]["rawfc_clean"]["required_root_fragments"] = [
        "rawfc-learned_marginal_proxy"
    ]
    _write_json(manifest, payload)
    _marker(proxy_root)
    _metrics(proxy_root, "val", "rawfc3")
    _metrics(proxy_root, "test", "rawfc3")

    summary = summarize_clean_results(manifest, repo_root=tmp_path)

    assert summary["status"] == "invalid"
    assert summary["sections"]["rawfc_clean"]["status"] == "invalid"
    assert "forbidden root fragment" in summary["sections"]["rawfc_clean"]["invalid_artifacts"][0]["error"]
