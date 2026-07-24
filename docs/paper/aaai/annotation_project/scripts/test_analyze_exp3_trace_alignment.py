from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp3_trace_alignment import (  # noqa: E402
    _resolve_bundle_path,
    analyze,
    claim_label_swap_randomization,
    design_weights,
    load_frozen_bundle,
    load_project_annotations,
    logistic_token_sensitivity,
    parse_preference_result,
    parse_transition_result,
    preference_agreement,
    token_robustness_summary,
    unblind_preference_rows,
    validate_project_pair,
)


def _preference_result(
    choice: str,
    *,
    issues: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> str:
    result = [
        {
            "from_name": "overall_preference",
            "value": {"choices": [choice]},
        }
    ]
    if issues:
        result.append(
            {"from_name": "data_issue", "value": {"choices": list(issues)}}
        )
    if notes:
        result.append({"from_name": "notes", "value": {"text": list(notes)}})
    return json.dumps(result)


def _transition_result(validity: str, contribution: str) -> str:
    return json.dumps(
        [
            {
                "from_name": "transition_validity",
                "value": {"choices": [validity]},
            },
            {
                "from_name": "marginal_contribution",
                "value": {"choices": [contribution]},
            },
        ]
    )


def _public_sha(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _annotation(
    blind_id: str,
    choice: str,
    *,
    annotation_id: int,
) -> dict:
    return {
        "blind_task_id": blind_id,
        "annotation_id": annotation_id,
        "completed_by_id": annotation_id,
        "annotator_email": f"rater-{annotation_id}@example.org",
        "created_at": "2026-07-24T00:00:00",
        "updated_at": "2026-07-24T00:00:01",
        "overall_preference": choice,
        "data_issue": [],
        "notes": [],
    }


def _key(
    blind_id: str,
    event_id: str,
    comparison_type: str,
    evi_side: str,
    *,
    difference: int = 0,
) -> dict:
    control_side = "B" if evi_side == "A" else "A"
    return {
        "blind_task_id": blind_id,
        "task_type": "preference",
        "phase": "formal",
        "comparison_type": comparison_type,
        "event_id": event_id,
        "stratum": f"stratum-{event_id}",
        "method_to_side": {
            "evitrace": evi_side,
            "control": control_side,
        },
        "evi_token_count": 100 + difference,
        "control_token_count": 100,
        "token_count_difference_evi_minus_control": difference,
        "same_evidence_set": comparison_type == "order_only",
    }


def _preference_row(
    blind_id: str,
    event_id: str,
    score: int,
    annotator: int,
    difference: int,
) -> dict:
    return {
        "blind_task_id": blind_id,
        "event_id": event_id,
        "stratum": "s-even" if int(event_id[1:]) % 2 == 0 else "s-odd",
        "annotator_index": annotator,
        "evitrace_score": score,
        "collapsed_outcome": (
            "evitrace" if score > 0 else "control" if score < 0 else "tie"
        ),
        "token_count_difference_evi_minus_control": difference,
        "data_issue": [],
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE project (
            id INTEGER PRIMARY KEY,
            title TEXT,
            maximum_annotations INTEGER NOT NULL,
            deleted_at TEXT
        );
        CREATE TABLE task (
            id INTEGER PRIMARY KEY,
            inner_id INTEGER,
            data TEXT NOT NULL,
            project_id INTEGER NOT NULL
        );
        CREATE TABLE htx_user (
            id INTEGER PRIMARY KEY,
            email TEXT,
            first_name TEXT,
            last_name TEXT
        );
        CREATE TABLE task_completion (
            id INTEGER PRIMARY KEY,
            task_id INTEGER,
            project_id INTEGER,
            result TEXT,
            created_at TEXT,
            updated_at TEXT,
            completed_by_id INTEGER,
            was_cancelled INTEGER NOT NULL
        );
        """
    )


class ResultParsingTest(unittest.TestCase):
    def test_bundle_resolution_prefers_exact_formal_artifact_over_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pilot = root / "pilot_preference_tasks.jsonl"
            formal = root / "preference_tasks.jsonl"
            pilot.write_text('{"blind_task_id":"pilot"}\n', encoding="utf-8")
            formal.write_text('{"blind_task_id":"formal"}\n', encoding="utf-8")
            manifest = {
                "artifacts": {
                    "pilot_preference_tasks": {
                        "path": pilot.name,
                        "sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
                    },
                    "preference_tasks": {
                        "path": formal.name,
                        "sha256": hashlib.sha256(formal.read_bytes()).hexdigest(),
                    },
                }
            }
            resolved, digest = _resolve_bundle_path(
                root / "task_manifest.json",
                manifest,
                None,
                "preference_tasks.jsonl",
                ("preference_tasks", "formal_preference"),
            )
            self.assertEqual(resolved, formal.resolve())
            self.assertEqual(digest, hashlib.sha256(formal.read_bytes()).hexdigest())

    def test_preference_parses_by_control_name_and_keeps_optional_issues(self) -> None:
        raw = json.loads(
            _preference_result(
                "prefer_b", issues=("translation",), notes=("check text",)
            )
        )
        raw.reverse()
        parsed = parse_preference_result(raw)
        self.assertEqual(parsed["overall_preference"], "prefer_b")
        self.assertEqual(parsed["data_issue"], ["translation"])
        self.assertEqual(parsed["notes"], ["check text"])

    def test_missing_required_submission_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing overall_preference"):
            parse_preference_result([])
        with self.assertRaisesRegex(ValueError, "Missing transition submissions"):
            parse_transition_result(
                [
                    {
                        "from_name": "transition_validity",
                        "value": {"choices": ["valid"]},
                    }
                ]
            )


class PreferenceUnblindingTest(unittest.TestCase):
    def test_synthetic_ab_reversal_maps_to_opposite_evitrace_scores(self) -> None:
        keys = {
            "left": _key("left", "e1", "main", "A", difference=1),
            "right": _key("right", "e2", "main", "B", difference=-1),
        }
        project = {
            "project_id": 1,
            "records": {
                "left": _annotation("left", "prefer_a", annotation_id=1),
                "right": _annotation("right", "prefer_a", annotation_id=2),
            },
        }
        rows = unblind_preference_rows([project], keys)
        scores = {row["blind_task_id"]: row["evitrace_score"] for row in rows}
        self.assertEqual(scores, {"left": 1, "right": -1})

    def test_tie_is_orientation_invariant_and_counted_in_agreement(self) -> None:
        rows = [
            _preference_row("p1", "e1", 0, 0, 0),
            _preference_row("p1", "e1", 0, 1, 0),
            _preference_row("p2", "e2", 1, 0, 0),
            _preference_row("p2", "e2", -1, 1, 0),
        ]
        result = preference_agreement(rows)
        self.assertEqual(
            result["five_level"]["exact_agreement_count"],
            1,
        )
        self.assertEqual(
            result["collapsed_evitrace_tie_control"]["exact_agreement_count"],
            1,
        )


class SqliteCompletionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        _create_schema(self.connection)
        self.connection.execute(
            "INSERT INTO project VALUES (1, 'formal preference', 1, NULL)"
        )
        self.connection.execute(
            "INSERT INTO htx_user VALUES (7, 'rater@example.org', 'A', 'Rater')"
        )

    def tearDown(self) -> None:
        self.connection.close()

    def test_identical_duplicate_completion_is_collapsed_and_audited(self) -> None:
        task = {
            "blind_task_id": "blind-1",
            "claim_en": "claim",
            "claim_zh": "主张",
            "sequence_a_html": "<p>A</p>",
            "sequence_b_html": "<p>B</p>",
        }
        self.connection.execute(
            "INSERT INTO task VALUES (10, 1, ?, 1)",
            (json.dumps(task),),
        )
        result = _preference_result("tie")
        for annotation_id in (100, 101):
            self.connection.execute(
                "INSERT INTO task_completion VALUES (?, 10, 1, ?, ?, ?, 7, 0)",
                (
                    annotation_id,
                    result,
                    "2026-07-24T00:00:00",
                    f"2026-07-24T00:00:{annotation_id - 100:02d}",
                ),
            )
        project = load_project_annotations(
            self.connection, 1, {"blind-1": task}, "preference"
        )
        self.assertEqual(project["semantic_annotation_count"], 1)
        self.assertEqual(project["active_completion_count"], 2)
        self.assertEqual(project["duplicate_completion_count"], 1)
        self.assertEqual(project["records"]["blind-1"]["annotation_id"], 101)

    def test_missing_completion_is_rejected(self) -> None:
        task = {
            "blind_task_id": "blind-1",
            "claim_en": "claim",
            "claim_zh": "主张",
            "sequence_a_html": "<p>A</p>",
            "sequence_b_html": "<p>B</p>",
        }
        self.connection.execute(
            "INSERT INTO task VALUES (10, 1, ?, 1)",
            (json.dumps(task),),
        )
        with self.assertRaisesRegex(ValueError, "missing submissions"):
            load_project_annotations(
                self.connection, 1, {"blind-1": task}, "preference"
            )

    def test_conflicting_duplicate_completion_is_rejected(self) -> None:
        task = {
            "blind_task_id": "blind-1",
            "claim_en": "claim",
            "claim_zh": "主张",
            "sequence_a_html": "<p>A</p>",
            "sequence_b_html": "<p>B</p>",
        }
        self.connection.execute(
            "INSERT INTO task VALUES (10, 1, ?, 1)",
            (json.dumps(task),),
        )
        for annotation_id, choice in ((100, "prefer_a"), (101, "prefer_b")):
            self.connection.execute(
                "INSERT INTO task_completion VALUES (?, 10, 1, ?, ?, ?, 7, 0)",
                (
                    annotation_id,
                    _preference_result(choice),
                    "2026-07-24T00:00:00",
                    f"2026-07-24T00:00:{annotation_id - 100:02d}",
                ),
            )
        with self.assertRaisesRegex(ValueError, "Conflicting active completions"):
            load_project_annotations(
                self.connection, 1, {"blind-1": task}, "preference"
            )

    def test_same_person_in_both_projects_is_rejected(self) -> None:
        common = {
            "records": {"blind-1": {}},
            "annotator": {"completed_by_id": 7},
        }
        with self.assertRaisesRegex(ValueError, "distinct annotators"):
            validate_project_pair(
                {"project_id": 1, **common},
                {"project_id": 2, **common},
            )


class SensitivityTest(unittest.TestCase):
    def test_twelve_stratum_design_weights_use_manifest_not_sample_counts(self) -> None:
        strata = [
            {
                "stratum": f"s{index}",
                "pool_size": index,
                "sampled_count": 10,
            }
            for index in range(1, 13)
        ]
        rows = [
            {
                "stratum": f"s{index}",
                "event_id": f"e{index}",
            }
            for index in range(1, 13)
        ]
        weights, source = design_weights(
            {"sampling": {"formal": {"main": {"strata": strata}}}},
            "main",
            rows,
        )
        self.assertEqual(source, "manifest_pool_size")
        self.assertAlmostEqual(weights["s1"], 1 / sum(range(1, 13)))
        self.assertAlmostEqual(weights["s12"], 12 / sum(range(1, 13)))
        explicit = [
            {**entry, "design_weight": 1 / 12} for entry in strata
        ]
        weights, source = design_weights(
            {"sampling": {"formal": {"main": {"strata": explicit}}}},
            "main",
            rows,
        )
        self.assertEqual(source, "manifest_design_weight")
        self.assertTrue(all(abs(value - 1 / 12) < 1e-12 for value in weights.values()))

    def test_token_subset_uses_absolute_difference_at_most_64(self) -> None:
        rows = []
        for index, difference in enumerate((-65, -64, 0, 64, 65), start=1):
            for annotator in (0, 1):
                rows.append(
                    _preference_row(
                        f"p{index}",
                        f"e{index}",
                        1 if index % 2 else -1,
                        annotator,
                        difference,
                    )
                )
        manifest = {
            "sampling": {
                "main": {
                    "strata": [
                        {"stratum": "s-even", "pool_size": 2},
                        {"stratum": "s-odd", "pool_size": 3},
                    ]
                }
            }
        }
        result = token_robustness_summary(
            rows, manifest, bootstrap_reps=0, seed=1
        )
        subset = result["pre_registered_absolute_difference_le_64"]["raw"]
        self.assertEqual(subset["claim_count"], 3)
        self.assertEqual(subset["annotation_count"], 6)

    def test_complete_separation_suppresses_logistic_coefficients(self) -> None:
        rows = [
            _preference_row(f"p{i}", f"e{i}", 1, annotator, i * 10)
            for i in range(1, 5)
            for annotator in (0, 1)
        ]
        result = logistic_token_sensitivity(rows)
        self.assertEqual(result["status"], "not_reported_complete_separation")
        self.assertNotIn("coefficients", result)

    def test_logistic_sensitivity_estimates_token_and_annotator_effects(self) -> None:
        rows = []
        for index, difference in enumerate(
            (-128, -96, -64, -32, 0, 32, 64, 96, 128, 160)
        ):
            for annotator in (0, 1):
                score = 1 if (index + annotator) % 3 else -1
                rows.append(
                    _preference_row(
                        f"p{index}",
                        f"e{index}",
                        score,
                        annotator,
                        difference,
                    )
                )
        result = logistic_token_sensitivity(rows)
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["claim_cluster_count"], 10)
        self.assertIn(
            "token_difference_evi_minus_control_per_64",
            result["coefficients"],
        )
        self.assertIn("annotator_1_fixed_effect", result["coefficients"])

    def test_claim_level_swap_flips_both_raters_together(self) -> None:
        rows = [
            _preference_row("p1", "e1", 1, 0, 0),
            _preference_row("p1", "e1", 1, 1, 0),
            _preference_row("p2", "e2", -1, 0, 0),
            _preference_row("p2", "e2", -1, 1, 0),
        ]
        result = claim_label_swap_randomization(
            rows, {"s-odd": 0.5, "s-even": 0.5}, reps=4, seed=1
        )
        self.assertEqual(result["unit"], "claim")
        self.assertEqual(result["mode"], "exact")
        self.assertEqual(result["replicates"], 4)


class EndToEndAnalysisTest(unittest.TestCase):
    def test_small_frozen_bundle_produces_hashed_incomplete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            private.mkdir()
            preference_public = [
                {
                    "blind_task_id": "pref-main",
                    "claim_en": "main claim",
                    "claim_zh": "主比较",
                    "sequence_a_html": "<p>A</p>",
                    "sequence_b_html": "<p>B</p>",
                },
                {
                    "blind_task_id": "pref-order",
                    "claim_en": "order claim",
                    "claim_zh": "顺序比较",
                    "sequence_a_html": "<p>A</p>",
                    "sequence_b_html": "<p>B</p>",
                },
            ]
            transition_public = [
                {
                    "blind_task_id": "trans-change",
                    "claim_en": "change claim",
                    "claim_zh": "变化",
                    "focal_atom_en": "atom",
                    "focal_atom_zh": "原子",
                    "state_legend_html": "<p>legend</p>",
                    "prior_evidence_html": "<p>prior</p>",
                    "current_evidence_html": "<p>current</p>",
                    "proposed_transition": "U -> S",
                },
                {
                    "blind_task_id": "trans-self",
                    "claim_en": "self claim",
                    "claim_zh": "自转移",
                    "focal_atom_en": "atom",
                    "focal_atom_zh": "原子",
                    "state_legend_html": "<p>legend</p>",
                    "prior_evidence_html": "<p>prior</p>",
                    "current_evidence_html": "<p>current</p>",
                    "proposed_transition": "S -> S",
                },
            ]
            artifact_sha = {
                "build": "a" * 64,
                "evitrace": "b" * 64,
                "s4": "c" * 64,
            }
            manifest_artifact_sha = {
                f"test_{name}": value for name, value in artifact_sha.items()
            }
            preference_key = [
                {
                    **_key("pref-main", "event-main", "main", "A", difference=12),
                    "phase": "formal",
                    "split": "test",
                    "gold_label": "true",
                    "complexity": "single",
                    "k_visible": 2,
                    "evi_candidate_uids": ["u1", "u2"],
                    "control_candidate_uids": ["u3", "u4"],
                    "artifact_sha256": artifact_sha,
                    "public_task_sha256": _public_sha(preference_public[0]),
                },
                {
                    **_key("pref-order", "event-order", "order_only", "B"),
                    "phase": "formal",
                    "split": "test",
                    "gold_label": "false",
                    "complexity": "multi",
                    "k_visible": 2,
                    "evi_candidate_uids": ["u1", "u2"],
                    "control_candidate_uids": ["u2", "u1"],
                    "artifact_sha256": artifact_sha,
                    "public_task_sha256": _public_sha(preference_public[1]),
                },
            ]
            transition_key = [
                {
                    "blind_task_id": "trans-change",
                    "phase": "formal",
                    "task_type": "transition",
                    "comparison_type": "transition",
                    "split": "test",
                    "event_id": "event-change",
                    "gold_label": "true",
                    "complexity": "single",
                    "artifact_sha256": artifact_sha,
                    "operation": "OPEN",
                    "atom_id": "A1",
                    "state_before": "U",
                    "state_after": "S",
                    "transition_kind": "change",
                    "step": 1,
                    "candidate_uid": "u5",
                    "public_task_sha256": _public_sha(transition_public[0]),
                },
                {
                    "blind_task_id": "trans-self",
                    "phase": "formal",
                    "task_type": "transition",
                    "comparison_type": "transition",
                    "split": "test",
                    "event_id": "event-self",
                    "gold_label": "false",
                    "complexity": "multi",
                    "artifact_sha256": artifact_sha,
                    "operation": "FALLBACK",
                    "atom_id": "A2",
                    "state_before": "S",
                    "state_after": "S",
                    "transition_kind": "self",
                    "step": 2,
                    "candidate_uid": "u6",
                    "public_task_sha256": _public_sha(transition_public[1]),
                },
            ]

            files = {
                "preference_tasks": root / "preference_tasks.jsonl",
                "transition_tasks": root / "transition_tasks.jsonl",
                "blinding_key": private / "blinding_key.jsonl",
                "transition_key": private / "transition_key.jsonl",
            }
            payloads = {
                "preference_tasks": preference_public,
                "transition_tasks": transition_public,
                "blinding_key": preference_key,
                "transition_key": transition_key,
            }
            for name, path in files.items():
                path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in payloads[name]
                    ),
                    encoding="utf-8",
                )
            manifest = {
                "schema_version": "test",
                "complete": True,
                "artifact_sha256": manifest_artifact_sha,
                "sampling": {
                    "main": {
                        "strata": [
                            {
                                "stratum": "stratum-event-main",
                                "pool_size": 1,
                                "sampled_count": 1,
                            }
                        ]
                    },
                    "order_only": {
                        "strata": [
                            {
                                "stratum": "stratum-event-order",
                                "pool_size": 1,
                                "sampled_count": 1,
                            }
                        ]
                    },
                },
                "artifacts": {
                    name: {
                        "path": str(path.relative_to(root)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rows": len(payloads[name]),
                    }
                    for name, path in files.items()
                },
                "private_key_commitments": {
                    "blinding_key_sha256": hashlib.sha256(
                        files["blinding_key"].read_bytes()
                    ).hexdigest(),
                    "transition_key_sha256": hashlib.sha256(
                        files["transition_key"].read_bytes()
                    ).hexdigest(),
                    "blind_seed_sha256": "d" * 64,
                },
            }
            manifest_path = root / "task_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            db_path = root / "label_studio.sqlite3"
            connection = sqlite3.connect(db_path)
            _create_schema(connection)
            connection.executemany(
                "INSERT INTO htx_user VALUES (?, ?, ?, ?)",
                [
                    (10, "a@example.org", "A", "Rater"),
                    (11, "b@example.org", "B", "Rater"),
                ],
            )
            project_rows = [
                (1, "preference A", 1, None),
                (2, "preference B", 1, None),
                (3, "transition A", 1, None),
                (4, "transition B", 1, None),
            ]
            connection.executemany("INSERT INTO project VALUES (?, ?, ?, ?)", project_rows)
            task_id = 1
            annotation_id = 100
            for project_id, tasks, task_type, user_id in (
                (1, preference_public, "preference", 10),
                (2, preference_public, "preference", 11),
                (3, transition_public, "transition", 10),
                (4, transition_public, "transition", 11),
            ):
                for inner_id, task in enumerate(tasks, start=1):
                    connection.execute(
                        "INSERT INTO task VALUES (?, ?, ?, ?)",
                        (task_id, inner_id, json.dumps(task), project_id),
                    )
                    if task_type == "preference":
                        choice = "prefer_a" if task["blind_task_id"] == "pref-main" else "tie"
                        result = _preference_result(choice)
                    else:
                        result = _transition_result("valid", "clear")
                    connection.execute(
                        "INSERT INTO task_completion VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                        (
                            annotation_id,
                            task_id,
                            project_id,
                            result,
                            "2026-07-24T00:00:00",
                            "2026-07-24T00:00:01",
                            user_id,
                        ),
                    )
                    task_id += 1
                    annotation_id += 1
            connection.commit()
            connection.close()

            output = root / "analysis"
            metrics = analyze(
                db_path,
                manifest_path,
                (1, 2),
                (3, 4),
                output,
                bootstrap_reps=20,
                randomization_reps=20,
                seed=7,
            )
            self.assertEqual(
                metrics["completion_contract"]["formal_claim_counts"],
                {"main": 1, "order_only": 1, "transition": 2},
            )
            self.assertFalse(metrics["completion_contract"]["satisfied"])
            self.assertEqual(
                metrics["preference"]["main"]["raw"]["evitrace_win_count"], 2
            )
            self.assertEqual(
                metrics["transition"]["change_step_validity"][
                    "validity_distribution"
                ]["valid"],
                2,
            )
            output_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(output_manifest["complete"])
            self.assertEqual(
                set(output_manifest["expected_artifacts"]),
                set(output_manifest["artifacts"]),
            )
            for name, record in output_manifest["artifacts"].items():
                self.assertEqual(
                    record["sha256"],
                    hashlib.sha256((output / name).read_bytes()).hexdigest(),
                )
            self.assertEqual(
                len(
                    (output / "raw_double_annotations.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                8,
            )
            self.assertIn(
                "observable evidence selection",
                (output / "paper_insert.md").read_text(encoding="utf-8"),
            )
            table = (output / "paper_table.tex").read_text(encoding="utf-8")
            self.assertIn(r"\begin{table}", table)
            self.assertIn("Order-only", table)
            files["blinding_key"].write_text(
                files["blinding_key"].read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_frozen_bundle(manifest_path)


if __name__ == "__main__":
    unittest.main()
