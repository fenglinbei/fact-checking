from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_exp3_trace_alignment import (  # noqa: E402
    CHANGE_OPERATIONS,
    PREFERENCE_PUBLIC_FIELDS,
    SELF_TRANSITION_OPERATIONS,
    TRANSITION_PUBLIC_FIELDS,
    ExportError,
    TranslationCache,
    WhitespaceTokenCounter,
    export_experiment,
    load_aligned_events,
    recompute_s4_source_score_order,
    sha256_file,
)


LABELS = (
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
)
OPERATIONS = ("OPEN", "CONTRAST", "BRIDGE", "CORROBORATE", "FALLBACK")


def _uid(event_id: str, key: str) -> str:
    return hashlib.sha1(f"{event_id}|{key}".encode()).hexdigest()[:12]


def _states(operation: str) -> tuple[str, str]:
    if operation == "OPEN":
        return "U", "S"
    if operation == "CONTRAST":
        return "S", "C"
    return "S", "S"


def _candidate(event_id: str, index: int) -> dict:
    key = f"canonical {event_id} candidate {index}"
    if index == 0:
        text = "Shared evidence <tag>."
    elif index == 4:
        text = "Check: hidden atom cue\nEvidence <script>alert(1)</script>"
    else:
        text = f"Evidence for {event_id}, item {index} & detail."
    return {
        "candidate_uid": _uid(event_id, key),
        "candidate_key": key,
        "canonical_text": key,
        "text": text,
        "mrec_token_cost": 8 + index,
        "report_id": 1000 + index,
        "source_report": {
            "report_id": 1000 + index,
            "domain": f"https://source{index}.example/article",
        },
        "from_baseline": True,
        "baseline_rank": index + 1,
        "atom_pool_rank": index + 1,
        "union_pool_rank": index + 1,
        "atom_rrf_score": 0.0,
        "atom_route_hit_count": 0,
        "atom_max_route_hybrid": 0.0,
    }


def _event_rows(split: str, index: int) -> tuple[dict, dict, dict]:
    event_id = f"{split}-{index:04d}.json"
    label = LABELS[index % len(LABELS)]
    complexity = "single" if (index // len(LABELS)) % 2 == 0 else "multi"
    atom_count = 1 if complexity == "single" else 2
    atoms = [
        {
            "atom_id": f"A{atom_index + 1}",
            "proposition": f"Focal atom {event_id} number {atom_index + 1}",
            "text": f"Focal atom {event_id} number {atom_index + 1}",
            "type": "other",
        }
        for atom_index in range(atom_count)
    ]
    claim = f"Synthetic claim {event_id} <unsafe>."
    evi_pool = [_candidate(event_id, candidate_index) for candidate_index in range(5)]
    s4_pool = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"candidate_uid", "candidate_key", "mrec_token_cost"}
        }
        for candidate in evi_pool
    ]
    ranked = recompute_s4_source_score_order(s4_pool)
    score_by_key = {
        candidate["canonical_text"]: candidate["atom_union_source_score"]
        for candidate in ranked
    }
    for candidate in s4_pool:
        candidate["atom_union_source_score"] = score_by_key[
            candidate["canonical_text"]
        ]

    # Roughly one ninth are intentionally identical and must be removed from
    # the order-only eligible pool.  Other events use a different selection
    # and order, so main can also exercise unequal token totals.
    selected_indices = [0, 1, 2, 3] if index % 9 == 0 else [4, 0, 1, 2]
    operation = OPERATIONS[index % len(OPERATIONS)]
    before, after = _states(operation)
    steps = []
    for step_number, candidate_index in enumerate(selected_indices + [3], start=1):
        candidate = evi_pool[candidate_index]
        if step_number == 1:
            step_operation = operation
            state_before, state_after = before, after
        else:
            step_operation = "BRIDGE"
            state_before = state_after = after
        steps.append(
            {
                "step": step_number,
                "candidate_uid": candidate["candidate_uid"],
                "candidate_key": candidate["candidate_key"],
                "evidence_text": candidate["text"],
                "atom_id": "A1",
                "atom_text": atoms[0]["proposition"],
                "operation": step_operation,
                "state_before": state_before,
                "state_after": state_after,
                "relation": "support",
                "directness": "direct",
                "map_confidence": 0.9,
            }
        )
    build_candidates = [evi_pool[candidate_index] for candidate_index in selected_indices]
    build = {
        "event_id": event_id,
        "claim": claim,
        "label": label,
        "gold_label": label,
        "evidence_count": 3,
        "evidence_count_before": 4,
        "candidates": build_candidates,
    }
    evitrace = {
        "event_id": event_id,
        "claim": claim,
        "gold_label": label,
        "claim_atoms": atoms,
        "candidate_pool": evi_pool,
        "mrec_steps": steps,
    }
    s4 = {
        "event_id": event_id,
        "claim": claim,
        "label": label,
        "gold_label": label,
        "candidate_pool": s4_pool,
        "selected_candidates": ranked,
    }
    return build, evitrace, s4


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_artifacts(root: Path, split: str, count: int) -> tuple[Path, Path, Path]:
    triples = [_event_rows(split, index) for index in range(count)]
    paths = (
        root / f"build_{split}.jsonl",
        root / f"evitrace_{split}.jsonl",
        root / f"s4_{split}.jsonl",
    )
    for path, position in zip(paths, range(3)):
        _write_jsonl(path, [triple[position] for triple in triples])
    return paths


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _complete_translation_cache(
    test_count: int,
    val_count: int,
) -> TranslationCache:
    translations: dict[str, str] = {
        "Shared evidence <tag>.": "共享证据。",
        "Evidence <script>alert(1)</script>": "已转义的证据。",
    }
    for split, count in (("test", test_count), ("val", val_count)):
        for index in range(count):
            event_id = f"{split}-{index:04d}.json"
            translations[f"Synthetic claim {event_id} <unsafe>."] = (
                f"合成声明 {event_id}。"
            )
            for atom_index in range(2):
                atom = f"Focal atom {event_id} number {atom_index + 1}"
                translations[atom] = f"焦点原子 {event_id} {atom_index + 1}"
            for candidate_index in (1, 2, 3):
                evidence = (
                    f"Evidence for {event_id}, item {candidate_index} & detail."
                )
                translations[evidence] = (
                    f"证据 {event_id} {candidate_index}。"
                )
    return TranslationCache(translations)


class ExportTraceAlignmentTest(unittest.TestCase):
    def test_s4_audit_rejects_order_drift_and_duplicate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            build, evitrace, s4 = _make_artifacts(root, "test", 3)

            rows = _load_jsonl(s4)
            rows[1]["selected_candidates"][0], rows[1]["selected_candidates"][1] = (
                rows[1]["selected_candidates"][1],
                rows[1]["selected_candidates"][0],
            )
            _write_jsonl(s4, rows)
            with self.assertRaisesRegex(ExportError, "stored S4 order differs"):
                load_aligned_events(
                    build,
                    evitrace,
                    s4,
                    split="test",
                )

            build, evitrace, s4 = _make_artifacts(root, "test", 3)
            rows = _load_jsonl(evitrace)
            rows[0]["candidate_pool"][1]["candidate_uid"] = rows[0][
                "candidate_pool"
            ][0]["candidate_uid"]
            _write_jsonl(evitrace, rows)
            with self.assertRaisesRegex(ExportError, "duplicate EviTrace candidate_uid"):
                load_aligned_events(
                    build,
                    evitrace,
                    s4,
                    split="test",
                )

    def test_full_export_quotas_blinding_hashes_and_reproducibility(self) -> None:
        test_count = 720
        val_count = 240
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            build_test, evi_test, s4_test = _make_artifacts(
                root, "test", test_count
            )
            build_val, evi_val, s4_val = _make_artifacts(root, "val", val_count)
            preference_xml = root / "preference.xml"
            transition_xml = root / "transition.xml"
            preference_xml.write_text("<View></View>\n", encoding="utf-8")
            transition_xml.write_text("<View></View>\n", encoding="utf-8")
            translations = _complete_translation_cache(test_count, val_count)

            outputs = [root / "out-a", root / "out-b"]
            manifests = []
            for output in outputs:
                manifests.append(
                    export_experiment(
                        build_test_path=build_test,
                        evitrace_test_path=evi_test,
                        s4_test_path=s4_test,
                        build_val_path=build_val,
                        evitrace_val_path=evi_val,
                        s4_val_path=s4_val,
                        output_dir=output,
                        blind_secret=b"fixed-test-blinding-secret",
                        translation_cache=translations,
                        token_counter=WhitespaceTokenCounter(),
                        expected_test_events=None,
                        expected_val_events=None,
                        test_atom_parse_path=None,
                        preference_config_path=preference_xml,
                        transition_config_path=transition_xml,
                    )
                )

            manifest = manifests[0]
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["translation_cache"]["miss_count"], 0)
            self.assertEqual(
                manifest["config_sha256"],
                {
                    "preference": sha256_file(preference_xml),
                    "transition": sha256_file(transition_xml),
                },
            )
            self.assertEqual(
                manifest["counts"]["formal_preference"],
                200,
            )
            self.assertEqual(manifest["counts"]["formal_transition"], 100)
            self.assertEqual(manifest["counts"]["pilot_preference"], 30)
            self.assertEqual(manifest["counts"]["pilot_transition"], 15)

            preference = _load_jsonl(outputs[0] / "preference_tasks.jsonl")
            transition = _load_jsonl(outputs[0] / "transition_tasks.jsonl")
            private_preference = _load_jsonl(
                outputs[0] / "private/blinding_key.jsonl"
            )
            private_transition = _load_jsonl(
                outputs[0] / "private/transition_key.jsonl"
            )
            self.assertTrue(
                all(set(row) == PREFERENCE_PUBLIC_FIELDS for row in preference)
            )
            self.assertTrue(
                all(set(row) == TRANSITION_PUBLIC_FIELDS for row in transition)
            )
            public_serialized = json.dumps(
                preference + transition, ensure_ascii=False
            )
            for forbidden in (
                "candidate_uid",
                "gold_label",
                "mrec_",
                "source_score",
                "Check:",
                "<script>",
            ):
                self.assertNotIn(forbidden, public_serialized)
            self.assertIn("<unsafe>", public_serialized)
            self.assertIn("&lt;script&gt;", public_serialized)
            self.assertIn("共享证据。", public_serialized)

            formal_private = [
                row for row in private_preference if row["phase"] == "formal"
            ]
            main = [
                row for row in formal_private if row["comparison_type"] == "main"
            ]
            order = [
                row
                for row in formal_private
                if row["comparison_type"] == "order_only"
            ]
            self.assertEqual(
                Counter(row["stratum"] for row in main),
                Counter(
                    {
                        f"{label}|{complexity}": 10
                        for label in LABELS
                        for complexity in ("single", "multi")
                    }
                ),
            )
            self.assertEqual(
                Counter(row["method_to_side"]["evitrace"] for row in main),
                Counter({"A": 60, "B": 60}),
            )
            self.assertEqual(
                Counter(row["method_to_side"]["evitrace"] for row in order),
                Counter({"A": 40, "B": 40}),
            )
            self.assertEqual(
                Counter(row["complexity"] for row in order),
                Counter({"single": 40, "multi": 40}),
            )
            for row in formal_private:
                self.assertEqual(row["k_visible"], 3)
                self.assertEqual(row["k_selected"], 4)
                self.assertEqual(len(row["evi_candidate_uids"]), 3)
                self.assertEqual(len(row["control_candidate_uids"]), 3)
                self.assertNotIn("annotator_order_rank", row)
            for row in order:
                self.assertEqual(
                    set(row["evi_candidate_uids"]),
                    set(row["control_candidate_uids"]),
                )
                self.assertNotEqual(
                    row["evi_candidate_uids"],
                    row["control_candidate_uids"],
                )
                self.assertEqual(row["evi_token_count"], row["control_token_count"])

            formal_transitions = [
                row for row in private_transition if row["phase"] == "formal"
            ]
            self.assertEqual(
                Counter(row["operation"] for row in formal_transitions),
                Counter(
                    {
                        "OPEN": 40,
                        "CONTRAST": 20,
                        "BRIDGE": 20,
                        "CORROBORATE": 10,
                        "FALLBACK": 10,
                    }
                ),
            )
            self.assertEqual(
                len({row["event_id"] for row in formal_transitions}),
                100,
            )
            for row in formal_transitions:
                if row["operation"] in CHANGE_OPERATIONS:
                    self.assertEqual(row["transition_kind"], "change")
                if row["operation"] in SELF_TRANSITION_OPERATIONS:
                    self.assertEqual(row["transition_kind"], "self")

            main_ids = {row["event_id"] for row in main}
            order_ids = {row["event_id"] for row in order}
            transition_ids = {row["event_id"] for row in formal_transitions}
            self.assertFalse(main_ids & order_ids)
            self.assertFalse(main_ids & transition_ids)
            self.assertFalse(order_ids & transition_ids)

            for name, entry in manifest["artifacts"].items():
                artifact_path = outputs[0] / entry["path"]
                self.assertEqual(entry["sha256"], sha256_file(artifact_path), name)
            for relative_path in (
                "preference_tasks.jsonl",
                "transition_tasks.jsonl",
                "pilot_preference_tasks.jsonl",
                "pilot_transition_tasks.jsonl",
                "private/blinding_key.jsonl",
                "private/transition_key.jsonl",
                "translation_inventory.jsonl",
                "translation_requests.jsonl",
                "sampling_stats.json",
            ):
                self.assertEqual(
                    (outputs[0] / relative_path).read_bytes(),
                    (outputs[1] / relative_path).read_bytes(),
                    relative_path,
                )

    def test_missing_translation_cache_fails_manifest_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            build_test, evi_test, s4_test = _make_artifacts(root, "test", 72)
            build_val, evi_val, s4_val = _make_artifacts(root, "val", 72)
            preference_xml = root / "preference.xml"
            transition_xml = root / "transition.xml"
            preference_xml.write_text("<View></View>\n", encoding="utf-8")
            transition_xml.write_text("<View></View>\n", encoding="utf-8")
            manifest = export_experiment(
                build_test_path=build_test,
                evitrace_test_path=evi_test,
                s4_test_path=s4_test,
                build_val_path=build_val,
                evitrace_val_path=evi_val,
                s4_val_path=s4_val,
                output_dir=root / "out",
                blind_secret=b"secret",
                n_main=12,
                n_order=12,
                n_transition=10,
                n_pilot_main=12,
                n_pilot_order=6,
                n_pilot_transition=5,
                translation_cache=TranslationCache(),
                token_counter=WhitespaceTokenCounter(),
                expected_test_events=None,
                expected_val_events=None,
                test_atom_parse_path=None,
                preference_config_path=preference_xml,
                transition_config_path=transition_xml,
            )
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"], "translation_cache_incomplete")
            self.assertGreater(manifest["translation_cache"]["miss_count"], 0)
            self.assertGreater(
                len(_load_jsonl(root / "out/translation_requests.jsonl")),
                0,
            )


if __name__ == "__main__":
    unittest.main()
