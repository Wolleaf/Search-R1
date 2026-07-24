#!/usr/bin/env python3
"""Pure-stdlib tests for strict per-question evaluation pairing."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "paired_eval.py"
SPEC = importlib.util.spec_from_file_location("autodl_paired_eval", MODULE_PATH)
assert SPEC and SPEC.loader
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)

FORMAL_CONTROL_DIGEST = "a" * 64
FORMAL_COST_DIGEST = "b" * 64


class PairedEvalTest(unittest.TestCase):

    @staticmethod
    def record(
        stage: str,
        sample_id: int,
        em: int,
        searches: int,
        question: str | None = None,
    ) -> dict[str, object]:
        return {
            "stage": stage,
            "checkpoint_digest": f"digest-{stage}",
            "sample_id": sample_id,
            "question": question or f"Question {sample_id}?",
            "gold_answers": [f"gold-{sample_id}"],
            "raw_trajectory": f"<think>trace {stage} {sample_id}</think>",
            "turns": [{"turn": 0, "think": f"trace {sample_id}"}],
            "extracted_answer": (
                f"gold-{sample_id}" if em else f"wrong-{stage}-{sample_id}"
            ),
            "em": em,
            "executed_search_count": searches,
            "posthoc_utility": em - 0.10 * searches / 4,
        }

    @staticmethod
    def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def make_inputs(self, root: Path) -> dict[str, Path]:
        paths = {
            "control": root / "control.jsonl",
            "cost_aware_old": root / "cost-old.jsonl",
            "cost_aware_gated": root / "cost-gated.jsonl",
        }
        records: dict[str, list[dict[str, object]]] = {
            role: [] for role in paths
        }
        stages = {
            "control": "B",
            "cost_aware_old": "C-old",
            "cost_aware_gated": "C-gated",
        }
        for sample_id in range(128):
            position = sample_id % 4
            baseline_em = int(position in (0, 1))
            candidate_em = int(position in (0, 2))
            baseline_searches = sample_id % 5
            records["control"].append(
                self.record("B", sample_id, baseline_em, baseline_searches)
            )
            for role in ("cost_aware_old", "cost_aware_gated"):
                candidate_searches = (baseline_searches + (1 if role.endswith("gated") else 2)) % 5
                records[role].append(
                    self.record(
                        stages[role], sample_id, candidate_em, candidate_searches
                    )
                )
        for role, path in paths.items():
            self.write_jsonl(path, records[role])
        return paths

    def make_catalog(self, root: Path, control_path: Path) -> Path:
        catalog_path = root / "catalog.jsonl"
        control_records = [
            json.loads(line) for line in control_path.read_text().splitlines()
        ]
        self.write_jsonl(
            catalog_path,
            [
                {
                    "sample_id": record["sample_id"],
                    "question": record["question"],
                    "golden_answers": record["gold_answers"],
                }
                for record in control_records
            ],
        )
        return catalog_path

    def make_formal_inputs(
        self, root: Path
    ) -> tuple[dict[str, Path], Path, Path]:
        paths = self.make_inputs(root)
        identities = {
            "control": ("qwen_native_b", FORMAL_CONTROL_DIGEST),
            "cost_aware_gated": ("qwen_native_c", FORMAL_COST_DIGEST),
        }
        for role, (stage, digest) in identities.items():
            records = [
                json.loads(line) for line in paths[role].read_text().splitlines()
            ]
            for record in records:
                record["stage"] = stage
                record["checkpoint_digest"] = digest
            self.write_jsonl(paths[role], records)
        catalog = self.make_catalog(root, paths["control"])
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "prompt_contract": {"tool_protocol": "qwen35_native"},
                    "artifacts": {
                        "catalog": {
                            "file": "catalog.jsonl",
                            "sha256": hashlib.sha256(
                                catalog.read_bytes()
                            ).hexdigest(),
                        },
                        "val": {
                            "file": "val_128.parquet",
                            "rows": 128,
                            "sha256": "c" * 64,
                            "sample_ids": list(range(128)),
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return paths, catalog, manifest

    @staticmethod
    def args(
        paths: dict[str, Path],
        output_dir: Path,
        *,
        catalog: Path | None = None,
        include_old: bool = True,
        data_manifest: Path | None = None,
        control_digest: str | None = None,
        cost_digest: str | None = None,
    ) -> Namespace:
        return Namespace(
            control=paths["control"],
            cost_aware_old=(paths["cost_aware_old"] if include_old else None),
            cost_aware_gated=paths["cost_aware_gated"],
            catalog=catalog,
            data_manifest=data_manifest,
            expected_control_checkpoint_digest=control_digest,
            expected_cost_aware_gated_checkpoint_digest=cost_digest,
            output_dir=output_dir,
            expected_rows=128,
            cost_lambda=0.10,
            max_searches=4,
            utility_tolerance=1e-6,
        )

    def formal_args(
        self,
        paths: dict[str, Path],
        output_dir: Path,
        catalog: Path,
        manifest: Path,
    ) -> Namespace:
        return self.args(
            paths,
            output_dir,
            catalog=catalog,
            include_old=False,
            data_manifest=manifest,
            control_digest=FORMAL_CONTROL_DIGEST,
            cost_digest=FORMAL_COST_DIGEST,
        )

    def test_writes_paired_outputs_and_four_category_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            output_dir = root / "results"

            PAIRED_EVAL.analyze(self.args(paths, output_dir))

            expected_files = {
                "paired_results.csv",
                "correct_questions.csv",
                "wrong_questions.csv",
                "search_transition.csv",
                "summary.json",
                "summary.md",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected_files
            )
            with (output_dir / "paired_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                paired = list(csv.DictReader(handle))
            self.assertEqual(len(paired), 128)
            self.assertEqual(paired[0]["sample_id"], "0")
            self.assertEqual(paired[0]["cost_aware_gated_category_vs_control"], "both_correct")
            self.assertEqual(
                paired[0]["cost_aware_gated_raw_trajectory"],
                "<think>trace C-gated 0</think>",
            )

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["expected_rows"], 128)
            self.assertEqual(summary["stages"]["control"]["K_correct"], 64)
            self.assertEqual(summary["stages"]["cost_aware_gated"]["K_correct"], 64)
            self.assertEqual(
                summary["stages"]["control"]["S_total_searches"],
                sum(sample_id % 5 for sample_id in range(128)),
            )
            categories = summary["comparisons"]["cost_aware_gated"]["categories"]
            self.assertEqual(set(categories), set(PAIRED_EVAL.CATEGORY_ORDER))
            self.assertTrue(all(values["count"] == 32 for values in categories.values()))
            self.assertIsNotNone(
                summary["stages"]["control"]["mean_searches_given_correct"]
            )
            self.assertIsNotNone(
                summary["stages"]["control"]["mean_searches_given_wrong"]
            )

            with (output_dir / "correct_questions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3 * 64)
            with (output_dir / "wrong_questions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3 * 64)
            with (output_dir / "search_transition.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                transitions = list(csv.DictReader(handle))
            for candidate in ("cost_aware_old", "cost_aware_gated"):
                self.assertEqual(
                    sum(
                        int(row["count"])
                        for row in transitions
                        if row["candidate_role"] == candidate
                    ),
                    128,
                )
            self.assertIn(
                "Search reductions in `both_correct`",
                (output_dir / "summary.md").read_text(),
            )

    def test_two_arm_mode_omits_cost_aware_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            output_dir = root / "two-arm-results"

            PAIRED_EVAL.analyze(
                self.args(paths, output_dir, include_old=False)
            )

            with (output_dir / "paired_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                paired = list(csv.DictReader(handle))
            self.assertEqual(len(paired), 128)
            self.assertFalse(
                any(field.startswith("cost_aware_old_") for field in paired[0])
            )
            self.assertIn("cost_aware_gated_category_vs_control", paired[0])

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(
                list(summary["inputs"]), ["control", "cost_aware_gated"]
            )
            self.assertEqual(
                list(summary["stages"]), ["control", "cost_aware_gated"]
            )
            self.assertEqual(
                list(summary["comparisons"]), ["cost_aware_gated"]
            )
            self.assertNotIn("catalog", summary)

            with (output_dir / "correct_questions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2 * 64)
            with (output_dir / "wrong_questions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2 * 64)
            with (output_dir / "search_transition.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                transitions = list(csv.DictReader(handle))
            self.assertEqual(
                {row["candidate_role"] for row in transitions},
                {"cost_aware_gated"},
            )
            self.assertEqual(sum(int(row["count"]) for row in transitions), 128)

            markdown = (output_dir / "summary.md").read_text()
            self.assertIn("All 2 active stages", markdown)
            self.assertNotIn("cost_aware_old", markdown)

            parsed = PAIRED_EVAL.build_parser().parse_args([
                "--control",
                str(paths["control"]),
                "--cost-aware-gated",
                str(paths["cost_aware_gated"]),
                "--output-dir",
                str(root / "cli-results"),
            ])
            self.assertIsNone(parsed.cost_aware_old)

    def test_catalog_replay_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            catalog = self.make_catalog(root, paths["control"])
            output_dir = root / "catalog-results"

            PAIRED_EVAL.analyze(
                self.args(
                    paths,
                    output_dir,
                    catalog=catalog,
                    include_old=False,
                )
            )

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["catalog"]["path"], str(catalog.resolve()))
            self.assertEqual(
                summary["catalog"]["sha256"],
                hashlib.sha256(catalog.read_bytes()).hexdigest(),
            )
            self.assertEqual(summary["catalog"]["replay_status"], "passed")
            self.assertEqual(
                summary["catalog"]["strict_em_scorer"], "qa_em.em_check"
            )
            self.assertEqual(summary["catalog"]["row_count"], 128)
            self.assertEqual(summary["catalog"]["matched_rows"], 128)

    def test_catalog_replay_accepts_sealed_catalog_superset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            catalog = self.make_catalog(root, paths["control"])
            records = [
                json.loads(line) for line in catalog.read_text().splitlines()
            ]
            records.extend({
                "sample_id": sample_id,
                "question": f"Unused train question {sample_id}?",
                "golden_answers": [f"unused-{sample_id}"],
            } for sample_id in range(128, 640))
            self.write_jsonl(catalog, records)
            output_dir = root / "catalog-superset-results"

            PAIRED_EVAL.analyze(
                self.args(
                    paths,
                    output_dir,
                    catalog=catalog,
                    include_old=False,
                )
            )

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["catalog"]["row_count"], 640)
            self.assertEqual(summary["catalog"]["matched_rows"], 128)

    def test_catalog_replay_rejects_trace_and_identity_tampering(self) -> None:
        cases = (
            (
                "question",
                "cost_aware_gated",
                "question mismatch with catalog",
            ),
            (
                "gold_answers",
                "cost_aware_old",
                "gold_answers mismatch with catalog",
            ),
            (
                "em",
                "cost_aware_gated",
                "strict EM mismatch with catalog replay",
            ),
            (
                "typed_sample_id",
                None,
                "sample-set mismatch between catalog",
            ),
        )
        for mutation, target_role, expected_error in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self.make_inputs(root)
                catalog = self.make_catalog(root, paths["control"])

                if mutation == "typed_sample_id":
                    records = [
                        json.loads(line) for line in catalog.read_text().splitlines()
                    ]
                    records[0]["sample_id"] = "0"
                    self.write_jsonl(catalog, records)
                else:
                    records = [
                        json.loads(line)
                        for line in paths[target_role].read_text().splitlines()
                    ]
                    if mutation == "question":
                        records[0]["question"] = "Tampered question?"
                    elif mutation == "gold_answers":
                        records[0]["gold_answers"] = ["tampered-gold"]
                    else:
                        records[0]["em"] = 0
                        records[0]["posthoc_utility"] = (
                            -0.10 * records[0]["executed_search_count"] / 4
                        )
                    self.write_jsonl(paths[target_role], records)

                with self.assertRaisesRegex(ValueError, expected_error):
                    PAIRED_EVAL.analyze(
                        self.args(
                            paths,
                            root / "tampered-output",
                            catalog=catalog,
                        )
                    )

    def test_formal_two_arm_contract_binds_manifest_stages_and_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, catalog, manifest = self.make_formal_inputs(root)
            output_dir = root / "formal-output"

            PAIRED_EVAL.analyze(
                self.formal_args(paths, output_dir, catalog, manifest)
            )

            summary = json.loads((output_dir / "summary.json").read_text())
            contract = summary["formal_contract"]
            self.assertEqual(contract["mode"], "qwen35_native_v2_b_c")
            self.assertEqual(
                contract["data_manifest"]["sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(contract["val"]["sample_set_status"], "exact")
            self.assertEqual(
                contract["endpoints"]["control"],
                {
                    "stage": "qwen_native_b",
                    "checkpoint_digest": FORMAL_CONTROL_DIGEST,
                },
            )
            self.assertEqual(
                contract["endpoints"]["cost_aware_gated"],
                {
                    "stage": "qwen_native_c",
                    "checkpoint_digest": FORMAL_COST_DIGEST,
                },
            )

    def test_formal_contract_rejects_stage_and_checkpoint_drift(self) -> None:
        cases = (
            ("control", "stage", "B", "formal control stage"),
            (
                "cost_aware_gated",
                "stage",
                "C-gated",
                "formal cost_aware_gated stage",
            ),
            (
                "control",
                "checkpoint_digest",
                "d" * 64,
                "formal control checkpoint_digest",
            ),
            (
                "cost_aware_gated",
                "checkpoint_digest",
                "e" * 64,
                "formal cost_aware_gated checkpoint_digest",
            ),
        )
        for role, field, value, error in cases:
            with self.subTest(
                role=role, field=field
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths, catalog, manifest = self.make_formal_inputs(root)
                records = [
                    json.loads(line)
                    for line in paths[role].read_text().splitlines()
                ]
                for record in records:
                    record[field] = value
                self.write_jsonl(paths[role], records)

                with self.assertRaisesRegex(ValueError, error):
                    PAIRED_EVAL.analyze(
                        self.formal_args(
                            paths, root / "formal-drift", catalog, manifest
                        )
                    )

    def test_formal_contract_rejects_val_sample_set_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, catalog, manifest = self.make_formal_inputs(root)
            payload = json.loads(manifest.read_text())
            payload["artifacts"]["val"]["sample_ids"][-1] = 1000
            manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not exactly match"):
                PAIRED_EVAL.analyze(
                    self.formal_args(
                        paths, root / "sample-drift", catalog, manifest
                    )
                )

    def test_formal_contract_rejects_catalog_path_and_digest_drift(self) -> None:
        for mutation, error in (
            ("path", "catalog path does not match"),
            ("digest", "catalog digest does not match"),
        ):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths, catalog, manifest = self.make_formal_inputs(root)
                selected_catalog = catalog
                if mutation == "path":
                    selected_catalog = root / "other-catalog.jsonl"
                    selected_catalog.write_bytes(catalog.read_bytes())
                else:
                    payload = json.loads(manifest.read_text())
                    payload["artifacts"]["catalog"]["sha256"] = "f" * 64
                    manifest.write_text(
                        json.dumps(payload) + "\n", encoding="utf-8"
                    )

                with self.assertRaisesRegex(ValueError, error):
                    PAIRED_EVAL.analyze(
                        self.formal_args(
                            paths,
                            root / "catalog-drift",
                            selected_catalog,
                            manifest,
                        )
                    )

    def test_formal_contract_arguments_are_all_or_none_and_two_arm_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, catalog, manifest = self.make_formal_inputs(root)
            partial = self.args(
                paths,
                root / "partial",
                catalog=catalog,
                include_old=False,
                data_manifest=manifest,
            )
            with self.assertRaisesRegex(ValueError, "all-or-none"):
                PAIRED_EVAL.analyze(partial)

            wrong_rows = self.formal_args(
                paths, root / "wrong-rows", catalog, manifest
            )
            wrong_rows.expected_rows = 64
            with self.assertRaisesRegex(ValueError, "fixed to sealed val_128"):
                PAIRED_EVAL.analyze(wrong_rows)

            three_arm = self.args(
                paths,
                root / "three-arm",
                catalog=catalog,
                include_old=True,
                data_manifest=manifest,
                control_digest=FORMAL_CONTROL_DIGEST,
                cost_digest=FORMAL_COST_DIGEST,
            )
            with self.assertRaisesRegex(ValueError, "only control"):
                PAIRED_EVAL.analyze(three_arm)

    def test_rejects_duplicate_ids_and_wrong_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            records = [json.loads(line) for line in paths["control"].read_text().splitlines()]
            records[-1]["sample_id"] = records[0]["sample_id"]
            self.write_jsonl(paths["control"], records)
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                PAIRED_EVAL.analyze(self.args(paths, root / "duplicate-output"))

            paths = self.make_inputs(root)
            records = [json.loads(line) for line in paths["control"].read_text().splitlines()]
            self.write_jsonl(paths["control"], records[:-1])
            with self.assertRaisesRegex(ValueError, "expected 128 rows"):
                PAIRED_EVAL.analyze(self.args(paths, root / "short-output"))

    def test_rejects_schema_and_sample_set_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            records = [
                json.loads(line)
                for line in paths["cost_aware_gated"].read_text().splitlines()
            ]
            records[7].pop("turns")
            self.write_jsonl(paths["cost_aware_gated"], records)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                PAIRED_EVAL.analyze(self.args(paths, root / "schema-output"))

            paths = self.make_inputs(root)
            records = [
                json.loads(line)
                for line in paths["cost_aware_gated"].read_text().splitlines()
            ]
            records[-1]["sample_id"] = 1000
            records[-1]["question"] = "Question 1000?"
            records[-1]["gold_answers"] = ["gold-1000"]
            self.write_jsonl(paths["cost_aware_gated"], records)
            with self.assertRaisesRegex(ValueError, "sample-set mismatch"):
                PAIRED_EVAL.analyze(self.args(paths, root / "sample-output"))

    def test_rejects_question_and_utility_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            records = [
                json.loads(line)
                for line in paths["cost_aware_old"].read_text().splitlines()
            ]
            records[12]["question"] = "A different question?"
            self.write_jsonl(paths["cost_aware_old"], records)
            with self.assertRaisesRegex(ValueError, "question mismatch"):
                PAIRED_EVAL.analyze(self.args(paths, root / "question-output"))

            paths = self.make_inputs(root)
            records = [json.loads(line) for line in paths["control"].read_text().splitlines()]
            records[0]["posthoc_utility"] = 123.0
            self.write_jsonl(paths["control"], records)
            with self.assertRaisesRegex(ValueError, "posthoc_utility is inconsistent"):
                PAIRED_EVAL.analyze(self.args(paths, root / "utility-output"))


if __name__ == "__main__":
    unittest.main()
