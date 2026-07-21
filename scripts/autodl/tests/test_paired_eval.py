#!/usr/bin/env python3
"""Pure-stdlib tests for strict per-question evaluation pairing."""

from __future__ import annotations

import csv
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
            "extracted_answer": f"answer-{stage}-{sample_id}",
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

    @staticmethod
    def args(paths: dict[str, Path], output_dir: Path) -> Namespace:
        return Namespace(
            control=paths["control"],
            cost_aware_old=paths["cost_aware_old"],
            cost_aware_gated=paths["cost_aware_gated"],
            output_dir=output_dir,
            expected_rows=128,
            cost_lambda=0.10,
            max_searches=4,
            utility_tolerance=1e-6,
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
