#!/usr/bin/env python3
"""Pure-stdlib tests for the fixed multihop search-opportunity gate."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from search_r1.trajectory_trace import TraceJsonlWriter

MODULE_PATH = Path(
    __file__).resolve().parents[1] / "search_opportunity_gate.py"
SPEC = importlib.util.spec_from_file_location("autodl_search_opportunity_gate",
                                              MODULE_PATH)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

TEST_SOURCE_BYTES = {
    dataset: f"fixed source bytes for {dataset}\n".encode("ascii")
    for dataset in GATE.DATASETS
}
GATE.SOURCE_FILE_SPECS = {
    dataset: {
        "repo_file": f"{dataset}/dev.jsonl",
        "file": f"sources/{dataset}/dev.jsonl",
        "bytes": len(TEST_SOURCE_BYTES[dataset]),
        "sha256": hashlib.sha256(TEST_SOURCE_BYTES[dataset]).hexdigest(),
    }
    for dataset in GATE.DATASETS
}


class SearchOpportunityGateTest(unittest.TestCase):

    @staticmethod
    def canonical(value: object) -> bytes:
        return (json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) +
                "\n").encode("utf-8")

    @staticmethod
    def write_manifest(path: Path, payload: dict[str, object]) -> None:
        raw = SearchOpportunityGateTest.canonical(payload)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="ascii")

    @staticmethod
    def catalog_record(dataset: str, source_index: int) -> dict[str, object]:
        sample_id = f"{dataset}:test:{source_index}"
        return {
            "sample_id":
            sample_id,
            "data_source":
            dataset,
            "source_split":
            "dev",
            "source_index":
            source_index,
            "question":
            f"Question for {dataset} source {source_index}?",
            "golden_answers": [f"gold answer {dataset} {source_index}"],
            "type":
            "comparison" if dataset == "hotpotqa" else "compositional",
            "level":
            "hard" if dataset == "hotpotqa" else None,
            "supporting_titles": [
                f"{dataset} title {source_index} A",
                f"{dataset} title {source_index} B",
            ],
            "hop_proxy":
            2,
        }

    @staticmethod
    def trace_record(
            catalog: dict[str, object],
            position: int,
            mode: str,
            alignment_mismatch: bool = False,
            trace_question: str | None = None,
            trace_answers: list[str] | None = None) -> dict[str, object]:
        sample_id = str(catalog["sample_id"])
        gold_answers = (trace_answers if trace_answers is not None else list(
            catalog["golden_answers"]))
        if mode == "go":
            em = int(position % 16 < 5)
            searches = position % 5
        elif mode == "no_go":
            em = 0
            searches = 0
        else:
            raise AssertionError(f"unsupported fixture mode: {mode}")
        extracted = gold_answers[0] if em else f"wrong answer {position}"

        events = []
        turns = []
        queries = [
            f"topic {sample_id}",
            f"context {sample_id}",
            f"topic {sample_id}",
            f"followup {sample_id}",
        ]
        doc_ids = [
            ("a", "b"),
            ("c", "d"),
            ("a", "b"),
            ("e", "f"),
        ]
        raw_parts = []
        for turn_index in range(searches):
            query = queries[turn_index]
            documents = [{
                "document_id": f"{sample_id}-{suffix}",
                "document": {
                    "title": f"title-{suffix}"
                },
            } for suffix in doc_ids[turn_index]]
            observation = (f"The evidence contains {gold_answers[0]}."
                           if turn_index == 0 else
                           f"Observation {turn_index} for {sample_id}.")
            event_query = (f"mismatched {query}" if alignment_mismatch
                           and position == 1 and turn_index == 0 else query)
            events.append({
                "turn": turn_index,
                "query": event_query,
                "documents": documents,
                "observation": observation,
            })
            turns.append({
                "turn": len(turns),
                "think": f"Need evidence at search {turn_index}.",
                "action": "search",
                "search_query": query,
                "answer": None,
                "observation": observation,
                "invalid_text": [],
                "valid_action": True,
                "generation_turn": turn_index,
                "environment_action": True,
                "synthetic_environment_action": False,
                "retrieved_docs": documents,
                "retrieval_executed": True,
            })
            raw_parts.append(
                f"<think>Need evidence</think><search>{query}</search>"
                f"<information>{observation}</information>")
        turns.append({
            "turn": len(turns),
            "think": "Return the answer.",
            "action": "answer",
            "search_query": None,
            "answer": extracted,
            "observation": None,
            "invalid_text": [],
            "valid_action": True,
            "generation_turn": searches,
            "environment_action": True,
            "synthetic_environment_action": False,
            "retrieved_docs": [],
            "retrieval_executed": False,
        })
        raw_parts.append(
            f"<think>Return the answer</think><answer>{extracted}</answer>")
        return {
            "sample_id": sample_id,
            "source_index": catalog["source_index"],
            "source_split": "test",
            "data_source": catalog["data_source"],
            "question": trace_question or str(catalog["question"]),
            "gold_answers": gold_answers,
            "raw_trajectory": "".join(raw_parts),
            "turns": turns,
            "extracted_answer": extracted,
            "em": em,
            "executed_search_count": searches,
            "posthoc_utility": em - 0.10 * searches / 4,
            "response_tokens": 32 + searches,
            "response_clipped": False,
            "turns_used": len(turns),
            "invalid_action_count": 0,
            "retrieval_events": events,
            "checkpoint_digest": "a" * 64,
            "max_searches": 4,
        }

    def make_fixture(
        self,
        root: Path,
        *,
        mode: str = "go",
        counts: tuple[int, int] = (128, 128),
        question_mismatch: bool = False,
        gold_mismatch: bool = False,
        alignment_mismatch: bool = False,
        invalid_metadata: bool = False,
        trace_max_searches: int = 4,
        mixed_checkpoint_digests: bool = False,
    ) -> dict[str, Path]:
        data_dir = root / "data"
        trace_dir = root / "trace"
        root.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir()
        trace_dir.mkdir()

        catalog = []
        original_question = None
        original_answers = None
        for dataset, count in zip(GATE.DATASETS, counts):
            for source_index in range(count):
                record = self.catalog_record(dataset, source_index)
                if not catalog:
                    original_question = str(record["question"])
                    original_answers = list(record["golden_answers"])
                    if question_mismatch:
                        record["question"] = "A different catalog question?"
                    if gold_mismatch:
                        record["golden_answers"] = [
                            "a different catalog answer"
                        ]
                    if invalid_metadata:
                        record["level"] = 7
                catalog.append(record)

        catalog_path = data_dir / "catalog.jsonl"
        catalog_path.write_bytes(b"".join(
            self.canonical(record) for record in catalog))
        eval_path = data_dir / "eval_256.parquet"
        eval_path.write_bytes(b"sealed dummy parquet bytes for stdlib tests\n")
        for dataset in GATE.DATASETS:
            source_path = data_dir / GATE.SOURCE_FILE_SPECS[dataset]["file"]
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(TEST_SOURCE_BYTES[dataset])
        manifest = {
            "schema_version": 1,
            "source": {
                "dataset": GATE.DATASET_NAME,
                "revision": GATE.DATASET_REVISION,
                "configs": list(GATE.DATASETS),
                "source_split": "dev",
                "seed": 42,
                "per_config_size": 128,
                "selection_policy": GATE.SELECTION_POLICY,
                "files": {
                    dataset: dict(GATE.SOURCE_FILE_SPECS[dataset])
                    for dataset in GATE.DATASETS
                },
            },
            "artifacts": {
                "eval_parquet": {
                    "file": eval_path.name,
                    "rows": 256,
                    "sha256":
                    hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                },
                "catalog_jsonl": {
                    "file":
                    catalog_path.name,
                    "rows":
                    256,
                    "sha256":
                    hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                },
            },
            "configs": {
                dataset: {
                    "rows":
                    count,
                    "source_indices": [
                        record["source_index"] for record in catalog
                        if record["data_source"] == dataset
                    ],
                }
                for dataset, count in zip(GATE.DATASETS, counts)
            },
            "sample_ids": [record["sample_id"] for record in catalog],
            "overlap_checks": {
                "passed": True,
                "normalized_question_duplicates": 0,
            },
        }
        manifest_path = data_dir / "manifest.json"
        self.write_manifest(manifest_path, manifest)

        trace_path = trace_dir / "eval_predictions.jsonl"
        writer = TraceJsonlWriter(
            trace_path,
            record_type="eval",
            expected_rows=len(catalog),
            run_id="search-opportunity-test",
            stage="search_opportunity",
            schema_version=1,
        )
        try:
            for position, record in enumerate(catalog):
                trace_record = self.trace_record(
                    record,
                    position,
                    mode,
                    alignment_mismatch=alignment_mismatch,
                    trace_question=(original_question if question_mismatch
                                    and position == 0 else None),
                    trace_answers=(original_answers if gold_mismatch
                                   and position == 0 else None),
                )
                trace_record["max_searches"] = trace_max_searches
                if mixed_checkpoint_digests and position == 1:
                    trace_record["checkpoint_digest"] = "b" * 64
                writer.append(trace_record)
            writer.finalize()
        finally:
            writer.close()
        return {
            "trace": trace_path,
            "catalog": catalog_path,
            "data_manifest": manifest_path,
        }

    @staticmethod
    def args(paths: dict[str, Path],
             output_dir: Path,
             extra: list[str] | None = None):
        raw = [
            "--trace",
            str(paths["trace"]),
            "--catalog",
            str(paths["catalog"]),
            "--data-manifest",
            str(paths["data_manifest"]),
            "--output-dir",
            str(output_dir),
            "--expected-checkpoint-digest",
            "a" * 64,
        ]
        if extra:
            raw.extend(extra)
        return GATE.build_parser().parse_args(raw)

    def test_go_path_writes_complete_metrics_trajectories_and_candidates(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_fixture(root)
            output = root / "results"

            summary = GATE.analyze(self.args(paths, output))

            self.assertEqual({path.name
                              for path in output.iterdir()}, GATE.OUTPUT_FILES)
            self.assertEqual(summary["decision"], "GO")
            self.assertEqual(summary["groups"]["overall"]["N"], 256)
            self.assertEqual(summary["groups"]["dataset"]["hotpotqa"]["N"],
                             128)
            self.assertEqual(
                summary["groups"]["dataset"]["2wikimultihopqa"]["N"], 128)
            distribution = summary["groups"]["overall"]["search_distribution"]
            self.assertEqual(
                sum(item["count"] for item in distribution.values()), 256)
            self.assertEqual(
                sum(item["correct_count"] for item in distribution.values()),
                summary["groups"]["overall"]["K_correct"],
            )
            self.assertTrue(all("EM" in item
                                for item in distribution.values()))
            self.assertEqual(summary["thresholds"]["min_correct"], 20)
            self.assertGreaterEqual(
                summary["redundancy"]["strong_plus_medium_candidates"], 5)

            records = [
                json.loads(line)
                for line in (output / "per_question.jsonl").read_text(
                    encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 256)
            self.assertIn("<think>", records[0]["raw_trajectory"])
            self.assertTrue(records[0]["turns"])
            candidate = next(
                record for record in records
                if record["redundancy_evidence"]["classification"] == "strong")
            self.assertIn("duplicate_query",
                          candidate["redundancy_evidence"]["reason_codes"])
            self.assertIn("no_new_document",
                          candidate["redundancy_evidence"]["reason_codes"])
            self.assertTrue(
                candidate["redundancy_evidence"]["gold_seen_before_third"])

            with (output / "strata.csv").open(encoding="utf-8",
                                              newline="") as handle:
                strata = list(csv.DictReader(handle))
            self.assertIn(("overall", "all"), {(row["stratum"], row["value"])
                                               for row in strata})
            self.assertIn(("dataset", "hotpotqa"),
                          {(row["stratum"], row["value"])
                           for row in strata})
            with (output / "redundant_search_candidates.csv").open(
                    encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle))
            self.assertTrue(candidates)
            self.assertTrue(
                all(row["redundancy_classification"] in ("strong", "medium")
                    for row in candidates))

    def test_no_go_path_preserves_denominator_safe_null_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_fixture(root, mode="no_go")
            output = root / "results"

            GATE.analyze(self.args(paths, output))

            decision = json.loads((output / "go_no_go.json").read_text())
            self.assertEqual(decision["decision"], "NO-GO")
            ratio = decision["criteria"][
                "correct_with_two_plus_searches_ratio"]
            self.assertIsNone(ratio["observed"])
            self.assertFalse(ratio["pass"])
            candidate_ratio = decision["criteria"][
                "candidate_ratio_among_three_plus"]
            self.assertIsNone(candidate_ratio["observed"])
            self.assertFalse(candidate_ratio["pass"])

    def test_rejects_wrong_total_and_per_dataset_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short_paths = self.make_fixture(root / "short", counts=(128, 127))
            with self.assertRaisesRegex(ValueError, "exactly 256 rows"):
                GATE.analyze(self.args(short_paths, root / "short-output"))

            uneven_paths = self.make_fixture(root / "uneven",
                                             counts=(129, 127))
            with self.assertRaisesRegex(ValueError,
                                        "exactly 128 unique hotpotqa"):
                GATE.analyze(self.args(uneven_paths, root / "uneven-output"))

    def test_rejects_catalog_question_and_gold_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            question_paths = self.make_fixture(root / "question",
                                               question_mismatch=True)
            with self.assertRaisesRegex(ValueError, "question mismatch"):
                GATE.analyze(
                    self.args(question_paths, root / "question-output"))

            gold_paths = self.make_fixture(root / "gold", gold_mismatch=True)
            with self.assertRaisesRegex(ValueError, "gold-answer mismatch"):
                GATE.analyze(self.args(gold_paths, root / "gold-output"))

    def test_rejects_search_retrieval_event_alignment_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_fixture(root, alignment_mismatch=True)
            with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                GATE.analyze(self.args(paths, root / "output"))

    def test_rejects_trace_config_and_checkpoint_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_paths = self.make_fixture(root / "config",
                                             trace_max_searches=99)
            with self.assertRaisesRegex(ValueError, "trace max_searches"):
                GATE.analyze(self.args(config_paths, root / "config-output"))

            digest_paths = self.make_fixture(root / "digest",
                                             mixed_checkpoint_digests=True)
            with self.assertRaisesRegex(ValueError, "one checkpoint_digest"):
                GATE.analyze(self.args(digest_paths, root / "digest-output"))

            expected_paths = self.make_fixture(root / "expected-digest")
            args = self.args(expected_paths, root / "expected-digest-output")
            args.expected_checkpoint_digest = "b" * 64
            with self.assertRaisesRegex(ValueError, "historical B checkpoint"):
                GATE.analyze(args)

    def test_rejects_data_manifest_revision_digest_and_metadata_mismatch(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision_paths = self.make_fixture(root / "revision")
            manifest = json.loads(revision_paths["data_manifest"].read_text())
            manifest["source"]["revision"] = "0" * 40
            self.write_manifest(revision_paths["data_manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "source/revision"):
                GATE.analyze(
                    self.args(revision_paths, root / "revision-output"))

            digest_paths = self.make_fixture(root / "digest")
            digest_sidecar = digest_paths["data_manifest"].with_suffix(
                ".json.sha256")
            digest_sidecar.write_text(f"{'0' * 64}  manifest.json\n",
                                      encoding="ascii")
            with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
                GATE.analyze(self.args(digest_paths, root / "digest-output"))

            metadata_paths = self.make_fixture(root / "metadata",
                                               invalid_metadata=True)
            with self.assertRaisesRegex(ValueError, "catalog level"):
                GATE.analyze(
                    self.args(metadata_paths, root / "metadata-output"))

            boolean_paths = self.make_fixture(root / "boolean")
            boolean_manifest = json.loads(
                boolean_paths["data_manifest"].read_text())
            boolean_manifest["schema_version"] = True
            self.write_manifest(boolean_paths["data_manifest"],
                                boolean_manifest)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                GATE.analyze(self.args(boolean_paths, root / "boolean-output"))

            float_paths = self.make_fixture(root / "float")
            float_manifest = json.loads(
                float_paths["data_manifest"].read_text())
            float_manifest["source"]["seed"] = 42.0
            self.write_manifest(float_paths["data_manifest"], float_manifest)
            with self.assertRaisesRegex(ValueError, "source/revision"):
                GATE.analyze(self.args(float_paths, root / "float-output"))

            source_paths = self.make_fixture(root / "source")
            source_file = (source_paths["data_manifest"].parent /
                           GATE.SOURCE_FILE_SPECS["hotpotqa"]["file"])
            source_file.write_bytes(source_file.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(ValueError,
                                        "source byte count mismatch"):
                GATE.analyze(self.args(source_paths, root / "source-output"))

    def test_refuses_existing_output_directory_without_modifying_it(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_fixture(root)
            output = root / "results"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError,
                                        "refusing to overwrite"):
                GATE.analyze(self.args(paths, output))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual({path.name
                              for path in output.iterdir()}, {"keep.txt"})

    def test_cli_threshold_override_changes_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_fixture(root)
            strict_args = self.args(paths, root / "strict-output",
                                    ["--min-correct", "256"])
            self.assertEqual(strict_args.max_searches, 4)
            strict = GATE.analyze(strict_args)
            self.assertEqual(strict["decision"], "NO-GO")

            relaxed_args = self.args(
                paths,
                root / "relaxed-output",
                [
                    "--min-correct",
                    "1",
                    "--min-correct-two-plus",
                    "1",
                    "--min-correct-two-plus-ratio",
                    "0.01",
                    "--min-three-plus",
                    "1",
                    "--min-redundancy-candidates",
                    "1",
                    "--min-candidate-ratio",
                    "0.01",
                ],
            )
            relaxed = GATE.analyze(relaxed_args)
            self.assertEqual(relaxed["decision"], "GO")


if __name__ == "__main__":
    unittest.main()
