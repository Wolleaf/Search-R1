import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.data_process import nq_small


class FakeDataset:

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])

    def map(self, function, with_indices):
        assert with_indices
        mapped = []
        for index, row in enumerate(self.rows):
            merged = dict(row)
            merged.update(function(row, index))
            mapped.append(merged)
        return FakeDataset(mapped)

    def to_parquet(self, path):
        Path(path).write_text(json.dumps(self.rows, sort_keys=True), encoding="utf-8")


def make_rows(prefix, count):
    return [{
        "question": f"{prefix} question {index}",
        "golden_answers": [f"answer {index}"],
    } for index in range(count)]


class NqSmallTest(unittest.TestCase):

    def test_selection_is_deterministic_and_has_no_question_overlap(self):
        train = FakeDataset(make_rows("train", 12))
        test_rows = make_rows("test", 8)
        test_rows[0]["question"] = "  TRAIN   QUESTION 0  "
        test = FakeDataset(test_rows)

        first = nq_small.select_split_indices(train, test, 5, 3, 4, seed=42)
        second = nq_small.select_split_indices(train, test, 5, 3, 4, seed=42)

        self.assertEqual(first, second)
        overlaps = nq_small.assert_no_overlap(train, test, first)
        self.assertTrue(all(count == 0 for count in overlaps.values()))

    def test_record_matches_search_r1_schema(self):
        record = nq_small.make_record({
            "question": "Who wrote Hamlet",
            "golden_answers": ["William Shakespeare", "Shakespeare"],
        }, "val", 17)

        self.assertEqual(record["data_source"], "nq")
        self.assertEqual(record["ability"], "fact-reasoning")
        self.assertEqual(record["extra_info"], {"split": "val", "index": 17})
        self.assertEqual(record["reward_model"]["ground_truth"]["target"],
                         ["William Shakespeare", "Shakespeare"])
        self.assertTrue(record["prompt"][0]["content"].endswith("Who wrote Hamlet?\n"))

    def test_run_uses_pinned_defaults_and_writes_manifest(self):
        source = {
            "train": FakeDataset(make_rows("train", 10)),
            "test": FakeDataset(make_rows("test", 7)),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            args = nq_small.parse_args([
                "--local-dir", output_dir, "--train-size", "4", "--val-size", "2",
                "--test-size", "3"
            ])
            with mock.patch.object(nq_small, "load_source_dataset", return_value=source) as load:
                manifest_path = nq_small.run(args)

            load.assert_called_once_with(nq_small.DEFAULT_DATASET, nq_small.DEFAULT_CONFIG,
                                         nq_small.DEFAULT_REVISION)
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"]["revision"], nq_small.DEFAULT_REVISION)
            self.assertTrue(manifest["overlap_checks"]["passed"])
            self.assertEqual(manifest["splits"]["train"]["rows"], 4)
            self.assertEqual(len(manifest["splits"]["val"]["sample_ids"]), 2)
            self.assertTrue((Path(output_dir) / "test_3.parquet").is_file())

            source_rows = json.loads(
                (Path(output_dir) / "test_3.parquet").read_text(
                    encoding="utf-8"))
            with mock.patch.object(nq_small,
                                   "_read_parquet",
                                   return_value=source_rows):
                native, sample_ids, source_path = (
                    nq_small.load_native_test_records(
                        Path(manifest_path),
                        Path(output_dir) / "test_3.parquet"))
            self.assertEqual(sample_ids, manifest["splits"]["test"]["sample_ids"])
            self.assertEqual(source_path.name, "test_3.parquet")
            self.assertEqual([row["prompt"] for row in native], [
                nq_small.qwen35_messages(
                    row["prompt"][0]["content"].rsplit("Question: ", 1)[1].strip())
                for row in source_rows
            ])
            self.assertEqual(
                [{key: value for key, value in row.items() if key != "prompt"}
                 for row in native],
                [{key: value for key, value in row.items() if key != "prompt"}
                 for row in source_rows])

    def test_selection_fails_when_unique_examples_are_insufficient(self):
        duplicate_train = FakeDataset([{
            "question": "same question",
            "golden_answers": ["answer"],
        }] * 4)
        test = FakeDataset(make_rows("test", 3))

        with self.assertRaisesRegex(ValueError, "unique examples"):
            nq_small.select_split_indices(duplicate_train, test, 2, 1, 2, seed=42)

    def test_overlap_check_rejects_question_mark_variant(self):
        train = FakeDataset(make_rows("train", 3))
        test = FakeDataset(make_rows("test", 3))
        test.rows[0]["question"] = "TRAIN QUESTION 0?"
        selections = {"train": [0], "val": [1], "test": [0]}

        with self.assertRaisesRegex(ValueError, "overlap"):
            nq_small.assert_no_overlap(train, test, selections)


if __name__ == "__main__":
    unittest.main()
