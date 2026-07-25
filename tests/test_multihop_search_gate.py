import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.data_process import multihop_search_gate as gate


def make_rows(config, count):
    rows = []
    for index in range(count):
        metadata = {
            "type": "bridge" if config == "hotpotqa" else "compositional",
            "supporting_facts": {
                "title":
                [f"{config} support {index} A", f"{config} support {index} B"],
                "sent_id": [0, 1],
            },
            "context": {
                "title": [f"SECRET_CONTEXT_TITLE_{config}_{index}"],
                "content": [[f"SECRET_CONTEXT_BODY_{config}_{index}"]],
            },
        }
        if config == "hotpotqa":
            metadata["level"] = "hard"
        rows.append({
            "id": f"{config}-{index}",
            "question": f"  {config}   question {index}  ",
            "golden_answers": [f" answer {config} {index} "],
            "metadata": metadata,
        })
    return rows


def make_sources(count=20):
    return {
        config: {
            "train": make_rows(config, 2),
            "dev": make_rows(config, count),
        }
        for config in gate.CONFIGS
    }


def fake_write_eval(records, path):
    Path(path).write_bytes(gate.canonical_json_bytes(list(records)))


def fake_read_eval(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_catalog(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def reseal_manifest(path, payload):
    path = Path(path)
    raw = gate.canonical_json_bytes(payload)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n")


def write_source_files(output_dir, sources):
    specs = {}
    for config in gate.CONFIGS:
        relative = f"sources/{config}/dev.jsonl"
        path = Path(output_dir) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b"".join(
            gate.canonical_json_bytes(row) for row in sources[config]["dev"])
        path.write_bytes(raw)
        specs[config] = {
            "repo_file": f"{config}/dev.jsonl",
            "file": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return specs


class MultihopSearchGateTest(unittest.TestCase):

    def test_build_is_deterministic_pinned_and_context_free(self):
        sources = make_sources()

        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir, \
                mock.patch.object(gate, "SOURCE_FILE_SPECS",
                                  write_source_files(first_dir, sources)), \
                mock.patch.object(gate, "PER_CONFIG_SIZE", 3), \
                mock.patch.object(gate, "write_eval_parquet", side_effect=fake_write_eval), \
                mock.patch.object(gate, "read_eval_parquet", side_effect=fake_read_eval):
            write_source_files(second_dir, sources)
            first_manifest = gate.build_artifacts(Path(first_dir))
            second_manifest = gate.build_artifacts(Path(second_dir))

            self.assertEqual(first_manifest.read_bytes(),
                             second_manifest.read_bytes())
            self.assertEqual((Path(first_dir) / gate.EVAL_FILE).read_bytes(),
                             (Path(second_dir) / gate.EVAL_FILE).read_bytes())
            self.assertEqual(
                (Path(first_dir) / gate.CATALOG_FILE).read_bytes(),
                (Path(second_dir) / gate.CATALOG_FILE).read_bytes())
            gate.verify_manifest(first_manifest)

            manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["configs"], list(gate.CONFIGS))
            self.assertEqual(manifest["source"]["source_split"], "dev")
            self.assertEqual(manifest["source"]["seed"], 42)
            self.assertEqual(manifest["source"]["files"],
                             gate._source_contract())
            self.assertEqual(len(manifest["sample_ids"]), 6)

            eval_records = fake_read_eval(Path(first_dir) / gate.EVAL_FILE)
            catalog = read_catalog(Path(first_dir) / gate.CATALOG_FILE)
            self.assertEqual(len(eval_records), 6)
            self.assertEqual(len(catalog), 6)
            for row, item in zip(eval_records, catalog):
                self.assertEqual(
                    set(row), {
                        "data_source", "prompt", "ability", "reward_model",
                        "extra_info"
                    })
                self.assertEqual(item["source_split"], "dev")
                self.assertEqual(
                    item["sample_id"],
                    f"{item['data_source']}:test:{item['source_index']}")
                self.assertEqual(item["hop_proxy"], 2)
                self.assertEqual(row["extra_info"], {
                    "split": "test",
                    "index": item["source_index"],
                })
                prompt = row["prompt"][0]["content"]
                self.assertNotIn("SECRET_CONTEXT_TITLE", prompt)
                self.assertNotIn("SECRET_CONTEXT_BODY", prompt)
                self.assertNotIn("metadata", prompt)
                self.assertNotIn("context", prompt)

            native, native_ids, source_eval = gate.load_native_eval_records(
                first_manifest, Path(first_dir) / gate.CATALOG_FILE)
            self.assertEqual(native_ids, manifest["sample_ids"])
            self.assertEqual(source_eval, Path(first_dir) / gate.EVAL_FILE)
            self.assertEqual([row["prompt"] for row in native], [
                gate.qwen35_messages(item["question"]) for item in catalog
            ])
            self.assertEqual(
                [{key: value for key, value in row.items() if key != "prompt"}
                 for row in native],
                [{key: value for key, value in row.items() if key != "prompt"}
                 for row in eval_records])

        self.assertEqual(gate.parse_args([]).command, "build")
        self.assertEqual(
            gate.parse_args(["--local-dir", "unused"]).command, "build")
        verify_args = gate.parse_args(
            ["verify", "--manifest", "manifest.json"])
        self.assertEqual(verify_args.command, "verify")
        self.assertEqual(verify_args.manifest, Path("manifest.json"))
        self.assertEqual(gate.PER_CONFIG_SIZE, 128)
        self.assertEqual(gate.EVAL_FILE, "eval_256.parquet")

    def test_download_requests_only_the_two_pinned_dev_files(self):
        sources = make_sources(count=2)
        with tempfile.TemporaryDirectory() as output_dir, \
                tempfile.TemporaryDirectory() as cache_dir:
            specs = write_source_files(cache_dir, sources)
            cached = {
                spec["repo_file"]: Path(cache_dir) / str(spec["file"])
                for spec in specs.values()
            }
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                return str(cached[kwargs["filename"]])

            fake_module = types.SimpleNamespace(hf_hub_download=fake_download)
            with mock.patch.object(gate, "SOURCE_FILE_SPECS", specs), \
                    mock.patch.dict(sys.modules,
                                    {"huggingface_hub": fake_module}):
                paths = gate.download_source_files(Path(output_dir))

            self.assertEqual(
                [call["filename"] for call in calls],
                [f"{config}/dev.jsonl" for config in gate.CONFIGS])
            self.assertTrue(
                all(call["repo_id"] == gate.DATASET_NAME and call["repo_type"]
                    == "dataset" and call["revision"] == gate.DATASET_REVISION
                    for call in calls))
            self.assertEqual(set(paths), set(gate.CONFIGS))
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_selection_globally_deduplicates_and_seed_changes_indices(self):
        shared_hotpot = make_rows("hotpotqa", 5)
        for row in shared_hotpot:
            row["question"] = "Shared question?"
        shared_2wiki = make_rows("2wikimultihopqa", 5)
        for row in shared_2wiki[:-1]:
            row["question"] = " shared   QUESTION "
        shared_2wiki[-1]["question"] = "Unique 2Wiki question"

        selected = gate.select_samples(
            {
                "hotpotqa": shared_hotpot,
                "2wikimultihopqa": shared_2wiki,
            },
            per_config_size=1)
        questions = [
            sample.question for config in gate.CONFIGS
            for sample in selected[config]
        ]
        self.assertEqual(
            len({gate.normalize_question(question)
                 for question in questions}), 2)
        self.assertEqual(selected["2wikimultihopqa"][0].question,
                         "Unique 2Wiki question?")

        unique = {config: make_rows(config, 30) for config in gate.CONFIGS}
        first = gate.select_samples(unique, per_config_size=5, seed=42)
        again = gate.select_samples(unique, per_config_size=5, seed=42)
        changed = gate.select_samples(unique, per_config_size=5, seed=43)
        first_indices = {
            config: [sample.source_index for sample in first[config]]
            for config in gate.CONFIGS
        }
        self.assertEqual(
            first_indices, {
                config: [sample.source_index for sample in again[config]]
                for config in gate.CONFIGS
            })
        self.assertNotEqual(
            first_indices, {
                config: [sample.source_index for sample in changed[config]]
                for config in gate.CONFIGS
            })

    def test_rejects_missing_dev_and_invalid_core_fields(self):
        sources = make_sources(count=2)
        del sources["hotpotqa"]["dev"]
        with self.assertRaisesRegex(ValueError, "must contain a dev split"):
            gate._source_dev_splits(sources)

        invalid = make_sources(count=1)
        invalid["hotpotqa"]["dev"][0]["golden_answers"] = []
        dev_splits = gate._source_dev_splits(invalid)
        with self.assertRaisesRegex(ValueError, "at least one golden answer"):
            gate.select_samples(dev_splits, per_config_size=1)

        invalid = make_sources(count=1)
        invalid["hotpotqa"]["dev"][0]["question"] = "   "
        dev_splits = gate._source_dev_splits(invalid)
        with self.assertRaisesRegex(ValueError, "non-empty string question"):
            gate.select_samples(dev_splits, per_config_size=1)

    def test_verify_rejects_hash_and_resealed_semantic_tampering(self):
        sources = make_sources(count=10)
        with tempfile.TemporaryDirectory() as output_dir:
            specs = write_source_files(output_dir, sources)
            with mock.patch.object(gate, "SOURCE_FILE_SPECS", specs), \
                mock.patch.object(gate, "PER_CONFIG_SIZE", 2), \
                mock.patch.object(gate, "write_eval_parquet", side_effect=fake_write_eval), \
                mock.patch.object(gate, "read_eval_parquet", side_effect=fake_read_eval):
                output = Path(output_dir)
                manifest_path = gate.build_artifacts(output, sources)

                catalog_path = output / gate.CATALOG_FILE
                catalog_path.write_bytes(catalog_path.read_bytes() +
                                         b"tampered\n")
                with self.assertRaisesRegex(ValueError,
                                            "Artifact checksum mismatch"):
                    gate.verify_manifest(manifest_path)

                manifest_path = gate.build_artifacts(output, sources)
                eval_path = output / gate.EVAL_FILE
                eval_records = fake_read_eval(eval_path)
                eval_records[0]["reward_model"]["ground_truth"]["target"] = [
                    "wrong"
                ]
                fake_write_eval(eval_records, eval_path)
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts"]["eval_parquet"][
                    "sha256"] = gate.sha256_file(eval_path)
                reseal_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError,
                                            "golden answers mismatch"):
                    gate.verify_manifest(manifest_path)

                manifest_path = gate.build_artifacts(output, sources)
                catalog = read_catalog(output / gate.CATALOG_FILE)
                catalog[0]["question"] = "Semantically tampered question?"
                catalog_raw = b"".join(
                    gate.canonical_json_bytes(record) for record in catalog)
                (output / gate.CATALOG_FILE).write_bytes(catalog_raw)
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts"]["catalog_jsonl"][
                    "sha256"] = gate.sha256_file(output / gate.CATALOG_FILE)
                reseal_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError,
                                            "deterministic selection"):
                    gate.verify_manifest(manifest_path)

                manifest_path = gate.build_artifacts(output, sources)
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
                manifest["sample_ids"][0] = "hotpotqa:test:999999"
                reseal_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "sample IDs mismatch"):
                    gate.verify_manifest(manifest_path)

                manifest_path = gate.build_artifacts(output, sources)
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
                manifest["source"]["files"]["hotpotqa"]["bytes"] = float(
                    specs["hotpotqa"]["bytes"])
                reseal_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError,
                                            "identity types mismatch"):
                    gate.verify_manifest(manifest_path)

                manifest_path = gate.build_artifacts(output, sources)
                link_dir = output / "manifest-link"
                link_dir.mkdir(exist_ok=True)
                link = link_dir / gate.MANIFEST_FILE
                try:
                    link.symlink_to(manifest_path)
                except OSError:
                    pass
                else:
                    with self.assertRaisesRegex(ValueError,
                                                "must not be a symlink"):
                        gate.verify_manifest(link)

                injected = make_sources(count=10)
                for row in injected["hotpotqa"]["dev"]:
                    row["question"] = "Injected " + row["question"]
                with self.assertRaisesRegex(ValueError,
                                            "deterministic selection"):
                    gate.build_artifacts(output, injected)

                manifest_path = gate.build_artifacts(output, sources)
                source = output / str(specs["hotpotqa"]["file"])
                source.write_bytes(source.read_bytes() + b"tampered\n")
                with self.assertRaisesRegex(ValueError,
                                            "source byte count mismatch"):
                    gate.verify_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
