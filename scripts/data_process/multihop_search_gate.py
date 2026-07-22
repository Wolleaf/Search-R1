#!/usr/bin/env python3
"""Build and verify the fixed multihop search-gate evaluation set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
CONFIGS = ("hotpotqa", "2wikimultihopqa")
SOURCE_SPLIT = "dev"
DEFAULT_SEED = 42
PER_CONFIG_SIZE = 128
SCHEMA_VERSION = 1
SELECTION_POLICY = "seeded-shuffle-global-question-dedup-v1"
EVAL_FILE = "eval_256.parquet"
CATALOG_FILE = "catalog.jsonl"
MANIFEST_FILE = "manifest.json"
SOURCE_FILE_SPECS = {
    "hotpotqa": {
        "repo_file":
        "hotpotqa/dev.jsonl",
        "file":
        "sources/hotpotqa/dev.jsonl",
        "bytes":
        48_191_374,
        "sha256":
        "434ec155867019396312f3d466ce3406c71fe8a2917f49f63bb646ec5ad2ff52",
    },
    "2wikimultihopqa": {
        "repo_file":
        "2wikimultihopqa/dev.jsonl",
        "file":
        "sources/2wikimultihopqa/dev.jsonl",
        "bytes":
        54_682_944,
        "sha256":
        "b4d74d73d1cb3c0632c665a85e1e5060e3411968b807721618cfd2464fc905e3",
    },
}
CONFIG_SALTS = {
    "hotpotqa": 0,
    "2wikimultihopqa": 1_000_003,
}


@dataclass(frozen=True)
class SelectedSample:
    config: str
    source_index: int
    question: str
    golden_answers: tuple[str, ...]
    sample_type: Optional[str]
    level: Optional[str]
    supporting_titles: tuple[str, ...]


class JsonlRows:
    """Index fixed JSONL without retaining its large contexts in memory."""

    def __init__(self, path: Path):
        self.path = path
        self.offsets = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(
                        f"Pinned source contains a blank line: {path}")
                self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        if type(index) is not int:
            raise TypeError("JSONL row index must be an integer")
        if index < 0:
            index += len(self.offsets)
        if not 0 <= index < len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            line = handle.readline()
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid pinned source JSON on line {index + 1}: {self.path}"
            ) from error
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Pinned source line {index + 1} must be a JSON object: {self.path}"
            )
        return value


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) +
            "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                             dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def clean_question(question: object) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Every selected example must have a non-empty string question")
    cleaned = re.sub(r"\s+", " ", question).strip()
    return cleaned if cleaned.endswith("?") else cleaned + "?"


def normalize_question(question: object) -> str:
    cleaned = clean_question(question).casefold()
    return cleaned[:-1].rstrip()


def clean_answers(answers: object) -> tuple[str, ...]:
    if isinstance(answers, str):
        values = [answers]
    elif isinstance(answers,
                    Sequence) and not isinstance(answers, (bytes, bytearray)):
        values = list(answers)
    else:
        raise ValueError(
            "golden_answers must be a string or a sequence of strings")
    if not values:
        raise ValueError(
            "Every selected example must have at least one golden answer")
    cleaned = []
    for answer in values:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Golden answers must be non-empty strings")
        cleaned.append(answer.strip())
    return tuple(cleaned)


def make_prefix(question: str) -> str:
    return (
        "Answer the given question. You must conduct reasoning inside <think> "
        "and </think> first every time you get new information. After reasoning, "
        "if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results "
        "between <information> and </information>. You can search as many times "
        "as your want. If you find no further external knowledge needed, you can "
        "directly provide the answer inside <answer> and </answer>, without "
        "detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n")


def _optional_text(value: object, field: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when present")
    return value.strip() or None


def _supporting_titles(example: Mapping[str, object],
                       metadata: Mapping[str, object]) -> tuple[str, ...]:
    supporting = metadata.get("supporting_facts",
                              example.get("supporting_facts"))
    if supporting is None:
        return ()

    raw_titles: list[object] = []
    if isinstance(supporting, Mapping):
        title_value = supporting.get("title", [])
        if isinstance(title_value, str):
            raw_titles = [title_value]
        elif isinstance(title_value, Sequence):
            raw_titles = list(title_value)
        else:
            raise ValueError(
                "supporting_facts.title must be a string sequence")
    elif isinstance(supporting, Sequence) and not isinstance(supporting, str):
        for fact in supporting:
            if isinstance(fact, Mapping):
                raw_titles.append(fact.get("title"))
            elif isinstance(fact,
                            Sequence) and not isinstance(fact, str) and fact:
                raw_titles.append(fact[0])
            else:
                raise ValueError("Unsupported supporting_facts entry")
    else:
        raise ValueError("supporting_facts must be a mapping or sequence")

    titles = []
    seen = set()
    for title in raw_titles:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Supporting titles must be non-empty strings")
        cleaned = title.strip()
        if cleaned not in seen:
            seen.add(cleaned)
            titles.append(cleaned)
    return tuple(titles)


def prepare_sample(config: str, source_index: int,
                   example: Mapping[str, object]) -> SelectedSample:
    if config not in CONFIGS:
        raise ValueError(f"Unsupported config: {config}")
    if not isinstance(example, Mapping):
        raise ValueError("Dataset rows must be mappings")
    metadata_value = example.get("metadata", {})
    if metadata_value is None:
        metadata_value = {}
    if not isinstance(metadata_value, Mapping):
        raise ValueError("metadata must be a mapping when present")
    sample_type = _optional_text(
        metadata_value.get("type", example.get("type")), "type")
    level = _optional_text(metadata_value.get("level", example.get("level")),
                           "level")
    return SelectedSample(
        config=config,
        source_index=source_index,
        question=clean_question(example.get("question")),
        golden_answers=clean_answers(example.get("golden_answers")),
        sample_type=sample_type,
        level=level,
        supporting_titles=_supporting_titles(example, metadata_value),
    )


def select_samples(
        dev_by_config: Mapping[str, object],
        per_config_size: Optional[int] = None,
        seed: int = DEFAULT_SEED) -> dict[str, list[SelectedSample]]:
    size = PER_CONFIG_SIZE if per_config_size is None else per_config_size
    if type(size) is not int or size <= 0:
        raise ValueError("per_config_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    selected: dict[str, list[SelectedSample]] = {}
    used_questions: set[str] = set()
    for config in CONFIGS:
        if config not in dev_by_config:
            raise ValueError(f"Missing {config} {SOURCE_SPLIT} split")
        dataset = dev_by_config[config]
        try:
            candidates = list(range(len(dataset)))  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                f"{config} {SOURCE_SPLIT} split is not indexable") from error
        random.Random(seed + CONFIG_SALTS[config]).shuffle(candidates)
        config_samples = []
        for source_index in candidates:
            example = dataset[source_index]  # type: ignore[index]
            question_key = normalize_question(
                example.get("question")) if isinstance(example,
                                                       Mapping) else ""
            if question_key in used_questions:
                continue
            sample = prepare_sample(config, source_index, example)
            used_questions.add(question_key)
            config_samples.append(sample)
            if len(config_samples) == size:
                break
        if len(config_samples) != size:
            raise ValueError(
                f"{config} has only {len(config_samples)} globally unique valid questions; "
                f"{size} are required")
        selected[config] = config_samples
    return selected


def make_eval_record(sample: SelectedSample) -> dict[str, object]:
    return {
        "data_source": sample.config,
        "prompt": [{
            "role": "user",
            "content": make_prefix(sample.question),
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": list(sample.golden_answers),
            },
        },
        "extra_info": {
            "split": "test",
            "index": sample.source_index,
        },
    }


def make_catalog_record(sample: SelectedSample) -> dict[str, object]:
    return {
        "sample_id": f"{sample.config}:test:{sample.source_index}",
        "data_source": sample.config,
        "source_split": SOURCE_SPLIT,
        "source_index": sample.source_index,
        "question": sample.question,
        "golden_answers": list(sample.golden_answers),
        "type": sample.sample_type,
        "level": sample.level,
        "supporting_titles": list(sample.supporting_titles),
        "hop_proxy": len(sample.supporting_titles),
    }


def write_eval_parquet(records: Sequence[Mapping[str, object]],
                       path: Path) -> None:
    import datasets

    datasets.Dataset.from_list(list(records)).to_parquet(str(path))


def read_eval_parquet(path: Path) -> list[dict[str, Any]]:
    import datasets

    dataset = datasets.Dataset.from_parquet(str(path))
    return [dataset[index] for index in range(len(dataset))]


def _atomic_write_parquet(records: Sequence[Mapping[str, object]],
                          path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                             suffix=".tmp",
                                             dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        write_eval_parquet(records, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _source_contract() -> dict[str, dict[str, object]]:
    return {config: dict(SOURCE_FILE_SPECS[config]) for config in CONFIGS}


def _source_path(local_dir: Path, config: str) -> Path:
    spec = SOURCE_FILE_SPECS[config]
    relative = Path(str(spec["file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid source path contract for {config}")
    root = local_dir.resolve()
    path = root / relative
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Source path escapes local directory for {config}") from error
    return path


def verify_source_file(local_dir: Path, config: str) -> Path:
    path = _source_path(local_dir, config)
    spec = SOURCE_FILE_SPECS[config]
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"Pinned source file is missing or symlinked for {config}")
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Pinned source byte count mismatch for {config}")
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Pinned source SHA-256 mismatch for {config}")
    return path


def download_source_files(local_dir: Path) -> dict[str, Path]:
    result = {}
    for config in CONFIGS:
        target = _source_path(local_dir, config)
        if target.is_file() and not target.is_symlink():
            result[config] = verify_source_file(local_dir, config)
            continue
        if target.exists() or target.is_symlink():
            raise ValueError(
                f"Pinned source target is not a regular file for {config}")
        from huggingface_hub import hf_hub_download

        spec = SOURCE_FILE_SPECS[config]
        cached = Path(
            hf_hub_download(
                repo_id=DATASET_NAME,
                repo_type="dataset",
                filename=str(spec["repo_file"]),
                revision=DATASET_REVISION,
            ))
        if (not cached.is_file() or cached.stat().st_size != spec["bytes"]
                or sha256_file(cached) != spec["sha256"]):
            raise ValueError(
                f"Downloaded source identity mismatch for {config}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.",
                                                 dir=target.parent)
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            shutil.copyfile(cached, temporary_path)
            with temporary_path.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        result[config] = verify_source_file(local_dir, config)
    return result


def load_source_dataset(path: Path) -> object:
    return {SOURCE_SPLIT: JsonlRows(path)}


def _source_dev_splits(
        source_datasets: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for config in CONFIGS:
        if config not in source_datasets:
            raise ValueError(f"Missing fixed source config: {config}")
        source = source_datasets[config]
        try:
            has_dev = SOURCE_SPLIT in source  # type: ignore[operator]
        except TypeError as error:
            raise ValueError(
                f"{config} source is not a dataset mapping") from error
        if not has_dev:
            raise ValueError(
                f"{config} source must contain a {SOURCE_SPLIT} split")
        result[config] = source[SOURCE_SPLIT]  # type: ignore[index]
    return result


def _build_manifest(selected: Mapping[str, Sequence[SelectedSample]],
                    eval_path: Path, catalog_path: Path, seed: int,
                    per_config_size: int) -> dict[str, object]:
    samples = [sample for config in CONFIGS for sample in selected[config]]
    sample_ids = [
        f"{sample.config}:test:{sample.source_index}" for sample in samples
    ]
    total_rows = per_config_size * len(CONFIGS)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": DATASET_NAME,
            "revision": DATASET_REVISION,
            "configs": list(CONFIGS),
            "source_split": SOURCE_SPLIT,
            "seed": seed,
            "per_config_size": per_config_size,
            "selection_policy": SELECTION_POLICY,
            "files": _source_contract(),
        },
        "artifacts": {
            "eval_parquet": {
                "file": eval_path.name,
                "rows": total_rows,
                "sha256": sha256_file(eval_path),
            },
            "catalog_jsonl": {
                "file": catalog_path.name,
                "rows": total_rows,
                "sha256": sha256_file(catalog_path),
            },
        },
        "configs": {
            config: {
                "rows":
                len(selected[config]),
                "source_indices":
                [sample.source_index for sample in selected[config]],
            }
            for config in CONFIGS
        },
        "sample_ids": sample_ids,
        "overlap_checks": {
            "passed": True,
            "normalized_question_duplicates": 0,
        },
    }


def build_artifacts(
        local_dir: Path,
        source_datasets: Optional[Mapping[str, object]] = None) -> Path:
    local_dir = local_dir.resolve()
    source_files = {
        config: verify_source_file(local_dir, config)
        for config in CONFIGS
    } if source_datasets is not None else download_source_files(local_dir)
    if source_datasets is None:
        source_datasets = {
            config: load_source_dataset(source_files[config])
            for config in CONFIGS
        }
    dev_by_config = _source_dev_splits(source_datasets)
    selected = select_samples(dev_by_config)
    samples = [sample for config in CONFIGS for sample in selected[config]]
    eval_records = [make_eval_record(sample) for sample in samples]
    catalog_records = [make_catalog_record(sample) for sample in samples]

    eval_path = local_dir / EVAL_FILE
    catalog_path = local_dir / CATALOG_FILE
    manifest_path = local_dir / MANIFEST_FILE
    _atomic_write_parquet(eval_records, eval_path)
    atomic_write(
        catalog_path,
        b"".join(canonical_json_bytes(record) for record in catalog_records))
    manifest = _build_manifest(selected, eval_path, catalog_path, DEFAULT_SEED,
                               PER_CONFIG_SIZE)
    manifest_raw = canonical_json_bytes(manifest)
    atomic_write(manifest_path, manifest_raw)
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    atomic_write(manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
                 f"{manifest_digest}  {manifest_path.name}\n".encode("ascii"))
    verify_manifest(manifest_path)
    return manifest_path


def _require_exact_keys(value: object, expected: set[str],
                        label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}",
                                                  value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_catalog(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("catalog.jsonl must end with a newline")
    records = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if line == b"\n":
            raise ValueError("catalog.jsonl must not contain blank lines")
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid catalog JSON on line {line_number}") from error
        if canonical_json_bytes(record) != line:
            raise ValueError(
                f"Catalog line {line_number} is not canonical JSON")
        records.append(record)
    return records


def _validate_catalog_record(record: object) -> Mapping[str, Any]:
    entry = _require_exact_keys(
        record, {
            "sample_id",
            "data_source",
            "source_split",
            "source_index",
            "question",
            "golden_answers",
            "type",
            "level",
            "supporting_titles",
            "hop_proxy",
        }, "catalog record")
    config = entry["data_source"]
    if config not in CONFIGS:
        raise ValueError(f"Unknown catalog data_source: {config!r}")
    source_index = entry["source_index"]
    if type(source_index) is not int or source_index < 0:
        raise ValueError("catalog source_index must be a non-negative integer")
    if entry["source_split"] != SOURCE_SPLIT:
        raise ValueError("catalog source_split must be dev")
    expected_id = f"{config}:test:{source_index}"
    if entry["sample_id"] != expected_id:
        raise ValueError(f"catalog sample_id mismatch for {expected_id}")
    if clean_question(entry["question"]) != entry["question"]:
        raise ValueError(
            f"catalog question is not normalized for {expected_id}")
    if list(clean_answers(entry["golden_answers"])) != entry["golden_answers"]:
        raise ValueError(
            f"catalog golden answers are not normalized for {expected_id}")
    if _optional_text(entry["type"], "type") != entry["type"]:
        raise ValueError("catalog type must be normalized when present")
    if _optional_text(entry["level"], "level") != entry["level"]:
        raise ValueError("catalog level must be normalized when present")
    titles = entry["supporting_titles"]
    if not isinstance(titles, list):
        raise ValueError("catalog supporting_titles must be a list")
    for title in titles:
        if not isinstance(title,
                          str) or not title.strip() or title != title.strip():
            raise ValueError(
                "catalog supporting titles must be normalized strings")
    if len(set(titles)) != len(titles):
        raise ValueError("catalog supporting_titles must be unique")
    if type(entry["hop_proxy"]) is not int or entry["hop_proxy"] != len(
            titles):
        raise ValueError(
            "catalog hop_proxy must equal the supporting-title count")
    return entry


def _verify_eval_row(row: object, catalog: Mapping[str, Any]) -> None:
    value = _require_exact_keys(
        row,
        {"data_source", "prompt", "ability", "reward_model", "extra_info"},
        "eval row")
    sample_id = catalog["sample_id"]
    if value["data_source"] != catalog["data_source"]:
        raise ValueError(f"data_source mismatch for {sample_id}")
    expected_prompt = [{
        "role": "user",
        "content": make_prefix(catalog["question"]),
    }]
    if value["prompt"] != expected_prompt:
        raise ValueError(f"question/prompt mismatch for {sample_id}")
    if value["ability"] != "fact-reasoning":
        raise ValueError(f"ability mismatch for {sample_id}")
    expected_reward = {
        "style": "rule",
        "ground_truth": {
            "target": catalog["golden_answers"],
        },
    }
    if value["reward_model"] != expected_reward:
        raise ValueError(f"golden answers mismatch for {sample_id}")
    expected_extra = {
        "split": "test",
        "index": catalog["source_index"],
    }
    if value["extra_info"] != expected_extra:
        raise ValueError(f"extra_info mismatch for {sample_id}")


def verify_manifest(manifest_path: Path) -> Mapping[str, Any]:
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink():
        raise ValueError("Manifest must not be a symlink")
    manifest_path = manifest_path.resolve()
    if manifest_path.name != MANIFEST_FILE or not manifest_path.is_file():
        raise ValueError(
            f"Manifest must be a regular file named {MANIFEST_FILE}")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError("Manifest SHA-256 sidecar is missing or a symlink")
    raw = manifest_path.read_bytes()
    sidecar_match = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)\n",
                                 sidecar.read_bytes())
    if sidecar_match is None or sidecar_match.group(2).decode(
            "ascii") != MANIFEST_FILE:
        raise ValueError("Malformed manifest SHA-256 sidecar")
    if hashlib.sha256(raw).hexdigest() != sidecar_match.group(1).decode(
            "ascii"):
        raise ValueError("Manifest SHA-256 sidecar mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Manifest is not valid UTF-8 JSON") from error
    if canonical_json_bytes(payload) != raw:
        raise ValueError("Manifest is not canonical JSON")
    manifest = _require_exact_keys(
        payload, {
            "schema_version",
            "source",
            "artifacts",
            "configs",
            "sample_ids",
            "overlap_checks",
        }, "manifest")
    if (type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Manifest schema_version mismatch")
    expected_source = {
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "configs": list(CONFIGS),
        "source_split": SOURCE_SPLIT,
        "seed": DEFAULT_SEED,
        "per_config_size": PER_CONFIG_SIZE,
        "selection_policy": SELECTION_POLICY,
        "files": _source_contract(),
    }
    source = _require_exact_keys(manifest["source"], set(expected_source),
                                 "manifest source")
    if (type(source["seed"]) is not int
            or type(source["per_config_size"]) is not int
            or source != expected_source):
        raise ValueError("Manifest source contract mismatch")
    files = _require_exact_keys(source["files"], set(CONFIGS),
                                "manifest source files")
    for config in CONFIGS:
        entry = _require_exact_keys(files[config],
                                    {"repo_file", "file", "bytes", "sha256"},
                                    f"manifest source file {config}")
        if (type(entry["repo_file"]) is not str
                or type(entry["file"]) is not str
                or type(entry["bytes"]) is not int
                or type(entry["sha256"]) is not str):
            raise ValueError(
                f"Manifest source identity types mismatch for {config}")
        verify_source_file(manifest_path.parent, config)
    if manifest["overlap_checks"] != {
            "passed": True,
            "normalized_question_duplicates": 0,
    }:
        raise ValueError("Manifest overlap contract mismatch")

    total_rows = PER_CONFIG_SIZE * len(CONFIGS)
    artifacts = _require_exact_keys(manifest["artifacts"],
                                    {"eval_parquet", "catalog_jsonl"},
                                    "manifest artifacts")
    expected_artifacts = {
        "eval_parquet": EVAL_FILE,
        "catalog_jsonl": CATALOG_FILE,
    }
    artifact_paths = {}
    for key, expected_file in expected_artifacts.items():
        entry = _require_exact_keys(artifacts[key], {"file", "rows", "sha256"},
                                    f"artifact {key}")
        if entry["file"] != expected_file or entry["rows"] != total_rows:
            raise ValueError(f"Artifact contract mismatch for {key}")
        expected_digest = _validate_digest(entry["sha256"], f"artifact {key}")
        path = manifest_path.parent / expected_file
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"Artifact is missing or a symlink: {expected_file}")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"Artifact checksum mismatch: {expected_file}")
        artifact_paths[key] = path

    catalog_records = [
        _validate_catalog_record(record)
        for record in _read_catalog(artifact_paths["catalog_jsonl"])
    ]
    if len(catalog_records) != total_rows:
        raise ValueError("Catalog row count mismatch")
    sample_ids = [record["sample_id"] for record in catalog_records]
    if len(set(
            sample_ids)) != total_rows or manifest["sample_ids"] != sample_ids:
        raise ValueError("Manifest/catalog sample IDs mismatch")
    question_keys = [
        normalize_question(record["question"]) for record in catalog_records
    ]
    if len(set(question_keys)) != total_rows:
        raise ValueError("Catalog contains duplicate normalized questions")

    fixed_sources = {
        config:
        load_source_dataset(verify_source_file(manifest_path.parent, config))
        for config in CONFIGS
    }
    selected = select_samples(_source_dev_splits(fixed_sources))
    expected_catalog = [
        make_catalog_record(sample) for config in CONFIGS
        for sample in selected[config]
    ]
    if catalog_records != expected_catalog:
        raise ValueError(
            "Catalog does not match the deterministic selection from pinned sources"
        )

    configs = _require_exact_keys(manifest["configs"], set(CONFIGS),
                                  "manifest configs")
    for config in CONFIGS:
        config_entry = _require_exact_keys(configs[config],
                                           {"rows", "source_indices"},
                                           f"manifest config {config}")
        config_records = [
            record for record in catalog_records
            if record["data_source"] == config
        ]
        source_indices = [record["source_index"] for record in config_records]
        if (config_entry["rows"] != PER_CONFIG_SIZE
                or len(config_records) != PER_CONFIG_SIZE
                or config_entry["source_indices"] != source_indices
                or len(set(source_indices)) != PER_CONFIG_SIZE):
            raise ValueError(f"Manifest/catalog config mismatch for {config}")

    try:
        eval_records = read_eval_parquet(artifact_paths["eval_parquet"])
    except Exception as error:
        raise ValueError("Eval Parquet cannot be read") from error
    if len(eval_records) != total_rows:
        raise ValueError("Eval Parquet row count mismatch")
    for row, catalog in zip(eval_records, catalog_records):
        _verify_eval_row(row, catalog)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0].startswith("-"):
        raw.insert(0, "build")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser(
        "build", help="build and verify the fixed eval set")
    build_parser.add_argument("--local-dir",
                              type=Path,
                              default=Path("./data/multihop_search_gate"))
    verify_parser = commands.add_parser(
        "verify", help="verify an existing sealed eval set")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_artifacts(args.local_dir)
            print(f"Built and verified multihop search gate: {manifest}")
        else:
            verify_manifest(args.manifest)
            print(f"Verified multihop search gate: {args.manifest.resolve()}")
    except (ImportError, OSError, ValueError) as error:
        print(f"multihop search-gate error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
