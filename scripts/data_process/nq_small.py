# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build deterministic, non-overlapping NQ subsets for the small run."""

import argparse
import hashlib
import json
import os
import random
import re
from typing import Dict, Iterable, List, Mapping, MutableSet, Optional, Sequence


DEFAULT_DATASET = "RUC-NLPIR/FlashRAG_datasets"
DEFAULT_CONFIG = "nq"
DEFAULT_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
DATA_SOURCE = "nq"


def normalize_question(question: str) -> str:
    """Return the comparison key used to prevent split leakage."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Every example must contain a non-empty string question")
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    return normalized[:-1].rstrip() if normalized.endswith("?") else normalized


def make_prefix(question: str, template_type: str = "base") -> str:
    if template_type != "base":
        raise ValueError(f"Unsupported template type: {template_type}")

    question = re.sub(r"\s+", " ", question).strip()
    if not question.endswith("?"):
        question += "?"
    return (
        "Answer the given question. You must conduct reasoning inside <think> "
        "and </think> first every time you get new information. After reasoning, "
        "if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results "
        "between <information> and </information>. You can search as many times "
        "as your want. If you find no further external knowledge needed, you can "
        "directly provide the answer inside <answer> and </answer>, without "
        "detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )


def make_record(example: Mapping[str, object], split: str, source_index: int,
                template_type: str = "base") -> Dict[str, object]:
    """Create the trainer fields used by the existing nq_search.py output."""
    question = example.get("question")
    normalize_question(question)  # type: ignore[arg-type]

    answers = example.get("golden_answers")
    if isinstance(answers, str):
        answers = [answers]
    elif answers is not None:
        answers = list(answers)  # type: ignore[arg-type]
    if not answers:
        raise ValueError(f"Example at source index {source_index} has no golden answers")

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": make_prefix(question, template_type)}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": answers,
            },
        },
        "extra_info": {
            "split": split,
            "index": source_index,
        },
    }


def _shuffled_indices(length: int, seed: int, salt: int) -> List[int]:
    indices = list(range(length))
    random.Random(seed + salt).shuffle(indices)
    return indices


def _select_unique(dataset: Sequence[Mapping[str, object]], candidates: Iterable[int],
                   count: int, excluded: MutableSet[str]) -> List[int]:
    selected = []
    for index in candidates:
        key = normalize_question(dataset[index].get("question"))  # type: ignore[arg-type]
        if key in excluded:
            continue
        excluded.add(key)
        selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Requested {count} unique examples, but only found {len(selected)} "
        "after excluding duplicate questions")


def select_split_indices(train_dataset: Sequence[Mapping[str, object]],
                         test_dataset: Sequence[Mapping[str, object]], train_size: int,
                         val_size: int, test_size: int,
                         seed: int) -> Dict[str, List[int]]:
    """Select deterministic indices and reject question overlap across splits."""
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("All split sizes must be positive")
    if train_size + val_size > len(train_dataset):
        raise ValueError("train_size + val_size exceeds the source train split")
    if test_size > len(test_dataset):
        raise ValueError("test_size exceeds the source test split")

    train_order = _shuffled_indices(len(train_dataset), seed, salt=0)
    used_questions: MutableSet[str] = set()
    train_indices = _select_unique(train_dataset, train_order, train_size, used_questions)
    train_index_set = set(train_indices)
    remaining_train = (index for index in train_order if index not in train_index_set)
    val_indices = _select_unique(train_dataset, remaining_train, val_size, used_questions)

    test_order = _shuffled_indices(len(test_dataset), seed, salt=1_000_003)
    test_indices = _select_unique(test_dataset, test_order, test_size, used_questions)
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def _question_keys(dataset: Sequence[Mapping[str, object]], indices: Sequence[int]) -> set:
    return {
        normalize_question(dataset[index].get("question"))  # type: ignore[arg-type]
        for index in indices
    }


def assert_no_overlap(train_dataset: Sequence[Mapping[str, object]],
                      test_dataset: Sequence[Mapping[str, object]],
                      selections: Mapping[str, Sequence[int]]) -> Dict[str, int]:
    """Fail before writing if source IDs or normalized questions overlap."""
    train_ids = {f"train:{index}" for index in selections["train"]}
    val_ids = {f"train:{index}" for index in selections["val"]}
    test_ids = {f"test:{index}" for index in selections["test"]}
    question_sets = {
        "train": _question_keys(train_dataset, selections["train"]),
        "val": _question_keys(train_dataset, selections["val"]),
        "test": _question_keys(test_dataset, selections["test"]),
    }
    id_sets = {"train": train_ids, "val": val_ids, "test": test_ids}

    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        id_overlap = len(id_sets[left] & id_sets[right])
        question_overlap = len(question_sets[left] & question_sets[right])
        overlaps[f"{left}_{right}_source_ids"] = id_overlap
        overlaps[f"{left}_{right}_questions"] = question_overlap
    if any(overlaps.values()):
        raise ValueError(f"Selected splits overlap: {overlaps}")
    return overlaps


def _output_dataset(source_dataset: object, indices: Sequence[int], split: str,
                    template_type: str) -> object:
    selected = source_dataset.select(list(indices))

    def process(example: Mapping[str, object], local_index: int) -> Dict[str, object]:
        return make_record(example, split, indices[local_index], template_type)

    return selected.map(function=process, with_indices=True)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_ids(source_split: str, indices: Sequence[int]) -> List[str]:
    return [f"{DATA_SOURCE}:{source_split}:{index}" for index in indices]


def build_manifest(dataset_name: str, config_name: str, revision: str, seed: int,
                   files: Mapping[str, str], selections: Mapping[str, Sequence[int]],
                   overlaps: Mapping[str, int]) -> Dict[str, object]:
    source_splits = {"train": "train", "val": "train", "test": "test"}
    split_data = {}
    for split in ("train", "val", "test"):
        path = files[split]
        source_split = source_splits[split]
        split_data[split] = {
            "file": os.path.basename(path),
            "rows": len(selections[split]),
            "sha256": _sha256(path),
            "source_split": source_split,
            "sample_ids": _sample_ids(source_split, selections[split]),
        }
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset_name,
            "config": config_name,
            "revision": revision,
        },
        "seed": seed,
        "splits": split_data,
        "overlap_checks": {
            "passed": not any(overlaps.values()),
            "counts": dict(overlaps),
        },
    }


def load_source_dataset(dataset_name: str, config_name: str, revision: str) -> object:
    import datasets

    return datasets.load_dataset(dataset_name, config_name, revision=revision)


def run(args: argparse.Namespace, source_dataset: Optional[Mapping[str, object]] = None) -> str:
    dataset = source_dataset or load_source_dataset(args.dataset, args.config, args.revision)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("NQ source dataset must contain train and test splits")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    selections = select_split_indices(train_dataset, test_dataset, args.train_size,
                                      args.val_size, args.test_size, args.seed)
    overlaps = assert_no_overlap(train_dataset, test_dataset, selections)

    os.makedirs(args.local_dir, exist_ok=True)
    sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    source_by_output = {
        "train": train_dataset,
        "val": train_dataset,
        "test": test_dataset,
    }
    files = {}
    for split in ("train", "val", "test"):
        path = os.path.join(args.local_dir, f"{split}_{sizes[split]}.parquet")
        output = _output_dataset(source_by_output[split], selections[split], split,
                                 args.template_type)
        output.to_parquet(path)
        files[split] = path

    manifest = build_manifest(args.dataset, args.config, args.revision, args.seed, files,
                              selections, overlaps)
    manifest_path = os.path.join(args.local_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", "--local_dir", default="./data/nq_small")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--template-type", choices=("base", ), default="base")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest_path = run(args)
    print(f"Wrote deterministic NQ subsets and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
