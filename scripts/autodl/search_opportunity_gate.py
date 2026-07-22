#!/usr/bin/env python3
"""Audit one fixed HotpotQA/2Wiki baseline trace for search-cost opportunity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from search_r1.trajectory_trace import verify_trace_manifest  # noqa: E402
from scripts.data_process.multihop_search_gate import SOURCE_FILE_SPECS  # noqa: E402

DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
DATASETS = ("hotpotqa", "2wikimultihopqa")
SOURCE_SPLIT = "dev"
TRACE_SPLIT = "test"
SELECTION_POLICY = "seeded-shuffle-global-question-dedup-v1"
EXPECTED_ROWS = 256
PER_DATASET_ROWS = 128
CATALOG_FIELDS = {
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
}
REQUIRED_TRACE_FIELDS = {
    "data_source",
    "source_split",
    "sample_id",
    "source_index",
    "question",
    "gold_answers",
    "em",
    "executed_search_count",
    "turns",
    "retrieval_events",
    "response_clipped",
    "invalid_action_count",
    "extracted_answer",
    "raw_trajectory",
    "max_searches",
    "checkpoint_digest",
}
OUTPUT_FILES = {
    "summary.json",
    "summary.md",
    "go_no_go.json",
    "per_question.jsonl",
    "correct_questions.csv",
    "wrong_questions.csv",
    "two_plus_search.csv",
    "redundant_search_candidates.csv",
    "strata.csv",
}


def _canonical_data_json(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) +
            "\n").encode("utf-8")


def _canonical_report_json(value: object) -> bytes:
    return (json.dumps(value,
                       ensure_ascii=False,
                       sort_keys=True,
                       separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) +
            "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(value: object, expected: set[str],
                        label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"{label} must be a regular, non-symlink file: {path}")
    return path.resolve()


def _clean_question(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("question must be a non-empty string")
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned if cleaned.endswith("?") else cleaned + "?"


def _normalize_question(value: object) -> str:
    cleaned = _clean_question(value).casefold()
    return cleaned[:-1].rstrip()


def _clean_answers(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    answers = []
    for answer in value:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{label} must contain non-empty strings")
        answers.append(answer.strip())
    return answers


def _optional_clean_text(value: object, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value,
                      str) or not value.strip() or value != value.strip():
        raise ValueError(
            f"{label} must be null or a normalized non-empty string")
    return value


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("catalog JSONL must end with a newline")
    records = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            raise ValueError(
                f"catalog JSONL has a blank line at {line_number}")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid catalog JSON on line {line_number}") from error
        if _canonical_data_json(value) != line:
            raise ValueError(
                f"catalog line {line_number} is not canonical JSON")
        entry = dict(
            _require_exact_keys(value, CATALOG_FIELDS,
                                f"catalog line {line_number}"))
        dataset = entry["data_source"]
        index = entry["source_index"]
        if dataset not in DATASETS:
            raise ValueError(f"unsupported catalog dataset: {dataset!r}")
        if type(index) is not int or index < 0:
            raise ValueError(
                "catalog source_index must be a non-negative integer")
        if entry["source_split"] != SOURCE_SPLIT:
            raise ValueError(f"catalog source_split must be {SOURCE_SPLIT!r}")
        expected_id = f"{dataset}:{TRACE_SPLIT}:{index}"
        if entry["sample_id"] != expected_id:
            raise ValueError(f"catalog sample_id mismatch for {expected_id}")
        if _clean_question(entry["question"]) != entry["question"]:
            raise ValueError(
                f"catalog question is not normalized for {expected_id}")
        if (_clean_answers(entry["golden_answers"], "catalog golden_answers")
                != entry["golden_answers"]):
            raise ValueError(
                f"catalog golden_answers are not normalized for {expected_id}")
        _optional_clean_text(entry["type"], "catalog type")
        _optional_clean_text(entry["level"], "catalog level")
        titles = entry["supporting_titles"]
        if not isinstance(titles, list):
            raise ValueError("catalog supporting_titles must be a list")
        if any(not isinstance(title, str) or not title.strip()
               or title != title.strip() for title in titles):
            raise ValueError(
                "catalog supporting_titles must be normalized strings")
        if len(set(titles)) != len(titles):
            raise ValueError("catalog supporting_titles must be unique")
        if type(entry["hop_proxy"]) is not int or entry["hop_proxy"] != len(
                titles):
            raise ValueError(
                "catalog hop_proxy must equal supporting-title count")
        records.append(entry)
    return records


def _verify_data_manifest(
        catalog_path: Path,
        manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog_path = _regular_file(catalog_path, "catalog")
    manifest_path = _regular_file(manifest_path, "data manifest")
    if manifest_path.name != "manifest.json":
        raise ValueError("data manifest must be named manifest.json")
    expected_catalog = (manifest_path.parent / "catalog.jsonl").resolve()
    if catalog_path != expected_catalog:
        raise ValueError(
            "--catalog is not the catalog bound by --data-manifest")

    sidecar_path = _regular_file(
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
        "data manifest SHA-256 sidecar",
    )
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected_sidecar = f"{digest}  {manifest_path.name}\n"
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError(
            "data manifest SHA-256 sidecar is not ASCII") from error
    if sidecar != expected_sidecar:
        raise ValueError("data manifest SHA-256 sidecar mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("data manifest is not valid UTF-8 JSON") from error
    if raw != _canonical_data_json(value):
        raise ValueError("data manifest is not canonical JSON")
    manifest = dict(
        _require_exact_keys(
            value,
            {
                "schema_version", "source", "artifacts", "configs",
                "sample_ids", "overlap_checks"
            },
            "data manifest",
        ))
    if type(manifest["schema_version"]
            ) is not int or manifest["schema_version"] != 1:
        raise ValueError("data manifest schema_version mismatch")
    expected_source = {
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "configs": list(DATASETS),
        "source_split": SOURCE_SPLIT,
        "seed": 42,
        "per_config_size": PER_DATASET_ROWS,
        "selection_policy": SELECTION_POLICY,
        "files": {
            dataset: dict(SOURCE_FILE_SPECS[dataset])
            for dataset in DATASETS
        },
    }
    source = _require_exact_keys(manifest["source"], set(expected_source),
                                 "data manifest source")
    if (type(source["seed"]) is not int
            or type(source["per_config_size"]) is not int
            or source != expected_source):
        raise ValueError("data manifest source/revision contract mismatch")
    source_files = _require_exact_keys(source["files"], set(DATASETS),
                                       "data manifest source files")
    source_root = manifest_path.parent.resolve()
    for dataset in DATASETS:
        expected_file = SOURCE_FILE_SPECS[dataset]
        declared_file = _require_exact_keys(
            source_files[dataset], {"repo_file", "file", "bytes", "sha256"},
            f"data manifest source file {dataset}")
        if (type(declared_file["repo_file"]) is not str
                or type(declared_file["file"]) is not str
                or type(declared_file["bytes"]) is not int
                or type(declared_file["sha256"]) is not str
                or declared_file != expected_file):
            raise ValueError(
                f"data manifest source identity mismatch for {dataset}")
        source_path = _regular_file(source_root / declared_file["file"],
                                    f"pinned source file {dataset}")
        try:
            source_path.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"pinned source path escapes data root for {dataset}"
            ) from error
        if source_path.stat().st_size != declared_file["bytes"]:
            raise ValueError(
                f"pinned source byte count mismatch for {dataset}")
        if _sha256(source_path) != declared_file["sha256"]:
            raise ValueError(f"pinned source SHA-256 mismatch for {dataset}")
    overlap = _require_exact_keys(
        manifest["overlap_checks"],
        {"passed", "normalized_question_duplicates"},
        "data manifest overlap_checks",
    )
    if (overlap["passed"] is not True
            or type(overlap["normalized_question_duplicates"]) is not int
            or overlap["normalized_question_duplicates"] != 0):
        raise ValueError("data manifest overlap contract mismatch")

    artifacts = _require_exact_keys(manifest["artifacts"],
                                    {"eval_parquet", "catalog_jsonl"},
                                    "data manifest artifacts")
    expected_artifacts = {
        "eval_parquet": "eval_256.parquet",
        "catalog_jsonl": "catalog.jsonl",
    }
    for role, filename in expected_artifacts.items():
        artifact = _require_exact_keys(artifacts[role],
                                       {"file", "rows", "sha256"},
                                       f"data manifest artifact {role}")
        if (artifact["file"] != filename or type(artifact["rows"]) is not int
                or artifact["rows"] != EXPECTED_ROWS):
            raise ValueError(
                f"data manifest artifact contract mismatch for {role}")
        if not isinstance(artifact["sha256"], str) or re.fullmatch(
                r"[0-9a-f]{64}", artifact["sha256"]) is None:
            raise ValueError(
                f"invalid SHA-256 in data manifest artifact {role}")
        artifact_path = _regular_file(manifest_path.parent / filename,
                                      f"data artifact {role}")
        if _sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"data artifact SHA-256 mismatch for {role}")

    catalog = _read_catalog(catalog_path)
    if len(catalog) != EXPECTED_ROWS:
        raise ValueError(f"catalog must contain exactly {EXPECTED_ROWS} rows")
    sample_ids = [entry["sample_id"] for entry in catalog]
    if len(set(sample_ids)) != EXPECTED_ROWS:
        raise ValueError("catalog sample_id values must be unique")
    if manifest["sample_ids"] != sample_ids:
        raise ValueError("data manifest sample_ids do not match catalog order")
    question_keys = [
        _normalize_question(entry["question"]) for entry in catalog
    ]
    if len(set(question_keys)) != EXPECTED_ROWS:
        raise ValueError("catalog contains duplicate normalized questions")

    configs = _require_exact_keys(manifest["configs"], set(DATASETS),
                                  "data manifest configs")
    counts = Counter(entry["data_source"] for entry in catalog)
    for dataset in DATASETS:
        config = _require_exact_keys(configs[dataset],
                                     {"rows", "source_indices"},
                                     f"data manifest config {dataset}")
        indices = [
            entry["source_index"] for entry in catalog
            if entry["data_source"] == dataset
        ]
        declared_indices = config["source_indices"]
        if (type(config["rows"]) is not int
                or not isinstance(declared_indices, list)
                or any(type(index) is not int for index in declared_indices)
                or counts[dataset] != PER_DATASET_ROWS
                or config["rows"] != PER_DATASET_ROWS
                or declared_indices != indices
                or len(set(indices)) != PER_DATASET_ROWS):
            raise ValueError(
                f"catalog/data-manifest must contain exactly {PER_DATASET_ROWS} "
                f"unique {dataset} rows")
    return manifest, catalog


def _read_trace(
        trace_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_path = _regular_file(trace_path, "trace")
    manifest_path = trace_path.with_suffix(".manifest.json")
    _regular_file(manifest_path, "trace manifest")
    _regular_file(manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
                  "trace manifest SHA-256 sidecar")
    manifest = verify_trace_manifest(manifest_path,
                                     expected_rows=EXPECTED_ROWS)
    artifact = manifest["artifact"]
    if (type(manifest["schema_version"]) is not int
            or type(artifact["record_schema_version"]) is not int):
        raise ValueError("trace manifest schema versions must be integers")
    if artifact["path"] != trace_path.name:
        raise ValueError(
            "trace manifest does not bind the supplied trace path")
    if artifact["record_type"] != "eval":
        raise ValueError("search-opportunity trace must contain eval records")
    if manifest["stage"] != "search_opportunity":
        raise ValueError("search-opportunity trace stage mismatch")

    records = []
    schemas: set[frozenset[str]] = set()
    with trace_path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"trace contains a blank line at {line_number}")
            record = json.loads(line.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"trace line {line_number} must be an object")
            missing = REQUIRED_TRACE_FIELDS - set(record)
            if missing:
                raise ValueError(
                    f"trace line {line_number} is missing metadata: {sorted(missing)}"
                )
            schemas.add(frozenset(record))
            records.append(record)
    if len(schemas) != 1:
        raise ValueError("trace records have inconsistent top-level schemas")
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"trace must contain exactly {EXPECTED_ROWS} rows")
    if len({record["sample_id"] for record in records}) != EXPECTED_ROWS:
        raise ValueError("trace sample_id values must be unique")
    return manifest, records


def _document_id(document: object) -> Optional[str]:
    if not isinstance(document, Mapping):
        raise ValueError("retrieval documents must be objects")
    if "document_id" not in document:
        return None
    value = document["document_id"]
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("retrieval document_id must be a string or integer")
    if isinstance(value, str) and not value.strip():
        raise ValueError("retrieval document_id must not be blank")
    return str(value)


def _validate_trace_record(record: dict[str, Any], max_searches: int) -> None:
    dataset = record["data_source"]
    index = record["source_index"]
    if dataset not in DATASETS:
        raise ValueError(f"unsupported trace dataset: {dataset!r}")
    if record["source_split"] != TRACE_SPLIT:
        raise ValueError(f"trace source_split must be {TRACE_SPLIT!r}")
    if type(index) is not int or index < 0:
        raise ValueError("trace source_index must be a non-negative integer")
    expected_id = f"{dataset}:{TRACE_SPLIT}:{index}"
    if record["sample_id"] != expected_id:
        raise ValueError(f"trace sample_id mismatch for {expected_id}")
    searches = record["executed_search_count"]
    if type(searches) is not int or not 0 <= searches <= max_searches:
        raise ValueError(
            f"executed_search_count must be an integer in [0, {max_searches}]")
    if type(record["schema_version"]) is not int:
        raise ValueError("trace schema_version must be an integer")
    if type(record["max_searches"]
            ) is not int or record["max_searches"] != max_searches:
        raise ValueError(
            f"trace max_searches must equal CLI max_searches={max_searches}")
    if (not isinstance(record["checkpoint_digest"], str) or re.fullmatch(
            r"[0-9a-f]{64}", record["checkpoint_digest"]) is None):
        raise ValueError(
            "trace checkpoint_digest must be a lowercase SHA-256 digest")
    if not isinstance(record["response_clipped"], bool):
        raise ValueError("response_clipped must be boolean")
    if type(record["invalid_action_count"]
            ) is not int or record["invalid_action_count"] < 0:
        raise ValueError("invalid_action_count must be a non-negative integer")
    _clean_question(record["question"])
    _clean_answers(record["gold_answers"], "trace gold_answers")

    events = record["retrieval_events"]
    turns = record["turns"]
    if not isinstance(events, list) or len(events) != searches:
        raise ValueError(
            f"retrieval event count mismatch for {record['sample_id']}")
    if not isinstance(turns, list):
        raise ValueError("turns must be a list")
    executed_turns = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("trace turns must be objects")
        for field in ("generation_turn", "environment_action",
                      "retrieved_docs", "retrieval_executed"):
            if field not in turn:
                raise ValueError(
                    f"trace turn is missing alignment field {field}")
        if type(turn["generation_turn"]
                ) is not int or turn["generation_turn"] < 0:
            raise ValueError(
                "turn generation_turn must be a non-negative integer")
        if not isinstance(turn["environment_action"], bool):
            raise ValueError("turn environment_action must be boolean")
        if not isinstance(turn["retrieval_executed"], bool):
            raise ValueError("turn retrieval_executed must be boolean")
        if not isinstance(turn["retrieved_docs"], list):
            raise ValueError("turn retrieved_docs must be a list")
        if turn["retrieval_executed"]:
            executed_turns.append(turn)
    if len(executed_turns) != searches:
        raise ValueError(
            f"executed retrieval turn count mismatch for {record['sample_id']}"
        )

    event_turns = []
    for event, turn in zip(events, executed_turns):
        if not isinstance(event, Mapping):
            raise ValueError("retrieval events must be objects")
        missing = {"turn", "query", "documents", "observation"} - set(event)
        if missing:
            raise ValueError(
                f"retrieval event is missing fields: {sorted(missing)}")
        if type(event["turn"]) is not int or event["turn"] < 0:
            raise ValueError(
                "retrieval event turn must be a non-negative integer")
        if not isinstance(event["query"], str):
            raise ValueError("retrieval event query must be a string")
        if not isinstance(event["documents"], list):
            raise ValueError("retrieval event documents must be a list")
        if not isinstance(event["observation"], str):
            raise ValueError("retrieval event observation must be a string")
        for document in event["documents"]:
            _document_id(document)
        event_turns.append(event["turn"])
        if (turn["generation_turn"] != event["turn"]
                or turn["environment_action"] is not True
                or turn["action"] != "search"
                or turn["valid_action"] is not True
                or turn["search_query"] != event["query"]
                or turn["observation"] != event["observation"]
                or turn["retrieved_docs"] != event["documents"]):
            raise ValueError(
                f"turn/retrieval event alignment mismatch for {record['sample_id']}"
            )
    if event_turns != sorted(event_turns) or len(
            set(event_turns)) != len(event_turns):
        raise ValueError(
            "retrieval event turn values must be unique and ordered")


def _join_trace_catalog(
        trace: Sequence[dict[str, Any]], catalog: Sequence[dict[str, Any]],
        max_searches: int,
        expected_checkpoint_digest: str) -> list[dict[str, Any]]:
    trace_by_id = {}
    for record in trace:
        _validate_trace_record(record, max_searches)
        trace_by_id[record["sample_id"]] = record
    checkpoint_digests = {record["checkpoint_digest"] for record in trace}
    if len(checkpoint_digests) != 1:
        raise ValueError("trace records must share one checkpoint_digest")
    if checkpoint_digests != {expected_checkpoint_digest}:
        raise ValueError(
            "trace checkpoint_digest does not match historical B checkpoint")
    catalog_ids = {entry["sample_id"] for entry in catalog}
    if set(trace_by_id) != catalog_ids:
        missing = sorted(catalog_ids - set(trace_by_id))[:5]
        extra = sorted(set(trace_by_id) - catalog_ids)[:5]
        raise ValueError(
            f"trace/catalog sample set mismatch; missing={missing}, extra={extra}"
        )

    joined = []
    for entry in catalog:
        record = trace_by_id[entry["sample_id"]]
        if (record["data_source"] != entry["data_source"]
                or record["source_index"] != entry["source_index"]):
            raise ValueError(
                f"trace/catalog identity mismatch for {entry['sample_id']}")
        if _normalize_question(record["question"]) != _normalize_question(
                entry["question"]):
            raise ValueError(
                f"trace/catalog question mismatch for {entry['sample_id']}")
        if _clean_answers(record["gold_answers"],
                          "trace gold_answers") != entry["golden_answers"]:
            raise ValueError(
                f"trace/catalog gold-answer mismatch for {entry['sample_id']}")
        joined.append({"catalog": entry, "trace": record})
    return joined


def _tokens(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return set(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _jaccard(left: object, right: object) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _normalized_phrase(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _phrase_seen(phrase: object, observations: str) -> bool:
    needle = _normalized_phrase(phrase)
    haystack = _normalized_phrase(observations)
    return bool(needle) and f" {needle} " in f" {haystack} "


def _redundancy_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    searches = record["executed_search_count"]
    if searches < 3:
        return {
            "eligible": False,
            "classification": "not_applicable",
            "reason_codes": [],
            "duplicate_query": None,
            "max_query_jaccard": None,
            "document_identity_complete": None,
            "no_new_document": None,
            "third_new_document_ratio": None,
            "low_document_novelty": None,
            "answer_seen_before_third": None,
            "gold_seen_before_third": None,
        }

    events = record["retrieval_events"]
    third = events[2]
    similarities = [
        _jaccard(third["query"], prior["query"]) for prior in events[:2]
    ]
    max_similarity = max(similarities)
    duplicate_query = max_similarity >= 0.8

    relevant_documents = [
        document for event in events[:3] for document in event["documents"]
    ]
    document_identity_complete = all(
        _document_id(document) is not None for document in relevant_documents)
    no_new_document: Optional[bool] = None
    novelty_ratio: Optional[float] = None
    low_novelty: Optional[bool] = None
    if document_identity_complete:
        previous_ids = {
            _document_id(document)
            for event in events[:2]
            for document in event["documents"]
        }
        third_ids = {_document_id(document) for document in third["documents"]}
        new_ids = third_ids - previous_ids
        no_new_document = not new_ids
        novelty_ratio = len(new_ids) / len(third_ids) if third_ids else 0.0
        low_novelty = novelty_ratio <= (1.0 / 3.0)

    previous_observations = "\n".join(event["observation"]
                                      for event in events[:2])
    answer_seen = _phrase_seen(record["extracted_answer"],
                               previous_observations)
    gold_seen = any(
        _phrase_seen(answer, previous_observations)
        for answer in record["gold_answers"])
    evidence_seen = answer_seen or gold_seen
    em = int(float(record["em"]))
    if em == 1 and evidence_seen and (duplicate_query
                                      or no_new_document is True):
        classification = "strong"
    elif ((em == 1 and evidence_seen and low_novelty is True)
          or (em == 0 and duplicate_query and no_new_document is True)):
        classification = "medium"
    else:
        classification = "review"

    flags = {
        "duplicate_query": duplicate_query,
        "no_new_document": no_new_document,
        "low_document_novelty": low_novelty,
        "answer_seen_before_third": answer_seen,
        "gold_seen_before_third": gold_seen,
    }
    return {
        "eligible":
        True,
        "classification":
        classification,
        "reason_codes":
        [name for name, present in flags.items() if present is True],
        "duplicate_query":
        duplicate_query,
        "max_query_jaccard":
        max_similarity,
        "document_identity_complete":
        document_identity_complete,
        "no_new_document":
        no_new_document,
        "third_new_document_ratio":
        novelty_ratio,
        "low_document_novelty":
        low_novelty,
        "answer_seen_before_third":
        answer_seen,
        "gold_seen_before_third":
        gold_seen,
    }


def _enrich(joined: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for item in joined:
        catalog = item["catalog"]
        trace = item["trace"]
        enriched.append({
            "sample_id":
            trace["sample_id"],
            "dataset":
            trace["data_source"],
            "source_split":
            catalog["source_split"],
            "trace_split":
            trace["source_split"],
            "source_index":
            trace["source_index"],
            "metadata_type":
            (catalog["type"] if catalog["type"] is not None else "unknown"),
            "level":
            catalog["level"] if catalog["level"] is not None else "unknown",
            "supporting_titles":
            catalog["supporting_titles"],
            "hop_proxy":
            catalog["hop_proxy"],
            "hop_proxy_definition":
            "distinct supporting-title count",
            "question":
            trace["question"],
            "gold_answers":
            trace["gold_answers"],
            "extracted_answer":
            trace["extracted_answer"],
            "em":
            int(float(trace["em"])),
            "executed_search_count":
            trace["executed_search_count"],
            "response_clipped":
            trace["response_clipped"],
            "invalid_action_count":
            trace["invalid_action_count"],
            "raw_trajectory":
            trace["raw_trajectory"],
            "turns":
            trace["turns"],
            "retrieval_events":
            trace["retrieval_events"],
            "redundancy_evidence":
            _redundancy_evidence(trace),
        })
    return enriched


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[int]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _metrics(records: Sequence[dict[str, Any]],
             max_searches: int) -> dict[str, Any]:
    n = len(records)
    correct = [record for record in records if record["em"] == 1]
    wrong = [record for record in records if record["em"] == 0]
    distribution = Counter(record["executed_search_count"]
                           for record in records)
    clipped = sum(record["response_clipped"] for record in records)
    invalid_actions = sum(record["invalid_action_count"] for record in records)
    invalid_questions = sum(record["invalid_action_count"] > 0
                            for record in records)
    searches = [record["executed_search_count"] for record in records]
    distribution_correct = Counter(record["executed_search_count"]
                                   for record in correct)
    return {
        "N":
        n,
        "K_correct":
        len(correct),
        "EM":
        _ratio(len(correct), n),
        "search_distribution": {
            str(search_count): {
                "count":
                distribution[search_count],
                "ratio":
                _ratio(distribution[search_count], n),
                "correct_count":
                distribution_correct[search_count],
                "EM":
                _ratio(distribution_correct[search_count],
                       distribution[search_count]),
            }
            for search_count in range(max_searches + 1)
        },
        "S_total_searches":
        sum(searches),
        "mean_searches":
        _mean(searches),
        "mean_searches_given_correct":
        _mean([record["executed_search_count"] for record in correct]),
        "mean_searches_given_wrong":
        _mean([record["executed_search_count"] for record in wrong]),
        "P_searches_ge_2":
        _ratio(sum(record["executed_search_count"] >= 2 for record in records),
               n),
        "P_searches_ge_3":
        _ratio(sum(record["executed_search_count"] >= 3 for record in records),
               n),
        "clipped_count":
        clipped,
        "clipped_ratio":
        _ratio(clipped, n),
        "invalid_action_count":
        invalid_actions,
        "invalid_action_question_count":
        invalid_questions,
        "invalid_action_ratio":
        _ratio(invalid_questions, n),
    }


def _group_metrics(
        records: Sequence[dict[str, Any]],
        max_searches: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    definitions = (
        ("dataset", lambda record: record["dataset"]),
        ("metadata_type", lambda record: record["metadata_type"]),
        ("level", lambda record: record["level"]),
        ("hop_proxy", lambda record: str(record["hop_proxy"])),
    )
    groups: dict[str, Any] = {"overall": _metrics(records, max_searches)}
    strata = [{"stratum": "overall", "value": "all", **groups["overall"]}]
    for dimension, key_fn in definitions:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            buckets[str(key_fn(record))].append(record)
        groups[dimension] = {}
        sort_key = (
            lambda value: int(value)) if dimension == "hop_proxy" else None
        for value in sorted(buckets, key=sort_key):
            metrics = _metrics(buckets[value], max_searches)
            groups[dimension][value] = metrics
            strata.append({"stratum": dimension, "value": value, **metrics})
    return groups, strata


def _criterion(observed: Optional[float | int], threshold: float | int,
               comparison: str, label: str) -> dict[str, Any]:
    if observed is None:
        passed = False
        reason = f"{label}: denominator unavailable"
    elif comparison == ">=":
        passed = observed >= threshold
        operator = ">=" if passed else "<"
        reason = f"{label}: {observed:.6g} {operator} {threshold:.6g}"
    elif comparison == "<=":
        passed = observed <= threshold
        operator = "<=" if passed else ">"
        reason = f"{label}: {observed:.6g} {operator} {threshold:.6g}"
    else:
        raise AssertionError(f"unsupported comparison: {comparison}")
    return {
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "pass": passed,
        "reason": reason,
    }


def _decision(records: Sequence[dict[str, Any]],
              thresholds: Mapping[str, Any]) -> dict[str, Any]:
    correct = [record for record in records if record["em"] == 1]
    correct_two_plus = sum(record["executed_search_count"] >= 2
                           for record in correct)
    three_plus = [
        record for record in records if record["executed_search_count"] >= 3
    ]
    candidates = [
        record for record in three_plus
        if record["redundancy_evidence"]["classification"] in ("strong",
                                                               "medium")
    ]
    clipped = sum(record["response_clipped"] for record in records)
    invalid_questions = sum(record["invalid_action_count"] > 0
                            for record in records)
    criteria = {
        "total_correct":
        _criterion(len(correct), thresholds["min_correct"], ">=",
                   "correct answers"),
        "correct_with_two_plus_searches_count":
        _criterion(correct_two_plus, thresholds["min_correct_two_plus"], ">=",
                   "correct answers with S>=2"),
        "correct_with_two_plus_searches_ratio":
        _criterion(_ratio(correct_two_plus, len(correct)),
                   thresholds["min_correct_two_plus_ratio"], ">=",
                   "share of correct answers with S>=2"),
        "all_with_three_plus_searches_count":
        _criterion(len(three_plus), thresholds["min_three_plus"], ">=",
                   "questions with S>=3"),
        "redundancy_candidate_count":
        _criterion(len(candidates), thresholds["min_redundancy_candidates"],
                   ">=", "strong+medium redundancy candidates"),
        "candidate_ratio_among_three_plus":
        _criterion(_ratio(len(candidates),
                          len(three_plus)), thresholds["min_candidate_ratio"],
                   ">=", "candidate share among S>=3"),
        "clipped_ratio":
        _criterion(_ratio(clipped,
                          len(records)), thresholds["max_clipped_ratio"], "<=",
                   "response-clipped question ratio"),
        "invalid_action_ratio":
        _criterion(_ratio(invalid_questions, len(records)),
                   thresholds["max_invalid_action_ratio"], "<=",
                   "questions with invalid actions"),
    }
    return {
        "decision":
        "GO" if all(item["pass"] for item in criteria.values()) else "NO-GO",
        "criteria":
        criteria,
        "note":
        ("GO means the fixed baseline exposes enough observable search opportunity "
         "to justify the pre-registered cost-aware follow-up; redundancy labels are "
         "heuristic candidates, not proof that a search was unnecessary."),
    }


def _csv_scalar(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value,
                          ensure_ascii=False,
                          sort_keys=True,
                          separators=(",", ":"),
                          allow_nan=False)
    if value is None:
        return ""
    return value


QUESTION_FIELDS = [
    "sample_id",
    "dataset",
    "source_split",
    "trace_split",
    "source_index",
    "metadata_type",
    "level",
    "supporting_titles",
    "hop_proxy",
    "question",
    "gold_answers",
    "extracted_answer",
    "em",
    "executed_search_count",
    "response_clipped",
    "invalid_action_count",
    "redundancy_classification",
    "redundancy_reason_codes",
    "max_query_jaccard",
    "no_new_document",
    "third_new_document_ratio",
    "answer_seen_before_third",
    "gold_seen_before_third",
    "raw_trajectory",
    "turns",
    "retrieval_events",
]


def _question_csv_rows(
        records: Iterable[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        evidence = record["redundancy_evidence"]
        row = {
            **{
                field: record[field]
                for field in QUESTION_FIELDS if field in record
            },
            "redundancy_classification": evidence["classification"],
            "redundancy_reason_codes": evidence["reason_codes"],
            "max_query_jaccard": evidence["max_query_jaccard"],
            "no_new_document": evidence["no_new_document"],
            "third_new_document_ratio": evidence["third_new_document_ratio"],
            "answer_seen_before_third": evidence["answer_seen_before_third"],
            "gold_seen_before_third": evidence["gold_seen_before_third"],
        }
        rows.append(
            {field: _csv_scalar(row.get(field))
             for field in QUESTION_FIELDS})
    return rows


def _csv_bytes(fieldnames: Sequence[str],
               rows: Iterable[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,
                            fieldnames=fieldnames,
                            lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _strata_csv(strata: Sequence[dict[str, Any]], max_searches: int) -> bytes:
    fields = [
        "stratum",
        "value",
        "N",
        "K_correct",
        "EM",
        "S_total_searches",
        "mean_searches",
        "mean_searches_given_correct",
        "mean_searches_given_wrong",
        "P_searches_ge_2",
        "P_searches_ge_3",
        "clipped_count",
        "clipped_ratio",
        "invalid_action_count",
        "invalid_action_question_count",
        "invalid_action_ratio",
    ]
    for search_count in range(max_searches + 1):
        fields.extend([
            f"S_{search_count}_count",
            f"S_{search_count}_ratio",
            f"S_{search_count}_correct_count",
            f"S_{search_count}_EM",
        ])
    rows = []
    for item in strata:
        row = {field: item.get(field, "") for field in fields}
        for search_count in range(max_searches + 1):
            distribution = item["search_distribution"][str(search_count)]
            row[f"S_{search_count}_count"] = distribution["count"]
            row[f"S_{search_count}_ratio"] = distribution["ratio"]
            row[f"S_{search_count}_correct_count"] = distribution[
                "correct_count"]
            row[f"S_{search_count}_EM"] = distribution["EM"]
        rows.append({field: _csv_scalar(row[field]) for field in fields})
    return _csv_bytes(fields, rows)


def _summary_markdown(summary: Mapping[str, Any],
                      decision: Mapping[str, Any]) -> str:
    overall = summary["groups"]["overall"]
    lines = [
        "# Search Opportunity Gate",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        "This is a baseline-only audit of 128 HotpotQA and 128 2WikiMultiHopQA "
        "questions. It does not compare checkpoints or prove that any search was redundant.",
        "",
        "## Baseline Metrics",
        "",
        "| Questions | Correct | EM | Mean searches | P(S>=2) | P(S>=3) | "
        "Clipped | Invalid-action questions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (f"| {overall['N']} | {overall['K_correct']} | {overall['EM']:.4f} | "
         f"{overall['mean_searches']:.4f} | {overall['P_searches_ge_2']:.4f} | "
         f"{overall['P_searches_ge_3']:.4f} | {overall['clipped_ratio']:.4f} | "
         f"{overall['invalid_action_ratio']:.4f} |"),
        "",
        "## Gate Criteria",
        "",
        "| Criterion | Observed | Rule | Pass |",
        "| --- | ---: | ---: | :---: |",
    ]
    for name, criterion in decision["criteria"].items():
        observed = "n/a" if criterion[
            "observed"] is None else f"{criterion['observed']:.6g}"
        lines.append(
            f"| {name} | {observed} | {criterion['comparison']} "
            f"{criterion['threshold']:.6g} | {'yes' if criterion['pass'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Evidence Notes",
        "",
        "- `per_question.jsonl` preserves every question, final answer, full raw trajectory, "
        "structured turns, retrieval events, and heuristic redundancy evidence.",
        "- `hop_proxy` is only the count of distinct annotated supporting-fact titles; it is "
        "not a semantic or oracle hop count.",
        "- Strong/medium labels inspect only observable third-search query similarity, document "
        "novelty, and whether answer text was already present before that search.",
        "",
    ])
    return "\n".join(lines)


def _validate_thresholds(args: argparse.Namespace) -> dict[str, Any]:
    if type(args.max_searches) is not int or args.max_searches < 3:
        raise ValueError("max_searches must be an integer >= 3")
    if (not isinstance(args.expected_checkpoint_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", args.expected_checkpoint_digest) is None):
        raise ValueError(
            "expected_checkpoint_digest must be a lowercase SHA-256 digest")
    count_names = (
        "min_correct",
        "min_correct_two_plus",
        "min_three_plus",
        "min_redundancy_candidates",
    )
    for name in count_names:
        value = getattr(args, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    ratio_names = (
        "min_correct_two_plus_ratio",
        "min_candidate_ratio",
        "max_clipped_ratio",
        "max_invalid_action_ratio",
    )
    for name in ratio_names:
        value = getattr(args, name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be a finite ratio in [0, 1]")
    return {name: getattr(args, name) for name in (*count_names, *ratio_names)}


def _publish(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != OUTPUT_FILES:
        raise AssertionError("internal output artifact set mismatch")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, payload in payloads.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite output directory: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = _validate_thresholds(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite output directory: {output_dir}")

    data_manifest, catalog = _verify_data_manifest(Path(args.catalog),
                                                   Path(args.data_manifest))
    trace_manifest, trace = _read_trace(Path(args.trace))
    joined = _join_trace_catalog(trace, catalog, args.max_searches,
                                 args.expected_checkpoint_digest)
    records = _enrich(joined)
    groups, strata = _group_metrics(records, args.max_searches)
    decision = _decision(records, thresholds)

    three_plus = [
        record for record in records if record["executed_search_count"] >= 3
    ]
    candidates = [
        record for record in three_plus
        if record["redundancy_evidence"]["classification"] in ("strong",
                                                               "medium")
    ]
    trace_path = Path(args.trace).resolve()
    catalog_path = Path(args.catalog).resolve()
    data_manifest_path = Path(args.data_manifest).resolve()
    trace_manifest_path = trace_path.with_suffix(".manifest.json")
    checkpoint_digest = trace[0]["checkpoint_digest"]
    summary = {
        "schema_version": 1,
        "scope": "baseline-only fixed multihop search-opportunity audit",
        "expected_rows": EXPECTED_ROWS,
        "per_dataset_rows": PER_DATASET_ROWS,
        "max_searches": args.max_searches,
        "hop_proxy_definition":
        "distinct supporting-title count; not semantic hop count",
        "thresholds": thresholds,
        "inputs": {
            "trace": {
                "path": str(trace_path),
                "sha256": _sha256(trace_path)
            },
            "trace_manifest": {
                "path": str(trace_manifest_path),
                "sha256": _sha256(trace_manifest_path),
                "run_id": trace_manifest["run_id"],
                "stage": trace_manifest["stage"],
                "checkpoint_digest": checkpoint_digest,
                "expected_checkpoint_digest": args.expected_checkpoint_digest,
            },
            "catalog": {
                "path": str(catalog_path),
                "sha256": _sha256(catalog_path)
            },
            "data_manifest": {
                "path": str(data_manifest_path),
                "sha256": _sha256(data_manifest_path),
                "dataset": data_manifest["source"]["dataset"],
                "revision": data_manifest["source"]["revision"],
            },
        },
        "groups": groups,
        "redundancy": {
            "three_plus_search_questions":
            len(three_plus),
            "strong_candidates":
            sum(record["redundancy_evidence"]["classification"] == "strong"
                for record in candidates),
            "medium_candidates":
            sum(record["redundancy_evidence"]["classification"] == "medium"
                for record in candidates),
            "strong_plus_medium_candidates":
            len(candidates),
            "candidate_ratio_among_three_plus":
            _ratio(len(candidates), len(three_plus)),
            "disclaimer":
            "heuristic review candidates, not proven redundant searches",
        },
        "decision": decision["decision"],
    }

    payloads = {
        "summary.json":
        _pretty_json(summary),
        "summary.md":
        _summary_markdown(summary, decision).encode("utf-8"),
        "go_no_go.json":
        _pretty_json(decision),
        "per_question.jsonl":
        b"".join(_canonical_report_json(record) for record in records),
        "correct_questions.csv":
        _csv_bytes(
            QUESTION_FIELDS,
            _question_csv_rows(record for record in records
                               if record["em"] == 1)),
        "wrong_questions.csv":
        _csv_bytes(
            QUESTION_FIELDS,
            _question_csv_rows(record for record in records
                               if record["em"] == 0)),
        "two_plus_search.csv":
        _csv_bytes(
            QUESTION_FIELDS,
            _question_csv_rows(record for record in records
                               if record["executed_search_count"] >= 2)),
        "redundant_search_candidates.csv":
        _csv_bytes(QUESTION_FIELDS, _question_csv_rows(candidates)),
        "strata.csv":
        _strata_csv(strata, args.max_searches),
    }
    _publish(output_dir, payloads)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(
        "Strictly join one 256-row search_opportunity eval trace with the "
        "sealed HotpotQA/2Wiki catalog and emit a baseline-only opportunity gate."
    ))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-digest", required=True)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--min-correct", type=int, default=20)
    parser.add_argument("--min-correct-two-plus", type=int, default=8)
    parser.add_argument("--min-correct-two-plus-ratio",
                        type=float,
                        default=0.20)
    parser.add_argument("--min-three-plus", type=int, default=10)
    parser.add_argument("--min-redundancy-candidates", type=int, default=5)
    parser.add_argument("--min-candidate-ratio", type=float, default=0.30)
    parser.add_argument("--max-clipped-ratio", type=float, default=0.05)
    parser.add_argument("--max-invalid-action-ratio", type=float, default=0.05)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = analyze(args)
    except (OSError, ValueError) as error:
        print(f"search opportunity gate error: {error}", file=sys.stderr)
        return 1
    print(f"Search opportunity decision: {summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
