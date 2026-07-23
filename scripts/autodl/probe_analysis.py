#!/usr/bin/env python3
"""Validate and summarize the fixed 64-question grouped behavior probe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence
import unicodedata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from search_r1.trajectory_trace import validate_trace_record  # noqa: E402

ANALYSIS_SCHEMA = "search-r1.grouped-probe-analysis"
ANALYSIS_SCHEMA_VERSION = 2
EXPECTED_GROUPS = 64
GROUP_SIZE = 5
EXPECTED_ROWS = EXPECTED_GROUPS * GROUP_SIZE
MAX_SEARCHES = 4
MIN_MULTI_SEARCHES = 2
MAX_QUERY_TOKEN_JACCARD = 0.8
NEAR_MISS_DIAGNOSTIC_THRESHOLD = 16
NEAR_MISS_EXAMPLE_LIMIT = 5
CATEGORY_GROUPS = {"comparison": 16, "bridge": 48}
THRESHOLDS = {
    "valid_correct_multi_search_count": 16,
    "covered_question_count": 8,
    "learnable_group_count": 8,
    "max_clipped_ratio": 0.05,
    "max_invalid_action_ratio": 0.05,
}
OUTPUT_FILES = {
    "summary.json",
    "summary.md",
    "go_no_go.json",
    "per_trajectory.jsonl",
    "per_question.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"{label} must be a regular, non-symlink file: {path}")
    return path.resolve()


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(character if character.isalnum() else " "
                         for character in normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _phrase_visible(phrases: Sequence[str], text: str) -> bool:
    haystack = f" {_normalize_text(text)} "
    return any(normalized and f" {normalized} " in haystack
               for normalized in (_normalize_text(value) for value in phrases))


def _cover_em(prediction: object, golden_answers: Sequence[str]) -> bool:
    normalized_prediction = _normalize_text(prediction)
    if not normalized_prediction:
        return False
    padded_prediction = f" {normalized_prediction} "
    for answer in golden_answers:
        normalized_answer = _normalize_text(answer)
        if not normalized_answer:
            continue
        padded_answer = f" {normalized_answer} "
        if (padded_prediction in padded_answer
                or padded_answer in padded_prediction):
            return True
    return False


def _query_token_jaccard(first: str, second: str) -> float:
    first_tokens = set(_normalize_text(first).split())
    second_tokens = set(_normalize_text(second).split())
    if not first_tokens or not second_tokens:
        return 1.0
    return len(first_tokens & second_tokens) / len(first_tokens
                                                   | second_tokens)


def _identity(value: object, label: str) -> tuple[str, str | int]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be a string or integer")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} must not be empty")
        return ("str", value)
    return ("int", value)


def _require_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise ValueError(
                    f"{label} contains a blank line at {line_number}")
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{label} contains invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{label} line {line_number} must be an object")
            record["_line_number"] = line_number
            records.append(record)
    return records


def _clean_answers(value: object, label: str) -> list[str]:
    if (not isinstance(value, list) or not value or not all(
            isinstance(answer, str) and answer.strip() for answer in value)):
        raise ValueError(f"{label} must be a non-empty list of strings")
    return [answer.strip() for answer in value]


def _load_probe_catalog(
        path: Path) -> dict[tuple[str, str | int], dict[str, Any]]:
    selected: dict[tuple[str, str | int], dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    for raw_record in _read_jsonl(path, "search-mix catalog"):
        line_number = raw_record.pop("_line_number")
        if (raw_record.get("output_split") != "val"
                or raw_record.get("category") not in CATEGORY_GROUPS):
            continue
        location = f"catalog line {line_number}"
        required = {
            "sample_id",
            "data_source",
            "source_split",
            "source_index",
            "category",
            "question",
            "golden_answers",
            "supporting_titles",
        }
        missing = required - set(raw_record)
        if missing:
            raise ValueError(
                f"{location} is missing fields: {sorted(missing)}")
        key = _identity(raw_record["sample_id"], f"{location} sample_id")
        if key in selected:
            raise ValueError(
                f"duplicate probe sample_id in catalog: {raw_record['sample_id']!r}"
            )
        category = raw_record["category"]
        question = raw_record["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{location} question must be a non-empty string")
        answers = _clean_answers(raw_record["golden_answers"],
                                 f"{location} golden_answers")
        titles = raw_record["supporting_titles"]
        if (not isinstance(titles, list) or len(titles) != 2 or not all(
                isinstance(title, str) and title.strip() for title in titles)
                or len({_normalize_text(title)
                        for title in titles}) != 2):
            raise ValueError(
                f"{location} must have two distinct supporting titles")
        source_index = _require_int(raw_record["source_index"],
                                    f"{location} source_index")
        if raw_record["data_source"] != "hotpotqa" or raw_record[
                "source_split"] != "train":
            raise ValueError(f"{location} is not a HotpotQA train sample")
        selected[key] = {
            "sample_id": raw_record["sample_id"],
            "data_source": "hotpotqa",
            "source_split": "train",
            "source_index": source_index,
            "category": category,
            "question": question.strip(),
            "gold_answers": answers,
            "supporting_titles": [title.strip() for title in titles],
        }
        category_counts[category] += 1

    if category_counts != Counter(CATEGORY_GROUPS):
        raise ValueError(
            "probe catalog category counts must be "
            f"{CATEGORY_GROUPS}, found {dict(sorted(category_counts.items()))}"
        )
    if len(selected) != EXPECTED_GROUPS:
        raise ValueError(
            f"probe catalog must select {EXPECTED_GROUPS} samples, found {len(selected)}"
        )
    return selected


def _document_id(document: object, location: str) -> str:
    if not isinstance(document, Mapping):
        raise ValueError(f"{location} document must be an object")
    value = document.get("document_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{location} document_id must be a string or integer")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{location} document_id must not be empty")
    return str(value)


def _document_title(document: Mapping[str, Any]) -> str:
    candidates: list[object] = [document.get("title")]
    payload = document.get("document")
    if isinstance(payload, Mapping):
        candidates.extend((payload.get("title"), payload.get("contents")))
    candidates.append(document.get("contents"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().splitlines()[0].strip()
    return ""


def _validate_retrievals(record: Mapping[str, Any], location: str,
                         searches: int) -> list[dict[str, Any]]:
    raw_events = record.get("retrieval_events")
    if not isinstance(raw_events, list) or len(raw_events) != searches:
        raise ValueError(
            f"{location} retrieval_events must contain exactly {searches} events"
        )
    events: list[dict[str, Any]] = []
    for index, raw_event in enumerate(raw_events):
        event_location = f"{location} retrieval_events[{index}]"
        if not isinstance(raw_event, Mapping):
            raise ValueError(f"{event_location} must be an object")
        query = raw_event.get("query")
        observation = raw_event.get("observation")
        visible_observation = raw_event.get("visible_observation")
        documents = raw_event.get("documents")
        if not isinstance(query, str):
            raise ValueError(f"{event_location}.query must be a string")
        if not isinstance(observation, str):
            raise ValueError(f"{event_location}.observation must be a string")
        if not isinstance(visible_observation, str):
            raise ValueError(
                f"{event_location}.visible_observation must be a string")
        if not isinstance(documents, list):
            raise ValueError(f"{event_location}.documents must be a list")
        ids = [
            _document_id(document, event_location) for document in documents
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"{event_location} contains duplicate document_id values")
        events.append({
            "query": query.strip(),
            "observation": observation,
            "visible_observation": visible_observation,
            "documents": list(documents),
            "document_ids": ids,
        })

    executed_turns = [
        turn for turn in record["turns"]
        if isinstance(turn, Mapping) and turn.get("retrieval_executed") is True
    ]
    if len(executed_turns) != searches:
        raise ValueError(
            f"{location} turns must expose exactly {searches} executed retrievals"
        )
    for index, (turn, event) in enumerate(zip(executed_turns, events)):
        turn_location = f"{location} executed turn {index}"
        turn_docs = turn.get("retrieved_docs")
        if not isinstance(turn_docs, list):
            raise ValueError(f"{turn_location}.retrieved_docs must be a list")
        turn_ids = [
            _document_id(document, turn_location) for document in turn_docs
        ]
        if turn_ids != event["document_ids"]:
            raise ValueError(
                f"{turn_location} document IDs disagree with retrieval_events")
        if (_normalize_text(turn.get("search_query"))
                != _normalize_text(event["query"])):
            raise ValueError(
                f"{turn_location} query disagrees with retrieval_events")
        if turn.get("observation") != event["visible_observation"]:
            raise ValueError(
                f"{turn_location} visible observation disagrees with retrieval_events"
            )
    return events


def _evidence_assessment(
        record: Mapping[str, Any], catalog: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(events) < MIN_MULTI_SEARCHES:
        return {
            "query_distinct": None,
            "query_token_jaccard": None,
            "new_document_ids": [],
            "answer_seen_first": None,
            "answer_seen_second": None,
            "new_supporting_titles": [],
            "evidence_branch": None,
        }

    first, second = events[0], events[1]
    query_token_jaccard = _query_token_jaccard(first["query"], second["query"])
    query_distinct = query_token_jaccard < MAX_QUERY_TOKEN_JACCARD
    first_visible = first["visible_observation"]
    second_visible = second["visible_observation"]
    first_ids = set(first["document_ids"])
    visible_new_documents = []
    for document, document_id in zip(second["documents"],
                                     second["document_ids"]):
        title = _document_title(document)
        if (document_id not in first_ids and title
                and _phrase_visible([title], second_visible)):
            visible_new_documents.append((document_id, title))
    new_ids = [document_id for document_id, _ in visible_new_documents]
    answers = catalog["gold_answers"]
    answer_seen_first = _phrase_visible(answers, first_visible)
    answer_seen_second = _phrase_visible(answers, second_visible)

    second_new_titles = {
        _normalize_text(title)
        for _, title in visible_new_documents
    }
    supporting_titles = {
        _normalize_text(title): title
        for title in catalog["supporting_titles"]
    }
    new_supporting_titles = sorted(
        original for normalized, original in supporting_titles.items()
        if normalized and normalized in second_new_titles
        and not _phrase_visible([original], first_visible))
    answer_evidence = not answer_seen_first and answer_seen_second
    supporting_evidence = bool(new_supporting_titles)
    branch = None
    if answer_evidence and supporting_evidence:
        branch = "answer_and_supporting_title"
    elif answer_evidence:
        branch = "answer"
    elif supporting_evidence:
        branch = "supporting_title"
    return {
        "query_distinct": query_distinct,
        "query_token_jaccard": query_token_jaccard,
        "new_document_ids": new_ids,
        "answer_seen_first": answer_seen_first,
        "answer_seen_second": answer_seen_second,
        "new_supporting_titles": new_supporting_titles,
        "evidence_branch": branch,
    }


def _qualification(record: Mapping[str, Any],
                   evidence: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if int(record["em"]) != 1:
        failures.append("incorrect")
    if record["executed_search_count"] < MIN_MULTI_SEARCHES:
        failures.append("fewer_than_two_searches")
    if record["response_clipped"]:
        failures.append("response_clipped")
    if record["invalid_action_count"] > 0:
        failures.append("invalid_action")
    if record["executed_search_count"] >= MIN_MULTI_SEARCHES:
        if not evidence["query_distinct"]:
            failures.append("near_duplicate_or_empty_query")
        if not evidence["new_document_ids"]:
            failures.append("no_new_document")
        if evidence["evidence_branch"] is None:
            failures.append("no_new_supporting_or_answer_evidence")
    return not failures, failures


def _load_trace(
        path: Path, catalog: Mapping[tuple[str, str | int], Mapping[str, Any]],
        expected_checkpoint_digest: str) -> tuple[list[dict[str, Any]], str]:
    raw_records = _read_jsonl(path, "grouped eval trace")
    if len(raw_records) != EXPECTED_ROWS:
        raise ValueError(
            f"grouped eval trace must contain {EXPECTED_ROWS} rows, found {len(raw_records)}"
        )
    schemas: set[frozenset[str]] = set()
    record_ids: set[str] = set()
    has_group_uid: set[bool] = set()
    stages: set[str] = set()
    run_ids: set[str] = set()
    observed_samples: set[tuple[str, str | int]] = set()
    records: list[dict[str, Any]] = []

    for raw_record in raw_records:
        line_number = raw_record.pop("_line_number")
        location = f"trace line {line_number}"
        schemas.add(frozenset(raw_record))
        validate_trace_record(raw_record, expected_record_type="eval")
        record_id = raw_record["record_id"]
        if record_id in record_ids:
            raise ValueError(
                f"duplicate record_id in grouped eval trace: {record_id}")
        record_ids.add(record_id)
        stages.add(raw_record["stage"])
        run_ids.add(raw_record["run_id"])
        if raw_record["checkpoint_digest"] != expected_checkpoint_digest:
            raise ValueError(
                f"{location} checkpoint_digest does not match the expected model"
            )
        if raw_record.get("max_searches") != MAX_SEARCHES:
            raise ValueError(
                f"{location} max_searches must equal {MAX_SEARCHES}")
        searches = raw_record["executed_search_count"]
        if searches > MAX_SEARCHES:
            raise ValueError(
                f"{location} executed_search_count exceeds {MAX_SEARCHES}")
        if _require_int(raw_record.get("group_size"),
                        f"{location} group_size",
                        minimum=1) != GROUP_SIZE:
            raise ValueError(f"{location} group_size must equal {GROUP_SIZE}")
        slot = _require_int(raw_record.get("group_slot"),
                            f"{location} group_slot")
        if slot >= GROUP_SIZE:
            raise ValueError(
                f"{location} group_slot must be in [0, {GROUP_SIZE - 1}]")
        present = "group_uid" in raw_record
        has_group_uid.add(present)
        if present and (not isinstance(raw_record["group_uid"], str)
                        or not raw_record["group_uid"].strip()):
            raise ValueError(
                f"{location} group_uid must be a non-empty string")

        sample_key = _identity(raw_record["sample_id"],
                               f"{location} sample_id")
        if sample_key not in catalog:
            raise ValueError(
                f"{location} sample_id is absent from the fixed probe catalog")
        metadata = catalog[sample_key]
        if (raw_record["source_index"] != metadata["source_index"]
                or raw_record.get("data_source") != metadata["data_source"]
                or raw_record.get("source_split") != metadata["source_split"]):
            raise ValueError(
                f"{location} source identity disagrees with the probe catalog")
        if _normalize_text(raw_record["question"]) != _normalize_text(
                metadata["question"]):
            raise ValueError(
                f"{location} question disagrees with the probe catalog")
        trace_answers = {
            _normalize_text(answer)
            for answer in raw_record["gold_answers"]
        }
        catalog_answers = {
            _normalize_text(answer)
            for answer in metadata["gold_answers"]
        }
        if trace_answers != catalog_answers:
            raise ValueError(
                f"{location} gold_answers disagree with the probe catalog")

        events = _validate_retrievals(raw_record, location, searches)
        evidence = _evidence_assessment(raw_record, metadata, events)
        qualifies, failures = _qualification(raw_record, evidence)
        clean = (not raw_record["response_clipped"]
                 and raw_record["invalid_action_count"] == 0)
        near_miss = int(raw_record["em"]) == 0 and failures == ["incorrect"]
        records.append({
            "trace":
            raw_record,
            "catalog":
            metadata,
            "events":
            events,
            "evidence":
            evidence,
            "qualifies":
            qualifies,
            "qualification_failures":
            failures,
            "clean":
            clean,
            "near_miss":
            near_miss,
            "cover_em":
            _cover_em(raw_record["extracted_answer"],
                      metadata["gold_answers"]),
            "sample_key":
            sample_key,
        })
        observed_samples.add(sample_key)

    if len(schemas) != 1:
        raise ValueError(
            "grouped eval trace records have inconsistent schemas")
    if len(has_group_uid) != 1:
        raise ValueError(
            "group_uid must be present on every trace row or none")
    if len(stages) != 1 or len(run_ids) != 1:
        raise ValueError(
            "grouped eval trace must contain exactly one stage and run_id")
    if observed_samples != set(catalog):
        missing = sorted(
            str(catalog[key]["sample_id"])
            for key in set(catalog) - observed_samples)
        raise ValueError(
            f"grouped eval trace does not cover the fixed probe catalog: {missing[:5]}"
        )
    return records, "group_uid" if True in has_group_uid else "sample_id"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator


def _group_records(records: Sequence[dict[str, Any]],
                   key_mode: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str | int], list[dict[str,
                                                   Any]]] = defaultdict(list)
    for record in records:
        trace = record["trace"]
        key = (("uid", trace["group_uid"])
               if key_mode == "group_uid" else record["sample_key"])
        buckets[key].append(record)
    if len(buckets) != EXPECTED_GROUPS:
        raise ValueError(
            f"trace must contain {EXPECTED_GROUPS} groups, found {len(buckets)}"
        )

    sample_owners: dict[tuple[str, str | int], tuple[str, str | int]] = {}
    normalized_questions: set[str] = set()
    output: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda item: (item[0], str(item[1]))):
        members = buckets[key]
        if len(members) != GROUP_SIZE:
            raise ValueError(
                f"group {key[1]!r} must contain {GROUP_SIZE} trajectories")
        slots = [member["trace"]["group_slot"] for member in members]
        if set(slots) != set(range(GROUP_SIZE)) or len(slots) != len(
                set(slots)):
            raise ValueError(
                f"group {key[1]!r} must contain slots 0..{GROUP_SIZE - 1} exactly once"
            )
        sample_keys = {member["sample_key"] for member in members}
        if len(sample_keys) != 1:
            raise ValueError(f"group {key[1]!r} mixes sample identities")
        sample_key = next(iter(sample_keys))
        if sample_key in sample_owners:
            raise ValueError(
                f"sample {members[0]['trace']['sample_id']!r} appears in multiple groups"
            )
        sample_owners[sample_key] = key
        questions = {member["trace"]["question"] for member in members}
        answers = {
            tuple(member["trace"]["gold_answers"])
            for member in members
        }
        if len(questions) != 1 or len(answers) != 1:
            raise ValueError(
                f"group {key[1]!r} has inconsistent question metadata")
        normalized_question = _normalize_text(next(iter(questions)))
        if normalized_question in normalized_questions:
            raise ValueError(
                "different probe samples have duplicate normalized questions")
        normalized_questions.add(normalized_question)

        members.sort(key=lambda member: member["trace"]["group_slot"])
        qualifying = [member for member in members if member["qualifies"]]
        near_misses = [member for member in members if member["near_miss"]]
        clean_wrong = [
            member for member in members
            if member["clean"] and int(member["trace"]["em"]) == 0
        ]
        clean_correct = [
            member for member in members
            if member["clean"] and int(member["trace"]["em"]) == 1
        ]
        clean_em = {
            int(member["trace"]["em"])
            for member in members if member["clean"]
        }
        clean_searches = {
            member["trace"]["executed_search_count"]
            for member in members if member["clean"]
        }
        correct_searches = {
            member["trace"]["executed_search_count"]
            for member in clean_correct
        }
        metadata = members[0]["catalog"]
        failure_counts = Counter(
            reason for member in members
            for reason in member["qualification_failures"])
        output.append({
            "group_key":
            str(key[1]),
            "group_uid": (members[0]["trace"].get("group_uid")
                          if key_mode == "group_uid" else None),
            "sample_id":
            metadata["sample_id"],
            "category":
            metadata["category"],
            "question":
            metadata["question"],
            "gold_answers":
            metadata["gold_answers"],
            "supporting_titles":
            metadata["supporting_titles"],
            "trajectory_count":
            len(members),
            "slots": [member["trace"]["group_slot"] for member in members],
            "clean_count":
            sum(member["clean"] for member in members),
            "correct_count":
            sum(int(member["trace"]["em"]) == 1 for member in members),
            "correct_multi_search_count":
            sum(
                int(member["trace"]["em"]) == 1 and member["trace"]
                ["executed_search_count"] >= MIN_MULTI_SEARCHES
                for member in members),
            "valid_correct_multi_search_count":
            len(qualifying),
            "qualifying_slots":
            [member["trace"]["group_slot"] for member in qualifying],
            "near_miss_count":
            len(near_misses),
            "near_miss_slots":
            [member["trace"]["group_slot"] for member in near_misses],
            "near_miss_cover_em_count":
            sum(member["cover_em"] for member in near_misses),
            "near_miss_covered":
            bool(near_misses),
            "clean_wrong_count":
            len(clean_wrong),
            "covered":
            bool(qualifying),
            "outcome_diverse":
            len(clean_em) >= 2,
            "search_diverse":
            len(clean_searches) >= 2,
            "learnable":
            bool(qualifying and clean_wrong),
            "cost_contrast":
            len(clean_correct) >= 2 and len(correct_searches) >= 2,
            "qualification_failure_counts":
            dict(sorted(failure_counts.items())),
        })
    return output


def _aggregate(records: Sequence[dict[str, Any]],
               groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    trajectory_count = len(records)
    clipped = sum(record["trace"]["response_clipped"] for record in records)
    invalid = sum(record["trace"]["invalid_action_count"] > 0
                  for record in records)
    valid_correct_multi = sum(record["qualifies"] for record in records)
    near_misses = [record for record in records if record["near_miss"]]
    failures = Counter(reason for record in records
                       for reason in record["qualification_failures"])
    branches = Counter(record["evidence"]["evidence_branch"]
                       for record in records if record["qualifies"])
    return {
        "trajectory_count":
        trajectory_count,
        "question_count":
        len(groups),
        "correct_count":
        sum(int(record["trace"]["em"]) == 1 for record in records),
        "multi_search_count":
        sum(record["trace"]["executed_search_count"] >= MIN_MULTI_SEARCHES
            for record in records),
        "correct_multi_search_count":
        sum(
            int(record["trace"]["em"]) == 1
            and record["trace"]["executed_search_count"] >= MIN_MULTI_SEARCHES
            for record in records),
        "clean_count":
        sum(record["clean"] for record in records),
        "valid_correct_multi_search_count":
        valid_correct_multi,
        "valid_correct_multi_search_ratio":
        _ratio(valid_correct_multi, trajectory_count),
        "near_miss_count":
        len(near_misses),
        "near_miss_ratio":
        _ratio(len(near_misses), trajectory_count),
        "near_miss_covered_question_count":
        sum(group["near_miss_covered"] for group in groups),
        "near_miss_cover_em_count":
        sum(record["cover_em"] for record in near_misses),
        "near_miss_cover_em_ratio":
        (_ratio(sum(record["cover_em"] for record in near_misses),
                len(near_misses)) if near_misses else 0.0),
        "covered_question_count":
        sum(group["covered"] for group in groups),
        "learnable_group_count":
        sum(group["learnable"] for group in groups),
        "outcome_diverse_group_count":
        sum(group["outcome_diverse"] for group in groups),
        "search_diverse_group_count":
        sum(group["search_diverse"] for group in groups),
        "cost_contrast_group_count":
        sum(group["cost_contrast"] for group in groups),
        "clipped_count":
        clipped,
        "clipped_ratio":
        _ratio(clipped, trajectory_count),
        "invalid_action_trajectory_count":
        invalid,
        "invalid_action_ratio":
        _ratio(invalid, trajectory_count),
        "qualification_failure_counts":
        dict(sorted(failures.items())),
        "qualifying_evidence_branches": {
            key: branches[key]
            for key in ("answer", "supporting_title",
                        "answer_and_supporting_title")
        },
    }


def _criterion(observed: int | float, threshold: int | float,
               comparison: str) -> dict[str, Any]:
    passed = observed >= threshold if comparison == ">=" else observed <= threshold
    return {
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
        "passed": passed,
    }


def _decision(overall: Mapping[str, Any]) -> dict[str, Any]:
    criteria = {
        "valid_correct_multi_search_count":
        _criterion(overall["valid_correct_multi_search_count"],
                   THRESHOLDS["valid_correct_multi_search_count"], ">="),
        "covered_question_count":
        _criterion(overall["covered_question_count"],
                   THRESHOLDS["covered_question_count"], ">="),
        "learnable_group_count":
        _criterion(overall["learnable_group_count"],
                   THRESHOLDS["learnable_group_count"], ">="),
        "clipped_ratio":
        _criterion(overall["clipped_ratio"], THRESHOLDS["max_clipped_ratio"],
                   "<="),
        "invalid_action_ratio":
        _criterion(overall["invalid_action_ratio"],
                   THRESHOLDS["max_invalid_action_ratio"], "<="),
    }
    failed = [
        name for name, criterion in criteria.items() if not criterion["passed"]
    ]
    return {
        "decision": "GO" if not failed else "NO-GO",
        "criteria": criteria,
        "failed_criteria": failed,
    }


def _near_miss_example(record: Mapping[str, Any]) -> dict[str, Any]:
    trace = record["trace"]
    evidence = record["evidence"]
    return {
        "record_id": trace["record_id"],
        "sample_id": trace["sample_id"],
        "group_slot": trace["group_slot"],
        "category": record["catalog"]["category"],
        "question": trace["question"],
        "gold_answers": trace["gold_answers"],
        "extracted_answer": trace["extracted_answer"],
        "cover_em": record["cover_em"],
        "executed_search_count": trace["executed_search_count"],
        "queries": [event["query"] for event in record["events"]],
        "query_token_jaccard": evidence["query_token_jaccard"],
        "new_document_ids": evidence["new_document_ids"],
        "new_supporting_titles": evidence["new_supporting_titles"],
        "evidence_branch": evidence["evidence_branch"],
        "raw_trajectory": trace["raw_trajectory"],
        "turns": trace["turns"],
        "retrieval_events": trace["retrieval_events"],
    }


def _near_miss_diagnostic(records: Sequence[dict[str, Any]],
                          decision: str) -> dict[str, Any]:
    near_misses = sorted(
        (record for record in records if record["near_miss"]),
        key=lambda record:
        (str(record["trace"]["sample_id"]), record["trace"]["group_slot"]))
    by_category = Counter(record["catalog"]["category"]
                          for record in near_misses)
    covered_questions = {record["sample_key"] for record in near_misses}
    if decision == "GO":
        diagnosis = "strict_gate_passed"
    elif len(near_misses) >= NEAR_MISS_DIAGNOSTIC_THRESHOLD:
        diagnosis = "retrieval_chain_present_review_answer_extraction"
    else:
        diagnosis = "correct_multisearch_exploration_absent_or_rare"
    return {
        "definition":
        ("EM=0 while every non-EM valid-multisearch condition passes: "
         "searches>=2, not clipped, no invalid action, query Jaccard<0.8, "
         "a visible new document, and new supporting-title or answer evidence"
         ),
        "diagnostic_threshold":
        NEAR_MISS_DIAGNOSTIC_THRESHOLD,
        "threshold_met":
        len(near_misses) >= NEAR_MISS_DIAGNOSTIC_THRESHOLD,
        "affects_go_no_go":
        False,
        "diagnosis":
        diagnosis,
        "trajectory_count":
        len(near_misses),
        "covered_question_count":
        len(covered_questions),
        "by_category": {
            category: by_category[category]
            for category in CATEGORY_GROUPS
        },
        "cover_em_count":
        sum(record["cover_em"] for record in near_misses),
        "cover_em_definition":
        ("after Unicode/case/punctuation normalization, prediction and any "
         "gold alias contain one another on token boundaries; diagnostic only"
         ),
        "examples": [
            _near_miss_example(record)
            for record in near_misses[:NEAR_MISS_EXAMPLE_LIMIT]
        ],
    }


def _trajectory_report(record: Mapping[str, Any]) -> dict[str, Any]:
    trace = record["trace"]
    metadata = record["catalog"]
    evidence = record["evidence"]
    return {
        "record_id": trace["record_id"],
        "group_uid": trace.get("group_uid"),
        "group_slot": trace["group_slot"],
        "group_size": trace["group_size"],
        "sample_id": trace["sample_id"],
        "category": metadata["category"],
        "question": trace["question"],
        "gold_answers": trace["gold_answers"],
        "supporting_titles": metadata["supporting_titles"],
        "extracted_answer": trace["extracted_answer"],
        "em": int(trace["em"]),
        "executed_search_count": trace["executed_search_count"],
        "response_clipped": trace["response_clipped"],
        "invalid_action_count": trace["invalid_action_count"],
        "clean": record["clean"],
        "cover_em": record["cover_em"],
        "near_miss": record["near_miss"],
        "valid_correct_multi_search": record["qualifies"],
        "qualification_failures": record["qualification_failures"],
        **evidence,
        "raw_trajectory": trace["raw_trajectory"],
        "turns": trace["turns"],
        "retrieval_events": trace["retrieval_events"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    decision = summary["go_no_go"]
    near_miss = summary["near_miss_diagnostic"]
    lines = [
        "# Grouped Probe Analysis",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        f"The fixed probe contains {overall['question_count']} questions and "
        f"{overall['trajectory_count']} sampled trajectories. It found "
        f"{overall['valid_correct_multi_search_count']} valid correct multi-search "
        f"trajectories across {overall['covered_question_count']} questions.",
        "",
        "## Gate Criteria",
        "",
        "| Criterion | Observed | Requirement | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, criterion in decision["criteria"].items():
        observed = criterion["observed"]
        threshold = criterion["threshold"]
        if isinstance(observed, float):
            observed = f"{observed:.4f}"
        if isinstance(threshold, float):
            threshold = f"{threshold:.4f}"
        lines.append(
            f"| `{name}` | {observed} | {criterion['comparison']} {threshold} | "
            f"{'PASS' if criterion['passed'] else 'FAIL'} |")
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- Learnable groups: {overall['learnable_group_count']}; cost-contrast "
        f"groups: {overall['cost_contrast_group_count']}.",
        f"- Clipped trajectories: {overall['clipped_count']} "
        f"({overall['clipped_ratio']:.2%}); invalid-action trajectories: "
        f"{overall['invalid_action_trajectory_count']} "
        f"({overall['invalid_action_ratio']:.2%}).",
        f"- Near misses: {near_miss['trajectory_count']} trajectories across "
        f"{near_miss['covered_question_count']} questions; cover-EM "
        f"{near_miss['cover_em_count']}; diagnosis: `{near_miss['diagnosis']}`.",
        "- Category metrics are available in `summary.json`; question- and "
        "trajectory-level evidence is in the companion JSONL files.",
        "",
        "## Definitions",
        "",
        "A valid correct multi-search trajectory has EM=1, at least two searches, "
        "no clipping or invalid action, query-token Jaccard below 0.8, a new "
        "second-round document, and new supporting-title or first-visible answer evidence.",
        "A learnable group contains at least one such trajectory and at least one "
        "clean wrong trajectory, so correctness-gated group reward has non-zero contrast.",
        "A near miss has EM=0 but passes every other valid multi-search condition. "
        "The >=16 diagnostic threshold never changes the strict GO/NO-GO decision.",
        "",
    ])
    if near_miss["examples"]:
        lines.extend(["## Near-Miss Examples", ""])
        for example in near_miss["examples"]:
            lines.append(
                f"- `{example['record_id']}` ({example['category']}, "
                f"{example['executed_search_count']} searches): extracted "
                f"`{example['extracted_answer']}`; cover-EM="
                f"{str(example['cover_em']).lower()}.")
        lines.append("")
    return "\n".join(lines)


def _json_bytes(value: object, pretty: bool = True) -> bytes:
    separators = None if pretty else (",", ":")
    return (json.dumps(value,
                       ensure_ascii=False,
                       sort_keys=True,
                       indent=2 if pretty else None,
                       separators=separators,
                       allow_nan=False) + "\n").encode("utf-8")


def _publish(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != OUTPUT_FILES:
        raise ValueError("probe analysis output set is incomplete")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.",
                         dir=str(output_dir.parent)))
    try:
        for name, payload in payloads.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    trace_path = _regular_file(Path(args.trace), "grouped eval trace")
    catalog_path = _regular_file(Path(args.catalog), "search-mix catalog")
    expected_digest = args.expected_checkpoint_digest
    if (not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None):
        raise ValueError(
            "expected checkpoint digest must be a lowercase SHA-256 digest")
    catalog = _load_probe_catalog(catalog_path)
    records, key_mode = _load_trace(trace_path, catalog, expected_digest)
    groups = _group_records(records, key_mode)
    overall = _aggregate(records, groups)
    by_category: dict[str, Any] = {}
    for category in CATEGORY_GROUPS:
        category_records = [
            record for record in records
            if record["catalog"]["category"] == category
        ]
        category_groups = [
            group for group in groups if group["category"] == category
        ]
        by_category[category] = _aggregate(category_records, category_groups)
    decision = _decision(overall)
    near_miss_diagnostic = _near_miss_diagnostic(records, decision["decision"])
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "decision": decision["decision"],
        "input": {
            "trace_path": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "checkpoint_digest": expected_digest,
            "stage": records[0]["trace"]["stage"],
            "run_id": records[0]["trace"]["run_id"],
            "group_key_mode": key_mode,
        },
        "contract": {
            "expected_questions":
            EXPECTED_GROUPS,
            "trajectories_per_question":
            GROUP_SIZE,
            "expected_trajectories":
            EXPECTED_ROWS,
            "max_searches":
            MAX_SEARCHES,
            "multi_search_minimum":
            MIN_MULTI_SEARCHES,
            "max_query_token_jaccard":
            MAX_QUERY_TOKEN_JACCARD,
            "near_miss_diagnostic_threshold":
            NEAR_MISS_DIAGNOSTIC_THRESHOLD,
            "category_questions":
            CATEGORY_GROUPS,
            "thresholds":
            THRESHOLDS,
            "valid_correct_multi_search_definition":
            ("EM=1; searches>=2; not clipped; invalid_action_count=0; "
             "normalized query-token Jaccard(query 1, query 2)<0.8; retrieval 2 adds a "
             "document_id; retrieval 2 adds a catalog supporting title or "
             "makes a gold alias visible for the first time"),
            "learnable_group_definition":
            ("at least one valid correct multi-search trajectory and at least "
             "one unclipped, invalid-action-free EM=0 trajectory"),
            "near_miss_definition":
            ("EM=0 and every non-EM valid-correct-multisearch condition passes; "
             "diagnostic only and excluded from GO criteria"),
            "invalid_action_ratio_denominator":
            "trajectories",
        },
        "overall": overall,
        "by_category": by_category,
        "near_miss_diagnostic": near_miss_diagnostic,
        "go_no_go": decision,
    }
    per_trajectory = sorted((_trajectory_report(record) for record in records),
                            key=lambda item:
                            (str(item["sample_id"]), item["group_slot"]))
    payloads = {
        "summary.json":
        _json_bytes(summary),
        "summary.md":
        _markdown(summary).encode("utf-8"),
        "go_no_go.json":
        _json_bytes({
            "schema":
            ANALYSIS_SCHEMA,
            "schema_version":
            ANALYSIS_SCHEMA_VERSION,
            "decision":
            decision["decision"],
            "criteria":
            decision["criteria"],
            "failed_criteria":
            decision["failed_criteria"],
            "near_miss_diagnosis":
            near_miss_diagnostic["diagnosis"],
            "near_miss_count":
            near_miss_diagnostic["trajectory_count"],
            "near_miss_threshold_met":
            near_miss_diagnostic["threshold_met"],
            "trace_sha256":
            summary["input"]["trace_sha256"],
            "catalog_sha256":
            summary["input"]["catalog_sha256"],
            "checkpoint_digest":
            expected_digest,
        }),
        "per_trajectory.jsonl":
        b"".join(
            _json_bytes(record, pretty=False) for record in per_trajectory),
        "per_question.jsonl":
        b"".join(_json_bytes(group, pretty=False) for group in groups),
    }
    _publish(Path(args.output_dir), payloads)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the fixed 64x5 Search-R1 grouped behavior probe.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fail-on-no-go",
        action="store_true",
        help=
        "Return 2 for a scientific NO-GO; default is exit 0 so it is not retried.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = analyze(args)
    except (OSError, ValueError) as error:
        print(f"grouped probe analysis error: {error}", file=sys.stderr)
        return 1
    print(f"Grouped probe decision: {summary['decision']}")
    if args.fail_on_no_go and summary["decision"] == "NO-GO":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
