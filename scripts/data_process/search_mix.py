#!/usr/bin/env python3
"""Build a retrieval-verified NQ/HotpotQA training mixture."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence
import unicodedata

from search_r1.llm_agent.tool_protocol import (
    LEGACY_XML,
    QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_MODEL_REVISION,
    QWEN35_NATIVE,
    QWEN35_PROMPT_VERSION,
    SUPPORTED_TOOL_PROTOCOLS,
    normalize_tool_protocol,
    qwen35_messages,
    qwen35_tools,
    qwen35_tool_schema_sha256,
    render_qwen35_prompt,
    validate_qwen35_messages,
)

DATASET_NAME = "RUC-NLPIR/FlashRAG_datasets"
DATASET_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
BM25_REVISION = "2c7554f25f425038c4bcb155735a0f831851fd78"
CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
CORPUS_SHA256 = "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"
SEED = 42
TOPK = 3
MAX_OBS_LENGTH = 384
SCHEMA_VERSION = 2
MATERIALIZED_SCHEMA_VERSION = 3
SELECTION_POLICY = "retrieval-verified-search-mix-v1"
LEGACY_PROMPT_VERSION = "search-r1-legacy-xml-v1"

SOURCE_SPECS = {
    "nq": {
        "repo_file":
        "nq/train.jsonl",
        "file":
        "sources/nq/train.jsonl",
        "bytes":
        9_960_189,
        "sha256":
        "572685d3f384d9c37479b1fb14232984c85ff13b58923c0c9442232bd5d5647b",
    },
    "hotpotqa": {
        "repo_file":
        "hotpotqa/train.jsonl",
        "file":
        "sources/hotpotqa/train.jsonl",
        "bytes":
        569_520_788,
        "sha256":
        "a81274abafa899ec0ee073102edbe6bb694a8a1174201b4e43e2bc6c98964d1a",
    },
}

CANDIDATE_LIMITS = {
    "single": 8192,
    "comparison": 8192,
    "bridge": 16384,
}
RETRIEVAL_TARGETS = {
    # These are feasibility floors; materialization applies the stricter
    # tokenizer, exclusion, and fixed-quota checks to the full evidence pool.
    "comparison": 240,
    "bridge": 144,
}
QUOTAS = {
    "single": {
        "train": 192,
        "val": 64
    },
    "comparison": {
        "train": 56,
        "val": 16
    },
    "bridge": {
        "train": 264,
        "val": 48
    },
}

EVIDENCE_FILE = "retrieval_evidence.jsonl"
RETRIEVAL_LEDGER_FILE = "retrieval_ledger.json"
CATALOG_FILE = "catalog.jsonl"
EXCLUSIONS_FILE = "exclusions.json"
MANIFEST_FILE = "manifest.json"
REPLAY_FILE = "retrieval_replay.json"
SELECTION_FUNNEL_FILE = "selection_funnel.json"
OUTPUT_FILES = {
    "train": "train_512.parquet",
    "val": "val_128.parquet",
    "probe": "probe_multi_64.parquet",
}
NATIVE_PROBE_FILES = {
    "probe_g0": ("probe_g0_8.parquet", 8, False),
    "probe_forced": ("probe_forced_16.parquet", 16, True),
    "probe_autonomous": ("probe_autonomous_32.parquet", 32, False),
}
BAD_ANSWERS = {"yes", "no", "true", "false", "unknown"}


class JsonlRows:
    """Index a JSONL source without retaining its large contexts in memory."""

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(f"Blank line in pinned source: {path}")
                self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if type(index) is not int or not 0 <= index < len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            raw = handle.readline()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid source JSON at {self.path}:{index + 1}") from error
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Source row is not an object at {self.path}:{index + 1}")
        return value


class SelectionQuotaError(ValueError):
    """Carry deterministic selection counts when a fixed quota is infeasible."""

    def __init__(self, shortfalls: Mapping[str, Mapping[str, int]],
                 rejection_counts: Counter[str], stats: Mapping[str, Any]):
        self.shortfalls = {
            category: dict(values)
            for category, values in shortfalls.items()
        }
        self.rejection_counts = Counter(rejection_counts)
        self.stats = dict(stats)
        details = ", ".join(
            f"{category}={values['available']}/{values['required']}"
            for category, values in sorted(self.shortfalls.items()))
        super().__init__(f"Materialization quota shortfall: {details}")


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
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_digest_sidecar(path: Path) -> None:
    atomic_write(path.with_suffix(path.suffix + ".sha256"),
                 f"{sha256_file(path)}  {path.name}\n".encode("ascii"))


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character if character.isalnum() else " "
                    for character in value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_question(value: object) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("Question must be a non-empty string")
    return normalized


def clean_question(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Question must be a non-empty string")
    return re.sub(r"\s+", " ", value).strip()


def clean_answers(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_answers: Sequence[object] = [value]
    elif isinstance(value, Sequence):
        raw_answers = value
    else:
        raise ValueError("golden_answers must be a string sequence")
    answers: list[str] = []
    seen: set[str] = set()
    for answer in raw_answers:
        if not isinstance(answer, str) or not answer.strip():
            continue
        cleaned = re.sub(r"\s+", " ", answer).strip()
        key = normalize_text(cleaned)
        if key and key not in seen:
            answers.append(cleaned)
            seen.add(key)
    if not answers:
        raise ValueError("Every sample needs a non-empty golden answer")
    return tuple(answers)


def usable_answers(answers: Sequence[str]) -> bool:
    normalized = [normalize_text(answer) for answer in answers]
    return any(
        len(answer) >= 3 and answer not in BAD_ANSWERS
        for answer in normalized)


def phrase_visible(phrases: Iterable[str], text: str) -> bool:
    haystack = f" {normalize_text(text)} "
    return any(phrase and f" {phrase} " in haystack
               for phrase in (normalize_text(value) for value in phrases))


def stable_key(namespace: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}|{namespace}|{sample_id}".encode("utf-8")).hexdigest()


def prompt_contract(tool_protocol: str) -> dict[str, Any]:
    protocol = normalize_tool_protocol(tool_protocol)
    return {
        "tool_protocol": protocol,
        "prompt_version": (QWEN35_PROMPT_VERSION
                           if protocol == QWEN35_NATIVE else
                           LEGACY_PROMPT_VERSION),
        "tool_schema_sha256": (qwen35_tool_schema_sha256()
                               if protocol == QWEN35_NATIVE else None),
        "tokenizer_revision": (QWEN35_MODEL_REVISION
                               if protocol == QWEN35_NATIVE else
                               MODEL_REVISION),
        "chat_template_sha256": (QWEN35_CHAT_TEMPLATE_SHA256
                                 if protocol == QWEN35_NATIVE else None),
    }


def make_prefix(question: str) -> str:
    question = clean_question(question)
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
        f"Question: {question}\n")


def make_prompt(question: object,
                tool_protocol: str = LEGACY_XML,
                force_search: bool = False) -> list[dict[str, str]]:
    protocol = normalize_tool_protocol(tool_protocol)
    if protocol == QWEN35_NATIVE:
        return qwen35_messages(str(question), force_search=force_search)
    if force_search:
        raise ValueError("force_search is only supported by qwen35_native")
    return [{"role": "user", "content": make_prefix(str(question))}]


def source_path(local_dir: Path, source: str) -> Path:
    root = local_dir.resolve()
    relative = Path(str(SOURCE_SPECS[source]["file"]))
    path = root / relative
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Source path escapes local directory: {source}") from error
    return path


def verify_source(local_dir: Path, source: str) -> Path:
    path = source_path(local_dir, source)
    spec = SOURCE_SPECS[source]
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Pinned source is missing or symlinked: {source}")
    if path.stat().st_size != spec["bytes"] or sha256_file(
            path) != spec["sha256"]:
        raise ValueError(f"Pinned source identity mismatch: {source}")
    return path


def download_sources(local_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    result = {}
    for source, spec in SOURCE_SPECS.items():
        target = source_path(local_dir, source)
        if target.is_file() and not target.is_symlink():
            result[source] = verify_source(local_dir, source)
            continue
        if target.exists() or target.is_symlink():
            raise ValueError(f"Invalid pinned source target: {target}")
        cached = Path(
            hf_hub_download(repo_id=DATASET_NAME,
                            repo_type="dataset",
                            filename=str(spec["repo_file"]),
                            revision=DATASET_REVISION))
        if cached.stat().st_size != spec["bytes"] or sha256_file(
                cached) != spec["sha256"]:
            raise ValueError(f"Downloaded source identity mismatch: {source}")
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
        result[source] = verify_source(local_dir, source)
    return result


def _supporting_metadata(
        row: Mapping[str,
                     Any]) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("HotpotQA row is missing metadata")
    supporting = metadata.get("supporting_facts")
    context = metadata.get("context")
    if not isinstance(supporting, Mapping) or not isinstance(context, Mapping):
        raise ValueError("HotpotQA row is missing supporting facts or context")
    titles = supporting.get("title")
    sentence_ids = supporting.get("sent_id")
    context_titles = context.get("title")
    context_sentences = context.get("sentences")
    if not all(
            isinstance(value, Sequence) and not isinstance(value, str)
            for value in (titles, sentence_ids, context_titles,
                          context_sentences)):
        raise ValueError("HotpotQA metadata has an invalid sequence field")
    if len(titles) != len(sentence_ids) or len(context_titles) != len(
            context_sentences):
        raise ValueError("HotpotQA metadata arrays are not aligned")

    unique_titles: list[str] = []
    for title in titles:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("HotpotQA supporting title is invalid")
        cleaned = title.strip()
        if cleaned not in unique_titles:
            unique_titles.append(cleaned)
    context_by_title = {
        str(title).strip(): sentences
        for title, sentences in zip(context_titles, context_sentences)
        if isinstance(title, str) and isinstance(sentences, Sequence)
        and not isinstance(sentences, str)
    }
    facts: dict[str, list[str]] = {title: [] for title in unique_titles}
    for title, sentence_id in zip(titles, sentence_ids):
        if type(sentence_id) is not int or title not in context_by_title:
            raise ValueError("HotpotQA supporting fact cannot be resolved")
        sentences = context_by_title[title]
        if not 0 <= sentence_id < len(sentences):
            raise ValueError(
                "HotpotQA supporting sentence id is outside context")
        sentence = sentences[sentence_id]
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("HotpotQA supporting sentence is invalid")
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if cleaned not in facts[title]:
            facts[title].append(cleaned)
    if any(not facts[title] for title in unique_titles):
        raise ValueError(
            "HotpotQA supporting title has no supporting sentence")
    return tuple(unique_titles), facts


def _hotpot_type(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    sample_type = metadata.get("type") if isinstance(metadata,
                                                     Mapping) else None
    if sample_type not in ("comparison", "bridge"):
        raise ValueError("HotpotQA row is not comparison or bridge")
    return str(sample_type)


def _candidate(source: str, source_index: int,
               row: Mapping[str, Any]) -> dict[str, Any]:
    question = clean_question(row.get("question"))
    answers = clean_answers(row.get("golden_answers"))
    if not usable_answers(answers):
        raise ValueError("Sample has an ambiguous short or boolean answer")
    sample_id = f"{source}:train:{source_index}"
    base: dict[str, Any] = {
        "source_id": sample_id,
        "data_source": source,
        "source_split": "train",
        "source_index": source_index,
        "question": question,
        "golden_answers": list(answers),
    }
    if source == "nq":
        if phrase_visible(answers, question):
            raise ValueError("NQ answer is already visible in the question")
        base["category"] = "single"
        return base

    sample_type = _hotpot_type(row)
    titles, facts = _supporting_metadata(row)
    if len(titles) != 2:
        raise ValueError(
            "HotpotQA candidate must have exactly two supporting titles")
    visible_titles = [
        title for title in titles if phrase_visible([title], question)
    ]
    if sample_type == "comparison" and len(visible_titles) != 2:
        raise ValueError(
            "Comparison titles must both be visible in the question")
    if sample_type == "bridge" and len(visible_titles) != 1:
        raise ValueError(
            "Bridge question must expose exactly one supporting title")
    if phrase_visible(answers, question):
        raise ValueError("HotpotQA answer is already visible in the question")
    base.update({
        "category": sample_type,
        "type": sample_type,
        "level": row.get("metadata", {}).get("level"),
        "supporting_titles": list(titles),
        "supporting_facts": facts,
    })
    return base


def collect_candidates(
    rows: JsonlRows, source: str, limits: Mapping[str, int]
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        category: []
        for category in limits
    }
    rejected: Counter[str] = Counter()
    for index in range(len(rows)):
        try:
            candidate = _candidate(source, index, rows[index])
        except (TypeError, ValueError) as error:
            rejected[f"prescreen:{str(error)}"] += 1
            continue
        category = str(candidate["category"])
        if category not in candidates:
            continue
        candidates[category].append(candidate)
    after_prescreen = {
        category: len(values)
        for category, values in candidates.items()
    }
    for category, values in candidates.items():
        values.sort(
            key=lambda item: stable_key("candidate", str(item["source_id"])))
        candidates[category] = values[:limits[category]]
    stats = {
        "source_rows": len(rows),
        "candidates_after_prescreen": after_prescreen,
        "candidates_after_cap": {
            category: len(values)
            for category, values in candidates.items()
        },
    }
    return candidates, rejected, stats


def _selection_funnel_payload(
        status: str,
        stage: str,
        retrieval: Mapping[str, Any],
        materialize: Optional[Mapping[str, Any]] = None,
        failure: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "seed": SEED,
        "status": status,
        "stage": stage,
        "quotas": QUOTAS,
        "candidate_limits": CANDIDATE_LIMITS,
        "retrieval_targets": RETRIEVAL_TARGETS,
        "retrieval": dict(retrieval),
        "materialize": None if materialize is None else dict(materialize),
        "failure": None if failure is None else dict(failure),
    }


def _write_selection_funnel(local_dir: Path, payload: Mapping[str,
                                                              Any]) -> Path:
    path = local_dir / SELECTION_FUNNEL_FILE
    atomic_write(path, canonical_json_bytes(payload))
    write_digest_sidecar(path)
    return path


def _serialize_hits(hits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for rank, hit in enumerate(hits, 1):
        document = hit.get("document")
        if not isinstance(document, Mapping):
            raise ValueError("BM25 result is missing its document")
        contents = document.get("contents")
        if not isinstance(contents, str) or not contents.strip():
            raise ValueError("BM25 result has empty contents")
        title = document.get("title")
        if not isinstance(title, str) or not title.strip():
            title = contents.partition("\n")[0].strip('" ')
        document_id = hit.get("document_id")
        score = float(hit.get("score", 0.0))
        if document_id is None or not str(document_id).strip():
            raise ValueError("BM25 result has an invalid document id")
        if not math.isfinite(score):
            raise ValueError("BM25 result has a non-finite score")
        output.append({
            "rank": rank,
            "document_id": str(document_id),
            "title": title.strip(),
            "score": score,
            "contents": contents,
        })
    return output


def _title_coverage(titles: Sequence[str],
                    hits: Sequence[Mapping[str, Any]]) -> list[str]:
    hit_titles = {normalize_text(hit.get("title")) for hit in hits}
    return [title for title in titles if normalize_text(title) in hit_titles]


def _observation_parts(hits: Sequence[Mapping[str, Any]]) -> list[str]:
    output = []
    for index, hit in enumerate(hits, 1):
        contents = str(hit["contents"])
        raw_title, _, body = contents.partition("\n")
        output.append(f"Doc {index}(Title: {raw_title}) {body}\n")
    return output


def _observation_text(hits: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_observation_parts(hits))


def retrieve_evidence(local_dir: Path, index_path: Path, corpus_path: Path,
                      offsets_path: Path) -> Path:
    from search_r1.search.bm25_server import BM25Retriever

    local_dir = local_dir.resolve()
    nq_rows = JsonlRows(verify_source(local_dir, "nq"))
    hotpot_rows = JsonlRows(verify_source(local_dir, "hotpotqa"))
    nq_candidates, nq_rejected, nq_stats = collect_candidates(
        nq_rows, "nq", {"single": CANDIDATE_LIMITS["single"]})
    hotpot_candidates, hotpot_rejected, hotpot_stats = collect_candidates(
        hotpot_rows, "hotpotqa", {
            "comparison": CANDIDATE_LIMITS["comparison"],
            "bridge": CANDIDATE_LIMITS["bridge"],
        })
    retriever = BM25Retriever(str(index_path),
                              topk=TOPK,
                              corpus_path=str(corpus_path),
                              offsets_path=str(offsets_path))
    rejection_counts = nq_rejected + hotpot_rejected
    evidence: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    bm25_query_calls = 0

    for candidate in nq_candidates["single"]:
        ordered_ids.append(str(candidate["source_id"]))
        bm25_query_calls += 1
        first = _serialize_hits(
            retriever.search(candidate["question"],
                             topk=TOPK,
                             return_scores=True))
        if not first:
            rejection_counts["retrieve:no_first_results"] += 1
            continue
        evidence.append({
            **candidate, "first_query": candidate["question"],
            "first_results": first,
            "second_query": None,
            "second_results": []
        })

    for category in ("comparison", "bridge"):
        for candidate in hotpot_candidates[category]:
            ordered_ids.append(str(candidate["source_id"]))
            bm25_query_calls += 1
            first = _serialize_hits(
                retriever.search(candidate["question"],
                                 topk=TOPK,
                                 return_scores=True))
            coverage = _title_coverage(candidate["supporting_titles"], first)
            if len(coverage) != 1:
                rejection_counts[
                    f"retrieve:{category}:first_support_count_{len(coverage)}"] += 1
                continue
            missing = next(title for title in candidate["supporting_titles"]
                           if title != coverage[0])
            if category == "comparison":
                derivable = phrase_visible([missing], candidate["question"])
                derivable_from = "question"
            else:
                derivable = phrase_visible([missing], _observation_text(first))
                derivable_from = "first_observation"
            if not derivable:
                rejection_counts[
                    f"retrieve:{category}:second_query_not_derivable"] += 1
                continue
            bm25_query_calls += 1
            second = _serialize_hits(
                retriever.search(missing, topk=TOPK, return_scores=True))
            if missing not in _title_coverage([missing], second):
                rejection_counts[
                    f"retrieve:{category}:second_support_missing"] += 1
                continue
            first_ids = {hit["document_id"] for hit in first}
            new_document_count = sum(hit["document_id"] not in first_ids
                                     for hit in second)
            new_support = any(
                hit["document_id"] not in first_ids
                and normalize_text(hit["title"]) == normalize_text(missing)
                for hit in second)
            if new_document_count == 0 or not new_support:
                rejection_counts[f"retrieve:{category}:no_new_document"] += 1
                continue
            evidence.append({
                **candidate,
                "first_query": candidate["question"],
                "first_results": first,
                "first_supporting_title": coverage[0],
                "second_query": missing,
                "second_query_derivable_from": derivable_from,
                "second_results": second,
                "new_document_count": new_document_count,
            })

    evidence.sort(
        key=lambda item: stable_key("evidence", str(item["source_id"])))
    order_digest = hashlib.sha256("".join(
        f"{sample_id}\n"
        for sample_id in ordered_ids).encode("utf-8")).hexdigest()
    candidates_after_prescreen = {
        **nq_stats["candidates_after_prescreen"],
        **hotpot_stats["candidates_after_prescreen"],
    }
    candidates_after_cap = {
        **nq_stats["candidates_after_cap"],
        **hotpot_stats["candidates_after_cap"],
    }
    structurally_valid = Counter(
        str(record["category"]) for record in evidence)
    retrieval_summary = {
        "source_rows": {
            "nq": nq_stats["source_rows"],
            "hotpotqa": hotpot_stats["source_rows"],
        },
        "candidates_after_prescreen": candidates_after_prescreen,
        "candidates_after_cap": candidates_after_cap,
        "candidates_truncated_by_cap": {
            category:
            candidates_after_prescreen[category] -
            candidates_after_cap[category]
            for category in CANDIDATE_LIMITS
        },
        "candidate_items_queried": len(ordered_ids),
        "bm25_query_calls": bm25_query_calls,
        "structurally_valid": {
            category: structurally_valid[category]
            for category in CANDIDATE_LIMITS
        },
        "evidence_rows": len(evidence),
        "candidate_order_sha256": order_digest,
        "evidence_sha256": None,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    shortfalls = {
        category: {
            "available": structurally_valid[category],
            "required": required,
            "missing": required - structurally_valid[category],
        }
        for category, required in RETRIEVAL_TARGETS.items()
        if structurally_valid[category] < required
    }
    if shortfalls:
        _write_selection_funnel(
            local_dir,
            _selection_funnel_payload("failed",
                                      "retrieve",
                                      retrieval_summary,
                                      failure={
                                          "stage": "retrieve",
                                          "shortfalls": shortfalls,
                                      }))
        details = ", ".join(
            f"{category}={values['available']}/{values['required']}"
            for category, values in sorted(shortfalls.items()))
        raise ValueError(f"Retrieval quota shortfall: {details}")

    evidence_path = local_dir / EVIDENCE_FILE
    atomic_write(evidence_path,
                 b"".join(canonical_json_bytes(item) for item in evidence))
    write_digest_sidecar(evidence_path)
    retrieval_summary["evidence_sha256"] = sha256_file(evidence_path)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "seed": SEED,
        "topk": TOPK,
        "candidate_limits": CANDIDATE_LIMITS,
        "retrieval_targets": RETRIEVAL_TARGETS,
        "candidate_order_sha256": order_digest,
        "candidate_queries": len(ordered_ids),
        "evidence_rows": len(evidence),
        "evidence_sha256": sha256_file(evidence_path),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "bm25_revision": BM25_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "corpus_sha256": CORPUS_SHA256,
    }
    ledger_path = local_dir / RETRIEVAL_LEDGER_FILE
    atomic_write(ledger_path, canonical_json_bytes(ledger))
    write_digest_sidecar(ledger_path)
    _write_selection_funnel(
        local_dir,
        _selection_funnel_payload("retrieval_complete", "retrieve",
                                  retrieval_summary))
    return ledger_path


def read_canonical_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record,
                              Mapping) or canonical_json_bytes(record) != line:
                raise ValueError(
                    f"Non-canonical JSONL at {path}:{line_number}")
            records.append(record)
    return records


def _validate_hits(value: object, *,
                   allow_empty: bool) -> list[Mapping[str, Any]]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or (not allow_empty and not value) or len(value) > TOPK):
        raise ValueError("Retrieval results have an invalid row count")
    hits = list(value)
    document_ids: set[str] = set()
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, Mapping) or set(hit) != {
                "rank", "document_id", "title", "score", "contents"
        }:
            raise ValueError("Retrieval result has an invalid schema")
        document_id = hit["document_id"]
        if (hit["rank"] != rank or not isinstance(document_id, str)
                or not document_id or document_id in document_ids
                or not isinstance(hit["title"], str)
                or not hit["title"].strip()
                or not isinstance(hit["contents"], str)
                or not hit["contents"].strip()
                or type(hit["score"]) not in (int, float)
                or not math.isfinite(float(hit["score"]))):
            raise ValueError("Retrieval result has an invalid value")
        document_ids.add(document_id)
    return hits


def validate_evidence_sources(local_dir: Path,
                              evidence: Sequence[Mapping[str, Any]]) -> None:
    rows = {
        source: JsonlRows(verify_source(local_dir, source))
        for source in SOURCE_SPECS
    }
    source_ids: list[str] = []
    for record in evidence:
        if not isinstance(record, Mapping):
            raise ValueError("Retrieval evidence row must be an object")
        source = record.get("data_source")
        source_index = record.get("source_index")
        if (not isinstance(source, str) or source not in rows
                or type(source_index) is not int):
            raise ValueError(
                "Retrieval evidence has an invalid source identity")
        if not 0 <= source_index < len(rows[source]):
            raise ValueError(
                "Retrieval evidence source index is outside the pinned source")
        candidate = _candidate(str(source), source_index,
                               rows[str(source)][source_index])
        for key, expected in candidate.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"Retrieval evidence differs from pinned source: {candidate['source_id']}:{key}"
                )

        category = candidate["category"]
        candidate_fields = set(candidate)
        retrieval_fields = {
            "first_query", "first_results", "second_query", "second_results"
        }
        if category != "single":
            retrieval_fields.update({
                "first_supporting_title", "second_query_derivable_from",
                "new_document_count"
            })
        if set(record) != candidate_fields | retrieval_fields:
            raise ValueError(
                f"Retrieval evidence has unknown or missing fields: {candidate['source_id']}"
            )
        first = _validate_hits(record["first_results"], allow_empty=False)
        second = _validate_hits(record["second_results"],
                                allow_empty=category == "single")
        if record["first_query"] != candidate["question"]:
            raise ValueError(
                "First retrieval query must equal the source question")
        if category == "single":
            if record["second_query"] is not None or second:
                raise ValueError(
                    "Single-search evidence cannot contain a second query")
        else:
            coverage = _title_coverage(candidate["supporting_titles"], first)
            if len(coverage) != 1:
                raise ValueError(
                    "First retrieval must cover exactly one supporting title")
            missing = next(title for title in candidate["supporting_titles"]
                           if title != coverage[0])
            expected_derivation = ("question" if category == "comparison" else
                                   "first_observation")
            first_ids = {str(hit["document_id"]) for hit in first}
            new_document_count = sum(
                str(hit["document_id"]) not in first_ids for hit in second)
            new_support = any(
                str(hit["document_id"]) not in first_ids
                and normalize_text(hit["title"]) == normalize_text(missing)
                for hit in second)
            if (record["first_supporting_title"] != coverage[0]
                    or record["second_query"] != missing
                    or record["second_query_derivable_from"]
                    != expected_derivation
                    or record["new_document_count"] != new_document_count
                    or new_document_count <= 0 or not new_support):
                raise ValueError("Second retrieval evidence is inconsistent")
        source_ids.append(str(candidate["source_id"]))

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Retrieval evidence has duplicate source ids")
    expected_order = sorted(source_ids,
                            key=lambda value: stable_key("evidence", value))
    if source_ids != expected_order:
        raise ValueError("Retrieval evidence is not in deterministic order")


def _visible_observation_by_hit(
        tokenizer: Any, hits: Sequence[Mapping[str,
                                               Any]]) -> tuple[str, list[str]]:
    prefix = "\n\n<information>"
    parts = _observation_parts(hits)
    body = "".join(parts).strip()
    wrapped = f"{prefix}{body}</information>\n\n"
    spans = []
    cursor = 0
    for part in parts:
        start = min(cursor, len(body))
        cursor += len(part)
        end = min(cursor, len(body))
        spans.append((len(prefix) + start, len(prefix) + end))

    try:
        tokenized = tokenizer(wrapped,
                              add_special_tokens=False,
                              return_offsets_mapping=True)
    except (NotImplementedError, TypeError) as error:
        raise ValueError(
            "The tokenizer must provide offset_mapping for evidence filtering"
        ) from error
    input_ids = tokenized["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    offsets = tokenized.get("offset_mapping")
    if (offsets and len(offsets) == 1 and isinstance(offsets[0], list)
            and offsets[0] and isinstance(offsets[0][0], (list, tuple))):
        offsets = offsets[0]
    if (not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes))
            or len(offsets) != len(input_ids)):
        raise ValueError("Tokenizer returned an invalid offset_mapping")
    normalized_offsets = []
    for offset in offsets:
        if (not isinstance(offset, Sequence)
                or isinstance(offset, (str, bytes)) or len(offset) != 2
                or any(type(value) is not int for value in offset)):
            raise ValueError("Tokenizer returned an invalid offset_mapping")
        normalized_offsets.append((int(offset[0]), int(offset[1])))

    visible_ids = input_ids[:MAX_OBS_LENGTH]
    visible_offsets = normalized_offsets[:MAX_OBS_LENGTH]
    visible_end = max((end for start, end in visible_offsets if end > start),
                      default=0)
    visible_hits = [
        wrapped[start:min(end, visible_end)] if visible_end > start else ""
        for start, end in spans
    ]
    visible = tokenizer.decode(visible_ids, skip_special_tokens=True)
    return visible, visible_hits


def visible_observation(tokenizer: Any, hits: Sequence[Mapping[str,
                                                               Any]]) -> str:
    return _visible_observation_by_hit(tokenizer, hits)[0]


def _facts_visible(facts: Mapping[str, Sequence[str]], title: str,
                   text: str) -> bool:
    sentences = facts.get(title, [])
    return bool(sentences) and all(
        phrase_visible([sentence], text) for sentence in sentences)


def _any_fact_visible(facts: Mapping[str, Sequence[str]], title: str,
                      text: str) -> bool:
    return any(
        phrase_visible([sentence], text) for sentence in facts.get(title, []))


def _document_rank_with_phrase(hits: Sequence[Mapping[str, Any]],
                               phrases: Sequence[str]) -> int:
    for hit in hits:
        if phrase_visible(phrases, str(hit["contents"])):
            return int(hit["rank"])
    return TOPK + 1


def evaluate_evidence(record: Mapping[str, Any],
                      tokenizer: Any) -> tuple[bool, str, dict[str, Any]]:
    first = record.get("first_results")
    second = record.get("second_results")
    if not isinstance(first, Sequence) or not isinstance(second, Sequence):
        return False, "malformed_results", {}
    first_visible, _ = _visible_observation_by_hit(tokenizer, first)
    answers = list(clean_answers(record.get("golden_answers")))
    category = record.get("category")
    if category == "single":
        if not phrase_visible(answers, first_visible):
            return False, "single_gold_not_visible", {}
        audit = {
            "first_gold_seen":
            True,
            "second_gold_seen":
            False,
            "first_supporting_titles": [],
            "second_supporting_titles": [],
            "new_document_count":
            0,
            "quality": [
                _document_rank_with_phrase(first, answers),
                len(normalize_text(record["question"]))
            ],
        }
        return True, "accepted", audit

    titles = record.get("supporting_titles")
    facts = record.get("supporting_facts")
    if (category not in ("comparison", "bridge")
            or not isinstance(titles, Sequence)
            or not isinstance(facts, Mapping) or len(titles) != 2):
        return False, "malformed_multihop_metadata", {}
    second_visible, second_visible_hits = _visible_observation_by_hit(
        tokenizer, second)
    first_coverage = _title_coverage(titles, [
        hit
        for hit in first if phrase_visible([str(hit["title"])], first_visible)
    ])
    if len(first_coverage) != 1:
        return False, f"visible_first_support_count_{len(first_coverage)}", {}
    first_title = first_coverage[0]
    second_title = next(title for title in titles if title != first_title)
    if not phrase_visible([second_title], second_visible):
        return False, "visible_second_support_missing", {}
    if not _facts_visible(facts, first_title, first_visible):
        return False, "first_supporting_fact_not_visible", {}
    if _any_fact_visible(facts, second_title, first_visible):
        return False, "second_supporting_fact_visible_in_first", {}
    if not _facts_visible(facts, second_title, second_visible):
        return False, "second_supporting_fact_not_visible", {}
    first_ids = {str(hit["document_id"]) for hit in first}
    new_document_count = sum(
        str(hit["document_id"]) not in first_ids for hit in second)
    if new_document_count == 0:
        return False, "no_visible_new_document", {}
    first_gold = phrase_visible(answers, first_visible)
    second_gold = phrase_visible(answers, second_visible)
    if first_gold or not second_gold:
        return False, "gold_visibility_mismatch", {}
    visible_new_support = any(
        str(hit["document_id"]) not in first_ids
        and normalize_text(hit["title"]) == normalize_text(second_title)
        and phrase_visible([second_title], visible_hit)
        and _facts_visible(facts, second_title, visible_hit)
        and phrase_visible(answers, visible_hit)
        for hit, visible_hit in zip(second, second_visible_hits))
    if not visible_new_support:
        return False, "new_supporting_evidence_not_visible", {}
    if category == "bridge":
        if not phrase_visible([second_title], first_visible):
            return False, "bridge_query_not_visible", {}
    elif not all(
            phrase_visible([title], record["question"]) for title in titles):
        return False, "comparison_entities_not_visible", {}
    audit = {
        "first_gold_seen":
        first_gold,
        "second_gold_seen":
        second_gold,
        "first_supporting_titles": [first_title],
        "second_supporting_titles": [second_title],
        "new_document_count":
        new_document_count,
        "quality": [
            _document_rank_with_phrase(first, [first_title]),
            _document_rank_with_phrase(second, [second_title]),
            _document_rank_with_phrase(second, answers),
            len(normalize_text(record["question"])),
        ],
    }
    return True, "accepted", audit


def _document_catalog(
        hits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "rank":
        int(hit["rank"]),
        "document_id":
        str(hit["document_id"]),
        "title":
        str(hit["title"]),
        "score":
        float(hit["score"]),
        "content_sha256":
        hashlib.sha256(str(hit["contents"]).encode("utf-8")).hexdigest(),
    } for hit in hits]


def _catalog_base(record: Mapping[str, Any],
                  audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": record["source_id"],
        "data_source": record["data_source"],
        "source_split": record["source_split"],
        "source_index": int(record["source_index"]),
        "category": record["category"],
        "type": record.get("type"),
        "level": record.get("level"),
        "question": record["question"],
        "golden_answers": list(record["golden_answers"]),
        "supporting_titles": list(record.get("supporting_titles", [])),
        "supporting_facts": record.get("supporting_facts", {}),
        "first_query": record["first_query"],
        "second_oracle_query": record.get("second_query"),
        "second_query_derivable_from":
        record.get("second_query_derivable_from"),
        "first_documents": _document_catalog(record["first_results"]),
        "second_documents": _document_catalog(record["second_results"]),
        **audit,
    }


def _parquet_questions(path: Path) -> set[str]:
    questions: set[str] = set()
    for record in _read_parquet(path):
        prompt = record.get("prompt")
        if (not isinstance(prompt, Sequence)
                or isinstance(prompt, (str, bytes)) or len(prompt) != 1
                or not isinstance(prompt[0], Mapping)):
            raise ValueError(f"Excluded Parquet has an invalid prompt: {path}")
        content = prompt[0].get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"Excluded Parquet has a non-string prompt: {path}")
        match = re.search(r"Question:\s*(.+?)\s*$", content, re.DOTALL)
        if match is None:
            raise ValueError(
                f"Excluded Parquet prompt has no question: {path}")
        questions.add(normalize_question(match.group(1)))
    return questions


def _read_excluded_questions(
        eval_catalogs: Sequence[Path] = (),
        eval_parquets: Sequence[Path] = (),
) -> tuple[set[str], dict[str, Any]]:
    questions: set[str] = set()
    sources: list[dict[str, Any]] = []
    inputs = [("catalog", Path(path)) for path in eval_catalogs]
    inputs.extend(("parquet", Path(path)) for path in eval_parquets)
    for kind, path in inputs:
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"Evaluation exclusion is missing or symlinked: {path}")
        if kind == "catalog":
            source_questions = {
                normalize_question(record.get("question"))
                for record in read_canonical_jsonl(path)
            }
        else:
            source_questions = _parquet_questions(path)
        questions.update(source_questions)
        sources.append({
            "file": path.name,
            "kind": kind,
            "questions": len(source_questions),
            "sha256": sha256_file(path),
        })
    sources.sort(key=lambda item: (item["kind"], item["file"], item["sha256"]))
    return questions, {
        "sources": sources,
        "normalized_questions": sorted(questions),
    }


def select_catalog(
    evidence: Sequence[Mapping[str, Any]], tokenizer: Any,
    excluded_questions: set[str]
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    accepted: dict[str, list[tuple[Mapping[str, Any], dict[str, Any]]]] = {
        category: []
        for category in QUOTAS
    }
    rejected: Counter[str] = Counter()
    evidence_by_category: Counter[str] = Counter()
    seen_source_ids: set[str] = set()
    for record in evidence:
        source_id = str(record.get("source_id"))
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate evidence source_id: {source_id}")
        seen_source_ids.add(source_id)
        category = str(record.get("category"))
        if category not in accepted:
            raise ValueError(f"Evidence has an unknown category: {category}")
        evidence_by_category[category] += 1
        question_key = normalize_question(record.get("question"))
        if question_key in excluded_questions:
            rejected["materialize:question_in_existing_eval"] += 1
            continue
        valid, reason, audit = evaluate_evidence(record, tokenizer)
        if not valid:
            rejected[f"materialize:{category}:{reason}"] += 1
            continue
        accepted[category].append((record, audit))

    used_questions: set[str] = set(excluded_questions)
    chosen_by_category: dict[str, list[tuple[Mapping[str, Any],
                                             dict[str, Any]]]] = {}
    valid_after_visibility: dict[str, int] = {}
    unique_available: dict[str, int] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    for category, split_quotas in QUOTAS.items():
        values = accepted[category]
        values.sort(key=lambda item: (item[1][
            "quality"], stable_key("quality", str(item[0]["source_id"]))))
        needed = sum(split_quotas.values())
        unique: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
        local_questions: set[str] = set()
        for record, audit in values:
            question_key = normalize_question(record["question"])
            if question_key in used_questions or question_key in local_questions:
                rejected[f"materialize:{category}:duplicate_question"] += 1
                continue
            local_questions.add(question_key)
            unique.append((record, audit))
        valid_after_visibility[category] = len(values)
        unique_available[category] = len(unique)
        if len(unique) < needed:
            shortfalls[category] = {
                "available": len(unique),
                "required": needed,
                "missing": needed - len(unique),
            }
        chosen = unique[:needed]
        chosen_by_category[category] = chosen
        used_questions.update(
            normalize_question(record["question"]) for record, _ in chosen)

    stats = {
        "evidence_by_category": {
            category: evidence_by_category[category]
            for category in QUOTAS
        },
        "valid_after_visibility": valid_after_visibility,
        "unique_available": unique_available,
        "selected_by_category": {
            category: len(chosen_by_category[category])
            for category in QUOTAS
        },
    }
    if shortfalls:
        raise SelectionQuotaError(shortfalls, rejected, stats)

    selected: list[dict[str, Any]] = []
    for category, split_quotas in QUOTAS.items():
        chosen = chosen_by_category[category]
        chosen.sort(
            key=lambda item: stable_key("split", str(item[0]["source_id"])))
        train_count = split_quotas["train"]
        assignments = (["train"] * train_count + ["val"] * split_quotas["val"])
        for (record, audit), split in zip(chosen, assignments):
            catalog = _catalog_base(record, audit)
            catalog["output_split"] = split
            catalog["sample_id"] = record["source_id"]
            selected.append(catalog)
    return selected, rejected, stats


def make_record(catalog: Mapping[str, Any],
                tool_protocol: str = LEGACY_XML,
                force_search: bool = False) -> dict[str, Any]:
    return {
        "data_source":
        catalog["data_source"],
        "prompt": make_prompt(catalog["question"], tool_protocol,
                              force_search),
        "ability":
        "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": list(catalog["golden_answers"])
            },
        },
        "extra_info": {
            "split": catalog["source_split"],
            "index": int(catalog["source_index"]),
        },
    }


def _ordered_records(catalog: Sequence[Mapping[str, Any]],
                     split: str,
                     tool_protocol: str = LEGACY_XML,
                     limit: Optional[int] = None,
                     force_search: bool = False) -> list[dict[str, Any]]:
    if split == "probe":
        selected = [
            record for record in catalog
            if record["output_split"] == "val" and record["category"] in (
                "comparison", "bridge")
        ]
    else:
        selected = [
            record for record in catalog if record["output_split"] == split
        ]
    selected.sort(
        key=lambda item: stable_key(f"output:{split}", str(item["source_id"])))
    if limit is not None:
        if type(limit) is not int or limit <= 0 or limit > len(selected):
            raise ValueError("output limit must select a non-empty prefix")
        selected = selected[:limit]
    return [
        make_record(record, tool_protocol, force_search) for record in selected
    ]


def _atomic_write_parquet(records: Sequence[Mapping[str, Any]],
                          path: Path) -> None:
    import datasets

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                             suffix=".tmp",
                                             dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        datasets.Dataset.from_list(list(records)).to_parquet(
            str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_retrieval_contract(
        local_dir: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    evidence_path = local_dir / EVIDENCE_FILE
    ledger_path = local_dir / RETRIEVAL_LEDGER_FILE
    if not evidence_path.is_file() or not ledger_path.is_file():
        raise ValueError("Retrieval evidence and ledger must exist")
    _verify_sidecar(evidence_path)
    _verify_sidecar(ledger_path)
    raw = ledger_path.read_bytes()
    ledger = json.loads(raw)
    if not isinstance(ledger, Mapping) or canonical_json_bytes(ledger) != raw:
        raise ValueError("Retrieval ledger is not canonical JSON")
    expected_keys = {
        "schema_version", "selection_policy", "seed", "topk",
        "candidate_limits", "retrieval_targets", "candidate_order_sha256",
        "candidate_queries", "evidence_rows", "evidence_sha256",
        "rejection_counts", "bm25_revision", "corpus_revision", "corpus_sha256"
    }
    if set(ledger) != expected_keys:
        raise ValueError("Retrieval ledger has missing or unknown fields")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "seed": SEED,
        "topk": TOPK,
        "candidate_limits": CANDIDATE_LIMITS,
        "retrieval_targets": RETRIEVAL_TARGETS,
        "bm25_revision": BM25_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "corpus_sha256": CORPUS_SHA256,
    }
    if any(ledger.get(key) != value for key, value in fixed.items()):
        raise ValueError("Retrieval ledger contract mismatch")
    evidence = read_canonical_jsonl(evidence_path)
    counts = ledger.get("rejection_counts")
    if (not isinstance(counts, Mapping) or not all(
            isinstance(key, str) and type(value) is int and value >= 0
            for key, value in counts.items())
            or type(ledger.get("candidate_queries")) is not int
            or ledger["candidate_queries"] < len(evidence)
            or ledger.get("evidence_rows") != len(evidence)
            or ledger.get("evidence_sha256") != sha256_file(evidence_path)
            or not isinstance(ledger.get("candidate_order_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}",
                            ledger["candidate_order_sha256"]) is None):
        raise ValueError("Retrieval ledger values are invalid")
    validate_evidence_sources(local_dir, evidence)
    return ledger, evidence


def _load_selection_funnel(local_dir: Path, ledger: Mapping[str, Any],
                           expected_status: str) -> Mapping[str, Any]:
    path = local_dir / SELECTION_FUNNEL_FILE
    if not path.is_file() or path.is_symlink():
        raise ValueError("Selection funnel must be a regular file")
    _verify_sidecar(path)
    raw = path.read_bytes()
    funnel = json.loads(raw)
    expected_keys = {
        "schema_version", "selection_policy", "seed", "status", "stage",
        "quotas", "candidate_limits", "retrieval_targets", "retrieval",
        "materialize", "failure"
    }
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "seed": SEED,
        "quotas": QUOTAS,
        "candidate_limits": CANDIDATE_LIMITS,
        "retrieval_targets": RETRIEVAL_TARGETS,
    }
    if (not isinstance(funnel, Mapping) or set(funnel) != expected_keys
            or canonical_json_bytes(funnel) != raw
            or funnel.get("status") != expected_status
            or any(funnel.get(key) != value for key, value in fixed.items())):
        raise ValueError("Selection funnel contract mismatch")
    if expected_status == "retrieval_complete":
        if (funnel.get("stage") != "retrieve"
                or funnel.get("materialize") is not None
                or funnel.get("failure") is not None):
            raise ValueError("Retrieval selection funnel is not complete")
    elif expected_status == "complete":
        if (funnel.get("stage") != "materialize"
                or not isinstance(funnel.get("materialize"), Mapping)
                or funnel.get("failure") is not None):
            raise ValueError("Materialized selection funnel is not complete")
    else:
        raise ValueError(
            f"Unsupported selection funnel status: {expected_status}")

    retrieval = funnel.get("retrieval")
    retrieval_keys = {
        "source_rows", "candidates_after_prescreen", "candidates_after_cap",
        "candidates_truncated_by_cap", "candidate_items_queried",
        "bm25_query_calls", "structurally_valid", "evidence_rows",
        "candidate_order_sha256", "evidence_sha256", "rejection_counts"
    }
    if not isinstance(retrieval, Mapping) or set(retrieval) != retrieval_keys:
        raise ValueError("Selection funnel retrieval schema mismatch")
    category_fields = (
        "candidates_after_prescreen",
        "candidates_after_cap",
        "candidates_truncated_by_cap",
        "structurally_valid",
    )
    if any(not isinstance(retrieval.get(field), Mapping)
           or set(retrieval[field]) != set(CANDIDATE_LIMITS) or not all(
               type(value) is int and value >= 0
               for value in retrieval[field].values())
           for field in category_fields):
        raise ValueError("Selection funnel category counts are invalid")
    for category, limit in CANDIDATE_LIMITS.items():
        prescreen = retrieval["candidates_after_prescreen"][category]
        capped = retrieval["candidates_after_cap"][category]
        truncated = retrieval["candidates_truncated_by_cap"][category]
        if capped > limit or prescreen - capped != truncated:
            raise ValueError(
                "Selection funnel candidate cap counts are invalid")
    source_rows = retrieval.get("source_rows")
    counts = retrieval.get("rejection_counts")
    if (not isinstance(source_rows, Mapping)
            or set(source_rows) != {"nq", "hotpotqa"} or not all(
                type(value) is int and value >= 0
                for value in source_rows.values())
            or not isinstance(counts, Mapping) or not all(
                isinstance(key, str) and type(value) is int and value >= 0
                for key, value in counts.items())):
        raise ValueError("Selection funnel retrieval counts are invalid")
    candidate_items = retrieval.get("candidate_items_queried")
    query_calls = retrieval.get("bm25_query_calls")
    if (type(candidate_items) is not int or type(query_calls) is not int
            or query_calls < candidate_items
            or candidate_items != ledger["candidate_queries"]
            or retrieval.get("evidence_rows") != ledger["evidence_rows"]
            or retrieval.get("evidence_sha256") != ledger["evidence_sha256"]
            or retrieval.get("candidate_order_sha256")
            != ledger["candidate_order_sha256"]
            or retrieval.get("rejection_counts") != ledger["rejection_counts"]
            or sum(retrieval["structurally_valid"].values())
            != ledger["evidence_rows"]):
        raise ValueError("Selection funnel disagrees with retrieval ledger")
    if any(retrieval["structurally_valid"][category] < required
           for category, required in RETRIEVAL_TARGETS.items()):
        raise ValueError("Selection funnel does not meet retrieval targets")
    return funnel


def _validate_derived_from(value: object) -> dict[str, str]:
    expected_keys = {
        "source_manifest_sha256",
        "source_catalog_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("Native manifest lineage contract mismatch")
    lineage: dict[str, str] = {}
    for key in sorted(expected_keys):
        digest = value[key]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}",
                                                       digest) is None:
            raise ValueError("Native manifest lineage digest is invalid")
        lineage[key] = digest
    return lineage


def materialize(
    local_dir: Path,
    model_dir: Path,
    eval_catalogs: Sequence[Path] = (),
    eval_parquets: Sequence[Path] = (),
    tool_protocol: str = LEGACY_XML,
    derived_from: Optional[Mapping[str, str]] = None,
) -> Path:
    protocol = normalize_tool_protocol(tool_protocol)
    if protocol == QWEN35_NATIVE:
        if derived_from is None:
            raise ValueError(
                "Native data must be created with materialize-native")
        native_lineage = _validate_derived_from(derived_from)
    else:
        if derived_from is not None:
            raise ValueError("Legacy data must not define native lineage")
        native_lineage = None

    from transformers import AutoTokenizer

    local_dir = local_dir.resolve()
    replay_path = local_dir / REPLAY_FILE
    for stale in (replay_path,
                  replay_path.with_suffix(replay_path.suffix + ".sha256")):
        stale.unlink(missing_ok=True)
    for source in SOURCE_SPECS:
        verify_source(local_dir, source)
    ledger_path = local_dir / RETRIEVAL_LEDGER_FILE
    evidence_path = local_dir / EVIDENCE_FILE
    ledger, evidence = _load_retrieval_contract(local_dir)
    retrieval_funnel = _load_selection_funnel(local_dir, ledger,
                                              "retrieval_complete")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    excluded_questions, exclusions = _read_excluded_questions(
        eval_catalogs, eval_parquets)
    try:
        catalog, rejection_counts, selection_stats = select_catalog(
            evidence, tokenizer, excluded_questions)
    except SelectionQuotaError as error:
        materialize_summary = {
            **error.stats,
            "evidence_rows": len(evidence),
            "excluded_question_count": len(excluded_questions),
            "rejection_counts": dict(sorted(error.rejection_counts.items())),
        }
        _write_selection_funnel(
            local_dir,
            _selection_funnel_payload("failed",
                                      "materialize",
                                      retrieval_funnel["retrieval"],
                                      materialize=materialize_summary,
                                      failure={
                                          "stage": "materialize",
                                          "shortfalls": error.shortfalls,
                                      }))
        raise
    catalog.sort(
        key=lambda item: stable_key("catalog", str(item["source_id"])))

    exclusions_path = local_dir / EXCLUSIONS_FILE
    atomic_write(exclusions_path, canonical_json_bytes(exclusions))
    catalog_path = local_dir / CATALOG_FILE
    atomic_write(catalog_path,
                 b"".join(canonical_json_bytes(record) for record in catalog))
    artifacts: dict[str, dict[str, Any]] = {}
    output_records: dict[str, list[dict[str, Any]]] = {}
    for split, filename in OUTPUT_FILES.items():
        records = _ordered_records(catalog, split, protocol)
        output_records[split] = records
        path = local_dir / filename
        _atomic_write_parquet(records, path)
        artifacts[split] = {
            "file":
            filename,
            "rows":
            len(records),
            "sha256":
            sha256_file(path),
            "sample_ids": [
                f"{record['data_source']}:{record['extra_info']['split']}:{record['extra_info']['index']}"
                for record in records
            ],
        }
    if protocol == QWEN35_NATIVE:
        for label, (filename, rows, force_search) in NATIVE_PROBE_FILES.items():
            records = _ordered_records(catalog,
                                       "probe",
                                       protocol,
                                       limit=rows,
                                       force_search=force_search)
            output_records[label] = records
            path = local_dir / filename
            _atomic_write_parquet(records, path)
            artifacts[label] = {
                "file": filename,
                "rows": len(records),
                "sha256": sha256_file(path),
                "sample_ids": [
                    f"{record['data_source']}:{record['extra_info']['split']}:{record['extra_info']['index']}"
                    for record in records
                ],
            }
    for label, path in (("catalog", catalog_path), ("exclusions",
                                                    exclusions_path),
                        ("retrieval_evidence",
                         evidence_path), ("retrieval_ledger", ledger_path)):
        artifacts[label] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path)
        }

    train_questions = {
        normalize_question(record["question"])
        for record in catalog if record["output_split"] == "train"
    }
    val_questions = {
        normalize_question(record["question"])
        for record in catalog if record["output_split"] == "val"
    }
    if (train_questions & val_questions or train_questions & excluded_questions
            or val_questions & excluded_questions):
        raise ValueError(
            "Selected train/val questions overlap held-out questions")
    materialize_summary = {
        **selection_stats,
        "evidence_rows": len(evidence),
        "excluded_question_count": len(excluded_questions),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    selection_funnel_path = _write_selection_funnel(
        local_dir,
        _selection_funnel_payload("complete",
                                  "materialize",
                                  retrieval_funnel["retrieval"],
                                  materialize=materialize_summary))
    artifacts["selection_funnel"] = {
        "file": selection_funnel_path.name,
        "bytes": selection_funnel_path.stat().st_size,
        "sha256": sha256_file(selection_funnel_path),
    }
    manifest = {
        "schema_version": (MATERIALIZED_SCHEMA_VERSION
                           if protocol == QWEN35_NATIVE else SCHEMA_VERSION),
        "selection_policy": SELECTION_POLICY,
        "seed": SEED,
        "sources": {
            "dataset": DATASET_NAME,
            "revision": DATASET_REVISION,
            "files": SOURCE_SPECS,
        },
        "retrieval": {
            "topk": TOPK,
            "bm25_revision": BM25_REVISION,
            "corpus_revision": CORPUS_REVISION,
            "corpus_sha256": CORPUS_SHA256,
            "ledger_sha256": sha256_file(ledger_path),
        },
        "tokenizer": {
            "revision": MODEL_REVISION,
            "max_obs_length": MAX_OBS_LENGTH
        },
        "quotas": QUOTAS,
        "materialize_rejection_counts": dict(sorted(rejection_counts.items())),
        "overlap_checks": {
            "passed": True,
            "train_val_questions": 0,
            "train_existing_eval_questions": 0,
            "val_existing_eval_questions": 0,
            "probe_is_hotpot_val_subset": True,
        },
        "artifacts": artifacts,
    }
    if protocol == QWEN35_NATIVE:
        manifest["prompt_contract"] = prompt_contract(protocol)
        manifest["derived_from"] = native_lineage
    manifest_path = local_dir / MANIFEST_FILE
    atomic_write(manifest_path, canonical_json_bytes(manifest))
    write_digest_sidecar(manifest_path)
    return manifest_path


def _copy_regular_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Reusable artifact is missing or symlinked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def materialize_native(
    source_manifest: Path,
    output_dir: Path,
    model_dir: Path,
    eval_catalogs: Sequence[Path] = (),
    eval_parquets: Sequence[Path] = (),
    reselect_catalog: bool = True,
) -> Path:
    """Re-materialize native prompts from a verified, sealed catalog."""
    source_manifest = source_manifest.resolve()
    source_dir = source_manifest.parent
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite native data directory: {output_dir}")
    try:
        output_dir.relative_to(source_dir)
    except ValueError:
        pass
    else:
        raise ValueError("native output directory must not be inside source data")

    _, source_payload, _ = _verify_manifest_artifacts(
        source_manifest, expected_tool_protocol=LEGACY_XML)
    verify_replay_receipt(source_manifest)
    source_catalog_path = source_dir / CATALOG_FILE
    source_catalog = source_catalog_path.read_bytes()
    catalog = read_canonical_jsonl(source_catalog_path)
    derived_from = {
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_catalog_sha256": sha256_file(source_catalog_path),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for source in SOURCE_SPECS:
            _copy_regular_file(verify_source(source_dir, source),
                               source_path(staging, source))
        artifacts: dict[str, dict[str, Any]] = {}
        bound_labels = (
            "catalog",
            "exclusions",
            "retrieval_evidence",
            "retrieval_ledger",
            "selection_funnel",
        )
        sidecar_labels = {
            "retrieval_evidence",
            "retrieval_ledger",
            "selection_funnel",
        }
        for label in bound_labels:
            artifact = source_payload["artifacts"][label]
            source_path_value = source_dir / artifact["file"]
            target_path = staging / artifact["file"]
            _copy_regular_file(source_path_value, target_path)
            artifacts[label] = dict(artifact)
            if label not in sidecar_labels:
                continue
            _copy_regular_file(
                source_path_value.with_suffix(source_path_value.suffix +
                                              ".sha256"),
                target_path.with_suffix(target_path.suffix + ".sha256"))

        output_records: dict[str, list[dict[str, Any]]] = {}
        for split, filename in OUTPUT_FILES.items():
            records = _ordered_records(catalog, split, QWEN35_NATIVE)
            output_records[split] = records
            path = staging / filename
            _atomic_write_parquet(records, path)
            artifacts[split] = {
                "file": filename,
                "rows": len(records),
                "sha256": sha256_file(path),
                "sample_ids": [
                    f"{record['data_source']}:{record['extra_info']['split']}:{record['extra_info']['index']}"
                    for record in records
                ],
            }
        for label, (filename, rows, force_search) in NATIVE_PROBE_FILES.items():
            records = _ordered_records(catalog,
                                       "probe",
                                       QWEN35_NATIVE,
                                       limit=rows,
                                       force_search=force_search)
            output_records[label] = records
            path = staging / filename
            _atomic_write_parquet(records, path)
            artifacts[label] = {
                "file": filename,
                "rows": len(records),
                "sha256": sha256_file(path),
                "sample_ids": [
                    f"{record['data_source']}:{record['extra_info']['split']}:{record['extra_info']['index']}"
                    for record in records
                ],
            }

        manifest = {
            "schema_version": MATERIALIZED_SCHEMA_VERSION,
            "selection_policy": source_payload["selection_policy"],
            "seed": source_payload["seed"],
            "sources": source_payload["sources"],
            "retrieval": source_payload["retrieval"],
            "tokenizer": source_payload["tokenizer"],
            "quotas": source_payload["quotas"],
            "materialize_rejection_counts":
            source_payload["materialize_rejection_counts"],
            "overlap_checks": source_payload["overlap_checks"],
            "artifacts": artifacts,
            "prompt_contract": prompt_contract(QWEN35_NATIVE),
            "derived_from": derived_from,
        }
        staged_manifest = staging / MANIFEST_FILE
        atomic_write(staged_manifest, canonical_json_bytes(manifest))
        write_digest_sidecar(staged_manifest)

        if (staging / CATALOG_FILE).read_bytes() != source_catalog:
            raise ValueError(
                "Native materialization changed the selected catalog")
        for split in OUTPUT_FILES:
            source_rows = _read_parquet(source_dir / OUTPUT_FILES[split])
            source_non_prompt = [{
                key: value
                for key, value in row.items() if key != "prompt"
            } for row in source_rows]
            target_non_prompt = [{
                key: value
                for key, value in row.items() if key != "prompt"
            } for row in output_records[split]]
            if source_non_prompt != target_non_prompt:
                raise ValueError(
                    f"Native materialization changed non-prompt {split} fields"
                )
            if (artifacts[split]["sample_ids"] !=
                    source_payload["artifacts"][split]["sample_ids"]):
                raise ValueError(
                    f"Native materialization changed {split} sample IDs")
        verify_manifest(staged_manifest,
                        model_dir,
                        eval_catalogs,
                        eval_parquets,
                        expected_tool_protocol=QWEN35_NATIVE,
                        source_manifest=source_manifest,
                        reselect_catalog=reselect_catalog)
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir / MANIFEST_FILE


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = f"{sha256_file(path)}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.is_symlink() or sidecar.read_bytes(
    ) != expected:
        raise ValueError(f"Digest sidecar mismatch: {path}")


def _read_parquet(path: Path) -> list[Mapping[str, Any]]:
    import datasets

    dataset = datasets.Dataset.from_parquet(str(path))
    return [dataset[index] for index in range(len(dataset))]


def _flat_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (isinstance(value, list) and len(value) == 1
            and isinstance(value[0], list)):
        value = value[0]
    if not isinstance(value, list) or not all(
            type(token_id) is int for token_id in value):
        raise ValueError("tokenizer returned invalid input_ids")
    return value


def _validate_native_prompt(tokenizer: Any,
                            messages: Sequence[Mapping[str, Any]],
                            max_start_length: int = 1024) -> int:
    canonical = validate_qwen35_messages(messages)
    rendered = render_qwen35_prompt(tokenizer, canonical)
    encoded = tokenizer(rendered, add_special_tokens=False)
    rendered_ids = _flat_token_ids(encoded["input_ids"])
    direct_ids = _flat_token_ids(
        tokenizer.apply_chat_template(canonical,
                                      tools=qwen35_tools(),
                                      enable_thinking=False,
                                      add_generation_prompt=True,
                                      tokenize=True,
                                      return_dict=False))
    if rendered_ids != direct_ids:
        raise ValueError(
            "rendered-string and direct-template prompt tokens differ")
    if not rendered_ids:
        raise ValueError("native prompt tokenization is empty")
    if len(rendered_ids) > max_start_length:
        raise ValueError(
            f"native prompt has {len(rendered_ids)} tokens; maximum is "
            f"{max_start_length}")
    return len(rendered_ids)


def _manifest_tool_protocol(manifest: Mapping[str, Any]) -> str:
    schema_version = manifest.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        expected_keys = {
            "schema_version", "selection_policy", "seed", "sources",
            "retrieval", "tokenizer", "quotas",
            "materialize_rejection_counts", "overlap_checks", "artifacts"
        }
        if set(manifest) != expected_keys:
            raise ValueError("Legacy manifest contract mismatch")
        return LEGACY_XML
    if schema_version == MATERIALIZED_SCHEMA_VERSION:
        expected_keys = {
            "schema_version", "selection_policy", "seed", "sources",
            "retrieval", "tokenizer", "quotas",
            "materialize_rejection_counts", "overlap_checks", "artifacts",
            "prompt_contract", "derived_from"
        }
        if set(manifest) != expected_keys:
            raise ValueError("Native manifest contract mismatch")
        if manifest.get("prompt_contract") != prompt_contract(QWEN35_NATIVE):
            raise ValueError("Native manifest prompt contract mismatch")
        _validate_derived_from(manifest.get("derived_from"))
        return QWEN35_NATIVE
    raise ValueError("Manifest schema_version mismatch")


def _verify_manifest_artifacts(
    manifest_path: Path,
    expected_tool_protocol: Optional[str] = None,
) -> tuple[Path, Mapping[str, Any], str]:
    """Verify the manifest contract and byte identities without reselection."""
    manifest_path = manifest_path.resolve()
    if manifest_path.name != MANIFEST_FILE or not manifest_path.is_file(
    ) or manifest_path.is_symlink():
        raise ValueError(f"Manifest must be a regular {MANIFEST_FILE}")
    _verify_sidecar(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if canonical_json_bytes(manifest) != raw:
        raise ValueError("Manifest is not canonical JSON")
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be an object")
    protocol = _manifest_tool_protocol(manifest)
    if expected_tool_protocol is not None and protocol != normalize_tool_protocol(
            expected_tool_protocol):
        raise ValueError("Manifest tool protocol mismatch")
    if (manifest.get("selection_policy") != SELECTION_POLICY
            or manifest.get("seed") != SEED
            or manifest.get("quotas") != QUOTAS):
        raise ValueError("Manifest contract mismatch")
    local_dir = manifest_path.parent
    for source in SOURCE_SPECS:
        verify_source(local_dir, source)
    if manifest.get("sources") != {
            "dataset": DATASET_NAME,
            "revision": DATASET_REVISION,
            "files": SOURCE_SPECS,
    }:
        raise ValueError("Manifest source contract mismatch")
    expected_retrieval = {
        "topk": TOPK,
        "bm25_revision": BM25_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "corpus_sha256": CORPUS_SHA256,
        "ledger_sha256": sha256_file(local_dir / RETRIEVAL_LEDGER_FILE),
    }
    if manifest.get("retrieval") != expected_retrieval:
        raise ValueError("Manifest retrieval contract mismatch")
    if manifest.get("tokenizer") != {
            "revision": MODEL_REVISION,
            "max_obs_length": MAX_OBS_LENGTH,
    }:
        raise ValueError("Manifest tokenizer contract mismatch")
    if manifest.get("overlap_checks") != {
            "passed": True,
            "train_val_questions": 0,
            "train_existing_eval_questions": 0,
            "val_existing_eval_questions": 0,
            "probe_is_hotpot_val_subset": True,
    }:
        raise ValueError("Manifest overlap contract mismatch")
    artifacts = manifest.get("artifacts")
    expected_artifact_names = {
        "train": OUTPUT_FILES["train"],
        "val": OUTPUT_FILES["val"],
        "probe": OUTPUT_FILES["probe"],
        "catalog": CATALOG_FILE,
        "exclusions": EXCLUSIONS_FILE,
        "retrieval_evidence": EVIDENCE_FILE,
        "retrieval_ledger": RETRIEVAL_LEDGER_FILE,
        "selection_funnel": SELECTION_FUNNEL_FILE,
    }
    if protocol == QWEN35_NATIVE:
        expected_artifact_names.update({
            label: values[0]
            for label, values in NATIVE_PROBE_FILES.items()
        })
    parquet_labels = set(OUTPUT_FILES)
    if protocol == QWEN35_NATIVE:
        parquet_labels.update(NATIVE_PROBE_FILES)
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != set(expected_artifact_names)):
        raise ValueError("Manifest artifacts must be an object")
    for label, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError("Manifest artifact entry must be an object")
        expected_keys = ({"file", "rows", "sha256", "sample_ids"} if label
                         in parquet_labels else {"file", "bytes", "sha256"})
        if set(artifact) != expected_keys or artifact.get(
                "file") != expected_artifact_names[label]:
            raise ValueError(f"Manifest artifact schema mismatch: {label}")
        relative = Path(str(artifact["file"]))
        if relative.is_absolute() or relative.name != str(artifact["file"]):
            raise ValueError(f"Manifest artifact path is invalid: {label}")
        path = local_dir / relative
        if not path.is_file() or path.is_symlink() or sha256_file(
                path) != artifact["sha256"]:
            raise ValueError(f"Artifact identity mismatch: {path}")
        if "bytes" in artifact and path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Artifact byte count mismatch: {path}")
    return manifest_path, manifest, protocol


def verify_manifest(
        manifest_path: Path,
        model_dir: Path,
        eval_catalogs: Sequence[Path] = (),
        eval_parquets: Sequence[Path] = (),
        expected_tool_protocol: Optional[str] = None,
        source_manifest: Optional[Path] = None,
        reselect_catalog: bool = True,
) -> Mapping[str, Any]:
    manifest_path, manifest, protocol = _verify_manifest_artifacts(
        manifest_path, expected_tool_protocol)
    local_dir = manifest_path.parent
    artifacts = manifest["artifacts"]
    source_payload: Optional[Mapping[str, Any]] = None
    source_catalog_sha256: Optional[str] = None
    if protocol == QWEN35_NATIVE:
        if source_manifest is None:
            raise ValueError(
                "Native manifest verification requires --source-manifest")
        source_manifest = Path(source_manifest).resolve()
        source_payload = verify_manifest(
            source_manifest,
            model_dir,
            eval_catalogs,
            eval_parquets,
            expected_tool_protocol=LEGACY_XML,
            reselect_catalog=reselect_catalog)
        source_catalog_sha256 = sha256_file(source_manifest.parent /
                                            CATALOG_FILE)
        expected_lineage = {
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_catalog_sha256": source_catalog_sha256,
        }
        if manifest["derived_from"] != expected_lineage:
            raise ValueError("Native manifest lineage does not match source")
        if reselect_catalog:
            verify_replay_receipt(source_manifest)
    elif not reselect_catalog:
        # Skipping selection never skips the sealed retrieval provenance.
        verify_replay_receipt(manifest_path)
    if protocol == QWEN35_NATIVE:
        if sha256_file(local_dir / CATALOG_FILE) != source_catalog_sha256:
            raise ValueError("Native catalog does not match source catalog")
        assert source_payload is not None
        for label in (
                "catalog",
                "retrieval_evidence",
                "retrieval_ledger",
                "selection_funnel",
                "exclusions",
        ):
            if artifacts[label] != source_payload["artifacts"][label]:
                raise ValueError(
                    f"Native artifact identity does not match source: {label}"
                )
        if (manifest["seed"] != source_payload["seed"]
                or manifest["quotas"] != source_payload["quotas"]):
            raise ValueError("Native seed or quotas do not match source")
        if (manifest["materialize_rejection_counts"] !=
                source_payload["materialize_rejection_counts"]):
            raise ValueError(
                "Native rejection counts do not match source")
        for split in OUTPUT_FILES:
            if (artifacts[split]["sample_ids"] !=
                    source_payload["artifacts"][split]["sample_ids"]):
                raise ValueError(
                    f"Native {split} sample IDs do not match source")
        # Source verification already proved these byte-identical artifacts.
        # Check the copied sidecars without loading the evidence pool again.
        for filename in (EVIDENCE_FILE, RETRIEVAL_LEDGER_FILE,
                         SELECTION_FUNNEL_FILE):
            _verify_sidecar(local_dir / filename)
    elif reselect_catalog:
        ledger, evidence = _load_retrieval_contract(local_dir)
        funnel = _load_selection_funnel(local_dir, ledger, "complete")

    exclusions_path = local_dir / EXCLUSIONS_FILE
    exclusions_raw = exclusions_path.read_bytes()
    exclusions = json.loads(exclusions_raw)
    if (not isinstance(exclusions, Mapping)
            or set(exclusions) != {"sources", "normalized_questions"}
            or canonical_json_bytes(exclusions) != exclusions_raw
            or not isinstance(exclusions["sources"], list)
            or not isinstance(exclusions["normalized_questions"], list)):
        raise ValueError("Exclusions artifact contract mismatch")
    excluded_list = exclusions["normalized_questions"]
    if (excluded_list != sorted(set(excluded_list)) or not all(
            isinstance(question, str) and question
            for question in excluded_list)):
        raise ValueError("Exclusions contain invalid normalized questions")
    if eval_catalogs or eval_parquets:
        _, expected_exclusions = _read_excluded_questions(
            eval_catalogs, eval_parquets)
        if exclusions != expected_exclusions:
            raise ValueError("Exclusions do not match the evaluation sources")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    catalog = read_canonical_jsonl(local_dir / CATALOG_FILE)
    if protocol == LEGACY_XML and reselect_catalog:
        recomputed, rejection_counts, selection_stats = select_catalog(
            evidence, tokenizer, set(excluded_list))
        recomputed.sort(
            key=lambda item: stable_key("catalog", str(item["source_id"])))
        if catalog != recomputed:
            raise ValueError("Catalog does not match deterministic reselection")
        if manifest.get("materialize_rejection_counts") != dict(
                sorted(rejection_counts.items())):
            raise ValueError(
                "Manifest rejection counts do not match reselection")
        expected_materialize = {
            **selection_stats,
            "evidence_rows": len(evidence),
            "excluded_question_count": len(excluded_list),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        }
        if funnel.get("materialize") != expected_materialize:
            raise ValueError(
                "Selection funnel does not match deterministic reselection")
    counts = Counter(
        (record["category"], record["output_split"]) for record in catalog)
    for category, split_quotas in QUOTAS.items():
        for split, expected in split_quotas.items():
            if counts[(category, split)] != expected:
                raise ValueError(
                    f"Catalog quota mismatch for {category}/{split}")
    output_specs = [(split, filename, split, None, False)
                    for split, filename in OUTPUT_FILES.items()]
    if protocol == QWEN35_NATIVE:
        output_specs.extend((label, filename, "probe", rows, force_search)
                            for label, (filename, rows,
                                        force_search) in NATIVE_PROBE_FILES.items())
    for label, filename, split, limit, force_search in output_specs:
        expected_rows = _ordered_records(catalog,
                                         split,
                                         protocol,
                                         limit=limit,
                                         force_search=force_search)
        actual_rows = _read_parquet(local_dir / filename)
        if actual_rows != expected_rows:
            raise ValueError(f"Parquet rows do not match catalog: {filename}")
        if protocol == QWEN35_NATIVE:
            for row_index, row in enumerate(actual_rows):
                sample_id = (row.get("extra_info", {}) or {}).get(
                    "sample_id", f"row-{row_index}")
                try:
                    _validate_native_prompt(tokenizer, row["prompt"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Native prompt validation failed for {filename} "
                        f"row {row_index} ({sample_id}): {error}") from error
        artifact = artifacts[label]
        expected_ids = [
            f"{record['data_source']}:{record['extra_info']['split']}:{record['extra_info']['index']}"
            for record in expected_rows
        ]
        if (artifact["rows"] != len(expected_rows)
                or artifact["sample_ids"] != expected_ids):
            raise ValueError(f"Parquet manifest metadata mismatch: {filename}")
        if set(actual_rows[0]) != {
                "data_source", "prompt", "ability", "reward_model",
                "extra_info"
        }:
            raise ValueError(f"Parquet exposes unexpected fields: {filename}")

    questions = [normalize_question(record["question"]) for record in catalog]
    if len(questions) != len(set(questions)):
        raise ValueError("Catalog contains duplicate normalized questions")
    selected_questions = set(questions)
    if selected_questions & set(excluded_list):
        raise ValueError("Catalog overlaps evaluation exclusions")
    return manifest


def verify_replay_receipt(manifest_path: Path) -> Mapping[str, Any]:
    manifest_path = Path(manifest_path)
    if (manifest_path.name != MANIFEST_FILE or not manifest_path.is_file()
            or manifest_path.is_symlink()):
        raise ValueError(f"Replay receipt requires a regular {MANIFEST_FILE}")
    manifest_path = manifest_path.resolve()
    _verify_sidecar(manifest_path)
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if (canonical_json_bytes(manifest) != manifest_raw
            or _manifest_tool_protocol(manifest) != LEGACY_XML
            or manifest.get("selection_policy") != SELECTION_POLICY):
        raise ValueError("Replay receipt manifest contract mismatch")

    local_dir = manifest_path.parent
    catalog_path = local_dir / CATALOG_FILE
    evidence_path = local_dir / EVIDENCE_FILE
    for label, path in (("catalog", catalog_path),
                        ("retrieval_evidence", evidence_path)):
        artifact = manifest.get("artifacts", {}).get(label, {})
        if (not path.is_file() or path.is_symlink()
                or artifact.get("file") != path.name
                or artifact.get("sha256") != sha256_file(path)):
            raise ValueError(f"Replay receipt {label} is not bound by the manifest")

    catalog = read_canonical_jsonl(catalog_path)
    selected_rows = sum(sum(values.values()) for values in QUOTAS.values())
    if len(catalog) != selected_rows:
        raise ValueError("Replay receipt catalog row count mismatch")
    source_ids: list[str] = []
    query_count = 0
    for record in catalog:
        source_id = record.get("source_id")
        category = record.get("category")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Replay receipt catalog source_id is invalid")
        if category not in QUOTAS:
            raise ValueError("Replay receipt catalog category is invalid")
        source_ids.append(source_id)
        query_count += 1 if category == "single" else 2
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Replay receipt catalog source_ids are not unique")
    selected_order_sha256 = hashlib.sha256("".join(
        f"{source_id}\n" for source_id in source_ids).encode("utf-8")).hexdigest()
    expected = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "manifest_sha256": sha256_file(manifest_path),
        "catalog_sha256": sha256_file(catalog_path),
        "evidence_sha256": sha256_file(evidence_path),
        "bm25_revision": BM25_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "corpus_sha256": CORPUS_SHA256,
        "topk": TOPK,
        "selected_rows": selected_rows,
        "query_count": query_count,
        "selected_order_sha256": selected_order_sha256,
    }
    receipt_path = local_dir / REPLAY_FILE
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("Replay receipt is missing or symlinked")
    _verify_sidecar(receipt_path)
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    if canonical_json_bytes(receipt) != receipt_raw or receipt != expected:
        raise ValueError("Replay receipt contract mismatch")
    return receipt


def replay_selected_retrieval(manifest_path: Path, index_path: Path,
                              corpus_path: Path, offsets_path: Path) -> Path:
    from search_r1.search.bm25_server import BM25Retriever

    manifest_path = manifest_path.resolve()
    if manifest_path.name != MANIFEST_FILE:
        raise ValueError(f"Replay requires {MANIFEST_FILE}")
    _verify_sidecar(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (canonical_json_bytes(manifest) != raw
            or manifest.get("selection_policy") != SELECTION_POLICY):
        raise ValueError("Replay manifest contract mismatch")
    local_dir = manifest_path.parent
    _, evidence = _load_retrieval_contract(local_dir)
    catalog_path = local_dir / CATALOG_FILE
    catalog = read_canonical_jsonl(catalog_path)
    catalog_artifact = manifest.get("artifacts", {}).get("catalog", {})
    if (catalog_artifact.get("file") != CATALOG_FILE
            or catalog_artifact.get("sha256") != sha256_file(catalog_path)):
        raise ValueError("Replay catalog is not bound by the manifest")
    selected_ids = [str(record.get("source_id")) for record in catalog]
    if len(selected_ids) != sum(
            sum(values.values()) for values in QUOTAS.values()):
        raise ValueError("Replay catalog row count mismatch")
    evidence_by_id = {str(record["source_id"]): record for record in evidence}
    if any(source_id not in evidence_by_id for source_id in selected_ids):
        raise ValueError("Replay catalog contains an unknown evidence source")

    retriever = BM25Retriever(str(index_path),
                              topk=TOPK,
                              corpus_path=str(corpus_path),
                              offsets_path=str(offsets_path))
    query_count = 0
    for source_id in selected_ids:
        record = evidence_by_id[source_id]
        first = _serialize_hits(
            retriever.search(str(record["first_query"]),
                             topk=TOPK,
                             return_scores=True))
        query_count += 1
        if first != record["first_results"]:
            raise ValueError(f"First retrieval replay mismatch: {source_id}")
        if record["category"] != "single":
            second = _serialize_hits(
                retriever.search(str(record["second_query"]),
                                 topk=TOPK,
                                 return_scores=True))
            query_count += 1
            if second != record["second_results"]:
                raise ValueError(
                    f"Second retrieval replay mismatch: {source_id}")
    expected_queries = (sum(QUOTAS["single"].values()) + 2 * sum(
        sum(QUOTAS[category].values())
        for category in ("comparison", "bridge")))
    if query_count != expected_queries:
        raise ValueError("Retrieval replay query count mismatch")
    selected_order_sha256 = hashlib.sha256("".join(
        f"{source_id}\n"
        for source_id in selected_ids).encode("utf-8")).hexdigest()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "manifest_sha256": sha256_file(manifest_path),
        "catalog_sha256": sha256_file(catalog_path),
        "evidence_sha256": sha256_file(local_dir / EVIDENCE_FILE),
        "bm25_revision": BM25_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "corpus_sha256": CORPUS_SHA256,
        "topk": TOPK,
        "selected_rows": len(selected_ids),
        "query_count": query_count,
        "selected_order_sha256": selected_order_sha256,
    }
    receipt_path = local_dir / REPLAY_FILE
    atomic_write(receipt_path, canonical_json_bytes(receipt))
    write_digest_sidecar(receipt_path)
    verify_replay_receipt(manifest_path)
    return receipt_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--local-dir", type=Path, required=True)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--local-dir", type=Path, required=True)
    retrieve.add_argument("--index-path", type=Path, required=True)
    retrieve.add_argument("--corpus-path", type=Path, required=True)
    retrieve.add_argument("--offsets-path", type=Path, required=True)
    build = subparsers.add_parser("materialize")
    build.add_argument("--local-dir", type=Path, required=True)
    build.add_argument("--model-dir", type=Path, required=True)
    build.add_argument("--eval-catalog",
                       type=Path,
                       action="append",
                       default=[])
    build.add_argument("--eval-parquet",
                       type=Path,
                       action="append",
                       default=[])
    native = subparsers.add_parser(
        "materialize-native",
        help="create independent native data from sealed retrieval evidence")
    native.add_argument("--source-manifest", type=Path, required=True)
    native.add_argument("--output-dir", type=Path, required=True)
    native.add_argument("--model-dir", type=Path, required=True)
    native.add_argument("--eval-catalog",
                        type=Path,
                        action="append",
                        default=[])
    native.add_argument("--eval-parquet",
                        type=Path,
                        action="append",
                        default=[])
    native.add_argument("--no-reselection", action="store_true")
    for command in ("verify", "verify-no-reselection"):
        verify = subparsers.add_parser(command)
        verify.add_argument("--manifest", type=Path, required=True)
        verify.add_argument("--model-dir", type=Path, required=True)
        verify.add_argument("--eval-catalog",
                            type=Path,
                            action="append",
                            default=[])
        verify.add_argument("--eval-parquet",
                            type=Path,
                            action="append",
                            default=[])
        verify.add_argument("--expected-tool-protocol",
                            choices=SUPPORTED_TOOL_PROTOCOLS)
        verify.add_argument("--source-manifest", type=Path)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--index-path", type=Path, required=True)
    replay.add_argument("--corpus-path", type=Path, required=True)
    replay.add_argument("--offsets-path", type=Path, required=True)
    verify_replay = subparsers.add_parser("verify-replay")
    verify_replay.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "download":
        paths = download_sources(args.local_dir)
        print("Downloaded and verified: " +
              ", ".join(str(path) for path in paths.values()))
    elif args.command == "retrieve":
        ledger = retrieve_evidence(args.local_dir, args.index_path,
                                   args.corpus_path, args.offsets_path)
        print(f"Built retrieval evidence: {ledger}")
    elif args.command == "materialize":
        manifest = materialize(args.local_dir, args.model_dir,
                               args.eval_catalog, args.eval_parquet)
        print(f"Built retrieval-verified search mixture: {manifest}")
    elif args.command == "materialize-native":
        manifest = materialize_native(args.source_manifest, args.output_dir,
                                       args.model_dir, args.eval_catalog,
                                       args.eval_parquet,
                                       reselect_catalog=not args.no_reselection)
        print(f"Built Qwen3.5 native search mixture: {manifest}")
    elif args.command in ("verify", "verify-no-reselection"):
        verify_manifest(args.manifest, args.model_dir, args.eval_catalog,
                         args.eval_parquet, args.expected_tool_protocol,
                         args.source_manifest,
                         reselect_catalog=args.command == "verify")
        print(
            f"Verified retrieval-verified search mixture: {args.manifest.resolve()}"
        )
    elif args.command == "replay":
        receipt = replay_selected_retrieval(args.manifest, args.index_path,
                                            args.corpus_path,
                                            args.offsets_path)
        print(f"Replayed selected retrieval evidence: {receipt}")
    else:
        verify_replay_receipt(args.manifest)
        print(f"Verified retrieval replay receipt: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
