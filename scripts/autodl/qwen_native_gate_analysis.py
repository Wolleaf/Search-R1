#!/usr/bin/env python3
"""Analyze the bounded Qwen native protocol gates without changing training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "search-r1.qwen-native-gate"
SCHEMA_VERSION = 1
STAGE_SHAPES = {
    "g0_g1": (16, 2),
    "g2": (32, 3),
    "g3": (64, 5),
}
STAGE_ARTIFACTS = {
    "g0_g1": "probe_forced",
    "g2": "probe_autonomous",
    "g3": "probe",
}
DEGENERATE_QUERIES = {"", "query", "and"}
WORD = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "did", "do", "does", "for", "from", "how",
    "in", "is", "of", "on", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSONL is missing or symlinked: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"JSONL line is not terminated: {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON: {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            records.append(value)
    return records


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSON is missing or symlinked: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_data_contract(manifest_path: Path, catalog_path: Path,
                       stage: str) -> tuple[list[str], dict[str, str], list[str]]:
    manifest = load_json(manifest_path)
    prompt_contract = manifest.get("prompt_contract")
    if (manifest.get("schema_version") != 3
            or not isinstance(prompt_contract, Mapping)
            or prompt_contract.get("tool_protocol") != "qwen35_native"):
        raise ValueError("Qwen native data manifest contract mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Qwen native data manifest artifacts are missing")
    catalog_artifact = artifacts.get("catalog")
    if not isinstance(catalog_artifact, Mapping) or catalog_artifact.get(
            "file") != "catalog.jsonl":
        raise ValueError("Qwen native catalog artifact contract mismatch")
    expected_catalog = manifest_path.resolve().parent / "catalog.jsonl"
    if catalog_path.resolve() != expected_catalog.resolve():
        raise ValueError("catalog is not the one bound to the data manifest")
    if catalog_artifact.get("sha256") != sha256_file(catalog_path):
        raise ValueError("catalog digest does not match the data manifest")

    questions: dict[str, str] = {}
    for record in load_jsonl(catalog_path):
        sample_id = record.get("sample_id")
        question = record.get("question")
        if not isinstance(sample_id, str) or not sample_id or sample_id in questions:
            raise ValueError("catalog sample IDs must be unique non-empty strings")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"catalog question is invalid for {sample_id!r}")
        questions[sample_id] = question.strip()

    def artifact_ids(label: str, rows: int) -> list[str]:
        artifact = artifacts.get(label)
        if not isinstance(artifact, Mapping) or artifact.get("rows") != rows:
            raise ValueError(f"data artifact contract mismatch: {label}")
        sample_ids = artifact.get("sample_ids")
        if (not isinstance(sample_ids, list) or len(sample_ids) != rows
                or not all(isinstance(item, str) and item for item in sample_ids)
                or len(set(sample_ids)) != rows):
            raise ValueError(f"data artifact sample IDs are invalid: {label}")
        missing = [sample_id for sample_id in sample_ids if sample_id not in questions]
        if missing:
            raise ValueError(f"data artifact IDs are absent from catalog: {label}")
        return sample_ids

    question_count, _ = STAGE_SHAPES[stage]
    expected_ids = artifact_ids(STAGE_ARTIFACTS[stage], question_count)
    g0_ids = artifact_ids("probe_g0", 8) if stage == "g0_g1" else []
    return expected_ids, questions, g0_ids


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def criterion(observed: int | float, comparison: str,
              threshold: int | float) -> dict[str, Any]:
    passed = observed >= threshold if comparison == ">=" else observed <= threshold
    return {
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
        "passed": passed,
    }


def trace_key(record: Mapping[str, Any]) -> tuple[str, int]:
    sample_id = record.get("sample_id")
    group_slot = record.get("group_slot")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("trace sample_id must be a non-empty string")
    if isinstance(group_slot, bool) or not isinstance(group_slot, int):
        raise ValueError("trace group_slot must be an integer")
    return sample_id, group_slot


def validate_traces(records: list[dict[str, Any]], stage: str,
                    checkpoint_digest: str, expected_ids: Sequence[str],
                    expected_questions: Mapping[str, str]) -> None:
    questions, group_size = STAGE_SHAPES[stage]
    expected_rows = questions * group_size
    if len(records) != expected_rows:
        raise ValueError(f"{stage} expected {expected_rows} traces, found {len(records)}")
    groups: dict[str, set[int]] = defaultdict(set)
    keys = set()
    for record in records:
        key = trace_key(record)
        if key in keys:
            raise ValueError(f"duplicate trace identity: {key}")
        keys.add(key)
        groups[key[0]].add(key[1])
        if record.get("checkpoint_digest") != checkpoint_digest:
            raise ValueError(f"checkpoint digest mismatch for {key}")
        question = record.get("question")
        if (not isinstance(question, str)
                or question.strip() != expected_questions.get(key[0])):
            raise ValueError(f"trace question does not match the fixed catalog for {key}")
        if not isinstance(record.get("turns"), list):
            raise ValueError(f"trace turns must be a list for {key}")
        if record.get("executed_search_count") != sum(
                turn.get("retrieval_executed") is True for turn in record["turns"]):
            raise ValueError(f"executed search count is not aligned for {key}")
    if set(groups) != set(expected_ids):
        raise ValueError(f"{stage} trace sample IDs do not match the fixed data artifact")
    expected_slots = set(range(group_size))
    for sample_id, slots in groups.items():
        if slots != expected_slots:
            raise ValueError(f"group slots mismatch for {sample_id}: {sorted(slots)}")


def query_is_degenerate(value: Any) -> bool:
    return not isinstance(value, str) or value.strip().casefold() in DEGENERATE_QUERIES


def query_relevant(query: str, question: str) -> bool:
    query_words = {word.casefold() for word in WORD.findall(query)} - STOPWORDS
    question_words = {word.casefold() for word in WORD.findall(question)} - STOPWORDS
    return bool(query_words & question_words)


def first_turn_clipped(record: Mapping[str, Any]) -> bool:
    events = record.get("generation_events")
    if isinstance(events, list) and events and isinstance(events[0], Mapping):
        return events[0].get("clipped") is True
    return record.get("response_clipped") is True


def trace_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    turns = record["turns"]
    first = turns[0] if turns else {}
    search_turns = [turn for turn in turns if turn.get("action") == "search"]
    queries = [turn.get("search_query") for turn in search_turns]
    non_degenerate = [
        query for query in queries if isinstance(query, str) and not query_is_degenerate(query)
    ]
    aligned = [
        turn for turn in search_turns
        if turn.get("retrieval_executed") is True
        and isinstance(turn.get("observation"), str)
        and bool(turn["observation"].strip())
        and isinstance(turn.get("retrieved_docs"), list)
        and bool(turn["retrieved_docs"])
    ]
    normalized_queries = [query.strip().casefold() for query in non_degenerate]
    return {
        "first_action_legal": bool(first.get("valid_action") is True
                                   and first.get("action") in {"search", "answer"}),
        "first_search_non_degenerate": bool(first.get("action") == "search"
                                            and not query_is_degenerate(first.get("search_query"))),
        "first_search_degenerate": bool(first.get("action") == "search"
                                        and query_is_degenerate(first.get("search_query"))),
        "first_turn_clipped": first_turn_clipped(record),
        "search_turn_count": len(search_turns),
        "aligned_tool_response_count": len(aligned),
        "degenerate_search_count": sum(query_is_degenerate(query) for query in queries),
        "repeated_query_count": len(normalized_queries) - len(set(normalized_queries)),
        "query_relevant_count": sum(
            query_relevant(query, str(record.get("question", "")))
            for query in non_degenerate),
        "non_ascii_query_count": sum(any(ord(char) > 127 for char in query)
                                     for query in non_degenerate),
        "complete_two_search_chain": bool(
            len(search_turns) >= 2 and len(aligned) == len(search_turns)
            and len(set(normalized_queries)) >= 2),
    }


def analyze_protocol_probe(directory: Path, expected_sample_ids: Sequence[str],
                           checkpoint_digest: str) -> tuple[dict[str, Any],
                                                            list[dict[str, Any]]]:
    manifest = load_json(directory / "manifest.json")
    records = load_jsonl(directory / "records.jsonl")
    resolved = load_json(directory / "resolved-config.json")
    if manifest.get("schema") != "search-r1.qwen-native-protocol-probe" \
            or manifest.get("schema_version") != 1:
        raise ValueError("G0 protocol probe manifest schema mismatch")
    expected_mode_counts = {
        "direct": 16,
        "native_manager": 16,
        "legacy_manager": 16,
    }
    if manifest.get("mode_counts") != expected_mode_counts:
        raise ValueError("G0 protocol probe mode-count manifest mismatch")
    if manifest.get("records_sha256") != sha256_file(directory / "records.jsonl"):
        raise ValueError("G0 protocol probe records digest mismatch")
    if (manifest.get("resolved_config_sha256") != sha256_file(
            directory / "resolved-config.json")
            or manifest.get("checkpoint_digest") != checkpoint_digest
            or resolved.get("checkpoint_digest") != checkpoint_digest):
        raise ValueError("G0 protocol probe config or checkpoint digest mismatch")
    expected_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    }
    if resolved.get("sampling") != expected_sampling:
        raise ValueError("G0 protocol probe sampling config mismatch")
    by_mode: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        mode = record.get("mode")
        if mode not in {"direct", "native_manager", "legacy_manager"}:
            raise ValueError(f"unknown G0 mode: {mode!r}")
        key = trace_key(record)
        if key in by_mode[mode]:
            raise ValueError(f"duplicate G0 record: {mode}:{key}")
        by_mode[mode][key] = record
    if any(len(by_mode[mode]) != 16 for mode in
           ("direct", "native_manager", "legacy_manager")):
        raise ValueError("G0 requires 16 records for each comparison mode")
    keys = set(by_mode["direct"])
    if set(by_mode["native_manager"]) != keys or set(by_mode["legacy_manager"]) != keys:
        raise ValueError("G0 comparison modes are not sample-aligned")
    expected_keys = {(sample_id, slot) for sample_id in expected_sample_ids
                     for slot in range(2)}
    if keys != expected_keys:
        raise ValueError("G0 records do not match the fixed probe_g0 artifact")
    prompt_matches = sum(
        by_mode["direct"][key].get("prompt_token_sha256")
        == by_mode["native_manager"][key].get("prompt_token_sha256") for key in keys)
    raw_text_matches = sum(
        by_mode["direct"][key].get("raw_text")
        == by_mode["native_manager"][key].get("raw_text") for key in keys)
    direct_valid = sum(
        by_mode["direct"][key].get("parsed_action", {}).get("valid") is True
        for key in keys)
    native_valid = sum(
        by_mode["native_manager"][key].get("parsed_action", {}).get("valid") is True
        for key in keys)
    native_degenerate = sum(
        by_mode["native_manager"][key].get("parsed_action", {}).get("action") == "search"
        and query_is_degenerate(
            by_mode["native_manager"][key].get("parsed_action", {}).get("content"))
        for key in keys)
    criteria = {
        "prompt_token_match_count": criterion(prompt_matches, ">=", 16),
        "raw_text_match_count": criterion(raw_text_matches, ">=", 16),
        "direct_parseable_count": criterion(direct_valid, ">=", 15),
        "native_parseable_count": criterion(native_valid, ">=", 15),
        "native_degenerate_query_count": criterion(native_degenerate, "<=", 0),
    }
    return {
        "records": len(records),
        "prompt_token_match_count": prompt_matches,
        "raw_text_match_count": raw_text_matches,
        "direct_parseable_count": direct_valid,
        "native_parseable_count": native_valid,
        "native_degenerate_query_count": native_degenerate,
        "criteria": criteria,
    }, records


def per_question(records: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record, diagnostic in zip(records, diagnostics):
        grouped[str(record["sample_id"])].append((record, diagnostic))
    output = []
    for sample_id in sorted(grouped):
        members = grouped[sample_id]
        searches = [int(record["executed_search_count"]) for record, _ in members]
        clean_correct_multi = [
            record for record, _ in members
            if int(record.get("em", 0)) == 1
            and int(record["executed_search_count"]) >= 2
            and int(record.get("invalid_action_count", 0)) == 0
            and record.get("response_clipped") is not True
        ]
        clean_wrong = [
            record for record, _ in members
            if int(record.get("em", 0)) == 0
            and int(record.get("invalid_action_count", 0)) == 0
            and record.get("response_clipped") is not True
        ]
        output.append({
            "sample_id": sample_id,
            "question": members[0][0].get("question"),
            "trajectory_count": len(members),
            "correct_count": sum(int(record.get("em", 0)) == 1 for record, _ in members),
            "search_counts": searches,
            "valid_correct_multi_search_count": len(clean_correct_multi),
            "covered": bool(clean_correct_multi),
            "learnable": bool(clean_correct_multi and clean_wrong),
            "complete_two_search_chain_count": sum(
                diagnostic["complete_two_search_chain"] for _, diagnostic in members),
        })
    return output


def analyze_trace_stage(stage: str, records: list[dict[str, Any]]) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics = [trace_diagnostics(record) for record in records]
    questions = per_question(records, diagnostics)
    if stage == "g0_g1":
        legal = sum(item["first_action_legal"] for item in diagnostics)
        non_degenerate = sum(item["first_search_non_degenerate"] for item in diagnostics)
        degenerate = sum(item["first_search_degenerate"] for item in diagnostics)
        clipped = sum(item["first_turn_clipped"] for item in diagnostics)
        aligned = sum(item["aligned_tool_response_count"] for item in diagnostics)
        searches = sum(item["search_turn_count"] for item in diagnostics)
        criteria = {
            "legal_first_action_count": criterion(legal, ">=", 31),
            "non_degenerate_first_search_count": criterion(non_degenerate, ">=", 29),
            "degenerate_first_search_count": criterion(degenerate, "<=", 1),
            "first_turn_clipped_count": criterion(clipped, "<=", 1),
            "aligned_tool_response_count": criterion(aligned, ">=", searches),
        }
        overall = {
            "legal_first_action_count": legal,
            "non_degenerate_first_search_count": non_degenerate,
            "degenerate_first_search_count": degenerate,
            "first_turn_clipped_count": clipped,
            "search_turn_count": searches,
            "aligned_tool_response_count": aligned,
        }
    else:
        invalid = sum(int(record.get("invalid_action_count", 0)) > 0 for record in records)
        clipped = sum(record.get("response_clipped") is True for record in records)
        searches = sum(item["search_turn_count"] for item in diagnostics)
        degenerate = sum(item["degenerate_search_count"] for item in diagnostics)
        max_degenerate = math.floor(searches * 0.02)
        criteria = {
            "invalid_trajectory_count": criterion(invalid, "<=", 5),
            "clipped_trajectory_count": criterion(clipped, "<=", 5),
            "degenerate_search_count": criterion(degenerate, "<=", max_degenerate),
        }
        overall = {
            "invalid_trajectory_count": invalid,
            "clipped_trajectory_count": clipped,
            "search_turn_count": searches,
            "degenerate_search_count": degenerate,
            "degenerate_search_ratio": ratio(degenerate, searches),
            "query_relevant_count": sum(item["query_relevant_count"] for item in diagnostics),
            "complete_two_search_chain_count": sum(
                item["complete_two_search_chain"] for item in diagnostics),
        }
    overall["repeated_query_count"] = sum(item["repeated_query_count"] for item in diagnostics)
    overall["non_ascii_query_count"] = sum(item["non_ascii_query_count"] for item in diagnostics)
    overall["em_count"] = sum(int(record.get("em", 0)) == 1 for record in records)
    overall["em"] = ratio(overall["em_count"], len(records))
    overall["criteria"] = criteria
    decorated = [{"trace": record, "diagnostics": diagnostic}
                 for record, diagnostic in zip(records, diagnostics)]
    return overall, decorated, questions


def analyze_g3_with_registered_gate(args: argparse.Namespace) -> int:
    import probe_analysis

    return probe_analysis.main([
        "--trace", str(args.trace),
        "--catalog", str(args.catalog),
        "--expected-checkpoint-digest", args.expected_checkpoint_digest,
        "--output-dir", str(args.output_dir),
    ])


def write_outputs(args: argparse.Namespace, expected_ids: Sequence[str],
                  expected_questions: Mapping[str, str],
                  g0_ids: Sequence[str]) -> dict[str, Any]:
    if args.stage == "g3":
        rc = analyze_g3_with_registered_gate(args)
        if rc != 0:
            raise ValueError(f"registered G3 analyzer failed with exit code {rc}")
        summary = load_json(args.output_dir / "summary.json")
        if summary.get("input", {}).get("stage") != "qwen_native_g3":
            raise ValueError("registered G3 output has the wrong trace stage")
        decision = load_json(args.output_dir / "go_no_go.json")
        decision["stage"] = "g3"
        atomic_write(args.output_dir / "go_no_go.json",
                     canonical_bytes(decision))
        return decision

    records = load_jsonl(args.trace)
    validate_traces(records, args.stage, args.expected_checkpoint_digest,
                    expected_ids, expected_questions)
    overall, decorated, questions = analyze_trace_stage(args.stage, records)
    protocol = None
    protocol_records: list[dict[str, Any]] = []
    if args.stage == "g0_g1":
        if args.protocol_probe_dir is None:
            raise ValueError("G0+G1 analysis requires --protocol-probe-dir")
        protocol, protocol_records = analyze_protocol_probe(
            args.protocol_probe_dir, g0_ids, args.expected_checkpoint_digest)
    all_criteria = dict(overall["criteria"])
    if protocol is not None:
        all_criteria.update({f"g0_{name}": value
                             for name, value in protocol["criteria"].items()})
    failed = [name for name, item in all_criteria.items() if item["passed"] is not True]
    decision = "GO" if not failed else "NO-GO"
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "decision": decision,
        "checkpoint_digest": args.expected_checkpoint_digest,
        "trace_sha256": sha256_file(args.trace),
        "protocol_probe": protocol,
        "overall": overall,
        "criteria": all_criteria,
        "failed_criteria": failed,
    }
    go_no_go = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "decision": decision,
        "criteria": all_criteria,
        "failed_criteria": failed,
        "trace_sha256": summary["trace_sha256"],
    }
    output = args.output_dir
    atomic_write(output / "summary.json", canonical_bytes(summary))
    atomic_write(output / "go_no_go.json", canonical_bytes(go_no_go))
    atomic_write(output / "per_trajectory.jsonl",
                 b"".join(canonical_bytes(item) for item in decorated))
    atomic_write(output / "per_question.jsonl",
                 b"".join(canonical_bytes(item) for item in questions))
    if protocol_records:
        atomic_write(output / "protocol_records.jsonl",
                     b"".join(canonical_bytes(item) for item in protocol_records))
    lines = [
        f"# Qwen Native Gate {args.stage}",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Trajectories: {len(records)}",
        f"- EM: {overall['em_count']}/{len(records)} ({overall['em']:.3f})",
        f"- Searches: {overall['search_turn_count']}",
        f"- Repeated queries: {overall['repeated_query_count']}",
        f"- Non-ASCII queries: {overall['non_ascii_query_count']}",
        "",
        "## Criteria",
        "",
    ]
    for name, item in all_criteria.items():
        lines.append(
            f"- {name}: {item['observed']} {item['comparison']} {item['threshold']} "
            f"({'pass' if item['passed'] else 'fail'})")
    atomic_write(output / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return go_no_go


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGE_SHAPES), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-digest", required=True)
    parser.add_argument("--protocol-probe-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_digest):
            raise ValueError("expected checkpoint digest must be 64 lowercase hex")
        expected_ids, expected_questions, g0_ids = load_data_contract(
            args.data_manifest, args.catalog, args.stage)
        if args.stage != "g3":
            args.output_dir.mkdir(parents=True, exist_ok=False)
        result = write_outputs(args, expected_ids, expected_questions, g0_ids)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Qwen native gate analysis error: {error}", file=sys.stderr)
        return 1
    print(f"Qwen native {args.stage} decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
