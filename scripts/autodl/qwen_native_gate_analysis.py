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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REWARD_SCORE_DIR = REPOSITORY_ROOT / "verl" / "utils" / "reward_score"
if str(REWARD_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_SCORE_DIR))

import qa_em  # noqa: E402


SCHEMA = "search-r1.qwen-native-gate"
SCHEMA_VERSION = 3
ACTIVE_PROMPT_VERSION = "qwen35-native-search-v3-original-aligned"
ACTIVE_DATA_SCHEMA_VERSION = 4
LEGACY_DATA_SCHEMA_VERSION = 3
STAGE_SHAPES = {
    "g0_g1": (16, 2),
    "g2": (32, 3),
    "g3": (64, 5),
}
STAGE_ARTIFACTS = {
    "g0_g1": "probe_autonomous",
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


def load_data_contract(
        manifest_path: Path, catalog_path: Path,
        stage: str) -> tuple[list[str], dict[str, str],
                             dict[str, list[str]], list[str], bool]:
    manifest = load_json(manifest_path)
    prompt_contract = manifest.get("prompt_contract")
    data_schema_version = manifest.get("schema_version")
    if (data_schema_version not in {
            LEGACY_DATA_SCHEMA_VERSION, ACTIVE_DATA_SCHEMA_VERSION}
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
    gold_answers: dict[str, list[str]] = {}
    for record in load_jsonl(catalog_path):
        sample_id = record.get("sample_id")
        question = record.get("question")
        if not isinstance(sample_id, str) or not sample_id or sample_id in questions:
            raise ValueError("catalog sample IDs must be unique non-empty strings")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"catalog question is invalid for {sample_id!r}")
        answers = record.get("golden_answers")
        if (not isinstance(answers, list) or not answers
                or not all(isinstance(answer, str) and answer.strip()
                           for answer in answers)):
            raise ValueError(f"catalog golden answers are invalid for {sample_id!r}")
        questions[sample_id] = question.strip()
        gold_answers[sample_id] = list(answers)

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
    active_v3 = (data_schema_version == ACTIVE_DATA_SCHEMA_VERSION
                 and prompt_contract.get("prompt_version") == ACTIVE_PROMPT_VERSION)
    if ((data_schema_version == ACTIVE_DATA_SCHEMA_VERSION) != active_v3
            or (prompt_contract.get("prompt_version") == ACTIVE_PROMPT_VERSION
                and not active_v3)):
        raise ValueError("Qwen native data schema/prompt version mismatch")
    artifact_label = STAGE_ARTIFACTS[stage]
    if stage == "g0_g1" and not active_v3:
        artifact_label = "probe_forced"
    expected_ids = artifact_ids(artifact_label, question_count)
    g0_ids = artifact_ids("probe_g0", 8) if stage == "g0_g1" else []
    return expected_ids, questions, gold_answers, g0_ids, active_v3


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def criterion(observed: int | float, comparison: str,
              threshold: int | float) -> dict[str, Any]:
    if comparison == ">=":
        passed = observed >= threshold
    elif comparison == "<=":
        passed = observed <= threshold
    elif comparison == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"unknown criterion comparison: {comparison}")
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
                    expected_questions: Mapping[str, str],
                    active_v3: bool = False) -> None:
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
        if active_v3:
            events = record.get("generation_events")
            retrievals = record.get("retrieval_events")
            if record.get("schema_version") != 3:
                raise ValueError(f"active v3 trace schema mismatch for {key}")
            if not isinstance(events, list) or not isinstance(retrievals, list):
                raise ValueError(f"active v3 events are missing for {key}")
            if (record.get("max_action_budget") != 4
                    or record.get("action_count") != len(events)
                    or "max_searches" in record
                    or "terminal_search_request" in record):
                raise ValueError(f"active v3 action-budget contract mismatch for {key}")
            if record.get("executed_search_count") != len(retrievals):
                raise ValueError(f"active v3 retrieval count mismatch for {key}")
            for name in ("policy_token_count", "observation_token_count",
                         "observation_policy_token_count"):
                value = record.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"active v3 {name} is invalid for {key}")
            if not isinstance(record.get("info_mask_consistent"), bool):
                raise ValueError(f"active v3 info-mask evidence is missing for {key}")
    if set(groups) != set(expected_ids):
        raise ValueError(f"{stage} trace sample IDs do not match the fixed data artifact")
    expected_slots = set(range(group_size))
    for sample_id, slots in groups.items():
        if slots != expected_slots:
            raise ValueError(f"group slots mismatch for {sample_id}: {sorted(slots)}")


def replay_strict_exact_match(
        records: Sequence[Mapping[str, Any]],
        expected_gold_answers: Mapping[str, Sequence[str]],
) -> dict[tuple[str, int], dict[str, Any]]:
    replayed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = trace_key(record)
        catalog_gold = list(expected_gold_answers.get(key[0], ()))
        if not catalog_gold:
            raise ValueError(f"catalog gold answers are missing for {key}")
        if record.get("gold_answers") != catalog_gold:
            raise ValueError(f"trace gold answers do not match the catalog for {key}")

        answer_fields = [
            name for name in ("extracted_answer", "final_answer") if name in record
        ]
        if not answer_fields:
            raise ValueError(f"trace has no extracted or final answer for {key}")
        answer = record.get(answer_fields[0])
        if any(record.get(name) != answer for name in answer_fields[1:]):
            raise ValueError(f"trace extracted and final answers disagree for {key}")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"trace answer must be a string or null for {key}")

        # Match the training reward: a missing/blank extracted answer is always
        # wrong, even when a gold answer normalizes to the empty string.
        if answer is None or not answer.strip():
            strict_em = 0
            subem = 0
        else:
            strict_em = int(qa_em.em_check(answer, catalog_gold))
            subem = int(qa_em.subem_check(answer, catalog_gold))
        trace_em = record.get("em")
        if (isinstance(trace_em, bool) or not isinstance(trace_em, (int, float))
                or not math.isfinite(float(trace_em))
                or float(trace_em) not in (0.0, 1.0)):
            raise ValueError(f"trace EM must be exactly 0 or 1 for {key}")
        if int(trace_em) != strict_em:
            raise ValueError(f"trace EM does not match strict replay for {key}")
        replayed[key] = {
            "answer_field": answer_fields[0],
            "answer": answer,
            "catalog_gold_answers": catalog_gold,
            "strict_em": strict_em,
            "subem": subem,
        }
    return replayed


def verify_g3_registered_outputs(
        args: argparse.Namespace,
        reward_replay: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Bind the legacy G3 report to independently replayed strict EM."""
    summary = load_json(args.output_dir / "summary.json")
    decision = load_json(args.output_dir / "go_no_go.json")
    reports = load_jsonl(args.output_dir / "per_trajectory.jsonl")
    trace_digest = sha256_file(args.trace)
    catalog_digest = sha256_file(args.catalog)

    summary_input = summary.get("input")
    overall = summary.get("overall")
    if not isinstance(summary_input, Mapping) or not isinstance(overall, Mapping):
        raise ValueError("registered G3 summary is incomplete")
    expected_positive = sum(int(item["strict_em"])
                            for item in reward_replay.values())
    checks = {
        "stage": summary_input.get("stage") == "qwen_native_g3",
        "trace": summary_input.get("trace_sha256") == trace_digest,
        "catalog": summary_input.get("catalog_sha256") == catalog_digest,
        "checkpoint": summary_input.get("checkpoint_digest")
        == args.expected_checkpoint_digest,
        "correct_count": overall.get("correct_count") == expected_positive,
        "decision": summary.get("decision") == decision.get("decision"),
        "decision_trace": decision.get("trace_sha256") == trace_digest,
        "decision_catalog": decision.get("catalog_sha256") == catalog_digest,
        "decision_checkpoint": decision.get("checkpoint_digest")
        == args.expected_checkpoint_digest,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("registered G3 output disagrees with strict EM replay: "
                         + ", ".join(failed))

    seen: set[tuple[str, int]] = set()
    for report in reports:
        key = trace_key(report)
        if key in seen or key not in reward_replay:
            raise ValueError(f"registered G3 report has an invalid identity: {key}")
        seen.add(key)
        replay = reward_replay[key]
        if report.get("em") != replay["strict_em"]:
            raise ValueError(
                f"registered G3 report EM disagrees with strict replay for {key}")
        report["strict_em_replay"] = dict(replay)
    if seen != set(reward_replay):
        raise ValueError("registered G3 reports do not cover the replayed trajectories")
    return summary, decision, reports


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


def _token_prefix_integrity(event: Mapping[str, Any]) -> bool:
    raw_ids = event.get("raw_token_ids")
    action_ids = event.get("action_token_ids")
    if (not isinstance(raw_ids, list) or not isinstance(action_ids, list)
            or not all(isinstance(item, int) and not isinstance(item, bool)
                       and item >= 0 for item in raw_ids + action_ids)):
        return False
    return (raw_ids[:len(action_ids)] == action_ids
            and event.get("tail_dropped")
            == (len(action_ids) < len(raw_ids)))


def _thinking_diagnostic(event: Mapping[str, Any], context: str) -> dict[str, Any]:
    text = event.get("text")
    if not isinstance(text, str):
        text = ""
    action_positions = [position for marker in ("<tool_call>", "<answer>")
                        if (position := text.find(marker)) >= 0]
    action_position = min(action_positions) if action_positions else len(text)
    close_position = text.find("</think>")
    closing = (close_position >= 0 and close_position < action_position
               and text.count("</think>") == 1)
    reasoning = text[:close_position] if closing else ""
    if reasoning.lstrip().startswith("<think>"):
        reasoning = reasoning.lstrip()[len("<think>"):]
    return {
        "context": context,
        # The active renderer contract opens thinking in every assistant prefix.
        "template_opening_provided": True,
        "nonempty_reasoning": bool(reasoning.strip()),
        "closing_before_action": closing,
    }


def trace_structure(record: Mapping[str, Any]) -> dict[str, Any]:
    events = record.get("generation_events")
    if not isinstance(events, list):
        events = []
    requested = sum(event.get("action") == "search" for event in events
                    if isinstance(event, Mapping))
    if not events:
        requested = sum(turn.get("action") == "search"
                        for turn in record.get("turns", []))
    executed = int(record.get("executed_search_count", 0))
    retrievals = record.get("retrieval_events")
    retrieval_count = len(retrievals) if isinstance(retrievals, list) else 0
    tool_responses = sum(
        turn.get("retrieval_executed") is True
        and isinstance(turn.get("observation"), str)
        and bool(turn["observation"].strip())
        for turn in record.get("turns", []))
    prefix_ok = sum(_token_prefix_integrity(event) for event in events
                    if isinstance(event, Mapping))
    thinking = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        context = event.get("generation_context")
        if context not in {"initial_question", "tool_response", "user_retry"}:
            context = "initial_question" if index == 0 else "unknown"
        thinking.append(_thinking_diagnostic(event, str(context)))
    return {
        "generation_turn_count": len(events),
        "action_token_prefix_integrity_count": prefix_ok,
        "action_tail_leak_count": len(events) - prefix_ok,
        "requested_search_count": requested,
        "executed_search_count": executed,
        "retrieval_event_count": retrieval_count,
        "nonempty_tool_response_count": tool_responses,
        "retrieval_aligned": requested == executed == retrieval_count == tool_responses,
        "thinking": thinking,
    }


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
    result = {
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
    result.update(trace_structure(record))
    return result


def analyze_protocol_probe(directory: Path, expected_sample_ids: Sequence[str],
                           checkpoint_digest: str,
                           active_v3: bool = True) -> tuple[dict[str, Any],
                                                            list[dict[str, Any]]]:
    manifest = load_json(directory / "manifest.json")
    records = load_jsonl(directory / "records.jsonl")
    resolved = load_json(directory / "resolved-config.json")
    version = manifest.get("schema_version")
    if manifest.get("schema") != "search-r1.qwen-native-protocol-probe" \
            or version not in {1, 3}:
        raise ValueError("G0 protocol probe manifest schema mismatch")
    if active_v3 and version != 3:
        raise ValueError("active v3 gate requires a v3 protocol probe")
    expected_modes = (("direct", "native_manager") if version == 3 else
                      ("direct", "native_manager", "legacy_manager"))
    expected_mode_counts = {mode: 16 for mode in expected_modes}
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
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    if resolved.get("sampling") != expected_sampling:
        raise ValueError("G0 protocol probe sampling config mismatch")
    if version == 3 and (resolved.get("prompt_version") != ACTIVE_PROMPT_VERSION
                         or resolved.get("max_action_budget") != 4
                         or resolved.get("max_obs_length") != 500):
        raise ValueError("G0 v3 protocol probe contract mismatch")
    by_mode: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        mode = record.get("mode")
        if mode not in expected_modes:
            raise ValueError(f"unknown G0 mode: {mode!r}")
        key = trace_key(record)
        if key in by_mode[mode]:
            raise ValueError(f"duplicate G0 record: {mode}:{key}")
        by_mode[mode][key] = record
    if any(len(by_mode[mode]) != 16 for mode in expected_modes):
        raise ValueError("G0 requires 16 records for each comparison mode")
    keys = set(by_mode["direct"])
    if any(set(by_mode[mode]) != keys for mode in expected_modes[1:]):
        raise ValueError("G0 comparison modes are not sample-aligned")
    expected_keys = {(sample_id, slot) for sample_id in expected_sample_ids
                     for slot in range(2)}
    if keys != expected_keys:
        raise ValueError("G0 records do not match the fixed probe_g0 artifact")
    prompt_matches = sum(
        by_mode["direct"][key].get("prompt_token_sha256")
        == by_mode["native_manager"][key].get("prompt_token_sha256") for key in keys)
    if version == 1:
        raw_matches = sum(
            by_mode["direct"][key].get("raw_text")
            == by_mode["native_manager"][key].get("raw_text") for key in keys)
        criteria = {
            "prompt_token_match_count": criterion(prompt_matches, "==", 16),
            "raw_text_match_count": criterion(raw_matches, "==", 16),
        }
        return {
            "schema_version": 1,
            "legacy_read_only": True,
            "records": len(records),
            "prompt_token_match_count": prompt_matches,
            "raw_text_match_count": raw_matches,
            "criteria": criteria,
        }, records

    if manifest.get("environment_replay_sha256") != sha256_file(
            directory / "environment-replay.json"):
        raise ValueError("G0 E0 environment replay digest mismatch")
    replay = load_json(directory / "environment-replay.json")
    replay_count_fields = (
        "requested_search_count", "executed_search_count",
        "retrieval_event_count", "nonempty_tool_response_count",
        "retrieved_document_count", "tool_response_policy_token_count",
    )
    if any(isinstance(replay.get(name), bool)
           or not isinstance(replay.get(name), int)
           or replay[name] < 0 for name in replay_count_fields):
        raise ValueError("G0 E0 environment replay counts are invalid")
    first_action_matches = sum(
        by_mode["direct"][key].get("action_token_ids")
        == by_mode["native_manager"][key].get("action_token_ids") for key in keys)
    direct_integrity = sum(_token_prefix_integrity(by_mode["direct"][key])
                           for key in keys)
    manager_integrity = sum(
        _token_prefix_integrity(by_mode["native_manager"][key]) for key in keys)
    roundtrip = int(
        replay.get("requested_search_count") == 1
        and replay.get("executed_search_count") == 1
        and replay.get("retrieval_event_count") == 1
        and replay.get("nonempty_tool_response_count") == 1)
    tool_role = int(replay.get("tool_role_rendered") is True)
    mask_leak = int(replay.get("tool_response_policy_token_count", -1))
    if replay.get("info_mask_consistent") is not True:
        mask_leak = max(mask_leak, 1)
    criteria = {
        "prompt_token_match_count": criterion(prompt_matches, "==", 16),
        "first_action_token_match_count": criterion(first_action_matches, "==", 16),
        "direct_action_prefix_integrity_count": criterion(direct_integrity, "==", 16),
        "manager_action_prefix_integrity_count": criterion(manager_integrity, "==", 16),
        "e0_search_roundtrip_count": criterion(roundtrip, "==", 1),
        "e0_retrieved_document_count": criterion(
            int(replay.get("retrieved_document_count", -1)), "==", 3),
        "e0_tool_role_count": criterion(tool_role, "==", 1),
        "e0_mask_leak_count": criterion(mask_leak, "==", 0),
    }
    return {
        "schema_version": 3,
        "records": len(records),
        "prompt_token_match_count": prompt_matches,
        "first_action_token_match_count": first_action_matches,
        "direct_action_prefix_integrity_count": direct_integrity,
        "manager_action_prefix_integrity_count": manager_integrity,
        "environment_replay": replay,
        "criteria": criteria,
    }, records


def per_question(records: list[dict[str, Any]],
                 diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record, diagnostic in zip(records, diagnostics):
        grouped[str(record["sample_id"])].append((record, diagnostic))
    output = []
    for sample_id in sorted(grouped):
        members = grouped[sample_id]
        searches = [int(record["executed_search_count"]) for record, _ in members]
        has_reward_replay = all(
            "strict_em" in diagnostic and "subem" in diagnostic
            for _, diagnostic in members)
        strict_scores = [
            int(diagnostic.get("strict_em", record.get("em", 0)))
            for record, diagnostic in members
        ]
        subem_scores = [
            int(diagnostic.get("subem", strict_score))
            for strict_score, (_, diagnostic) in zip(strict_scores, members)
        ]
        clean_correct_multi = [
            record for record, diagnostic in members
            if int(diagnostic.get("strict_em", record.get("em", 0))) == 1
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
        question = {
            "sample_id": sample_id,
            "question": members[0][0].get("question"),
            "trajectory_count": len(members),
            "correct_count": sum(strict_scores),
            "search_counts": searches,
            "valid_correct_multi_search_count": len(clean_correct_multi),
            "covered": bool(clean_correct_multi),
            "learnable": bool(clean_correct_multi and clean_wrong),
            "complete_two_search_chain_count": sum(
                diagnostic["complete_two_search_chain"] for _, diagnostic in members),
        }
        if has_reward_replay:
            question.update({
                "strict_em_mixed": 0 < sum(strict_scores) < len(strict_scores),
                "subem_positive_count": sum(subem_scores),
                "subem_mixed": 0 < sum(subem_scores) < len(subem_scores),
            })
        output.append(question)
    return output


def analyze_trace_stage(
        stage: str, records: list[dict[str, Any]],
        reward_replay: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
        active_v3: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if stage == "g2" and reward_replay is None:
        raise ValueError("G2 analysis requires independent strict EM replay")
    diagnostics = [trace_diagnostics(record) for record in records]
    if reward_replay is not None:
        for record, diagnostic in zip(records, diagnostics):
            key = trace_key(record)
            replay = reward_replay.get(key)
            if replay is None:
                raise ValueError(f"strict EM replay is missing for {key}")
            diagnostic["strict_em"] = int(replay["strict_em"])
            diagnostic["subem"] = int(replay["subem"])
    questions = per_question(records, diagnostics)
    if stage == "g0_g1" and active_v3:
        generation_turns = sum(item["generation_turn_count"] for item in diagnostics)
        prefix_integrity = sum(
            item["action_token_prefix_integrity_count"] for item in diagnostics)
        tail_leaks = sum(item["action_tail_leak_count"] for item in diagnostics)
        mask_consistent = sum(record.get("info_mask_consistent") is True
                              for record in records)
        observation_policy_tokens = sum(
            int(record.get("observation_policy_token_count", 0)) for record in records)
        retrieval_errors = sum(not item["retrieval_aligned"] for item in diagnostics)
        thinking_rows = [row for item in diagnostics for row in item["thinking"]]
        contexts = {}
        for context in ("initial_question", "tool_response", "user_retry", "unknown"):
            rows = [row for row in thinking_rows if row["context"] == context]
            if rows:
                contexts[context] = {
                    "generation_turn_count": len(rows),
                    "template_opening_count": sum(
                        row["template_opening_provided"] for row in rows),
                    "nonempty_reasoning_count": sum(
                        row["nonempty_reasoning"] for row in rows),
                    "closing_before_action_count": sum(
                        row["closing_before_action"] for row in rows),
                }
        criteria = {
            "action_token_prefix_integrity_count": criterion(
                prefix_integrity, "==", generation_turns),
            "action_tail_leak_count": criterion(tail_leaks, "==", 0),
            "info_mask_consistent_count": criterion(mask_consistent, "==", len(records)),
            "observation_policy_token_count": criterion(
                observation_policy_tokens, "==", 0),
            "retrieval_alignment_error_count": criterion(retrieval_errors, "==", 0),
        }
        overall = {
            "legal_first_action_count": sum(
                item["first_action_legal"] for item in diagnostics),
            "first_action_search_count": sum(
                bool(record.get("turns"))
                and record["turns"][0].get("action") == "search"
                for record in records),
            "first_action_answer_count": sum(
                bool(record.get("turns"))
                and record["turns"][0].get("action") == "answer"
                for record in records),
            "answer_trajectory_count": sum(
                any(turn.get("action") == "answer"
                    for turn in record.get("turns", [])) for record in records),
            "invalid_trajectory_count": sum(
                int(record.get("invalid_action_count", 0)) > 0
                for record in records),
            "first_turn_clipped_count": sum(
                item["first_turn_clipped"] for item in diagnostics),
            "search_turn_count": sum(item["search_turn_count"] for item in diagnostics),
            "requested_search_count": sum(
                item["requested_search_count"] for item in diagnostics),
            "executed_search_count": sum(
                item["executed_search_count"] for item in diagnostics),
            "retrieval_event_count": sum(
                item["retrieval_event_count"] for item in diagnostics),
            "nonempty_tool_response_count": sum(
                item["nonempty_tool_response_count"] for item in diagnostics),
            "generation_turn_count": generation_turns,
            "search_count_distribution": {
                str(count): sum(int(record.get("executed_search_count", 0)) == count
                                for record in records)
                for count in range(5)
            },
            "thinking": {
                "diagnostic_only": True,
                "template_opening_source": "qwen35 v3 renderer contract",
                "generation_turn_count": len(thinking_rows),
                "template_opening_count": sum(
                    row["template_opening_provided"] for row in thinking_rows),
                "nonempty_reasoning_count": sum(
                    row["nonempty_reasoning"] for row in thinking_rows),
                "closing_before_action_count": sum(
                    row["closing_before_action"] for row in thinking_rows),
                "by_context": contexts,
            },
        }
    elif stage == "g0_g1":
        # Preserve the archived v2 interpretation without using its
        # forced-search capability thresholds for active v3 admission.
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
        strict_em_count = sum(item["strict_em"] for item in diagnostics)
        strict_em_mixed_groups = sum(item["strict_em_mixed"] for item in questions)
        subem_count = sum(item["subem"] for item in diagnostics)
        criteria = {
            "invalid_trajectory_count": criterion(invalid, "<=", 5),
            "clipped_trajectory_count": criterion(clipped, "<=", 5),
            "degenerate_search_count": criterion(degenerate, "<=", max_degenerate),
            "strict_em_positive_count": criterion(strict_em_count, ">=", 1),
            "strict_em_mixed_group_count": criterion(strict_em_mixed_groups, ">=", 1),
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
            "strict_em_positive_count": strict_em_count,
            "strict_em_mixed_group_count": strict_em_mixed_groups,
            "subem_positive_count": subem_count,
            "subem_mixed_group_count": sum(item["subem_mixed"] for item in questions),
        }
    overall["repeated_query_count"] = sum(item["repeated_query_count"] for item in diagnostics)
    overall["non_ascii_query_count"] = sum(item["non_ascii_query_count"] for item in diagnostics)
    overall["em_count"] = sum(
        int(diagnostic.get("strict_em", record.get("em", 0))) == 1
        for record, diagnostic in zip(records, diagnostics))
    overall["em"] = ratio(overall["em_count"], len(records))
    overall["criteria"] = criteria
    decorated = []
    for record, diagnostic in zip(records, diagnostics):
        item = {"trace": record, "diagnostics": diagnostic}
        if reward_replay is not None:
            item["reward_replay"] = dict(reward_replay[trace_key(record)])
        decorated.append(item)
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
                  expected_gold_answers: Mapping[str, Sequence[str]],
                  g0_ids: Sequence[str], active_v3: bool = False) -> dict[str, Any]:
    if args.stage == "g3":
        records = load_jsonl(args.trace)
        validate_traces(records, args.stage, args.expected_checkpoint_digest,
                        expected_ids, expected_questions, active_v3=active_v3)
        reward_replay = replay_strict_exact_match(records,
                                                   expected_gold_answers)
        output = args.output_dir
        if output.exists() or output.is_symlink():
            raise ValueError(f"refusing to overwrite output directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix=f".{output.name}.verified-", dir=output.parent) as temporary:
            staged_args = argparse.Namespace(**vars(args))
            staged_args.output_dir = Path(temporary) / "registered-output"
            rc = analyze_g3_with_registered_gate(staged_args)
            if rc != 0:
                raise ValueError(
                    f"registered G3 analyzer failed with exit code {rc}")
            summary, decision, reports = verify_g3_registered_outputs(
                staged_args, reward_replay)
            replay_summary = {
                "schema": "search-r1.strict-em-replay",
                "schema_version": 1,
                "source": "catalog.golden_answers+trace.extracted_answer/final_answer",
                "verified": True,
                "trajectory_count": len(reward_replay),
                "strict_em_positive_count": sum(
                    int(item["strict_em"]) for item in reward_replay.values()),
                "subem_positive_count": sum(
                    int(item["subem"]) for item in reward_replay.values()),
                "trace_sha256": sha256_file(args.trace),
                "catalog_sha256": sha256_file(args.catalog),
            }
            summary["strict_em_replay"] = replay_summary
            decision["stage"] = "g3"
            decision["strict_em_replay"] = replay_summary
            atomic_write(staged_args.output_dir / "summary.json",
                         canonical_bytes(summary))
            atomic_write(staged_args.output_dir / "go_no_go.json",
                         canonical_bytes(decision))
            atomic_write(
                staged_args.output_dir / "per_trajectory.jsonl",
                b"".join(canonical_bytes(report) for report in reports))
            os.replace(staged_args.output_dir, output)
        return decision

    records = load_jsonl(args.trace)
    validate_traces(records, args.stage, args.expected_checkpoint_digest,
                    expected_ids, expected_questions, active_v3=active_v3)
    reward_replay = (replay_strict_exact_match(records, expected_gold_answers)
                     if args.stage == "g2" else None)
    overall, decorated, questions = analyze_trace_stage(
        args.stage, records, reward_replay, active_v3=active_v3)
    protocol = None
    protocol_records: list[dict[str, Any]] = []
    if args.stage == "g0_g1":
        if args.protocol_probe_dir is None:
            raise ValueError("G0+G1 analysis requires --protocol-probe-dir")
        protocol, protocol_records = analyze_protocol_probe(
            args.protocol_probe_dir, g0_ids, args.expected_checkpoint_digest,
            active_v3=active_v3)
    if args.stage == "g0_g1" and active_v3:
        all_criteria = {f"g1_{name}": value
                        for name, value in overall["criteria"].items()}
    else:
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
    if args.stage == "g2":
        lines.insert(
            7,
            f"- SubEM (diagnostic only): {overall['subem_positive_count']}/{len(records)}",
        )
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
        (expected_ids, expected_questions, expected_gold_answers,
         g0_ids, active_v3) = load_data_contract(
             args.data_manifest, args.catalog, args.stage)
        if args.stage != "g3":
            args.output_dir.mkdir(parents=True, exist_ok=False)
        result = write_outputs(args, expected_ids, expected_questions,
                               expected_gold_answers, g0_ids, active_v3)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Qwen native gate analysis error: {error}", file=sys.stderr)
        return 1
    print(f"Qwen native {args.stage} decision: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
