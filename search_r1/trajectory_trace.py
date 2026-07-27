"""Dependency-light trajectory parsing and durable JSONL trace writing."""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from search_r1.llm_agent.tool_protocol import (
    QWEN35_REASONING_CONTINUATION,
    QWEN35_TERMINAL_PROMPT,
    QWEN35_TERMINAL_PROMPT_SHA256,
    QWEN35_TERMINAL_PROMPT_VERSION,
    locate_qwen35_action_boundary,
)

TRACE_SCHEMA = "search-r1.trajectory"
TRACE_SCHEMA_VERSION = 3
SUPPORTED_TRACE_SCHEMA_VERSIONS = (1, 2, TRACE_SCHEMA_VERSION)
PUBLISHABLE_TRACE_SCHEMA_VERSIONS = (1, TRACE_SCHEMA_VERSION)
MANIFEST_SCHEMA = "search-r1.trajectory-manifest"
MANIFEST_SCHEMA_VERSION = 1

_ID_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ASSISTANT_MARKER = re.compile(r"<\|im_start\|>assistant[ \t]*(?:\r?\n)?")
_TERMINAL_MARKER = re.compile(r"<\|(?:im_end|endoftext)\|>")
_TAG_BLOCK = re.compile(
    r"<(think|search|information|answer)>(.*?)</\1>",
    flags=re.DOTALL,
)

_COMMON_FIELDS = {
    "schema",
    "schema_version",
    "record_type",
    "record_id",
    "run_id",
    "stage",
    "sample_id",
    "source_index",
    "question",
    "gold_answers",
    "raw_trajectory",
    "turns",
    "extracted_answer",
    "em",
    "executed_search_count",
    "posthoc_utility",
    "response_tokens",
    "response_clipped",
    "turns_used",
    "invalid_action_count",
}
_V3_FIELDS = {
    "raw_generations",
    "max_action_budget",
    "action_count",
    "policy_token_count",
    "observation_token_count",
    "observation_policy_token_count",
    "info_mask_consistent",
}
_TRAIN_FIELDS = {
    "step",
    "group_uid",
    "group_slot",
    "reward_em_only",
    "train_reward",
    "group_correct_count",
    "group_reward_mean",
    "group_reward_std",
    "sequence_advantage",
}
_EVAL_FIELDS = {"checkpoint_digest"}


def _require_string(value: Any, name: str, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_int(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value,
                  bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


def stable_sample_id(data_source: str, source_split: str,
                     source_index: Any) -> str:
    """Return the stable source identity used by the NQ-small manifest."""
    components = (data_source, source_split, str(source_index))
    for name, component in zip(("data_source", "source_split", "source_index"),
                               components):
        if not _ID_COMPONENT.fullmatch(component):
            raise ValueError(
                f"{name} must match {_ID_COMPONENT.pattern!r}, got {component!r}"
            )
    if isinstance(source_index, bool):
        raise ValueError("source_index must not be boolean")
    return ":".join(components)


def _assistant_text(transcript: str) -> str:
    matches = list(_ASSISTANT_MARKER.finditer(transcript))
    text = transcript[matches[-1].end():] if matches else transcript
    terminal = _TERMINAL_MARKER.search(text)
    return text[:terminal.start()] if terminal else text


def _empty_turn(index: int) -> Dict[str, Any]:
    return {
        "turn": index,
        "think": "",
        "action": None,
        "search_query": None,
        "answer": None,
        "observation": None,
        "invalid_text": [],
        "valid_action": False,
    }


def parse_search_r1_transcript(transcript: str) -> List[Dict[str, Any]]:
    """Parse a decoded Search-R1 interaction into ordered, JSON-safe turns.

    Full chat-template strings are supported: only content after the final
    assistant marker is parsed, so tag examples in the user prompt are ignored.
    Unmatched tags and non-whitespace text outside complete blocks are retained
    in ``invalid_text`` instead of being silently discarded.
    """
    if not isinstance(transcript, str):
        raise TypeError("transcript must be a string")

    text = _assistant_text(transcript)
    turns: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        if current["action"] is None:
            current["action"] = "invalid"
            current["valid_action"] = False
        current["turn"] = len(turns)
        turns.append(current)
        current = None

    def start() -> Dict[str, Any]:
        nonlocal current
        if current is None:
            current = _empty_turn(len(turns))
        return current

    def invalid(fragment: str) -> None:
        if not fragment.strip():
            return
        turn = start()
        turn["invalid_text"].append(fragment.strip())

    position = 0
    for match in _TAG_BLOCK.finditer(text):
        invalid(text[position:match.start()])
        tag, content = match.group(1), match.group(2).strip()

        if tag == "think":
            if current is not None:
                finish()
            start()["think"] = content
        elif tag in ("search", "answer"):
            if current is not None and current["action"] is not None:
                # A preceding invalid fragment represents its own failed action.
                finish()
            turn = start()
            turn["action"] = tag
            turn["valid_action"] = True
            if tag == "search":
                turn["search_query"] = content
            else:
                turn["answer"] = content
        else:
            if (current is None or current["action"] != "search"
                    or current["observation"] is not None):
                finish()
                turn = start()
                turn["action"] = "invalid"
                turn["valid_action"] = False
            else:
                turn = current
            turn["observation"] = content

        position = match.end()

    invalid(text[position:])
    finish()
    return turns


def _validate_turn(turn: Any, index: int) -> None:
    if not isinstance(turn, Mapping):
        raise ValueError(f"turns[{index}] must be an object")
    required = {
        "turn",
        "think",
        "action",
        "search_query",
        "answer",
        "observation",
        "invalid_text",
        "valid_action",
    }
    missing = required - set(turn)
    if missing:
        raise ValueError(
            f"turns[{index}] is missing fields: {sorted(missing)}")
    if turn["turn"] != index:
        raise ValueError(f"turns[{index}].turn must equal {index}")
    _require_string(turn["think"], f"turns[{index}].think", allow_empty=True)
    if turn["action"] not in ("search", "answer", "invalid"):
        raise ValueError(f"turns[{index}].action is invalid")
    for name in ("search_query", "answer", "observation"):
        value = turn[name]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"turns[{index}].{name} must be a string or null")
    if not isinstance(turn["invalid_text"], list) or not all(
            isinstance(item, str) for item in turn["invalid_text"]):
        raise ValueError(
            f"turns[{index}].invalid_text must be a list of strings")
    if not isinstance(turn["valid_action"], bool):
        raise ValueError(f"turns[{index}].valid_action must be boolean")
    if turn["action"] == "search" and turn["search_query"] is None:
        raise ValueError(f"turns[{index}] search action has no search_query")
    if turn["action"] == "answer" and turn["answer"] is None:
        raise ValueError(f"turns[{index}] answer action has no answer")
    if turn["action"] == "invalid" and turn["valid_action"]:
        raise ValueError(f"turns[{index}] invalid action cannot be valid")
    if turn["action"] in ("search", "answer") and not turn["valid_action"]:
        raise ValueError(f"turns[{index}] recognized action must be valid")


def _validate_raw_generation(generation: Any, index: int) -> None:
    if not isinstance(generation, Mapping):
        raise ValueError(f"raw_generations[{index}] must be an object")
    required = {
        "turn",
        "raw_text",
        "raw_token_ids",
        "raw_token_count",
        "action_text",
        "action_token_ids",
        "action_token_count",
        "boundary",
        "tail_dropped",
        "raw_clipped",
    }
    missing = required - set(generation)
    if missing:
        raise ValueError(
            f"raw_generations[{index}] is missing fields: {sorted(missing)}")
    if generation["turn"] != index:
        raise ValueError(f"raw_generations[{index}].turn must equal {index}")
    for name in ("raw_text", "action_text"):
        _require_string(generation[name],
                        f"raw_generations[{index}].{name}",
                        allow_empty=True)
    for name in ("raw_token_ids", "action_token_ids"):
        values = generation[name]
        if (not isinstance(values, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0 for value in values)):
            raise ValueError(
                f"raw_generations[{index}].{name} must contain token IDs")
    _require_int(generation["raw_token_count"],
                 f"raw_generations[{index}].raw_token_count")
    _require_int(generation["action_token_count"],
                 f"raw_generations[{index}].action_token_count")
    raw_ids = generation["raw_token_ids"]
    action_ids = generation["action_token_ids"]
    if generation["raw_token_count"] != len(raw_ids):
        raise ValueError(
            f"raw_generations[{index}].raw_token_count does not match IDs")
    if generation["action_token_count"] != len(action_ids):
        raise ValueError(
            f"raw_generations[{index}].action_token_count does not match IDs")
    if raw_ids[:len(action_ids)] != action_ids:
        raise ValueError(
            f"raw_generations[{index}] action IDs are not a raw-token prefix")
    if generation["boundary"] not in ("tool_call", "answer", "eos", "length"):
        raise ValueError(f"raw_generations[{index}].boundary is invalid")
    for name in ("tail_dropped", "raw_clipped"):
        if not isinstance(generation[name], bool):
            raise ValueError(
                f"raw_generations[{index}].{name} must be boolean")
    if generation["tail_dropped"] != (raw_ids != action_ids):
        raise ValueError(
            f"raw_generations[{index}].tail_dropped does not match IDs")
    located = locate_qwen35_action_boundary(
        generation["action_text"], QWEN35_REASONING_CONTINUATION)
    declared_boundary = generation["boundary"]
    if declared_boundary in ("tool_call", "answer"):
        if located.error is not None or located.boundary != declared_boundary:
            raise ValueError(
                f"raw_generations[{index}] {declared_boundary} boundary is inconsistent"
            )
    elif located.boundary is not None:
        raise ValueError(
            f"raw_generations[{index}] {declared_boundary} boundary is inconsistent"
        )
    if "generation_context" in generation:
        if generation["generation_context"] not in {
                "initial_question", "tool_response", "user_retry",
                "terminal_answer"}:
            raise ValueError(
                f"raw_generations[{index}].generation_context is invalid")
    if ("terminal_generation" in generation
            and not isinstance(generation["terminal_generation"], bool)):
        raise ValueError(
            f"raw_generations[{index}].terminal_generation must be boolean")


def _stable_record_id(record: Mapping[str, Any]) -> str:
    identity: Dict[str, Any] = {
        "record_type": record["record_type"],
        "stage": record["stage"],
        "sample_id": record["sample_id"],
    }
    if record["record_type"] == "train":
        identity.update({
            "step": record["step"],
            "group_uid": record["group_uid"],
            "group_slot": record["group_slot"],
        })
    digest = hashlib.sha256(
        json.dumps(identity,
                   sort_keys=True,
                   separators=(",", ":"),
                   ensure_ascii=True).encode("ascii")).hexdigest()
    return f"trace:{digest[:24]}"


def prepare_trace_record(record: Mapping[str, Any], record_type: str,
                         run_id: str, stage: str) -> Dict[str, Any]:
    """Copy a caller record, add trace identity fields, and validate it."""
    prepared = dict(record)
    prepared.setdefault("schema", TRACE_SCHEMA)
    prepared.setdefault("schema_version", TRACE_SCHEMA_VERSION)
    prepared.setdefault("record_type", record_type)
    prepared.setdefault("run_id", run_id)
    prepared.setdefault("stage", stage)
    identity_fields = {"record_type", "stage", "sample_id"}
    if record_type == "train":
        identity_fields.update({"step", "group_uid", "group_slot"})
    missing_identity = identity_fields - set(prepared)
    if missing_identity:
        raise ValueError(
            f"trace record is missing identity fields: {sorted(missing_identity)}"
        )
    prepared.setdefault("record_id", _stable_record_id(prepared))
    validate_trace_record(prepared, expected_record_type=record_type)
    return prepared


def validate_trace_record(record: Mapping[str, Any],
                          expected_record_type: Optional[str] = None) -> None:
    """Validate one self-describing training or evaluation trace record."""
    if not isinstance(record, Mapping):
        raise ValueError("trace record must be an object")
    record_type = record.get("record_type")
    if record_type not in ("train", "eval"):
        raise ValueError("record_type must be 'train' or 'eval'")
    if expected_record_type is not None and record_type != expected_record_type:
        raise ValueError(
            f"record_type must be {expected_record_type!r}, got {record_type!r}"
        )

    schema_version = record.get("schema_version")
    if (isinstance(schema_version, bool) or not isinstance(schema_version, int)
            or schema_version not in SUPPORTED_TRACE_SCHEMA_VERSIONS):
        raise ValueError("trace record schema_version mismatch")
    required = _COMMON_FIELDS | (_TRAIN_FIELDS
                                 if record_type == "train" else _EVAL_FIELDS)
    if schema_version == TRACE_SCHEMA_VERSION:
        required |= _V3_FIELDS
    missing = required - set(record)
    if missing:
        raise ValueError(f"trace record is missing fields: {sorted(missing)}")
    if record["schema"] != TRACE_SCHEMA:
        raise ValueError("trace record schema mismatch")
    if schema_version == TRACE_SCHEMA_VERSION and "max_searches" in record:
        raise ValueError("v3 trace records must not contain max_searches")

    for name in ("record_id", "run_id", "stage", "sample_id", "question"):
        _require_string(record[name], name)
    if isinstance(record["source_index"],
                  bool) or not isinstance(record["source_index"], (str, int)):
        raise ValueError("source_index must be a string or integer")
    if not isinstance(record["gold_answers"],
                      list) or not record["gold_answers"]:
        raise ValueError("gold_answers must be a non-empty list")
    if not all(
            isinstance(answer, str) and answer.strip()
            for answer in record["gold_answers"]):
        raise ValueError("gold_answers must contain non-empty strings")
    _require_string(record["raw_trajectory"],
                    "raw_trajectory",
                    allow_empty=True)
    if record["extracted_answer"] is not None and not isinstance(
            record["extracted_answer"], str):
        raise ValueError("extracted_answer must be a string or null")

    if not isinstance(record["turns"], list):
        raise ValueError("turns must be a list")
    for index, turn in enumerate(record["turns"]):
        _validate_turn(turn, index)
    _require_int(record["turns_used"], "turns_used")
    if record["turns_used"] != len(record["turns"]):
        raise ValueError("turns_used must equal len(turns)")
    _require_int(record["invalid_action_count"], "invalid_action_count")
    invalid_actions = sum(not turn["valid_action"] for turn in record["turns"])
    if record["invalid_action_count"] != invalid_actions:
        raise ValueError(
            "invalid_action_count must equal the number of invalid parsed turns"
        )

    _require_number(record["em"], "em")
    if float(record["em"]) not in (0.0, 1.0):
        raise ValueError("em must be exactly 0 or 1")
    _require_int(record["executed_search_count"], "executed_search_count")
    _require_number(record["posthoc_utility"], "posthoc_utility")
    _require_int(record["response_tokens"], "response_tokens")
    if not isinstance(record["response_clipped"], bool):
        raise ValueError("response_clipped must be boolean")

    if schema_version == TRACE_SCHEMA_VERSION:
        _require_int(record["max_action_budget"],
                     "max_action_budget",
                     minimum=1)
        _require_int(record["action_count"], "action_count")
        max_generation_count = record["max_action_budget"] + 1
        if record["action_count"] > max_generation_count:
            raise ValueError(
                "action_count exceeds max_action_budget plus terminal generation"
            )
        if record["executed_search_count"] > record["max_action_budget"]:
            raise ValueError(
                "executed_search_count exceeds max_action_budget")
        if record["executed_search_count"] > record["action_count"]:
            raise ValueError("executed_search_count exceeds action_count")
        if not isinstance(record["raw_generations"], list):
            raise ValueError("raw_generations must be a list")
        for index, generation in enumerate(record["raw_generations"]):
            _validate_raw_generation(generation, index)
        if record["action_count"] != len(record["raw_generations"]):
            raise ValueError("action_count must equal len(raw_generations)")
        if ("generation_events" in record
                and len(record["generation_events"]) != record["action_count"]):
            raise ValueError("action_count must equal len(generation_events)")
        terminal_generations = [
            index for index, generation in enumerate(record["raw_generations"])
            if generation.get("terminal_generation", False)
        ]
        generation_events = record.get("generation_events")
        if (generation_events is not None
                and not all(isinstance(event, Mapping)
                            for event in generation_events)):
            raise ValueError("generation_events must contain objects")
        terminal_events = ([] if generation_events is None else [
            index for index, event in enumerate(generation_events)
            if event.get("terminal_generation", False)
        ])
        if record["action_count"] > record["max_action_budget"]:
            expected_terminal = [record["action_count"] - 1]
            if (terminal_generations != expected_terminal
                    or terminal_events != expected_terminal):
                raise ValueError(
                    "the extra generation must be the final terminal generation"
                )
            if generation_events is None:
                raise ValueError(
                    "terminal generation requires auditable generation_events")
            terminal_event = generation_events[-1]
            if (terminal_event.get("terminal_generation") is not True
                    or terminal_event.get("executed_search") is not False):
                raise ValueError(
                    "terminal generation event is not marked or executed retrieval"
                )
            if terminal_event.get("terminal_instruction_applied") is True:
                _validate_terminal_answer_only_event(terminal_event)
        elif terminal_generations or terminal_events:
            raise ValueError(
                "terminal generation is only valid after max_action_budget")
        for name in ("policy_token_count", "observation_token_count",
                     "observation_policy_token_count"):
            _require_int(record[name], name)
        if record["observation_policy_token_count"] != 0:
            raise ValueError("observation_policy_token_count must be zero")
        if record["info_mask_consistent"] is not True:
            raise ValueError("info_mask_consistent must be true")

    if record_type == "train":
        _require_int(record["step"], "step", minimum=1)
        _require_string(record["group_uid"], "group_uid")
        _require_int(record["group_slot"], "group_slot")
        _require_int(record["group_correct_count"], "group_correct_count")
        for name in ("reward_em_only", "train_reward", "group_reward_mean",
                     "group_reward_std", "sequence_advantage"):
            _require_number(record[name], name)
        if record["group_reward_std"] < 0:
            raise ValueError("group_reward_std must be non-negative")
    else:
        _require_string(record["checkpoint_digest"], "checkpoint_digest")


def _validate_terminal_answer_only_event(event: Mapping[str, Any]) -> None:
    """Validate the auditable v4 terminal instruction and allowlist result."""
    expected = {
        "terminal_prompt_version": QWEN35_TERMINAL_PROMPT_VERSION,
        "terminal_prompt_sha256": QWEN35_TERMINAL_PROMPT_SHA256,
        "terminal_prompt_text": QWEN35_TERMINAL_PROMPT,
        "terminal_prompt_policy_token_count": 0,
        "generation_context": "terminal_answer",
        "done": True,
        "executed_search": False,
    }
    for name, value in expected.items():
        if event.get(name) != value:
            raise ValueError(
                f"terminal answer-only event has invalid {name}")
    _require_int(event.get("terminal_followup_token_count"),
                 "terminal_followup_token_count",
                 minimum=1)
    actual_digest = hashlib.sha256(
        event["terminal_prompt_text"].encode("utf-8")).hexdigest()
    if actual_digest != event["terminal_prompt_sha256"]:
        raise ValueError("terminal prompt text does not match its SHA-256")

    requested_action = event.get("requested_action")
    action = event.get("action")
    parse_error = event.get("parse_error")
    rejection = event.get("terminal_rejection_reason")
    valid_action = event.get("valid_action")
    if requested_action == "answer":
        if parse_error is None:
            if (action != "answer" or rejection is not None
                    or valid_action is not True):
                raise ValueError(
                    "terminal answer was not accepted consistently")
        elif (action is not None or not isinstance(parse_error, str)
              or not parse_error or rejection != parse_error
              or valid_action is not False):
            raise ValueError(
                "terminal invalid answer was not rejected consistently")
    elif requested_action == "search":
        if (action is not None
                or parse_error != "search_disallowed_after_budget"
                or rejection != "search_disallowed_after_budget"
                or valid_action is not False):
            raise ValueError("terminal search was not rejected consistently")
    elif (requested_action is not None or action is not None
          or not isinstance(parse_error, str) or not parse_error
          or rejection != parse_error or valid_action is not False):
        raise ValueError("terminal invalid action was not rejected consistently")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value,
                       ensure_ascii=False,
                       sort_keys=True,
                       separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _inspect_jsonl(path: Path, expected_rows: int, record_type: Optional[str],
                   run_id: Optional[str], stage: Optional[str],
                   schema_version: Optional[int] = None) -> Dict[str, Any]:
    _require_int(expected_rows, "expected_rows", minimum=1)
    digest = hashlib.sha256()
    rows = 0
    record_ids: Set[str] = set()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"trace line {line_number} is not newline terminated")
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid JSON on trace line {line_number}") from error
            validate_trace_record(record, expected_record_type=record_type)
            if (schema_version is not None
                    and record["schema_version"] != schema_version):
                raise ValueError(
                    f"trace line {line_number} has a different schema_version")
            if line != _canonical_json(record):
                raise ValueError(
                    f"trace line {line_number} is not canonical JSON")
            if run_id is not None and record["run_id"] != run_id:
                raise ValueError(
                    f"trace line {line_number} has a different run_id")
            if stage is not None and record["stage"] != stage:
                raise ValueError(
                    f"trace line {line_number} has a different stage")
            if record["record_id"] in record_ids:
                raise ValueError(
                    f"duplicate record_id on trace line {line_number}")
            record_ids.add(record["record_id"])
            rows += 1
    if rows != expected_rows:
        raise ValueError(
            f"expected exactly {expected_rows} trace rows, found {rows}")
    return {
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": digest.hexdigest(),
    }


def inspect_trace_jsonl(path: Path,
                        expected_rows: int,
                        record_type: Optional[str] = None,
                        run_id: Optional[str] = None,
                        stage: Optional[str] = None,
                        schema_version: Optional[int] = None) -> Dict[str, Any]:
    """Re-validate a complete JSONL file and return its artifact metadata."""
    return _inspect_jsonl(Path(path), expected_rows, record_type, run_id,
                          stage, schema_version)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.",
                                                  suffix=".tmp",
                                                  dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class TraceJsonlWriter:
    """Append validated records to a partial file and atomically finalize it."""

    def __init__(self, path: Path, record_type: str, expected_rows: int,
                 run_id: str, stage: str,
                 schema_version: int = TRACE_SCHEMA_VERSION) -> None:
        if record_type not in ("train", "eval"):
            raise ValueError("record_type must be 'train' or 'eval'")
        _require_int(expected_rows, "expected_rows", minimum=1)
        _require_string(run_id, "run_id")
        _require_string(stage, "stage")
        if (isinstance(schema_version, bool) or not isinstance(schema_version, int)
                or schema_version not in PUBLISHABLE_TRACE_SCHEMA_VERSIONS):
            raise ValueError(
                "schema_version must be an explicitly publishable version")

        self.path = Path(path)
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        self.manifest_path = self.path.with_suffix(".manifest.json")
        self.manifest_sha256_path = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".sha256")
        self.record_type = record_type
        self.expected_rows = expected_rows
        self.run_id = run_id
        self.stage = stage
        self.schema_version = schema_version
        self._record_ids: Set[str] = set()
        self._rows = 0
        self._closed = False
        self._finalized = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        for artifact in (self.path, self.partial_path, self.manifest_path,
                         self.manifest_sha256_path):
            if artifact.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing trace artifact: {artifact}"
                )
        self._handle = self.partial_path.open("x",
                                              encoding="utf-8",
                                              newline="\n")

    @property
    def rows(self) -> int:
        return self._rows

    def append(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("trace writer is closed")
        if self._rows >= self.expected_rows:
            raise ValueError(
                f"cannot append more than expected_rows={self.expected_rows}")
        candidate = dict(record)
        candidate.setdefault("schema_version", self.schema_version)
        if self.schema_version == 1 and _V3_FIELDS.intersection(candidate):
            raise ValueError(
                "v1 trace writer cannot publish v3-only audit fields")
        prepared = prepare_trace_record(candidate, self.record_type,
                                        self.run_id, self.stage)
        if prepared["schema_version"] != self.schema_version:
            raise ValueError(
                "trace record schema_version does not match its writer")
        record_id = prepared["record_id"]
        if record_id in self._record_ids:
            raise ValueError(f"duplicate trace record_id: {record_id}")
        payload = _canonical_json(prepared).decode("utf-8")
        self._handle.write(payload)
        self._handle.flush()
        self._record_ids.add(record_id)
        self._rows += 1
        return prepared

    def sync(self) -> None:
        """Flush a completed training step through the operating system."""
        if self._closed:
            raise RuntimeError("trace writer is closed")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        """Close without publishing; the ``.partial`` file remains inspectable."""
        if not self._closed:
            self._handle.close()
            self._closed = True

    def finalize(self) -> Dict[str, Any]:
        """Validate exact cardinality, publish JSONL, and atomically write hashes."""
        if self._finalized:
            raise RuntimeError("trace writer is already finalized")
        if not self._closed:
            self.sync()
            self.close()

        artifact = _inspect_jsonl(
            self.partial_path,
            self.expected_rows,
            self.record_type,
            self.run_id,
            self.stage,
            self.schema_version,
        )
        if self.path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing trace: {self.path}")
        os.replace(str(self.partial_path), str(self.path))

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": self.stage,
            "artifact": {
                "path": self.path.name,
                "record_schema": TRACE_SCHEMA,
                "record_schema_version": self.schema_version,
                "record_type": self.record_type,
                "expected_rows": self.expected_rows,
                **artifact,
            },
        }
        manifest_bytes = _canonical_json(manifest)
        _atomic_write(self.manifest_path, manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar = f"{manifest_digest}  {self.manifest_path.name}\n".encode(
            "ascii")
        _atomic_write(self.manifest_sha256_path, sidecar)
        self._finalized = True
        return manifest


def verify_trace_manifest(
        manifest_path: Path,
        expected_rows: Optional[int] = None) -> Dict[str, Any]:
    """Verify a finalized manifest, sidecar, JSONL hash, schema, and row count."""
    manifest_path = Path(manifest_path)
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    manifest_bytes = manifest_path.read_bytes()
    sidecar = sidecar_path.read_text(encoding="ascii")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    expected_sidecar = f"{digest}  {manifest_path.name}\n"
    if sidecar != expected_sidecar:
        raise ValueError("trace manifest SHA256 sidecar mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trace manifest is not valid JSON") from error
    if manifest_bytes != _canonical_json(manifest):
        raise ValueError("trace manifest is not canonical JSON")
    if set(manifest) != {
            "schema", "schema_version", "run_id", "stage", "artifact"
    }:
        raise ValueError("trace manifest has missing or unknown fields")
    if (manifest["schema"] != MANIFEST_SCHEMA
            or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION):
        raise ValueError("trace manifest schema mismatch")

    artifact = manifest["artifact"]
    artifact_fields = {
        "path",
        "record_schema",
        "record_schema_version",
        "record_type",
        "expected_rows",
        "bytes",
        "rows",
        "sha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != artifact_fields:
        raise ValueError(
            "trace manifest artifact has missing or unknown fields")
    if (artifact["record_schema"] != TRACE_SCHEMA
            or artifact["record_schema_version"]
            not in SUPPORTED_TRACE_SCHEMA_VERSIONS):
        raise ValueError("trace record schema mismatch in manifest")
    if artifact["rows"] != artifact["expected_rows"]:
        raise ValueError("trace manifest rows do not match expected_rows")
    if expected_rows is not None and artifact["expected_rows"] != expected_rows:
        raise ValueError(
            f"expected exactly {expected_rows} trace rows, manifest declares "
            f"{artifact['expected_rows']}")

    trace_path = manifest_path.parent / artifact["path"]
    inspected = _inspect_jsonl(
        trace_path,
        artifact["expected_rows"],
        artifact["record_type"],
        manifest["run_id"],
        manifest["stage"],
        artifact["record_schema_version"],
    )
    if inspected != {
            "bytes": artifact["bytes"],
            "rows": artifact["rows"],
            "sha256": artifact["sha256"],
    }:
        raise ValueError("trace artifact metadata does not match manifest")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Expose final trace verification to the AutoDL shell gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="verify a trace manifest and its JSONL artifact")
    verify_parser.add_argument("--manifest", required=True, type=Path)
    verify_parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args(argv)

    if args.command == "verify":
        manifest = verify_trace_manifest(args.manifest, args.expected_rows)
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
