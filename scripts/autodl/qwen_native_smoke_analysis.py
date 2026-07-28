#!/usr/bin/env python3
"""Fail closed on the registered two-step Qwen native training smoke."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
import sys
import tempfile
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REWARD_SCORE_DIR = REPOSITORY_ROOT / "verl" / "utils" / "reward_score"
AUTODL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(REWARD_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_SCORE_DIR))
if str(AUTODL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AUTODL_SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import qa_em  # noqa: E402
from search_r1.llm_agent.tool_protocol import (  # noqa: E402
    QWEN35_TERMINAL_PROMPT,
    QWEN35_TERMINAL_PROMPT_SHA256,
    QWEN35_TERMINAL_PROMPT_VERSION,
)
from wandb_history import scan_offline_run  # noqa: E402


SCHEMA = "search-r1.qwen-native-smoke-decision"
SCHEMA_VERSION = 3
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WANDB_METRIC_REL_TOL = 1e-6
WANDB_METRIC_ABS_TOL = 1e-6
ACTOR_METRICS = (
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    "actor/ppo_kl",
)
NATIVE_EXACT_ONE = (
    "native_batch/contract_valid",
    "native_batch/info_loss_mask_match",
    "native_batch/policy_mask_subset",
    "native_batch/old_log_prob_finite_ratio",
    "native_batch/advantage_finite_ratio",
    "native_batch/reward_finite_ratio",
)
NATIVE_POSITIVE = (
    "native_batch/policy_tokens",
    "native_batch/policy_tokens_min_per_trajectory",
    "native_batch/policy_coverage",
)
NATIVE_NONNEGATIVE = (
    "native_batch/nonzero_advantage_tokens",
    "native_batch/advantage_abs_max",
)
MAX_ACTION_BUDGET = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or symlinked: {path}")


def _file_identity(path: Path, label: str) -> tuple[int, int, int, int]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"unterminated {label} row: {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid {label} JSON: {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{label} row must be an object: {path}:{line_number}")
            records.append(record)
    return records


def _materialized_native_question(question: str) -> str:
    question = re.sub(r"\s+", " ", question).strip()
    return question if question.endswith("?") else question + "?"


def _catalog(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(path, "catalog"):
        sample_id = record.get("sample_id")
        question = record.get("question")
        answers = record.get("golden_answers")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in records
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) and answer.strip() for answer in answers)
        ):
            raise ValueError("catalog identity or answer contract is invalid")
        records[sample_id] = {
            "question": _materialized_native_question(question),
            "gold_answers": list(answers),
        }
    return records


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _metric_lines(path: Path) -> dict[int, dict[str, list[float]]]:
    _regular_file(path, "train log")
    metrics: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        marker = line.find("step:")
        if marker < 0:
            continue
        fields = line[marker:].split(" - ")
        if not fields or not fields[0].startswith("step:"):
            continue
        try:
            step = int(fields[0].split(":", 1)[1])
        except ValueError:
            continue
        for field in fields[1:]:
            if ":" not in field:
                continue
            key, raw_value = field.rsplit(":", 1)
            try:
                value = float(raw_value)
            except ValueError:
                continue
            metrics[step][key].append(value)
    return metrics


def _metric_check(
    metrics: dict[int, dict[str, list[float]]],
    steps: range,
    key: str,
    predicate,
) -> bool:
    for step in steps:
        values = metrics.get(step, {}).get(key, [])
        if not values or not all(math.isfinite(value) and predicate(value) for value in values):
            return False
    return True


def _wandb_actor_metrics(
    scan: dict[str, Any], steps: range
) -> dict[int, dict[str, list[Any]]]:
    metrics: dict[int, dict[str, list[Any]]] = {
        step: {key: [] for key in ACTOR_METRICS} for step in steps
    }
    for record in scan["history"]:
        step = record["step"]
        if step not in metrics:
            continue
        for key in ACTOR_METRICS:
            if key in record["metrics"]:
                metrics[step][key].append(record["metrics"][key])
    return metrics


def _wandb_metrics_match_log(
    wandb_metrics: dict[int, dict[str, list[Any]]],
    log_metrics: dict[int, dict[str, list[float]]],
    steps: range,
) -> bool:
    logged_actor_steps = sorted(
        step
        for step, metrics in log_metrics.items()
        if any(metrics.get(key) for key in ACTOR_METRICS)
    )
    if logged_actor_steps != list(steps):
        return False
    for step in steps:
        for key in ACTOR_METRICS:
            history_values = wandb_metrics[step][key]
            logged_values = log_metrics.get(step, {}).get(key, [])
            if len(history_values) != len(logged_values) or not history_values:
                return False
            for history_value, logged_value in zip(history_values, logged_values):
                if (
                    isinstance(history_value, bool)
                    or not isinstance(history_value, (int, float))
                    or not math.isfinite(float(history_value))
                    or not math.isfinite(logged_value)
                    or not math.isclose(
                        float(history_value),
                        logged_value,
                        rel_tol=WANDB_METRIC_REL_TOL,
                        abs_tol=WANDB_METRIC_ABS_TOL,
                    )
                ):
                    return False
    return True


def _json_safe_metric(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe_metric(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_metric(item) for item in value]
    return value


def _terminal_diagnostics(record: Mapping[str, Any], location: str) -> dict[str, int]:
    """Replay the registered answer-only generation after budget exhaustion."""
    events = record.get("generation_events")
    action_count = record.get("action_count")
    if (
        record.get("max_action_budget") != MAX_ACTION_BUDGET
        or isinstance(action_count, bool)
        or not isinstance(action_count, int)
        or action_count < 0
        or action_count > MAX_ACTION_BUDGET + 1
        or not isinstance(events, list)
        or action_count != len(events)
        or not all(isinstance(event, Mapping) for event in events)
        or any(
            not isinstance(event.get("terminal_generation", False), bool)
            for event in events
        )
    ):
        raise ValueError(f"{location} has an invalid action-budget contract")

    terminal_indices = [
        index
        for index, event in enumerate(events)
        if event.get("terminal_generation") is True
    ]
    diagnostics = {
        "terminal_generation_count": len(terminal_indices),
        "terminal_instruction_applied_count": 0,
        "terminal_answer_count": 0,
        "terminal_requested_search_count": 0,
        "terminal_invalid_count": 0,
        "terminal_accepted_search_count": 0,
        "terminal_executed_search_count": 0,
        "terminal_prompt_policy_token_count": 0,
    }
    if action_count <= MAX_ACTION_BUDGET:
        if terminal_indices:
            raise ValueError(
                f"{location} has a terminal generation before budget exhaustion"
            )
        final_event = events[-1] if events else None
        if (
            not isinstance(final_event, Mapping)
            or final_event.get("requested_action") != "answer"
            or final_event.get("action") != "answer"
            or final_event.get("parse_error") is not None
            or final_event.get("valid_action") is not True
            or final_event.get("done") is not True
            or final_event.get("executed_search") is not False
        ):
            raise ValueError(
                f"{location} does not end with a valid early answer or "
                "the registered terminal generation"
            )
        return diagnostics
    if terminal_indices != [action_count - 1]:
        raise ValueError(f"{location} terminal generation is not uniquely final")

    terminal = events[-1]
    prompt_policy_tokens = terminal.get("terminal_prompt_policy_token_count")
    followup_tokens = terminal.get("terminal_followup_token_count")
    if (
        terminal.get("terminal_prompt_version") != QWEN35_TERMINAL_PROMPT_VERSION
        or terminal.get("terminal_prompt_sha256") != QWEN35_TERMINAL_PROMPT_SHA256
        or terminal.get("terminal_prompt_text") != QWEN35_TERMINAL_PROMPT
        or terminal.get("generation_context") != "terminal_answer"
        or terminal.get("done") is not True
        or not isinstance(terminal.get("executed_search"), bool)
        or isinstance(prompt_policy_tokens, bool)
        or not isinstance(prompt_policy_tokens, int)
        or prompt_policy_tokens < 0
        or isinstance(followup_tokens, bool)
        or not isinstance(followup_tokens, int)
        or followup_tokens < 1
    ):
        raise ValueError(f"{location} has an invalid terminal prompt contract")
    if hashlib.sha256(
        terminal["terminal_prompt_text"].encode("utf-8")
    ).hexdigest() != terminal["terminal_prompt_sha256"]:
        raise ValueError(f"{location} terminal prompt digest does not replay")

    diagnostics["terminal_instruction_applied_count"] = int(
        terminal.get("terminal_instruction_applied") is True
    )
    diagnostics["terminal_prompt_policy_token_count"] = prompt_policy_tokens
    diagnostics["terminal_executed_search_count"] = int(
        terminal.get("executed_search") is True
    )
    diagnostics["terminal_accepted_search_count"] = int(
        terminal.get("action") == "search" and terminal.get("parse_error") is None
    )

    requested_action = terminal.get("requested_action")
    action = terminal.get("action")
    parse_error = terminal.get("parse_error")
    rejection = terminal.get("terminal_rejection_reason")
    valid_action = terminal.get("valid_action")
    if requested_action == "answer":
        if parse_error is None:
            if action != "answer" or rejection is not None or valid_action is not True:
                raise ValueError(f"{location} terminal answer is inconsistent")
            diagnostics["terminal_answer_count"] = 1
        elif (
            action is not None
            or not isinstance(parse_error, str)
            or not parse_error
            or rejection != parse_error
            or valid_action is not False
        ):
            raise ValueError(f"{location} terminal answer rejection is inconsistent")
        else:
            diagnostics["terminal_invalid_count"] = 1
    elif requested_action == "search":
        diagnostics["terminal_requested_search_count"] = 1
        rejected = (
            action is None
            and parse_error == "search_disallowed_after_budget"
            and rejection == "search_disallowed_after_budget"
            and valid_action is False
        )
        coherently_accepted = (
            action == "search"
            and parse_error is None
            and rejection is None
            and valid_action is True
        )
        if not (rejected or coherently_accepted):
            raise ValueError(f"{location} terminal search result is inconsistent")
    elif (
        requested_action is not None
        or action is not None
        or not isinstance(parse_error, str)
        or not parse_error
        or rejection != parse_error
        or valid_action is not False
    ):
        raise ValueError(f"{location} terminal invalid result is inconsistent")
    else:
        diagnostics["terminal_invalid_count"] = 1
    return diagnostics


def _structured_wandb_metrics(
    scan: dict[str, Any], actor_metrics: dict[int, dict[str, list[Any]]]
) -> dict[str, Any]:
    return {
        "actor_metrics": {
            str(step): {
                key: [_json_safe_metric(value) for value in values]
                for key, values in metrics.items()
            }
            for step, metrics in actor_metrics.items()
        },
        "exit_codes": scan["exit_codes"],
        "files": scan["file_count"],
        "history_records": scan["history_count"],
        "history_steps": scan["history_steps"],
        "last_record_type": scan["last_record_type"],
        "record_counts": scan["record_counts"],
        "run_file": scan["run_file"],
        "run_file_sha256": scan["run_file_sha256"],
        "summary": _json_safe_metric(scan["summary"]),
        "summary_records": scan["summary_count"],
        "wandb_version": scan["wandb_version"],
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_steps != 2 or args.group_size != 5 or args.batch_size != 8:
        raise ValueError("the registered smoke shape is exactly 2 steps x 8 x 5")
    protected_inputs = {
        "catalog": args.catalog,
        "log": args.log,
        "trace": args.trace,
    }
    for label, path in protected_inputs.items():
        _regular_file(path, label)
    input_identities = {
        label: _file_identity(path, label)
        for label, path in protected_inputs.items()
    }
    input_digests = {
        label: _sha256_file(path) for label, path in protected_inputs.items()
    }
    expected_rows = args.expected_steps * args.batch_size * args.group_size
    records = _load_jsonl(args.trace, "training trace")
    if len(records) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} smoke trajectories, found {len(records)}"
        )
    catalog = _catalog(args.catalog)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    positive_count = 0
    nonzero_trace_advantages = 0
    terminal_metrics: dict[str, int] = defaultdict(int)

    for index, record in enumerate(records):
        location = f"trace row {index + 1}"
        if record.get("record_type") != "train" or record.get("stage") != "smoke":
            raise ValueError(f"{location} has the wrong trace identity")
        step = record.get("step")
        sample_id = record.get("sample_id")
        slot = record.get("group_slot")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step not in range(1, args.expected_steps + 1)
            or not isinstance(sample_id, str)
            or sample_id not in catalog
            or isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot not in range(args.group_size)
        ):
            raise ValueError(f"{location} has an invalid group identity")
        source = catalog[sample_id]
        if record.get("question", "").strip() != source["question"]:
            raise ValueError(f"{location} question differs from the sealed catalog")
        if record.get("gold_answers") != source["gold_answers"]:
            raise ValueError(f"{location} gold answers differ from the sealed catalog")
        prediction = record.get("extracted_answer")
        if prediction is not None and not isinstance(prediction, str):
            raise ValueError(f"{location} extracted answer is invalid")
        strict_em = int(qa_em.em_check(prediction or "", source["gold_answers"]))
        if record.get("em") != strict_em:
            raise ValueError(f"{location} strict EM does not replay")
        reward_em = _finite_number(record.get("reward_em_only"), f"{location} EM reward")
        train_reward = _finite_number(record.get("train_reward"), f"{location} reward")
        advantage = _finite_number(
            record.get("sequence_advantage"), f"{location} advantage"
        )
        if reward_em != strict_em or train_reward != strict_em:
            raise ValueError(f"{location} is not using the registered EM-only smoke reward")
        positive_count += strict_em
        nonzero_trace_advantages += int(abs(advantage) > 0.0)
        expected_uid = f"{step}:{sample_id}"
        if record.get("group_uid") != expected_uid:
            raise ValueError(f"{location} group UID is inconsistent")
        for name, value in _terminal_diagnostics(record, location).items():
            terminal_metrics[name] += value
        groups[(step, sample_id)].append(record)

    expected_groups = args.expected_steps * args.batch_size
    if len(groups) != expected_groups:
        raise ValueError(f"expected {expected_groups} smoke groups, found {len(groups)}")
    mixed_groups = 0
    groups_per_step: dict[int, int] = defaultdict(int)
    for (step, _), group in groups.items():
        groups_per_step[step] += 1
        slots = sorted(record["group_slot"] for record in group)
        if slots != list(range(args.group_size)):
            raise ValueError("smoke group slots are incomplete or duplicated")
        correct = sum(int(record["em"]) for record in group)
        rewards = [float(record["train_reward"]) for record in group]
        reward_std = statistics.stdev(rewards)
        for record in group:
            if record.get("group_correct_count") != correct:
                raise ValueError("trace group_correct_count does not replay")
            recorded_std = _finite_number(
                record.get("group_reward_std"), "trace group reward std"
            )
            if not math.isclose(recorded_std, reward_std, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("trace group_reward_std does not replay")
        mixed_groups += int(0 < correct < args.group_size and reward_std > 0.0)
    if set(groups_per_step) != set(range(1, args.expected_steps + 1)) or any(
        groups_per_step[step] != args.batch_size for step in groups_per_step
    ):
        raise ValueError("smoke groups are not batch-aligned by step")

    metrics = _metric_lines(args.log)
    steps = range(1, args.expected_steps + 1)
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, observed: Any) -> None:
        checks[name] = {"observed": observed, "passed": bool(passed)}

    add("strict_em_positive", positive_count > 0, positive_count)
    add("mixed_reward_group", mixed_groups > 0, mixed_groups)
    add(
        "nonzero_trace_advantage",
        nonzero_trace_advantages > 0,
        nonzero_trace_advantages,
    )
    terminal_generations = terminal_metrics["terminal_generation_count"]
    terminal_answer_rate = (
        terminal_metrics["terminal_answer_count"] / terminal_generations
        if terminal_generations
        else 1.0
    )
    terminal_requested_search_rate = (
        terminal_metrics["terminal_requested_search_count"] / terminal_generations
        if terminal_generations
        else 0.0
    )
    add(
        "terminal_instruction_applied_count",
        terminal_metrics["terminal_instruction_applied_count"]
        == terminal_generations,
        terminal_metrics["terminal_instruction_applied_count"],
    )
    add(
        "terminal_prompt_policy_token_count",
        terminal_metrics["terminal_prompt_policy_token_count"] == 0,
        terminal_metrics["terminal_prompt_policy_token_count"],
    )
    add(
        "terminal_accepted_search_count",
        terminal_metrics["terminal_accepted_search_count"] == 0,
        terminal_metrics["terminal_accepted_search_count"],
    )
    add(
        "terminal_executed_search_count",
        terminal_metrics["terminal_executed_search_count"] == 0,
        terminal_metrics["terminal_executed_search_count"],
    )
    for key in ACTOR_METRICS:
        add(
            f"finite_{key.replace('/', '_')}",
            _metric_check(metrics, steps, key, lambda _: True),
            {str(step): metrics.get(step, {}).get(key, []) for step in steps},
        )
    for key in NATIVE_EXACT_ONE:
        add(
            key.replace("/", "_"),
            _metric_check(metrics, steps, key, lambda value: value == 1.0),
            {str(step): metrics.get(step, {}).get(key, []) for step in steps},
        )
    for key in NATIVE_POSITIVE:
        add(
            key.replace("/", "_"),
            _metric_check(metrics, steps, key, lambda value: value > 0.0),
            {str(step): metrics.get(step, {}).get(key, []) for step in steps},
        )
    for key in NATIVE_NONNEGATIVE:
        add(
            key.replace("/", "_"),
            _metric_check(metrics, steps, key, lambda value: value >= 0.0),
            {str(step): metrics.get(step, {}).get(key, []) for step in steps},
        )
    wandb_scan = scan_offline_run(args.wandb_dir, metric_keys=ACTOR_METRICS)
    wandb_actor_metrics = _wandb_actor_metrics(wandb_scan, steps)
    expected_history_steps = list(steps)
    observed_history_steps = wandb_scan["history_steps"]
    add(
        "wandb_offline_history",
        wandb_scan["history_count"] > 0,
        wandb_scan["history_count"],
    )
    add(
        "wandb_history_steps",
        observed_history_steps == expected_history_steps,
        observed_history_steps,
    )
    add(
        "wandb_actor_metrics_match_log",
        _wandb_metrics_match_log(wandb_actor_metrics, metrics, steps),
        {
            str(step): {
                key: [_json_safe_metric(value) for value in values]
                for key, values in wandb_actor_metrics[step].items()
            }
            for step in steps
        },
    )
    add(
        "wandb_summary_present",
        wandb_scan["summary_count"] > 0,
        wandb_scan["summary_count"],
    )
    add(
        "wandb_exit_zero",
        wandb_scan["exit_codes"] == [0]
        and wandb_scan["last_record_type"] == "exit",
        {
            "codes": wandb_scan["exit_codes"],
            "last_record_type": wandb_scan["last_record_type"],
        },
    )
    for label, path in protected_inputs.items():
        identity_before_hash = _file_identity(path, label)
        digest_after_read = _sha256_file(path)
        identity_after_hash = _file_identity(path, label)
        if (
            identity_before_hash != input_identities[label]
            or identity_after_hash != identity_before_hash
            or digest_after_read != input_digests[label]
        ):
            raise ValueError("smoke analysis input changed while it was read")

    decision = "GO" if all(check["passed"] for check in checks.values()) else "NO-GO"
    return {
        "checks": checks,
        "decision": decision,
        "inputs": {
            "catalog_sha256": input_digests["catalog"],
            "log_sha256": input_digests["log"],
            "trace_sha256": input_digests["trace"],
            "wandb_tree_sha256": wandb_scan["tree_sha256"],
        },
        "metrics": {
            "groups": len(groups),
            "mixed_groups": mixed_groups,
            "nonzero_trace_advantages": nonzero_trace_advantages,
            "strict_em_positive_count": positive_count,
            **dict(terminal_metrics),
            "terminal_answer_rate": terminal_answer_rate,
            "terminal_requested_search_rate": terminal_requested_search_rate,
            "trajectories": len(records),
            "wandb": _structured_wandb_metrics(wandb_scan, wandb_actor_metrics),
        },
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = analyze(args)
        payload = (
            json.dumps(
                result,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        _atomic_write(args.output, payload)
    except (OSError, ValueError) as error:
        print(f"Qwen native smoke analysis failed: {error}", file=sys.stderr)
        return 1
    print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
