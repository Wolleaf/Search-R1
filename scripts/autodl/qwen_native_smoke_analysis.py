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
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REWARD_SCORE_DIR = REPOSITORY_ROOT / "verl" / "utils" / "reward_score"
if str(REWARD_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_SCORE_DIR))

import qa_em  # noqa: E402


SCHEMA = "search-r1.qwen-native-smoke-decision"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or symlinked: {path}")


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
            "question": question.strip(),
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


def _wandb_tree(path: Path) -> tuple[int, str]:
    if not path.is_dir() or path.is_symlink():
        return 0, ""
    root = path.resolve()
    entries: list[tuple[str, str]] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            try:
                target = item.resolve(strict=True)
            except OSError as error:
                raise ValueError(f"broken WandB symlink: {item}") from error
            if target != root and root not in target.parents:
                raise ValueError(f"WandB symlink escapes its run directory: {item}")
            continue
        if item.is_file():
            relative = item.relative_to(path).as_posix()
            entries.append((relative, _sha256_file(item)))
    if not entries or not any(name.endswith(".wandb") for name, _ in entries):
        return len(entries), ""
    digest = hashlib.sha256()
    for name, file_digest in entries:
        digest.update(f"{file_digest}  {name}\n".encode("utf-8"))
    return len(entries), digest.hexdigest()


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


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_steps != 2 or args.group_size != 5 or args.batch_size != 8:
        raise ValueError("the registered smoke shape is exactly 2 steps x 8 x 5")
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
    wandb_files, wandb_digest = _wandb_tree(args.wandb_dir)
    add("wandb_offline_history", bool(wandb_digest), wandb_files)

    decision = "GO" if all(check["passed"] for check in checks.values()) else "NO-GO"
    return {
        "checks": checks,
        "decision": decision,
        "inputs": {
            "catalog_sha256": _sha256_file(args.catalog),
            "log_sha256": _sha256_file(args.log),
            "trace_sha256": _sha256_file(args.trace),
            "wandb_tree_sha256": wandb_digest,
        },
        "metrics": {
            "groups": len(groups),
            "mixed_groups": mixed_groups,
            "nonzero_trace_advantages": nonzero_trace_advantages,
            "strict_em_positive_count": positive_count,
            "trajectories": len(records),
            "wandb_files": wandb_files,
        },
        "schema": SCHEMA,
        "schema_version": 1,
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
