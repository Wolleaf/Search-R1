#!/usr/bin/env python3
"""Validate and pair control with one or two cost-aware evaluation traces."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REWARD_SCORE_DIR = REPOSITORY_ROOT / "verl" / "utils" / "reward_score"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(REWARD_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_SCORE_DIR))

import qa_em  # noqa: E402
from search_r1.trajectory_trace import validate_trace_record  # noqa: E402


REQUIRED_FIELDS = frozenset({
    "stage",
    "sample_id",
    "question",
    "gold_answers",
    "extracted_answer",
    "em",
    "executed_search_count",
    "posthoc_utility",
})
PASSTHROUGH_FIELDS = ("checkpoint_digest", "raw_trajectory", "turns")
NATIVE_REPORT_FIELDS = (
    "response_tokens",
    "action_count",
    "invalid_action_count",
    "response_clipped",
    "generation_clipped_count",
)
EFFICIENCY_ROLE_ARGUMENTS = (
    ("control", "B / Control"),
    ("cost_aware_old", "C-old / Cost-aware"),
    ("cost_aware_gated", "C-gated / Cost-aware gated"),
)
CAPABILITY_ROLE_ARGUMENTS = (
    ("parent", "A / Parent (post-trained)"),
    ("reproduced", "R / Reproduced"),
)
CATEGORY_ORDER = (
    "both_correct",
    "baseline_correct_candidate_wrong",
    "baseline_wrong_candidate_correct",
    "both_wrong",
)
FORMAL_STAGE_NAMES = {
    "control": "qwen_native_b",
    "cost_aware_gated": "qwen_native_c",
}
V4_PROMPT_VERSION = "qwen35-native-search-v4-terminal-answer-only"
V3_PROMPT_VERSION = "qwen35-native-search-v3-original-aligned"
V2_PROMPT_VERSION = "qwen35-native-search-v2-answer-tag"
TERMINAL_PROMPT_VERSION = "qwen35-terminal-answer-v1"
TERMINAL_PROMPT_TEXT = (
    "The search budget is exhausted. You must not call the search tool again. "
    "Using only the question and information already available, give your best "
    "answer even if uncertain. After reasoning, output exactly one concise "
    "final answer inside <answer> and </answer>, with no text after </answer>."
)
TERMINAL_PROMPT_SHA256 = (
    "afc18b79afaafccece6927aec5ccd7898ef2ae17766bce7ffda244eef388d7f2"
)
V4_EVAL_ARTIFACTS = {
    "val": {
        "file": "val_128.parquet",
        "rows": 128,
        "stage_suffix": "val",
    },
    "nq_test_eval": {
        "file": "nq_test_128_native_v4.parquet",
        "rows": 128,
        "stage_suffix": "nq_test",
    },
    "multihop_eval": {
        "file": "multihop_eval_256_native_v4.parquet",
        "rows": 256,
        "stage_suffix": "multihop",
    },
}
V3_EVAL_ARTIFACTS = {
    "val": {
        "file": "val_128.parquet",
        "rows": 128,
        "stage_suffix": "val",
    },
    "nq_test_eval": {
        "file": "nq_test_128_native_v3.parquet",
        "rows": 128,
        "stage_suffix": "nq_test",
    },
    "multihop_eval": {
        "file": "multihop_eval_256_native_v3.parquet",
        "rows": 256,
        "stage_suffix": "multihop",
    },
}
ENDPOINT_EVAL_CONTRACT = {
    "group_size": 1,
    "rollouts_per_question": 1,
    "do_sample": False,
    "decoding": "greedy",
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 42,
    "group_slot": 0,
    "preserve_artifact_order": True,
    "pairing_key": "typed_sample_id",
}
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
QUESTION_PATTERN = re.compile(r"Question:\s*(.+?)\s*$", flags=re.DOTALL)


def _sample_key(sample_id: str | int) -> tuple[str, str | int]:
    """Keep integer and string IDs distinct while making them sortable."""
    if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
        raise ValueError(f"sample_id must be a string or integer, found {sample_id!r}")
    if isinstance(sample_id, str):
        if not sample_id:
            raise ValueError("sample_id must not be empty")
        return ("str", sample_id)
    return ("int", sample_id)


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def _comparison_mode(args: argparse.Namespace) -> str:
    efficiency_fields = (
        "control",
        "cost_aware_old",
        "cost_aware_gated",
        "expected_control_checkpoint_digest",
        "expected_cost_aware_gated_checkpoint_digest",
    )
    capability_fields = (
        "parent",
        "reproduced",
        "expected_parent_checkpoint_digest",
        "expected_reproduced_checkpoint_digest",
    )
    has_efficiency = any(getattr(args, field, None) is not None
                         for field in efficiency_fields)
    has_capability = any(getattr(args, field, None) is not None
                         for field in capability_fields)
    if has_efficiency == has_capability:
        raise ValueError("select exactly one of the B/C efficiency or A/R capability modes")
    return "efficiency" if has_efficiency else "capability"


def _active_role_arguments(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    mode = _comparison_mode(args)
    if mode == "capability":
        if getattr(args, "parent", None) is None:
            raise ValueError("parent trace is required")
        if getattr(args, "reproduced", None) is None:
            raise ValueError("reproduced trace is required")
        return CAPABILITY_ROLE_ARGUMENTS
    if getattr(args, "control", None) is None:
        raise ValueError("control trace is required")
    if getattr(args, "cost_aware_gated", None) is None:
        raise ValueError("cost_aware_gated trace is required")
    roles = [EFFICIENCY_ROLE_ARGUMENTS[0]]
    if getattr(args, "cost_aware_old", None) is not None:
        roles.append(EFFICIENCY_ROLE_ARGUMENTS[1])
    roles.append(EFFICIENCY_ROLE_ARGUMENTS[2])
    return tuple(roles)


def _load_formal_contract(args: argparse.Namespace) -> dict[str, Any] | None:
    comparison_mode = _comparison_mode(args)
    if comparison_mode == "efficiency":
        digest_arguments = {
            "control": getattr(args, "expected_control_checkpoint_digest", None),
            "cost_aware_gated": getattr(
                args, "expected_cost_aware_gated_checkpoint_digest", None
            ),
        }
    else:
        digest_arguments = {
            "parent": getattr(args, "expected_parent_checkpoint_digest", None),
            "reproduced": getattr(
                args, "expected_reproduced_checkpoint_digest", None
            ),
        }
    values = {
        "data_manifest": getattr(args, "data_manifest", None),
        **digest_arguments,
    }
    supplied = {name: value is not None for name, value in values.items()}
    if any(supplied.values()) and not all(supplied.values()):
        missing = sorted(name for name, present in supplied.items() if not present)
        raise ValueError(
            "formal paired evaluation arguments are all-or-none; missing "
            + ", ".join(missing)
        )
    if not any(supplied.values()):
        if getattr(args, "eval_artifact", None) is not None:
            raise ValueError("--eval-artifact requires the formal evaluation arguments")
        if comparison_mode == "capability":
            raise ValueError("A/R capability evaluation requires a formal native contract")
        return None
    if getattr(args, "cost_aware_old", None) is not None:
        raise ValueError("formal paired evaluation accepts only control and cost_aware_gated")

    checkpoint_digests = {
        role: _require_sha256(value, f"expected {role} checkpoint digest")
        for role, value in digest_arguments.items()
    }
    manifest_path = Path(values["data_manifest"])
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid data manifest JSON in {manifest_path}: {error.msg}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("formal data manifest must be a JSON object")
    manifest_schema_version = manifest.get("schema_version")
    prompt_contract = manifest.get("prompt_contract")
    if (
        not isinstance(prompt_contract, dict)
        or prompt_contract.get("tool_protocol") != "qwen35_native"
    ):
        raise ValueError(
            "formal data manifest must bind tool_protocol qwen35_native"
        )
    prompt_version = prompt_contract.get("prompt_version")
    if prompt_version == V4_PROMPT_VERSION:
        if manifest_schema_version != 5:
            raise ValueError("native v4 prompt requires data manifest schema_version 5")
        if (prompt_contract.get("terminal_answer_only") is not True
                or prompt_contract.get("terminal_prompt_version") !=
                TERMINAL_PROMPT_VERSION
                or prompt_contract.get("terminal_prompt_sha256") !=
                TERMINAL_PROMPT_SHA256):
            raise ValueError("native v4 terminal prompt contract mismatch")
        native_contract_version = "v4"
        artifact_specs = V4_EVAL_ARTIFACTS
    elif prompt_version == V3_PROMPT_VERSION:
        if manifest_schema_version != 4:
            raise ValueError("native v3 prompt requires data manifest schema_version 4")
        native_contract_version = "v3"
        artifact_specs = V3_EVAL_ARTIFACTS
    elif prompt_version == V2_PROMPT_VERSION:
        if manifest_schema_version != 3:
            raise ValueError("legacy v2 prompt requires data manifest schema_version 3")
        if comparison_mode != "efficiency":
            raise ValueError("A/R capability evaluation requires native data")
        native_contract_version = None
        artifact_specs = None
    else:
        raise ValueError(f"unsupported formal prompt_version: {prompt_version!r}")
    is_native_endpoint = native_contract_version is not None
    eval_artifact = getattr(args, "eval_artifact", None)
    if is_native_endpoint and eval_artifact not in artifact_specs:
        raise ValueError(
            f"native {native_contract_version} formal evaluation requires "
            "--eval-artifact with a sealed key"
        )
    if not is_native_endpoint and eval_artifact is not None:
        raise ValueError("--eval-artifact is supported only by a native contract")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("formal data manifest artifacts must be an object")
    if is_native_endpoint:
        tokenizer = manifest.get("tokenizer")
        if (
            not isinstance(tokenizer, dict)
            or not isinstance(tokenizer.get("revision"), str)
            or not tokenizer["revision"]
            or tokenizer.get("selection_observation_length") != 384
            or tokenizer.get("rollout_observation_length") != 500
        ):
            raise ValueError("native manifest tokenizer/observation contract mismatch")
        assert eval_artifact is not None
        assert artifact_specs is not None
        artifact_spec = artifact_specs[eval_artifact]
        selected_artifact = artifacts.get(eval_artifact)
        if not isinstance(selected_artifact, dict):
            raise ValueError(
                f"formal data manifest has no {eval_artifact} artifact"
            )
        if selected_artifact.get("file") != artifact_spec["file"]:
            raise ValueError(
                f"formal {eval_artifact} artifact file must be {artifact_spec['file']}"
            )
        if args.expected_rows != artifact_spec["rows"]:
            raise ValueError(
                f"formal {eval_artifact} evaluation is fixed to "
                f"{artifact_spec['rows']} rows"
            )
        if selected_artifact.get("rows") != args.expected_rows:
            raise ValueError(
                f"formal {eval_artifact} artifact row count does not match expected_rows"
            )
        artifact_digest = _require_sha256(
            selected_artifact.get("sha256"),
            f"formal {eval_artifact} artifact sha256",
        )
        artifact_path = manifest_path.resolve().parent / artifact_spec["file"]
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or _file_sha256(artifact_path) != artifact_digest
        ):
            raise ValueError(
                f"formal {eval_artifact} artifact identity does not match the manifest"
            )
        if getattr(args, "catalog", None) is not None:
            raise ValueError(
                "native formal evaluation replays the selected Parquet; omit --catalog"
            )
        catalog_path = None
        catalog_digest = None
        stage_suffix = str(artifact_spec["stage_suffix"])
        if comparison_mode == "efficiency":
            stage_names = {
                "control": f"qwen_native_b_{stage_suffix}",
                "cost_aware_gated": f"qwen_native_c_{stage_suffix}",
            }
            mode = f"qwen35_native_{native_contract_version}_b_c_efficiency"
        else:
            stage_names = {
                "parent": f"qwen_native_a_{stage_suffix}",
                "reproduced": f"qwen_native_r_{stage_suffix}",
            }
            mode = f"qwen35_native_{native_contract_version}_a_r_capability"
        artifact_key = eval_artifact
    else:
        if args.expected_rows != 128:
            raise ValueError("formal paired evaluation is fixed to sealed val_128")
        catalog_artifact = artifacts.get("catalog")
        if not isinstance(catalog_artifact, dict):
            raise ValueError("formal data manifest has no catalog artifact")
        if catalog_artifact.get("file") != "catalog.jsonl":
            raise ValueError("formal catalog artifact file must be catalog.jsonl")
        catalog_digest = _require_sha256(
            catalog_artifact.get("sha256"), "formal catalog artifact sha256"
        )
        catalog_argument = getattr(args, "catalog", None)
        if catalog_argument is None:
            raise ValueError("formal paired evaluation requires --catalog")
        catalog_path = Path(catalog_argument).resolve()
        expected_catalog_path = manifest_path.resolve().parent / "catalog.jsonl"
        if catalog_path != expected_catalog_path:
            raise ValueError(
                "catalog path does not match the catalog bound by the data manifest"
            )
        actual_catalog_digest = _file_sha256(catalog_path)
        if actual_catalog_digest != catalog_digest:
            raise ValueError(
                "catalog digest does not match the catalog bound by the data manifest"
            )
        selected_artifact = artifacts.get("val")
        if not isinstance(selected_artifact, dict):
            raise ValueError("formal data manifest has no val artifact")
        if selected_artifact.get("file") != "val_128.parquet":
            raise ValueError("formal val artifact file must be val_128.parquet")
        if selected_artifact.get("rows") != args.expected_rows:
            raise ValueError(
                "formal val artifact row count does not match expected_rows"
            )
        artifact_digest = _require_sha256(
            selected_artifact.get("sha256"), "formal val artifact sha256"
        )
        artifact_path = None
        stage_names = dict(FORMAL_STAGE_NAMES)
        mode = "qwen35_native_v2_b_c"
        artifact_key = "val"

    sample_ids = selected_artifact.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != args.expected_rows:
        raise ValueError(
            f"formal {artifact_key} artifact sample_ids must contain exactly "
            "expected_rows IDs"
        )
    sample_keys = [_sample_key(sample_id) for sample_id in sample_ids]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError(
            f"formal {artifact_key} artifact sample_ids must be unique by type and value"
        )
    sample_ids_digest = hashlib.sha256(
        json.dumps(
            sample_ids, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_schema_version": manifest_schema_version,
        "catalog_path": catalog_path,
        "catalog_sha256": catalog_digest,
        "artifact_key": artifact_key,
        "artifact_file": selected_artifact["file"],
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_digest,
        "mode": mode,
        "comparison_mode": comparison_mode,
        "native_contract_version": native_contract_version,
        "prompt_version": prompt_version,
        "stage_names": stage_names,
        "sample_ids": sample_ids,
        "sample_keys_order": tuple(sample_keys),
        "sample_keys": frozenset(sample_keys),
        "sample_ids_sha256": sample_ids_digest,
        "checkpoint_digests": checkpoint_digests,
    }


def _number(value: Any, field: str, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric in {location}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite in {location}")
    return number


def _validate_record(
    record: dict[str, Any],
    location: str,
    max_searches: int,
    cost_lambda: float,
    utility_tolerance: float,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError(f"missing required fields in {location}: {', '.join(missing)}")
    if not isinstance(record["stage"], str) or not record["stage"].strip():
        raise ValueError(f"stage must be a non-empty string in {location}")
    _sample_key(record["sample_id"])
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError(f"question must be a non-empty string in {location}")
    if (
        not isinstance(record["gold_answers"], list)
        or not record["gold_answers"]
        or not all(isinstance(answer, str) for answer in record["gold_answers"])
    ):
        raise ValueError(f"gold_answers must be a non-empty string list in {location}")
    if record["extracted_answer"] is not None and not isinstance(
        record["extracted_answer"], str
    ):
        raise ValueError(f"extracted_answer must be a string or null in {location}")

    em = _number(record["em"], "em", location)
    if em not in (0.0, 1.0):
        raise ValueError(f"em must be exactly 0 or 1 in {location}")
    searches_value = _number(
        record["executed_search_count"], "executed_search_count", location
    )
    if not searches_value.is_integer():
        raise ValueError(f"executed_search_count must be an integer in {location}")
    searches = int(searches_value)
    if searches < 0 or searches > max_searches:
        raise ValueError(
            f"executed_search_count must be in [0, {max_searches}] in {location}"
        )
    utility = _number(record["posthoc_utility"], "posthoc_utility", location)
    expected_utility = em - cost_lambda * searches / max_searches
    if not math.isclose(
        utility, expected_utility, rel_tol=0.0, abs_tol=utility_tolerance
    ):
        raise ValueError(
            f"posthoc_utility is inconsistent in {location}: found {utility}, "
            f"expected {expected_utility}"
        )

    if "checkpoint_digest" in record and (
        not isinstance(record["checkpoint_digest"], str)
        or not record["checkpoint_digest"]
    ):
        raise ValueError(f"checkpoint_digest must be a non-empty string in {location}")
    if "raw_trajectory" in record and not isinstance(record["raw_trajectory"], str):
        raise ValueError(f"raw_trajectory must be a string in {location}")
    if "turns" in record and not isinstance(record["turns"], list):
        raise ValueError(f"turns must be a list in {location}")

    normalized = dict(record)
    normalized["em"] = int(em)
    normalized["executed_search_count"] = searches
    normalized["posthoc_utility"] = utility
    return normalized


def _require_nonnegative_int(record: dict[str, Any], field: str, location: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer in {location}")
    return value


def _validate_native_record(
    record: dict[str, Any], location: str, native_contract_version: str
) -> None:
    if native_contract_version not in {"v3", "v4"}:
        raise ValueError("native contract version must be v3 or v4")
    try:
        validate_trace_record(record, expected_record_type="eval")
    except ValueError as error:
        raise ValueError(f"invalid native trajectory in {location}: {error}") from error
    required = {
        "schema",
        "schema_version",
        "record_type",
        "group_uid",
        "group_slot",
        "group_size",
        "response_tokens",
        "response_clipped",
        "turns_used",
        "invalid_action_count",
        "generation_clipped_count",
        "max_action_budget",
        "action_count",
        "raw_generations",
        "policy_token_count",
        "observation_token_count",
        "observation_policy_token_count",
        "info_mask_consistent",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(
            f"native trace is missing fields in {location}: {', '.join(missing)}"
        )
    if record["schema"] != "search-r1.trajectory" or record["schema_version"] != 3:
        raise ValueError(f"native trajectory schema mismatch in {location}")
    if record["record_type"] != "eval":
        raise ValueError(f"native endpoint trace must have record_type eval in {location}")
    if "max_searches" in record:
        raise ValueError(f"native trace must not contain legacy max_searches in {location}")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValueError(f"native sample_id must be a non-empty string in {location}")
    if record["group_uid"] != record["sample_id"]:
        raise ValueError(f"native group_uid must equal sample_id in {location}")
    if record["group_size"] != ENDPOINT_EVAL_CONTRACT["group_size"]:
        raise ValueError(f"native endpoint group_size must be 1 in {location}")
    if record["group_slot"] != ENDPOINT_EVAL_CONTRACT["group_slot"]:
        raise ValueError(f"native endpoint group_slot must be 0 in {location}")

    response_tokens = _require_nonnegative_int(record, "response_tokens", location)
    del response_tokens
    _require_nonnegative_int(record, "policy_token_count", location)
    _require_nonnegative_int(record, "observation_token_count", location)
    if _require_nonnegative_int(
        record, "observation_policy_token_count", location
    ) != 0:
        raise ValueError(f"observation_policy_token_count must be zero in {location}")
    if record["info_mask_consistent"] is not True:
        raise ValueError(f"info_mask_consistent must be true in {location}")
    turns_used = _require_nonnegative_int(record, "turns_used", location)
    invalid_count = _require_nonnegative_int(
        record, "invalid_action_count", location
    )
    clipped_count = _require_nonnegative_int(
        record, "generation_clipped_count", location
    )
    max_actions = _require_nonnegative_int(record, "max_action_budget", location)
    action_count = _require_nonnegative_int(record, "action_count", location)
    if max_actions != 4:
        raise ValueError(f"native max_action_budget must be 4 in {location}")
    if action_count > max_actions + 1:
        raise ValueError(
            f"native action_count exceeds its budget plus terminal generation in {location}"
        )
    if record["executed_search_count"] > action_count:
        raise ValueError(f"executed searches exceed action_count in {location}")
    if invalid_count > action_count:
        raise ValueError(f"invalid actions exceed action_count in {location}")
    if clipped_count > action_count:
        raise ValueError(f"clipped generations exceed action_count in {location}")
    if not isinstance(record["response_clipped"], bool):
        raise ValueError(f"response_clipped must be boolean in {location}")
    if record["response_clipped"] != (clipped_count > 0):
        raise ValueError(
            f"response_clipped disagrees with generation_clipped_count in {location}"
        )
    if turns_used != len(record["turns"]):
        raise ValueError(f"turns_used must equal len(turns) in {location}")
    raw_generations = record["raw_generations"]
    if not isinstance(raw_generations, list) or len(raw_generations) != action_count:
        raise ValueError(
            f"raw_generations must contain exactly action_count entries in {location}"
        )
    if action_count > max_actions:
        events = record.get("generation_events")
        if (not isinstance(events, list) or len(events) != action_count
                or not all(isinstance(event, dict) for event in events)
                or events[-1].get("terminal_generation") is not True
                or events[-1].get("executed_search") is not False
                or any(event.get("terminal_generation") is True
                       for event in events[:-1])):
            raise ValueError(
                f"native terminal generation is not auditable in {location}"
            )
        if (native_contract_version == "v4"
                and events[-1].get("terminal_instruction_applied") is not True):
            raise ValueError(
                f"native v4 terminal generation lacks answer-only evidence in {location}"
            )


def read_stage(
    path: Path,
    expected_rows: int,
    max_searches: int,
    cost_lambda: float,
    utility_tolerance: float,
    native_contract_version: str | None = None,
) -> tuple[
    dict[tuple[str, str | int], dict[str, Any]],
    frozenset[str],
    str,
    tuple[tuple[str, str | int], ...],
]:
    records: dict[tuple[str, str | int], dict[str, Any]] = {}
    order: list[tuple[str, str | int]] = []
    schema: frozenset[str] | None = None
    stages: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record in {path}:{line_number}")
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error.msg}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"JSONL record must be an object in {path}:{line_number}")
            record_schema = frozenset(raw_record)
            if schema is None:
                schema = record_schema
            elif record_schema != schema:
                missing = sorted(schema - record_schema)
                extra = sorted(record_schema - schema)
                raise ValueError(
                    f"schema mismatch in {path}:{line_number}; "
                    f"missing={missing}, extra={extra}"
                )
            location = f"{path}:{line_number}"
            record = _validate_record(
                raw_record,
                location,
                max_searches,
                cost_lambda,
                utility_tolerance,
            )
            if native_contract_version is not None:
                _validate_native_record(
                    record, location, native_contract_version)
            key = _sample_key(record["sample_id"])
            if key in records:
                raise ValueError(f"duplicate sample_id {record['sample_id']!r} in {path}")
            records[key] = record
            order.append(key)
            stages.add(record["stage"])

    if len(records) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {path}, found {len(records)}")
    if len(stages) != 1:
        raise ValueError(f"expected one stage value in {path}, found {sorted(stages)}")
    assert schema is not None
    return records, schema, next(iter(stages)), tuple(order)


def read_catalog(
    path: Path, expected_rows: int
) -> dict[tuple[str, str | int], dict[str, Any]]:
    catalog: dict[tuple[str, str | int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record in {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {error.msg}"
                ) from error
            location = f"{path}:{line_number}"
            if not isinstance(record, dict):
                raise ValueError(f"catalog record must be an object in {location}")
            missing = sorted(
                {"sample_id", "question", "golden_answers"} - record.keys()
            )
            if missing:
                raise ValueError(
                    f"missing catalog fields in {location}: {', '.join(missing)}"
                )
            key = _sample_key(record["sample_id"])
            if key in catalog:
                raise ValueError(
                    f"duplicate catalog sample_id {record['sample_id']!r} in {path}"
                )
            question = record["question"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"catalog question must be non-empty in {location}")
            answers = record["golden_answers"]
            if (
                not isinstance(answers, list)
                or not answers
                or not all(
                    isinstance(answer, str) and answer.strip() for answer in answers
                )
            ):
                raise ValueError(
                    f"catalog golden_answers must be a non-empty string list in {location}"
                )
            catalog[key] = {
                "sample_id": record["sample_id"],
                "question": question,
                "golden_answers": list(answers),
            }
    if len(catalog) < expected_rows:
        raise ValueError(
            f"expected at least {expected_rows} rows in {path}, "
            f"found {len(catalog)}"
        )
    return catalog


def read_eval_parquet_catalog(
    path: Path, expected_rows: int
) -> tuple[
    dict[tuple[str, str | int], dict[str, Any]],
    tuple[tuple[str, str | int], ...],
]:
    """Read the sealed endpoint artifact used to generate the evaluated traces."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ValueError(
            "pyarrow is required to replay a native endpoint artifact"
        ) from error

    rows = parquet.read_table(path).to_pylist()
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows in endpoint artifact {path}, found {len(rows)}"
        )
    catalog: dict[tuple[str, str | int], dict[str, Any]] = {}
    order: list[tuple[str, str | int]] = []
    for index, row in enumerate(rows, 1):
        location = f"{path}:row-{index}"
        if not isinstance(row, dict):
            raise ValueError(f"endpoint artifact row must be an object in {location}")
        data_source = row.get("data_source")
        extra_info = row.get("extra_info")
        if not isinstance(data_source, str) or not data_source:
            raise ValueError(f"endpoint data_source must be non-empty in {location}")
        if not isinstance(extra_info, dict):
            raise ValueError(f"endpoint extra_info must be an object in {location}")
        source_split = extra_info.get("split")
        source_index = extra_info.get("index")
        if (
            not isinstance(source_split, str)
            or not source_split
            or isinstance(source_index, bool)
            or not isinstance(source_index, (str, int))
        ):
            raise ValueError(f"endpoint source identity is invalid in {location}")
        sample_id = f"{data_source}:{source_split}:{source_index}"
        key = _sample_key(sample_id)
        if key in catalog:
            raise ValueError(f"duplicate endpoint sample_id {sample_id!r} in {path}")

        prompt = row.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or not isinstance(prompt[0], dict)
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
        ):
            raise ValueError(f"endpoint prompt must be one user message in {location}")
        match = QUESTION_PATTERN.search(prompt[0]["content"])
        if match is None or not match.group(1).strip():
            raise ValueError(f"endpoint prompt has no terminal Question in {location}")
        question = match.group(1).strip()
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict):
            raise ValueError(f"endpoint reward_model must be an object in {location}")
        ground_truth = reward_model.get("ground_truth")
        answers = ground_truth.get("target") if isinstance(ground_truth, dict) else None
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) and answer.strip() for answer in answers)
        ):
            raise ValueError(f"endpoint gold answers are invalid in {location}")
        catalog[key] = {
            "sample_id": sample_id,
            "question": question,
            "golden_answers": list(answers),
        }
        order.append(key)
    return catalog, tuple(order)


def validate_catalog_replay(
    records_by_role: dict[
        str, dict[tuple[str, str | int], dict[str, Any]]
    ],
    catalog: dict[tuple[str, str | int], dict[str, Any]],
) -> None:
    catalog_ids = set(catalog)
    for role, records in records_by_role.items():
        role_ids = set(records)
        if not role_ids.issubset(catalog_ids):
            missing = sorted(role_ids - catalog_ids)[:5]
            catalog_only = sorted(catalog_ids - role_ids)[:5]
            raise ValueError(
                f"sample-set mismatch between catalog and {role}; "
                f"missing_from_catalog={missing}, catalog_only={catalog_only}"
            )
        for sample_key in sorted(role_ids):
            record = records[sample_key]
            expected = catalog[sample_key]
            sample_id = record["sample_id"]
            if record["question"] != expected["question"]:
                raise ValueError(
                    f"question mismatch with catalog for sample_id {sample_id!r} in {role}"
                )
            if record["gold_answers"] != expected["golden_answers"]:
                raise ValueError(
                    f"gold_answers mismatch with catalog for sample_id {sample_id!r} in {role}"
                )
            prediction = record["extracted_answer"] or ""
            strict_em = int(qa_em.em_check(prediction, expected["golden_answers"]))
            if record["em"] != strict_em:
                raise ValueError(
                    f"strict EM mismatch with catalog replay for sample_id "
                    f"{sample_id!r} in {role}: trace={record['em']}, replay={strict_em}"
                )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category(baseline_em: int, candidate_em: int) -> str:
    if baseline_em and candidate_em:
        return "both_correct"
    if baseline_em:
        return "baseline_correct_candidate_wrong"
    if candidate_em:
        return "baseline_wrong_candidate_correct"
    return "both_wrong"


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


def _stage_summary(records: list[dict[str, Any]], max_searches: int) -> dict[str, Any]:
    correct = [record for record in records if record["em"] == 1]
    wrong = [record for record in records if record["em"] == 0]
    total_searches = sum(record["executed_search_count"] for record in records)
    total_utility = sum(record["posthoc_utility"] for record in records)
    distribution = {
        result: {
            str(searches): sum(
                1
                for record in subset
                if record["executed_search_count"] == searches
            )
            for searches in range(max_searches + 1)
        }
        for result, subset in (("correct", correct), ("wrong", wrong))
    }
    summary = {
        "N": len(records),
        "K_correct": len(correct),
        "wrong": len(wrong),
        "S_total_searches": total_searches,
        "EM": len(correct) / len(records),
        "mean_searches": total_searches / len(records),
        "no_search_count": sum(
            record["executed_search_count"] == 0 for record in records
        ),
        "no_search_ratio": sum(
            record["executed_search_count"] == 0 for record in records
        ) / len(records),
        "mean_searches_given_correct": _mean(
            sum(record["executed_search_count"] for record in correct), len(correct)
        ),
        "correct_only_mean_searches": _mean(
            sum(record["executed_search_count"] for record in correct), len(correct)
        ),
        "mean_searches_given_wrong": _mean(
            sum(record["executed_search_count"] for record in wrong), len(wrong)
        ),
        "total_utility": total_utility,
        "utility": total_utility / len(records),
        "search_distribution": distribution,
    }
    if all(set(NATIVE_REPORT_FIELDS).issubset(record) for record in records):
        summary.update({
            "mean_action_count": sum(record["action_count"] for record in records)
            / len(records),
            "mean_trajectory_tokens": sum(
                record["response_tokens"] for record in records
            ) / len(records),
            "mean_invalid_action_count": sum(
                record["invalid_action_count"] for record in records
            ) / len(records),
            "invalid_trajectory_count": sum(
                record["invalid_action_count"] > 0 for record in records
            ),
            "invalid_trajectory_ratio": sum(
                record["invalid_action_count"] > 0 for record in records
            ) / len(records),
            "clipped_trajectory_count": sum(
                record["response_clipped"] for record in records
            ),
            "clipped_trajectory_ratio": sum(
                record["response_clipped"] for record in records
            ) / len(records),
            "generation_clipped_total": sum(
                record["generation_clipped_count"] for record in records
            ),
        })
    return summary


def _comparison_summary(
    paired: list[dict[str, dict[str, Any]]], baseline_role: str, candidate_role: str
) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        subset = [
            row
            for row in paired
            if _category(row[baseline_role]["em"], row[candidate_role]["em"]) == category
        ]
        count = len(subset)
        b_searches = sum(row[baseline_role]["executed_search_count"] for row in subset)
        candidate_searches = sum(
            row[candidate_role]["executed_search_count"] for row in subset
        )
        b_utility = sum(row[baseline_role]["posthoc_utility"] for row in subset)
        candidate_utility = sum(
            row[candidate_role]["posthoc_utility"] for row in subset
        )
        categories[category] = {
            "count": count,
            "baseline_total_searches": b_searches,
            "candidate_total_searches": candidate_searches,
            "search_delta_candidate_minus_baseline": candidate_searches - b_searches,
            "baseline_mean_searches": _mean(b_searches, count),
            "candidate_mean_searches": _mean(candidate_searches, count),
            "baseline_total_utility": b_utility,
            "candidate_total_utility": candidate_utility,
            "total_utility_delta_candidate_minus_baseline": (
                candidate_utility - b_utility
            ),
            "mean_utility_delta_candidate_minus_baseline": _mean(
                candidate_utility - b_utility, count
            ),
        }

    baseline_correct = sum(row[baseline_role]["em"] for row in paired)
    candidate_correct = sum(row[candidate_role]["em"] for row in paired)
    baseline_searches = sum(
        row[baseline_role]["executed_search_count"] for row in paired
    )
    candidate_searches = sum(
        row[candidate_role]["executed_search_count"] for row in paired
    )
    baseline_utility = sum(row[baseline_role]["posthoc_utility"] for row in paired)
    candidate_utility = sum(
        row[candidate_role]["posthoc_utility"] for row in paired
    )
    return {
        "baseline_role": baseline_role,
        "candidate_role": candidate_role,
        "correct_count_delta_candidate_minus_baseline": candidate_correct - baseline_correct,
        "total_search_delta_candidate_minus_baseline": candidate_searches - baseline_searches,
        "total_utility_delta_candidate_minus_baseline": (
            candidate_utility - baseline_utility
        ),
        "mean_utility_delta_candidate_minus_baseline": (
            candidate_utility - baseline_utility
        ) / len(paired),
        "categories": categories,
    }


def _quantile(values: list[float], probability: float) -> float:
    values.sort()
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _bootstrap_statistics(
    rows: list[dict[str, dict[str, Any]]], baseline_role: str, candidate_role: str
) -> dict[str, float | None]:
    count = len(rows)
    control_correct = [row[baseline_role] for row in rows if row[baseline_role]["em"]]
    candidate_correct = [row[candidate_role] for row in rows if row[candidate_role]["em"]]
    statistics: dict[str, float | None] = {
        "em": sum(
            row[candidate_role]["em"] - row[baseline_role]["em"] for row in rows
        ) / count,
        "executed_searches": sum(
            row[candidate_role]["executed_search_count"]
            - row[baseline_role]["executed_search_count"]
            for row in rows
        ) / count,
        "correct_only_searches": None,
    }
    if control_correct and candidate_correct:
        statistics["correct_only_searches"] = (
            sum(record["executed_search_count"] for record in candidate_correct)
            / len(candidate_correct)
            - sum(record["executed_search_count"] for record in control_correct)
            / len(control_correct)
        )
    if all(set(NATIVE_REPORT_FIELDS).issubset(row[baseline_role]) for row in rows):
        for name, field in (
            ("action_count", "action_count"),
            ("trajectory_tokens", "response_tokens"),
            ("invalid_actions", "invalid_action_count"),
        ):
            statistics[name] = sum(
                row[candidate_role][field] - row[baseline_role][field] for row in rows
            ) / count
        statistics["clipping_rate"] = sum(
            int(row[candidate_role]["response_clipped"])
            - int(row[baseline_role]["response_clipped"])
            for row in rows
        ) / count
    return statistics


def _paired_bootstrap(
    paired: list[dict[str, dict[str, Any]]], baseline_role: str, candidate_role: str
) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = _bootstrap_statistics(paired, baseline_role, candidate_role)
    samples: dict[str, list[float]] = {name: [] for name in estimates}
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [paired[rng.randrange(len(paired))] for _ in paired]
        values = _bootstrap_statistics(resample, baseline_role, candidate_role)
        for name, value in values.items():
            if value is not None:
                samples[name].append(value)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    metrics = {}
    for name, estimate in estimates.items():
        values = samples[name]
        metric = {
            "estimate_candidate_minus_baseline": estimate,
            "ci_lower": _quantile(values, alpha) if values else None,
            "ci_upper": _quantile(values, 1.0 - alpha) if values else None,
            "valid_resamples": len(values),
        }
        if baseline_role == "control":
            metric["estimate_candidate_minus_control"] = estimate
        metrics[name] = metric
    return {
        "method": "paired_percentile_bootstrap",
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "pairing_key": ENDPOINT_EVAL_CONTRACT["pairing_key"],
        "baseline_role": baseline_role,
        "candidate_role": candidate_role,
        "metrics": metrics,
    }


def _csv_value(field: str, value: Any) -> Any:
    if field in ("gold_answers", "turns"):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _summary_markdown(summary: dict[str, Any]) -> str:
    capability = summary.get("report_type") == "parent_reproduced_capability"
    baseline_label = "A / Parent (post-trained)" if capability else "B / Control"
    report_title = (
        "# Parent vs Reproduced Capability Report"
        if capability else "# Control vs Cost-aware Efficiency Report"
    )
    lines = [
        report_title,
        "",
        f"All {len(summary['stages'])} active stages contain the same "
        f"{summary['expected_rows']} test samples. "
        f"Utility is `EM - {summary['cost_lambda']:.2f} * searches / "
        f"{summary['max_searches']}`.",
        "",
        "## Stage Totals",
        "",
        "| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role, stage in summary["stages"].items():
        lines.append(
            f"| {role} | {stage['stage']} | {stage['N']} | {stage['K_correct']} | "
            f"{stage['S_total_searches']} | {stage['EM']:.4f} | "
            f"{stage['mean_searches']:.4f} | "
            f"{_format_optional(stage['mean_searches_given_correct'])} | "
            f"{_format_optional(stage['mean_searches_given_wrong'])} | "
            f"{stage['utility']:.4f} |"
        )

    if summary["schema_version"] >= 2:
        contract = summary["formal_contract"]["endpoint_evaluation"]
        lines.extend([
            "",
            "## Deterministic Endpoint Contract",
            "",
            f"Group size `{contract['group_size']}`, `do_sample={str(contract['do_sample']).lower()}` "
            f"({contract['decoding']}), seed `{contract['seed']}`, group slot "
            f"`{contract['group_slot']}`. "
            f"{'A/R' if capability else 'B/C'} preserve the manifest sample order and pair by "
            f"`{contract['pairing_key']}`.",
            f"One rollout per question; temperature/top-p `{contract['temperature']:.1f}/"
            f"{contract['top_p']:.1f}`, top-k `{contract['top_k']}`, min-p "
            f"`{contract['min_p']:.1f}`, presence penalty "
            f"`{contract['presence_penalty']:.1f}`, repetition penalty "
            f"`{contract['repetition_penalty']:.1f}`.",
            "",
            "| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | "
            "Invalid trajectory ratio | Clipping ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for role, stage in summary["stages"].items():
            lines.append(
                f"| {role} | {stage['mean_action_count']:.4f} | "
                f"{stage['mean_trajectory_tokens']:.4f} | "
                f"{stage['mean_invalid_action_count']:.4f} | "
                f"{stage['invalid_trajectory_ratio']:.4f} | "
                f"{stage['clipped_trajectory_ratio']:.4f} |"
            )

    category_labels = {
        "both_correct": "Both correct",
        "baseline_correct_candidate_wrong": f"{baseline_label} correct, candidate wrong",
        "baseline_wrong_candidate_correct": f"{baseline_label} wrong, candidate correct",
        "both_wrong": "Both wrong",
    }
    for candidate_role, comparison in summary["comparisons"].items():
        lines.extend([
            "",
            f"## {baseline_label} vs {candidate_role}",
            "",
            f"| Category | Questions | {baseline_label} searches | Candidate searches | "
            "Search delta | Utility delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for category in CATEGORY_ORDER:
            values = comparison["categories"][category]
            lines.append(
                f"| {category_labels[category]} | {values['count']} | "
                f"{values['baseline_total_searches']} | "
                f"{values['candidate_total_searches']} | "
                f"{values['search_delta_candidate_minus_baseline']:+d} | "
                f"{values['total_utility_delta_candidate_minus_baseline']:+.4f} |"
            )
        lines.extend([
            "",
            f"Overall: correct-count delta "
            f"`{comparison['correct_count_delta_candidate_minus_baseline']:+d}`, "
            f"search delta `{comparison['total_search_delta_candidate_minus_baseline']:+d}`, "
            f"mean Utility delta "
            f"`{comparison['mean_utility_delta_candidate_minus_baseline']:+.4f}`.",
        ])
        if "paired_bootstrap" in comparison:
            bootstrap = comparison["paired_bootstrap"]
            lines.extend([
                "",
                f"Paired bootstrap: `{bootstrap['resamples']}` resamples, seed "
                f"`{bootstrap['seed']}`, {bootstrap['confidence_level']:.0%} percentile CI. "
                f"All estimates are candidate minus {baseline_label}.",
                "",
                "| Metric | Estimate | CI lower | CI upper |",
                "| --- | ---: | ---: | ---: |",
            ])
            for name, values in bootstrap["metrics"].items():
                lines.append(
                    f"| {name} | {_format_optional(values['estimate_candidate_minus_baseline'])} | "
                    f"{_format_optional(values['ci_lower'])} | "
                    f"{_format_optional(values['ci_upper'])} |"
                )
    lines.extend(["", (
        "A/R deltas measure Search-R1 capability change on identical deterministic "
        "endpoint samples; they are separate from the B/C cost-efficiency comparison."
        if capability else
        "Search reductions in `both_correct` are lossless efficiency gains. Reductions in "
        "`baseline_correct_candidate_wrong` are harmful savings; reductions in "
        "`both_wrong` only remove unsuccessful calls."
    ), ""])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    if args.expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if args.max_searches <= 0:
        raise ValueError("max_searches must be positive")
    if not math.isfinite(args.cost_lambda) or args.cost_lambda < 0:
        raise ValueError("cost_lambda must be finite and non-negative")
    if not math.isfinite(args.utility_tolerance) or args.utility_tolerance < 0:
        raise ValueError("utility_tolerance must be finite and non-negative")

    role_arguments = _active_role_arguments(args)
    active_roles = tuple(role for role, _ in role_arguments)
    baseline_role = active_roles[0]
    candidate_roles = active_roles[1:]
    comparison_mode = _comparison_mode(args)
    formal_contract = _load_formal_contract(args)
    native_contract_version = (
        formal_contract["native_contract_version"]
        if formal_contract is not None else None)
    require_native_trace = native_contract_version is not None
    paths = {role: Path(getattr(args, role)) for role in active_roles}
    records_by_role: dict[str, dict[tuple[str, str | int], dict[str, Any]]] = {}
    schemas: dict[str, frozenset[str]] = {}
    stage_names: dict[str, str] = {}
    stage_orders: dict[str, tuple[tuple[str, str | int], ...]] = {}
    for role in active_roles:
        records, schema, stage_name, stage_order = read_stage(
            paths[role],
            args.expected_rows,
            args.max_searches,
            args.cost_lambda,
            args.utility_tolerance,
            native_contract_version=native_contract_version,
        )
        records_by_role[role] = records
        schemas[role] = schema
        stage_names[role] = stage_name
        stage_orders[role] = stage_order

    baseline_schema = schemas[baseline_role]
    for role in candidate_roles:
        if schemas[role] != baseline_schema:
            raise ValueError(
                f"schema mismatch between {baseline_role} and {role}; "
                f"missing={sorted(baseline_schema - schemas[role])}, "
                f"extra={sorted(schemas[role] - baseline_schema)}"
            )
    if len(set(stage_names.values())) != len(stage_names):
        raise ValueError(f"stage values must be distinct across inputs: {stage_names}")
    if formal_contract is not None:
        for role, expected_stage in formal_contract["stage_names"].items():
            if stage_names[role] != expected_stage:
                raise ValueError(
                    f"formal {role} stage must be {expected_stage}, "
                    f"found {stage_names[role]}"
                )
        expected_digests = formal_contract["checkpoint_digests"]
        for role, expected_digest in expected_digests.items():
            for record in records_by_role[role].values():
                if record.get("checkpoint_digest") != expected_digest:
                    raise ValueError(
                        f"formal {role} checkpoint_digest does not match "
                        f"the selected endpoint for sample_id {record['sample_id']!r}"
                    )

    baseline_ids = set(records_by_role[baseline_role])
    for role in candidate_roles:
        role_ids = set(records_by_role[role])
        if role_ids != baseline_ids:
            missing = sorted(baseline_ids - role_ids)[:5]
            extra = sorted(role_ids - baseline_ids)[:5]
            raise ValueError(
                f"sample-set mismatch between {baseline_role} and {role}; "
                f"missing={missing}, extra={extra}"
            )
        if require_native_trace and stage_orders[role] != stage_orders[baseline_role]:
            raise ValueError(
                f"sample order mismatch between {baseline_role} and {role}"
            )
    if (
        formal_contract is not None
        and baseline_ids != formal_contract["sample_keys"]
    ):
        missing = sorted(formal_contract["sample_keys"] - baseline_ids)[:5]
        extra = sorted(baseline_ids - formal_contract["sample_keys"])[:5]
        raise ValueError(
            "formal trace sample-set does not exactly match manifest val_128; "
            f"missing={missing}, extra={extra}"
        )
    if (
        require_native_trace
        and stage_orders[baseline_role] != formal_contract["sample_keys_order"]
    ):
        raise ValueError(
            f"formal trace order does not exactly match manifest "
            f"{formal_contract['artifact_key']} sample_ids"
        )

    catalog_argument = getattr(args, "catalog", None)
    catalog_path = Path(catalog_argument) if catalog_argument is not None else None
    if require_native_trace:
        artifact_catalog, artifact_order = read_eval_parquet_catalog(
            formal_contract["artifact_path"], args.expected_rows
        )
        if artifact_order != formal_contract["sample_keys_order"]:
            raise ValueError(
                "endpoint Parquet order does not match manifest artifact sample_ids"
            )
        validate_catalog_replay(records_by_role, artifact_catalog)
    elif catalog_path is not None:
        catalog = read_catalog(catalog_path, args.expected_rows)
        validate_catalog_replay(records_by_role, catalog)

    paired: list[dict[str, dict[str, Any]]] = []
    paired_order = (
        formal_contract["sample_keys_order"]
        if require_native_trace else sorted(baseline_ids)
    )
    for sample_key in paired_order:
        row = {role: records_by_role[role][sample_key] for role in active_roles}
        baseline = row[baseline_role]
        for role in candidate_roles:
            candidate = row[role]
            if candidate["question"] != baseline["question"]:
                raise ValueError(
                    f"question mismatch for sample_id {baseline['sample_id']!r} in {role}"
                )
            if candidate["gold_answers"] != baseline["gold_answers"]:
                raise ValueError(
                    f"gold_answers mismatch for sample_id {baseline['sample_id']!r} in {role}"
                )
        paired.append(row)

    optional_fields = [field for field in PASSTHROUGH_FIELDS if field in baseline_schema]
    report_fields = [
        field for field in NATIVE_REPORT_FIELDS if field in baseline_schema]
    paired_fields = ["sample_id", "question", "gold_answers"]
    for role in active_roles:
        paired_fields.extend([
            f"{role}_stage",
            f"{role}_extracted_answer",
            f"{role}_em",
            f"{role}_executed_search_count",
            f"{role}_posthoc_utility",
        ])
        paired_fields.extend(
            f"{role}_{field}" for field in (*report_fields, *optional_fields)
        )
    for candidate_role in candidate_roles:
        paired_fields.extend([
            f"{candidate_role}_category_vs_{baseline_role}",
            f"{candidate_role}_search_delta_vs_{baseline_role}",
            f"{candidate_role}_utility_delta_vs_{baseline_role}",
        ])
    paired_csv_rows = []
    for row in paired:
        baseline = row[baseline_role]
        output: dict[str, Any] = {
            "sample_id": baseline["sample_id"],
            "question": baseline["question"],
            "gold_answers": _csv_value("gold_answers", baseline["gold_answers"]),
        }
        for role in active_roles:
            record = row[role]
            for field in (
                "stage",
                "extracted_answer",
                "em",
                "executed_search_count",
                "posthoc_utility",
                *report_fields,
                *optional_fields,
            ):
                output[f"{role}_{field}"] = _csv_value(field, record[field])
        for candidate_role in candidate_roles:
            candidate = row[candidate_role]
            output[f"{candidate_role}_category_vs_{baseline_role}"] = _category(
                baseline["em"], candidate["em"]
            )
            output[f"{candidate_role}_search_delta_vs_{baseline_role}"] = (
                candidate["executed_search_count"] - baseline["executed_search_count"]
            )
            output[f"{candidate_role}_utility_delta_vs_{baseline_role}"] = (
                candidate["posthoc_utility"] - baseline["posthoc_utility"]
            )
        paired_csv_rows.append(output)

    question_fields = [
        "role",
        "stage",
        "sample_id",
        "question",
        "gold_answers",
        "extracted_answer",
        "em",
        "executed_search_count",
        "posthoc_utility",
        *report_fields,
        *optional_fields,
    ]
    correctness_rows: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in paired:
        for role in active_roles:
            record = row[role]
            output = {
                "role": role,
                **{
                    field: _csv_value(field, record[field])
                    for field in question_fields
                    if field != "role"
                },
            }
            correctness_rows[record["em"]].append(output)

    transition_counts: Counter[tuple[str, str, int, int]] = Counter()
    for row in paired:
        baseline = row[baseline_role]
        for candidate_role in candidate_roles:
            candidate = row[candidate_role]
            transition_counts[
                (
                    candidate_role,
                    _category(baseline["em"], candidate["em"]),
                    baseline["executed_search_count"],
                    candidate["executed_search_count"],
                )
            ] += 1
    transition_rows = []
    for candidate_role in candidate_roles:
        for category in CATEGORY_ORDER:
            keys = sorted(
                key
                for key in transition_counts
                if key[0] == candidate_role and key[1] == category
            )
            for _, _, baseline_searches, candidate_searches in keys:
                transition_rows.append({
                    "comparison": f"{baseline_role}_vs_{candidate_role}",
                    "candidate_role": candidate_role,
                    "candidate_stage": stage_names[candidate_role],
                    "category": category,
                    f"{baseline_role}_searches": baseline_searches,
                    "candidate_searches": candidate_searches,
                    f"search_delta_candidate_minus_{baseline_role}": (
                        candidate_searches - baseline_searches
                    ),
                    "count": transition_counts[
                        (candidate_role, category, baseline_searches, candidate_searches)
                    ],
                })

    summary: dict[str, Any] = {
        "schema_version": (
            3 if native_contract_version == "v4" else
            2 if native_contract_version == "v3" else 1),
        "report_type": (
            "parent_reproduced_capability"
            if comparison_mode == "capability"
            else "control_cost_efficiency"
        ),
        "expected_rows": args.expected_rows,
        "cost_lambda": args.cost_lambda,
        "max_searches": args.max_searches,
        "inputs": {
            role: {
                "path": str(paths[role].resolve()),
                "sha256": _file_sha256(paths[role]),
            }
            for role in active_roles
        },
        "stages": {},
        "comparisons": {},
    }
    if catalog_path is not None:
        summary["catalog"] = {
            "path": str(catalog_path.resolve()),
            "sha256": _file_sha256(catalog_path),
            "row_count": len(catalog),
            "matched_rows": len(baseline_ids),
            "replay_status": "passed",
            "strict_em_scorer": "qa_em.em_check",
        }
    if formal_contract is not None:
        formal_summary: dict[str, Any] = {
            "mode": formal_contract["mode"],
            "native_contract_version": formal_contract["native_contract_version"],
            "data_manifest": {
                "path": str(formal_contract["manifest_path"]),
                "sha256": formal_contract["manifest_sha256"],
                "schema_version": formal_contract["manifest_schema_version"],
                "prompt_version": formal_contract["prompt_version"],
            },
            "endpoints": {
                role: {
                    "stage": formal_contract["stage_names"][role],
                    "checkpoint_digest": formal_contract["checkpoint_digests"][role],
                }
                for role in formal_contract["stage_names"]
            },
        }
        artifact_summary = {
            "key": formal_contract["artifact_key"],
            "file": formal_contract["artifact_file"],
            "rows": args.expected_rows,
            "sha256": formal_contract["artifact_sha256"],
            "sample_ids_sha256": formal_contract["sample_ids_sha256"],
            "sample_set_status": "exact",
            "sample_order_status": (
                "exact" if require_native_trace else "not_checked"),
        }
        if require_native_trace:
            artifact_summary["path"] = str(formal_contract["artifact_path"])
            artifact_summary["parquet_replay_status"] = "passed"
            formal_summary["evaluation_artifact"] = artifact_summary
            formal_summary["endpoint_evaluation"] = {
                **ENDPOINT_EVAL_CONTRACT,
                "bootstrap": {
                    "method": "paired_percentile_bootstrap",
                    "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
                    "seed": BOOTSTRAP_SEED,
                    "resamples": BOOTSTRAP_RESAMPLES,
                },
            }
        else:
            formal_summary["catalog"] = {
                "path": str(formal_contract["catalog_path"]),
                "sha256": formal_contract["catalog_sha256"],
            }
            artifact_summary.pop("key")
            artifact_summary.pop("sha256")
            artifact_summary.pop("sample_order_status")
            formal_summary["val"] = artifact_summary
        summary["formal_contract"] = formal_summary
    for role in active_roles:
        stage_summary = _stage_summary([row[role] for row in paired], args.max_searches)
        summary["stages"][role] = {"stage": stage_names[role], **stage_summary}
    for candidate_role in candidate_roles:
        summary["comparisons"][candidate_role] = _comparison_summary(
            paired, baseline_role, candidate_role
        )
        if require_native_trace:
            summary["comparisons"][candidate_role]["paired_bootstrap"] = (
                _paired_bootstrap(paired, baseline_role, candidate_role)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paired_results.csv", paired_fields, paired_csv_rows)
    _write_csv(
        args.output_dir / "correct_questions.csv",
        question_fields,
        correctness_rows[1],
    )
    _write_csv(
        args.output_dir / "wrong_questions.csv", question_fields, correctness_rows[0]
    )
    _write_csv(
        args.output_dir / "search_transition.csv",
        [
            "comparison",
            "candidate_role",
            "candidate_stage",
            "category",
            f"{baseline_role}_searches",
            "candidate_searches",
            f"search_delta_candidate_minus_{baseline_role}",
            "count",
        ],
        transition_rows,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic paired endpoint capability or efficiency reports."
    )
    parser.add_argument("--control", type=Path)
    parser.add_argument("--cost-aware-old", dest="cost_aware_old", type=Path)
    parser.add_argument("--cost-aware-gated", dest="cost_aware_gated", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--reproduced", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--eval-artifact", choices=tuple(V4_EVAL_ARTIFACTS))
    parser.add_argument("--expected-control-checkpoint-digest")
    parser.add_argument("--expected-cost-aware-gated-checkpoint-digest")
    parser.add_argument("--expected-parent-checkpoint-digest")
    parser.add_argument("--expected-reproduced-checkpoint-digest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=128)
    parser.add_argument("--cost-lambda", type=float, default=0.10)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--utility-tolerance", type=float, default=1e-6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        analyze(args)
    except (OSError, ValueError) as error:
        print(f"paired eval error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
