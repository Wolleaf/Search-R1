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
import re
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REWARD_SCORE_DIR = REPOSITORY_ROOT / "verl" / "utils" / "reward_score"
if str(REWARD_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_SCORE_DIR))

import qa_em  # noqa: E402


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
ROLE_ARGUMENTS = (
    ("control", "B / Control"),
    ("cost_aware_old", "C-old / Cost-aware"),
    ("cost_aware_gated", "C-gated / Cost-aware gated"),
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
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def _load_formal_contract(args: argparse.Namespace) -> dict[str, Any] | None:
    values = {
        "data_manifest": getattr(args, "data_manifest", None),
        "control_digest": getattr(
            args, "expected_control_checkpoint_digest", None
        ),
        "cost_digest": getattr(
            args, "expected_cost_aware_gated_checkpoint_digest", None
        ),
    }
    supplied = {name: value is not None for name, value in values.items()}
    if any(supplied.values()) and not all(supplied.values()):
        missing = sorted(name for name, present in supplied.items() if not present)
        raise ValueError(
            "formal paired evaluation arguments are all-or-none; missing "
            + ", ".join(missing)
        )
    if not any(supplied.values()):
        return None
    if getattr(args, "cost_aware_old", None) is not None:
        raise ValueError("formal paired evaluation accepts only control and cost_aware_gated")
    if args.expected_rows != 128:
        raise ValueError("formal paired evaluation is fixed to sealed val_128")

    control_digest = _require_sha256(
        values["control_digest"], "expected control checkpoint digest"
    )
    cost_digest = _require_sha256(
        values["cost_digest"], "expected cost-aware-gated checkpoint digest"
    )
    manifest_path = Path(values["data_manifest"])
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid data manifest JSON in {manifest_path}: {error.msg}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("formal data manifest must be a JSON object")
    if manifest.get("schema_version") != 3:
        raise ValueError("formal data manifest schema_version must be 3")
    prompt_contract = manifest.get("prompt_contract")
    if (
        not isinstance(prompt_contract, dict)
        or prompt_contract.get("tool_protocol") != "qwen35_native"
    ):
        raise ValueError(
            "formal data manifest must bind tool_protocol qwen35_native"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("formal data manifest artifacts must be an object")
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

    val_artifact = artifacts.get("val")
    if not isinstance(val_artifact, dict):
        raise ValueError("formal data manifest has no val artifact")
    if val_artifact.get("file") != "val_128.parquet":
        raise ValueError("formal val artifact file must be val_128.parquet")
    if val_artifact.get("rows") != args.expected_rows:
        raise ValueError(
            "formal val artifact row count does not match expected_rows"
        )
    _require_sha256(val_artifact.get("sha256"), "formal val artifact sha256")
    sample_ids = val_artifact.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != args.expected_rows:
        raise ValueError(
            "formal val artifact sample_ids must contain exactly expected_rows IDs"
        )
    sample_keys = [_sample_key(sample_id) for sample_id in sample_ids]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("formal val artifact sample_ids must be unique by type and value")
    sample_ids_digest = hashlib.sha256(
        json.dumps(
            sample_ids, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": _file_sha256(manifest_path),
        "catalog_path": catalog_path,
        "catalog_sha256": catalog_digest,
        "sample_keys": frozenset(sample_keys),
        "sample_ids_sha256": sample_ids_digest,
        "control_digest": control_digest,
        "cost_digest": cost_digest,
    }


def _active_role_arguments(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    if getattr(args, "control", None) is None:
        raise ValueError("control trace is required")
    if getattr(args, "cost_aware_gated", None) is None:
        raise ValueError("cost_aware_gated trace is required")
    roles = [ROLE_ARGUMENTS[0]]
    if getattr(args, "cost_aware_old", None) is not None:
        roles.append(ROLE_ARGUMENTS[1])
    roles.append(ROLE_ARGUMENTS[2])
    return tuple(roles)


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


def read_stage(
    path: Path,
    expected_rows: int,
    max_searches: int,
    cost_lambda: float,
    utility_tolerance: float,
) -> tuple[dict[tuple[str, str | int], dict[str, Any]], frozenset[str], str]:
    records: dict[tuple[str, str | int], dict[str, Any]] = {}
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
            key = _sample_key(record["sample_id"])
            if key in records:
                raise ValueError(f"duplicate sample_id {record['sample_id']!r} in {path}")
            records[key] = record
            stages.add(record["stage"])

    if len(records) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {path}, found {len(records)}")
    if len(stages) != 1:
        raise ValueError(f"expected one stage value in {path}, found {sorted(stages)}")
    assert schema is not None
    return records, schema, next(iter(stages))


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
    return {
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
        "mean_searches_given_wrong": _mean(
            sum(record["executed_search_count"] for record in wrong), len(wrong)
        ),
        "total_utility": total_utility,
        "utility": total_utility / len(records),
        "search_distribution": distribution,
    }


def _comparison_summary(
    paired: list[dict[str, dict[str, Any]]], candidate_role: str
) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        subset = [
            row
            for row in paired
            if _category(row["control"]["em"], row[candidate_role]["em"]) == category
        ]
        count = len(subset)
        b_searches = sum(row["control"]["executed_search_count"] for row in subset)
        candidate_searches = sum(
            row[candidate_role]["executed_search_count"] for row in subset
        )
        b_utility = sum(row["control"]["posthoc_utility"] for row in subset)
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

    baseline_correct = sum(row["control"]["em"] for row in paired)
    candidate_correct = sum(row[candidate_role]["em"] for row in paired)
    baseline_searches = sum(
        row["control"]["executed_search_count"] for row in paired
    )
    candidate_searches = sum(
        row[candidate_role]["executed_search_count"] for row in paired
    )
    baseline_utility = sum(row["control"]["posthoc_utility"] for row in paired)
    candidate_utility = sum(
        row[candidate_role]["posthoc_utility"] for row in paired
    )
    return {
        "baseline_role": "control",
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
    lines = [
        "# Paired Evaluation Summary",
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

    category_labels = {
        "both_correct": "Both correct",
        "baseline_correct_candidate_wrong": "B correct, candidate wrong",
        "baseline_wrong_candidate_correct": "B wrong, candidate correct",
        "both_wrong": "Both wrong",
    }
    for candidate_role, comparison in summary["comparisons"].items():
        lines.extend([
            "",
            f"## Control vs {candidate_role}",
            "",
            "| Category | Questions | B searches | Candidate searches | "
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
    lines.extend([
        "",
        "Search reductions in `both_correct` are lossless efficiency gains. Reductions in "
        "`baseline_correct_candidate_wrong` are harmful savings; reductions in "
        "`both_wrong` only remove unsuccessful calls.",
        "",
    ])
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

    formal_contract = _load_formal_contract(args)
    role_arguments = _active_role_arguments(args)
    active_roles = tuple(role for role, _ in role_arguments)
    candidate_roles = active_roles[1:]
    paths = {role: Path(getattr(args, role)) for role in active_roles}
    records_by_role: dict[str, dict[tuple[str, str | int], dict[str, Any]]] = {}
    schemas: dict[str, frozenset[str]] = {}
    stage_names: dict[str, str] = {}
    for role in active_roles:
        records, schema, stage_name = read_stage(
            paths[role],
            args.expected_rows,
            args.max_searches,
            args.cost_lambda,
            args.utility_tolerance,
        )
        records_by_role[role] = records
        schemas[role] = schema
        stage_names[role] = stage_name

    control_schema = schemas["control"]
    for role in candidate_roles:
        if schemas[role] != control_schema:
            raise ValueError(
                f"schema mismatch between control and {role}; "
                f"missing={sorted(control_schema - schemas[role])}, "
                f"extra={sorted(schemas[role] - control_schema)}"
            )
    if len(set(stage_names.values())) != len(stage_names):
        raise ValueError(f"stage values must be distinct across inputs: {stage_names}")
    if formal_contract is not None:
        for role, expected_stage in FORMAL_STAGE_NAMES.items():
            if stage_names[role] != expected_stage:
                raise ValueError(
                    f"formal {role} stage must be {expected_stage}, "
                    f"found {stage_names[role]}"
                )
        expected_digests = {
            "control": formal_contract["control_digest"],
            "cost_aware_gated": formal_contract["cost_digest"],
        }
        for role, expected_digest in expected_digests.items():
            for record in records_by_role[role].values():
                if record.get("checkpoint_digest") != expected_digest:
                    raise ValueError(
                        f"formal {role} checkpoint_digest does not match "
                        f"the selected endpoint for sample_id {record['sample_id']!r}"
                    )

    control_ids = set(records_by_role["control"])
    for role in candidate_roles:
        role_ids = set(records_by_role[role])
        if role_ids != control_ids:
            missing = sorted(control_ids - role_ids)[:5]
            extra = sorted(role_ids - control_ids)[:5]
            raise ValueError(
                f"sample-set mismatch between control and {role}; "
                f"missing={missing}, extra={extra}"
            )
    if (
        formal_contract is not None
        and control_ids != formal_contract["sample_keys"]
    ):
        missing = sorted(formal_contract["sample_keys"] - control_ids)[:5]
        extra = sorted(control_ids - formal_contract["sample_keys"])[:5]
        raise ValueError(
            "formal trace sample-set does not exactly match manifest val_128; "
            f"missing={missing}, extra={extra}"
        )

    catalog_argument = getattr(args, "catalog", None)
    catalog_path = Path(catalog_argument) if catalog_argument is not None else None
    if catalog_path is not None:
        catalog = read_catalog(catalog_path, args.expected_rows)
        validate_catalog_replay(records_by_role, catalog)

    paired: list[dict[str, dict[str, Any]]] = []
    for sample_key in sorted(control_ids):
        row = {role: records_by_role[role][sample_key] for role in active_roles}
        baseline = row["control"]
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

    optional_fields = [field for field in PASSTHROUGH_FIELDS if field in control_schema]
    paired_fields = ["sample_id", "question", "gold_answers"]
    for role in active_roles:
        paired_fields.extend([
            f"{role}_stage",
            f"{role}_extracted_answer",
            f"{role}_em",
            f"{role}_executed_search_count",
            f"{role}_posthoc_utility",
        ])
        paired_fields.extend(f"{role}_{field}" for field in optional_fields)
    for candidate_role in candidate_roles:
        paired_fields.extend([
            f"{candidate_role}_category_vs_control",
            f"{candidate_role}_search_delta_vs_control",
            f"{candidate_role}_utility_delta_vs_control",
        ])
    paired_csv_rows = []
    for row in paired:
        baseline = row["control"]
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
                *optional_fields,
            ):
                output[f"{role}_{field}"] = _csv_value(field, record[field])
        for candidate_role in candidate_roles:
            candidate = row[candidate_role]
            output[f"{candidate_role}_category_vs_control"] = _category(
                baseline["em"], candidate["em"]
            )
            output[f"{candidate_role}_search_delta_vs_control"] = (
                candidate["executed_search_count"] - baseline["executed_search_count"]
            )
            output[f"{candidate_role}_utility_delta_vs_control"] = (
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
        "executed_search_count",
        "posthoc_utility",
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
        baseline = row["control"]
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
                    "comparison": f"control_vs_{candidate_role}",
                    "candidate_role": candidate_role,
                    "candidate_stage": stage_names[candidate_role],
                    "category": category,
                    "control_searches": baseline_searches,
                    "candidate_searches": candidate_searches,
                    "search_delta_candidate_minus_control": (
                        candidate_searches - baseline_searches
                    ),
                    "count": transition_counts[
                        (candidate_role, category, baseline_searches, candidate_searches)
                    ],
                })

    summary: dict[str, Any] = {
        "schema_version": 1,
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
            "matched_rows": len(control_ids),
            "replay_status": "passed",
            "strict_em_scorer": "qa_em.em_check",
        }
    if formal_contract is not None:
        summary["formal_contract"] = {
            "mode": "qwen35_native_v2_b_c",
            "data_manifest": {
                "path": str(formal_contract["manifest_path"]),
                "sha256": formal_contract["manifest_sha256"],
                "schema_version": 3,
            },
            "catalog": {
                "path": str(formal_contract["catalog_path"]),
                "sha256": formal_contract["catalog_sha256"],
            },
            "val": {
                "file": "val_128.parquet",
                "rows": args.expected_rows,
                "sample_ids_sha256": formal_contract["sample_ids_sha256"],
                "sample_set_status": "exact",
            },
            "endpoints": {
                "control": {
                    "stage": FORMAL_STAGE_NAMES["control"],
                    "checkpoint_digest": formal_contract["control_digest"],
                },
                "cost_aware_gated": {
                    "stage": FORMAL_STAGE_NAMES["cost_aware_gated"],
                    "checkpoint_digest": formal_contract["cost_digest"],
                },
            },
        }
    for role in active_roles:
        stage_summary = _stage_summary([row[role] for row in paired], args.max_searches)
        summary["stages"][role] = {"stage": stage_names[role], **stage_summary}
    for candidate_role in candidate_roles:
        summary["comparisons"][candidate_role] = _comparison_summary(
            paired, candidate_role
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
            "control_searches",
            "candidate_searches",
            "search_delta_candidate_minus_control",
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
        description="Pair and summarize control and cost-aware test traces."
    )
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--cost-aware-old", dest="cost_aware_old", type=Path)
    parser.add_argument(
        "--cost-aware-gated", dest="cost_aware_gated", type=Path, required=True
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--expected-control-checkpoint-digest")
    parser.add_argument("--expected-cost-aware-gated-checkpoint-digest")
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
