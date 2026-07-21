#!/usr/bin/env python3
"""Validate and pair B/C-old/C-gated per-question evaluation traces."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


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


def _sample_key(sample_id: str | int) -> tuple[str, str | int]:
    """Keep integer and string IDs distinct while making them sortable."""
    if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
        raise ValueError(f"sample_id must be a string or integer, found {sample_id!r}")
    if isinstance(sample_id, str):
        if not sample_id:
            raise ValueError("sample_id must not be empty")
        return ("str", sample_id)
    return ("int", sample_id)


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
        f"All three stages contain the same {summary['expected_rows']} test samples. "
        f"Utility is `EM - {summary['cost_lambda']:.2f} * searches / "
        f"{summary['max_searches']}`.",
        "",
        "## Stage Totals",
        "",
        "| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role, _ in ROLE_ARGUMENTS:
        stage = summary["stages"][role]
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
    for candidate_role in ("cost_aware_old", "cost_aware_gated"):
        comparison = summary["comparisons"][candidate_role]
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

    paths = {role: Path(getattr(args, role)) for role, _ in ROLE_ARGUMENTS}
    records_by_role: dict[str, dict[tuple[str, str | int], dict[str, Any]]] = {}
    schemas: dict[str, frozenset[str]] = {}
    stage_names: dict[str, str] = {}
    for role, _ in ROLE_ARGUMENTS:
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
    for role, _ in ROLE_ARGUMENTS[1:]:
        if schemas[role] != control_schema:
            raise ValueError(
                f"schema mismatch between control and {role}; "
                f"missing={sorted(control_schema - schemas[role])}, "
                f"extra={sorted(schemas[role] - control_schema)}"
            )
    if len(set(stage_names.values())) != len(stage_names):
        raise ValueError(f"stage values must be distinct across inputs: {stage_names}")

    control_ids = set(records_by_role["control"])
    for role, _ in ROLE_ARGUMENTS[1:]:
        role_ids = set(records_by_role[role])
        if role_ids != control_ids:
            missing = sorted(control_ids - role_ids)[:5]
            extra = sorted(role_ids - control_ids)[:5]
            raise ValueError(
                f"sample-set mismatch between control and {role}; "
                f"missing={missing}, extra={extra}"
            )

    paired: list[dict[str, dict[str, Any]]] = []
    for sample_key in sorted(control_ids):
        row = {role: records_by_role[role][sample_key] for role, _ in ROLE_ARGUMENTS}
        baseline = row["control"]
        for role, _ in ROLE_ARGUMENTS[1:]:
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
    for role, _ in ROLE_ARGUMENTS:
        paired_fields.extend([
            f"{role}_stage",
            f"{role}_extracted_answer",
            f"{role}_em",
            f"{role}_executed_search_count",
            f"{role}_posthoc_utility",
        ])
        paired_fields.extend(f"{role}_{field}" for field in optional_fields)
    paired_fields.extend([
        "cost_aware_old_category_vs_control",
        "cost_aware_old_search_delta_vs_control",
        "cost_aware_old_utility_delta_vs_control",
        "cost_aware_gated_category_vs_control",
        "cost_aware_gated_search_delta_vs_control",
        "cost_aware_gated_utility_delta_vs_control",
    ])
    paired_csv_rows = []
    for row in paired:
        baseline = row["control"]
        output: dict[str, Any] = {
            "sample_id": baseline["sample_id"],
            "question": baseline["question"],
            "gold_answers": _csv_value("gold_answers", baseline["gold_answers"]),
        }
        for role, _ in ROLE_ARGUMENTS:
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
        for candidate_role in ("cost_aware_old", "cost_aware_gated"):
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
        for role, _ in ROLE_ARGUMENTS:
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
        for candidate_role in ("cost_aware_old", "cost_aware_gated"):
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
    for candidate_role in ("cost_aware_old", "cost_aware_gated"):
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
            for role, _ in ROLE_ARGUMENTS
        },
        "stages": {},
        "comparisons": {},
    }
    for role, _ in ROLE_ARGUMENTS:
        stage_summary = _stage_summary([row[role] for row in paired], args.max_searches)
        summary["stages"][role] = {"stage": stage_names[role], **stage_summary}
    for candidate_role in ("cost_aware_old", "cost_aware_gated"):
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
        description="Pair and summarize B, C-old, and C-gated test traces."
    )
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--cost-aware-old", dest="cost_aware_old", type=Path, required=True)
    parser.add_argument(
        "--cost-aware-gated", dest="cost_aware_gated", type=Path, required=True
    )
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
