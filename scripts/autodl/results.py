#!/usr/bin/env python3
"""Select validation checkpoints and summarize fixed-endpoint test runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys


EVAL_METRICS = {
    "utility": "val/utility",
    "em": "val/em",
    "search_count": "val/search_count",
    "no_search_ratio": "val/no_search_ratio",
}
UTILITY_LAMBDA = 0.10
MAX_SEARCHES = 4
UTILITY_TOLERANCE = 2e-6
PARENT_DISPLAY_NAME = "A / Parent (post-trained)"


def _metric_rows(log_path: Path, reject_duplicates: bool) -> list[tuple[int, dict[str, float]]]:
    rows = []
    for line_number, line in enumerate(log_path.read_text(errors="replace").splitlines(), 1):
        match = re.search(r"(?:^|\s)step:(\d+)(?:\s|$)", line)
        if not match:
            continue
        metrics: dict[str, float] = {}
        for token in line.split(" - "):
            if ":" not in token:
                continue
            key, raw_value = token.rsplit(":", 1)
            key = key.strip()
            try:
                value = float(raw_value.strip())
            except ValueError:
                continue
            if reject_duplicates and key in metrics:
                raise ValueError(f"duplicate metric {key!r} in {log_path}:{line_number}")
            metrics[key] = value
        rows.append((int(match.group(1)), metrics))
    return rows


def metric_rows(log_path: Path) -> list[tuple[int, dict[str, float]]]:
    return _metric_rows(log_path, reject_duplicates=False)


def mean_prefix(metrics: dict[str, float], prefix: str) -> float | None:
    prefix = prefix.rstrip("/")
    values = [
        value for key, value in metrics.items()
        if key == prefix or key.startswith(f"{prefix}/")
    ]
    return sum(values) / len(values) if values else None


def select(args: argparse.Namespace) -> None:
    candidates = []
    score_prefix = "val/em/" if args.variant == "baseline" else "val/utility/"
    for step, metrics in metric_rows(args.log):
        checkpoint = args.checkpoint_root / "actor" / f"global_step_{step}"
        score = mean_prefix(metrics, score_prefix)
        if score is None or not checkpoint.is_dir():
            continue
        searches = mean_prefix(metrics, "val/search_count/")
        candidates.append((step, score, math.inf if searches is None else searches, checkpoint))
    if not candidates:
        raise ValueError(f"no saved checkpoint has a {score_prefix} metric in {args.log}")
    if args.variant == "baseline":
        chosen = max(candidates, key=lambda item: (item[1], -item[0]))
    else:
        chosen = max(candidates, key=lambda item: (item[1], -item[2], -item[0]))
    step, score, searches, checkpoint = chosen
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "score": score,
        "score_metric": score_prefix,
        "search_count": None if math.isinf(searches) else searches,
        "step": step,
        "variant": args.variant,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_run_metadata(path: Path) -> dict[str, str]:
    values = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"invalid metadata line in {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"duplicate or empty metadata key in {path}:{line_number}")
        values[key] = value
    return values


def final_metrics(log_path: Path) -> dict[str, float]:
    evaluation_rows = []
    for step, metrics in _metric_rows(log_path, reject_duplicates=True):
        if any(mean_prefix(metrics, prefix) is not None for prefix in EVAL_METRICS.values()):
            evaluation_rows.append((step, metrics))
    if len(evaluation_rows) != 1:
        raise ValueError(
            f"expected exactly one evaluation record in {log_path}, found {len(evaluation_rows)}"
        )

    _, metrics = evaluation_rows[0]
    result = {}
    missing = []
    for name, prefix in EVAL_METRICS.items():
        value = mean_prefix(metrics, prefix)
        if value is None:
            missing.append(prefix)
        elif not math.isfinite(value):
            raise ValueError(f"non-finite {prefix} metric in {log_path}")
        else:
            result[name] = value
    if missing:
        raise ValueError(f"evaluation record in {log_path} is missing: {', '.join(missing)}")
    expected_utility = (
        result["em"] - UTILITY_LAMBDA * result["search_count"] / MAX_SEARCHES
    )
    if not math.isclose(
        result["utility"], expected_utility, rel_tol=0.0, abs_tol=UTILITY_TOLERANCE
    ):
        raise ValueError(
            f"utility in {log_path} is inconsistent with lambda={UTILITY_LAMBDA:.2f}: "
            f"found {result['utility']}, expected {expected_utility}"
        )
    return result


def training_fields(path: Path, expected_role: str) -> dict[str, str | int | float]:
    metadata = read_run_metadata(path)
    required = {
        "checkpoint",
        "checkpoint_digest",
        "checkout_commit",
        "cost_lambda",
        "cpu_handoff_digest",
        "elapsed_seconds",
        "gpu_count",
        "parent_checkpoint",
        "parent_checkpoint_digest",
        "price_per_hour",
        "resolved_config_sha256",
        "role",
        "seed",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"metadata {path} is missing: {', '.join(missing)}")
    if metadata["role"] != expected_role:
        raise ValueError(
            f"metadata {path} has role {metadata['role']!r}, expected {expected_role!r}"
        )
    provenance_keys = (
        "checkpoint",
        "checkpoint_digest",
        "checkout_commit",
        "cpu_handoff_digest",
        "parent_checkpoint",
        "parent_checkpoint_digest",
        "resolved_config_sha256",
    )
    if any(not metadata[key] for key in provenance_keys):
        raise ValueError(f"metadata {path} has an empty provenance field")

    elapsed = int(metadata["elapsed_seconds"])
    gpu_count = int(metadata["gpu_count"])
    price = float(metadata["price_per_hour"])
    seed = int(metadata["seed"])
    cost_lambda = float(metadata["cost_lambda"])
    expected_lambda = 0.10 if expected_role == "cost_aware" else 0.0
    if not math.isclose(cost_lambda, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"metadata {path} has cost_lambda {cost_lambda}, expected {expected_lambda}"
        )
    if (
        elapsed < 0
        or gpu_count <= 0
        or not math.isfinite(price)
        or price < 0
        or not math.isfinite(cost_lambda)
    ):
        raise ValueError(f"metadata {path} has invalid training resource values")
    return {
        "checkpoint": metadata["checkpoint"],
        "checkpoint_digest": metadata["checkpoint_digest"],
        "checkout_commit": metadata["checkout_commit"],
        "cost_lambda": cost_lambda,
        "cpu_handoff_digest": metadata["cpu_handoff_digest"],
        "parent_checkpoint": metadata["parent_checkpoint"],
        "parent_checkpoint_digest": metadata["parent_checkpoint_digest"],
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "seed": seed,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed * gpu_count / 3600,
        "actual_rmb": elapsed * price / 3600,
    }


def summarize(args: argparse.Namespace) -> None:
    run_fields = {
        "reproduced": training_fields(args.reproduced_run_metadata, "reproduced"),
        "control": training_fields(args.control_run_metadata, "control"),
        "cost_aware": training_fields(args.cost_aware_run_metadata, "cost_aware"),
    }
    base_checkpoint = str(args.base_checkpoint)
    base_checkpoint_digest = str(args.base_checkpoint_digest)
    if not base_checkpoint or not base_checkpoint_digest:
        raise ValueError("base checkpoint path and digest must not be empty")
    reproduced_checkpoint = run_fields["reproduced"]["checkpoint"]
    if run_fields["reproduced"]["parent_checkpoint"] != base_checkpoint:
        raise ValueError("reproduced parent checkpoint does not match the base checkpoint")
    if run_fields["reproduced"]["parent_checkpoint_digest"] != base_checkpoint_digest:
        raise ValueError("reproduced parent digest does not match the base checkpoint digest")
    for model in ("control", "cost_aware"):
        if run_fields[model]["parent_checkpoint"] != reproduced_checkpoint:
            raise ValueError(
                f"{model} parent checkpoint does not match the reproduced checkpoint"
            )
        if (
            run_fields[model]["parent_checkpoint_digest"]
            != run_fields["reproduced"]["checkpoint_digest"]
        ):
            raise ValueError(
                f"{model} parent digest does not match the reproduced checkpoint digest"
            )
    for provenance in ("checkout_commit", "cpu_handoff_digest"):
        values = {fields[provenance] for fields in run_fields.values()}
        if len(values) != 1:
            raise ValueError(f"R/B/C metadata disagree on {provenance}")
    if run_fields["control"]["seed"] != run_fields["cost_aware"]["seed"]:
        raise ValueError("control and cost-aware metadata disagree on seed")

    labels = {
        "base": PARENT_DISPLAY_NAME,
        "reproduced": "R / Reproduced",
        "control": "B / Control",
        "cost_aware": "C / Cost-aware",
    }
    eval_logs = {
        "base": args.base_log,
        "reproduced": args.reproduced_log,
        "control": args.control_log,
        "cost_aware": args.cost_aware_log,
    }
    output_rows = []
    for stage, model in zip("ARBC", ("base", "reproduced", "control", "cost_aware")):
        metrics = final_metrics(eval_logs[model])
        resource_fields = run_fields.get(model, {
            "checkpoint": base_checkpoint,
            "checkpoint_digest": base_checkpoint_digest,
            "checkout_commit": "",
            "cost_lambda": "",
            "cpu_handoff_digest": "",
            "parent_checkpoint": "",
            "parent_checkpoint_digest": "",
            "resolved_config_sha256": "",
            "seed": "",
            "elapsed_seconds": 0,
            "gpu_hours": 0.0,
            "actual_rmb": 0.0,
        })
        output_rows.append({"stage": stage, "model": model, **metrics, **resource_fields})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "model",
        "checkpoint",
        "checkpoint_digest",
        "parent_checkpoint",
        "parent_checkpoint_digest",
        "checkout_commit",
        "cpu_handoff_digest",
        "resolved_config_sha256",
        "seed",
        "cost_lambda",
        "em",
        "search_count",
        "no_search_ratio",
        "utility",
        "elapsed_seconds",
        "gpu_hours",
        "actual_rmb",
    ]
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    lines = [
        "# Test-128 Results",
        "",
        "| Model | NQ EM | Avg searches | No-search ratio | Utility (lambda=0.10) | Train seconds | GPU hours | RMB | Checkpoint | Parent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in output_rows:
        lines.append(
            f"| {labels[row['model']]} | {row['em']:.3f} | {row['search_count']:.3f} | "
            f"{row['no_search_ratio']:.3f} | {row['utility']:.3f} | "
            f"{row['elapsed_seconds']} | {row['gpu_hours']:.2f} | {row['actual_rmb']:.2f} | "
            f"{row['checkpoint']} | {row['parent_checkpoint'] or '-'} |"
        )
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--variant", choices=("baseline", "cost_aware"), required=True)
    choose.add_argument("--log", type=Path, required=True)
    choose.add_argument("--checkpoint-root", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    choose.set_defaults(handler=select)
    summary = commands.add_parser("summarize")
    summary.add_argument("--base-checkpoint", type=Path, required=True)
    summary.add_argument("--base-checkpoint-digest", required=True)
    summary.add_argument("--base-log", type=Path, required=True)
    summary.add_argument("--reproduced-log", type=Path, required=True)
    summary.add_argument("--control-log", type=Path, required=True)
    summary.add_argument("--cost-aware-log", type=Path, required=True)
    summary.add_argument("--reproduced-run-metadata", type=Path, required=True)
    summary.add_argument("--control-run-metadata", type=Path, required=True)
    summary.add_argument("--cost-aware-run-metadata", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.set_defaults(handler=summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (KeyError, OSError, ValueError) as error:
        print(f"results error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
