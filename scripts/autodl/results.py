#!/usr/bin/env python3
"""Select validation checkpoints and summarize the two fixed test runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys


def metric_rows(log_path: Path) -> list[tuple[int, dict[str, float]]]:
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
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
                metrics[key] = float(raw_value.strip())
            except ValueError:
                continue
        rows.append((int(match.group(1)), metrics))
    return rows


def mean_prefix(metrics: dict[str, float], prefix: str) -> float | None:
    values = [value for key, value in metrics.items() if key.startswith(prefix)]
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
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def final_metrics(log_path: Path) -> dict[str, float]:
    for _, metrics in reversed(metric_rows(log_path)):
        em = mean_prefix(metrics, "val/em/")
        if em is not None:
            search_count = mean_prefix(metrics, "val/search_count/")
            no_search_ratio = mean_prefix(metrics, "val/no_search_ratio/")
            utility = mean_prefix(metrics, "val/utility/")
            return {
                "em": em,
                "search_count": 0.0 if search_count is None else search_count,
                "no_search_ratio": 0.0 if no_search_ratio is None else no_search_ratio,
                "utility": em if utility is None else utility,
            }
    raise ValueError(f"no test metrics found in {log_path}")


def summarize(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for variant, eval_log, run_metadata in (
        ("baseline", args.baseline_log, args.baseline_run_metadata),
        ("cost_aware", args.cost_log, args.cost_run_metadata),
    ):
        metrics = final_metrics(eval_log)
        metadata = read_run_metadata(run_metadata)
        elapsed = int(metadata["elapsed_seconds"])
        gpu_count = int(metadata["gpu_count"])
        price = metadata.get("price_per_hour", "")
        metrics.update({
            "model": variant,
            "gpu_hours": elapsed * gpu_count / 3600,
            "actual_rmb": "" if not price else elapsed * float(price) / 3600,
        })
        output_rows.append(metrics)

    fields = ["model", "em", "search_count", "no_search_ratio", "utility", "gpu_hours", "actual_rmb"]
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    lines = [
        "# Test-128 Results",
        "",
        "| Model | NQ EM | Avg searches | No-search ratio | Utility | GPU hours | RMB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output_rows:
        rmb = "n/a" if row["actual_rmb"] == "" else f"{row['actual_rmb']:.2f}"
        lines.append(
            f"| {row['model']} | {row['em']:.3f} | {row['search_count']:.3f} | "
            f"{row['no_search_ratio']:.3f} | {row['utility']:.3f} | {row['gpu_hours']:.2f} | {rmb} |"
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
    summary.add_argument("--baseline-log", type=Path, required=True)
    summary.add_argument("--cost-log", type=Path, required=True)
    summary.add_argument("--baseline-run-metadata", type=Path, required=True)
    summary.add_argument("--cost-run-metadata", type=Path, required=True)
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
