#!/usr/bin/env python3
"""Export compact CSV and plots from the completed AutoDL experiment logs."""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_LINE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)$")
COMPARISON_STAGE_DESCRIPTION = (
    "A=Parent (post-trained), R=Reproduced, B=Control, C=Cost-aware"
)

STAGES = (
    ("R", "Reproduced", "reproduced_log", 60),
    ("B", "Control", "control_log", 20),
    ("C", "Cost-aware", "cost_aware_log", 20),
)

CSV_FIELDS = (
    "stage",
    "model",
    "step",
    "actor_pg_loss",
    "actor_kl_loss",
    "actor_entropy_loss",
    "actor_grad_norm",
    "actor_lr",
    "em",
    "reward",
    "avg_searches",
    "no_search_ratio",
    "search_cost",
    "posthoc_utility",
    "response_length",
    "response_clip_ratio",
    "step_seconds",
    "val_em",
    "val_avg_searches",
    "val_no_search_ratio",
    "val_utility",
)


def _metric(metrics: dict[str, float], key: str) -> float:
    try:
        return metrics[key]
    except KeyError as error:
        raise ValueError(f"training metric is missing: {key}") from error


def parse_training_log(
        path: Path,
        stage: str,
        model: str,
        cost_lambda: float,
        max_searches: int,
        expected_steps: int | None = None) -> list[dict[str, Any]]:
    """Parse the one aggregate training record emitted for each completed step."""
    rows: list[dict[str, Any]] = []
    seen_steps: set[int] = set()

    for raw_line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        match = STEP_LINE.search(line)
        if not match:
            continue

        metrics: dict[str, float] = {}
        for item in match.group(2).split(" - "):
            key, separator, value = item.rpartition(":")
            if not separator:
                continue
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue

        # Validation-only records do not contain an actor update.
        if "actor/pg_loss" not in metrics:
            continue

        step = int(match.group(1))
        if step in seen_steps:
            raise ValueError(f"duplicate completed step {step} in {path}")
        seen_steps.add(step)

        em = _metric(metrics, "env/em/mean")
        avg_searches = _metric(metrics, "env/executed_search_count/mean")
        row: dict[str, Any] = {
            "stage": stage,
            "model": model,
            "step": step,
            "actor_pg_loss": _metric(metrics, "actor/pg_loss"),
            "actor_kl_loss": _metric(metrics, "actor/kl_loss"),
            "actor_entropy_loss": _metric(metrics, "actor/entropy_loss"),
            "actor_grad_norm": _metric(metrics, "actor/grad_norm"),
            "actor_lr": _metric(metrics, "actor/lr"),
            "em": em,
            "reward": _metric(metrics, "critic/rewards/mean"),
            "avg_searches": avg_searches,
            "no_search_ratio": _metric(metrics, "env/no_search_ratio"),
            "search_cost": _metric(metrics, "env/search_cost/mean"),
            "posthoc_utility": em - cost_lambda * avg_searches / max_searches,
            "response_length": _metric(metrics, "response_length/mean"),
            "response_clip_ratio": _metric(metrics,
                                           "response_length/clip_ratio"),
            "step_seconds": _metric(metrics, "timing_s/step"),
            "val_em": metrics.get("val/em/nq", ""),
            "val_avg_searches": metrics.get("val/search_count/nq", ""),
            "val_no_search_ratio": metrics.get("val/no_search_ratio/nq", ""),
            "val_utility": metrics.get("val/utility/nq", ""),
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"no completed training steps found in {path}")
    steps = [row["step"] for row in rows]
    last_step = expected_steps if expected_steps is not None else steps[-1]
    if steps != list(range(1, last_step + 1)):
        raise ValueError(f"expected completed steps 1..{last_step} in {path}")
    return rows


def read_comparison(path: Path, cost_lambda: float,
                    max_searches: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row.get("stage") for row in rows] != list("ARBC"):
        raise ValueError("comparison CSV must contain A/R/B/C in order")
    for row in rows:
        for field in ("em", "search_count", "no_search_ratio", "utility"):
            row[field] = float(row[field])
        if not all(
                math.isfinite(row[field])
                for field in ("em", "search_count", "no_search_ratio",
                              "utility")):
            raise ValueError("comparison CSV contains a non-finite metric")
        expected_utility = row[
            "em"] - cost_lambda * row["search_count"] / max_searches
        if not math.isclose(
                row["utility"], expected_utility, rel_tol=0.0, abs_tol=2e-6):
            raise ValueError(
                "comparison utility does not match the requested formula")
    return rows


def write_training_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _annotate_bars(axis: Any, bars: Any, digits: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_training(path: Path, rows: list[dict[str, Any]],
                  cost_lambda: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"R": "#457b9d", "B": "#e9a23b", "C": "#df684d"}
    panels = (
        ("actor_pg_loss", "GRPO policy loss"),
        ("actor_kl_loss", "Reference KL loss"),
        ("actor_entropy_loss", "Policy entropy"),
        ("em", "Train-batch EM"),
        ("avg_searches", "Executed searches / prompt"),
        ("posthoc_utility", f"Post-hoc utility (lambda={cost_lambda:.2f})"),
    )
    figure, axes = plt.subplots(2,
                                3,
                                figsize=(14, 7.5),
                                constrained_layout=True)

    for axis, (field, title) in zip(axes.flat, panels):
        for stage, model, _, _ in STAGES:
            stage_rows = [row for row in rows if row["stage"] == stage]
            axis.plot(
                [row["step"] for row in stage_rows],
                [row[field] for row in stage_rows],
                color=colors[stage],
                linewidth=1.5,
                marker="o",
                markersize=2.5,
                alpha=0.9,
                label=f"{stage} / {model}",
            )
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Stage-local training step")
        axis.grid(alpha=0.2, linewidth=0.7)

    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Search-R1-small training dynamics (raw per-step aggregates)",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(path: Path, rows: list[dict[str, Any]],
                    cost_lambda: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["stage"] for row in rows]
    colors = ["#8d99ae", "#457b9d", "#e9a23b", "#df684d"]
    panels = (
        ("em", "NQ Exact Match", 3),
        ("search_count", "Average executed searches", 3),
        ("no_search_ratio", "No-search ratio", 3),
        ("utility", f"Utility (lambda={cost_lambda:.2f})", 3),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for axis, (field, title, digits) in zip(axes.flat, panels):
        bars = axis.bar(labels, [row[field] for row in rows],
                        color=colors,
                        width=0.68)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel(COMPARISON_STAGE_DESCRIPTION)
        axis.grid(axis="y", alpha=0.2, linewidth=0.7)
        _annotate_bars(axis, bars, digits)
        maximum = max(float(row[field]) for row in rows)
        axis.set_ylim(0, maximum * 1.18 if maximum > 0 else 1)

    figure.suptitle("Search-R1-small test-128 comparison",
                    fontsize=15,
                    fontweight="bold")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduced-log", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--cost-aware-log", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-lambda", type=float, default=0.10)
    parser.add_argument("--max-searches", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cost_lambda < 0 or args.max_searches <= 0:
        raise SystemExit(
            "cost lambda must be non-negative and max searches must be positive"
        )

    rows: list[dict[str, Any]] = []
    for stage, model, argument, expected_steps in STAGES:
        rows.extend(
            parse_training_log(
                getattr(args, argument),
                stage,
                model,
                args.cost_lambda,
                args.max_searches,
                expected_steps,
            ))

    comparison = read_comparison(args.results_csv, args.cost_lambda,
                                 args.max_searches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_training_csv(args.output_dir / "training_metrics.csv", rows)
    plot_training(args.output_dir / "training_curves.png", rows,
                  args.cost_lambda)
    plot_comparison(args.output_dir / "final_comparison.png", comparison,
                    args.cost_lambda)
    print(f"Exported {len(rows)} training rows to {args.output_dir}")


if __name__ == "__main__":
    main()
