#!/usr/bin/env python3
"""Export C-gated per-step metrics and a dependency-free SVG curve sheet."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from export_training_curves import parse_training_log, write_training_csv


PANELS = (
    ("actor_pg_loss", "GRPO policy loss"),
    ("actor_kl_loss", "Reference KL loss"),
    ("em", "Train-batch EM"),
    ("avg_searches", "Executed searches / prompt"),
    ("posthoc_utility", "Post-hoc utility (lambda=0.10)"),
    ("reward", "Gated train reward"),
)


def render_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1200, 720
    panel_width, panel_height = 380, 300
    margin_x, margin_y = 15, 55
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f4ed"/>',
        '<style>text{font-family:Georgia,serif;fill:#202522}.title{font-size:24px;font-weight:bold}'
        '.label{font-size:14px;font-weight:bold}.tick{font-size:10px;fill:#59615d}'
        '.grid{stroke:#d8d4ca;stroke-width:1}.line{fill:none;stroke:#176b57;stroke-width:2.5}</style>',
        '<text class="title" x="24" y="34">C-gated training dynamics (raw per-step aggregates)</text>',
    ]
    for index, (field, title) in enumerate(PANELS):
        column, row_index = index % 3, index // 3
        x0 = margin_x + column * 395
        y0 = margin_y + row_index * 325
        plot_left, plot_top = x0 + 50, y0 + 35
        plot_width, plot_height = panel_width - 70, panel_height - 75
        values = [float(row[field]) for row in rows]
        minimum, maximum = min(values), max(values)
        if maximum == minimum:
            padding = max(abs(maximum) * 0.05, 0.5)
            minimum -= padding
            maximum += padding
        else:
            padding = (maximum - minimum) * 0.08
            minimum -= padding
            maximum += padding
        pieces.append(f'<text class="label" x="{x0 + 8}" y="{y0 + 18}">{escape(title)}</text>')
        for tick in range(5):
            y = plot_top + tick * plot_height / 4
            value = maximum - tick * (maximum - minimum) / 4
            pieces.append(
                f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" '
                f'x2="{plot_left + plot_width}" y2="{y:.1f}"/>'
            )
            pieces.append(
                f'<text class="tick" x="{plot_left - 5}" y="{y + 3:.1f}" '
                f'text-anchor="end">{value:.3f}</text>'
            )
        points = []
        for position, item in enumerate(values):
            x = plot_left + position * plot_width / max(len(values) - 1, 1)
            y = plot_top + (maximum - item) * plot_height / (maximum - minimum)
            points.append(f"{x:.1f},{y:.1f}")
        pieces.append(f'<polyline class="line" points="{" ".join(points)}"/>')
        pieces.append(
            f'<text class="tick" x="{plot_left}" y="{plot_top + plot_height + 18}">step 1</text>'
        )
        pieces.append(
            f'<text class="tick" x="{plot_left + plot_width}" y="{plot_top + plot_height + 18}" '
            f'text-anchor="end">step {len(rows)}</text>'
        )
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_training_log(
        args.log, "C-gated", "Cost-aware gated", 0.10, 4, expected_steps=20
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_training_csv(args.output_dir / "gated_training_metrics.csv", rows)
    render_svg(args.output_dir / "gated_training_curves.svg", rows)
    print(f"Exported {len(rows)} C-gated steps to {args.output_dir}")


if __name__ == "__main__":
    main()
