#!/usr/bin/env python3
"""Render the single-column Evidence Capacity sensitivity figure."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "map2trace-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "evidence_capacity_data.csv"
DEFAULT_OUTPUT_DIR = HERE / "figures"


@dataclass(frozen=True)
class CapacityPoint:
    family: str
    code: str
    policy: str
    macro_f1: float
    mean_prompt_tokens: float
    status: str


FAMILY_ORDER = ("fixed", "minmax", "budget")
FAMILY_STYLE = {
    "fixed": {
        "marker": "o",
        "legend": "Fixed",
        "color": "#E69F00",
        "edge": "#9C6B00",
        "linestyle": "-",
    },
    "minmax": {
        "marker": "D",
        "legend": "Minmax",
        "color": "#0072B2",
        "edge": "#00537A",
        "linestyle": "-",
    },
    "budget": {
        "marker": "^",
        "legend": "Budget",
        "color": "#009E73",
        "edge": "#006B4E",
        "linestyle": "--",
    },
}
BEST_BLUE = "#0072B2"
FIGURE_WIDTH_PT = 285.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_PT / 72.0

CONFIG_LABEL = {
    "F3": "F3",
    "F5": "F5",
    "F7": "F7",
    "F9": "F9",
    "M3-8": "M3–8",
    "M3-10": "M3–10",
    "M5-10": "M5–10",
    "M5-12": "M5–12",
    "M7-12": "M7–12",
    "B512": "B512",
    "B768": "B768",
    "B1024": "B1024",
}

CONFIG_OFFSET = {
    "F3": (4, 5),
    "F5": (5, 5),
    "F7": (5, -8),
    "F9": (-5, 6),
    "M3-8": (-4, 7),
    "M3-10": (4, -8),
    "M5-10": (-7, 5),
    "M5-12": (5, 5),
    "M7-12": (6, -9),
    "B512": (5, 5),
    "B768": (-5, 6),
    "B1024": (-5, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stem",
        default="evidence_capacity_sensitivity",
        help="Output filename stem (default: evidence_capacity_sensitivity).",
    )
    return parser.parse_args()


def load_points(path: Path) -> list[CapacityPoint]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    points = [
        CapacityPoint(
            family=row["family"].strip(),
            code=row["code"].strip(),
            policy=row["policy"].strip(),
            macro_f1=float(row["macro_f1"]),
            mean_prompt_tokens=float(row["mean_prompt_tokens"]),
            status=row["status"].strip().lower(),
        )
        for row in rows
    ]

    unknown = sorted({point.family for point in points} - set(FAMILY_ORDER))
    if unknown:
        raise ValueError(f"Unknown policy families: {unknown}")
    return points


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def draw(points: list[CapacityPoint], output_dir: Path, stem: str) -> list[Path]:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.62))

    for family in FAMILY_ORDER:
        family_points = sorted(
            (point for point in points if point.family == family),
            key=lambda point: point.mean_prompt_tokens,
        )
        style = FAMILY_STYLE[family]
        ax.plot(
            [point.mean_prompt_tokens for point in family_points],
            [point.macro_f1 for point in family_points],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.05,
            alpha=0.68,
            zorder=1,
        )

    for point in points:
        if point.code == "M5-10":
            continue
        interpolated = point.status == "interpolated"
        style = FAMILY_STYLE[point.family]
        ax.scatter(
            [point.mean_prompt_tokens],
            [point.macro_f1],
            s=31,
            marker=style["marker"],
            facecolor="white" if interpolated else style["color"],
            edgecolor=style["edge"],
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )

    best = next(point for point in points if point.code == "M5-10")
    ax.scatter(
        [best.mean_prompt_tokens],
        [best.macro_f1],
        s=112,
        marker="*",
        facecolor=BEST_BLUE,
        edgecolor="#00537A",
        linewidth=0.7,
        zorder=5,
    )

    for point in points:
        dx, dy = CONFIG_OFFSET[point.code]
        annotation = ax.annotate(
            CONFIG_LABEL[point.code],
            xy=(point.mean_prompt_tokens, point.macro_f1),
            xytext=(dx, dy),
            textcoords="offset points",
            color=FAMILY_STYLE[point.family]["edge"],
            fontsize=6.5,
            fontweight="normal",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top",
            zorder=6,
        )
        annotation.set_path_effects(
            [
                path_effects.withStroke(linewidth=1.7, foreground="white"),
                path_effects.Normal(),
            ]
        )

    ax.set_xlim(400, 1020)
    ax.set_ylim(32.5, 37.05)
    ax.set_xlabel("Mean prompt tokens")
    ax.set_ylabel("Macro-F1 (%)")
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.tick_params(direction="in", length=3.0, width=0.65, pad=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [],
            [],
            color=FAMILY_STYLE[family]["color"],
            linestyle=FAMILY_STYLE[family]["linestyle"],
            linewidth=1.05,
            marker=FAMILY_STYLE[family]["marker"],
            markerfacecolor=FAMILY_STYLE[family]["color"],
            markeredgecolor=FAMILY_STYLE[family]["edge"],
            markeredgewidth=0.8,
            label=FAMILY_STYLE[family]["legend"],
        )
        for family in FAMILY_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.30,
        columnspacing=0.85,
    )

    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.21, top=0.84)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(outputs[0], facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = draw(load_points(args.data), args.output_dir, args.stem)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
