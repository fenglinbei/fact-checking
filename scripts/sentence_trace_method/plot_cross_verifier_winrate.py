#!/usr/bin/env python3
"""Plot conditional verifier win rates and primary Macro-F1 effects."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "evitrace-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    REPO_ROOT
    / "outputs/analysis/evitrace_cross_verifier_finetune_v1/analysis/metrics.json"
)
DEFAULT_OUTPUT_PREFIX = (
    REPO_ROOT
    / "outputs/analysis/evitrace_cross_verifier_finetune_v1/analysis/"
    "figure_winrate_panel"
)
DEFAULT_SINGLE_COLUMN_OUTPUT_PREFIX = (
    REPO_ROOT
    / "outputs/analysis/evitrace_cross_verifier_finetune_v1/analysis/"
    "figure_winrate_panel_singlecol"
)

EVITRACE_COLOR = "#0072B2"
S4_COLOR = "#D55E00"
NEUTRAL_COLOR = "#5F6368"
GRID_COLOR = "#D9D9D9"


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    conditional_evitrace_win_pct: float
    non_tie_pct: float
    tie_pct: float
    macro_f1_delta_pp: float
    macro_f1_ci95_pp: tuple[float, float]
    holm_pvalue: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the EviTrace conditional win-rate panel and the paired "
            "Macro-F1 effect-size panel."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help=f"Formal analysis metrics JSON (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path without a suffix; .pdf, .svg, and .png are written "
            "(default depends on --layout)"
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("wide", "single-column"),
        default="wide",
        help=(
            "Figure layout: the original two-column-wide version or a "
            "compact single-column paper version "
            "(default: wide)"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution (default: 400)",
    )
    return parser.parse_args()


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _load_comparison(
    metrics: dict[str, Any],
    *,
    key: str,
    name: str,
) -> ComparisonResult:
    primary = _require_mapping(metrics.get("primary"), "primary")
    result = _require_mapping(primary.get(key), f"primary.{key}")
    point = _require_mapping(result.get("point"), f"primary.{key}.point")
    panel = _require_mapping(point.get("panel"), f"primary.{key}.point.panel")
    rates = _require_mapping(
        panel.get("wlt_rates"),
        f"primary.{key}.point.panel.wlt_rates",
    )
    delta = _require_mapping(
        panel.get("delta"),
        f"primary.{key}.point.panel.delta",
    )
    bootstrap = _require_mapping(
        result.get("bootstrap"),
        f"primary.{key}.bootstrap",
    )
    ci95 = _require_mapping(
        bootstrap.get("ci95"),
        f"primary.{key}.bootstrap.ci95",
    )
    randomization = _require_mapping(
        result.get("randomization"),
        f"primary.{key}.randomization",
    )

    conditional = _require_number(
        rates.get("conditional_evitrace_win_rate"),
        f"primary.{key}.point.panel.wlt_rates.conditional_evitrace_win_rate",
    )
    evitrace_win = _require_number(
        rates.get("evitrace_win"),
        f"primary.{key}.point.panel.wlt_rates.evitrace_win",
    )
    s4_win = _require_number(
        rates.get("s4_win"),
        f"primary.{key}.point.panel.wlt_rates.s4_win",
    )
    tie = _require_number(
        rates.get("tie"),
        f"primary.{key}.point.panel.wlt_rates.tie",
    )
    ci = ci95.get("macro_f1_delta")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in ci
        )
    ):
        raise ValueError(
            f"primary.{key}.bootstrap.ci95.macro_f1_delta must be two numbers"
        )

    for label, value in {
        "conditional win rate": conditional,
        "EviTrace win rate": evitrace_win,
        "S4 win rate": s4_win,
        "tie rate": tie,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} {label} must be in [0, 1], got {value}")
    if not np.isclose(evitrace_win + s4_win + tie, 1.0, atol=1e-8):
        raise ValueError(f"{name} W/L/T rates do not sum to one")

    return ComparisonResult(
        name=name,
        conditional_evitrace_win_pct=100.0 * conditional,
        non_tie_pct=100.0 * (evitrace_win + s4_win),
        tie_pct=100.0 * tie,
        macro_f1_delta_pp=100.0
        * _require_number(
            delta.get("macro_f1"),
            f"primary.{key}.point.panel.delta.macro_f1",
        ),
        macro_f1_ci95_pp=(100.0 * float(ci[0]), 100.0 * float(ci[1])),
        holm_pvalue=_require_number(
            randomization.get("holm_pvalue"),
            f"primary.{key}.randomization.holm_pvalue",
        ),
    )


def load_results(metrics_path: Path) -> list[ComparisonResult]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics = _require_mapping(metrics, "metrics")
    return [
        _load_comparison(metrics, key="main", name="Main"),
        _load_comparison(metrics, key="order_only", name="Order-only"),
    ]


def _format_pvalue(value: float) -> str:
    if value < 0.001:
        return r"$p_\mathrm{Holm}<.001$"
    return rf"$p_\mathrm{{Holm}}={value:.3f}$"


def draw_figure(results: list[ComparisonResult]) -> plt.Figure:
    plt.rcParams.update(
        {
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.5,
        }
    )

    figure, (win_ax, effect_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": (1.28, 1.0)},
    )
    figure.subplots_adjust(
        bottom=0.18,
        left=0.105,
        right=0.985,
        top=0.80,
        wspace=0.39,
    )

    y_positions = np.array([1.0, 0.0])
    conditional_evi = np.array(
        [result.conditional_evitrace_win_pct for result in results]
    )
    conditional_s4 = 100.0 - conditional_evi

    win_ax.barh(
        y_positions,
        conditional_evi,
        height=0.48,
        color=EVITRACE_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="EviTrace win",
    )
    win_ax.barh(
        y_positions,
        conditional_s4,
        left=conditional_evi,
        height=0.48,
        color=S4_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Control win",
    )
    win_ax.axvline(50.0, color=NEUTRAL_COLOR, linewidth=0.9, linestyle="--", zorder=3)

    for y, result, evi_pct, s4_pct in zip(
        y_positions,
        results,
        conditional_evi,
        conditional_s4,
        strict=True,
    ):
        win_ax.text(
            evi_pct / 2.0,
            y,
            f"{evi_pct:.2f}%",
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
        )
        win_ax.text(
            evi_pct + s4_pct / 2.0,
            y,
            f"{s4_pct:.2f}%",
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
        )
        win_ax.text(
            50.0,
            y - 0.37,
            f"Non-ties {result.non_tie_pct:.2f}%  |  Ties {result.tie_pct:.2f}%",
            color=NEUTRAL_COLOR,
            fontsize=7.1,
            ha="center",
            va="top",
        )

    win_ax.set(
        xlabel="Share among non-tie comparisons (%)",
        xlim=(0.0, 100.0),
        xticks=(0, 25, 50, 75, 100),
        yticks=y_positions,
        yticklabels=[result.name for result in results],
        ylim=(-0.62, 1.42),
    )
    win_ax.set_title("(a) Conditional paired wins", loc="left", pad=38)
    win_ax.legend(
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.5,
        columnspacing=1.5,
    )

    delta = np.array([result.macro_f1_delta_pp for result in results])
    ci_low = np.array([result.macro_f1_ci95_pp[0] for result in results])
    ci_high = np.array([result.macro_f1_ci95_pp[1] for result in results])
    xerr = np.vstack((delta - ci_low, ci_high - delta))
    effect_ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=0.9, linestyle="--")
    effect_ax.errorbar(
        delta,
        y_positions,
        xerr=xerr,
        fmt="o",
        color=EVITRACE_COLOR,
        ecolor=EVITRACE_COLOR,
        elinewidth=1.6,
        capsize=4,
        capthick=1.2,
        markersize=5.5,
        zorder=3,
    )
    for y, result in zip(y_positions, results, strict=True):
        low, high = result.macro_f1_ci95_pp
        effect_ax.text(
            result.macro_f1_delta_pp,
            y + 0.24,
            (
                f"{result.macro_f1_delta_pp:+.2f} "
                f"[{low:+.2f}, {high:+.2f}]  "
                f"{_format_pvalue(result.holm_pvalue)}"
            ),
            fontsize=7.1,
            ha="center",
            va="bottom",
        )

    effect_ax.set(
        xlabel="EviTrace − Control (percentage points)",
        xlim=(-0.55, 2.65),
        xticks=(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
        yticks=y_positions,
        yticklabels=[result.name for result in results],
        ylim=(-0.48, 1.48),
    )
    effect_ax.set_title(r"(b) Primary $\Delta$Macro-F1", loc="left", pad=38)

    for axis in (win_ax, effect_ax):
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    return figure


def draw_single_column_figure(results: list[ComparisonResult]) -> plt.Figure:
    """Draw a compact, information-bearing panel for an AAAI single column."""
    minimum_font_size = 6.2
    plt.rcParams.update(
        {
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.8,
        }
    )

    figure, (win_ax, effect_ax) = plt.subplots(
        1,
        2,
        figsize=(3.20, 1.85),
        gridspec_kw={"width_ratios": (1.18, 1.0)},
    )
    figure.subplots_adjust(
        bottom=0.27,
        left=0.185,
        right=0.99,
        top=0.82,
        wspace=0.24,
    )

    y_positions = np.array([1.0, 0.0])
    conditional_evi = np.array(
        [result.conditional_evitrace_win_pct for result in results]
    )
    conditional_control = 100.0 - conditional_evi
    win_ax.barh(
        y_positions,
        conditional_evi,
        height=0.44,
        color=EVITRACE_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    win_ax.barh(
        y_positions,
        conditional_control,
        left=conditional_evi,
        height=0.44,
        color=S4_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    win_ax.axvline(
        50.0,
        color=NEUTRAL_COLOR,
        linewidth=0.8,
        linestyle="--",
        zorder=3,
    )
    for y, result, evi_pct, control_pct in zip(
        y_positions,
        results,
        conditional_evi,
        conditional_control,
        strict=True,
    ):
        win_ax.text(
            evi_pct / 2.0,
            y,
            f"Evi {evi_pct:.1f}",
            color="white",
            fontsize=8.1,
            fontweight="bold",
            ha="center",
            va="center",
        )
        win_ax.text(
            evi_pct + control_pct / 2.0,
            y,
            f"Ctl {control_pct:.1f}",
            color="white",
            fontsize=8.1,
            fontweight="bold",
            ha="center",
            va="center",
        )
        win_ax.text(
            50.0,
            y - 0.30,
            f"Ties {result.tie_pct:.1f}%",
            color=NEUTRAL_COLOR,
            fontsize=minimum_font_size,
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
        )
    win_ax.set(
        xlabel="Share among non-ties (%)",
        xlim=(0.0, 100.0),
        xticks=(0, 50, 100),
        yticks=y_positions,
        yticklabels=("Main", "Order-only"),
        ylim=(-0.58, 1.48),
    )
    win_ax.set_title("(a) Conditional wins", loc="left", pad=5)

    delta = np.array([result.macro_f1_delta_pp for result in results])
    ci_low = np.array([result.macro_f1_ci95_pp[0] for result in results])
    ci_high = np.array([result.macro_f1_ci95_pp[1] for result in results])
    xerr = np.vstack((delta - ci_low, ci_high - delta))
    effect_ax.axvline(
        0.0,
        color=NEUTRAL_COLOR,
        linewidth=0.8,
        linestyle="--",
    )
    effect_ax.errorbar(
        delta,
        y_positions,
        xerr=xerr,
        fmt="o",
        color=EVITRACE_COLOR,
        ecolor=EVITRACE_COLOR,
        elinewidth=1.3,
        capsize=3,
        capthick=1.0,
        markersize=4.5,
        zorder=3,
    )
    for y, result in zip(y_positions, results, strict=True):
        low, high = result.macro_f1_ci95_pp
        effect_ax.text(
            result.macro_f1_delta_pp,
            y + 0.21,
            (
                rf"$\mathbf{{{result.macro_f1_delta_pp:+.2f}}}$ "
                f"[{low:.2f}, {high:.2f}]"
            ),
            fontsize=6.6,
            ha="center",
            va="bottom",
        )
    effect_ax.set(
        xlabel=r"$\Delta$F1 (pp)",
        xlim=(-0.52, 2.58),
        xticks=(0.0, 1.0, 2.0),
        yticks=y_positions,
        yticklabels=("", ""),
        ylim=(-0.58, 1.48),
    )
    effect_ax.set_title(r"(b) $\Delta$Macro-F1", loc="left", pad=5)

    for axis in (win_ax, effect_ax):
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    return figure


def audit_minimum_font_size(
    figure: plt.Figure,
    minimum_font_size: float,
) -> float:
    """Fail if any visible text in the figure is smaller than requested."""
    visible_text = [
        artist
        for artist in figure.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    if not visible_text:
        raise ValueError("Figure contains no visible text to audit")
    violations = [
        (artist.get_text(), float(artist.get_fontsize()))
        for artist in visible_text
        if float(artist.get_fontsize()) + 1e-9 < minimum_font_size
    ]
    if violations:
        details = "; ".join(
            f"{text!r}={size:g}pt" for text, size in violations
        )
        raise ValueError(
            f"Figure contains text below {minimum_font_size:g} pt: {details}"
        )
    return min(float(artist.get_fontsize()) for artist in visible_text)


def audit_pdf_fonts(pdf_path: Path) -> list[str]:
    """Require every PDF font to be embedded with a six-letter subset prefix."""
    payload = pdf_path.read_bytes()
    descriptor_objects = re.findall(
        rb"<<\s*/Type\s*/FontDescriptor\b.*?>>",
        payload,
        flags=re.DOTALL,
    )
    if not descriptor_objects:
        raise ValueError(f"No PDF font descriptors found in {pdf_path}")

    descriptors: dict[bytes, bytes] = {}
    for descriptor in descriptor_objects:
        match = re.search(rb"/FontName\s*/([^\s/<>\[\]()]+)", descriptor)
        if match is None:
            raise ValueError(f"Font descriptor without /FontName in {pdf_path}")
        descriptors[match.group(1)] = descriptor

    font_objects = re.findall(
        rb"<<\s*/Type\s*/Font(?:\s|/).*?>>",
        payload,
        flags=re.DOTALL,
    )
    failures: list[str] = []
    base_fonts: set[bytes] = set()
    for font_object in font_objects:
        subtype_match = re.search(rb"/Subtype\s*/([^\s/<>\[\]()]+)", font_object)
        subtype = subtype_match.group(1) if subtype_match is not None else b"unknown"
        base_font_match = re.search(
            rb"/BaseFont\s*/([^\s/<>\[\]()]+)",
            font_object,
        )
        if subtype == b"Type3":
            failures.append("Type3 font object is not an embedded subset font")
        if base_font_match is None:
            failures.append(
                f"{subtype.decode('ascii', errors='replace')}: missing /BaseFont"
            )
        else:
            base_fonts.add(base_font_match.group(1))
    if not base_fonts:
        raise ValueError(f"No /BaseFont entries found in {pdf_path}")

    for font_name in sorted(base_fonts):
        printable_name = font_name.decode("ascii", errors="replace")
        if re.fullmatch(rb"[A-Z]{6}\+.+", font_name) is None:
            failures.append(f"{printable_name}: missing subset prefix")
        descriptor = descriptors.get(font_name)
        if descriptor is None:
            failures.append(f"{printable_name}: missing font descriptor")
        elif re.search(rb"/FontFile(?:2|3)?\s+\d+\s+0\s+R", descriptor) is None:
            failures.append(f"{printable_name}: font program is not embedded")
    if failures:
        raise ValueError(
            f"PDF font audit failed for {pdf_path}: " + "; ".join(failures)
        )
    return [
        font_name.decode("ascii", errors="strict")
        for font_name in sorted(base_fonts)
    ]


def save_figure(
    figure: plt.Figure,
    *,
    output_prefix: Path,
    dpi: int,
    minimum_font_size: float | None = None,
) -> list[Path]:
    if output_prefix.suffix:
        raise ValueError("--output-prefix must not include a file suffix")
    if dpi <= 0:
        raise ValueError("--dpi must be positive")
    if minimum_font_size is not None:
        observed_minimum = audit_minimum_font_size(
            figure,
            minimum_font_size,
        )
        print(f"Minimum visible font size: {observed_minimum:g} pt")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_prefix.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    metadata = {
        "Title": "EviTrace conditional verifier win rates",
        "Subject": "LIAR-RAW paired verifier evaluation",
        "Creator": "plot_cross_verifier_winrate.py",
    }
    for output in outputs:
        save_kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if output.suffix == ".png":
            save_kwargs["dpi"] = dpi
            save_kwargs["metadata"] = {
                "Title": metadata["Title"],
                "Description": metadata["Subject"],
                "Software": metadata["Creator"],
            }
        elif output.suffix == ".pdf":
            save_kwargs["metadata"] = metadata
        figure.savefig(output, **save_kwargs)
    embedded_subset_fonts = audit_pdf_fonts(output_prefix.with_suffix(".pdf"))
    print(
        "PDF embedded subset fonts: "
        + ", ".join(embedded_subset_fonts)
    )
    return outputs


def main() -> None:
    args = parse_args()
    results = load_results(args.metrics.resolve())
    if args.layout == "single-column":
        figure = draw_single_column_figure(results)
        default_output_prefix = DEFAULT_SINGLE_COLUMN_OUTPUT_PREFIX
        minimum_font_size = 6.2
    else:
        figure = draw_figure(results)
        default_output_prefix = DEFAULT_OUTPUT_PREFIX
        minimum_font_size = None
    output_prefix = args.output_prefix or default_output_prefix
    try:
        outputs = save_figure(
            figure,
            output_prefix=output_prefix.resolve(),
            dpi=args.dpi,
            minimum_font_size=minimum_font_size,
        )
    finally:
        plt.close(figure)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
