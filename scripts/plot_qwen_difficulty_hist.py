#!/usr/bin/env python3
"""Plot level-wise accuracy and prefix depth for Qwen3-8B."""

from __future__ import annotations

import json
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("runs/Qwen3-8B_examples.jsonl")
OUTPUT_PATH = Path("figs/qwen_level_metrics.png")
STYLE_COLOR = "#3f4042"
STYLE_FONT = "newpx"
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 11
LEGEND_SIZE = 12


def configure_fonts() -> None:
    """Prefer the newpx font while staying usable without a LaTeX install."""
    if shutil.which("latex") is None:
        warnings.warn(
            "LaTeX not found. Install LaTeX to render with the 'newpx' font; "
            "falling back to Matplotlib's default serif font.",
            RuntimeWarning,
        )
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": [STYLE_FONT, "DejaVu Serif"],
            }
        )
        return

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [STYLE_FONT],
            "text.latex.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
        }
    )


def load_level_metrics(path: Path) -> Dict[int, Dict[str, float]]:
    """Aggregate target metrics for each problem level."""
    aggregates: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "correct": 0.0,
            "latest_prefix_sum": 0.0,
        }
    )

    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            level = int(record["level"])
            if level == 3:
                continue

            aggregates[level]["count"] += 1
            aggregates[level]["correct"] += float(record["is_correct"])

            available_prefixes = record["available_prefixes"]
            latest_prefix = max(available_prefixes)
            aggregates[level]["latest_prefix_sum"] += latest_prefix

    metrics: Dict[int, Dict[str, float]] = {}
    for level, stats in aggregates.items():
        count = stats["count"]
        metrics[level] = {
            "accuracy": stats["correct"] / count,
            "latest_prefix": stats["latest_prefix_sum"] / count,
        }
    return metrics


def _scaled_colors(values: np.ndarray, cmap_name: str, light: float, dark: float) -> list:
    """Return colors that slightly darken with higher values."""
    cmap = plt.colormaps[cmap_name]
    if values.size == 0:
        return []
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        positions = np.full_like(values, (light + dark) / 2, dtype=float)
    else:
        positions = light + (values - vmin) / (vmax - vmin) * (dark - light)
    return [cmap(pos) for pos in positions]


def _annotate_bars(bars, axis, fmt: str) -> None:
    """Place numeric labels slightly above each bar."""
    upper = axis.get_ylim()[1] or 1.0
    offset = 0.015 * upper
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=STYLE_COLOR,
        )


def plot_histogram(metrics: Dict[int, Dict[str, float]], output_path: Path) -> None:
    """Create a grouped bar chart with accuracy and prefix depth per level."""
    levels = sorted(metrics)
    x_positions = np.arange(len(levels))
    bar_width = 0.4

    accuracy = np.array([metrics[level]["accuracy"] for level in levels], dtype=float)
    latest_prefix = np.array([metrics[level]["latest_prefix"] for level in levels], dtype=float)

    fig, ax_accuracy = plt.subplots(figsize=(8.5, 5.2))
    ax_prefix = ax_accuracy.twinx()

    accuracy_colors = _scaled_colors(accuracy, "Purples", light=0.35, dark=0.7)
    prefix_colors = _scaled_colors(latest_prefix, "Greens", light=0.35, dark=0.7)

    accuracy_bars = ax_accuracy.bar(
        x_positions - bar_width / 2,
        accuracy,
        width=bar_width,
        color=accuracy_colors,
        edgecolor=STYLE_COLOR,
        linewidth=0.75,
        label="Mean answer accuracy",
    )
    prefix_bars = ax_prefix.bar(
        x_positions + bar_width / 2,
        latest_prefix,
        width=bar_width,
        color=prefix_colors,
        edgecolor=STYLE_COLOR,
        linewidth=0.75,
        label="Mean latest available prefix",
    )

    ax_accuracy.set_xticks(x_positions)
    ax_accuracy.set_xticklabels([str(level) for level in levels], fontsize=TICK_SIZE, color=STYLE_COLOR)
    ax_accuracy.set_xlabel("Problem level", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_accuracy.set_ylabel("Mean answer accuracy", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_prefix.set_ylabel("Mean latest available prefix length", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_accuracy.set_ylim(0, 1.05)
    prefix_top = latest_prefix.max() if latest_prefix.size else 0.0
    if prefix_top <= 0:
        prefix_top = 1.0
    ax_prefix.set_ylim(0, prefix_top * 1.15)
    ax_accuracy.set_title(
        "Qwen3-8B Accuracy vs. Context Depth by Level",
        fontsize=TITLE_SIZE,
        pad=12,
        color=STYLE_COLOR,
    )

    ax_accuracy.spines["top"].set_visible(False)
    ax_prefix.spines["top"].set_visible(False)
    ax_accuracy.grid(False)
    ax_prefix.grid(False)

    ax_accuracy.tick_params(axis="x", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_accuracy.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_prefix.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)

    ax_accuracy.spines["left"].set_color(STYLE_COLOR)
    ax_accuracy.spines["bottom"].set_color(STYLE_COLOR)
    ax_prefix.spines["right"].set_color(STYLE_COLOR)

    all_rects = list(accuracy_bars) + list(prefix_bars)
    if all_rects:
        left_edge = min(rect.get_x() for rect in all_rects)
        right_edge = max(rect.get_x() + rect.get_width() for rect in all_rects)
        ax_accuracy.set_xlim(left_edge, right_edge)
        ax_prefix.set_xlim(left_edge, right_edge)
        ax_accuracy.spines["bottom"].set_bounds(left_edge, right_edge)
        ax_accuracy.margins(x=0)
        ax_prefix.margins(x=0)

    _annotate_bars(accuracy_bars, ax_accuracy, "{:.2f}")
    _annotate_bars(prefix_bars, ax_prefix, "{:.0f}")

    handles = [accuracy_bars, prefix_bars]
    labels = [accuracy_bars.get_label(), prefix_bars.get_label()]
    legend = ax_accuracy.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(STYLE_COLOR)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_fonts()
    metrics = load_level_metrics(DATA_PATH)
    plot_histogram(metrics, OUTPUT_PATH)


if __name__ == "__main__":
    main()
