from __future__ import annotations

import shutil
import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

STYLE_COLOR = "#3f4042"
STYLE_FONT = "newpx"
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12
ANNOTATION_SIZE = 11

_FONTS_CONFIGURED = False


def _configure_fonts() -> None:
    """Prefer the newpx font while remaining usable without LaTeX."""
    global _FONTS_CONFIGURED
    if _FONTS_CONFIGURED:
        return
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
        _FONTS_CONFIGURED = True
        return

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [STYLE_FONT],
            "text.latex.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
        }
    )
    _FONTS_CONFIGURED = True


def plot_probe_results(results_dict: Dict[int, Dict[str, float]], title: str, outpath: str) -> None:
    """
    Save a simple line plot showing probe AUC/ACC as a function of prefix length.
    """
    if not results_dict:
        return

    _configure_fonts()

    steps = sorted(results_dict.keys())
    aucs = [results_dict[t]["auc"] for t in steps]
    accs = [results_dict[t]["acc"] for t in steps]

    colors = {
        "auc": plt.colormaps["Purples"](0.7),
        "acc": plt.colormaps["Greens"](0.75),
    }
    styles = {
        "auc": {"label": "ROC-AUC", "linestyle": "-", "marker": "o"},
        "acc": {"label": "Accuracy", "linestyle": "--", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        steps,
        aucs,
        label=styles["auc"]["label"],
        linestyle=styles["auc"]["linestyle"],
        marker=styles["auc"]["marker"],
        color=colors["auc"],
        linewidth=2.0,
        markersize=6,
    )
    ax.plot(
        steps,
        accs,
        label=styles["acc"]["label"],
        linestyle=styles["acc"]["linestyle"],
        marker=styles["acc"]["marker"],
        color=colors["acc"],
        linewidth=2.0,
        markersize=6,
    )

    ax.set_xlabel("Reasoning prefix length (tokens)", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax.set_title(title, fontsize=TITLE_SIZE, color=STYLE_COLOR, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(steps)
    ax.tick_params(axis="x", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax.margins(x=0)

    ax.axhline(
        0.5,
        color="#535353",
        linestyle=(0, (1, 3)),
        linewidth=1.0,
        zorder=0.5,
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color=STYLE_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE_COLOR)
    ax.spines["bottom"].set_color(STYLE_COLOR)

    legend = ax.legend(loc="lower right", fontsize=LEGEND_SIZE, frameon=False)
    if legend:
        for text in legend.get_texts():
            text.set_color(STYLE_COLOR)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_multiple_probe_results(
    results_by_name: Dict[str, Dict[int, Dict[str, float]]],
    title: str,
    outpath: str,
) -> None:
    """
    Plot AUC/Accuracy curves for multiple result groups on shared axes.
    """
    filtered = {name: res for name, res in results_by_name.items() if res}
    if not filtered:
        return

    plt.figure(figsize=(7, 6))
    ax_auc = plt.subplot(2, 1, 1)
    ax_acc = plt.subplot(2, 1, 2, sharex=ax_auc)

    for name, results in filtered.items():
        steps = sorted(results.keys())
        aucs = [results[t]["auc"] for t in steps]
        accs = [results[t]["acc"] for t in steps]
        ax_auc.plot(steps, aucs, marker="o", label=name)
        ax_acc.plot(steps, accs, marker="s", label=name)

    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_title(title)
    ax_auc.grid(True)
    ax_auc.legend(loc="best")

    ax_acc.set_xlabel("Reasoning prefix length (tokens)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_difficulty_bar(labels: List[str], auc_values: List[float], title: str, outpath: str) -> None:
    if not labels or not auc_values:
        return

    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, auc_values, color=["#4C72B0", "#DD8452"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, auc_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_baseline_comparison(
    steps: List[int],
    series_by_name: Dict[str, List[float]],
    title: str,
    outpath: str,
) -> None:
    if not steps or not series_by_name:
        return

    _configure_fonts()

    colors = {
        "hidden_state": plt.colormaps["Purples"](0.65),
        "entropy": plt.colormaps["Greens"](0.75),
        "length": plt.colormaps["Greens"](0.35),
    }
    style_config = {
        "hidden_state": {"label": "Hidden-state probe", "linestyle": "-", "marker": "o"},
        "entropy": {"label": "Entropy baseline", "linestyle": "--", "marker": "s"},
        "length": {"label": "Length baseline", "linestyle": ":", "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name in ("hidden_state", "entropy", "length"):
        values = series_by_name.get(name)
        if values is None:
            continue
        style = style_config[name]
        ax.plot(
            steps,
            values,
            label=style["label"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=colors[name],
            linewidth=2.0,
            markersize=6,
        )

    ax.set_xlabel("Reasoning prefix length (tokens)", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax.set_ylabel("ROC-AUC", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax.set_title(title, fontsize=TITLE_SIZE, color=STYLE_COLOR, pad=10)
    ax.set_ylim(0.0, 1.02)
    visible_steps = set(steps)

    def _format_step(value: float, _pos: int) -> str:
        step = int(round(value))
        if step not in visible_steps or step in {4, 16}:
            return ""
        return str(step)

    ax.set_xticks(steps)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_step))
    ax.tick_params(axis="x", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax.margins(x=0)

    ax.axhline(
        0.5,
        color="#535353",
        linestyle=(0, (1, 3)),
        linewidth=1.0,
        zorder=0.5,
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color=STYLE_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE_COLOR)
    ax.spines["bottom"].set_color(STYLE_COLOR)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles, labels, loc="lower right", fontsize=LEGEND_SIZE, frameon=False)
        for text in legend.get_texts():
            text.set_color(STYLE_COLOR)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_probe_pca_profile(
    prefix_len: int,
    details: Dict[str, List[float]],
    metrics: Dict[str, float],
    outpath: str,
    top_k: int = 16,
    title: Optional[str] = None,
) -> None:
    """
    Visualise how PCA components feed the linear probe at a specific prefix length.
    """
    if not details:
        return

    variance = details.get("explained_variance_ratio")
    coeffs = details.get("logit_coefficients")
    if not variance or not coeffs:
        return

    _configure_fonts()

    k = min(top_k, len(variance), len(coeffs))
    if k == 0:
        return

    indices = np.arange(1, k + 1)
    var = np.asarray(variance[:k], dtype=float)
    cum_var = np.cumsum(var)
    coef = np.asarray(coeffs[:k], dtype=float)
    abs_coef = np.abs(coef)

    def _scaled_colors(values: np.ndarray, cmap_name: str, light: float, dark: float) -> List:
        cmap = plt.colormaps[cmap_name]
        if values.size == 0:
            return []
        vmin, vmax = float(values.min()), float(values.max())
        if np.isclose(vmin, vmax):
            positions = np.full_like(values, (light + dark) / 2, dtype=float)
        else:
            positions = light + (values - vmin) / (vmax - vmin) * (dark - light)
        return [cmap(float(pos)) for pos in positions]

    fig, (ax_var, ax_coef) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.6),
        sharex=True,
    )

    bar_colors = _scaled_colors(var, "Greens", 0.35, 0.75)
    bars = ax_var.bar(
        indices,
        var,
        color=bar_colors,
        edgecolor=STYLE_COLOR,
        linewidth=0.8,
    )
    ax_var.plot(
        indices,
        cum_var,
        color=plt.colormaps["Greens"](0.55),
        marker="o",
        linestyle="-",
        linewidth=1.6,
        label="Cumulative variance",
    )
    ax_var.set_ylabel("Explained variance ratio", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_var.set_ylim(0, max(var.max() * 1.2, 0.05))
    ax_var.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    ax_var.spines["left"].set_color(STYLE_COLOR)
    ax_var.spines["bottom"].set_visible(False)
    ax_var.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25, color=STYLE_COLOR)

    if bars:
        upper = ax_var.get_ylim()[1]
        offset = 0.018 * upper
        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax_var.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
                color=STYLE_COLOR,
            )

    coef_colors = _scaled_colors(abs_coef, "Purples", 0.35, 0.75)
    ax_coef.plot(
        indices,
        coef,
        color=plt.colormaps["Purples"](0.65),
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="Logistic coefficient",
    )
    ax_coef.vlines(
        indices,
        np.zeros_like(indices),
        coef,
        colors=coef_colors,
        linewidth=2.0,
        alpha=0.9,
    )
    ax_coef.axhline(0.0, color=STYLE_COLOR, linewidth=0.9, linestyle=(0, (1, 3)))
    ax_coef.set_ylabel("Coefficient weight", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_coef.set_xlabel("Principal component index", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_coef.tick_params(axis="x", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_coef.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_coef.spines["top"].set_visible(False)
    ax_coef.spines["right"].set_visible(False)
    ax_coef.spines["left"].set_color(STYLE_COLOR)
    ax_coef.spines["bottom"].set_color(STYLE_COLOR)
    ax_coef.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25, color=STYLE_COLOR)

    auc = metrics.get("auc")
    acc = metrics.get("acc")
    header = title or f"PCA→probe diagnostics at prefix $t={prefix_len}$"
    if auc is not None and acc is not None:
        header += f" (AUC {auc:.3f}, ACC {acc:.3f})"
    ax_var.set_title(header, fontsize=TITLE_SIZE, color=STYLE_COLOR, pad=10)

    legend = ax_var.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=False)
    if legend:
        for text in legend.get_texts():
            text.set_color(STYLE_COLOR)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
