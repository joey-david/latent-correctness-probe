from __future__ import annotations

import shutil
import warnings
import math
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


def _wilson_interval(p_hat: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n is None or n <= 0 or p_hat is None or np.isnan(p_hat):
        return float("nan"), float("nan")
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z ** 2) / (4.0 * n * n)) / denom
    lower = max(0.0, center - radius)
    upper = min(1.0, center + radius)
    return lower, upper


def _auc_confidence_interval(
    auc: float,
    n_pos: Optional[float],
    n_neg: Optional[float],
    z: float = 1.96,
) -> tuple[float, float]:
    if (
        auc is None
        or np.isnan(auc)
        or n_pos is None
        or n_neg is None
        or n_pos <= 0
        or n_neg <= 0
    ):
        return float("nan"), float("nan")
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    se_sq = (
        (auc * (1.0 - auc))
        + (n_pos - 1.0) * (q1 - auc * auc)
        + (n_neg - 1.0) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    if se_sq < 0.0:
        return float("nan"), float("nan")
    se = math.sqrt(se_sq)
    delta = z * se
    lower = max(0.0, auc - delta)
    upper = min(1.0, auc + delta)
    return lower, upper


def _counts_from_metrics(metrics: Dict[str, float]) -> tuple[Optional[float], Optional[float]]:
    if not metrics:
        return None, None
    n_pos = metrics.get("n_pos_test")
    n_neg = metrics.get("n_neg_test")
    if n_pos is None or n_neg is None:
        n_pos = metrics.get("n_pos_total")
        n_neg = metrics.get("n_neg_total")
    if n_pos is not None and n_neg is not None and n_pos > 0 and n_neg > 0:
        return float(n_pos), float(n_neg)

    n_test = metrics.get("n_test")
    p_pos = metrics.get("p_pos")
    if n_test is None or n_test <= 0 or p_pos is None or np.isnan(p_pos):
        return None, None
    approx_pos = max(1.0, round(p_pos * n_test))
    approx_neg = max(1.0, n_test - approx_pos)
    if approx_neg <= 0:
        approx_neg = 1.0
        approx_pos = max(1.0, n_test - approx_neg)
    return float(approx_pos), float(approx_neg)


def _make_step_formatter(steps: List[int]) -> FuncFormatter:
    visible_steps = set(int(s) for s in steps)

    def _format(value: float, _pos: int) -> str:
        candidate = int(round(value))
        if candidate not in visible_steps:
            return ""
        if candidate in {4, 16}:
            return ""
        return str(candidate)

    return FuncFormatter(_format)


def _fill_between(
    ax: plt.Axes,
    steps: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    alpha: float = 0.18,
) -> None:
    if steps.size == 0 or lower.size == 0 or upper.size == 0:
        return
    mask = ~(np.isnan(lower) | np.isnan(upper))
    if not np.any(mask):
        return
    ax.fill_between(
        steps[mask],
        lower[mask],
        upper[mask],
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=1.0,
    )


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
    steps_arr = np.asarray(steps, dtype=float)

    acc_ci_lower: List[float] = []
    acc_ci_upper: List[float] = []
    auc_ci_lower: List[float] = []
    auc_ci_upper: List[float] = []
    for t in steps:
        metrics = results_dict[t]
        acc = metrics.get("acc")
        n_test = metrics.get("n_test")
        acc_lo, acc_hi = _wilson_interval(acc, n_test)
        acc_ci_lower.append(acc_lo)
        acc_ci_upper.append(acc_hi)

        auc = metrics.get("auc")
        n_pos, n_neg = _counts_from_metrics(metrics)
        auc_lo, auc_hi = _auc_confidence_interval(auc, n_pos, n_neg)
        auc_ci_lower.append(auc_lo)
        auc_ci_upper.append(auc_hi)

    acc_ci_lower_arr = np.asarray(acc_ci_lower, dtype=float)
    acc_ci_upper_arr = np.asarray(acc_ci_upper, dtype=float)
    auc_ci_lower_arr = np.asarray(auc_ci_lower, dtype=float)
    auc_ci_upper_arr = np.asarray(auc_ci_upper, dtype=float)

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

    _fill_between(
        ax,
        steps_arr,
        auc_ci_lower_arr,
        auc_ci_upper_arr,
        colors["auc"],
    )
    _fill_between(
        ax,
        steps_arr,
        acc_ci_lower_arr,
        acc_ci_upper_arr,
        colors["acc"],
    )

    ax.set_xlabel("Reasoning prefix length (tokens)", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax.set_title(title, fontsize=TITLE_SIZE, color=STYLE_COLOR, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(steps)
    ax.xaxis.set_major_formatter(_make_step_formatter(steps))
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
    filtered = {
        name: res
        for name, res in results_by_name.items()
        if res and name not in {"overall", "fixed<=256"}
    }
    if not filtered:
        return

    _configure_fonts()

    names = sorted(filtered.keys())
    if not names:
        return

    def _color_for(name: str, index: int, total: int) -> str:
        lower = name.lower()
        if lower.startswith("easy"):
            return plt.colormaps["Purples"](0.5)
        if lower.startswith("hard"):
            return plt.colormaps["Greens"](0.6)
        frac = 0.35 + 0.55 * (index / max(1, total - 1))
        return plt.colormaps["Purples"](frac)

    fig, (ax_auc, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.4),
        sharex=True,
    )

    for idx, name in enumerate(names):
        results = filtered[name]
        steps = sorted(results.keys())
        steps_arr = np.asarray(steps, dtype=float)
        aucs = [results[t]["auc"] for t in steps]
        accs = [results[t]["acc"] for t in steps]

        acc_ci_lower: List[float] = []
        acc_ci_upper: List[float] = []
        auc_ci_lower: List[float] = []
        auc_ci_upper: List[float] = []
        for t in steps:
            metrics = results[t]
            acc_lo, acc_hi = _wilson_interval(metrics.get("acc"), metrics.get("n_test"))
            acc_ci_lower.append(acc_lo)
            acc_ci_upper.append(acc_hi)

            n_pos, n_neg = _counts_from_metrics(metrics)
            auc_lo, auc_hi = _auc_confidence_interval(metrics.get("auc"), n_pos, n_neg)
            auc_ci_lower.append(auc_lo)
            auc_ci_upper.append(auc_hi)

        acc_ci_lower_arr = np.asarray(acc_ci_lower, dtype=float)
        acc_ci_upper_arr = np.asarray(acc_ci_upper, dtype=float)
        auc_ci_lower_arr = np.asarray(auc_ci_lower, dtype=float)
        auc_ci_upper_arr = np.asarray(auc_ci_upper, dtype=float)

        color = _color_for(name, idx, len(names))
        ax_auc.plot(
            steps,
            aucs,
            marker="o",
            linewidth=2.0,
            label=name,
            color=color,
        )
        ax_acc.plot(
            steps,
            accs,
            marker="s",
            linewidth=2.0,
            label=name,
            color=color,
        )

        _fill_between(ax_auc, steps_arr, auc_ci_lower_arr, auc_ci_upper_arr, color)
        _fill_between(ax_acc, steps_arr, acc_ci_lower_arr, acc_ci_upper_arr, color)

    all_steps = sorted(
        {int(step) for result in filtered.values() for step in result.keys()}
    )
    if all_steps:
        ax_acc.set_xticks(all_steps)
        formatter = _make_step_formatter(all_steps)
        ax_acc.xaxis.set_major_formatter(formatter)
        ax_auc.xaxis.set_major_formatter(formatter)

    ax_auc.set_ylabel("ROC-AUC", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_acc.set_ylabel("Accuracy", fontsize=LABEL_SIZE, color=STYLE_COLOR)
    ax_acc.set_xlabel("Reasoning prefix length (tokens)", fontsize=LABEL_SIZE, color=STYLE_COLOR)

    ax_auc.set_title(title, fontsize=TITLE_SIZE, color=STYLE_COLOR, pad=10)
    ax_auc.set_ylim(0.0, 1.0)
    ax_acc.set_ylim(0.0, 1.0)
    ax_auc.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color=STYLE_COLOR)
    ax_acc.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color=STYLE_COLOR)

    ax_auc.tick_params(axis="x", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_auc.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_acc.tick_params(axis="x", colors=STYLE_COLOR, labelsize=TICK_SIZE)
    ax_acc.tick_params(axis="y", colors=STYLE_COLOR, labelsize=TICK_SIZE)

    for axis in (ax_auc, ax_acc):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(STYLE_COLOR)
        axis.spines["bottom"].set_color(STYLE_COLOR)

    ax_auc.axhline(
        0.5,
        color="#535353",
        linestyle=(0, (1, 3)),
        linewidth=1.0,
        zorder=0.5,
    )
    ax_acc.axhline(
        0.5,
        color="#535353",
        linestyle=(0, (1, 3)),
        linewidth=1.0,
        zorder=0.5,
    )

    legend = ax_auc.legend(loc="best", fontsize=LEGEND_SIZE, frameon=False)
    if legend:
        for text in legend.get_texts():
            text.set_color(STYLE_COLOR)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
def plot_baseline_comparison(
    steps: List[int],
    series_by_name: Dict[str, List[float]],
    title: str,
    outpath: str,
    support_by_name: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> None:
    if not steps or not series_by_name:
        return

    _configure_fonts()

    steps_arr = np.asarray(steps, dtype=float)
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
        (line,) = ax.plot(
            steps,
            values,
            label=style["label"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=colors[name],
            linewidth=2.0,
            markersize=6,
        )
        color = line.get_color()

        counts = support_by_name.get(name) if support_by_name else None
        if counts:
            n_pos_list = counts.get("n_pos", [])
            n_neg_list = counts.get("n_neg", [])
            lower: List[float] = []
            upper: List[float] = []
            for value, n_pos, n_neg in zip(values, n_pos_list, n_neg_list):
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    lower.append(float("nan"))
                    upper.append(float("nan"))
                    continue
                if np.isnan(value_float):
                    lower.append(float("nan"))
                    upper.append(float("nan"))
                    continue
                try:
                    n_pos_float = float(n_pos)
                    n_neg_float = float(n_neg)
                except (TypeError, ValueError):
                    lower.append(float("nan"))
                    upper.append(float("nan"))
                    continue
                if (
                    np.isnan(n_pos_float)
                    or np.isnan(n_neg_float)
                    or n_pos_float <= 0.0
                    or n_neg_float <= 0.0
                ):
                    lower.append(float("nan"))
                    upper.append(float("nan"))
                    continue
                lo, hi = _auc_confidence_interval(value_float, n_pos_float, n_neg_float)
                lower.append(lo)
                upper.append(hi)
            lower_arr = np.asarray(lower, dtype=float)
            upper_arr = np.asarray(upper, dtype=float)
            _fill_between(ax, steps_arr, lower_arr, upper_arr, color)

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
