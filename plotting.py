from typing import Dict, List

import matplotlib.pyplot as plt


def plot_probe_results(results_dict: Dict[int, Dict[str, float]], title: str, outpath: str) -> None:
    """
    Save a simple line plot showing probe AUC/ACC as a function of prefix length.
    """
    if not results_dict:
        return

    steps = sorted(results_dict.keys())
    aucs = [results_dict[t]["auc"] for t in steps]
    accs = [results_dict[t]["acc"] for t in steps]

    plt.figure()
    plt.plot(steps, aucs, marker="o", label="ROC-AUC")
    plt.plot(steps, accs, marker="s", label="Accuracy")
    plt.xlabel("Reasoning prefix length (tokens)")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


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


def plot_carryforward_curves(
    raw_results: Dict[int, Dict[str, float]],
    carry_data: Dict[str, List[float]],
    title: str,
    outpath: str,
) -> None:
    if not raw_results or not carry_data:
        return

    steps = carry_data["steps"]
    raw_auc = carry_data["raw_auc"]
    raw_acc = carry_data["raw_acc"]
    carry_auc = carry_data["carry_auc"]
    carry_acc = carry_data["carry_acc"]

    plt.figure(figsize=(7, 6))
    ax_auc = plt.subplot(2, 1, 1)
    ax_acc = plt.subplot(2, 1, 2, sharex=ax_auc)

    ax_auc.plot(steps, raw_auc, marker="o", linestyle="-", label="Raw ROC-AUC")
    ax_auc.plot(steps, carry_auc, marker="o", linestyle="--", label="Carry-forward ROC-AUC")
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_title(title)
    ax_auc.grid(True)
    ax_auc.legend(loc="best")

    ax_acc.plot(steps, raw_acc, marker="s", linestyle="-", label="Raw Accuracy")
    ax_acc.plot(steps, carry_acc, marker="s", linestyle="--", label="Carry-forward Accuracy")
    ax_acc.set_xlabel("Reasoning prefix length (tokens)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True)
    ax_acc.legend(loc="best")

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

    style_config = {
        "hidden_state": {"label": "Hidden-state probe", "linestyle": "-", "marker": "o"},
        "entropy": {"label": "Entropy baseline", "linestyle": "--", "marker": "s"},
        "length": {"label": "Length baseline", "linestyle": ":", "marker": "^"},
    }

    plt.figure(figsize=(7, 4))
    for name, values in series_by_name.items():
        if name not in style_config:
            continue
        style = style_config[name]
        plt.plot(
            steps,
            values,
            label=style["label"],
            linestyle=style["linestyle"],
            marker=style["marker"],
        )

    plt.xlabel("Reasoning prefix length (tokens)")
    plt.ylabel("ROC-AUC")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
