from typing import Dict

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
