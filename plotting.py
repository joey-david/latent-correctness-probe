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
