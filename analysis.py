import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from probes import run_probes_from_meta


def _to_serialisable(metrics: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if metrics is None:
        return None
    return {
        key: (float(value) if isinstance(value, (np.floating, np.float32, np.float64)) else value)
        for key, value in metrics.items()
    }


def analyze_difficulty_buckets(
    per_example_meta: List[Dict],
    target_t: int,
    seed: int,
    fig_path: Path,
    results_path: Path,
    plot_fn: Callable[[List[str], List[float], str, Path], None],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    def make_filter(level_predicate: Callable[[int], bool]) -> Callable[[Dict], bool]:
        def _filter(example: Dict) -> bool:
            level = example.get("level")
            if level is None:
                return False
            return level_predicate(level)

        return _filter

    easy_filter = make_filter(lambda level: level <= 2)
    hard_filter = make_filter(lambda level: level >= 4)

    easy_metrics, _, easy_indices = run_probes_from_meta(
        per_example_meta,
        checkpoints=[target_t],
        filter_fn=easy_filter,
        required_ts=[target_t],
        seed=seed,
    )
    hard_metrics, _, hard_indices = run_probes_from_meta(
        per_example_meta,
        checkpoints=[target_t],
        filter_fn=hard_filter,
        required_ts=[target_t],
        seed=seed,
    )

    easy_stats = easy_metrics.get(target_t)
    hard_stats = hard_metrics.get(target_t)

    if log_fn:
        log_fn(
            "Difficulty analysis: "
            f"easy_examples={len(easy_indices)} hard_examples={len(hard_indices)}"
        )

    if not easy_stats or not hard_stats:
        if log_fn:
            log_fn("Difficulty analysis skipped due to insufficient data in one bucket.")
        return None

    labels = ["easy<=2", "hard>=4"]
    auc_values = [float(easy_stats["auc"]), float(hard_stats["auc"])]

    try:
        plot_fn(labels, auc_values, f"Prefix t={target_t}", fig_path)
    except Exception as exc:  # pragma: no cover
        if log_fn:
            log_fn(f"Difficulty bar plot failed: {exc!r}")

    results_payload = {
        "t": target_t,
        "easy": _to_serialisable(easy_stats),
        "hard": _to_serialisable(hard_stats),
        "num_easy_examples": len(easy_indices),
        "num_hard_examples": len(hard_indices),
    }
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")
    return results_payload


def _logistic_cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int,
) -> Optional[np.ndarray]:
    if X.size == 0 or len(np.unique(y)) < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )

    for train_idx, test_idx in skf.split(X, y):
        try:
            clf.fit(X[train_idx], y[train_idx])
        except ValueError:
            return None
        preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return preds


def analyze_baselines(
    per_example_meta: List[Dict],
    probe_results: Dict[int, Dict[str, float]],
    seed: int,
    fig_path: Path,
    results_path: Path,
    plot_fn: Callable[[List[int], Dict[str, List[float]], str, Path], None],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[int, Dict[str, float]]]:
    records: List[Dict[str, float]] = []
    for example in per_example_meta:
        label = int(example.get("is_correct", 0))
        prefix_states = example.get("prefix_states", {})
        for t, info in prefix_states.items():
            if not info or info.get("leaky"):
                continue
            entropy = info.get("next_token_entropy")
            prefix_len = info.get("prefix_len", t)
            if entropy is None:
                continue
            records.append(
                {
                    "t": int(t),
                    "entropy": float(entropy),
                    "prefix_len": float(prefix_len),
                    "label": label,
                }
            )

    if not records:
        if log_fn:
            log_fn("Baseline analysis skipped (no entropy records available).")
        return None

    steps = sorted({rec["t"] for rec in records})
    y = np.array([rec["label"] for rec in records], dtype=int)
    entropy_feat = np.array([[rec["entropy"]] for rec in records], dtype=float)
    entropy_len_feat = np.array(
        [[rec["entropy"], rec["prefix_len"]] for rec in records], dtype=float
    )
    length_feat = np.array([[rec["prefix_len"]] for rec in records], dtype=float)

    class_counts = np.bincount(y)
    min_class = int(class_counts.min()) if len(class_counts) > 1 else 0
    n_splits = max(2, min(5, min_class)) if min_class >= 2 else 0
    if n_splits < 2:
        if log_fn:
            log_fn("Baseline analysis skipped (insufficient class balance).")
        return None

    entropy_probs = _logistic_cv_predictions(entropy_feat, y, seed, n_splits)
    entropy_len_probs = _logistic_cv_predictions(entropy_len_feat, y, seed, n_splits)
    length_probs = _logistic_cv_predictions(length_feat, y, seed, n_splits)

    auc_curves: Dict[str, List[float]] = {
        "entropy": [],
        "entropy_plus_length": [],
        "length": [],
        "hidden_state": [],
    }
    per_t_results: Dict[int, Dict[str, float]] = {}

    for t in steps:
        mask = np.array([rec["t"] == t for rec in records])
        labels_t = y[mask]
        if len(np.unique(labels_t)) < 2:
            continue

        result_row: Dict[str, float] = {}

        if entropy_probs is not None:
            auc_entropy = float(roc_auc_score(labels_t, entropy_probs[mask]))
            auc_curves["entropy"].append(auc_entropy)
            result_row["entropy_auc"] = auc_entropy
        if entropy_len_probs is not None:
            auc_entropy_len = float(roc_auc_score(labels_t, entropy_len_probs[mask]))
            auc_curves["entropy_plus_length"].append(auc_entropy_len)
            result_row["entropy_plus_length_auc"] = auc_entropy_len
        if length_probs is not None:
            auc_length = float(roc_auc_score(labels_t, length_probs[mask]))
            auc_curves["length"].append(auc_length)
            result_row["length_auc"] = auc_length

        hidden_stats = probe_results.get(t)
        if hidden_stats:
            hidden_auc = float(hidden_stats.get("auc", float("nan")))
            auc_curves["hidden_state"].append(hidden_auc)
            result_row["hidden_state_auc"] = hidden_auc

        per_t_results[t] = result_row

    if not per_t_results:
        if log_fn:
            log_fn("Baseline analysis produced no per-step results.")
        return None

    # Align curves by steps and fill missing entries with NaN for plotting consistency.
    def series_for(name: str) -> List[float]:
        values = []
        for t in steps:
            row = per_t_results.get(t)
            if not row or f"{name}_auc" not in row and name != "hidden_state":
                values.append(float("nan"))
            else:
                key = f"{name}_auc" if name != "hidden_state" else "hidden_state_auc"
                values.append(row.get(key, float("nan")))
        return values

    plot_payload = {
        "hidden_state": series_for("hidden_state"),
        "entropy": series_for("entropy"),
        "length": series_for("length"),
    }

    try:
        plot_fn(steps, plot_payload, "Probe vs baselines", fig_path)
    except Exception as exc:  # pragma: no cover
        if log_fn:
            log_fn(f"Baseline comparison plot failed: {exc!r}")

    serialisable = {
        str(t): {k: float(v) for k, v in row.items()}
        for t, row in sorted(per_t_results.items())
    }
    results_path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
    return per_t_results
