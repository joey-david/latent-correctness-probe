from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


ExampleFilter = Callable[[Dict[str, Any]], bool]


def _default_filter(_: Dict[str, Any]) -> bool:
    return True


def eligible_example_indices(
    per_example_meta: Sequence[Dict[str, Any]],
    checkpoints: Iterable[int],
    filter_fn: Optional[ExampleFilter] = None,
    required_ts: Optional[Iterable[int]] = None,
) -> List[int]:
    """
    Return indices of examples that satisfy the filter and contain non-leaky features
    for all checkpoints in required_ts.
    """

    filter_fn = filter_fn or _default_filter
    cps = set(checkpoints)
    req = {t for t in (required_ts or []) if t in cps}

    eligible: List[int] = []
    for idx, example in enumerate(per_example_meta):
        if not filter_fn(example):
            continue
        prefix_states = example.get("prefix_states", {})
        missing = False
        for t in req:
            info = prefix_states.get(t)
            if not info or info.get("leaky") or info.get("h_t") is None:
                missing = True
                break
        if missing:
            continue
        eligible.append(idx)
    return eligible


def collect_features_for_indices(
    per_example_meta: Sequence[Dict[str, Any]],
    checkpoints: Iterable[int],
    indices: Iterable[int],
):
    """
    Assemble feature/label lists keyed by checkpoint using the cached per-example data.
    """

    checkpoints = list(checkpoints)
    features_by_t: Dict[int, List[np.ndarray]] = {t: [] for t in checkpoints}
    labels_by_t: Dict[int, List[int]] = {t: [] for t in checkpoints}

    for idx in indices:
        example = per_example_meta[idx]
        label = int(example.get("is_correct", 0))
        prefix_states = example.get("prefix_states", {})
        for t in checkpoints:
            info = prefix_states.get(t)
            if not info or info.get("leaky") or info.get("h_t") is None:
                continue
            features_by_t[t].append(info["h_t"])
            labels_by_t[t].append(label)

    return features_by_t, labels_by_t


def run_probes_from_meta(
    per_example_meta: Sequence[Dict[str, Any]],
    checkpoints: Iterable[int],
    filter_fn: Optional[ExampleFilter] = None,
    required_ts: Optional[Iterable[int]] = None,
    seed: int = 42,
):
    """
    Convenience wrapper that filters examples, aggregates features, and trains probes.

    Returns a tuple of (probe_metrics_by_t, labels_by_t, eligible_indices).
    """

    checkpoints = list(checkpoints)
    indices = eligible_example_indices(
        per_example_meta,
        checkpoints=checkpoints,
        filter_fn=filter_fn,
        required_ts=required_ts,
    )
    features_by_t, labels_by_t = collect_features_for_indices(
        per_example_meta,
        checkpoints=checkpoints,
        indices=indices,
    )
    metrics = run_all_probes(features_by_t, labels_by_t, seed=seed)
    return metrics, labels_by_t, indices


def train_eval_probe(features: np.ndarray, labels: np.ndarray, seed: int = 42):
    """
    Fit a simple linear probe (PCA -> logistic regression) and report metrics.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return None
    if np.min(counts) < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # PCA keeps the probe compact and mitigates collinearity before LR.
    max_components = min(features.shape[1], 128, X_train.shape[0])
    if max_components < 1:
        return None
    pca_dim = max_components
    pca = PCA(n_components=pca_dim, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    y_prob = clf.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "acc": acc,
        "auc": auc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "p_pos": float(np.mean(labels)),
    }


def run_all_probes(features_by_t, labels_by_t, seed: int = 42) -> Dict[int, Dict[str, float]]:
    """
    Train a separate linear probe at each checkpoint using the collected features.
    """
    results: Dict[int, Dict[str, float]] = {}
    for t in sorted(features_by_t.keys()):
        feats_list = features_by_t[t]
        labs_list = labels_by_t[t]
        if len(feats_list) < 10:
            continue  # not enough samples for a meaningful split

        X = np.stack(feats_list, axis=0)
        y = np.array(labs_list)

        metrics = train_eval_probe(X, y, seed=seed)
        if metrics is not None:
            results[t] = metrics
    return results
