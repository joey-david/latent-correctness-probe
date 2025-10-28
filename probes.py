from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


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
