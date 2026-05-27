from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def aggregate_binary_outcomes(records: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    rows = list(records)
    n = len(rows)
    correct = sum(1 for row in rows if bool(row.get("is_correct", False)))
    tokens = sum(int(row.get("token_count", row.get("answer_len_tokens", 0)) or 0) for row in rows)
    parseable = sum(1 for row in rows if row.get("model_ans_norm") is not None or row.get("parseable"))
    return {
        "n": float(n),
        "correct": float(correct),
        "accuracy": safe_divide(correct, n),
        "token_count": float(tokens),
        "accuracy_per_1k_tokens": safe_divide(correct, tokens) * 1000.0,
        "parse_rate": safe_divide(parseable, n),
    }


def intervention_confusion(records: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    rows = list(records)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in rows:
        triggered = bool(row.get("triggered", False))
        was_correct = bool(row.get("is_correct", False))
        if triggered and not was_correct:
            true_positive += 1
        elif triggered and was_correct:
            false_positive += 1
        elif not triggered and was_correct:
            true_negative += 1
        else:
            false_negative += 1
    total = len(rows)
    return {
        "n": float(total),
        "intervention_rate": safe_divide(true_positive + false_positive, total),
        "false_positive_trigger_rate": safe_divide(false_positive, false_positive + true_negative),
        "false_negative_miss_rate": safe_divide(false_negative, false_negative + true_positive),
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "true_negative": float(true_negative),
        "false_negative": float(false_negative),
    }


def risk_coverage_curve(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "score",
    label_key: str = "is_correct",
) -> List[Dict[str, float]]:
    ordered = sorted(records, key=lambda row: float(row.get(score_key, 0.0)), reverse=True)
    n_total = len(ordered)
    curve: List[Dict[str, float]] = []
    kept: List[Mapping[str, Any]] = []
    for row in ordered:
        kept.append(row)
        n_kept = len(kept)
        correct = sum(1 for item in kept if bool(item.get(label_key, False)))
        coverage = safe_divide(n_kept, n_total)
        selective_accuracy = safe_divide(correct, n_kept)
        curve.append(
            {
                "coverage": coverage,
                "risk": 1.0 - selective_accuracy,
                "selective_accuracy": selective_accuracy,
                "threshold": float(row.get(score_key, 0.0)),
            }
        )
    return curve


def grouped_problem_accuracy(records: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    by_problem: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        problem_id = str(row.get("problem_id", "missing"))
        by_problem.setdefault(problem_id, []).append(row)
    if not by_problem:
        return {"n_problems": 0.0, "mean_problem_accuracy": 0.0}
    accuracies = [
        safe_divide(sum(1 for row in rows if bool(row.get("is_correct", False))), len(rows))
        for rows in by_problem.values()
    ]
    return {
        "n_problems": float(len(by_problem)),
        "mean_problem_accuracy": sum(accuracies) / len(accuracies),
    }
