from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SPLIT_NAMES = ("train", "validation", "final_test")


def stable_problem_id(problem: str, subject: str | None = None) -> str:
    normalized = " ".join(problem.strip().split())
    payload = f"{subject or ''}\n{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def attach_problem_ids(examples: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for example in examples:
        record = dict(example)
        problem = str(record.get("problem", ""))
        subject = record.get("subject")
        problem_id = str(record.get("problem_id") or stable_problem_id(problem, subject))
        if problem_id in seen:
            continue
        seen.add(problem_id)
        record["problem_id"] = problem_id
        records.append(record)
    return records


def build_split_manifest(
    examples: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    final_test_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    total_fraction = train_fraction + validation_fraction + final_test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    records = attach_problem_ids(examples)
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_fraction))
    n_validation = int(round(n_total * validation_fraction))
    if n_train + n_validation > n_total:
        n_validation = max(0, n_total - n_train)

    splits = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "final_test": shuffled[n_train + n_validation :],
    }
    manifest = {
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "final_test": final_test_fraction,
        },
        "counts": {name: len(items) for name, items in splits.items()},
        "problem_ids": {
            name: [str(item["problem_id"]) for item in items] for name, items in splits.items()
        },
        "metadata": {
            name: [
                {
                    "problem_id": str(item["problem_id"]),
                    "subject": item.get("subject"),
                    "level": item.get("level"),
                }
                for item in items
            ]
            for name, items in splits.items()
        },
    }
    assert_problem_disjoint(manifest)
    return manifest


def assert_problem_disjoint(manifest: Mapping[str, Any]) -> None:
    split_sets = {
        name: set(manifest.get("problem_ids", {}).get(name, [])) for name in SPLIT_NAMES
    }
    for left in SPLIT_NAMES:
        for right in SPLIT_NAMES:
            if left >= right:
                continue
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(f"Problem leakage between {left} and {right}: {sample}")


def write_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    assert_problem_disjoint(manifest)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def read_split_manifest(path: str | Path) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_problem_disjoint(manifest)
    return manifest
