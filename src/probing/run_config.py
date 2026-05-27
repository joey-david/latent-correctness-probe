from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "id": "Qwen/Qwen3-8B",
        "revision": None,
        "device_map": "auto",
        "torch_dtype": "auto",
    },
    "dataset": {
        "name": "EleutherAI/hendrycks_math",
        "config": "all",
        "source_split": "test",
        "subjects": [],
        "min_level": 1,
        "require_numeric": True,
        "split_manifest": "data/splits/math_problem_splits.json",
    },
    "prompt": {
        "id": "math_boxed_v1",
        "template": "model_family_default",
    },
    "rollouts": {
        "per_problem": 1,
        "max_problems": 16,
    },
    "decoding": {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "features": {
        "prefix_checkpoints": [4, 8, 16, 32, 64, 128],
        "selected_layers": ["last"],
        "pool_last_k_tokens": 4,
        "max_prefix_tokens": 1024,
    },
    "splits": {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "final_test_fraction": 0.2,
        "seed": 356,
    },
    "intervention": {
        "policy": "none",
        "checkpoint": 32,
        "threshold": 0.5,
        "prompt": "reassess",
        "max_extra_tokens": 256,
        "score_file": None,
    },
    "routing": {
        "difficulty_signal": "none",
        "low_confidence_threshold": 0.5,
        "high_confidence_threshold": 0.8,
    },
    "seeds": [356],
    "outputs": {
        "root": "artifacts",
        "rollouts": "artifacts/rollouts.jsonl",
        "features": "artifacts/features",
        "probes": "artifacts/probes",
        "baselines": "artifacts/baselines.json",
        "interventions": "artifacts/interventions.jsonl",
        "figures": "artifacts/figures",
        "manifests": "artifacts/manifests",
    },
    "run": {
        "dry_run": False,
        "notes": "",
    },
}

REQUIRED_TOP_LEVEL = {
    "model",
    "dataset",
    "prompt",
    "rollouts",
    "decoding",
    "features",
    "splits",
    "intervention",
    "seeds",
    "outputs",
}


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def load_config(path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config_path = Path(path)
    loaded = load_yaml(config_path)
    merged = deep_merge(DEFAULT_CONFIG, loaded)
    if overrides:
        merged = deep_merge(merged, overrides)
    merged["_config_path"] = str(config_path)
    validate_config(merged)
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(config.keys()))
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    rollouts = int(config["rollouts"].get("per_problem", 0))
    if rollouts < 1:
        raise ValueError("rollouts.per_problem must be >= 1")

    checkpoints = config["features"].get("prefix_checkpoints", [])
    if not checkpoints or any(int(t) <= 0 for t in checkpoints):
        raise ValueError("features.prefix_checkpoints must contain positive integers")

    seeds = config.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")

    fractions = [
        float(config["splits"].get("train_fraction", 0.0)),
        float(config["splits"].get("validation_fraction", 0.0)),
        float(config["splits"].get("final_test_fraction", 0.0)),
    ]
    if any(value <= 0.0 for value in fractions):
        raise ValueError("all split fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")


def output_path(config: Mapping[str, Any], key: str) -> Path:
    outputs = config.get("outputs", {})
    if key not in outputs:
        raise KeyError(f"Unknown output key: {key}")
    return Path(str(outputs[key]))


def ensure_output_dirs(config: Mapping[str, Any], keys: Optional[Iterable[str]] = None) -> None:
    selected = list(keys) if keys is not None else list(config.get("outputs", {}).keys())
    for key in selected:
        path = output_path(config, key)
        suffix = path.suffix.lower()
        directory = path if suffix == "" else path.parent
        directory.mkdir(parents=True, exist_ok=True)
