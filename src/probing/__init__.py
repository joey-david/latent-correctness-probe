"""Core probing pipeline utilities.

The package initializer intentionally avoids importing GPU/scientific stacks.
Import concrete submodules directly, for example ``from probing.data import ...``
or ``from probing.probes import ...``.
"""

_EXPORTS = {
    "analyze_baselines": "analysis",
    "analyze_difficulty_buckets": "analysis",
    "build_probe_data": "collection",
    "CHECKPOINT_STEPS": "config",
    "DEVICE": "config",
    "MAX_NEW_TOKENS": "config",
    "MAX_PREFIX_TOKENS": "config",
    "SEED": "config",
    "MATH_SUBJECTS": "data",
    "extract_boxed": "data",
    "extract_last_number": "data",
    "load_math_split": "data",
    "normalize_gold_answer": "data",
    "normalize_num_str": "data",
    "parse_model_answer": "data",
    "contains_answer": "features",
    "get_prefix_hidden_states": "features",
    "build_prompt": "generation",
    "generate_cot": "generation",
    "compute_correct_label": "labels",
    "eligible_example_indices": "probes",
    "collect_features_for_indices": "probes",
    "run_probes_from_meta": "probes",
    "run_all_probes": "probes",
    "train_eval_probe": "probes",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
