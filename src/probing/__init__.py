"""Core probing pipeline utilities."""

from .analysis import analyze_baselines, analyze_difficulty_buckets
from .collection import build_probe_data
from .config import CHECKPOINT_STEPS, DEVICE, MAX_NEW_TOKENS, MAX_PREFIX_TOKENS, SEED
from .data import (
    MATH_SUBJECTS,
    extract_boxed,
    extract_last_number,
    load_math_split,
    normalize_gold_answer,
    normalize_num_str,
    parse_model_answer,
)
from .features import contains_answer, get_prefix_hidden_states
from .generation import build_prompt, generate_cot
from .labels import compute_correct_label
from .probes import (
    eligible_example_indices,
    collect_features_for_indices,
    run_probes_from_meta,
    run_all_probes,
    train_eval_probe,
)

__all__ = [
    "analyze_baselines",
    "analyze_difficulty_buckets",
    "build_probe_data",
    "CHECKPOINT_STEPS",
    "DEVICE",
    "MAX_NEW_TOKENS",
    "MAX_PREFIX_TOKENS",
    "SEED",
    "MATH_SUBJECTS",
    "extract_boxed",
    "extract_last_number",
    "load_math_split",
    "normalize_gold_answer",
    "normalize_num_str",
    "parse_model_answer",
    "contains_answer",
    "get_prefix_hidden_states",
    "build_prompt",
    "generate_cot",
    "compute_correct_label",
    "eligible_example_indices",
    "collect_features_for_indices",
    "run_probes_from_meta",
    "run_all_probes",
    "train_eval_probe",
]
