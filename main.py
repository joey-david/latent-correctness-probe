import json
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_dataset import build_probe_data
from data import MATH_SUBJECTS, load_math_split
from plotting import plot_probe_results
from probes import run_all_probes


def load_model(model_id: str):
    """
    Load a reasoning model with FP16 weights and automatic device placement.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def pretty_print_results(tag: str, results: dict) -> None:
    ordered = {int(k): v for k, v in sorted(results.items(), key=lambda item: item[0])}
    print(f"{tag} results:")
    print(json.dumps(ordered, indent=2))


def load_math_data(
    split: str,
    subjects: List[str],
    min_level: int,
    require_numeric: bool,
) -> List[dict]:
    print(f"Loading MATH split '{split}' (subjects={subjects}, min_level={min_level})...")
    return load_math_split(
        split=split,
        subjects=subjects,
        min_level=min_level,
        require_numeric=require_numeric,
    )


def main():
    subjects = list(MATH_SUBJECTS)
    min_level = 1  # raise to 3 or higher if you only want harder problems
    require_numeric = True  # drop examples whose gold answer cannot be normalised

    math_train = load_math_data(
        split="train",
        subjects=subjects,
        min_level=min_level,
        require_numeric=require_numeric,
    )

    qwen_id = "Qwen/Qwen2.5-7B-Instruct"
    other_id = "mistralai/Mistral-7B-Instruct-v0.3"
    max_items = 400  # per model; increase if you have compute budget

    print("Loading Qwen model...")
    qwen_model, qwen_tokenizer = load_model(qwen_id)
    print("Collecting hidden states for Qwen...")
    q_features_by_t, q_labels_by_t, _ = build_probe_data(
        qwen_model, qwen_tokenizer, math_train, max_items=max_items
    )
    q_results = run_all_probes(q_features_by_t, q_labels_by_t)
    plot_probe_results(
        q_results,
        title="Qwen correctness probe vs prefix length",
        outpath="qwen_curve.png",
    )
    pretty_print_results("Qwen", q_results)

    print("Loading secondary model...")
    other_model, other_tokenizer = load_model(other_id)
    print("Collecting hidden states for secondary model...")
    other_features_by_t, other_labels_by_t, _ = build_probe_data(
        other_model, other_tokenizer, math_train, max_items=max_items
    )
    other_results = run_all_probes(other_features_by_t, other_labels_by_t)
    plot_probe_results(
        other_results,
        title="Other model correctness probe vs prefix length",
        outpath="other_curve.png",
    )
    pretty_print_results("Other", other_results)


if __name__ == "__main__":
    main()
