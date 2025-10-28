import argparse
import json
from typing import List

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_dataset import build_probe_data
from data import MATH_SUBJECTS, load_math_split
from plotting import plot_probe_results
from probes import run_all_probes


def load_model(model_id: str):
    """
    Load a reasoning model with FP16 weights and automatic device placement.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
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
    show_progress: bool = False,
) -> List[dict]:
    print(f"Loading MATH split '{split}' (subjects={subjects}, min_level={min_level})...")
    return load_math_split(
        split=split,
        subjects=subjects,
        min_level=min_level,
        require_numeric=require_numeric,
        show_progress=show_progress,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-8B", help="HF model identifier"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=800,
        help="Maximum number of dataset items to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    subjects = list(MATH_SUBJECTS)
    min_level = 1  # raise to 3 or higher if you only want harder problems
    require_numeric = True  # drop examples whose gold answer cannot be normalised

    math_train = load_math_data(
        split="train",
        subjects=subjects,
        min_level=min_level,
        require_numeric=require_numeric,
        show_progress=True,
    )

    model_id = args.model_id
    tag = model_id.split("/")[-1]

    print(f"Loading model '{model_id}'...")
    model, tokenizer = load_model(model_id)
    print("Collecting hidden states...")
    total_examples = min(len(math_train), args.max_items)
    if total_examples == 0:
        print("No examples available after filtering; exiting.")
        return
    with tqdm(
        total=total_examples,
        desc="Collecting hidden states",
        unit="example",
    ) as progress:
        features_by_t, labels_by_t, _ = build_probe_data(
            model,
            tokenizer,
            math_train,
            max_items=args.max_items,
            progress_bar=progress,
        )
    probe_results = run_all_probes(features_by_t, labels_by_t)
    plot_probe_results(
        probe_results,
        title=f"{tag} correctness probe vs prefix length",
        outpath=f"{tag}_curve.png",
    )
    pretty_print_results(tag, probe_results)


if __name__ == "__main__":
    main()
