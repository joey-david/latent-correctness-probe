import argparse
import json
from typing import List

from datetime import datetime
from pathlib import Path

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
    parser.add_argument(
        "--print-prefixes",
        action="store_true",
        help="Print generated token prefixes for each checkpoint capture",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_path = Path("log.txt")
    log_path.touch(exist_ok=True)

    def make_logger(file_handle):
        def _log(message: str) -> None:
            timestamp = datetime.now().isoformat(timespec="seconds")
            file_handle.write(f"[{timestamp}] {message}\n")
            file_handle.flush()
        return _log

    with log_path.open("a", encoding="utf-8") as log_file:
        log_line = make_logger(log_file)
        log_line(
            f"=== Run started === model={args.model_id} max_items={args.max_items} "
            f"print_prefixes={args.print_prefixes}"
        )
        subjects = list(MATH_SUBJECTS)
        min_level = 1  # raise to 3 or higher for harder problems
        require_numeric = True  # drop examples whose gold answer cannot be normalized

        try:
            math_train = load_math_data(
                split="train",
                subjects=subjects,
                min_level=min_level,
                require_numeric=require_numeric,
                show_progress=True,
            )
            log_line(f"Loaded {len(math_train)} math examples.")

            model_id = args.model_id
            tag = model_id.split("/")[-1]

            print(f"Loading model '{model_id}'...")
            log_line(f"Loading model '{model_id}'")
            model, tokenizer = load_model(model_id)
            print("Collecting hidden states...")
            log_line("Collecting hidden states...")
            total_examples = min(len(math_train), args.max_items)
            log_line(f"Processing up to {total_examples} examples.")
            if total_examples == 0:
                log_line("No examples available after filtering; exiting.")
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
                    print_prefixes=args.print_prefixes,
                    log_fn=log_line,
                )
            log_line("Finished collecting hidden states.")
            for t in sorted(features_by_t.keys()):
                n_total = len(labels_by_t[t])
                pos = int(sum(labels_by_t[t])) if n_total else 0
                log_line(f"Checkpoint t={t}: n={n_total}, pos={pos}")
            probe_results = run_all_probes(features_by_t, labels_by_t)
            for t, metrics in sorted(probe_results.items()):
                log_line(f"Probe t={t}: {metrics}")
            if not probe_results:
                log_line("No probe results available (insufficient data).")
            out_path = f"{tag}_curve.png"
            plot_probe_results(
                probe_results,
                title=f"{tag} correctness probe vs prefix length",
                outpath=out_path,
            )
            log_line(f"Saved probe plot to {out_path}")
            pretty_print_results(tag, probe_results)
            log_line("Run completed successfully.")
        except Exception as exc:
            log_line(f"Run failed with error: {exc!r}")
            raise


if __name__ == "__main__":
    main()
