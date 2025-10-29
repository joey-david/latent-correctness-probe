import argparse
import json
from collections import Counter
from typing import List

from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_dataset import build_probe_data
from core_deps import CHECKPOINT_STEPS, SEED
from data import MATH_SUBJECTS, load_math_split, sample_balanced_by_difficulty
from plotting import plot_multiple_probe_results, plot_probe_results
from probes import run_all_probes, run_probes_from_meta


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


def parse_level_list(spec: str) -> List[int]:
    levels: List[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            levels.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid level '{token}' in specification '{spec}'") from exc
    if not levels:
        raise ValueError(f"No levels provided in specification '{spec}'")
    return levels


def compute_base_accuracy(meta: List[dict], indices: List[int]) -> float:
    if not indices:
        return float("nan")
    correct = sum(int(meta[idx].get("is_correct", 0)) for idx in indices)
    return correct / len(indices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-8B", help="HF model identifier"
    )
    parser.add_argument(
        "--total-examples",
        "--max-items",
        dest="total_examples",
        type=int,
        default=1500,
        help="Total number of dataset items to process (must be even for balanced sampling)",
    )
    parser.add_argument(
        "--easy-levels",
        type=str,
        default="1,2",
        help="Comma-separated list of MATH difficulty levels treated as 'easy'",
    )
    parser.add_argument(
        "--hard-levels",
        type=str,
        default="4,5",
        help="Comma-separated list of MATH difficulty levels treated as 'hard'",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=SEED,
        help="Seed used when sampling the difficulty-balanced subset",
    )
    parser.add_argument(
        "--fixed-prefix-max",
        type=int,
        default=256,
        help="Require examples to reach at least this prefix length for fixed-subset analysis",
    )
    parser.add_argument(
        "--skip-fixed-subset",
        action="store_true",
        help="Disable the fixed-prefix subset analysis",
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
        easy_levels = parse_level_list(args.easy_levels)
        hard_levels = parse_level_list(args.hard_levels)
        total_examples = args.total_examples
        log_line(
            "=== Run started === "
            f"model={args.model_id} total_examples={total_examples} "
            f"easy_levels={easy_levels} hard_levels={hard_levels} "
            f"balance_seed={args.balance_seed} fixed_prefix_max={args.fixed_prefix_max} "
            f"skip_fixed_subset={args.skip_fixed_subset} print_prefixes={args.print_prefixes}"
        )
        subjects = list(MATH_SUBJECTS)
        min_level = 1  # raise to 3 or higher for harder problems
        require_numeric = True  # drop examples whose gold answer cannot be normalized

        try:
            if total_examples <= 0:
                log_line("Requested total_examples <= 0; exiting early.")
                print("No examples requested; exiting.")
                return
            if total_examples % 2 != 0:
                raise ValueError("total_examples must be even for balanced sampling")

            math_train = load_math_data(
                split="train",
                subjects=subjects,
                min_level=min_level,
                require_numeric=require_numeric,
                show_progress=True,
            )
            log_line(f"Loaded {len(math_train)} math examples.")

            balanced_examples = sample_balanced_by_difficulty(
                math_train,
                total=total_examples,
                easy_levels=easy_levels,
                hard_levels=hard_levels,
                seed=args.balance_seed,
            )
            difficulty_counts = Counter(
                ex.get("difficulty_bin", "unknown") for ex in balanced_examples
            )
            level_counts = Counter(ex.get("level") for ex in balanced_examples)
            log_line(
                "Balanced subset stats: "
                f"difficulty_counts={dict(difficulty_counts)} level_counts={dict(level_counts)}"
            )

            model_id = args.model_id
            tag = model_id.split("/")[-1]

            print(f"Loading model '{model_id}'...")
            log_line(f"Loading model '{model_id}'")
            model, tokenizer = load_model(model_id)
            print("Collecting hidden states...")
            log_line("Collecting hidden states...")
            total_examples = len(balanced_examples)
            log_line(f"Processing {total_examples} difficulty-balanced examples.")
            with tqdm(
                total=total_examples,
                desc="Collecting hidden states",
                unit="example",
            ) as progress:
                features_by_t, labels_by_t, per_example_meta = build_probe_data(
                    model,
                    tokenizer,
                    balanced_examples,
                    max_items=total_examples,
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
            subset_results = {"overall": probe_results}

            def log_subset(name: str, labels_dict: dict, indices: List[int]) -> None:
                base_acc = compute_base_accuracy(per_example_meta, indices)
                log_line(
                    f"Subset '{name}': examples={len(indices)} base_acc={base_acc:.3f}"
                )
                for step in sorted(labels_dict.keys()):
                    count = len(labels_dict[step])
                    if count == 0:
                        continue
                    pos = int(sum(labels_dict[step]))
                    log_line(f"  t={step}: n={count}, pos={pos}")

            log_subset("overall", labels_by_t, list(range(len(per_example_meta))))

            easy_metrics, easy_labels_by_t, easy_indices = run_probes_from_meta(
                per_example_meta,
                checkpoints=CHECKPOINT_STEPS,
                filter_fn=lambda ex: ex.get("difficulty_bin") == "easy",
                seed=args.balance_seed,
            )
            subset_results["easy"] = easy_metrics

            hard_metrics, hard_labels_by_t, hard_indices = run_probes_from_meta(
                per_example_meta,
                checkpoints=CHECKPOINT_STEPS,
                filter_fn=lambda ex: ex.get("difficulty_bin") == "hard",
                seed=args.balance_seed,
            )
            subset_results["hard"] = hard_metrics

            log_subset("easy", easy_labels_by_t, easy_indices)
            log_subset("hard", hard_labels_by_t, hard_indices)

            fixed_metrics = {}
            fixed_labels_by_t = {}
            fixed_indices: List[int] = []
            if not args.skip_fixed_subset:
                required_ts = [t for t in CHECKPOINT_STEPS if t <= args.fixed_prefix_max]
                if required_ts:
                    fixed_metrics, fixed_labels_by_t, fixed_indices = run_probes_from_meta(
                        per_example_meta,
                        checkpoints=CHECKPOINT_STEPS,
                        required_ts=required_ts,
                        seed=args.balance_seed,
                    )
                    fixed_key = f"fixed<={args.fixed_prefix_max}"
                    subset_results[fixed_key] = fixed_metrics
                    log_subset(
                        fixed_key,
                        fixed_labels_by_t,
                        fixed_indices,
                    )
                else:
                    log_line(
                        "Fixed subset skipped because no checkpoints fall under the threshold."
                    )

            out_path = f"{tag}_curve_overall.png"
            plot_probe_results(
                probe_results,
                title=f"{tag} correctness probe vs prefix length",
                outpath=out_path,
            )
            log_line(f"Saved overall probe plot to {out_path}")

            multi_out_path = f"{tag}_difficulty_curves.png"
            plot_multiple_probe_results(
                subset_results,
                title=f"{tag} probe performance by subset",
                outpath=multi_out_path,
            )
            log_line(f"Saved stratified probe plot to {multi_out_path}")

            pretty_print_results(f"{tag} overall", probe_results)
            pretty_print_results(f"{tag} easy", easy_metrics)
            pretty_print_results(f"{tag} hard", hard_metrics)
            if fixed_metrics:
                pretty_print_results(f"{tag} fixed<={args.fixed_prefix_max}", fixed_metrics)

            results_payload = {
                name: {str(t): metrics for t, metrics in sorted(res.items())}
                for name, res in subset_results.items()
                if res
            }
            results_path = Path(f"{tag}_results.json")
            results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")
            log_line(f"Persisted combined results to {results_path}")
            log_line("Run completed successfully.")
        except Exception as exc:
            log_line(f"Run failed with error: {exc!r}")
            raise


if __name__ == "__main__":
    main()
