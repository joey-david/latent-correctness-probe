from typing import Any, Callable, Dict, List, Optional, Tuple

from core_deps import CHECKPOINT_STEPS, MAX_PREFIX_TOKENS
from features import get_prefix_hidden_states
from gen import generate_cot
from labels import compute_correct_label


def build_probe_data(
    model,
    tokenizer,
    math_data: List[Dict[str, Any]],
    max_items: int = 400,
    progress_bar: Optional[Any] = None,
    print_prefixes: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[int, List[Any]], Dict[int, List[int]], List[Dict[str, Any]]]:
    """
    Runs generation + feature extraction over the dataset and returns:
    - features_by_t: checkpoints -> [hidden_state vectors]
    - labels_by_t: checkpoints -> [0/1 correctness labels]
    - per_example_meta: optional metadata for manual inspection
    """
    features_by_t: Dict[int, List[Any]] = {t: [] for t in CHECKPOINT_STEPS}
    labels_by_t: Dict[int, List[int]] = {t: [] for t in CHECKPOINT_STEPS}
    totals_by_t: Dict[int, Dict[str, int]] = {
        t: {"total": 0, "pos": 0} for t in CHECKPOINT_STEPS
    }
    num_correct = 0
    per_example_meta: List[Dict[str, Any]] = []

    writer = progress_bar.write if (progress_bar is not None) else print

    for idx, ex in enumerate(math_data[:max_items], start=1):
        question = ex["problem"]
        cot_info = generate_cot(model, tokenizer, question)
        gen_token_ids = cot_info["gen_token_ids"]
        gen_token_list = gen_token_ids.tolist()
        answer_text = cot_info.get("answer_text") or ""
        answer_len_tokens = len(gen_token_list)
        answer_len_chars = len(answer_text)
        last_token_ids = gen_token_list[-4:] if answer_len_tokens >= 4 else gen_token_list
        last_token_pieces = tokenizer.convert_ids_to_tokens(last_token_ids)
        last_token_text = tokenizer.convert_tokens_to_string(last_token_pieces).strip()
        label = compute_correct_label(ex, cot_info["model_ans_norm"])
        num_correct += int(label)
        forced = bool(cot_info.get("forced_completion", False))

        prefix_states = get_prefix_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt_text=cot_info["prompt_text"],
            gen_token_ids=cot_info["gen_token_ids"],
            checkpoints=CHECKPOINT_STEPS,
            max_prefix_tokens=MAX_PREFIX_TOKENS,
            leak_answer_str=cot_info["model_ans_raw"],
            include_prefix_text=print_prefixes,
        )

        if print_prefixes:
            writer(f"Example {idx}:")
            writer(f"  Question: {question.strip()}")
            writer(f"  Model answer: {cot_info.get('answer_text')}")
            if forced:
                writer("  (forced completion)")
            for t in sorted(prefix_states.keys()):
                info = prefix_states[t]
                prefix_text = info.get("prefix_text", "")
                status = "leaky" if info["leaky"] else "clean"
                writer(f"    t={t:>3} ({status}): {prefix_text}")
        if log_fn and print_prefixes:
            log_fn(f"Example {idx}:")
            log_fn(f"  Question: {question.strip()}")
            log_fn(f"  Model answer: {cot_info.get('answer_text')}")
            if forced:
                log_fn("  (forced completion)")
            for t in sorted(prefix_states.keys()):
                info = prefix_states[t]
                prefix_text = info.get("prefix_text", "")
                status = "leaky" if info["leaky"] else "clean"
                log_fn(f"    t={t:>3} ({status}): {prefix_text}")

        for t, info in prefix_states.items():
            if info["leaky"]:
                continue  # skip prefixes where the final answer already appeared
            features_by_t[t].append(info["h_t"])
            labels_by_t[t].append(label)
            totals_by_t[t]["total"] += 1
            totals_by_t[t]["pos"] += int(label)

        if log_fn:
            leaky_ts = [t for t, info in prefix_states.items() if info["leaky"]]
            stored_ts = [t for t, info in prefix_states.items() if not info["leaky"]]
            stats = ", ".join(
                f"t={t}:n={totals_by_t[t]['total']},pos={totals_by_t[t]['pos']}"
                for t in CHECKPOINT_STEPS
                if totals_by_t[t]["total"] > 0
            )
            running_acc = num_correct / idx if idx > 0 else 0.0
            log_fn(
                f"[example {idx}] correct={bool(label)} forced={forced} "
                f"stored={stored_ts} leaky={leaky_ts} running_acc={running_acc:.3f}"
            )
            if stats:
                log_fn(f"  prefix_stats: {stats}")
            log_fn(
                f"  answers: model={cot_info['model_ans_raw']} gold={ex.get('gold_answer_raw')}"
            )

        per_example_meta.append(
            {
                "question": question,
                "model_final_answer": cot_info["model_ans_raw"],
                "gold_answer": ex.get("gold_answer_raw"),
                "is_correct": label,
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "difficulty_bin": ex.get("difficulty_bin"),
                "think_text": cot_info.get("think_text"),
                "answer_text": cot_info.get("answer_text"),
                "answer_len_tokens": answer_len_tokens,
                "answer_len_chars": answer_len_chars,
                "last_token_ids": last_token_ids,
                "last_token_pieces": last_token_pieces,
                "last_token_text": last_token_text,
                "prefix_states": prefix_states,
                "forced_completion": forced,
            }
        )
        if progress_bar is not None:
            progress_bar.update(1)

    return features_by_t, labels_by_t, per_example_meta
