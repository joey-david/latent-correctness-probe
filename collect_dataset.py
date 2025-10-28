from typing import Any, Dict, List, Optional, Tuple

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
) -> Tuple[Dict[int, List[Any]], Dict[int, List[int]], List[Dict[str, Any]]]:
    """
    Runs generation + feature extraction over the dataset and returns:
    - features_by_t: checkpoints -> [hidden_state vectors]
    - labels_by_t: checkpoints -> [0/1 correctness labels]
    - per_example_meta: optional metadata for manual inspection
    """
    features_by_t: Dict[int, List[Any]] = {t: [] for t in CHECKPOINT_STEPS}
    labels_by_t: Dict[int, List[int]] = {t: [] for t in CHECKPOINT_STEPS}
    per_example_meta: List[Dict[str, Any]] = []

    for ex in math_data[:max_items]:
        question = ex["problem"]
        cot_info = generate_cot(model, tokenizer, question)
        label = compute_correct_label(ex, cot_info["model_ans_norm"])

        prefix_states = get_prefix_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt_text=cot_info["prompt_text"],
            gen_token_ids=cot_info["gen_token_ids"],
            checkpoints=CHECKPOINT_STEPS,
            max_prefix_tokens=MAX_PREFIX_TOKENS,
            leak_answer_str=cot_info["model_ans_raw"],
        )

        for t, info in prefix_states.items():
            if info["leaky"]:
                continue  # skip prefixes where the final answer already appeared
            features_by_t[t].append(info["h_t"])
            labels_by_t[t].append(label)

        per_example_meta.append(
            {
                "question": question,
                "model_final_answer": cot_info["model_ans_raw"],
                "gold_answer": ex.get("gold_answer_raw"),
                "is_correct": label,
                "think_text": cot_info.get("think_text"),
                "answer_text": cot_info.get("answer_text"),
                "prefix_states": prefix_states,
            }
        )
        if progress_bar is not None:
            progress_bar.update(1)

    return features_by_t, labels_by_t, per_example_meta
