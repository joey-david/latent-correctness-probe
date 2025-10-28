from typing import Any, Dict, Optional

from data import gold_answer_str, normalize_num_str


def compute_correct_label(example_gsm: Dict[str, Any], model_ans_norm: Optional[str]) -> int:
    """
    Binary correctness: 1 if the model's normalized numeric answer matches gold.
    """
    gold_ans = gold_answer_str(example_gsm)
    gold_norm = normalize_num_str(gold_ans)
    if gold_norm is None or model_ans_norm is None:
        return 0
    return int(gold_norm == model_ans_norm)
