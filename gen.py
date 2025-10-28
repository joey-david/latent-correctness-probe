from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_deps import MAX_NEW_TOKENS
from data import parse_model_answer


def build_prompt(question: str) -> str:
    """
    Phrasing emphasises deliberate reasoning so the model emits a chain-of-thought.
    """
    return (
        "You are a careful math tutor. Solve the following problem step by step, "
        "showing your reasoning clearly, then give the final numeric answer.\n\n"
        f"Question: {question}\n\n"
        "Reasoning:\n"
    )


@torch.no_grad()
def generate_cot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """
    Runs greedy decoding to produce a full chain-of-thought completion.
    Returns the generated text plus the parsed numeric answer.
    """
    prompt = build_prompt(question)

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out_ids = out[0]

    prompt_len = enc["input_ids"].shape[1]
    gen_ids = out_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    parsed = parse_model_answer(gen_text)

    return {
        "prompt_text": prompt,
        "gen_text": gen_text,
        "model_ans_raw": parsed["raw"],
        "model_ans_norm": parsed["norm"],
        "gen_token_ids": gen_ids.detach().cpu(),
    }
