from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_deps import MAX_NEW_TOKENS
from data import parse_model_answer


def _is_qwen3(tokenizer: AutoTokenizer) -> bool:
    name = getattr(tokenizer, "name_or_path", "") or ""
    if "qwen3" in name.lower():
        return True
    template = getattr(tokenizer, "chat_template", None)
    if template and "qwen" in template.lower() and "think" in template.lower():
        return True
    return False


def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """
    Builds the prompt for the model. Uses Qwen3 chat templating to enable thinking mode when available.
    """
    if _is_qwen3(tokenizer):
        messages = [
            {
                "role": "system",
                "content": (
                    "Think step by step inside <think>...</think> before responding. "
                    "After thinking, output the final numeric answer only inside \\boxed{...}."
                ),
            },
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    return (
        f"Question: {question}\n\n"
        "Answer:\n"
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
    prompt = build_prompt(question, tokenizer)

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
    think_text = None
    answer_text = gen_text
    if _is_qwen3(tokenizer) and "</think>" in gen_text:
        pre, post = gen_text.split("</think>", 1)
        think_text = pre.replace("<think>", "").strip()
        answer_text = post.strip()

    parsed = parse_model_answer(answer_text)
    reached_limit = gen_ids.shape[0] >= max_new_tokens
    forced_completion = False

    if parsed["raw"] is None or reached_limit:
        forced_completion = True

        needs_think_close = "<think>" in gen_text and "</think>" not in gen_text
        forced_suffix_parts = []
        if needs_think_close:
            forced_suffix_parts.append("</think>")
        forced_suffix_parts.append(FORCED_SUFFIX_SUFFIX)
        forced_suffix = "".join(forced_suffix_parts)

        forced_suffix_ids = tokenizer(
            forced_suffix,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].to(model.device)

        base_input_ids = torch.cat(
            [enc["input_ids"][0], gen_ids, forced_suffix_ids], dim=-1
        )
        continuation = model.generate(
            input_ids=base_input_ids.unsqueeze(0),
            attention_mask=torch.ones_like(base_input_ids).unsqueeze(0),
            max_new_tokens=FORCED_SUFFIX_EXTRA_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation_ids = continuation[0][base_input_ids.shape[0]:]

        gen_ids = torch.cat([gen_ids, forced_suffix_ids, continuation_ids], dim=0)
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if _is_qwen3(tokenizer) and "</think>" in gen_text:
            pre, post = gen_text.split("</think>", 1)
            think_text = pre.replace("<think>", "").strip()
            answer_text = post.strip()
        else:
            think_text = None
            answer_text = gen_text

        parsed = parse_model_answer(answer_text)

    return {
        "prompt_text": prompt,
        "gen_text": gen_text,
        "think_text": think_text,
        "answer_text": answer_text,
        "model_ans_raw": parsed["raw"],
        "model_ans_norm": parsed["norm"],
        "gen_token_ids": gen_ids.detach().cpu(),
        "forced_completion": forced_completion,
    }
FORCED_SUFFIX_SUFFIX = "\nThus the final numeric answer is is \\boxed{"
FORCED_SUFFIX_EXTRA_TOKENS = 32
