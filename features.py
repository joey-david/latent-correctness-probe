from typing import Any, Dict, List, Optional

import torch

def contains_answer(prefix_text: str, answer_str: Optional[str]) -> bool:
    """
    Extremely simple leakage test: if the numeric answer is already visible in the prefix.
    """
    if answer_str is None:
        return False
    return answer_str in prefix_text


@torch.no_grad()
def get_prefix_hidden_states(
    model,
    tokenizer,
    prompt_text: str,
    gen_token_ids: torch.Tensor,
    checkpoints: List[int],
    max_prefix_tokens: int,
    leak_answer_str: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    For each checkpoint length t, run the model forward on prompt + first t generated tokens
    and record the final hidden state vector. This is the feature we probe.
    """
    results: Dict[int, Dict[str, Any]] = {}

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids[0].to(model.device)

    for t in checkpoints:
        if t > len(gen_token_ids):
            continue
        if t > max_prefix_tokens:
            continue

        reasoning_slice = gen_token_ids[:t].to(model.device)
        full_ids = torch.cat([prompt_ids, reasoning_slice], dim=-1).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids, device=model.device)

        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_state = outputs.hidden_states[-1][:, -1, :].squeeze(0).detach().cpu().numpy()

        prefix_text = tokenizer.decode(reasoning_slice, skip_special_tokens=True)
        think_closed = "</think>" in prefix_text
        leaky = contains_answer(prefix_text, leak_answer_str) or think_closed

        results[t] = {"h_t": hidden_state, "leaky": leaky, "has_think_close": think_closed}

    return results
