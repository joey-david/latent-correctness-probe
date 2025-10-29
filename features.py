from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

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
    include_prefix_text: bool = False,
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

        layer_hidden = outputs.hidden_states[-1].squeeze(0).detach()
        seq_len = layer_hidden.shape[0]
        reasoning_len = min(reasoning_slice.shape[-1], seq_len)
        if reasoning_len == 0:
            pooled_hidden = layer_hidden[-1]
        else:
            pool_width = min(4, reasoning_len)
            reason_end = seq_len
            reason_start = seq_len - reasoning_len
            start_idx = max(reason_start, reason_end - pool_width)
            pooled_hidden = layer_hidden[start_idx:reason_end].mean(dim=0)
        # Numpy cannot represent bfloat16 tensors, so cast to float32 before moving to CPU.
        hidden_state = pooled_hidden.to(torch.float32).cpu().numpy()

        prefix_text = tokenizer.decode(reasoning_slice, skip_special_tokens=True)
        think_closed = "</think>" in prefix_text
        leaky = contains_answer(prefix_text, leak_answer_str) or think_closed

        next_token_logprob: Optional[float] = None
        next_token_entropy: Optional[float] = None
        if hasattr(outputs, "logits") and outputs.logits is not None:
            next_token_logits = outputs.logits[0, -1].to(torch.float32)
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            probs = log_probs.exp()
            next_token_entropy = float(-(probs * log_probs).sum().item())
            if t < len(gen_token_ids):
                next_token_id = int(gen_token_ids[t].item())
                next_token_logprob = float(log_probs[next_token_id].item())

        entry = {
            "h_t": hidden_state,
            "leaky": leaky,
            "has_think_close": think_closed,
            "next_token_logprob": next_token_logprob,
            "next_token_entropy": next_token_entropy,
            "prefix_len": t,
        }
        if include_prefix_text:
            entry["prefix_text"] = prefix_text
        results[t] = entry

    return results
