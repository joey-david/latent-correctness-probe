from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


INTERVENTION_PROMPTS = {
    "reassess": (
        "Wait. This path may be flawed. Reassess the solution from the beginning, "
        "identify the first possible mistake, and correct it before giving the final answer."
    ),
    "restart": (
        "Stop this attempt. Solve the original problem again from scratch, using an "
        "independent derivation, then give the final numeric answer in \\boxed{...}."
    ),
    "alternate_strategy": (
        "Use a different solution method or verification equation. Check the answer "
        "independently before giving the final numeric answer in \\boxed{...}."
    ),
    "selective_self_consistency": (
        "Sample additional independent solutions and use the majority or most verified "
        "answer after checking for arithmetic consistency."
    ),
}


@dataclass(frozen=True)
class InterventionDecision:
    triggered: bool
    policy: str
    prompt_id: str
    prompt_text: str
    reason: str
    score: float
    threshold: float


def decide_intervention(score: float, config: Mapping[str, Any]) -> InterventionDecision:
    intervention = config.get("intervention", config)
    policy = str(intervention.get("policy", "none"))
    threshold = float(intervention.get("threshold", 0.5))
    prompt_id = str(intervention.get("prompt", "reassess"))
    prompt_text = INTERVENTION_PROMPTS.get(prompt_id, INTERVENTION_PROMPTS["reassess"])

    if policy == "none":
        return InterventionDecision(
            triggered=False,
            policy=policy,
            prompt_id=prompt_id,
            prompt_text="",
            reason="policy_none",
            score=float(score),
            threshold=threshold,
        )

    triggered = float(score) < threshold
    reason = "low_hidden_confidence" if triggered else "above_threshold"
    return InterventionDecision(
        triggered=triggered,
        policy=policy,
        prompt_id=prompt_id,
        prompt_text=prompt_text if triggered else "",
        reason=reason,
        score=float(score),
        threshold=threshold,
    )


def route_by_difficulty(score: float, difficulty: str, config: Mapping[str, Any]) -> str:
    routing = config.get("routing", {})
    low = float(routing.get("low_confidence_threshold", 0.5))
    high = float(routing.get("high_confidence_threshold", 0.8))
    difficulty_norm = difficulty.lower()
    if score >= high and difficulty_norm == "easy":
        return "finish_or_early_stop"
    if score < low and difficulty_norm == "easy":
        return "reassess_once"
    if score >= high and difficulty_norm == "hard":
        return "continue_then_verify"
    if score < low and difficulty_norm == "hard":
        return "allocate_extra_samples"
    return "self_consistency"
