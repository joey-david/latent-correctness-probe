# Plan: Strengthen Temporal Predictors Into an Actionable Reasoning-Control Paper

## Goal

Turn the current preprint and codebase from a diagnostic probing paper into a stronger experimental paper showing whether latent correctness signals can improve reasoning at inference time.

The central hypothesis to test is:

> When a reasoning model is wrong, or internally represents that it is likely wrong, a hidden-state monitor can trigger reassessment, restart, steering, or difficulty-conditioned routing that improves final accuracy over verified baselines.

The paper should not be framed around the basic fact that early hidden states predict correctness. That is now the baseline result. The stronger paper needs to establish whether the signal is useful for adaptive control, and if not, explain precisely why.

All GPU work should be run through `ssh lamgate`, using the smallest GPU that fits the current job according to the local lamgate GPU-selection guide. Do not default to the largest available GPU unless memory or runtime measurements require it.

## Current Repo State To Account For

- The manuscript in `paper/main.tex` is a short two-column draft centered on greedy single-rollout probing.
- Existing results show strong early Qwen3-8B hidden-state predictability on MATH, with Qwen `t=4` ROC-AUC around `0.86`.
- The current experiment pipeline supports fixed-prefix hidden-state extraction and PCA plus logistic-regression probes.
- The repo does not yet support multi-rollout matched-problem evaluation, reassessment interventions, activation steering, difficulty-conditioned routing, or final locked test evaluations.
- `literature/` exists but is empty, despite `progress.md` referring to literature files.
- `main.py` is missing, while `scripts/run_llama3_8b.sh` still refers to it.
- Local Python dependencies are not installed in the current `.venv`.
- `Dockerfile` is currently deleted in the worktree; do not restore or modify it unless explicitly needed.

## Phase 1: Repair And Reproducibility

Create a runnable experiment entrypoint before adding new research logic.

- Add a config-driven CLI with subcommands or scripts for:
  - generation,
  - hidden-state extraction,
  - probe training,
  - baseline evaluation,
  - intervention runs,
  - plotting.
- Keep all major run settings in versioned YAML configs:
  - model ID,
  - dataset and split,
  - prompt template,
  - number of rollouts per problem,
  - decoding parameters,
  - prefix checkpoints,
  - selected layers,
  - intervention policy,
  - random seeds,
  - output paths.
- Every run must write a manifest containing:
  - git commit or dirty-tree marker,
  - host,
  - GPU name and memory,
  - command line,
  - config path,
  - model ID,
  - dataset version,
  - split manifest,
  - output artifact paths.
- Add deterministic split manifests:
  - train,
  - validation,
  - final test.
- Use problem-disjoint splits only.
- Never tune thresholds, steering coefficients, or prompt policies on the final test split.
- Add smoke tests that do not require a GPU:
  - answer parsing,
  - leakage detection,
  - split construction,
  - metric aggregation,
  - config loading,
  - manifest writing.
- Add one small GPU smoke command to verify the full pipeline on lamgate.

## Phase 2: Literature Review And Positioning

Rebuild the literature review from scratch and store it under `literature/`.

Create:

- `literature/sota_review.md`
- `literature/paper_matrix.md`
- `literature/source_links.md`
- `literature/references.bib`

Use Zotero as the working source of truth for collection, tagging, notes, and BibTeX export. The review should be kept current through May 2026.

Required literature buckets:

- Hidden-state truth and correctness:
  - Burns et al., latent knowledge / CCS.
  - Azaria and Mitchell, internal state knows when lying.
  - Inside-Out hidden factual knowledge.
  - question-only answer-accuracy probes.
- Chain-of-thought hidden-state correctness:
  - papers probing intermediate CoT correctness or future outcome.
  - reasoning-model self-verification work.
  - hidden error-awareness papers.
- Reasoning trajectories and overthinking:
  - CoT prompting,
  - self-consistency,
  - least-to-most,
  - overthinking and recovery/failure trajectories.
- Inference-time control:
  - early stopping CoT,
  - answer convergence,
  - entropy stopping,
  - DeepConf-style trace confidence,
  - selective self-consistency.
- Self-correction and reassessment:
  - Self-Refine,
  - Reflexion-style methods,
  - verbal critique/restart prompts,
  - evidence on when self-correction works or fails.
- Representation steering and activation intervention:
  - activation addition,
  - contrastive activation addition,
  - representation engineering,
  - activation patching/causal mediation in LMs.
- Causal and latent reasoning:
  - causal sufficiency/necessity for CoT,
  - latent CoT,
  - internalized reasoning,
  - depth-recurrent or recurrent latent reasoning models.

For each paper, record:

- citation key,
- URL or DOI,
- model family,
- dataset,
- signal type,
- intervention type if any,
- whether claims are diagnostic, causal, or control-oriented,
- what baseline it threatens in this project,
- how this paper should cite or distinguish it.

## Phase 3: Strong Diagnostic Baseline

Before testing interventions, rebuild the diagnostic result in a harder and more defensible setting.

### Multi-Rollout Dataset

- Generate multiple sampled rollouts per problem, not just greedy traces.
- Use MATH or MATH500 as the core benchmark.
- Include enough problems with mixed outcomes:
  - at least one correct and one incorrect rollout for the same problem.
- Save stable IDs:
  - `problem_id`,
  - `rollout_id`,
  - `model_id`,
  - `prompt_id`,
  - `seed`,
  - `decoding_config_id`.
- Store rollouts as JSONL.
- Store hidden states separately in compressed shards.
- Keep enough metadata to rerun or audit any example.

### Matched-Problem Evaluation

Measure both:

- problem-level predictability: is this problem intrinsically easy or hard?
- trajectory-level predictability: among rollouts for the same problem, is this particular trajectory headed toward a correct answer?

Required evaluations:

- standard train/test AUC and accuracy,
- grouped bootstrap confidence intervals by problem,
- matched correct-vs-incorrect rollout comparisons for the same problem,
- held-out subject transfer,
- easy-to-hard and hard-to-easy transfer,
- short-to-long and long-to-short transfer,
- shuffled-label sanity checks,
- randomized-prefix sanity checks.

### Baselines

Compare hidden-state probes against:

- question-only activation probe,
- partial-CoT text classifier,
- masked partial-CoT text classifier with numeric answer cues removed,
- next-token entropy,
- next-token logprob,
- global entropy summaries,
- prefix length,
- final length,
- difficulty label,
- subject label,
- verbalized confidence,
- answer convergence,
- self-consistency vote margins,
- ES-CoT-style early stopping,
- DeepConf-style trace confidence.

The paper should only claim a latent trajectory signal if the hidden-state probe beats these baselines under problem-disjoint and matched-problem evaluation.

## Phase 4: Reassessment And Restart Interventions

This is the main new paper direction.

Train a calibrated hidden-state monitor on training data, tune thresholds on validation, then use it to trigger reassessment on final test.

### Text-Based Interventions

Test at prefix checkpoints such as `t = 16`, `32`, and `64`.

When the monitor score is below a validation-selected threshold, run one of:

- continue with explicit reassessment:
  - `Wait. This path may be flawed. Reassess the solution from the beginning, identify the first possible mistake, and correct it before giving the final answer.`
- forced restart:
  - stop the current rollout and solve the original problem again from scratch with a restart prompt.
- alternate strategy prompt:
  - ask for a different solution method, verification equation, or independent check.
- selective self-consistency:
  - sample extra rollouts only for low-confidence traces.

Use validation to choose:

- trigger checkpoint,
- confidence threshold,
- intervention prompt,
- maximum extra token budget.

Final test metrics:

- accuracy,
- token count,
- accuracy per token,
- risk-coverage curve,
- selective accuracy,
- intervention rate,
- false-positive trigger rate,
- false-negative miss rate.

### Baselines For Text Interventions

Compare against:

- greedy full CoT,
- same-token-budget random restart,
- always reassess,
- always restart,
- fixed self-consistency,
- entropy-triggered reassessment,
- answer-convergence-triggered reassessment,
- difficulty-triggered reassessment,
- hidden probe without intervention.

The result is meaningful only if hidden-triggered reassessment beats same-token-budget baselines.

## Phase 5: Activation Steering And Causal Stress Tests

Run only after the monitor is validated.

### Steering Direction

Learn a correctness direction from training activations:

- contrast correct vs incorrect rollouts,
- preferably within matched problems,
- compute direction per layer and checkpoint,
- normalize and store the direction with its training config.

### Intervention Grid

Tune on validation:

- layer,
- checkpoint,
- coefficient,
- duration,
- add direction vs subtract direction,
- single-token vs multi-token steering window.

Measure on final test:

- answer accuracy,
- coherence,
- parse rate,
- token count,
- repetition rate,
- invalid output rate.

### Controls

Include:

- random direction,
- opposite direction,
- shuffled labels,
- steering at non-predictive layers,
- always-steer,
- low-confidence-only steer,
- no-steer.

Do not make causal claims unless steering or ablation consistently changes outcomes without destroying coherence.

If steering fails, keep the result: it supports the distinction between diagnostic signals and causal control variables.

## Phase 6: Difficulty-Conditioned Policy

Build a policy that uses both latent outcome score and detected difficulty.

Policy candidates:

- easy and high confidence:
  - finish normally or early stop.
- easy and low confidence:
  - trigger reassessment once.
- hard and high confidence:
  - continue normally, optionally verify final answer.
- hard and low confidence:
  - allocate extra samples, restart, or switch to alternate strategy.
- ambiguous confidence:
  - use self-consistency rather than steering.

Difficulty signals can include:

- dataset difficulty when available,
- question-only probe,
- early hidden-state difficulty classifier,
- output length predictor,
- entropy summary.

Report whether difficulty-conditioned policies improve the accuracy/token Pareto curve beyond a single global threshold.

## Phase 7: Figures And Manuscript Rewrite

Rewrite the paper only after the intervention and matched-rollout results are known.

### Replace The PCA Figure

The current PCA graph should be removed. It does not clearly support the story.

Replace it with one of:

- layer-by-prefix AUC heatmap,
- layer-by-prefix calibration heatmap,
- held-out 2D projection colored by correctness and difficulty,
- matched-problem correct-vs-incorrect separation plot.

The strongest figure set should be:

1. matched-rollout setup diagram,
2. diagnostic monitor performance against baselines,
3. layer-by-time signal heatmap,
4. intervention accuracy/token Pareto curve,
5. difficulty-conditioned routing result,
6. qualitative examples of failure, reassessment, and recovery.

### Manuscript Framing

The paper should make one of two claims, depending on results.

If interventions improve performance:

> Hidden states provide an early outcome monitor that can improve inference-time reasoning when used to selectively trigger reassessment, restart, or extra computation.

If interventions do not improve performance:

> Hidden states robustly encode future correctness, but the signal is mostly diagnostic; standard text and activation interventions do not reliably convert it into better reasoning.

Do not overclaim that the model "knows" it is wrong unless the evidence distinguishes latent difficulty, trajectory quality, and answer leakage.

## Acceptance Criteria

The project is ready for a strong paper draft when:

- current diagnostic results are reproduced from a runnable CLI,
- multi-rollout matched-problem data exists,
- hidden probes beat text, entropy, length, difficulty, and self-consistency baselines,
- intervention thresholds are tuned only on validation,
- final test results are run once with locked configs,
- adaptive policies are compared to same-token-budget baselines,
- PCA figure is replaced,
- literature review is updated through May 2026,
- all claims in the paper are backed by saved artifacts and manifest files.

The project should not claim a positive control result unless hidden-triggered reassessment, restart, steering, or routing improves final test accuracy or the accuracy/token Pareto curve over verified baselines.

## Execution Notes

- Run GPU jobs via `ssh lamgate`.
- Use the smallest lamgate GPU that fits the job, following the local lamgate dynamic GPU-selection guide.
- Start with small debug runs before launching full hidden-state collection.
- Save every long-running command, log path, host, GPU, and expected artifact in `progress.md`.
- Keep `upgrade_plan.md` aligned with this plan or replace it after `plan.md` becomes the primary checklist.
- Preserve existing untracked manuscript files and figures unless explicitly replacing them as part of the paper rewrite.
