# SOTA Review: Latent Correctness Signals for Reasoning Control

Updated through May 2026. Zotero should remain the working source of truth; this
file is the project-facing synthesis used to decide baselines and claims.

## Positioning

The old paper framing, "early hidden states predict final correctness," is now a
baseline result. The stronger claim must be control-oriented: a latent monitor
must improve accuracy, token efficiency, or selective risk over matched
same-budget baselines. If interventions fail, the paper should explicitly argue
for a diagnostic-control gap.

## Hidden-State Truth And Correctness

- Burns et al. introduce Contrast-Consistent Search as an unsupervised way to
  elicit latent truth-like directions from activations. This motivates probing
  hidden states but is not itself an inference-time reasoning-control baseline.
- Azaria and Mitchell show hidden states can reveal whether generated statements
  are likely truthful. This threatens broad "model knows" language unless this
  project separates truth, difficulty, answer leakage, and trajectory quality.
- Inside-Out and related factual-knowledge probing work make question-only and
  answer-only controls mandatory: latent correctness may reflect problem
  difficulty or memorized answer availability rather than rollout quality.

## CoT Hidden-State Correctness

Recent reasoning-model work directly overlaps this project. Hidden error
awareness papers report strong AUROC from early reasoning states, including May
2026 work arguing that the signal is diagnostic rather than causal. This makes
matched rollouts, text baselines, and intervention outcomes central. The paper
should cite these results as the strongest adjacent diagnostic baseline and then
ask whether validation-locked triggers can improve final inference.

## Reasoning Trajectories And Overthinking

CoT prompting, self-consistency, least-to-most prompting, and process-supervised
reasoning all establish that extra reasoning can improve final answers, but also
that longer traces can drift or overthink. This project needs trajectory-level
analysis: for the same problem, does a particular sampled trace have a latent
signature that differs between correct and incorrect futures?

## Inference-Time Control

DeepConf-style trace confidence, entropy stopping, answer convergence, selective
self-consistency, and early-stopping CoT are the most important baselines. They
can improve the accuracy-token Pareto curve without hidden-state probes. Hidden
monitors only matter if they beat these under the same token budget and
problem-disjoint splits.

## Self-Correction And Reassessment

Self-Refine, Reflexion-style restart/critique, and verbal reassessment prompts
show that self-correction is conditional: it often helps when external feedback
or independent samples expose the error, and often fails when the model simply
endorses its first wrong path. This motivates selective triggers and same-budget
random/always-reassess controls.

## Representation Steering And Activation Intervention

Activation Addition, contrastive activation addition, representation
engineering, function vectors, and activation patching provide the intervention
toolkit. They do not justify causal claims by themselves. This project should
only claim a causal correctness variable if steering or ablation changes final
outcomes without destroying parseability, coherence, or token efficiency.

## Causal And Latent Reasoning

CoT monitorability and faithfulness papers show that visible reasoning can be
incomplete or strategically unfaithful, while hidden-state work suggests some
reasoning-relevant information is internalized. This supports a cautious frame:
hidden correctness monitors may detect internal trajectory state, but the
project must not equate detection with controllability.

## Required Baseline Consequences

- Always report question-only, difficulty, prefix-length, final-length, entropy,
  logprob, partial-CoT text, masked partial-CoT text, verbal confidence,
  answer-convergence, self-consistency margin, ES-CoT, and DeepConf-style
  confidence baselines.
- Use problem-disjoint train/validation/final-test splits.
- Tune thresholds, steering coefficients, and intervention prompts only on
  validation.
- Preserve negative intervention results as evidence for the diagnostic-control
  gap.
