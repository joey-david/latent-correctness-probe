# Progress

## Current Objective

Upgrade the latent correctness probe project into an ICLR-ready research program by testing trajectory-level latent outcome monitoring, strong confound controls, and adaptive inference use.

## Active Checklist

- [x] Use `plan.md` as the working checklist.
- [x] Use `literature/sota_review.md`, `literature/paper_matrix.md`, `literature/source_links.md`, and `literature/references.bib` for SOTA positioning.
- [ ] Keep big paper-identity decisions after experimental evidence, per `plan.md`.
- [ ] Update this file at the start/end of substantive work and before context transitions.

## Session Log

### 2026-05-27 17:28 CEST

- Implemented the local reproducibility/control scaffold from `plan.md`:
  - config-driven CLI in `main.py` / `src/probing/cli.py`,
  - YAML configs under `configs/`,
  - manifest writing with git, host, GPU, command, config, split, and artifact metadata,
  - deterministic problem-disjoint split utilities,
  - CPU-safe answer parsing, leakage, metrics, intervention policy, and JSONL helpers,
  - dry-run generation and hidden-state extraction commands for local validation,
  - baseline/intervention/plotting command plumbing.
- Rebuilt `literature/` with SOTA review, matrix, links, and BibTeX starter file through May 2026.
- Added CPU smoke tests in `tests/test_smoke.py`.
- Verified:
  - `PYTHONPATH=src python3 -m unittest discover -s tests`
  - `PYTHONPATH=src python3 -m probing.cli generation --config configs/qwen_math_smoke.yaml --dry-run`
  - `PYTHONPATH=src python3 -m probing.cli hidden-state-extraction --config configs/qwen_math_smoke.yaml --dry-run`
  - `PYTHONPATH=src python3 -m py_compile main.py src/probing/*.py tests/test_smoke.py`
- Local `.venv` remains broken because its Python symlink points to a missing mise install; system `python3` was used for smoke tests.
- No GPU jobs were launched locally. Run `configs/lamgate_gpu_smoke.yaml` through `ssh lamgate` for the required GPU smoke.

# Restart Notes

- Start future substantial work by reading this file, then `plan.md`.
- Remote working copy is available at `s205:~/research/independent/latent_correctness_probe/`.
- Remote Codex is already running in tmux session `latent-correctness-codex`; attach before starting a duplicate.
- The immediate research priority is multi-rollout same-problem experiments to separate problem-level difficulty from trajectory-level correctness.
- The closest SOTA threats are documented in `literature/sota_review.md`; do not pitch simple early correctness probing as the main novelty.
