# Latent Correctness Probe

This repository packages code for studying whether latent correctness signals in
reasoning models are merely diagnostic or can be used for inference-time
control. The workflow is organised into two Python packages:

- `src/probing/`: data loading, prompting, hidden-state extraction, and probe training.
- `src/plotting/`: Matplotlib helpers for visualising probe metrics and diagnostics.

Cached generations (JSONL in `runs/`) and derived metrics (JSON/png in `results/` and `figs/`) can be regenerated or inspected with the utilities below.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Setting `PYTHONPATH` (or running scripts with `PYTHONPATH=src`) makes the packages importable as `import probing` and `import plotting`.

## Config-Driven CLI

The experiment entrypoint is `main.py`. Every substantive command takes a
versioned YAML config and writes a manifest with git state, host/GPU metadata,
command line, config path, split manifest, and output artifact paths.

```bash
PYTHONPATH=src python3 main.py make-splits --config configs/qwen_math_smoke.yaml
PYTHONPATH=src python3 main.py generation --config configs/qwen_math_smoke.yaml --split validation
PYTHONPATH=src python3 main.py hidden-state-extraction --config configs/qwen_math_smoke.yaml
PYTHONPATH=src python3 main.py probe-training --config configs/qwen_math_smoke.yaml
PYTHONPATH=src python3 main.py baseline-evaluation --config configs/qwen_math_smoke.yaml
PYTHONPATH=src python3 main.py intervention-run --config configs/qwen_math_smoke.yaml
PYTHONPATH=src python3 main.py plotting --config configs/qwen_math_smoke.yaml
```

Use dry runs locally to validate paths and manifests without GPU dependencies:

```bash
PYTHONPATH=src python3 main.py generation --config configs/qwen_math_smoke.yaml --dry-run
PYTHONPATH=src python3 main.py hidden-state-extraction --config configs/qwen_math_smoke.yaml --dry-run
```

GPU jobs should be launched through `ssh lamgate` with the smallest GPU that
fits the job according to the local lamgate GPU-selection guide. The minimal
full-pipeline smoke config is `configs/lamgate_gpu_smoke.yaml`.

## Typical Workflow

1. **Load Hendrycks MATH examples**

   ```python
   from probing.data import load_math_split

   math_examples = load_math_split(
       split="train",
       min_level=1,
       require_numeric=True,
   )
   ```

   The helpers in `probing.data` normalise boxed answers and filter to numeric targets.

2. **Generate chains of thought and cache prefix states**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from probing.collection import build_probe_data
   from probing.config import CHECKPOINT_STEPS, MAX_NEW_TOKENS

   model_id = "Qwen/Qwen3-8B"
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

   features_by_t, labels_by_t, per_example_meta = build_probe_data(
       model=model,
       tokenizer=tokenizer,
       math_data=math_examples,
       max_items=1500,
   )
   ```

   `build_probe_data` uses the model-specific prompt templates in `probing.generation` and extracts hidden states with `probing.features`. Non-leaky prefixes and auxiliary metadata are returned for later analysis.

3. **Train probes and analyse baselines**

   ```python
   from probing.probes import run_all_probes
   from probing.analysis import analyze_baselines
   from pathlib import Path

   metrics = run_all_probes(features_by_t, labels_by_t, seed=356)

   analyze_baselines(
       per_example_meta=per_example_meta,
       probe_results=metrics,
       seed=356,
       fig_path=Path("figs/qwen/baselines.png"),
       results_path=Path("results/qwen/baselines.json"),
       plot_fn=lambda steps, curves, title, path, _: None,  # hook up plotting later
   )
   ```

   The analysis module offers convenience wrappers for difficulty splits and entropy/length baselines.

4. **Plot results**

   ```python
   from plotting import plot_probe_results, plot_baseline_comparison
   from pathlib import Path

   plot_probe_results(metrics, title="Qwen3-8B probe", outpath="figs/qwen_curve_overall.png")

   baseline_records = Path("results/qwen/baselines.json").read_text()
   # ... load JSON and pass to plot_baseline_comparison if you captured it earlier.
   ```

   Additional helpers (PCA variance, multiple-model curves) live in `plotting.metrics`.

## Repository Layout

```
src/
  probing/
    analysis.py          # Baseline comparisons and difficulty-slice summaries
    collection.py        # End-to-end generation + feature extraction loop
    config.py            # Shared constants and seeding helper
    data.py              # Hendrycks MATH parsing and normalisation
    features.py          # Prefix hidden-state harvesting and leakage checks
    generation.py        # Prompt builders and greedy CoT decoding
    labels.py            # Gold-vs-prediction correctness checks
    probes.py            # PCA + logistic regression probes
  plotting/
    metrics.py           # Matplotlib styling for probe diagnostics
scripts/
  prepare_math.py        # Optional JSONL dump of MATH subsets
  plot_*.py              # Standalone figure scripts for cached results
results/                 # Probe metrics and metadata (JSON, PNG)
runs/                    # Raw per-example generation logs (JSONL)
figs/                    # Published figures
requirements.txt         # Python dependencies
```

## Handy Commands

- Dump MATH splits to JSONL (offline reuse):
  ```bash
  PYTHONPATH=src python scripts/prepare_math.py --out-train data/math_train.jsonl
  ```
- Plot cached PCA ellipses:
  ```bash
  PYTHONPATH=src python scripts/plot_pca_prefix_ellipses.py \
    --input results/qwen/Qwen__Qwen3-8B_probe_details.json \
    --output figs/qwen/Qwen__Qwen3-8B_pca_prefix_ellipses.png
  ```
- Regenerate the difficulty histogram for Qwen:
  ```bash
  PYTHONPATH=src python scripts/plot_qwen_difficulty_hist.py \
    --input runs/Qwen__Qwen3-8B_examples.jsonl
  ```
- Run CPU smoke tests:
  ```bash
  PYTHONPATH=src python3 -m unittest discover -s tests
  ```

## Notes

- The packages rely on PyTorch, scikit-learn, Hugging Face Transformers, and `datasets`.
- GPU inference is assumed; adjust `probing.config.DEVICE` or pass `device_map` overrides when loading models.
- Prefix leakage checks flag prefixes that already contain the boxed final answer or close the `<think>` block early.
- Do not tune thresholds, steering coefficients, or prompt policies on the
  final test split. Use train for fitting and validation for policy selection.
