# Latent Correctness Probe

This repo measures whether a reasoning language model already "knows" if it will solve a *hard* math problem correctly before finishing its chain-of-thought. For two instruction-tuned models (one Qwen-like, one Mistral-like) we:

- Decode greedy chain-of-thought answers on **Hendrycks MATH (NeurIPS 2021)** problems.
- Snapshot the model's last hidden state after the first `t` reasoning tokens (`t ∈ {1,2,4,8,16,32,64,128}`).
- Train a linear probe (PCA → Logistic Regression) to predict whether the eventual final answer is correct.
- Plot probe accuracy and ROC-AUC versus reasoning prefix length.

If probe performance rises with `t`, it suggests the internal representation already encodes correctness before the model finishes reasoning.

## Assumptions & Limitations

- Greedy decoding only; no sampling or self-consistency.
- Gold answers are pulled from the final `\boxed{…}` in the official MATH solutions. By default we keep only problems whose boxed answer normalises to a numeric string (override if you want symbolic targets).
- Predictions are parsed by first looking for a final `\boxed{…}` in the model output, falling back to the last numeric literal in the text.
- Only prefixes up to 192 tokens are considered to avoid very long forwards.
- Prefixes that already contain the final numeric answer are skipped to prevent label leakage.

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

### 1. Prepare Hendrycks MATH data

The default pipeline loads MATH directly from Hugging Face using `datasets.load_dataset`. If you prefer to cache JSONL files (e.g., for offline runs) use:

```bash
python scripts/prepare_math.py --min-level 1 --out-train data/math_train.jsonl --out-test data/math_test.jsonl
```

This utility concatenates all seven subjects (`algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`) and filters to examples whose answers can be normalised. Adjust `--subjects`, `--min-level`, `--keep-non-numeric`, or `--max-items` for custom subsets.

`main.py` consumes the dataset through `load_math_split(...)`, so the JSONLs are optional.

### 2. Configure model checkpoints

`main.py` defaults to:

- Qwen: `Qwen/Qwen3-8B`
- Mistral: `mistralai/Mistral-7B-Instruct-v0.3`

Edit `qwen_id` / `other_id` in `main.py` to change them. Models are loaded with `torch_dtype=torch.float16` and `device_map="auto"`, so you need a GPU configuration that Hugging Face Accelerate can target automatically.

### 3. Expected outputs

Running `python main.py` will:

1. Load MATH training data (respecting the `min_level` and numeric filter in `main.py`).
2. Decode CoT answers for each example with both models.
3. Collect hidden states at every checkpoint (skipping leaky prefixes).
4. Train logistic probes and evaluate with a train/test split.
5. Print metrics dictionary per prefix length, e.g.:

   ```json
   {
     "1": {"acc": 0.55, "auc": 0.57, "...": "..."},
     "2": {...},
     "4": {...}
   }
   ```

6. Save plots:
   - `qwen_curve.png`
   - `other_curve.png`

Each plot shows ROC-AUC and accuracy versus prefix length.

## File Overview

- `core_deps.py`: shared imports, constants, random seeds.
- `data.py`: Hendrycks MATH helpers (boxed answer parsing, dataset loading, numeric normalisation).
- `gen.py`: prompt builder and greedy chain-of-thought generation.
- `labels.py`: binary correctness label computation.
- `features.py`: prefix hidden-state extraction with leakage checks.
- `collect_dataset.py`: loop over data to build probe-ready datasets.
- `probes.py`: PCA + logistic regression probes and evaluation.
- `plotting.py`: matplotlib plot helper for probe curves.
- `scripts/prepare_math.py`: optional utility to dump MATH JSONL files.
- `main.py`: end-to-end script (load data/models, build dataset, train probes).
- `requirements.txt`: Python dependencies.
