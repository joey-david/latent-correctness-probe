#!/usr/bin/env bash

set -euo pipefail

# Run the probe pipeline with Meta-Llama 3.1 8B and capture enhanced metadata.
python main.py \
  --model-id meta-llama/Llama-3.1-8B \
  --total-examples 1500
