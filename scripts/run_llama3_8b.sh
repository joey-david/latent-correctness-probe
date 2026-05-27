#!/usr/bin/env bash

set -euo pipefail

# Run the config-driven pipeline with Meta-Llama 3.1 8B after adapting a YAML
# config's model.id. Keep full jobs on lamgate per plan.md.
PYTHONPATH=src python3 main.py generation \
  --config "${1:-configs/qwen_math_smoke.yaml}" \
  --split validation
