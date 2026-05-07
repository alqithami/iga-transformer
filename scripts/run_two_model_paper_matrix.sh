#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.mac_m4 2>/dev/null || true
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m iga_llm.run_matrix \
  --configs configs/llama3_8b_iga.yaml configs/mistral_7b_iga.yaml \
  --seeds 1 2 3 \
  --limit_train "${LIMIT_TRAIN:-1000}" \
  --limit_dev "${LIMIT_DEV:-300}" \
  --limit_eval "${LIMIT_EVAL:-300}" \
  --epochs "${EPOCHS:-1}" \
  --run_ablations
