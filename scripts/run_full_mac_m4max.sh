#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.mac_m4 2>/dev/null || true
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIGS="${CONFIGS:-configs/mistral_7b_iga.yaml}"
SEEDS="${SEEDS:-1 2 3}"
LIMIT_TRAIN="${LIMIT_TRAIN:-1000}"
LIMIT_DEV="${LIMIT_DEV:-300}"
LIMIT_EVAL="${LIMIT_EVAL:-300}"
EPOCHS="${EPOCHS:-1}"
OUT_DIR="${OUT_DIR:-}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_LATENCY="${SKIP_LATENCY:-0}"
EXTRA_ARGS=()
if [[ -n "$OUT_DIR" ]]; then EXTRA_ARGS+=(--out_dir "$OUT_DIR"); fi
if [[ "$RUN_ABLATIONS" == "1" ]]; then EXTRA_ARGS+=(--run_ablations); fi
if [[ "$SKIP_GENERATION" == "1" ]]; then EXTRA_ARGS+=(--skip_generation); fi
if [[ "$SKIP_LATENCY" == "1" ]]; then EXTRA_ARGS+=(--skip_latency); fi
python -m iga_llm.run_matrix \
  --configs $CONFIGS \
  --seeds $SEEDS \
  --limit_train "$LIMIT_TRAIN" \
  --limit_dev "$LIMIT_DEV" \
  --limit_eval "$LIMIT_EVAL" \
  --epochs "$EPOCHS" \
  "${EXTRA_ARGS[@]}"
