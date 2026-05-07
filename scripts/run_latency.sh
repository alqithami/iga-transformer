#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.mac_m4 2>/dev/null || true
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${CONFIG:-configs/mistral_7b_iga.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
OUT_DIR="${OUT_DIR:-results/manual_latency}"
mkdir -p "$OUT_DIR"
for TOKS in ${PROMPT_TOKENS:-256 512 1024}; do
  python -m iga_llm.benchmark_latency --config "$CONFIG" --method vanilla --prompt_tokens "$TOKS" --out "$OUT_DIR/vanilla_${TOKS}.jsonl"
  if [[ -n "$CHECKPOINT" ]]; then
    python -m iga_llm.benchmark_latency --config "$CONFIG" --method iga --iga_checkpoint "$CHECKPOINT" --prompt_tokens "$TOKS" --out "$OUT_DIR/iga_${TOKS}.jsonl"
  fi
done
python -m iga_llm.report --out_dir "$OUT_DIR/aggregate" --latency "$OUT_DIR"/*.jsonl
