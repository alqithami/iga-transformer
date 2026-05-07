#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.mac_m4 2>/dev/null || true
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${CONFIG:-configs/mistral_7b_iga.yaml}"
DATA="${DATA:-results/latest/data/truthfulqa_eval.jsonl}"
OUT_DIR="${OUT_DIR:-results/manual_generation}"
SEED="${SEED:-1}"
mkdir -p "$OUT_DIR"
for METHOD in vanilla_gen contrastive_gen dola_gen self_refine_gen cove_gen; do
  python -m iga_llm.evaluate --config "$CONFIG" --data "$DATA" --out "$OUT_DIR/${METHOD}.jsonl" --method "$METHOD" --seed "$SEED"
done
python -m iga_llm.report --out_dir "$OUT_DIR/aggregate" --predictions "$OUT_DIR"/*.jsonl
