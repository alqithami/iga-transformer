#!/usr/bin/env bash
set -euo pipefail
cd "${IGA_ROOT:-$HOME/IGA_GPU}"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED="${1:?Usage: bash scripts/run_mistral_lora_plus_iga_one_seed.sh <seed>}"
BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
MISTRAL_CFG="${MISTRAL_CFG:-configs/mistral_7b_iga_lowrank_g2_tau035_bf16.yaml}"
ADAPTER="results/mistral_7b_lora_matched/seed${SEED}/run/lora_adapter"
OUT="results/mistral_7b_lora_plus_iga_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
RUN_NAME="mistral_7b_lora_plus_iga_seed${SEED}"
mkdir -p "$OUT/run" "$OUT/predictions" "$OUT/aggregate" logs
python -m iga_llm.train --config "$MISTRAL_CFG" --train_jsonl "$BASE_OUT/data/calibration_train_mix.jsonl" --dev_jsonl "$BASE_OUT/data/calibration_dev_mix.jsonl" --output_dir "$OUT/run" --seed "$SEED" --epochs 1 --lora_adapter_dir "$ADAPTER" | tee "logs/mistral_lora_plus_iga_seed${SEED}_train_$(date +%Y%m%d_%H%M%S).log"
CKPT="$OUT/run/iga_modules.pt"
for BENCH in truthfulqa fever halueval; do
  python -m iga_llm.evaluate --config "$MISTRAL_CFG" --data "$BASE_OUT/data/${BENCH}_eval.jsonl" --out "$OUT/predictions/${RUN_NAME}_${BENCH}_lora_iga_mc.jsonl" --method iga_mc --report_method lora_iga_mc --seed "$SEED" --run_id "$RUN_NAME" --iga_checkpoint "$CKPT" --lora_adapter_dir "$ADAPTER" --max_new_tokens 32 --num_samples 5 --temperature 0.7 --top_p 0.95
done
python scripts/audit_prediction_nans.py "$OUT"/predictions/*.jsonl
python -m iga_llm.report --out_dir "$OUT/aggregate" --predictions "$OUT"/predictions/*.jsonl
