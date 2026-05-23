#!/usr/bin/env bash
set -euo pipefail
cd ~/IGA_GPU
source .venv/bin/activate
export BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export QWEN_CFG="${QWEN_CFG:-configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml}"
export SHUF_OUT="results/qwen_choice_shuffle_eval_seed17"
mkdir -p "$SHUF_OUT/data"
for BENCH in truthfulqa fever halueval; do
  python scripts/shuffle_choice_jsonl.py --input "$BASE_OUT/data/${BENCH}_eval.jsonl" --output "$SHUF_OUT/data/${BENCH}_eval_shuffled.jsonl" --seed 17
done
BASE_FIXED_SHUF="results/qwen2_5_7b_base_fixed_shuffle_eval"
mkdir -p "$BASE_FIXED_SHUF/predictions"
for BENCH in truthfulqa fever halueval; do
  python scripts/evaluate_base_choice_fixed.py \
    --model_id "$MODEL_ID" \
    --data "$SHUF_OUT/data/${BENCH}_eval_shuffled.jsonl" \
    --out "$BASE_FIXED_SHUF/predictions/qwen2_5_7b_${BENCH}_vanilla_fixed_shuffle_mc.jsonl" \
    --seed 1 \
    --run_id qwen2_5_7b_base_fixed_shuffle_seed1 \
    --method vanilla_fixed_shuffle_mc \
    --dtype bfloat16
done
python scripts/audit_prediction_nans.py "$BASE_FIXED_SHUF"/predictions/*.jsonl
python -m iga_llm.report --out_dir "$BASE_FIXED_SHUF/aggregate" --predictions "$BASE_FIXED_SHUF"/predictions/*.jsonl
LORA_SHUF="results/qwen2_5_7b_lora_fixed_shuffle_eval"
for SEED in 1 2 3; do
  ADAPTER="results/qwen2_5_7b_lora_matched/seed${SEED}/run/lora_adapter"
  OUT_DIR="$LORA_SHUF/seed${SEED}"
  RUN_ID="qwen2_5_7b_lora_matched_seed${SEED}_shuffle"
  mkdir -p "$OUT_DIR/predictions"
  for BENCH in truthfulqa fever halueval; do
    python scripts/evaluate_lora_choice_fixed.py \
      --model_id "$MODEL_ID" \
      --adapter_dir "$ADAPTER" \
      --data "$SHUF_OUT/data/${BENCH}_eval_shuffled.jsonl" \
      --out "$OUT_DIR/predictions/${RUN_ID}_${BENCH}_lora_shuffle_mc.jsonl" \
      --seed "$SEED" \
      --run_id "$RUN_ID" \
      --method lora_shuffle_mc \
      --dtype bfloat16
  done
done
python scripts/audit_prediction_nans.py "$LORA_SHUF"/seed*/predictions/*.jsonl
python -m iga_llm.report --out_dir "$LORA_SHUF/aggregate" --predictions "$LORA_SHUF"/seed*/predictions/*.jsonl
IGA_SHUF="results/qwen2_5_7b_iga_shuffle_eval"
for SEED in 1 2 3; do
  ROOT="$(ls -td results/qwen2_5_7b_lowrank_g2_matched_seed${SEED}_* | head -1)"
  RUN_DIR="$(find "$ROOT/runs" -maxdepth 1 -mindepth 1 -type d | head -1)"
  RUN_NAME="$(basename "$RUN_DIR")"
  CKPT="$RUN_DIR/iga_modules.pt"
  OUT_DIR="$IGA_SHUF/seed${SEED}"
  mkdir -p "$OUT_DIR/predictions"
  for BENCH in truthfulqa fever halueval; do
    python -m iga_llm.evaluate \
      --config "$QWEN_CFG" \
      --data "$SHUF_OUT/data/${BENCH}_eval_shuffled.jsonl" \
      --out "$OUT_DIR/predictions/${RUN_NAME}_${BENCH}_iga_shuffle_mc.jsonl" \
      --method iga_mc \
      --seed "$SEED" \
      --run_id "${RUN_NAME}_shuffle" \
      --iga_checkpoint "$CKPT" \
      --max_new_tokens 32 \
      --num_samples 5 \
      --temperature 0.7 \
      --top_p 0.95 \
      --report_method "iga_shuffle_mc"
  done
done
python scripts/audit_prediction_nans.py "$IGA_SHUF"/seed*/predictions/*.jsonl
python -m iga_llm.report --out_dir "$IGA_SHUF/aggregate" --predictions "$IGA_SHUF"/seed*/predictions/*.jsonl
