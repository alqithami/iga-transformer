#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_paper_data

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
QWEN_CFG="${QWEN_CFG:-configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml}"
SHUFFLE_SEED="${SHUFFLE_SEED:-17}"
SHUFFLE_ROOT="${SHUFFLE_ROOT:-results/qwen_choice_shuffle_eval_seed${SHUFFLE_SEED}}"
SEEDS="${SEEDS:-1 2 3}"

require_file "$QWEN_CFG"
mkdir -p "$SHUFFLE_ROOT/data"
for benchmark in truthfulqa fever halueval; do
  "$PYTHON_BIN" scripts/shuffle_choice_jsonl.py \
    --input "$DATA_ROOT/data/${benchmark}_eval.jsonl" \
    --output "$SHUFFLE_ROOT/data/${benchmark}_eval_shuffled.jsonl" \
    --seed "$SHUFFLE_SEED"
done

base_out="results/qwen2_5_7b_base_fixed_shuffle_eval"
mkdir -p "$base_out/predictions"
for benchmark in truthfulqa fever halueval; do
  "$PYTHON_BIN" scripts/evaluate_base_choice_fixed.py \
    --model_id "$MODEL_ID" \
    --data "$SHUFFLE_ROOT/data/${benchmark}_eval_shuffled.jsonl" \
    --out "$base_out/predictions/qwen2_5_7b_${benchmark}_vanilla_fixed_shuffle_mc.jsonl" \
    --seed 1 \
    --run_id qwen2_5_7b_base_fixed_shuffle_seed1 \
    --method vanilla_fixed_shuffle_mc \
    --dtype bfloat16
done
"$PYTHON_BIN" scripts/audit_prediction_nans.py "$base_out"/predictions/*.jsonl
"$PYTHON_BIN" -m iga_llm.report --out_dir "$base_out/aggregate" --predictions "$base_out"/predictions/*.jsonl

lora_out="results/qwen2_5_7b_lora_fixed_shuffle_eval"
for seed in $SEEDS; do
  adapter="results/qwen2_5_7b_lora_matched/seed${seed}/run/lora_adapter"
  require_dir "$adapter"
  seed_out="$lora_out/seed${seed}"
  run_id="qwen2_5_7b_lora_matched_seed${seed}_shuffle"
  mkdir -p "$seed_out/predictions"
  for benchmark in truthfulqa fever halueval; do
    "$PYTHON_BIN" scripts/evaluate_lora_choice_fixed.py \
      --model_id "$MODEL_ID" \
      --adapter_dir "$adapter" \
      --data "$SHUFFLE_ROOT/data/${benchmark}_eval_shuffled.jsonl" \
      --out "$seed_out/predictions/${run_id}_${benchmark}_lora_shuffle_mc.jsonl" \
      --seed "$seed" \
      --run_id "$run_id" \
      --method lora_shuffle_mc \
      --dtype bfloat16
  done
done
"$PYTHON_BIN" scripts/audit_prediction_nans.py "$lora_out"/seed*/predictions/*.jsonl
"$PYTHON_BIN" -m iga_llm.report --out_dir "$lora_out/aggregate" --predictions "$lora_out"/seed*/predictions/*.jsonl

iga_out="results/qwen2_5_7b_iga_shuffle_eval"
for seed in $SEEDS; do
  checkpoint="$($PYTHON_BIN - "$seed" <<'PY'
from pathlib import Path
import sys
seed = int(sys.argv[1])
for root in sorted(Path("results").glob(f"qwen2_5_7b_lowrank_g2_matched_seed{seed}_*"), reverse=True):
    candidates = list((root / "runs").glob("*/iga_modules.pt"))
    if candidates:
        print(candidates[0])
        break
else:
    raise SystemExit(f"No dense IGA checkpoint found for seed {seed}")
PY
)"
  seed_out="$iga_out/seed${seed}"
  run_id="qwen2_5_7b_iga_shuffle_seed${seed}"
  mkdir -p "$seed_out/predictions"
  for benchmark in truthfulqa fever halueval; do
    "$PYTHON_BIN" -m iga_llm.evaluate \
      --config "$QWEN_CFG" \
      --data "$SHUFFLE_ROOT/data/${benchmark}_eval_shuffled.jsonl" \
      --out "$seed_out/predictions/${run_id}_${benchmark}_iga_shuffle_mc.jsonl" \
      --method iga_mc \
      --seed "$seed" \
      --run_id "$run_id" \
      --iga_checkpoint "$checkpoint" \
      --max_new_tokens 32 \
      --num_samples 5 \
      --temperature 0.7 \
      --top_p 0.95 \
      --report_method iga_shuffle_mc
  done
done
"$PYTHON_BIN" scripts/audit_prediction_nans.py "$iga_out"/seed*/predictions/*.jsonl
"$PYTHON_BIN" -m iga_llm.report --out_dir "$iga_out/aggregate" --predictions "$iga_out"/seed*/predictions/*.jsonl

echo "Choice-shuffle audit complete."
