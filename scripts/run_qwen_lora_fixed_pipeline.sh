#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_paper_data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN_CFG="${QWEN_CFG:-configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
SEEDS="${SEEDS:-1 2 3}"
require_file "$QWEN_CFG"
mkdir -p logs

for seed in $SEEDS; do
  run_root="results/qwen2_5_7b_lora_matched/seed${seed}"
  run_name="qwen2_5_7b_lora_matched_seed${seed}"
  mkdir -p "$run_root/run" "$run_root/predictions" "$run_root/aggregate"

  "$PYTHON_BIN" scripts/run_lora_choice.py train \
    --config "$QWEN_CFG" \
    --train_jsonl "$DATA_ROOT/data/calibration_train_mix.jsonl" \
    --dev_jsonl "$DATA_ROOT/data/calibration_dev_mix.jsonl" \
    --output_dir "$run_root/run" \
    --seed "$seed" \
    --epochs 1 \
    --lr 0.0001 \
    --max_length 768 \
    --lora_r 2 \
    --lora_alpha 4 \
    --target_modules q_proj,v_proj,o_proj \
    --max_trainable 2000000 \
    2>&1 | tee "logs/qwen_lora_seed${seed}_train_$(date +%Y%m%d_%H%M%S).log"

  require_dir "$run_root/run/lora_adapter"
  fixed_root="results/qwen2_5_7b_lora_fixed_eval/seed${seed}"
  mkdir -p "$fixed_root/predictions"
  for benchmark in truthfulqa fever halueval; do
    "$PYTHON_BIN" scripts/evaluate_lora_choice_fixed.py \
      --model_id "$MODEL_ID" \
      --adapter_dir "$run_root/run/lora_adapter" \
      --data "$DATA_ROOT/data/${benchmark}_eval.jsonl" \
      --out "$fixed_root/predictions/${run_name}_${benchmark}_lora_mc.jsonl" \
      --seed "$seed" \
      --run_id "$run_name" \
      --method lora_mc \
      --dtype bfloat16
  done
  "$PYTHON_BIN" scripts/audit_prediction_nans.py "$fixed_root"/predictions/*.jsonl
  "$PYTHON_BIN" -m iga_llm.report --out_dir "$fixed_root/aggregate" --predictions "$fixed_root"/predictions/*.jsonl
done

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import subprocess, sys
files = sorted(Path("results/qwen2_5_7b_lora_fixed_eval").glob("seed*/predictions/*.jsonl"))
if len(files) != 9:
    raise SystemExit(f"Expected 9 LoRA evaluation files, found {len(files)}")
out = Path("results/qwen2_5_7b_lora_fixed_compare/aggregate")
out.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "iga_llm.report", "--out_dir", str(out), "--predictions", *map(str, files)], check=True)
print(f"Wrote {out}")
PY
