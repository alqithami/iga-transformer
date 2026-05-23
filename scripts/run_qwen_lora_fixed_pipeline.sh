#!/usr/bin/env bash
set -euo pipefail
cd ~/IGA_GPU
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
export QWEN_CFG="${QWEN_CFG:-configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p logs
for SEED in 1 2 3; do
  RUN_ROOT="results/qwen2_5_7b_lora_matched/seed${SEED}"
  RUN_NAME="qwen2_5_7b_lora_matched_seed${SEED}"
  mkdir -p "$RUN_ROOT/run" "$RUN_ROOT/predictions" "$RUN_ROOT/aggregate"
  python scripts/run_lora_choice.py train \
    --config "$QWEN_CFG" \
    --train_jsonl "$BASE_OUT/data/calibration_train_mix.jsonl" \
    --dev_jsonl "$BASE_OUT/data/calibration_dev_mix.jsonl" \
    --output_dir "$RUN_ROOT/run" \
    --seed "$SEED" \
    --epochs 1 \
    --lr 0.0001 \
    --max_length 768 \
    --lora_r 2 \
    --lora_alpha 4 \
    --target_modules q_proj,v_proj,o_proj \
    --max_trainable 2000000 \
    2>&1 | tee "logs/qwen_lora_seed${SEED}_train_$(date +%Y%m%d_%H%M%S).log"
  FIXED_ROOT="results/qwen2_5_7b_lora_fixed_eval/seed${SEED}"
  mkdir -p "$FIXED_ROOT/predictions"
  for BENCH in truthfulqa fever halueval; do
    python scripts/evaluate_lora_choice_fixed.py \
      --model_id "$MODEL_ID" \
      --adapter_dir "$RUN_ROOT/run/lora_adapter" \
      --data "$BASE_OUT/data/${BENCH}_eval.jsonl" \
      --out "$FIXED_ROOT/predictions/${RUN_NAME}_${BENCH}_lora_mc.jsonl" \
      --seed "$SEED" \
      --run_id "$RUN_NAME" \
      --method lora_mc \
      --dtype bfloat16
  done
  python scripts/audit_prediction_nans.py "$FIXED_ROOT"/predictions/*.jsonl
  python -m iga_llm.report --out_dir "$FIXED_ROOT/aggregate" --predictions "$FIXED_ROOT"/predictions/*.jsonl
done
python - <<'PY'
from pathlib import Path
import subprocess, sys
out = Path('results/qwen2_5_7b_lora_fixed_compare')
agg = out / 'aggregate'
agg.mkdir(parents=True, exist_ok=True)
files = []
for root in sorted(Path('results').glob('qwen2_5_7b_lowrank_g2_matched_seed*')):
    files.extend(sorted((root / 'predictions').glob('*.jsonl')))
files.extend(sorted(Path('results/qwen2_5_7b_lora_fixed_eval').glob('seed*/predictions/*.jsonl')))
seen=set(); files=[f for f in files if not (str(f) in seen or seen.add(str(f)))]
subprocess.run([sys.executable, '-m', 'iga_llm.report', '--out_dir', str(agg), '--predictions', *map(str, files)], check=True)
print('Wrote', agg)
PY
