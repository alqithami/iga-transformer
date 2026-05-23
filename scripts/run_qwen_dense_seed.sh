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
SEED="${1:-1}"
export QWEN_OUT="results/qwen2_5_7b_lowrank_g2_matched_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
export RUN_NAME="qwen2_5_7b_iga_lowrank_g2_tau035_matched_seed${SEED}"
mkdir -p "$QWEN_OUT/runs/$RUN_NAME" "$QWEN_OUT/predictions" "$QWEN_OUT/aggregate" logs
python -m iga_llm.train \
  --config "$QWEN_CFG" \
  --train_jsonl "$BASE_OUT/data/calibration_train_mix.jsonl" \
  --dev_jsonl "$BASE_OUT/data/calibration_dev_mix.jsonl" \
  --output_dir "$QWEN_OUT/runs/$RUN_NAME" \
  --seed "$SEED" \
  --epochs 1 \
  2>&1 | tee "logs/qwen_seed${SEED}_train_$(date +%Y%m%d_%H%M%S).log"
python - <<'PY'
import json, os
p = os.environ['QWEN_OUT'] + '/runs/' + os.environ['RUN_NAME'] + '/train_summary.json'
s = json.load(open(p))
print(json.dumps({k:s.get(k) for k in ['trainable_parameters','total_parameters','trainable_fraction','risk_labels_computed','risk_loss_weight']}, indent=2))
assert s.get('risk_labels_computed', 0) in (0, None)
assert s.get('trainable_parameters', 10**12) < 2_000_000
PY
export CKPT="$QWEN_OUT/runs/$RUN_NAME/iga_modules.pt"
for BENCH in truthfulqa fever halueval; do
  python -m iga_llm.evaluate \
    --config "$QWEN_CFG" \
    --data "$BASE_OUT/data/${BENCH}_eval.jsonl" \
    --out "$QWEN_OUT/predictions/${RUN_NAME}_${BENCH}_eval_iga_mc.jsonl" \
    --method iga_mc \
    --seed "$SEED" \
    --run_id "$RUN_NAME" \
    --iga_checkpoint "$CKPT" \
    --max_new_tokens 32 \
    --num_samples 5 \
    --temperature 0.7 \
    --top_p 0.95 \
    --report_method "iga_v2_lowrank_mc"
done
for METHOD in vanilla_mc temperature_mc semantic_entropy_mc self_consistency_mc; do
  for BENCH in truthfulqa fever halueval; do
    python -m iga_llm.evaluate \
      --config "$QWEN_CFG" \
      --data "$BASE_OUT/data/${BENCH}_eval.jsonl" \
      --out "$QWEN_OUT/predictions/${RUN_NAME}_${BENCH}_${METHOD}.jsonl" \
      --method "$METHOD" \
      --seed "$SEED" \
      --run_id "$RUN_NAME" \
      --max_new_tokens 32 \
      --num_samples 5 \
      --temperature 0.7 \
      --top_p 0.95
  done
done
python scripts/audit_prediction_nans.py "$QWEN_OUT"/predictions/*.jsonl
python -m iga_llm.report --out_dir "$QWEN_OUT/aggregate" --predictions "$QWEN_OUT"/predictions/*.jsonl
echo "DONE: $QWEN_OUT"
