#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_paper_data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN_CFG="${QWEN_CFG:-configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml}"
SEED="${1:-1}"
QWEN_OUT="${QWEN_OUT:-results/qwen2_5_7b_lowrank_g2_matched_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="qwen2_5_7b_iga_lowrank_g2_tau035_matched_seed${SEED}"

require_file "$QWEN_CFG"
mkdir -p "$QWEN_OUT/runs/$RUN_NAME" "$QWEN_OUT/predictions" "$QWEN_OUT/aggregate" logs
export QWEN_OUT RUN_NAME

"$PYTHON_BIN" -m iga_llm.train \
  --config "$QWEN_CFG" \
  --train_jsonl "$DATA_ROOT/data/calibration_train_mix.jsonl" \
  --dev_jsonl "$DATA_ROOT/data/calibration_dev_mix.jsonl" \
  --output_dir "$QWEN_OUT/runs/$RUN_NAME" \
  --seed "$SEED" \
  --epochs 1 \
  2>&1 | tee "logs/qwen_seed${SEED}_train_$(date +%Y%m%d_%H%M%S).log"

"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["QWEN_OUT"]) / "runs" / os.environ["RUN_NAME"] / "train_summary.json"
summary = json.loads(path.read_text())
keys = ["trainable_parameters", "total_parameters", "trainable_fraction", "steps", "train_examples", "dev_examples", "risk_labels_computed", "risk_loss_weight"]
print(json.dumps({key: summary.get(key) for key in keys}, indent=2))
assert summary.get("risk_labels_computed", 0) in (0, None)
assert summary.get("trainable_parameters", 10**12) < 2_000_000
assert summary.get("steps", 0) > 0
PY

checkpoint="$QWEN_OUT/runs/$RUN_NAME/iga_modules.pt"
require_file "$checkpoint"

for benchmark in truthfulqa fever halueval; do
  "$PYTHON_BIN" -m iga_llm.evaluate \
    --config "$QWEN_CFG" \
    --data "$DATA_ROOT/data/${benchmark}_eval.jsonl" \
    --out "$QWEN_OUT/predictions/${RUN_NAME}_${benchmark}_eval_iga_mc.jsonl" \
    --method iga_mc \
    --seed "$SEED" \
    --run_id "$RUN_NAME" \
    --iga_checkpoint "$checkpoint" \
    --max_new_tokens 32 \
    --num_samples 5 \
    --temperature 0.7 \
    --top_p 0.95 \
    --report_method iga_v2_lowrank_mc
done

for method in vanilla_mc temperature_mc semantic_entropy_mc self_consistency_mc; do
  for benchmark in truthfulqa fever halueval; do
    "$PYTHON_BIN" -m iga_llm.evaluate \
      --config "$QWEN_CFG" \
      --data "$DATA_ROOT/data/${benchmark}_eval.jsonl" \
      --out "$QWEN_OUT/predictions/${RUN_NAME}_${benchmark}_${method}.jsonl" \
      --method "$method" \
      --seed "$SEED" \
      --run_id "$RUN_NAME" \
      --max_new_tokens 32 \
      --num_samples 5 \
      --temperature 0.7 \
      --top_p 0.95
  done
done

"$PYTHON_BIN" scripts/audit_prediction_nans.py "$QWEN_OUT"/predictions/*.jsonl
"$PYTHON_BIN" -m iga_llm.report \
  --out_dir "$QWEN_OUT/aggregate" \
  --predictions "$QWEN_OUT"/predictions/*.jsonl

echo "Dense Qwen run complete: $QWEN_OUT"
