#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_paper_data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

seed="${1:?Usage: bash scripts/run_mistral_lora_plus_iga_one_seed.sh <seed>}"
MISTRAL_CFG="${MISTRAL_CFG:-configs/mistral_7b_iga_lowrank_g2_tau035_bf16.yaml}"
adapter="results/mistral_7b_lora_matched/seed${seed}/run/lora_adapter"
out="results/mistral_7b_lora_plus_iga_seed${seed}_$(date +%Y%m%d_%H%M%S)"
run_name="mistral_7b_lora_plus_iga_seed${seed}"

require_file "$MISTRAL_CFG"
require_dir "$adapter"
mkdir -p "$out/run" "$out/predictions" "$out/aggregate" logs

"$PYTHON_BIN" -m iga_llm.train \
  --config "$MISTRAL_CFG" \
  --train_jsonl "$DATA_ROOT/data/calibration_train_mix.jsonl" \
  --dev_jsonl "$DATA_ROOT/data/calibration_dev_mix.jsonl" \
  --output_dir "$out/run" \
  --seed "$seed" \
  --epochs 1 \
  --lora_adapter_dir "$adapter" \
  2>&1 | tee "logs/mistral_lora_plus_iga_seed${seed}_train_$(date +%Y%m%d_%H%M%S).log"

"$PYTHON_BIN" - "$out/run/train_summary.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
summary = json.loads(path.read_text())
keys = ["seed", "steps", "train_examples", "dev_examples", "trainable_parameters", "lora_merged_for_iga", "phi_source", "risk_labels_computed", "risk_loss_weight"]
print(json.dumps({key: summary.get(key) for key in keys}, indent=2))
assert summary.get("lora_merged_for_iga") is True
assert summary.get("phi_source") == "lora_adapted_backbone"
assert summary.get("risk_labels_computed", 0) in (0, None)
assert summary.get("trainable_parameters", 10**12) < 2_000_000
assert summary.get("steps", 0) > 0
PY

checkpoint="$out/run/iga_modules.pt"
require_file "$checkpoint"
for benchmark in truthfulqa fever halueval; do
  "$PYTHON_BIN" -m iga_llm.evaluate \
    --config "$MISTRAL_CFG" \
    --data "$DATA_ROOT/data/${benchmark}_eval.jsonl" \
    --out "$out/predictions/${run_name}_${benchmark}_lora_iga_mc.jsonl" \
    --method iga_mc \
    --report_method lora_iga_mc \
    --seed "$seed" \
    --run_id "$run_name" \
    --iga_checkpoint "$checkpoint" \
    --lora_adapter_dir "$adapter" \
    --max_new_tokens 32 \
    --num_samples 5 \
    --temperature 0.7 \
    --top_p 0.95
done

"$PYTHON_BIN" scripts/audit_prediction_nans.py "$out"/predictions/*.jsonl
"$PYTHON_BIN" -m iga_llm.report --out_dir "$out/aggregate" --predictions "$out"/predictions/*.jsonl

echo "Mistral composition seed $seed complete: $out"
