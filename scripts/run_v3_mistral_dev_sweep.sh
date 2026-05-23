#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env.mac_m4
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
OUT_ROOT="${OUT_ROOT:-results/iga_v3_mistral_dev_sweep_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-1}"
CONFIGS="${CONFIGS:-configs/mistral_7b_iga_v3_entropy_sparse.yaml configs/mistral_7b_iga_v3_hybrid_sparse.yaml configs/mistral_7b_iga_v3_risk_sparse.yaml}"
BENCHES="${BENCHES:-truthfulqa fever halueval}"

mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/predictions" "$OUT_ROOT/aggregate"
echo "BASE_OUT=$BASE_OUT"
echo "OUT_ROOT=$OUT_ROOT"
echo "CONFIGS=$CONFIGS"

for CFG in $CONFIGS; do
  NAME=$(basename "$CFG" .yaml)
  RUN_DIR="$OUT_ROOT/runs/${NAME}_seed${SEED}"
  CKPT="$RUN_DIR/iga_modules.pt"
  if [ ! -f "$CKPT" ]; then
    caffeinate -dimsu python -m iga_llm.train \
      --config "$CFG" \
      --train_jsonl "$BASE_OUT/data/calibration_train_mix.jsonl" \
      --dev_jsonl "$BASE_OUT/data/calibration_dev_mix.jsonl" \
      --output_dir "$RUN_DIR" \
      --seed "$SEED" \
      --epochs 1
  else
    echo "Skipping existing checkpoint $CKPT"
  fi
  for BENCH in $BENCHES; do
    DATA="$BASE_OUT/data/${BENCH}_dev.jsonl"
    OUT_FILE="$OUT_ROOT/predictions/${NAME}_seed${SEED}_${BENCH}_dev_iga_mc.jsonl"
    EXPECTED=$(wc -l < "$DATA" | tr -d ' ')
    if [ -f "$OUT_FILE" ] && [ "$(wc -l < "$OUT_FILE" | tr -d ' ')" = "$EXPECTED" ]; then
      echo "Skipping complete $OUT_FILE"
      continue
    fi
    caffeinate -dimsu python -m iga_llm.evaluate \
      --config "$CFG" \
      --data "$DATA" \
      --out "$OUT_FILE" \
      --method iga_mc \
      --seed "$SEED" \
      --run_id "${NAME}_seed${SEED}" \
      --iga_checkpoint "$CKPT" \
      --max_new_tokens 32 \
      --num_samples 5 \
      --temperature 0.7 \
      --top_p 0.95 \
      --report_method "$NAME"
  done
done

python -m iga_llm.report --out_dir "$OUT_ROOT/aggregate" --predictions "$OUT_ROOT"/predictions/*.jsonl

echo "Done. Inspect:"
echo "  column -s, -t < $OUT_ROOT/aggregate/summary_by_model_benchmark_method.csv | less -S"
