#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_paper_data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN_CFG="${QWEN_CFG:-configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml}"
OUT="${OUT:-results/qwen2_5_7b_selfconsistency20}"
SEEDS="${SEEDS:-1 2 3}"
BENCHES="${BENCHES:-truthfulqa fever halueval}"

require_file "$QWEN_CFG"
mkdir -p "$OUT/predictions" "$OUT/aggregate" logs

row_count() {
  local path="$1"
  if [[ -f "$path" ]]; then wc -l < "$path"; else printf '0\n'; fi
}

for seed in $SEEDS; do
  for benchmark in $BENCHES; do
    output="$OUT/predictions/qwen2_5_7b_seed${seed}_${benchmark}_self_consistency20_mc.jsonl"
    expected="$(expected_rows "$benchmark")"
    current="$(row_count "$output")"

    if [[ "$current" -eq "$expected" ]]; then
      echo "SKIP seed=$seed benchmark=$benchmark rows=$current/$expected"
      continue
    fi

    echo "RUN seed=$seed benchmark=$benchmark rows=$current/$expected"
    rm -f "$output"
    "$PYTHON_BIN" -m iga_llm.evaluate \
      --config "$QWEN_CFG" \
      --data "$DATA_ROOT/data/${benchmark}_eval.jsonl" \
      --out "$output" \
      --method self_consistency_mc \
      --report_method self_consistency20_mc \
      --seed "$seed" \
      --run_id "qwen2_5_7b_self_consistency20_seed${seed}" \
      --max_new_tokens 32 \
      --num_samples 20 \
      --temperature 0.7 \
      --top_p 0.95 \
      2>&1 | tee "logs/qwen_sc20_seed${seed}_${benchmark}_$(date +%Y%m%d_%H%M%S).log"
  done
done

"$PYTHON_BIN" scripts/audit_prediction_nans.py "$OUT"/predictions/*.jsonl
"$PYTHON_BIN" -m iga_llm.report \
  --out_dir "$OUT/aggregate" \
  --predictions "$OUT"/predictions/*.jsonl

echo "SC20 evaluation complete: $OUT"
