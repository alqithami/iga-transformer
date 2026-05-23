#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env.mac_m4
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
CFG="${CFG:-configs/mistral_7b_iga_v3_risk_sparse.yaml}"
SEED="${SEED:-1}"
OUT_ROOT="${OUT_ROOT:-results/iga_v3_selected_eval_$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-$(basename "$CFG" .yaml)}"
BENCHES="${BENCHES:-truthfulqa fever halueval}"
REPORT_METHOD="${REPORT_METHOD:-iga_v3_risk_sparse_mc}"
INCLUDE_MISTRAL_BASELINES="${INCLUDE_MISTRAL_BASELINES:-1}"

mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/predictions" "$OUT_ROOT/aggregate"
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
  DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
  OUT_FILE="$OUT_ROOT/predictions/${NAME}_seed${SEED}_${BENCH}_eval_iga_mc.jsonl"
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
    --report_method "$REPORT_METHOD"
done

# Aggregate with original Mistral matched baselines when present.
python - <<'PY'
from pathlib import Path
import subprocess, sys, os
base = Path(os.environ.get("BASE_OUT", "results/mistral_full_generation_20260507_234820"))
out = Path(os.environ["OUT_ROOT"])
include_mistral_baselines = os.environ.get("INCLUDE_MISTRAL_BASELINES", "1") == "1"
clean = Path("results/mistral_matched_baseline_seeds_2_3_CLEAN")
files = []
if include_mistral_baselines:
    if clean.exists():
        files.extend(sorted((clean / "predictions").glob("*.jsonl")))
    else:
        for m in ["vanilla_mc", "temperature_mc", "semantic_entropy_mc", "self_consistency_mc"]:
            files.extend(sorted((base / "predictions").glob(f"mistral_7b_iga_seed1_*_{m}.jsonl")))
files.extend(sorted((out / "predictions").glob("*.jsonl")))
if not files:
    raise SystemExit("No files to aggregate")
subprocess.run([sys.executable, "-m", "iga_llm.report", "--out_dir", str(out / "aggregate"), "--predictions", *map(str, files)], check=True)
PY

echo "Done. Inspect:"
echo "  column -s, -t < $OUT_ROOT/aggregate/summary_by_model_benchmark_method.csv | less -S"
