#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate
source .env.mac_m4

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
CFG="${CFG:-configs/llama3_8b_iga_lowrank_g2_tau035.yaml}"
OUT_ROOT="${OUT_ROOT:-results/llama3_8b_lowrank_g2_multiseed}"
SEEDS="${SEEDS:-2 3}"
BENCHES="${BENCHES:-truthfulqa fever halueval}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/aggregate" "$OUT_ROOT/logs" configs

echo "BASE_OUT=$BASE_OUT"
echo "CFG=$CFG"
echo "OUT_ROOT=$OUT_ROOT"
echo "SEEDS=$SEEDS"
echo "BENCHES=$BENCHES"

# Create the Llama low-rank IGA config if it does not exist.
if [ ! -f "$CFG" ]; then
  echo "Creating config: $CFG"
  cat > "$CFG" <<'YAML'
model_id: meta-llama/Meta-Llama-3-8B-Instruct
model_revision: null
device: auto
dtype: float16
attn_implementation: eager
trust_remote_code: false
seed: 1
iga:
  layers: late
  rank: 16
  pattern_type: low_rank
  uncertainty_mode: entropy
  gamma_max: 2.0
  beta: 8.0
  tau: 0.35
  init_head_strength: 0.05
  train_gamma_threshold: false
  head_selection: all
  pair_hidden_mult: 2.0
train:
  epochs: 1
  batch_size: 1
  lr: 0.00015
  weight_decay: 0.0
  gamma_reg: 0.003
  max_grad_norm: 1.0
  max_length: 768
  dev_eval_limit: 300
YAML
fi

for BENCH in $BENCHES; do
  DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
  if [ ! -f "$DATA" ]; then
    echo "ERROR: data file not found: $DATA"
    exit 1
  fi
done

for SEED in $SEEDS; do
  RUN_OUT="$OUT_ROOT/seed${SEED}"
  RUN_NAME="llama3_8b_iga_lowrank_g2_tau035_seed${SEED}"
  RUN_DIR="$RUN_OUT/runs/$RUN_NAME"
  PRED_DIR="$RUN_OUT/predictions"

  mkdir -p "$RUN_DIR" "$PRED_DIR"

  CKPT="$RUN_DIR/iga_modules.pt"

  if [ -f "$CKPT" ]; then
    echo "Skipping training; checkpoint exists: $CKPT"
  else
    echo
    echo "Training Llama IGA-v2 seed=$SEED"
    caffeinate -dimsu python -m iga_llm.train \
      --config "$CFG" \
      --train_jsonl "$BASE_OUT/data/calibration_train_mix.jsonl" \
      --dev_jsonl "$BASE_OUT/data/calibration_dev_mix.jsonl" \
      --output_dir "$RUN_DIR" \
      --seed "$SEED" \
      --epochs 1
  fi

  if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint missing after training: $CKPT"
    exit 1
  fi

  for BENCH in $BENCHES; do
    DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
    EXPECTED="$(wc -l < "$DATA" | tr -d ' ')"
    OUT_FILE="$PRED_DIR/${RUN_NAME}_${BENCH}_eval_iga_mc.jsonl"

    if [ -f "$OUT_FILE" ]; then
      N="$(wc -l < "$OUT_FILE" | tr -d ' ')"
      if [ "$N" = "$EXPECTED" ]; then
        echo "Skipping complete file: $OUT_FILE ($N/$EXPECTED)"
        continue
      else
        echo "Removing incomplete file: $OUT_FILE ($N/$EXPECTED)"
        rm -f "$OUT_FILE"
      fi
    fi

    echo
    echo "Evaluating Llama IGA-v2: seed=$SEED bench=$BENCH expected=$EXPECTED"
    caffeinate -dimsu python -m iga_llm.evaluate \
      --config "$CFG" \
      --data "$DATA" \
      --out "$OUT_FILE" \
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
done

echo
echo "Aggregating Llama seed-1 baselines plus IGA seeds..."

python - <<'PY'
from pathlib import Path
import os
import subprocess
import sys

out_root = Path(os.environ.get("OUT_ROOT", "results/llama3_8b_lowrank_g2_multiseed"))
agg = out_root / "aggregate"
agg.mkdir(parents=True, exist_ok=True)

files = []

# Existing Llama seed-1 run, if present.
seed1_runs = sorted(Path("results").glob("llama3_8b_lowrank_g2_seed1_*"))
if seed1_runs:
    seed1 = seed1_runs[-1]
    print("Using Llama seed-1 folder:", seed1)
    files.extend(sorted((seed1 / "predictions").glob("*.jsonl")))
else:
    print("WARNING: no existing llama3_8b_lowrank_g2_seed1_* folder found.")

# New IGA seeds 2 and 3.
files.extend(sorted((out_root).glob("seed*/predictions/*.jsonl")))

# Deduplicate.
seen = set()
deduped = []
for f in files:
    s = str(f)
    if s not in seen:
        seen.add(s)
        deduped.append(f)

print(f"Using {len(deduped)} prediction files:")
for f in deduped:
    print(" ", f)

if not deduped:
    raise SystemExit("No prediction files found.")

cmd = [
    sys.executable,
    "-m",
    "iga_llm.report",
    "--out_dir",
    str(agg),
    "--predictions",
    *[str(f) for f in deduped],
]
subprocess.run(cmd, check=True)
print("\nWrote:", agg)
PY

echo
echo "Done. Inspect:"
echo "  column -s, -t < $OUT_ROOT/aggregate/summary_by_model_benchmark_method.csv | less -S"
