#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate
source .env.mac_m4

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

BASE_OUT="${BASE_OUT:-results/mistral_full_generation_20260507_234820}"
TUNE_OUT="${TUNE_OUT:-results/mistral_tuning_20260510_065427}"
CFG="${CFG:-$BASE_OUT/resolved_configs/mistral_7b_iga.yaml}"
OUT_DIR="${OUT_DIR:-results/mistral_matched_baseline_seeds_2_3}"

SEEDS="${SEEDS:-2 3}"
METHODS="${METHODS:-vanilla_mc temperature_mc semantic_entropy_mc self_consistency_mc}"
BENCHES="${BENCHES:-truthfulqa fever halueval}"

mkdir -p "$OUT_DIR/predictions" "$OUT_DIR/aggregate" "$OUT_DIR/logs"

echo "BASE_OUT=$BASE_OUT"
echo "TUNE_OUT=$TUNE_OUT"
echo "CFG=$CFG"
echo "OUT_DIR=$OUT_DIR"
echo "SEEDS=$SEEDS"
echo "METHODS=$METHODS"
echo "BENCHES=$BENCHES"

if [ ! -f "$CFG" ]; then
  echo "ERROR: config not found: $CFG"
  exit 1
fi

for BENCH in $BENCHES; do
  DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
  if [ ! -f "$DATA" ]; then
    echo "ERROR: data file not found: $DATA"
    exit 1
  fi
done

for SEED in $SEEDS; do
  RUN_ID="mistral_7b_baseline_seed${SEED}"

  for METHOD in $METHODS; do
    for BENCH in $BENCHES; do
      DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
      EXPECTED="$(wc -l < "$DATA" | tr -d ' ')"
      OUT_FILE="$OUT_DIR/predictions/${RUN_ID}_${BENCH}_${METHOD}.jsonl"

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
      echo "Running Mistral baseline: seed=$SEED method=$METHOD bench=$BENCH expected=$EXPECTED"
      caffeinate -dimsu python -m iga_llm.evaluate \
        --config "$CFG" \
        --data "$DATA" \
        --out "$OUT_FILE" \
        --method "$METHOD" \
        --seed "$SEED" \
        --run_id "$RUN_ID" \
        --max_new_tokens 32 \
        --num_samples 5 \
        --temperature 0.7 \
        --top_p 0.95
    done
  done
done

echo
echo "Aggregating Mistral matched baseline seeds with existing IGA-v2 seeds..."

python - <<'PY'
from pathlib import Path
import os
import subprocess
import sys

base = Path(os.environ.get("BASE_OUT", "results/mistral_full_generation_20260507_234820"))
tune = Path(os.environ.get("TUNE_OUT", "results/mistral_tuning_20260510_065427"))
out = Path(os.environ.get("OUT_DIR", "results/mistral_matched_baseline_seeds_2_3"))
agg = out / "aggregate"
agg.mkdir(parents=True, exist_ok=True)

files = []

# Existing seed-1 baselines.
for method in [
    "vanilla_mc",
    "temperature_mc",
    "semantic_entropy_mc",
    "self_consistency_mc",
]:
    files.extend(sorted((base / "predictions").glob(
        f"mistral_7b_iga_seed1_*_{method}.jsonl"
    )))

# Newly created baseline seeds 2 and 3.
files.extend(sorted((out / "predictions").glob("*.jsonl")))

# Existing selected IGA-v2 seed 1.
files.extend(sorted((tune / "eval_predictions").glob(
    "mistral_7b_iga_lowrank_g2_tau035_*_eval_iga_mc.jsonl"
)))

# Existing selected IGA-v2 seeds 2 and 3.
for seed in [2, 3]:
    files.extend(sorted(Path(
        f"results/mistral_lowrank_g2_seed{seed}/predictions"
    ).glob("*.jsonl")))

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
echo "  column -s, -t < $OUT_DIR/aggregate/summary_by_model_benchmark_method.csv | less -S"
