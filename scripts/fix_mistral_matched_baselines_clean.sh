#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate
source .env.mac_m4

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

BASE_OUT="results/mistral_full_generation_20260507_234820"
TUNE_OUT="results/mistral_tuning_20260510_065427"
MISTRAL_CFG="$BASE_OUT/resolved_configs/mistral_7b_iga.yaml"

OLD_PRED_DIR="results/mistral_matched_baseline_seeds_2_3/predictions"
CLEAN_OUT="results/mistral_matched_baseline_seeds_2_3_CLEAN"
CLEAN_PRED_DIR="$CLEAN_OUT/predictions"
CLEAN_AGG_DIR="$CLEAN_OUT/aggregate"

SEEDS="2 3"
METHODS="vanilla_mc temperature_mc semantic_entropy_mc self_consistency_mc"
BENCHES="truthfulqa fever halueval"

mkdir -p "$CLEAN_PRED_DIR" "$CLEAN_AGG_DIR" "$CLEAN_OUT/logs"

echo "=== Mistral repair settings ==="
echo "BASE_OUT=$BASE_OUT"
echo "TUNE_OUT=$TUNE_OUT"
echo "MISTRAL_CFG=$MISTRAL_CFG"
echo "OLD_PRED_DIR=$OLD_PRED_DIR"
echo "CLEAN_OUT=$CLEAN_OUT"
echo

if [ ! -f "$MISTRAL_CFG" ]; then
  echo "ERROR: Mistral config not found: $MISTRAL_CFG"
  exit 1
fi

if ! grep -q "$MISTRAL_MODEL" "$MISTRAL_CFG"; then
  echo "ERROR: config does not contain expected Mistral model_id:"
  echo "Expected: $MISTRAL_MODEL"
  echo "Config:   $MISTRAL_CFG"
  echo
  cat "$MISTRAL_CFG"
  exit 1
fi

for BENCH in $BENCHES; do
  DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
  if [ ! -f "$DATA" ]; then
    echo "ERROR: missing eval data: $DATA"
    exit 1
  fi
done

echo "=== Step 1: copy only valid old Mistral baseline files, if any ==="

python - <<'PY'
from pathlib import Path
import json
import shutil

MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

old_dir = Path("results/mistral_matched_baseline_seeds_2_3/predictions")
clean_dir = Path("results/mistral_matched_baseline_seeds_2_3_CLEAN/predictions")
clean_dir.mkdir(parents=True, exist_ok=True)

expected_counts = {
    "truthfulqa": 149,
    "fever": 300,
    "halueval": 300,
}

if not old_dir.exists():
    print("No old prediction directory found; will regenerate all baseline seed-2/3 files.")
    raise SystemExit(0)

kept = 0
skipped = 0

for path in sorted(old_dir.glob("*.jsonl")):
    # Only consider baseline files from this folder.
    if not any(m in path.name for m in ["vanilla_mc", "temperature_mc", "semantic_entropy_mc", "self_consistency_mc"]):
        continue

    try:
        with open(path) as f:
            first = json.loads(next(f))
    except Exception as e:
        print("SKIP unreadable:", path, e)
        skipped += 1
        continue

    model_id = first.get("model_id")
    bench = next((b for b in expected_counts if b in path.name), None)
    expected = expected_counts.get(bench)

    n = sum(1 for _ in open(path))
    if model_id == MISTRAL_MODEL and expected is not None and n == expected:
        dst = clean_dir / path.name
        shutil.copy2(path, dst)
        print(f"KEEP {path.name}: model={model_id}, rows={n}/{expected}")
        kept += 1
    else:
        print(f"SKIP {path.name}: model={model_id}, rows={n}/{expected}")
        skipped += 1

print(f"\nCopied valid old Mistral files: {kept}")
print(f"Skipped old files: {skipped}")
PY

echo
echo "=== Step 2: run only missing Mistral baseline seed-2/3 files ==="

for SEED in $SEEDS; do
  RUN_ID="mistral_7b_baseline_seed${SEED}"

  for METHOD in $METHODS; do
    for BENCH in $BENCHES; do
      DATA="$BASE_OUT/data/${BENCH}_eval.jsonl"
      EXPECTED="$(wc -l < "$DATA" | tr -d ' ')"
      OUT_FILE="$CLEAN_PRED_DIR/${RUN_ID}_${BENCH}_${METHOD}.jsonl"

      if [ -f "$OUT_FILE" ]; then
        N="$(wc -l < "$OUT_FILE" | tr -d ' ')"
        MODEL_ID="$(python - <<PY
import json
with open("$OUT_FILE") as f:
    row=json.loads(next(f))
print(row.get("model_id"))
PY
)"
        if [ "$N" = "$EXPECTED" ] && [ "$MODEL_ID" = "$MISTRAL_MODEL" ]; then
          echo "Skipping complete Mistral file: $OUT_FILE ($N/$EXPECTED)"
          continue
        else
          echo "Removing bad/incomplete file: $OUT_FILE rows=$N/$EXPECTED model=$MODEL_ID"
          rm -f "$OUT_FILE"
        fi
      fi

      echo
      echo "Running TRUE Mistral baseline: seed=$SEED method=$METHOD bench=$BENCH expected=$EXPECTED"

      caffeinate -dimsu python -m iga_llm.evaluate \
        --config "$MISTRAL_CFG" \
        --data "$DATA" \
        --out "$OUT_FILE" \
        --method "$METHOD" \
        --seed "$SEED" \
        --run_id "$RUN_ID" \
        --max_new_tokens 32 \
        --num_samples 5 \
        --temperature 0.7 \
        --top_p 0.95

      python - <<PY
import json
from pathlib import Path
p = Path("$OUT_FILE")
n = sum(1 for _ in open(p))
with open(p) as f:
    row = json.loads(next(f))
model_id = row.get("model_id")
if model_id != "$MISTRAL_MODEL":
    raise SystemExit(f"ERROR: wrong model_id in {p}: {model_id}")
if n != int("$EXPECTED"):
    raise SystemExit(f"ERROR: wrong row count in {p}: {n}/$EXPECTED")
print(f"Verified {p.name}: model={model_id}, rows={n}")
PY
    done
  done
done

echo
echo "=== Step 3: aggregate a clean Mistral-only result ==="

python - <<'PY'
from pathlib import Path
import json
import subprocess
import sys

MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

base = Path("results/mistral_full_generation_20260507_234820")
tune = Path("results/mistral_tuning_20260510_065427")
clean = Path("results/mistral_matched_baseline_seeds_2_3_CLEAN")
agg = clean / "aggregate"
agg.mkdir(parents=True, exist_ok=True)

candidate_files = []

# Existing valid seed-1 Mistral baselines.
for method in [
    "vanilla_mc",
    "temperature_mc",
    "semantic_entropy_mc",
    "self_consistency_mc",
]:
    candidate_files.extend(sorted((base / "predictions").glob(
        f"mistral_7b_iga_seed1_*_{method}.jsonl"
    )))

# Clean baseline seeds 2 and 3.
candidate_files.extend(sorted((clean / "predictions").glob("*.jsonl")))

# Existing selected IGA-v2 seed 1.
candidate_files.extend(sorted((tune / "eval_predictions").glob(
    "mistral_7b_iga_lowrank_g2_tau035_*_eval_iga_mc.jsonl"
)))

# Existing selected IGA-v2 seeds 2 and 3.
for seed in [2, 3]:
    candidate_files.extend(sorted(Path(
        f"results/mistral_lowrank_g2_seed{seed}/predictions"
    ).glob("*.jsonl")))

# Deduplicate and filter strictly by model_id.
seen = set()
files = []
bad = []
for p in candidate_files:
    s = str(p)
    if s in seen:
        continue
    seen.add(s)

    try:
        with open(p) as f:
            row = json.loads(next(f))
    except Exception as e:
        bad.append((str(p), f"unreadable: {e}"))
        continue

    model_id = row.get("model_id")
    if model_id == MISTRAL_MODEL:
        files.append(p)
    else:
        bad.append((str(p), model_id))

print(f"Candidate files: {len(candidate_files)}")
print(f"Clean Mistral files: {len(files)}")
if bad:
    print("\nExcluded non-Mistral/unreadable files:")
    for p, reason in bad:
        print(" ", p, "=>", reason)

if not files:
    raise SystemExit("No clean Mistral files found.")

print("\nFiles included in clean aggregate:")
for p in files:
    print(" ", p)

subprocess.run(
    [
        sys.executable,
        "-m",
        "iga_llm.report",
        "--out_dir",
        str(agg),
        "--predictions",
        *[str(p) for p in files],
    ],
    check=True,
)

print("\nWrote:", agg)
PY

echo
echo "=== Done ==="
echo "Inspect:"
echo "  column -s, -t < $CLEAN_AGG_DIR/summary_by_model_benchmark_method.csv | less -S"
echo "  cat $CLEAN_AGG_DIR/main_results_table.tex"
