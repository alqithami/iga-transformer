#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.mac_m4 2>/dev/null || true
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p data runs results/smoke
python -m iga_llm.prepare_data --benchmark synthetic --out data/synthetic.jsonl
python -m iga_llm.train \
  --config configs/tiny_debug.yaml \
  --train_jsonl data/synthetic.jsonl \
  --dev_jsonl data/synthetic.jsonl \
  --output_dir runs/smoke_tiny_iga \
  --epochs 1 \
  --limit 2 \
  --print_every 1
python -m iga_llm.evaluate \
  --config configs/tiny_debug.yaml \
  --data data/synthetic.jsonl \
  --out results/smoke/vanilla_mc.jsonl \
  --method vanilla_mc \
  --run_id smoke \
  --seed 1 \
  --limit 3
python -m iga_llm.evaluate \
  --config configs/tiny_debug.yaml \
  --data data/synthetic.jsonl \
  --out results/smoke/iga_mc.jsonl \
  --method iga_mc \
  --iga_checkpoint runs/smoke_tiny_iga \
  --run_id smoke \
  --seed 1 \
  --limit 3
python -m iga_llm.benchmark_latency \
  --config configs/tiny_debug.yaml \
  --method vanilla \
  --out results/smoke/latency_vanilla.jsonl \
  --prompt_tokens 32 \
  --max_new_tokens 4 \
  --warmup 0 \
  --runs 1
python -m iga_llm.report \
  --out_dir results/smoke/aggregate \
  --predictions results/smoke/vanilla_mc.jsonl results/smoke/iga_mc.jsonl \
  --latency results/smoke/latency_vanilla.jsonl
printf '\nSmoke test complete. See results/smoke/aggregate/summary_by_model_benchmark_method.csv\n'
