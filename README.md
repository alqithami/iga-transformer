# Inhibitory-Gate Attention (IGA)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.XXXXX-b31b1b.svg)](https://arxiv.org/)

This repository contains code for **Inhibitory-Gate Attention: Learned Negative Routing for Factual Calibration in Language Models**.

Inhibitory-Gate Attention (IGA) is a frozen-backbone intervention for decoder-only language models. It learns a nonnegative attention-logit penalty and applies it before the attention softmax:

```text
A' = softmax(S - Gamma),   Gamma >= 0
```

The main implementation is dense low-rank IGA, where a factorized query-key penalty is scaled by a row-wise uncertainty gate. The repository also includes sparse inhibition, risk-gate diagnostics, LoRA and LoRA+IGA composition workflows, choice-shuffle audits, higher-sample self-consistency diagnostics, DoLa generation-harness evaluation, and reporting utilities.

## Repository scope

The repository provides source code, configuration files, and scripts for reproducing the experimental workflow. It does not include pretrained model weights, trained adapter checkpoints, benchmark corpora, or large generated result directories. Those assets should be generated locally or stored outside version control.

## Repository layout

```text
configs/                 Model and experiment configuration files
scripts/                 Experiment, audit, bootstrap, and utility scripts
src/iga_llm/             IGA package source code
docs/                    Reproducibility and artifact documentation
pyproject.toml           Python package metadata
requirements_gpu.txt     GPU environment requirements
```

## Installation

Create and activate a Python environment, then install the package in editable mode.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements_gpu.txt
python -m pip install -e .
```

For gated Hugging Face models, authenticate before running experiments.

```bash
hf auth login
```

The GPU experiments used CUDA-capable hardware with bfloat16 support. The smoke test can run on smaller devices because it uses a tiny model and synthetic data.

## Smoke test

```bash
bash scripts/run_smoke.sh
```

The smoke test verifies that data loading, IGA installation, training, evaluation, latency measurement, and reporting execute successfully on a minimal example.

## Data protocol

The pipeline works with deterministic JSONL splits and manifests for:

- TruthfulQA-MC1
- FEVER
- HaluEval-QA

Training uses a calibration mixture built from train/development partitions. Final metrics are computed on held-out evaluation partitions. FEVER is loaded through JSONL sources to avoid deprecated dataset-script behavior. Split files and manifests are written under each run directory.

## Core workflow

### Dense IGA training and evaluation

The main training entry point is:

```bash
python -m iga_llm.train \
  --config <config.yaml> \
  --train_jsonl <calibration_train_mix.jsonl> \
  --dev_jsonl <calibration_dev_mix.jsonl> \
  --output_dir <run_dir> \
  --seed <seed> \
  --epochs 1
```

Choice evaluation is run with:

```bash
python -m iga_llm.evaluate \
  --config <config.yaml> \
  --data <eval.jsonl> \
  --out <predictions.jsonl> \
  --method iga_mc \
  --iga_checkpoint <run_dir>/iga_modules.pt \
  --seed <seed> \
  --run_id <run_id>
```

Supported choice-mode methods include:

```text
vanilla_mc
temperature_mc
semantic_entropy_mc
self_consistency_mc
iga_mc
```

### Reporting

Aggregate prediction JSONL files with:

```bash
python -m iga_llm.report \
  --out_dir <aggregate_out> \
  --predictions <prediction_files...>
```

The report module writes:

```text
summary_per_seed.csv
summary_by_model_benchmark_method.csv
calibration_bins.csv
paired_deltas_vs_vanilla.csv
paired_delta_summary.csv
main_results_table.tex
ablation_table.tex
latency_summary.csv
efficiency_table.tex
REPORT.json
```

## Qwen dense IGA workflow

Run the three dense Qwen seeds:

```bash
bash scripts/run_qwen_dense_seed.sh 1
bash scripts/run_qwen_dense_seed.sh 2
bash scripts/run_qwen_dense_seed.sh 3
```

Aggregate the three-seed Qwen matrix:

```bash
bash scripts/aggregate_qwen_three_seed.sh
```

## LoRA and LoRA+IGA composition

The composition workflow loads a trained LoRA adapter, merges and freezes it into the backbone, and trains only the IGA controller on top of the LoRA-adapted model. In this setting, the entropy gate is computed from the LoRA-adapted model.

Run the Qwen LoRA baseline:

```bash
bash scripts/run_qwen_lora_fixed_pipeline.sh
```

Run Qwen LoRA+IGA composition:

```bash
bash scripts/run_qwen_lora_plus_iga_three_seeds.sh
```

For Mistral, train LoRA adapters with `scripts/train_lora_choice.py`, then run the LoRA+IGA diagnostic one seed at a time:

```bash
python scripts/train_lora_choice.py \
  --model_id mistralai/Mistral-7B-Instruct-v0.3 \
  --train_jsonl results/mistral_full_generation_20260507_234820/data/calibration_train_mix.jsonl \
  --dev_jsonl results/mistral_full_generation_20260507_234820/data/calibration_dev_mix.jsonl \
  --output_dir results/mistral_7b_lora_matched/seed1/run \
  --seed 1 \
  --epochs 1 \
  --lora_r 2 \
  --lora_alpha 4 \
  --target_modules q_proj,v_proj,o_proj \
  --dtype bfloat16

bash scripts/run_mistral_lora_plus_iga_one_seed.sh 1
```

## Analysis utilities

### Selective prediction and no-HaluEval diagnostics

```bash
python scripts/reviewer_risk_checks.py \
  --out_dir <diagnostic_out> \
  --predictions <prediction_files...>
```

This script reports pooled accuracy, ECE10, AURC, accuracy at fixed coverage, and a no-HaluEval aggregate.

### Paired bootstrap

```bash
python scripts/bootstrap_two_methods.py \
  --a_label <method_a> \
  --b_label <method_b> \
  --a <method_a_prediction_files...> \
  --b <method_b_prediction_files...> \
  --out <bootstrap.csv>
```

### Choice-shuffle audit

```bash
bash scripts/run_qwen_choice_shuffle_audit.sh
```

The shuffle audit evaluates robustness to answer-order artifacts by deterministically permuting answer choices and recomputing choice likelihoods.

### Higher-sample self-consistency

```bash
bash scripts/resume_qwen_sc20.sh
```

This workflow evaluates a 20-sample Qwen self-consistency diagnostic and resumes missing or incomplete files when interrupted.

## Generated artifacts

Generated results are intentionally kept outside version control. Typical generated artifacts include:

```text
results/**/predictions/*.jsonl
results/**/aggregate/*.csv
results/**/aggregate/*.json
runs/**/iga_modules.pt
**/lora_adapter/
```

The repository `.gitignore` excludes model weights, adapter checkpoints, and large generated outputs.

## Citation

If this code is used in academic work, cite the accompanying paper and the upstream datasets and model checkpoints used in the experiments.
