# Inhibitory-Gate Attention (IGA)

This repository contains the research code for **Inhibitory-Gate Attention: Learned Negative Routing for Factual Calibration in Language Models**.

IGA is a frozen-backbone intervention for decoder-only language models. It learns a small, nonnegative inhibitory field and subtracts it from attention logits:

```text
A' = softmax(S - Gamma),   Gamma >= 0
```

The main implementation is dense low-rank IGA: a low-rank query--key penalty scaled by a row-wise entropy gate. The repository also includes sparse IGA, learned-risk-gate diagnostics, LoRA and LoRA+IGA composition scripts, choice-shuffle audits, higher-sample self-consistency checks, DoLa generation-harness diagnostics, and reviewer-oriented bootstrap/selective-prediction analyses.

## What this code supports

- Frozen-backbone IGA training and choice evaluation.
- Mistral-7B, Llama-3-8B, and Qwen2.5-7B configs.
- Dense and sparse IGA variants.
- LoRA-only and frozen-LoRA + trainable-IGA composition experiments.
- Choice-likelihood baselines: vanilla, temperature scaling, semantic entropy, and self-consistency.
- Higher-sample self-consistency diagnostics.
- Choice-shuffle audits for answer-order artifacts.
- DoLa custom-generation harness diagnostics.
- Aggregate reporting: factual accuracy, ECE, Brier score, latency, AURC/selective prediction, calibration bins, paired deltas, and bootstrap intervals.

The code does **not** include model weights, trained adapters, or benchmark data. Those should be regenerated locally or supplied through a separate data artifact.

## Setup

### GPU environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -U \
  transformers datasets accelerate safetensors sentencepiece protobuf tokenizers \
  tqdm numpy pandas scikit-learn pyyaml psutil peft "huggingface_hub[cli]"
python -m pip install -e .
```

For gated models, authenticate with Hugging Face:

```bash
hf auth login
```

### Smoke test

```bash
bash scripts/run_smoke.sh
```

The smoke test uses a tiny model and synthetic data. Do not report smoke-test numbers.

## Data protocol

The pipeline prepares deterministic JSONL splits and manifests for:

- TruthfulQA-MC1
- FEVER
- HaluEval-QA

The calibration mixture uses train/dev partitions; final evaluation uses held-out evaluation partitions. FEVER is loaded from JSONL sources to avoid deprecated dataset-script behavior.

## Core dense IGA runs

### Qwen dense IGA, one seed

```bash
bash scripts/run_qwen_dense_seed.sh 1
bash scripts/run_qwen_dense_seed.sh 2
bash scripts/run_qwen_dense_seed.sh 3
```

### Aggregate Qwen dense/baseline runs

```bash
bash scripts/aggregate_qwen_three_seed.sh
```

### Mistral/Llama paper runs

Use the model-specific configs in `configs/` and the runner scripts in `scripts/`. The full Mac runner remains available for local Apple Silicon reproduction, but GPU execution is recommended for Qwen and LoRA+IGA.

## LoRA and LoRA+IGA composition

The composition experiment loads a trained LoRA adapter, merges/freezes it into the backbone, and trains only the IGA controller on top of the LoRA-adapted model. The entropy gate is computed from the LoRA-adapted backbone, not from vanilla predictions.

### Train/evaluate Qwen LoRA baseline

```bash
bash scripts/run_qwen_lora_fixed_pipeline.sh
```

### Qwen LoRA+IGA composition

```bash
bash scripts/run_qwen_lora_plus_iga_three_seeds.sh
```

### Mistral LoRA and Mistral LoRA+IGA diagnostic

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
bash scripts/run_mistral_lora_plus_iga_one_seed.sh 2
bash scripts/run_mistral_lora_plus_iga_one_seed.sh 3
```

## Reviewer diagnostics

### Selective prediction, no-HaluEval, and AURC

```bash
python scripts/reviewer_risk_checks.py \
  --out_dir results/reviewer_checks_qwen_lora_plus_iga \
  --predictions results/qwen2_5_7b_lora_fixed_eval/seed*/predictions/*.jsonl \
                results/qwen2_5_7b_lora_plus_iga_seed*/predictions/*.jsonl
```

### Paired bootstrap

```bash
python scripts/bootstrap_two_methods.py \
  --a_label lora_iga \
  --b_label lora \
  --a results/qwen2_5_7b_lora_plus_iga_seed*/predictions/*.jsonl \
  --b results/qwen2_5_7b_lora_fixed_eval/seed*/predictions/*.jsonl \
  --out results/reviewer_checks_qwen_lora_plus_iga/bootstrap_lora_iga_vs_lora.csv
```

### Qwen 20-sample self-consistency diagnostic

```bash
# Use the resume script pattern if long runs are interrupted.
bash scripts/resume_qwen_sc20.sh
```

If the resume script is not present, run `iga_llm.evaluate` with `--method self_consistency_mc --report_method self_consistency20_mc --num_samples 20` for each seed/benchmark.

### Choice-shuffle audit

```bash
bash scripts/run_qwen_choice_shuffle_audit.sh
```

## Reporting

Aggregate any set of prediction JSONL files with:

```bash
python -m iga_llm.report --out_dir <aggregate_out> --predictions <prediction_files...>
```

Typical outputs:

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

## Artifact policy

Do not commit large generated artifacts to this repository:

- model weights
- LoRA adapter weights
- `.pt`, `.bin`, `.safetensors`
- raw full result folders
- large prediction archives

For review, provide code and data artifacts separately. The ARR software artifact should contain this repository code/configs/scripts; the ARR data artifact should contain processed splits, aggregate CSVs, generated tables, and weight-free prediction archives.

## Notes on scope

The main paper claims concern labeled choice/verification scoring. Generation-mode DoLa is included as a harness diagnostic only; deployment-grade autoregressive IGA requires fused inhibitory-attention kernels.
