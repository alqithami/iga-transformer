# Inhibitory-Gate Attention

Inhibitory-Gate Attention (IGA) is a constrained attention controller for decoder-only language models. It augments a frozen transformer with a learned nonnegative edge-cost field and subtracts that field from the attention logits before row-wise normalization:

```text
A' = softmax(S - Gamma),   Gamma >= 0
```

The principal implementation uses low-rank query--key interactions and a row-wise uncertainty gate. The codebase also provides sparse inhibition, learned-risk diagnostics, LoRA baselines, LoRA+IGA composition, choice-order audits, higher-sample self-consistency evaluation, latency measurement, and result aggregation.

## Scope

The repository contains source code, experiment configurations, data-preparation utilities, and analysis scripts. Pretrained model weights, trained IGA checkpoints, LoRA adapters, benchmark corpora, and generated result directories are not versioned.

The experimental protocol covers:

- Mistral-7B-Instruct-v0.3
- Llama-3-8B-Instruct
- Qwen2.5-7B-Instruct
- TruthfulQA-MC1, FEVER, and HaluEval-QA
- factual accuracy, ECE, Brier score, AURC, selective prediction, latency, and inhibitory-controller diagnostics

## Repository structure

```text
configs/                 Model and experiment configurations
scripts/                 Reproduction, evaluation, and analysis commands
src/iga_llm/             Python package
docs/                    Protocol and artifact documentation
pyproject.toml            Package metadata
requirements_gpu.txt      Tested CUDA environment
```

## Installation

Python 3.10 or later is required. The reference GPU environment uses PyTorch 2.5.1 with CUDA 12.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements_gpu.txt
python -m pip install -e .
```

Authenticate with Hugging Face before using gated checkpoints:

```bash
hf auth login
```

## Smoke test

The smoke test uses a tiny model and synthetic examples. It verifies data preparation, IGA installation, training, choice evaluation, latency measurement, and reporting.

```bash
bash scripts/run_smoke.sh
```

Expected outputs are written under `results/smoke/`.

## Data preparation

The full matrix runner creates deterministic JSONL splits and manifests. TruthfulQA-MC1 and HaluEval-QA are hash-partitioned; FEVER training/development examples are drawn from the training split and final evaluation examples from `labelled_dev`.

```bash
python -m iga_llm.run_matrix \
  --configs configs/mistral_7b_iga_lowrank_g2_tau035.yaml \
  --out_dir results/paper_data \
  --seeds 1 \
  --limit_train 1000 \
  --limit_dev 300 \
  --limit_eval 300 \
  --skip_training \
  --skip_generation \
  --skip_latency
```

This command prepares:

```text
results/paper_data/data/calibration_train_mix.jsonl
results/paper_data/data/calibration_dev_mix.jsonl
results/paper_data/data/truthfulqa_eval.jsonl
results/paper_data/data/fever_eval.jsonl
results/paper_data/data/halueval_eval.jsonl
```

The experiment scripts accept `DATA_ROOT` to identify the directory containing these files:

```bash
export DATA_ROOT="$PWD/results/paper_data"
```

## Dense IGA

Run one Qwen seed:

```bash
DATA_ROOT="$PWD/results/paper_data" bash scripts/run_qwen_dense_seed.sh 1
```

Run seeds 1--3, then aggregate one complete result directory per seed:

```bash
for seed in 1 2 3; do
  DATA_ROOT="$PWD/results/paper_data" bash scripts/run_qwen_dense_seed.sh "$seed"
done
bash scripts/aggregate_qwen_three_seed.sh
```

The underlying entry points are also available directly:

```bash
python -m iga_llm.train \
  --config configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml \
  --train_jsonl "$DATA_ROOT/data/calibration_train_mix.jsonl" \
  --dev_jsonl "$DATA_ROOT/data/calibration_dev_mix.jsonl" \
  --output_dir results/qwen_seed1/run \
  --seed 1 \
  --epochs 1

python -m iga_llm.evaluate \
  --config configs/qwen2_5_7b_iga_lowrank_g2_tau035_matched_bf16.yaml \
  --data "$DATA_ROOT/data/truthfulqa_eval.jsonl" \
  --out results/qwen_seed1/predictions/truthfulqa_iga.jsonl \
  --method iga_mc \
  --iga_checkpoint results/qwen_seed1/run/iga_modules.pt \
  --seed 1 \
  --run_id qwen_seed1
```

## LoRA and LoRA+IGA composition

The composition pipeline merges a trained LoRA adapter into the backbone, freezes the adapted model, and trains only IGA. The entropy gate is therefore computed from the LoRA-adapted model.

```bash
DATA_ROOT="$PWD/results/paper_data" bash scripts/run_qwen_lora_fixed_pipeline.sh
DATA_ROOT="$PWD/results/paper_data" bash scripts/run_qwen_lora_plus_iga_three_seeds.sh
```

For Mistral, train a rank-2 LoRA adapter and then run the composition diagnostic:

```bash
python scripts/train_lora_choice.py \
  --model_id mistralai/Mistral-7B-Instruct-v0.3 \
  --train_jsonl "$DATA_ROOT/data/calibration_train_mix.jsonl" \
  --dev_jsonl "$DATA_ROOT/data/calibration_dev_mix.jsonl" \
  --output_dir results/mistral_7b_lora_matched/seed1/run \
  --seed 1 \
  --epochs 1 \
  --lora_r 2 \
  --lora_alpha 4 \
  --target_modules q_proj,v_proj,o_proj \
  --dtype bfloat16

DATA_ROOT="$PWD/results/paper_data" bash scripts/run_mistral_lora_plus_iga_one_seed.sh 1
```

Each composition training summary records `lora_merged_for_iga`, `phi_source`, and `risk_labels_computed` so the composition path can be audited.

## Additional evaluations

Choice-order audit:

```bash
DATA_ROOT="$PWD/results/paper_data" bash scripts/run_qwen_choice_shuffle_audit.sh
```

Twenty-sample self-consistency diagnostic:

```bash
DATA_ROOT="$PWD/results/paper_data" bash scripts/resume_qwen_sc20.sh
```

Generation-mode DoLa harness:

```bash
python scripts/evaluate_custom_dola_gen.py --help
```

The DoLa script is a generation-harness diagnostic; it is not a matched choice-likelihood implementation.

## Analysis and reporting

Aggregate prediction files:

```bash
python -m iga_llm.report \
  --out_dir results/aggregate \
  --predictions results/**/predictions/*.jsonl
```

The report module writes per-seed and aggregate CSV files, calibration bins, paired deltas, latency summaries, JSON manifests, and LaTeX tables.

Selective-prediction and no-HaluEval diagnostics:

```bash
python scripts/reviewer_risk_checks.py \
  --out_dir results/diagnostics \
  --predictions <prediction-files...>
```

Paired bootstrap comparison:

```bash
python scripts/bootstrap_two_methods.py \
  --a_label method_a \
  --b_label method_b \
  --a <method-a-files...> \
  --b <method-b-files...> \
  --out results/diagnostics/bootstrap.csv
```

## Reproducibility checks

```bash
bash scripts/static_check.sh
```

The check compiles Python sources, validates shell syntax, parses YAML configurations, and scans tracked source files for common machine-specific paths and generated binary artifacts.

Further protocol details are provided in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/RESULTS_ARTIFACTS.md`](docs/RESULTS_ARTIFACTS.md).

## Generated artifacts

The repository excludes generated outputs and model state:

```text
results/
runs/
logs/
*.pt
*.pth
*.bin
*.safetensors
**/lora_adapter/
```

A source-only archive can be created with:

```bash
bash scripts/export_source_artifact.sh
```

## License

The software is released under the MIT License. See [`LICENSE`](LICENSE).
