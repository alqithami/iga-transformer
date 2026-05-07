# IGA reviewer-grade experiment pipeline

This repository is the runnable local pipeline for **Uncertainty-Gated Inhibitory Attention (IGA)** experiments.

It is designed to generate the result package reviewers will ask for:

- factual accuracy / hallucination proxy / ECE / Brier score
- answer rate, refusal rate, generation length, parse-success rate
- seed-separated and model-separated summaries
- raw per-example JSONL prediction records
- calibration bins and paired deltas against vanilla
- latency, tokens/sec, and memory traces at multiple prompt lengths
- IGA diagnostics: gamma/gate/pattern statistics
- ablations for gate, inhibition pattern, layer placement, head selection, rank, gamma strength, threshold training, and inference-only IGA

The code intentionally does **not** fabricate numerical results. Tables are generated only from raw JSONL outputs.

## 1. Setup on Mac M4 Max

```bash
cd code
bash scripts/setup_mac_m4.sh
source .venv/bin/activate
source .env.mac_m4
```

For gated models such as Llama 3, authenticate first:

```bash
huggingface-cli login
```

## 2. Smoke test

This verifies the code path using a tiny Llama-like checkpoint and synthetic data. Do not report these numbers.

```bash
bash scripts/run_smoke.sh
```

Outputs:

```text
results/smoke/aggregate/summary_by_model_benchmark_method.csv
results/smoke/aggregate/main_results_table.tex
```

## 3. First serious M4 run

Start with Mistral and smaller limits:

```bash
CONFIGS="configs/mistral_7b_iga.yaml" \
SEEDS="1" \
LIMIT_TRAIN=300 \
LIMIT_DEV=100 \
LIMIT_EVAL=100 \
RUN_ABLATIONS=1 \
SKIP_GENERATION=1 \
bash scripts/run_full_mac_m4max.sh
```

This produces the reviewer-critical multiple-choice/verification evidence without the slower free-form decoding baselines.

## 4. Full paper matrix

```bash
CONFIGS="configs/llama3_8b_iga.yaml configs/mistral_7b_iga.yaml" \
SEEDS="1 2 3" \
LIMIT_TRAIN=1000 \
LIMIT_DEV=300 \
LIMIT_EVAL=300 \
RUN_ABLATIONS=1 \
bash scripts/run_full_mac_m4max.sh
```

Or simply:

```bash
bash scripts/run_two_model_paper_matrix.sh
```

## 5. Where results appear

A full run writes:

```text
results/full_<timestamp>/
  data/
    truthfulqa_train.jsonl
    truthfulqa_dev.jsonl
    truthfulqa_eval.jsonl
    fever_train.jsonl
    fever_dev.jsonl
    fever_eval.jsonl
    halueval_train.jsonl
    halueval_dev.jsonl
    halueval_eval.jsonl
    calibration_train_mix.jsonl
    calibration_dev_mix.jsonl
  runs/
    <model>_seed<seed>/full_iga/iga_modules.pt
    <model>_seed<seed>/<ablation>/iga_modules.pt
  predictions/
    *.jsonl
  latency/
    *.jsonl
  aggregate/
    summary_per_seed.csv
    summary_by_model_benchmark_method.csv
    calibration_bins.csv
    paired_deltas_vs_vanilla.csv
    paired_delta_summary.csv
    latency_summary.csv
    main_results_table.tex
    ablation_table.tex
    efficiency_table.tex
    REPORT.json
  matrix_manifest.json
```

## 6. What is trained

The pretrained LLM backbone is frozen. IGA installs small trainable modules into selected self-attention layers.

Default IGA uses:

```text
attention_logits' = attention_logits - Gamma
Gamma_ijh = head_strength_h * gate(phi_i) * f_theta(h_i, h_j)
gate(phi_i) = gamma_max * sigmoid(beta * (phi_i - tau))
```

The default `f_theta` is a paper-aligned pair MLP over low-rank q/k interaction features:

```text
f_theta(i,j) = softplus(MLP([q_i ⊙ k_j ; |q_i-k_j| ; q_i^T k_j]))
```

Config fields:

```yaml
iga:
  layers: late
  rank: 16
  pattern_type: pair_mlp
  uncertainty_mode: entropy
  gamma_max: 2.0
  beta: 8.0
  tau: 0.35
  train_gamma_threshold: false
  head_selection: all
```

By default, `gamma_max`, `beta`, and `tau` are fixed. The `ablate_train_threshold` condition learns them.

## 7. Baselines included

Choice / verifier-mode baselines:

```text
vanilla_mc
temperature_mc
self_consistency_mc
semantic_entropy_mc
iga_mc
```

Generation-classification baselines:

```text
vanilla_gen
contrastive_gen
dola_gen
self_refine_gen
cove_gen
iga_gen
```

Generation baselines are parsed back to benchmark labels and include parse-success flags. Use them as secondary evidence unless you add an external factuality judge.

## 8. Ablations included

```text
ablate_full_iga
ablate_no_uncertainty_gate
ablate_no_inhibition
ablate_constant_inhibition
ablate_low_rank_dot
ablate_pair_mlp
ablate_early_layers
ablate_middle_layers
ablate_all_layers
ablate_selected_heads_even
ablate_train_threshold
ablate_rank_8
ablate_rank_64
ablate_gamma_0p5
ablate_gamma_4
ablate_inference_only
```

The default ablation run uses a compact subset that covers the key reviewer questions.

## 9. Data leakage prevention

The runner prepares disjoint calibration/dev/evaluation files:

- TruthfulQA: deterministic hash split of validation into train/dev/test partitions.
- HaluEval: deterministic hash split of the selected subset, default `qa`, into train/dev/test partitions.
- FEVER: train/dev come from FEVER train; final evaluation comes from `labelled_dev`.

All JSONL files and manifests are hashed.

## 10. Paper integration

After a run:

```bash
cp results/full_<timestamp>/aggregate/main_results_table.tex ../paper/tables/main_results.tex
cp results/full_<timestamp>/aggregate/ablation_table.tex ../paper/tables/ablations.tex
cp results/full_<timestamp>/aggregate/efficiency_table.tex ../paper/tables/efficiency.tex
```

Do not claim fixed overhead such as “8%” until `efficiency_table.tex` supports it.
