# Reviewer audit runbook

## Minimal defensible table run

```bash
CONFIGS="configs/mistral_7b_iga.yaml" \
SEEDS="1 2 3" \
LIMIT_TRAIN=1000 \
LIMIT_DEV=300 \
LIMIT_EVAL=300 \
RUN_ABLATIONS=1 \
SKIP_GENERATION=1 \
bash scripts/run_full_mac_m4max.sh
```

This generates the main factuality/calibration tables from labeled tasks.

## Full NeurIPS package run

```bash
CONFIGS="configs/llama3_8b_iga.yaml configs/mistral_7b_iga.yaml" \
SEEDS="1 2 3" \
LIMIT_TRAIN=1000 \
LIMIT_DEV=300 \
LIMIT_EVAL=300 \
RUN_ABLATIONS=1 \
bash scripts/run_full_mac_m4max.sh
```

## Expected reviewer checks and where to find evidence

| Reviewer question | Evidence file |
|---|---|
| What exactly was trained? | `runs/*/train_summary.json`, `runs/*/resolved_config.yaml` |
| Were train and eval examples separated? | `data/*.manifest.json`, `matrix_manifest.json` |
| Are Llama and Mistral separated? | `aggregate/summary_by_model_benchmark_method.csv` |
| Are seeds separated? | `aggregate/summary_per_seed.csv` |
| Is IGA improving factuality or just refusing? | `answer_rate`, `refusal_rate`, `parse_success_rate`, `mean_generated_tokens` columns |
| Is calibration improved? | `ece`, `brier`, `calibration_bins.csv` |
| Is overhead measured honestly? | `latency/*.jsonl`, `aggregate/latency_summary.csv`, `aggregate/efficiency_table.tex` |
| Which examples changed? | `aggregate/paired_deltas_vs_vanilla.csv` |
| Is the mechanism doing anything? | `iga_gamma_mean`, `iga_gate_mean`, `iga_pattern_mean` columns |
| Which component matters? | `aggregate/ablation_table.tex` and ablation JSONL files |

## Recommended paper reporting discipline

Use labeled choice/verification results as the primary main table. Put generation-classification baselines and parse success in an appendix unless the parse success rate is high and stable. Report overhead from the latency benchmark, not from the choice-evaluation latency.
