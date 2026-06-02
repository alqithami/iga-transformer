# Result artifact guide

Large generated results are intentionally excluded from the GitHub repository. For review or reproduction, keep them in a separate data artifact.

Recommended data artifact layout:

```text
processed_splits/
results_inputs/
generated_tables/
raw_prediction_archives/
MANIFEST.json
README_REVIEWER_DATA.md
```

Do not include model weights or adapters in the data artifact unless explicitly permitted by the venue and license.

## Minimal files for recomputation

- processed split JSONLs and manifests
- raw prediction JSONLs or weight-free archives
- aggregate CSV/JSON files
- calibration bin CSVs
- bootstrap CSVs
- generated LaTeX tables

## Important diagnostics

- `reviewer_checks_qwen_lora_plus_iga/selective_prediction_summary.csv`
- `reviewer_checks_qwen_lora_plus_iga/bootstrap_lora_iga_vs_lora.csv`
- `qwen2_5_7b_selfconsistency20/aggregate/summary_by_model_benchmark_method.csv`
- `mistral_lora_plus_iga_corrected/summary_by_model_benchmark_method.csv`
