# Reproducibility notes

This document summarizes the reproducibility protocol for the IGA experiments.

## Main ingredients

- Frozen pretrained backbone models.
- Trainable IGA modules only, unless running explicit LoRA baselines.
- Deterministic JSONL train/dev/eval splits with manifests and hashes.
- Three-seed aggregation for main reported rows.
- Per-example prediction JSONL records for recomputation.

## Key result families

1. Dense IGA versus vanilla, temperature scaling, semantic entropy, and self-consistency.
2. Sparse IGA latency/factuality tradeoff.
3. Risk-gated V3 diagnostics.
4. Qwen LoRA-only and LoRA+IGA composition.
5. Mistral LoRA+IGA calibration-only diagnostic.
6. Qwen 20-sample self-consistency diagnostic.
7. Choice-shuffle audits.
8. DoLa custom-generation harness diagnostic.

## Recomputing tables

Use:

```bash
python -m iga_llm.report --out_dir <aggregate_out> --predictions <prediction_files...>
```

The report module recomputes factual accuracy, hallucination/error rate, ECE, Brier score, latency, calibration bins, paired deltas, and LaTeX-ready tables.

## Avoiding duplicate seeds

Some experiments create timestamped result folders. When aggregating reruns, select exactly one complete folder per seed. A complete choice-evaluation seed has one JSONL file per benchmark/method with expected row counts:

- TruthfulQA-MC1: 149
- FEVER: 300
- HaluEval-QA: 300

## LoRA+IGA composition check

For composition runs, confirm each `train_summary.json` contains:

```json
{
  "lora_merged_for_iga": true,
  "phi_source": "lora_adapted_backbone",
  "risk_labels_computed": 0
}
```

These fields verify that the entropy gate was computed from the LoRA-adapted model and that the learned-risk path was not accidentally used.
