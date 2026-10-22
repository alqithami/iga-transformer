# Error diagnosis — 21 October 2026

This is post-hoc analysis of the saved coverage-replication predictions, not a new training or heldout-confirmation result. It retains the inherited labels. Source: the verified `Error_Analysis_20261021` handoff; machine-readable values are in [error_diagnosis.json](error_diagnosis.json).

## Clean errors versus background-induced errors

For training selection B, average counts per separately scored fitted state on the same 800 examples are:

| Category | IGA | LoRA |
|---|---:|---:|
| Wrong on clean evidence | 160.67 | 142.67 |
| Wrong clean and with eight backgrounds | 129.33 | 114.67 |
| Correct clean, then wrong with backgrounds | 45.00 | 52.33 |
| Wrong clean, then correct with backgrounds | 31.33 | 28.00 |
| Total errors with eight backgrounds | 174.33 | 167.00 |

For IGA, 74.2% of eight-background errors were already errors on the clean prompt. The corresponding shares for the earlier IGA 3k and selection-A states are 75.0% and 72.4%. Persistent errors need not preserve the same wrong label. The partition identities are:

```text
clean_errors   = persistent_errors + corrected_clean_errors
context_errors = persistent_errors + new_errors
context_errors - clean_errors = new_errors - corrected_clean_errors
```

These counts describe behavior, not a neural cause. IGA is active on clean inputs, and annotation ambiguity or inconsistency remains possible. The raw smaller accuracy drop with backgrounds must not be interpreted as proven robustness.

## Compare the same clean-correct examples

Restricting to targets that both methods classify correctly on clean input gives common-clean denominators of 619, 622 and 628 for the three paired initializations. These are model/example observations, not 1,869 independent unique targets. With eight backgrounds, IGA makes 40, 41 and 41 errors on those subsets; LoRA makes 43, 33 and 41.

The pooled conditional error rates are 6.53% for IGA and 6.26% for LoRA. The difference is +0.27 percentage points with post-hoc nominal 95% interval [−1.32, +1.90]. This does not establish a robustness advantage.

## Clean-input label breakdown

| Inherited class | IGA | LoRA | Examples per state |
|---|---:|---:|---:|
| Supported | 93.67% | 93.50% | 400 |
| Refuted | 64.17% | 69.33% | 200 |
| Not enough information | 68.17% | 72.33% | 200 |

The observed deficit is concentrated in contradiction and insufficient-evidence cases rather than supported claims. This suggests testing unsupported-detail acceptance and contradiction discrimination after label quality is reviewed. It does not establish an architecture-specific explanation.

## Annotation concern and review status

One example describes the HD+ resolution of 1600 by 900 pixels in a 16:9 aspect ratio as 900p in both claim and evidence, but has an inherited insufficient-information label. All six selection-B IGA/LoRA states classify it as supported. This is a candidate annotation inconsistency, not a corrected gold label or an estimate of overall label noise. No labels or scores have been changed.

A blinded packet of all 800 clean examples and a separate 135-case diagnostic collection were prepared locally. Reviewer judgment fields remain blank. The packet is not distributed in this source repository, and no independent human adjudication is claimed.

## Next decision

Review the clean-evidence labels without revealing model outcomes to reviewers. If the deficit remains on unambiguous examples, test contradiction and unsupported-detail supervision using separate training sources and the same supervision opportunity for independent LoRA. Do not treat this recommendation as an executed experiment, a revised model, or a positive result.
