# Coverage experiments: retain the development finding and the unconfirmed replication

Source: the completed 18 October code/run-validity handoff and the independently audited 20 October coverage-replication handoff. Numeric values are reported as observed; no new training, model selection or inference is represented here.

## Fixed experimental recipe

Standalone normalized radius-12 IGA and independent rank-10 query/key LoRA adapt the frozen Qwen2.5-7B-Instruct backbone. They train 1,183,010 and 1,126,400 parameters respectively in attention blocks 18–27. Their fixed peak learning rates are 0.00015 and 0.0005. Ordinary three-class cross-entropy, the numerical casting, training exposure and 3,000-update budget are matched across the coverage comparison. This is not an optimal all-projection LoRA comparison.

The small-data arm presents 3,000 unique examples four times. Each expanded arm presents 12,000 unique examples once: the original 3,000 plus 9,000 additional target examples. Every model receives 12,000 presentations and 3,000 optimizer updates, not four times more computation. The first 750 updates are shared exactly. Subsequent matched slots preserve label-pair stratum, background passages/count, label mapping, and target position; additional target evidence is the intervention.

Each family is the mean of three separately scored model initializations. Predictions from different seeds or models are not ensembled. Training selections A and B share the original 3k and slot backgrounds; their 9k additions are source-disjoint. They are not two independent complete datasets or six independent initialization seeds.

## Initial inspected-development study

The twelve new fits completed on 18 October. These scores use the same 400 previously inspected engineering-development examples per condition.

| Method / coverage | Clean | Four backgrounds | Eight backgrounds | Calibrated Brier, eight (lower is better) |
|---|---:|---:|---:|---:|
| IGA, 3k × 4 | 81.08% | 79.75% | 79.58% | 0.299682 |
| IGA, 12k selection A × 1 | 81.83% | 80.08% | 80.75% | 0.286519 |
| LoRA, 3k × 4 | 80.50% | 80.17% | 80.17% | 0.288202 |
| LoRA, 12k selection A × 1 | 80.58% | 79.83% | 80.00% | 0.288531 |

IGA's eight-background coverage gain is +1.17 percentage points, nominal 95% interval [−0.75, +3.33]. Per-initialization gains are +2.25, +0.75 and +0.50 points. Its Brier change is −0.01316, nominal 95% [−0.02630, −0.00106]. These are descriptive development findings, unadjusted for the entire comparison family and sequential exploration. IGA minus 12k LoRA is +0.75 points, interval [−1.75, +3.33]; this does not establish superiority.

## Source-separated replication

Six new fits use training selection B. All eighteen states (twelve reused, six new) are evaluated on 800 examples from 400 previously unused official-DEV target pages. Calibration uses 200 examples from a separate 100 pages. All models are frozen before calibration; all temperatures are frozen before confirmation scoring. The source audit checks normalized page, claim, evidence and identifier exclusions. Backgrounds intentionally reuse earlier slot passages. This is not official TEST, a new benchmark, or pretraining-corpus deduplication.

| Fixed method / coverage | Clean | Four backgrounds | Eight backgrounds | Calibrated Brier, eight |
|---|---:|---:|---:|---:|
| IGA, 3k × 4 | 80.12% | 78.46% | 78.12% | 0.321495 |
| LoRA, 3k × 4 | 81.46% | 79.37% | 79.08% | 0.311226 |
| IGA, 12k selection A | 80.29% | 78.17% | 78.38% | 0.316937 |
| LoRA, 12k selection A | 82.04% | 79.88% | 79.00% | 0.308753 |
| IGA, 12k selection B | 79.92% | 78.79% | 78.21% | 0.318005 |
| LoRA, 12k selection B | 82.17% | 79.38% | 79.13% | 0.310722 |

The two prespecified primary eight-background contrasts are:

- **IGA B minus 3k IGA: +0.08 percentage points**, marginal 97.5% paired initialization/page interval [−1.12, +1.25]. The per-initialization effects are −0.375, +0.375 and +0.250 points.
- **IGA B minus LoRA B: −0.92 points**, marginal 97.5% interval [−2.75, +0.96]. Per-initialization effects are −0.375, −1.125 and −1.250 points.

The 97.5% marginal levels account for the two predeclared primary accuracy comparisons under the approximate bootstrap procedure; they do not adjust for all historical searches. Selection A also yields only +0.25 points over 3k IGA on these new targets, nominal 95% [−1.08, +1.50]. This is not explained solely by an unfavorable second training selection.

## Decision and limitations

Retain both studies. The early improvement did not clearly confirm on the new pages. Neither primary comparison establishes superiority, equivalence or non-inferiority. Lower IGA mean scores do not alone establish universal LoRA superiority either. Do not replace the current precision-matched LoRA scores with a weaker historical baseline.

There are three adapter initializations, two added-source selections, one confirmation cohort, and inherited labels without new human adjudication. These results do not establish convergence, optimal hyperparameters, arbitrary data-selection robustness, or natural-distractor robustness. The source-scoped prompt explicitly identifies the admissible evidence passage.

No threshold was relaxed, favorable seed selected, third draw launched, or manuscript result rewritten by this publication update.
