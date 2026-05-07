# Status

This package replaces the earlier scaffold with a complete local experiment runner.

Completed:

- Fixed training API and removed failure masking.
- Added disjoint train/dev/eval data preparation.
- Added paper-aligned pair-MLP IGA with entropy gate and nonnegative inhibition.
- Added frozen-backbone training with trainable IGA modules only.
- Added model/seed/method/config/dataset hashes to prediction rows.
- Added MC baselines, generation baselines, ablations, latency runs, calibration bins, paired deltas, and LaTeX table generation.
- Added smoke, M4 Max, and two-model paper matrix scripts.

Still dependent on local resources:

- Full 7B/8B experiments require downloading/opening the Hugging Face model checkpoints.
- Llama 3 requires Hugging Face access approval and login.
- DoLa support depends on the installed Transformers version and model compatibility; any unsupported baseline error is logged per row rather than hidden.
- Free-form generation factuality is label-parsed, not judged by an external verifier. Treat generation baselines as secondary unless an external judge is added.
