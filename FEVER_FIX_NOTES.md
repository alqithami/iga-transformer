# FEVER loader fix

This package loads FEVER v1.0 from the original `fever.ai` JSONL URLs through the Hugging Face `json` builder rather than `load_dataset("fever", "v1.0", ...)`.

Reason: recent Hugging Face Datasets versions reject legacy dataset scripts on the Hub, which causes:

```text
RuntimeError: Dataset scripts are no longer supported, but found fever.py
```

Default FEVER usage in the matrix remains:

- training/calibration from `train.jsonl`, hash-partitioned into train/dev;
- final evaluation from `shared_task_dev.jsonl` / `labelled_dev`;
- labelled examples only; unlabelled rows are skipped.

No paper result should be reported from unlabelled FEVER splits.
