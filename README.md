# Inhibitory-Gate Attention

Inhibitory-Gate Attention (IGA) studies learned nonnegative attention costs in decoder-only language models:

```text
A' = softmax(S - Gamma),  Gamma >= 0
```

## Research status — 22 October 2026

The latest completed work is the **standalone IGA coverage replication and clean/background error diagnosis**. The early development-set coverage improvement did not clearly confirm on previously unused target pages. These results **do not establish IGA superiority over independent LoRA**.

Start with the [October research update](docs/research/2026-10/README.md), [complete findings](docs/research/2026-10/FINDINGS.md), and [error diagnosis](docs/research/2026-10/ERROR_DIAGNOSIS.md). Machine-readable summaries and historical audit identities are included. Negative and inconclusive results are retained.

The October studies compare frozen-backbone grouped-radius-12 IGA against an independent query/key LoRA model. They are **not** the older LoRA-parent composition workflow. Their source-scoped classification prompts identify the admissible evidence passage; the findings should not be generalized to natural retrieval or open-ended generation without further evidence.

The latest diagnosis finds that 74.2% of selection-B IGA errors with eight backgrounds were already errors on clean evidence. On the common clean-correct targets, the observed background-induced error-rate difference does not establish a robustness advantage. A candidate annotation inconsistency is recorded, but no inherited label has been changed and independent human adjudication remains pending.

## What this repository update contains

```text
src/iga_llm/                              Existing implementation, preserved
configs/                                 Existing experiment configurations
scripts/                                 Existing commands plus portable error reanalysis
tests/test_error_transitions_publication.py
                                         Synthetic and published-summary checks
docs/research/2026-10/                    Latest findings, numeric extracts, audit history
README_LEGACY.md                          Previous README, preserved byte-for-byte
```

This is a research-results and offline-analysis update, **not a full synchronization of the research workstation**. The workstation was offline when this update was assembled from verified exported handoffs. The latest historical training modules, full raw archives, trained states, and blinded-review packet are not added by this commit. See [publication scope](docs/research/2026-10/PUBLICATION_SCOPE.md) for the exact boundary. Existing `src/iga_llm` code must not be mistaken for the exact implementation of every October experiment.

## Offline error-transition analysis

Python 3.10 or later; no GPU or third-party packages are required for the new utility. Supply the retained complete replication archive:

```bash
python scripts/analyze_iga_error_transitions.py \
  --archive /path/to/IGA_COVERAGE_REPLICATION_V1_REVIEW.zip \
  --output /path/to/new_error_transition_report.json
```

The utility validates archive identity and payload hashes, reconstructs semantic probabilities from stored logits, and reports paired clean/background transitions. It does not train, load a language model, alter labels, or overwrite earlier reports. It reproduces point estimates, not the historical bootstrap intervals or full source-history audit.

```bash
python -m unittest discover -s tests -p 'test_error_transitions_publication.py' -v
```

The ten local publication tests passed. They are synthetic and algebraic checks, not a fresh execution of the original experiment. End-to-end use of the new utility on the complete archive remains to be checked where that artifact is available.

## Existing installation and historical workflows

The [previous README](README_LEGACY.md) retains the installation commands, model configurations, dense IGA, LoRA, LoRA+IGA composition, and evaluation workflows. Those commands and their historical environment have not been silently rewritten. For the existing package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements_gpu.txt
python -m pip install -e .
bash scripts/static_check.sh
```

Further historical information is in [reproducibility documentation](docs/REPRODUCIBILITY.md) and [artifact documentation](docs/RESULTS_ARTIFACTS.md). Pretrained weights, benchmark corpora, trained adapters, private access files, and unpublished manuscripts are not included in the new update.

## License

The software is released under the MIT License. See [LICENSE](LICENSE). Third-party models and datasets retain their own licenses.
