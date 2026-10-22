# Standalone IGA research update — October 2026

Published 22 October 2026; latest completed analysis: 21 October 2026.

## Current conclusion

The fixed-budget data-coverage improvement observed on the inspected development examples did **not** clearly confirm on previously unused target pages. These studies do not establish standalone IGA superiority over the independent LoRA comparator. The error diagnosis shows that most remaining IGA errors were already present on clean evidence. This motivates checking clean-evidence labels and contradiction/insufficient-evidence behavior before another suppression change.

This update concerns standalone normalized grouped-radius-12 IGA in Qwen2.5-7B-Instruct. IGA and LoRA are independent models, not IGA on a LoRA parent. The older package and composition workflows elsewhere in this repository remain historical implementations; they must not be presented as the exact implementation of these experiments.

## Evidence index

- [Coverage and replication findings](FINDINGS.md): initial development result and source-separated replication, with unfavorable results retained.
- [Error diagnosis](ERROR_DIAGNOSIS.md): persistent clean errors, background-induced errors, common-clean comparison, and annotation-review limitations.
- [Machine-readable study summary](coverage_results.json): selected means and prespecified primary contrasts, not full predictions.
- [Machine-readable error analysis](error_diagnosis.json): historical point estimates and explicitly post-hoc intervals.
- [Historical audit summary](AUDIT_HISTORY.json): what was checked on the original runs; this is not a new execution.
- [Executed scientific-source hashes](EXECUTED_SOURCE_BINDINGS.json): exact hashes from the verified replication handoff. Hash publication is not source-code synchronization.
- [Publication scope](PUBLICATION_SCOPE.md): included materials, intentionally excluded assets, and the remaining workstation-only files.

## Portable offline point-estimate check

Python 3.10 or later; standard library only. Supply the original complete replication archive retained with the research artifacts:

```bash
python scripts/analyze_iga_error_transitions.py \
  --archive /path/to/IGA_COVERAGE_REPLICATION_V1_REVIEW.zip \
  --output /path/to/new_error_transition_report.json
```

The utility requires archive SHA-256 `1f8b058bf8983451c5d8c5d582d229a0aa9b1c1c0024710eccc3d10fd3a56938`. It verifies every archive payload, reconstructs six-format semantic probabilities, pairs clean and background predictions by source ID, and writes transition counts, family means and common-clean point estimates. It never loads model weights, trains, changes labels, contacts a server, or overwrites an existing report. The input archive is not distributed in this repository.

This is a **new portable publication utility**, not the exact historical 21 October program. It does not reproduce the historical bootstrap intervals, refit temperatures, or redo the training/source-exclusion audit. The historical program's hash is retained in `AUDIT_HISTORY.json` so the distinction is explicit.

Run the local synthetic and published-summary checks:

```bash
python -m unittest discover -s tests -p 'test_error_transitions_publication.py' -v
```

Ten checks passed during publication preparation. They validate synthetic transitions, probability arithmetic, and algebraic identities in the published extracts. The new utility was **not** run end-to-end on the 111 MB original archive during this update because that archive and its workstation were unavailable. Do not confuse these tests with the prior full experimental audit or model training.

## Next decision

The proposed next step is blinded clean-evidence annotation review, then a separately controlled training study of contradiction and unsupported-detail discrimination if the deficit remains. No review judgments have been supplied, labels have not been corrected, and no new training follows automatically from this update.
