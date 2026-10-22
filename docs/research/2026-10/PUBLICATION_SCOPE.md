# Publication scope and reproducibility boundary

## This commit includes

- A public research-status update covering the initial coverage study, the source-separated replication, and the subsequent error diagnosis.
- Selected numerical extracts and historical audit/provenance records, including negative and inconclusive results.
- A new, standard-library offline utility for reconstructing clean/background point estimates from the original replication ZIP, with ten synthetic/published-summary tests.

The new utility is distinct from the historical 21 October analysis program. Its tests passed during publication preparation; it has not been executed against the complete raw archive in this publication session. Historical audits are reported as historical, not claimed to have been rerun here.

## This is not a full workstation synchronization

The project workstation was offline at publication time. This commit uses previously exported verified handoffs. It does not claim that every local research script or unpublished change has been synchronized. In particular, the full historical error-analysis program, per-example transitions, diagnostic cases, and 800-example blinded review packet remain in the retained research artifacts. Existing training implementations in `src/iga_llm` are preserved, not silently substituted for the newer experimental implementation.

The new utility requires `IGA_COVERAGE_REPLICATION_V1_REVIEW.zip` (111,015,622 bytes), SHA-256 `1f8b058bf8983451c5d8c5d582d229a0aa9b1c1c0024710eccc3d10fd3a56938`. This archive is not included. It contains 390 verified payloads and all eighteen compact fitted states. The earlier fixed-budget archive is `IGA_DATA_COVERAGE_V1_REVIEW.zip` (76,257,991 bytes), SHA-256 `a175cb01e3f8fc128ef82dcf4e249026623f9eb4ca1c678b5326a0c5869cb769`.

Publishing hashes permits identity checks for readers who already have the corresponding artifacts; it does not make absent artifacts available or establish complete public reproducibility.

## Excluded and preserved

No pretrained weights, trained adapters, raw benchmark corpora, raw prediction archives, blind-review answer keys, server launch/access helpers, SSH material, cloud account identifiers, tokens, private workstation paths, or unpublished manuscripts are added. Repository visibility and access settings are unchanged. No new model training, GPU inference, human label review, manuscript revision, or submission is performed by this update.

The anonymous manuscript and the coauthor's separate manuscript remain outside this publication update. Legacy repository workflows remain available for historical reference; their environment and model variants should not be confused with the October standalone experiments.
