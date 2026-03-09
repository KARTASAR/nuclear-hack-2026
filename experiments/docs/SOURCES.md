# Sources (Experiments Layer)

This file defines the source map for the `experiments/` documentation package and provides a publication-oriented claim-to-reference matrix.

## 1) Internal Project Sources

Core reading order for implementation/reproducibility in the `experiments/` layer:

1. [experiments/README.md](../README.md)
2. [experiments/docs/REPRODUCIBILITY.md](./REPRODUCIBILITY.md)
3. [experiments/docs/EXPERIMENTS_APPENDIX.md](./EXPERIMENTS_APPENDIX.md)
4. [experiments/docs/general/10.5.Physically-grounded-Raman-XAI-smart-ticket.md](./general/10.5.Physically-grounded-Raman-XAI-smart-ticket.md)
5. [experiments/docs/general/10.6.Domain-validation-and-physics-aware-Raman-XAI-smart-ticket.md](./general/10.6.Domain-validation-and-physics-aware-Raman-XAI-smart-ticket.md)
6. [experiments/docs/general/10.8.Raman-XAI-pipeline-usage.md](./general/10.8.Raman-XAI-pipeline-usage.md)

## 2) Formal Bibliography

Formal bibliography with DOI/venue/year mapping:

- [experiments/docs/BIBLIOGRAPHY.md](./BIBLIOGRAPHY.md)

## 3) Claim-to-Reference Matrix

| claim/result block | internal evidence (docs/tests/artifacts) | bibliography keys |
|---|---|---|
| XAI architecture is implemented as shared, reusable runtime components | [10.5.Physically-grounded-Raman-XAI-smart-ticket.md](./general/10.5.Physically-grounded-Raman-XAI-smart-ticket.md), [test_explainability_schema.py](../tests/test_explainability_schema.py), [test_explain_batch_cli.py](../tests/test_explain_batch_cli.py), [test_cli_entrypoints.py](../tests/test_cli_entrypoints.py) | N/A (implementation-state engineering evidence) |
| Domain-validation metrics and reporting are implemented and executable | [10.6.Domain-validation-and-physics-aware-Raman-XAI-smart-ticket.md](./general/10.6.Domain-validation-and-physics-aware-Raman-XAI-smart-ticket.md), [test_band_registry.py](../tests/test_band_registry.py), [test_band_aggregation.py](../tests/test_band_aggregation.py), [test_physics_metrics.py](../tests/test_physics_metrics.py), [test_artifact_masks.py](../tests/test_artifact_masks.py), [test_explainer_concordance.py](../tests/test_explainer_concordance.py), [test_validate_explanations_cli.py](../tests/test_validate_explanations_cli.py) | B01, B02, B03, B07, B12 |
| Reproducibility command surface and artifact policy are documented and stable | [REPRODUCIBILITY.md](./REPRODUCIBILITY.md), [README.md](../README.md), [test_cli_entrypoints.py](../tests/test_cli_entrypoints.py) | N/A (operational reproducibility evidence) |
| Preprocessing rationale and spectral correction choices are literature-backed | [4.1.Internet-review-raman-preprocessing.md](./general/4.1.Internet-review-raman-preprocessing.md), [4.2.Shortlist-from-literature.md](./general/4.2.Shortlist-from-literature.md), [7.1.Tech-stack-and-uv-setup.md](./general/7.1.Tech-stack-and-uv-setup.md) | B01, B02, B04, B05, B06, B08, B09, B10, B11, B12 |
| DL architecture exploration and outcome tracking are historically documented | [10.1.DL-architectures-from-literature.md](./general/10.1.DL-architectures-from-literature.md), [10.2.DL-wave-results.md](./general/10.2.DL-wave-results.md), [10.3.Prioritized-hypotheses-after-dl-breakthrough.md](./general/10.3.Prioritized-hypotheses-after-dl-breakthrough.md), [10.4.Full-DL-architecture-expansion-plan.md](./general/10.4.Full-DL-architecture-expansion-plan.md), [docs/artifacts/*](./artifacts/) | B13, B14 |

Note: test references above describe repository evidence linkage; pass/fail status can depend on local environment completeness (for example optional/native ML dependencies).

## 4) Scope Note

- `experiments/docs/general/*` contains historical depth and implementation snapshots.
- Canonical current-state operational interface for external readers is:
  1. [experiments/README.md](../README.md)
  2. [experiments/docs/REPRODUCIBILITY.md](./REPRODUCIBILITY.md)
  3. [experiments/docs/SOURCES.md](./SOURCES.md)
  4. [experiments/docs/BIBLIOGRAPHY.md](./BIBLIOGRAPHY.md)
