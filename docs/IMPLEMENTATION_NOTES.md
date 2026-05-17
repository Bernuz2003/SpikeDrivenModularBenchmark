# Implementation Notes — Milestone 1

## Mapping requisiti → codice

| Requisito | Implementazione |
|---|---|
| FR-1 Dataset management | `m1_benchmark.datasets`, dataset sintetico + adapter opzionale tonic |
| FR-2 Encoding registry | `build_encoder` in `datasets/__init__.py`, encoders in `encoders/binary.py` |
| FR-3 Feature extractor registry | `build_feature_extractor` in `models/feature_extractors.py` |
| FR-4 Residual validator | `validate_milestone1_config` + `validate_model_static` |
| FR-5 Identity attention fixed | `build_attention`, config validator |
| FR-6 Head registry | `build_head` in `models/heads.py` |
| FR-7 Metrics logger layer-wise | `metrics/collector.py`, token-aware temporal density, operator-class summary |
| FR-8 Reproducibility | `training/train.py` saves config YAML/JSON, split indices, checkpoints, history, environment and Git metadata |
| FR-9 Report generation | `reporting/report.py`, `scripts/aggregate_reports.py`, Pareto plots and decision-table draft |
| FR-10 Failure tests | `tests/test_validators.py`, `tests/test_smoke.py` |

## Concetti importanti

- I tensori hidden comunicati tra feature extractor, identity attention e head sono validati come binari.
- Le somme real-valued sono ammesse solo come correnti/stati interni al blocco, seguite immediatamente da LIF.
- La head produce logits real-valued solo come terminal boundary.
- I SOPs sono proxy diagnostici, non numeri post-sintesi.
- Robustness e temporal sensitivity sono valutate sullo stesso checkpoint, non tramite training con perturbazioni, quando `evaluation.robustness.enabled=true`.
- Gli output hidden dichiarati con `emits_hidden_spikes=true` sono validati dinamicamente come binari.
