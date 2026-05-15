# Milestone 1 — Spike-Driven Modular Benchmark

Framework modulare per la **Milestone 1** della tesi: valutazione quantitativa pre-attention di encoding, feature extractor/tokenizer e classification head per event-based vision, con attention fissata a `Identity` e pipeline hidden quanto più possibile spike-driven.

## Cosa implementa

- Dataset/event interface standard: `(event_time, x, y, polarity, label)`.
- Dataset sintetico per smoke test (`synthetic_dvs`).
- Supporto opzionale a `tonic` per `CIFAR10-DVS` e `DVS128 Gesture`.
- Encoding spike-driven binari: fixed time-bin, fixed event-count, voxel-grid binary, temporal subsampling.
- Feature extractor spike-driven FE0–FE5, con FE6 opzionale dual-path.
- Residual validator: sono ammessi solo `none` o `ms`; `sew` fallisce.
- Attention bloccata a `identity` per Milestone 1.
- Head spike-driven con readout terminale esplicitamente contabilizzato.
- Logger layer-wise: firing rate, spike count, burstiness, memory proxy, SOPs proxy, operator classes, high-frequency ratio.
- Report automatico run-level e aggregazione Pareto.
- Test automatici per smoke run e quality gates.

## Installazione rapida

```bash
cd milestone1_framework
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

`torch`, `numpy`, `pyyaml`, `pandas` e `matplotlib` sono sufficienti per gli smoke test. `tonic` è opzionale per caricare dataset DVS reali.

## Smoke test

```bash
python -m m1_benchmark.training.train --config configs/milestone1/smoke/synthetic_smoke.yaml
```

Output atteso:

```text
runs/milestone1/synthetic_smoke/
  config.yaml
  metrics_layerwise.csv
  metrics_summary.json
  profile.json
  checkpoint_best.pt
  checkpoint_final.pt
  report.md
```

## Test automatici

```bash
pytest -q
```

## Esecuzione sweep

```bash
python scripts/run_sweep.py --config-dir configs/milestone1/encoding_sweep
python scripts/run_sweep.py --config-dir configs/milestone1/feature_extractor_sweep
python scripts/run_sweep.py --config-dir configs/milestone1/head_sweep
```

## Aggregazione report

```bash
python scripts/aggregate_reports.py --runs-dir runs/milestone1 --out-dir reports/milestone1
```

## Dataset reali

Per usare `CIFAR10-DVS` o `DVS128 Gesture`, installare `tonic` e impostare `dataset.root` nel file YAML:

```yaml
dataset:
  name: cifar10_dvs
  root: /path/to/datasets
```

Il framework resta utilizzabile anche senza dataset reali tramite `synthetic_dvs`, utile per testare training loop, validator, logging e report.

## Principi implementati

1. Hidden communication binaria: gli output principali dei moduli hidden devono essere spike in `{0,1}`.
2. Attention Milestone 1 = Identity; ogni altra attention fallisce.
3. Residual non-MS fallisce.
4. Encoding count-valued non binarizzato fallisce.
5. Terminal readout può produrre logits real-valued, ma viene marcato come confine terminale.

## Nota metodologica

Questo repository non cerca accuracy SOTA. Serve a costruire una base sperimentale tracciabile e hardware-aware per decidere quali componenti pre-attention portare nella Milestone 2.
