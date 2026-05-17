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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternativa equivalente:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Per i dataset reali installare anche `tonic`:

```bash
python -m pip install -e ".[dev,datasets]"
```

oppure:

```bash
python -m pip install -r requirements-datasets.txt
python -m pip install -e .
```

Se serve una build specifica di PyTorch per CUDA, installare prima PyTorch seguendo la versione CUDA del server e poi installare il progetto.

## Smoke test

```bash
python -m m1_benchmark.training.train --config configs/milestone1/smoke/synthetic_smoke.yaml
```

Output atteso:

```text
runs/milestone1/synthetic_smoke/
  config.yaml
  config.json
  dataset_split.json
  environment.json
  model_metadata.json
  input_spike_profile.json
  input_spike_density.csv
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

Il report aggregato produce `summary.csv`, `pareto_front.csv`, grafici Pareto accuracy/costo e una decision table preliminare per Milestone 2.

Per abilitare robustness e temporal sensitivity su una run, impostare:

```yaml
evaluation:
  robustness:
    enabled: true
```

La valutazione usa lo stesso checkpoint addestrato su dati puliti e misura event drop, temporal jitter, polarity drop, timestep shuffle ed early accuracy.

Durante i run lunghi il training stampa un riepilogo a fine epoca e, se `training.log_interval_batches` è maggiore di zero, anche ogni N batch:

```yaml
training:
  log_interval_batches: 25
```

Gli output sono flushati, quindi `screen -L` e `tail -f` li mostrano mentre il processo è in esecuzione.

## Dataset reali

Per usare `CIFAR10-DVS` o `DVS128 Gesture`, installare `tonic` e impostare `dataset.root` nel file YAML:

```yaml
dataset:
  name: cifar10_dvs
  root: /path/to/datasets/tonic
```

Config di partenza:

```bash
python -m m1_benchmark.training.train --config configs/milestone1/real/cifar10_dvs_smoke.yaml
python -m m1_benchmark.training.train --config configs/milestone1/real/dvs128_gesture_smoke.yaml
```

Il framework resta utilizzabile anche senza dataset reali tramite `synthetic_dvs`, utile per testare training loop, validator, logging e report. Per la struttura consigliata dei dataset vedere `docs/DATASETS.md`.

## Principi implementati

1. Hidden communication binaria: gli output principali dei moduli hidden devono essere spike in `{0,1}`.
2. Attention Milestone 1 = Identity; ogni altra attention fallisce.
3. Residual non-MS fallisce.
4. Encoding count-valued non binarizzato fallisce.
5. Terminal readout può produrre logits real-valued, ma viene marcato come confine terminale.

## Nota metodologica

Questo repository non cerca accuracy SOTA. Serve a costruire una base sperimentale tracciabile e hardware-aware per decidere quali componenti pre-attention portare nella Milestone 2.
