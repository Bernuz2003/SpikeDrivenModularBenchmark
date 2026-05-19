# Pre-Attention Benchmark

Framework modulare per valutare componenti spike-driven prima dell'introduzione di un modulo di attention reale: encoder event-to-spike, feature extractor/tokenizer, head di classificazione, metriche hardware-aware e report Pareto.

## Cosa Implementa

- Interfaccia eventi uniforme: tempo, coordinate, polarita e label.
- Supporto reale via `tonic` per `CIFAR10-DVS` e `DVS128 Gesture`.
- Encoder binari: fixed time-bin e fixed event-count, con soglia configurabile su istogramma locale.
- Cache encoded per run in memoria (`uint8`/`bool`), cosi lo stesso sample non viene ricodificato a ogni epoca.
- Feature extractor spike-driven con varianti convolutional, residual MS, depthwise separable, hierarchical tokenizer e dual-path.
- Attention forzata a `identity`, per misurare solo la qualita dei componenti pre-attention.
- Head terminali: spatio-temporal average, SpikeVision-like spatial pooling e class-neuron accumulator.
- Metriche layer-wise: firing rate, spike count, burstiness, memory proxy, SOPs proxy, classi operative e high-frequency ratio.
- Report run-level, aggregazione Pareto e decision table.
- Test automatici sui validator e sulle parti eseguibili senza dataset esterni.

## Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,datasets]"
```

Se sul server serve una build PyTorch CUDA specifica, installare prima PyTorch seguendo la versione CUDA disponibile, poi rieseguire l'installazione editable del progetto.

## Path Locali

Le config versionate usano placeholder come `${PREATTN_DATA_ROOT}` e `${PREATTN_RUNS_ROOT}`. I path reali non vanno committati: mettili in un file locale ignorato da Git.

```bash
mkdir -p configs/local
cp configs/PATHS.example.yaml configs/local/paths.yaml
```

Poi modifica `configs/local/paths.yaml`:

```yaml
PREATTN_DATA_ROOT: /path/to/datasets/tonic
PREATTN_RUNS_ROOT: /path/to/runs/pre_attention_benchmark
PREATTN_REPORTS_ROOT: /path/to/reports/pre_attention_benchmark
```

In alternativa puoi esportare le stesse variabili d'ambiente; hanno precedenza sul file locale.

## Quick Check Reale

Prima di lanciare sweep lunghi:

```bash
python -m pre_attention_benchmark.training.train --config configs/pre_attention_benchmark/real/cifar10_dvs_quick_check.yaml
```

Output principale:

```text
${PREATTN_RUNS_ROOT}/cifar10_dvs_quick_check/
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

## Test Automatici

```bash
pytest -q
```

## Esecuzione Sweep

Ogni directory contiene config YAML eseguibili in sequenza:

```bash
python scripts/run_sweep.py --config-dir configs/pre_attention_benchmark/encoding_sweep
python scripts/run_sweep.py --config-dir configs/pre_attention_benchmark/feature_extractor_sweep
python scripts/run_sweep.py --config-dir configs/pre_attention_benchmark/head_sweep
python scripts/run_sweep.py --config-dir configs/pre_attention_benchmark/robustness
```

Per run ufficiali e riproducibili puoi copiare le config in `configs/local/` solo se vuoi modificare iperparametri o subset. I path reali restano in `configs/local/paths.yaml`.

Quando hai preparato una copia locale con la stessa struttura puoi lanciare tutto in sequenza:

```bash
bash scripts/run_pre_attention_suite.sh configs/local/pre_attention_benchmark
```

## Aggregazione Report

```bash
python scripts/aggregate_reports.py
```

Il report aggregato produce `summary.csv`, `pareto_front.csv`, grafici Pareto accuracy/costo e una decision table preliminare per scegliere i candidati da portare nella fase attention.

Ogni run usa un solo dataset, definito da `dataset.name` nella YAML. Non c'e merge automatico tra CIFAR10-DVS e DVS128 Gesture. I risultati restano separati nel campo `dataset` e nei path sotto `${PREATTN_RUNS_ROOT}/<dataset>/...`.

Per aggregare solo un dataset:

```bash
python scripts/aggregate_reports.py --dataset cifar10_dvs
python scripts/aggregate_reports.py --dataset dvs128_gesture --out-dir ${PREATTN_REPORTS_ROOT}/dvs128_gesture
```

## Log Durante Run Lunghe

Il training stampa sempre un riepilogo a fine epoca. Per avere output periodico anche dentro epoche lunghe:

```yaml
training:
  log_interval_batches: 50
```