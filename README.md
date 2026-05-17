# Pre-Attention Benchmark

Framework modulare per valutare componenti spike-driven prima dell'introduzione di un modulo di attention reale: encoder event-to-spike, feature extractor/tokenizer, head di classificazione, metriche hardware-aware e report Pareto.

## Cosa Implementa

- Interfaccia eventi uniforme: tempo, coordinate, polarita e label.
- Supporto reale via `tonic` per `CIFAR10-DVS` e `DVS128 Gesture`.
- Encoder binari: fixed time-bin, fixed event-count, binary voxel-grid e temporal subsampling.
- Feature extractor spike-driven con varianti convolutional, residual MS, depthwise separable, hierarchical tokenizer e dual-path.
- Attention forzata a `identity`, per misurare solo la qualita dei componenti pre-attention.
- Head spike-driven con readout terminale esplicito.
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

## Quick Check Reale

Prima di lanciare sweep lunghi, copia una config reale in `configs/local/` e imposta `dataset.root`:

```bash
mkdir -p configs/local
cp configs/pre_attention_benchmark/real/cifar10_dvs_quick_check.yaml configs/local/cifar10_dvs_quick_check.yaml
```

Nel file copiato:

```yaml
dataset:
  name: cifar10_dvs
  root: /path/to/datasets/tonic
```

Poi:

```bash
python -m pre_attention_benchmark.training.train --config configs/local/cifar10_dvs_quick_check.yaml
```

Output principale:

```text
runs/pre_attention_benchmark/cifar10_dvs_quick_check/
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

Per run ufficiali e riproducibili e meglio copiare le config in `configs/local/`, correggere `dataset.root` e, se necessario, portare `logging.output_dir` su una directory persistente del server.

Quando hai preparato una copia locale con la stessa struttura puoi lanciare tutto in sequenza:

```bash
bash scripts/run_pre_attention_suite.sh configs/local/pre_attention_benchmark
```

## Aggregazione Report

```bash
python scripts/aggregate_reports.py --runs-dir runs/pre_attention_benchmark --out-dir reports/pre_attention_benchmark
```

Il report aggregato produce `summary.csv`, `pareto_front.csv`, grafici Pareto accuracy/costo e una decision table preliminare per scegliere i candidati da portare nella fase attention.

## Log Durante Run Lunghe

Il training stampa sempre un riepilogo a fine epoca. Per avere output periodico anche dentro epoche lunghe:

```yaml
training:
  log_interval_batches: 25
```

Le `print` sono flushate, quindi funzionano bene con `screen -L`, `tail -f` e log file su server remoti.

## Principi

1. La comunicazione hidden deve restare binaria: spike in `{0,1}`.
2. L'attention rimane `identity`.
3. I residual ammessi sono `none` e `ms`; `sew` fallisce.
4. Gli encoding count-valued devono essere binarizzati.
5. I logits real-valued sono ammessi solo nel readout terminale.
