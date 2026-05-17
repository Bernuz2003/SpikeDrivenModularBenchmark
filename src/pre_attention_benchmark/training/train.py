from __future__ import annotations

import argparse
from pathlib import Path
import sys
import platform
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pre_attention_benchmark.config import (
    has_unresolved_placeholders,
    load_config,
    save_config,
    save_config_json,
    validate_pre_attention_config,
    get_git_commit,
    get_git_metadata,
)
from pre_attention_benchmark.datasets import build_datasets, dataset_metadata
from pre_attention_benchmark.models import build_model
from pre_attention_benchmark.models.validators import validate_model_static, validate_hidden_outputs
from pre_attention_benchmark.metrics import MetricsCollector, summarize_layer_metrics
from pre_attention_benchmark.reporting.report import generate_run_report
from pre_attention_benchmark.training.utils import seed_everything, get_device, accuracy_top1, save_json, count_params, configure_torch_runtime


def train_one_epoch(model, loader, optimizer, device, epoch: int | None = None, log_interval_batches: int = 0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    total_batches = len(loader) if hasattr(loader, '__len__') else None
    for batch_idx, (spikes, labels) in enumerate(loader, start=1):
        spikes = spikes.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(spikes)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        bs = labels.numel()
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_top1(logits.detach(), labels) * bs
        n += bs
        if log_interval_batches and batch_idx % int(log_interval_batches) == 0:
            total = f'/{total_batches}' if total_batches is not None else ''
            ep = f'epoch={epoch:03d} ' if epoch is not None else ''
            print(
                f"{ep}batch={batch_idx}{total} train_loss={total_loss / max(1, n):.4f} train_acc={total_acc / max(1, n):.3f}",
                flush=True,
            )
    return total_loss / max(1, n), total_acc / max(1, n)


def evaluate(model, loader, device, collector: MetricsCollector | None = None, profile_batches: int = 1):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    profiled = 0
    with torch.no_grad():
        for spikes, labels in loader:
            spikes = spikes.to(device)
            labels = labels.to(device)
            # Profilare tutti i batch rende le run lunghe inutilmente pesanti;
            # bastano pochi batch stabili per confrontare i candidati.
            if collector is not None and profiled < profile_batches:
                collector.enable()
                profiled += 1
            else:
                if collector is not None:
                    collector.disable()
            logits = model(spikes, validate_hidden_binary=True)
            loss = F.cross_entropy(logits, labels)
            bs = labels.numel()
            total_loss += float(loss.item()) * bs
            total_acc += accuracy_top1(logits, labels) * bs
            n += bs
    if collector is not None:
        collector.disable()
    return total_loss / max(1, n), total_acc / max(1, n)


def make_loader(ds, cfg: dict, shuffle: bool = False) -> DataLoader:
    num_workers = int(cfg['training'].get('num_workers', 0))
    return DataLoader(
        ds,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(cfg['training'].get('pin_memory', torch.cuda.is_available())),
        persistent_workers=num_workers > 0,
    )


def input_spike_profile(loader, max_batches: int = 4) -> dict:
    total_spikes = 0.0
    total_numel = 0
    per_timestep_sum = None
    batches = 0
    samples = 0
    for spikes, _ in loader:
        # Profilo dell'input prima del modello: aiuta a capire se un encoder e
        # troppo denso o se svuota alcuni timestep.
        batches += 1
        samples += int(spikes.shape[0])
        total_spikes += float(spikes.sum().item())
        total_numel += int(spikes.numel())
        dens = spikes.float().mean(dim=tuple(i for i in range(spikes.dim()) if i != 1))
        per_timestep_sum = dens if per_timestep_sum is None else per_timestep_sum + dens
        if batches >= max_batches:
            break
    if batches == 0:
        return {'input_spike_count': 0, 'input_spike_density': 0, 'input_spike_density_per_timestep': [], 'profiled_samples': 0}
    per_timestep = (per_timestep_sum / batches).cpu().tolist()
    return {
        'input_spike_count': total_spikes / batches,
        'input_spike_density': total_spikes / max(1, total_numel),
        'input_spike_density_per_timestep': [float(v) for v in per_timestep],
        'profiled_samples': samples,
    }


def save_input_density_csv(profile: dict, out_path: Path) -> None:
    lines = ['timestep,spike_density']
    for i, v in enumerate(profile.get('input_spike_density_per_timestep', [])):
        lines.append(f'{i},{v}')
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def robustness_specs(cfg: dict, T: int) -> list[tuple[str, dict]]:
    rob = cfg.get('evaluation', {}).get('robustness', {})
    specs: list[tuple[str, dict]] = []
    for rate in rob.get('event_drop_rates', []):
        specs.append((f'event_drop_{int(float(rate) * 100):02d}', {'event_drop': float(rate)}))
    for jitter in rob.get('temporal_jitter', []):
        specs.append((f'temporal_jitter_{str(jitter).replace(".", "_")}', {'temporal_jitter': float(jitter)}))
    for drop in rob.get('polarity_drop', []):
        specs.append((f'polarity_drop_{int(float(drop) * 100):02d}', {'polarity_drop': float(drop)}))
    if rob.get('timestep_shuffle', True):
        specs.append(('timestep_shuffle', {'timestep_shuffle': True}))
    if rob.get('early_accuracy', True):
        for t in range(1, max(1, T)):
            specs.append((f'early_truncation_{t}', {'early_truncation': t}))
    return specs


def evaluate_robustness(model, cfg: dict, device: torch.device, clean_acc: float) -> dict:
    rob = cfg.get('evaluation', {}).get('robustness', {})
    if not rob.get('enabled', False):
        return {
            'temporal_shuffle_drop': None,
            'robustness_score': None,
            'robustness': {},
            'early_accuracy': {},
        }
    T = int(cfg['encoder']['T'])
    results = {}
    drops = []
    early = {}
    for name, aug in robustness_specs(cfg, T):
        print(f"robustness_eval={name} augmentations={aug}", flush=True)
        # La robustezza riusa i pesi addestrati su dati puliti e cambia solo la
        # validation: cosi misuriamo degradazione, non un nuovo training.
        _, val_ds, _ = build_datasets(cfg, train_augmentations_override={}, val_augmentations_override=aug)
        val_loader = make_loader(val_ds, cfg, shuffle=False)
        loss, acc = evaluate(model, val_loader, device)
        drop = float(clean_acc - acc)
        rec = {'loss_val': float(loss), 'accuracy_top1': float(acc), 'drop_from_clean': drop, 'augmentations': aug}
        results[name] = rec
        if name.startswith('early_truncation_'):
            early[name.rsplit('_', 1)[-1]] = float(acc)
        else:
            drops.append(max(0.0, drop))
    temporal_shuffle_drop = results.get('timestep_shuffle', {}).get('drop_from_clean')
    robustness_score = 1.0 - (sum(drops) / max(1, len(drops)))
    return {
        'temporal_shuffle_drop': temporal_shuffle_drop,
        'robustness_score': float(robustness_score),
        'robustness': results,
        'early_accuracy': early,
    }


def ensure_runtime_paths_resolved(cfg: dict) -> None:
    paths = {
        'dataset.root': cfg.get('dataset', {}).get('root'),
        'logging.output_dir': cfg.get('logging', {}).get('output_dir'),
    }
    unresolved = [name for name, value in paths.items() if has_unresolved_placeholders(value)]
    if unresolved:
        joined = ', '.join(unresolved)
        raise RuntimeError(
            f"Unresolved path placeholders in {joined}. "
            "Create configs/local/paths.yaml or export PREATTN_DATA_ROOT/PREATTN_RUNS_ROOT."
        )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int):
    name = str(cfg.get('training', {}).get('scheduler', 'cosine')).lower()
    if name in {'none', 'null', 'false', 'off'}:
        return None
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
            eta_min=float(cfg['training'].get('min_lr', 1e-5)),
        )
    raise ValueError(f'Unknown scheduler: {name}')


def run(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_pre_attention_config(cfg)
    ensure_runtime_paths_resolved(cfg)
    configure_torch_runtime(cfg.get('training', {}).get('torch_num_threads', 1))
    seed_everything(int(cfg['experiment']['seed']))
    device = get_device(cfg['training'].get('device', 'auto'))
    out_dir = Path(cfg['logging']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / 'config.yaml')
    save_config_json(cfg, out_dir / 'config.json')

    train_ds, val_ds, encoder = build_datasets(cfg)
    save_json(dataset_metadata(train_ds, val_ds, encoder), out_dir / 'dataset_split.json')
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    input_profile = input_spike_profile(val_loader, max_batches=max(1, int(cfg['training'].get('profile_batches', 1))))
    save_json(input_profile, out_dir / 'input_spike_profile.json')
    save_input_density_csv(input_profile, out_dir / 'input_spike_density.csv')

    model = build_model(cfg, encoder, device)
    validate_model_static(model)
    # Quality gate runtime su un sample: intercetta subito output hidden non binari.
    sample, _ = train_ds[0]
    validate_hidden_outputs(model, sample, device)

    optimizer_name = cfg['training'].get('optimizer', 'adamw').lower()
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training'].get('weight_decay', 1e-4)))
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg['training']['lr']), momentum=0.9)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    epochs = int(cfg['training']['epochs'])
    scheduler = build_scheduler(optimizer, cfg, epochs)
    best_acc = -1.0
    best_state = None
    history = []
    start_epoch = 1
    resume_from = cfg['training'].get('resume_from')
    if resume_from:
        # Resume minimale ma completo: modello, optimizer, history e best score.
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        history = list(ckpt.get('history', []))
        best_acc = float(ckpt.get('best_acc', ckpt.get('best_accuracy_top1', -1.0)))
        start_epoch = int(ckpt.get('epoch', len(history))) + 1
    for epoch in range(start_epoch, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            log_interval_batches=int(cfg['training'].get('log_interval_batches', 25) or 0),
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]['lr'])
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'lr': current_lr})
        print(f"epoch={epoch:03d} lr={current_lr:.6g} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt = {'model': best_state, 'optimizer': optimizer.state_dict(), 'cfg': cfg, 'history': history, 'epoch': epoch, 'best_acc': best_acc}
            if scheduler is not None:
                ckpt['scheduler'] = scheduler.state_dict()
            torch.save(ckpt, out_dir / 'checkpoint_best.pt')

    final_ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'cfg': cfg, 'history': history, 'epoch': epochs, 'best_acc': best_acc}
    if scheduler is not None:
        final_ckpt['scheduler'] = scheduler.state_dict()
    torch.save(final_ckpt, out_dir / 'checkpoint_final.pt')

    collector = MetricsCollector(model, run_id=cfg['experiment']['name'])
    collector.attach()
    collector.reset()
    print('profiling=layerwise_metrics', flush=True)
    val_loss, val_acc = evaluate(model, val_loader, device, collector=collector, profile_batches=int(cfg['training'].get('profile_batches', 1)))
    layer_df = collector.save(out_dir / 'metrics_layerwise.csv', out_dir / 'profile.json')
    collector.close()
    metric_summary = summarize_layer_metrics(layer_df)
    robustness = evaluate_robustness(model, cfg, device, clean_acc=val_acc)
    save_json({'clean_accuracy_top1': float(val_acc), **robustness}, out_dir / 'robustness.json')
    git_meta = get_git_metadata()
    environment = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'git': git_meta,
        'device': str(device),
    }
    save_json(environment, out_dir / 'environment.json')
    save_json(model.metadata, out_dir / 'model_metadata.json')

    summary = {
        'run_id': cfg['experiment']['name'],
        'dataset': cfg['dataset']['name'],
        'encoder': cfg['encoder']['name'],
        'feature_extractor': cfg['model']['feature_extractor']['name'],
        'attention': cfg['model']['attention']['name'],
        'head': cfg['model']['head']['name'],
        'accuracy_top1': float(val_acc),
        'loss_val': float(val_loss),
        'best_accuracy_top1': float(best_acc),
        'total_params': int(count_params(model)),
        'git_commit': get_git_commit(),
        'git_dirty': git_meta.get('dirty'),
        'git_remote': git_meta.get('remote'),
        'git_branch': git_meta.get('branch'),
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'encoder_metadata': model.metadata.get('encoder', {}),
        'feature_extractor_metadata': model.metadata.get('feature_extractor', {}),
        'head_metadata': model.metadata.get('head', {}),
        **input_profile,
        **metric_summary,
        'temporal_shuffle_drop': robustness['temporal_shuffle_drop'],
        'robustness_score': robustness['robustness_score'],
        'early_accuracy': robustness['early_accuracy'],
    }
    save_json(summary, out_dir / 'metrics_summary.json')
    save_json({'history': history}, out_dir / 'history.json')
    generate_run_report(out_dir, summary, layer_df)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
