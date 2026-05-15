from __future__ import annotations

import argparse
from pathlib import Path
import sys
import platform
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from m1_benchmark.config import load_config, save_config, validate_milestone1_config, get_git_commit
from m1_benchmark.datasets import build_datasets
from m1_benchmark.models import build_model
from m1_benchmark.models.validators import validate_model_static, validate_hidden_outputs
from m1_benchmark.metrics import MetricsCollector, summarize_layer_metrics
from m1_benchmark.reporting.report import generate_run_report
from m1_benchmark.training.utils import seed_everything, get_device, accuracy_top1, save_json, count_params, configure_torch_runtime


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for spikes, labels in loader:
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


def run(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_milestone1_config(cfg)
    configure_torch_runtime(cfg.get('training', {}).get('torch_num_threads', 1))
    seed_everything(int(cfg['experiment']['seed']))
    device = get_device(cfg['training'].get('device', 'auto'))
    out_dir = Path(cfg['logging']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / 'config.yaml')

    train_ds, val_ds, encoder = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, num_workers=int(cfg['training'].get('num_workers', 0)))
    val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, num_workers=int(cfg['training'].get('num_workers', 0)))

    model = build_model(cfg, encoder, device)
    validate_model_static(model)
    # Runtime quality gate on one sample.
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

    best_acc = -1.0
    best_state = None
    history = []
    epochs = int(cfg['training']['epochs'])
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        print(f"epoch={epoch:03d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({'model': best_state, 'cfg': cfg, 'history': history}, out_dir / 'checkpoint_best.pt')

    torch.save({'model': model.state_dict(), 'cfg': cfg, 'history': history}, out_dir / 'checkpoint_final.pt')

    collector = MetricsCollector(model, run_id=cfg['experiment']['name'])
    collector.attach()
    collector.reset()
    val_loss, val_acc = evaluate(model, val_loader, device, collector=collector, profile_batches=int(cfg['training'].get('profile_batches', 1)))
    layer_df = collector.save(out_dir / 'metrics_layerwise.csv', out_dir / 'profile.json')
    collector.close()
    metric_summary = summarize_layer_metrics(layer_df)

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
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        **metric_summary,
        'temporal_shuffle_drop': None,
        'robustness_score': None,
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
