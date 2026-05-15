from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from m1_benchmark.metrics.pareto import pareto_front


def generate_run_report(out_dir: str | Path, summary: dict, layer_df: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    _plot_layerwise(out_dir, layer_df)
    lines = []
    lines.append(f"# Milestone 1 Run Report — {summary.get('run_id')}\n")
    lines.append('## Configuration\n')
    for k in ['dataset', 'encoder', 'feature_extractor', 'attention', 'head']:
        lines.append(f"- **{k}**: `{summary.get(k)}`")
    lines.append('\n## Main metrics\n')
    for k in ['accuracy_top1', 'loss_val', 'total_params', 'total_sops_proxy', 'total_spike_count', 'mean_firing_rate', 'total_weight_mem_bits', 'total_activation_mem_bits', 'total_state_mem_bits', 'max_buffer_mem_bits']:
        lines.append(f"- **{k}**: {summary.get(k)}")
    lines.append('\n## Layer-wise excerpt\n')
    if layer_df.empty:
        lines.append('No layer-wise metrics were collected.\n')
    else:
        cols = ['layer_name', 'module_type', 'output_shape', 'is_binary_output', 'firing_rate', 'burstiness', 'sops_proxy', 'state_mem_bits', 'hf_ratio']
        lines.append(layer_df[cols].head(30).to_markdown(index=False))
    (out_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def _plot_layerwise(out_dir: Path, layer_df: pd.DataFrame) -> None:
    if layer_df.empty:
        return
    # Keep plots readable by showing the first 40 tracked layers.
    df = layer_df.head(40).copy()
    labels = [str(x).replace('feature_extractor.', 'FE.').replace('stages.', 's.') for x in df['layer_name']]
    def bar(col: str, fname: str, title: str):
        if col not in df.columns:
            return
        plt.figure(figsize=(max(6, 0.28 * len(df)), 4))
        plt.bar(range(len(df)), df[col].fillna(0.0).astype(float))
        plt.xticks(range(len(df)), labels, rotation=90, fontsize=6)
        plt.ylabel(col)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()
    bar('firing_rate', 'firing_rate_layers.png', 'Layer-wise firing rate')
    bar('sops_proxy', 'sops_layers.png', 'Layer-wise SOPs proxy')
    if 'hf_ratio' in df.columns and df['hf_ratio'].notna().any():
        bar('hf_ratio', 'high_frequency_layers.png', 'Layer-wise high-frequency ratio')


def aggregate_reports(runs_dir: str | Path, out_dir: str | Path) -> None:
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in runs_dir.glob('*/metrics_summary.json'):
        try:
            rows.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        (out_dir / 'Milestone1_Report.md').write_text('# Milestone 1 Aggregate Report\n\nNo runs found.\n', encoding='utf-8')
        return
    df.to_csv(out_dir / 'summary.csv', index=False)
    pf = pareto_front(df)
    pf.to_csv(out_dir / 'pareto_front.csv', index=False)

    def scatter(x, y, fname):
        if x not in df.columns or y not in df.columns:
            return
        plt.figure()
        plt.scatter(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{y} vs {x}')
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()

    scatter('total_sops_proxy', 'accuracy_top1', 'pareto_accuracy_sops.png')
    scatter('total_state_mem_bits', 'accuracy_top1', 'pareto_accuracy_state_mem.png')
    scatter('total_spike_count', 'accuracy_top1', 'pareto_accuracy_spike_count.png')

    md = ['# Milestone 1 Aggregate Report\n']
    md.append('## Runs\n')
    show_cols = [c for c in ['run_id','dataset','encoder','feature_extractor','head','accuracy_top1','total_sops_proxy','mean_firing_rate','total_state_mem_bits'] if c in df.columns]
    md.append(df[show_cols].sort_values('accuracy_top1', ascending=False).to_markdown(index=False))
    md.append('\n## Pareto front\n')
    md.append(pf[show_cols].sort_values('accuracy_top1', ascending=False).to_markdown(index=False))
    (out_dir / 'Milestone1_Report.md').write_text('\n'.join(md), encoding='utf-8')
