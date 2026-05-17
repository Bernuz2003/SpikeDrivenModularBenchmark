from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from pre_attention_benchmark.metrics.pareto import pareto_front


def generate_run_report(out_dir: str | Path, summary: dict, layer_df: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    # I plot sono best-effort: se non ci sono metriche layer-wise il report resta
    # comunque leggibile e utile per debug.
    _plot_layerwise(out_dir, layer_df)
    _plot_spike_density_timestep(out_dir, layer_df)
    lines = []
    lines.append(f"# Pre-Attention Benchmark Run Report — {summary.get('run_id')}\n")
    lines.append('## Configuration\n')
    for k in ['dataset', 'encoder', 'feature_extractor', 'attention', 'head']:
        lines.append(f"- **{k}**: `{summary.get(k)}`")
    lines.append('\n## Main metrics\n')
    for k in ['accuracy_top1', 'loss_val', 'total_params', 'total_sops_proxy', 'total_spike_count', 'mean_firing_rate', 'total_weight_mem_bits', 'total_activation_mem_bits', 'total_state_mem_bits', 'max_buffer_mem_bits']:
        lines.append(f"- **{k}**: {summary.get(k)}")
    if summary.get('robustness_score') is not None:
        lines.append(f"- **robustness_score**: {summary.get('robustness_score')}")
        lines.append(f"- **temporal_shuffle_drop**: {summary.get('temporal_shuffle_drop')}")
    lines.append('\n## Module metadata\n')
    for key in ['encoder_metadata', 'feature_extractor_metadata', 'head_metadata']:
        val = summary.get(key, {})
        lines.append(f"### {key}\n")
        lines.append('```json')
        lines.append(json.dumps(val, indent=2, ensure_ascii=False))
        lines.append('```')
    lines.append('\n## Layer-wise excerpt\n')
    if layer_df.empty:
        lines.append('No layer-wise metrics were collected.\n')
    else:
        cols = ['layer_name', 'module_type', 'output_shape', 'is_binary_output', 'firing_rate', 'burstiness', 'sops_proxy', 'state_mem_bits', 'hf_ratio']
        lines.append(_df_to_markdown(layer_df[cols].head(30)))
    (out_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as Markdown without making tabulate a hard runtime dependency."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        pass
    if df.empty:
        return ''
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[c]).replace('\n', ' ').replace('|', '\\|') for c in df.columns])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt(cells):
        return '| ' + ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)) + ' |'
    sep = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    return '\n'.join([fmt(headers), sep] + [fmt(row) for row in rows])


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


def _plot_spike_density_timestep(out_dir: Path, layer_df: pd.DataFrame) -> None:
    if layer_df.empty or 'spike_density_timestep' not in layer_df.columns:
        return
    plt.figure(figsize=(7, 4))
    plotted = 0
    for _, row in layer_df.iterrows():
        raw = row.get('spike_density_timestep')
        try:
            vals = json.loads(raw) if isinstance(raw, str) else []
        except Exception:
            vals = []
        if not vals:
            continue
        # Limitiamo le curve: con tutte le layer il grafico diventa una matassa.
        label = str(row['layer_name']).replace('feature_extractor.', 'FE.').replace('stages.', 's.')
        plt.plot(range(len(vals)), vals, label=label[:48])
        plotted += 1
        if plotted >= 8:
            break
    if plotted == 0:
        plt.close()
        return
    plt.xlabel('timestep')
    plt.ylabel('spike density')
    plt.title('Spike density per timestep')
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / 'spike_density_timestep.png', dpi=160)
    plt.close()


def aggregate_reports(runs_dir: str | Path, out_dir: str | Path) -> None:
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in runs_dir.glob('*/metrics_summary.json'):
        try:
            rows.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            # Un run parziale non deve bloccare l'aggregato; resta nei log della run.
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        (out_dir / 'PreAttentionBenchmark_Report.md').write_text('# Pre-Attention Benchmark Aggregate Report\n\nNo runs found.\n', encoding='utf-8')
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
    scatter('total_activation_mem_bits', 'accuracy_top1', 'pareto_accuracy_activation_mem.png')
    scatter('max_buffer_mem_bits', 'accuracy_top1', 'pareto_accuracy_buffer_mem.png')

    md = ['# Pre-Attention Benchmark Aggregate Report\n']
    md.append('## Runs\n')
    show_cols = [c for c in ['run_id','dataset','encoder','feature_extractor','head','accuracy_top1','loss_val','total_sops_proxy','mean_firing_rate','total_spike_count','total_state_mem_bits','total_activation_mem_bits','max_buffer_mem_bits','robustness_score'] if c in df.columns]
    md.append(_df_to_markdown(df[show_cols].sort_values('accuracy_top1', ascending=False)))
    md.append('\n## Ranking by accuracy\n')
    md.append(_df_to_markdown(df[show_cols].sort_values('accuracy_top1', ascending=False).head(10)))
    if 'mean_firing_rate' in df.columns:
        md.append('\n## Ranking by firing rate\n')
        md.append(_df_to_markdown(df[show_cols].sort_values('mean_firing_rate', ascending=True).head(10)))
    if 'total_sops_proxy' in df.columns:
        md.append('\n## Ranking by SOPs proxy\n')
        md.append(_df_to_markdown(df[show_cols].sort_values('total_sops_proxy', ascending=True).head(10)))
    md.append('\n## Pareto front\n')
    md.append(_df_to_markdown(pf[show_cols].sort_values('accuracy_top1', ascending=False)))
    md.append('\n## Decision table draft\n')
    md.append(_decision_table(df, pf))
    (out_dir / 'PreAttentionBenchmark_Report.md').write_text('\n'.join(md), encoding='utf-8')


def _decision_table(df: pd.DataFrame, pf: pd.DataFrame) -> str:
    rows = []
    for field, label in [('encoder', 'Encoding'), ('feature_extractor', 'Feature extractor'), ('head', 'Head')]:
        if field not in df.columns:
            continue
        candidates = ', '.join(str(x) for x in sorted(df[field].dropna().unique()))
        selected = ', '.join(str(x) for x in sorted(pf[field].dropna().unique())) if field in pf.columns and not pf.empty else 'TBD'
        evidence = 'Pareto aggregate over accuracy, SOPs, state memory and spike count'
        risk = 'Requires confirmation on full real DVS runs' if str(df['dataset'].iloc[0]).lower() == 'cifar10_dvs' else 'Check robustness and temporal sensitivity before the attention phase'
        rows.append([label, candidates, selected or 'TBD', evidence, risk])
    return _df_to_markdown(pd.DataFrame(rows, columns=['Design choice', 'Candidates tested', 'Selected candidate(s)', 'Evidence', 'Risk']))
