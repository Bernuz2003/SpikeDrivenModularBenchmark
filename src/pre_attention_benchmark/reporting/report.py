from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from pre_attention_benchmark.metrics.pareto import pareto_front


def generate_run_report(out_dir: str | Path, summary: dict, layer_df: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    _plot_layerwise(out_dir, layer_df)
    _plot_spike_density_timestep(out_dir, layer_df)
    lines = [f"# Pre-Attention Benchmark Run Report — {summary.get('run_id')}\n"]
    lines.append('## Configuration\n')
    for key in ['dataset', 'encoder', 'feature_extractor', 'attention', 'head']:
        lines.append(f"- **{key}**: `{summary.get(key)}`")
    lines.append('\n## Main metrics\n')
    for key in [
        'accuracy_top1',
        'loss_val',
        'best_accuracy_top1',
        'profiled_checkpoint',
        'total_params',
        'total_params_profiled',
        'input_spike_density',
        'input_spike_count_mean_per_sample',
        'weighted_output_firing_rate',
        'mean_layer_output_firing_rate',
        'max_layer_output_firing_rate',
        'profiled_layers',
        'profiled_binary_layers',
    ]:
        lines.append(f"- **{key}**: {summary.get(key)}")
    if summary.get('robustness_score') is not None:
        lines.append(f"- **robustness_score**: {summary.get('robustness_score')}")
        lines.append(f"- **temporal_shuffle_drop**: {summary.get('temporal_shuffle_drop')}")
    lines.append('\n## Module metadata\n')
    for key in ['encoder_metadata', 'feature_extractor_metadata', 'head_metadata']:
        lines.append(f"### {key}\n")
        lines.append('```json')
        lines.append(json.dumps(summary.get(key, {}), indent=2, ensure_ascii=False))
        lines.append('```')
    lines.append('\n## Layer-wise excerpt\n')
    if layer_df.empty:
        lines.append('No layer-wise activity metrics were collected.\n')
    else:
        cols = [
            'layer_name',
            'module_type',
            'profile_scope',
            'output_shape',
            'params',
            'is_binary_input',
            'is_binary_output',
            'input_firing_rate',
            'output_firing_rate',
            'output_spike_count',
            'burstiness',
            'time_steps',
        ]
        lines.append(_df_to_markdown(layer_df[[c for c in cols if c in layer_df.columns]].head(30)))
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

    bar('output_firing_rate', 'firing_rate_layers.png', 'Layer-wise output firing rate')
    bar('output_spike_count', 'spike_count_layers.png', 'Layer-wise output spike count')


def _plot_spike_density_timestep(out_dir: Path, layer_df: pd.DataFrame) -> None:
    if layer_df.empty or 'spike_density_timestep' not in layer_df.columns:
        return
    plt.figure(figsize=(7, 4))
    plotted = 0
    for _, row in layer_df.iterrows():
        raw = row.get('spike_density_timestep')
        try:
            values = json.loads(raw) if isinstance(raw, str) else []
        except Exception:
            values = []
        if not values:
            continue
        label = str(row['layer_name']).replace('feature_extractor.', 'FE.').replace('stages.', 's.')
        plt.plot(range(len(values)), values, label=label[:48])
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


def aggregate_reports(runs_dir: str | Path, out_dir: str | Path, dataset: str | None = None) -> None:
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in runs_dir.rglob('metrics_summary.json'):
        try:
            rows.append(json.loads(path.read_text(encoding='utf-8')))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if dataset and not df.empty and 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() == dataset.lower()].copy()
    if df.empty:
        (out_dir / 'PreAttentionBenchmark_Report.md').write_text(
            '# Pre-Attention Benchmark Aggregate Report\n\nNo runs found.\n',
            encoding='utf-8',
        )
        return
    df.to_csv(out_dir / 'summary.csv', index=False)
    pf = pareto_front(df)
    pf.to_csv(out_dir / 'pareto_front.csv', index=False)

    def scatter(x: str, y: str, fname: str):
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

    scatter('total_params', 'accuracy_top1', 'accuracy_params.png')
    scatter('input_spike_density', 'accuracy_top1', 'accuracy_input_density.png')
    scatter('weighted_output_firing_rate', 'accuracy_top1', 'accuracy_output_activity.png')

    md = ['# Pre-Attention Benchmark Aggregate Report\n']
    show_cols = [
        c
        for c in [
            'run_id',
            'dataset',
            'encoder',
            'feature_extractor',
            'head',
            'accuracy_top1',
            'loss_val',
            'total_params',
            'input_spike_density',
            'weighted_output_firing_rate',
            'robustness_score',
        ]
        if c in df.columns
    ]
    md.append('## Runs\n')
    md.append(_df_to_markdown(df[show_cols].sort_values('accuracy_top1', ascending=False)))
    md.append('\n## Ranking by accuracy\n')
    md.append(_df_to_markdown(df[show_cols].sort_values('accuracy_top1', ascending=False).head(10)))
    if 'weighted_output_firing_rate' in df.columns:
        md.append('\n## Ranking by output activity\n')
        md.append(_df_to_markdown(df[show_cols].sort_values('weighted_output_firing_rate', ascending=True).head(10)))
    md.append('\n## Activity trade-off front\n')
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
        evidence = 'Trade-off over accuracy, parameter count, input density and output activity'
        risk = 'Confirm on full real DVS runs and robustness probes'
        rows.append([label, candidates, selected or 'TBD', evidence, risk])
    return _df_to_markdown(pd.DataFrame(rows, columns=['Design choice', 'Candidates tested', 'Selected candidate(s)', 'Evidence', 'Risk']))
