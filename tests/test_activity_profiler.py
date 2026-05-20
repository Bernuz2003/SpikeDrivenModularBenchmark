from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pre_attention_benchmark.metrics.activity import ActivityProfiler, summarize_activity_metrics
from pre_attention_benchmark.models.blocks import TDMaxPool
from pre_attention_benchmark.models.lif import MultiStepLIF


def test_activity_profiler_records_conv_input_activity_and_structure():
    profiler = ActivityProfiler(nn.Identity(), run_id='test', time_steps=4)
    conv = nn.Conv2d(2, 4, kernel_size=3, padding=1, groups=1)
    x = torch.ones(3, 2, 8, 8)
    y = conv(x)
    rec = profiler._make_record('conv', conv, (x,), y)
    assert rec is not None
    assert rec.module_type == 'Conv2d'
    assert rec.is_binary_input is True
    assert rec.input_firing_rate == 1.0
    assert rec.is_binary_output is False
    assert rec.kernel_size == '[3, 3]'
    assert rec.padding == '[1, 1]'
    assert rec.in_channels == 2
    assert rec.out_channels == 4
    assert rec.time_steps == 4


def test_activity_profiler_temporal_density_for_map_and_tokens():
    profiler = ActivityProfiler(nn.Identity(), run_id='test')
    maps = torch.zeros(2, 3, 4, 5, 5)
    maps[:, 1] = 1
    mean, std, burst, per_timestep = profiler._temporal_density(maps, True, scope='leaf')
    assert per_timestep == [0.0, 1.0, 0.0]
    assert mean > 0
    assert std > 0
    assert burst > 0

    tokens = torch.zeros(2, 3, 16, 5)
    tokens[:, 2] = 1
    _, _, _, token_timestep = profiler._temporal_density(tokens, True, scope='feature_extractor')
    assert token_timestep == [0.0, 0.0, 1.0]


def test_activity_profiler_tracks_temporal_pool_without_hardware_buffer_metrics():
    profiler = ActivityProfiler(nn.Identity(), run_id='test', time_steps=3)
    pool = TDMaxPool(kernel_size=2, stride=2)
    x = torch.ones(2, 3, 4, 8, 8)
    y = pool(x)
    rec = profiler._make_record('pool', pool, (x,), y)
    assert rec is not None
    assert rec.module_type == 'TDMaxPool'
    assert rec.is_binary_input is True
    assert rec.is_binary_output is True
    assert rec.output_firing_rate == 1.0
    assert rec.kernel_size == '[2]'
    assert rec.stride == '[2]'


def test_activity_summary_keeps_only_basic_activity_metrics():
    df = pd.DataFrame(
        [
            {
                'layer_name': 'lif0',
                'params': 0,
                'is_binary_output': True,
                'output_spike_count': 10.0,
                'output_numel': 20,
                'output_firing_rate': 0.5,
            },
            {
                'layer_name': 'fc',
                'params': 33,
                'is_binary_output': False,
                'output_spike_count': 0.0,
                'output_numel': 10,
                'output_firing_rate': 0.0,
            },
        ]
    )
    summary = summarize_activity_metrics(df)
    assert summary['total_params_profiled'] == 33
    assert summary['mean_layer_output_firing_rate'] == 0.5
    assert summary['weighted_output_firing_rate'] == 0.5
    assert summary['profiled_binary_layers'] == 1


def test_activity_profiler_collects_hooks_end_to_end():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lif = MultiStepLIF()
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            y = self.lif(x)
            return self.head(y.mean(dim=1))

    model = Tiny()
    profiler = ActivityProfiler(model, run_id='test', time_steps=3)
    profiler.attach()
    profiler.enable()
    _ = model(torch.ones(2, 3, 4))
    profiler.disable()
    profiler.close()
    df = profiler.to_dataframe()
    assert set(df['module_type']) == {'MultiStepLIF', 'Linear'}
