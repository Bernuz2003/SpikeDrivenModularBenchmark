from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pre_attention_benchmark.metrics.collector import MetricsCollector, summarize_layer_metrics


def test_state_memory_is_counted_per_sample():
    collector = MetricsCollector(nn.Identity(), run_id='test')
    module = nn.Module()
    module.last_state_shape = (4, 3, 2)
    assert collector._state_bits_per_sample(module) == 3 * 2 * 32


def test_conv_buffer_uses_input_bit_width():
    collector = MetricsCollector(nn.Identity(), run_id='test')
    conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
    inp = torch.ones(4, 2, 16, 16)
    out = torch.randn(4, 4, 16, 16)
    assert collector._buffer_bits(conv, inp, out, is_bin=False) == 2 * 3 * 16


def test_summary_does_not_emit_misleading_global_spike_or_activation_totals():
    df = pd.DataFrame(
        [
            {
                'layer_name': 'lif0',
                'params': 0,
                'sops_proxy': 10.0,
                'spike_count': 5.0,
                'firing_rate': 0.5,
                'weight_mem_bits': 0,
                'activation_mem_bits': 64,
                'state_mem_bits_per_sample': 32,
                'buffer_mem_bits': 16,
                'module_type': 'MultiStepLIF',
                'mac_dense_ops': 0.0,
                'ac_sparse_ops': 0.0,
                'and_ops': 0.0,
                'compare_ops': 10.0,
                'shift_ops': 0.0,
                'maxpool_compare_ops': 0.0,
            },
            {
                'layer_name': 'identity',
                'params': 0,
                'sops_proxy': 0.0,
                'spike_count': 5.0,
                'firing_rate': 0.5,
                'weight_mem_bits': 0,
                'activation_mem_bits': 64,
                'state_mem_bits_per_sample': 0,
                'buffer_mem_bits': 0,
                'module_type': 'IdentityAttention',
                'mac_dense_ops': 0.0,
                'ac_sparse_ops': 0.0,
                'and_ops': 0.0,
                'compare_ops': 0.0,
                'shift_ops': 0.0,
                'maxpool_compare_ops': 0.0,
            },
        ]
    )
    summary = summarize_layer_metrics(df)
    assert 'total_spike_count' not in summary
    assert 'mean_firing_rate' not in summary
    assert 'total_activation_mem_bits' not in summary
    assert 'total_state_mem_bits' not in summary
    assert summary['max_state_mem_bits_per_sample'] == 32
