from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import torch
from torch import nn
import pandas as pd

from .frequency import high_frequency_ratio
from pre_attention_benchmark.training.utils import tensor_is_binary


@dataclass
class LayerMetric:
    run_id: str
    layer_name: str
    module_type: str
    input_shape: str
    output_shape: str
    is_binary_output: bool
    spike_count: float
    firing_rate: float
    spike_density_mean: float
    spike_density_std: float
    spike_density_timestep: str
    burstiness: float
    params: int
    weight_mem_bits: int
    activation_mem_bits: int
    state_mem_bits: int
    buffer_mem_bits: int
    sops_proxy: float
    mac_dense_ops: float
    ac_sparse_ops: float
    and_ops: float
    compare_ops: float
    shift_ops: float
    maxpool_compare_ops: float
    hf_ratio: float | None


class MetricsCollector:
    """Forward-hook based layer-wise profiler.

    The metrics are proxy-level diagnostics, not post-synthesis FPGA numbers.
    """

    def __init__(self, model: nn.Module, run_id: str, bit_weights: int = 32, bit_membrane: int = 32) -> None:
        self.model = model
        self.run_id = run_id
        self.bit_weights = int(bit_weights)
        self.bit_membrane = int(bit_membrane)
        self.handles = []
        self.enabled = False
        self.records: list[LayerMetric] = []
        self.profile_layers: list[dict[str, Any]] = []

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if name == '':
                continue
            if self._should_track(module):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.records.clear()
        self.profile_layers.clear()

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _should_track(self, module: nn.Module) -> bool:
        if getattr(module, '_pre_attention_skip_metrics', False):
            return False
        return (
            getattr(module, 'track_metrics', False)
            or isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.MaxPool2d))
            or module.__class__.__name__.lower().endswith('lif')
        )

    def _hook(self, name: str):
        def fn(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if not self.enabled:
                return
            try:
                rec = self._make_record(name, module, inputs, output)
                if rec is not None:
                    self.records.append(rec)
                    self.profile_layers.append(self._profile_entry(name, module, inputs, output))
            except Exception as e:
                # Profiling must not silently corrupt a run, but during training
                # we prefer storing an explicit record rather than crashing hooks.
                raise RuntimeError(f'Metrics hook failed at layer {name}: {e}') from e
        return fn

    def _tensor(self, x: Any) -> torch.Tensor | None:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
            return x[0]
        return None

    def _make_record(self, name: str, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> LayerMetric | None:
        inp = self._tensor(inputs)
        out = self._tensor(output)
        if out is None:
            return None
        input_shape = str(list(inp.shape)) if inp is not None else '[]'
        output_shape = str(list(out.shape))
        is_bin = tensor_is_binary(out.detach())
        numel = int(out.numel())
        spike_count = float(out.detach().sum().item()) if is_bin else 0.0
        firing_rate = spike_count / max(1, numel) if is_bin else 0.0
        density_mean, density_std, burst, density_timestep = self._temporal_density(out.detach(), is_bin)
        params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        weight_bits = params * self.bit_weights
        act_bits = numel * (1 if is_bin else 32)
        state_bits = self._state_bits(module)
        buffer_bits = self._buffer_bits(module, inp, out, is_bin)
        # Sono proxy architetturali, non numeri da sintesi FPGA: servono per
        # confrontare candidati in modo consistente prima del deployment vero.
        ops = self._ops(module, inp, out, is_bin)
        hf = high_frequency_ratio(out.detach()) if is_bin else None
        return LayerMetric(
            run_id=self.run_id,
            layer_name=name,
            module_type=module.__class__.__name__,
            input_shape=input_shape,
            output_shape=output_shape,
            is_binary_output=is_bin,
            spike_count=spike_count,
            firing_rate=firing_rate,
            spike_density_mean=density_mean,
            spike_density_std=density_std,
            spike_density_timestep=json.dumps(density_timestep),
            burstiness=burst,
            params=params,
            weight_mem_bits=weight_bits,
            activation_mem_bits=act_bits,
            state_mem_bits=state_bits,
            buffer_mem_bits=buffer_bits,
            sops_proxy=ops['sops_proxy'],
            mac_dense_ops=ops['mac_dense_ops'],
            ac_sparse_ops=ops['ac_sparse_ops'],
            and_ops=ops['and_ops'],
            compare_ops=ops['compare_ops'],
            shift_ops=ops['shift_ops'],
            maxpool_compare_ops=ops['maxpool_compare_ops'],
            hf_ratio=hf,
        )

    def _temporal_density(self, out: torch.Tensor, is_bin: bool) -> tuple[float, float, float, list[float]]:
        if not is_bin or out.dim() < 3:
            return 0.0, 0.0, 0.0, []
        # [B,T,...] expected for spike tensors. Conv hooks may be [B*T,C,H,W]; those do not expose T.
        if out.dim() in (4, 5):
            if out.dim() == 5:
                dens = out.float().mean(dim=tuple(i for i in range(out.dim()) if i != 1))
            elif out.dim() == 4:
                dens = out.float().mean(dim=(0, 2, 3))
            mean = float(dens.mean().cpu())
            std = float(dens.std(unbiased=False).cpu())
            var = float(dens.var(unbiased=False).cpu())
            burst = var / (mean + 1e-12)
            return mean, std, burst, [float(v) for v in dens.detach().cpu().flatten()]
        return 0.0, 0.0, 0.0, []

    def _state_bits(self, module: nn.Module) -> int:
        shape = getattr(module, 'last_state_shape', None)
        if shape is None:
            return 0
        n = 1
        for s in shape:
            n *= int(s)
        return n * self.bit_membrane

    def _buffer_bits(self, module: nn.Module, inp: torch.Tensor | None, out: torch.Tensor, is_bin: bool) -> int:
        bit = 1 if is_bin else 32
        if isinstance(module, nn.Conv2d) and inp is not None:
            k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            # Proxy di line buffer streaming: canali in input * righe del kernel * larghezza.
            width = int(inp.shape[-1]) if inp.dim() >= 4 else 1
            return int(module.in_channels * k * width * bit)
        if isinstance(module, (nn.MaxPool2d,)) or module.__class__.__name__ == 'TDMaxPool':
            return int(out.numel() * bit)
        return 0

    def _ops(self, module: nn.Module, inp: torch.Tensor | None, out: torch.Tensor, is_bin: bool) -> dict[str, float]:
        z = dict(sops_proxy=0.0, mac_dense_ops=0.0, ac_sparse_ops=0.0, and_ops=0.0, compare_ops=0.0, shift_ops=0.0, maxpool_compare_ops=0.0)
        if isinstance(module, nn.Conv2d) and inp is not None:
            out_elems = out.numel()
            k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            per_out = (module.in_channels // module.groups) * k_h * k_w
            dense = float(out_elems * per_out)
            inp_binary = tensor_is_binary(inp.detach())
            if inp_binary:
                fr_in = float(inp.detach().sum().item() / max(1, inp.numel()))
                z['ac_sparse_ops'] = dense * fr_in
                z['sops_proxy'] = z['ac_sparse_ops']
            else:
                z['mac_dense_ops'] = dense
                z['sops_proxy'] = dense
        elif isinstance(module, nn.Linear) and inp is not None:
            dense = float(inp.numel() * module.out_features)
            z['mac_dense_ops'] = dense
            z['sops_proxy'] = dense
        elif isinstance(module, nn.MaxPool2d) or module.__class__.__name__ == 'TDMaxPool':
            kernel = getattr(module, 'kernel_size', 1)
            k_h, k_w = kernel if isinstance(kernel, tuple) else (kernel, kernel)
            z['maxpool_compare_ops'] = float(out.numel() * max(0, k_h * k_w - 1))
            z['sops_proxy'] = z['maxpool_compare_ops']
        elif module.__class__.__name__.lower().endswith('lif') or 'lif' in module.__class__.__name__.lower():
            z['compare_ops'] = float(out.numel())
            z['sops_proxy'] = z['compare_ops']
        return z

    def _profile_entry(self, name: str, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> dict[str, Any]:
        inp = self._tensor(inputs)
        out = self._tensor(output)
        entry = {
            'name': name,
            'type': module.__class__.__name__,
            'input_shape': list(inp.shape) if inp is not None else None,
            'output_shape': list(out.shape) if out is not None else None,
            'op_class': getattr(module, 'op_class', module.__class__.__name__),
            'requires_state': bool(getattr(module, 'requires_state', False)),
            'terminal_readout': bool(getattr(module, 'terminal_readout', False)),
            'emits_hidden_spikes': bool(getattr(module, 'emits_hidden_spikes', False)),
        }
        if isinstance(module, nn.Conv2d):
            entry.update({
                'kernel': list(module.kernel_size),
                'stride': list(module.stride),
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'groups': module.groups,
            })
            if inp is not None and inp.dim() >= 4:
                k_h = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                entry['buffer_shape_proxy'] = [module.in_channels, k_h, int(inp.shape[-1])]
        if module.__class__.__name__ == 'TDMaxPool' and out is not None:
            entry['buffer_shape_proxy'] = list(out.shape)
        if hasattr(module, 'last_membrane_range'):
            entry['membrane_range'] = getattr(module, 'last_membrane_range')
        weight = getattr(module, 'weight', None)
        if isinstance(weight, torch.Tensor):
            w = weight.detach()
            entry['weight_stats'] = {
                'min': float(w.min().cpu()) if w.numel() else 0.0,
                'max': float(w.max().cpu()) if w.numel() else 0.0,
                'mean': float(w.mean().cpu()) if w.numel() else 0.0,
                'std': float(w.std(unbiased=False).cpu()) if w.numel() else 0.0,
            }
        return entry

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def save(self, layerwise_path: str | Path, profile_path: str | Path) -> pd.DataFrame:
        df = self.to_dataframe()
        layerwise_path = Path(layerwise_path)
        layerwise_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(layerwise_path, index=False)
        profile = {'layers': self.profile_layers}
        profile_path = Path(profile_path)
        with profile_path.open('w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
        return df


def summarize_layer_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            'total_params_profiled': 0,
            'total_sops_proxy': 0,
            'total_spike_count': 0,
            'mean_firing_rate': 0,
            'total_weight_mem_bits': 0,
            'total_activation_mem_bits': 0,
            'total_state_mem_bits': 0,
            'max_buffer_mem_bits': 0,
            'operator_class_count': {},
            'total_mac_dense_ops': 0,
            'total_ac_sparse_ops': 0,
            'total_and_ops': 0,
            'total_compare_ops': 0,
            'total_shift_ops': 0,
            'total_maxpool_compare_ops': 0,
        }
    # Hooks may fire multiple times across profiled batches. Aggregate by layer
    # first so memory/params describe one representative forward pass.
    grouped = df.groupby('layer_name', as_index=False).agg({
        'params': 'first',
        'sops_proxy': 'mean',
        'spike_count': 'mean',
        'firing_rate': 'mean',
        'weight_mem_bits': 'first',
        'activation_mem_bits': 'max',
        'state_mem_bits': 'max',
        'buffer_mem_bits': 'max',
        'module_type': 'first',
        'mac_dense_ops': 'mean',
        'ac_sparse_ops': 'mean',
        'and_ops': 'mean',
        'compare_ops': 'mean',
        'shift_ops': 'mean',
        'maxpool_compare_ops': 'mean',
    })
    op_counts = df.groupby('module_type')['layer_name'].nunique().to_dict()
    return {
        'total_params_profiled': int(grouped['params'].sum()),
        'total_sops_proxy': float(grouped['sops_proxy'].sum()),
        'total_spike_count': float(grouped['spike_count'].sum()),
        'mean_firing_rate': float(df.loc[df['is_binary_output'] == True, 'firing_rate'].mean()) if (df['is_binary_output'] == True).any() else 0.0,
        'total_weight_mem_bits': int(grouped['weight_mem_bits'].sum()),
        'total_activation_mem_bits': int(grouped['activation_mem_bits'].sum()),
        'total_state_mem_bits': int(grouped['state_mem_bits'].sum()),
        'max_buffer_mem_bits': int(grouped['buffer_mem_bits'].max()),
        'operator_class_count': {str(k): int(v) for k, v in op_counts.items()},
        'total_mac_dense_ops': float(grouped['mac_dense_ops'].sum()),
        'total_ac_sparse_ops': float(grouped['ac_sparse_ops'].sum()),
        'total_and_ops': float(grouped['and_ops'].sum()),
        'total_compare_ops': float(grouped['compare_ops'].sum()),
        'total_shift_ops': float(grouped['shift_ops'].sum()),
        'total_maxpool_compare_ops': float(grouped['maxpool_compare_ops'].sum()),
    }
