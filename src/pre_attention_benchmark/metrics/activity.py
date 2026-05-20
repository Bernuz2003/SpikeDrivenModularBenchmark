from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd
import torch
from torch import nn

from pre_attention_benchmark.training.utils import tensor_is_binary


@dataclass
class ActivityLayerRecord:
    run_id: str
    layer_name: str
    module_type: str
    profile_scope: str
    input_shape: str
    output_shape: str
    input_numel: int
    output_numel: int
    params: int
    is_binary_input: bool
    is_binary_output: bool
    input_spike_count: float
    output_spike_count: float
    input_firing_rate: float
    output_firing_rate: float
    spike_density_mean: float
    spike_density_std: float
    spike_density_timestep: str
    burstiness: float
    kernel_size: str
    stride: str
    padding: str
    dilation: str
    groups: int | None
    in_channels: int | None
    out_channels: int | None
    in_features: int | None
    out_features: int | None
    time_steps: int


class ActivityProfiler:
    """Misura attivita spike e struttura osservata, senza stimare costi hardware."""

    def __init__(self, model: nn.Module, run_id: str, time_steps: int | None = None) -> None:
        self.model = model
        self.run_id = run_id
        self.time_steps = int(time_steps or 0)
        self.handles = []
        self.enabled = False
        self.records: list[ActivityLayerRecord] = []
        self.profile_layers: list[dict[str, Any]] = []

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if name == '':
                continue
            if self._should_track(name, module):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.records.clear()
        self.profile_layers.clear()

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _should_track(self, name: str, module: nn.Module) -> bool:
        if name in {'feature_extractor', 'attention', 'head'}:
            return True
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            return True
        cls = module.__class__.__name__
        return cls in {'MultiStepLIF', 'TDMaxPool', 'StatelessThreshold'}

    def _scope(self, name: str, module: nn.Module) -> str:
        if name in {'feature_extractor', 'attention', 'head'}:
            return name
        if not any(True for _ in module.children()):
            return 'leaf'
        return 'block'

    def _hook(self, name: str):
        def fn(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if not self.enabled:
                return
            try:
                record = self._make_record(name, module, inputs, output)
                if record is not None:
                    self.records.append(record)
                    self.profile_layers.append(self._profile_entry(record, module))
            except Exception as exc:
                raise RuntimeError(f'Activity profiler hook failed at layer {name}: {exc}') from exc

        return fn

    def _tensor(self, x: Any) -> torch.Tensor | None:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
            return x[0]
        return None

    def _binary_stats(self, tensor: torch.Tensor | None) -> tuple[bool, float, float, int]:
        if tensor is None:
            return False, 0.0, 0.0, 0
        detached = tensor.detach()
        numel = int(detached.numel())
        is_binary = tensor_is_binary(detached)
        if not is_binary:
            return False, 0.0, 0.0, numel
        spike_count = float(detached.sum().item())
        firing_rate = spike_count / max(1, numel)
        return True, spike_count, firing_rate, numel

    def _make_record(self, name: str, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> ActivityLayerRecord | None:
        inp = self._tensor(inputs)
        out = self._tensor(output)
        if out is None:
            return None
        is_binary_input, input_spikes, input_fr, input_numel = self._binary_stats(inp)
        is_binary_output, output_spikes, output_fr, output_numel = self._binary_stats(out)
        density_mean, density_std, burstiness, density_timestep = self._temporal_density(
            out.detach(),
            is_binary_output,
            self._scope(name, module),
        )
        structure = self._structure(module)
        return ActivityLayerRecord(
            run_id=self.run_id,
            layer_name=name,
            module_type=module.__class__.__name__,
            profile_scope=self._scope(name, module),
            input_shape=json.dumps(list(inp.shape)) if inp is not None else '[]',
            output_shape=json.dumps(list(out.shape)),
            input_numel=input_numel,
            output_numel=output_numel,
            params=sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad),
            is_binary_input=is_binary_input,
            is_binary_output=is_binary_output,
            input_spike_count=input_spikes,
            output_spike_count=output_spikes,
            input_firing_rate=input_fr,
            output_firing_rate=output_fr,
            spike_density_mean=density_mean,
            spike_density_std=density_std,
            spike_density_timestep=json.dumps(density_timestep),
            burstiness=burstiness,
            kernel_size=structure.get('kernel_size', ''),
            stride=structure.get('stride', ''),
            padding=structure.get('padding', ''),
            dilation=structure.get('dilation', ''),
            groups=structure.get('groups'),
            in_channels=structure.get('in_channels'),
            out_channels=structure.get('out_channels'),
            in_features=structure.get('in_features'),
            out_features=structure.get('out_features'),
            time_steps=self.time_steps,
        )

    def _temporal_density(self, out: torch.Tensor, is_binary: bool, scope: str = 'leaf') -> tuple[float, float, float, list[float]]:
        if not is_binary:
            return 0.0, 0.0, 0.0, []
        if out.dim() == 5:
            dens = out.float().mean(dim=tuple(i for i in range(out.dim()) if i != 1))
        elif out.dim() == 4 and scope in {'feature_extractor', 'attention'}:
            dens = out.float().mean(dim=(0, 2, 3))
        else:
            return 0.0, 0.0, 0.0, []
        mean = float(dens.mean().cpu())
        std = float(dens.std(unbiased=False).cpu())
        var = float(dens.var(unbiased=False).cpu())
        burstiness = var / (mean + 1e-12)
        return mean, std, burstiness, [float(v) for v in dens.detach().cpu().flatten()]

    def _structure(self, module: nn.Module) -> dict[str, Any]:
        if isinstance(module, nn.Conv2d):
            return {
                'kernel_size': json.dumps(list(module.kernel_size)),
                'stride': json.dumps(list(module.stride)),
                'padding': json.dumps(list(module.padding)),
                'dilation': json.dumps(list(module.dilation)),
                'groups': int(module.groups),
                'in_channels': int(module.in_channels),
                'out_channels': int(module.out_channels),
            }
        if isinstance(module, nn.Linear):
            return {
                'in_features': int(module.in_features),
                'out_features': int(module.out_features),
            }
        cls = module.__class__.__name__
        if cls == 'TDMaxPool':
            kernel = getattr(module, 'kernel_size', None)
            stride = getattr(module, 'stride', None)
            return {
                'kernel_size': json.dumps(_as_list(kernel)),
                'stride': json.dumps(_as_list(stride)),
            }
        return {}

    def _profile_entry(self, record: ActivityLayerRecord, module: nn.Module) -> dict[str, Any]:
        entry = {
            'name': record.layer_name,
            'type': record.module_type,
            'profile_scope': record.profile_scope,
            'input_shape': json.loads(record.input_shape),
            'output_shape': json.loads(record.output_shape),
            'params': record.params,
            'is_binary_input': record.is_binary_input,
            'is_binary_output': record.is_binary_output,
            'input_firing_rate': record.input_firing_rate,
            'output_firing_rate': record.output_firing_rate,
            'time_steps': record.time_steps,
        }
        structure = {}
        for key in ['kernel_size', 'stride', 'padding', 'dilation']:
            value = getattr(record, key)
            if value:
                structure[key] = json.loads(value)
        for key in ['groups', 'in_channels', 'out_channels', 'in_features', 'out_features']:
            value = getattr(record, key)
            if value is not None:
                structure[key] = value
        if structure:
            entry['structure'] = structure
        return entry

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])

    def save(self, layerwise_path: str | Path, profile_path: str | Path) -> pd.DataFrame:
        df = self.to_dataframe()
        layerwise_path = Path(layerwise_path)
        layerwise_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(layerwise_path, index=False)
        profile_path = Path(profile_path)
        with profile_path.open('w', encoding='utf-8') as f:
            json.dump({'layers': self.profile_layers}, f, indent=2)
        return df


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return [int(v) for v in value]
    return [int(value)]


def summarize_activity_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            'total_params_profiled': 0,
            'mean_layer_output_firing_rate': 0.0,
            'weighted_output_firing_rate': 0.0,
            'max_layer_output_firing_rate': 0.0,
            'profiled_binary_layers': 0,
            'profiled_layers': 0,
        }
    grouped = df.groupby('layer_name', as_index=False).agg({
        'params': 'first',
        'is_binary_output': 'first',
        'output_spike_count': 'mean',
        'output_numel': 'mean',
        'output_firing_rate': 'mean',
    })
    binary = grouped[grouped['is_binary_output'] == True]
    total_spikes = float(binary['output_spike_count'].sum()) if not binary.empty else 0.0
    total_numel = float(binary['output_numel'].sum()) if not binary.empty else 0.0
    return {
        'total_params_profiled': int(grouped['params'].sum()),
        'mean_layer_output_firing_rate': float(binary['output_firing_rate'].mean()) if not binary.empty else 0.0,
        'weighted_output_firing_rate': total_spikes / max(1.0, total_numel),
        'max_layer_output_firing_rate': float(binary['output_firing_rate'].max()) if not binary.empty else 0.0,
        'profiled_binary_layers': int(len(binary)),
        'profiled_layers': int(len(grouped)),
    }
