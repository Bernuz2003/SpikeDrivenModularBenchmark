from __future__ import annotations

from typing import Any
import torch
from torch import nn
from .lif import MultiStepLIF


def pool_spikes(x: torch.Tensor, mode: str = 'max') -> torch.Tensor:
    """Pool spike maps/tokens without producing hidden multi-bit communication.

    For Milestone 1 we use max pooling across spatial/token dimensions, preserving
    binary values before the terminal readout.
    """
    if x.dim() == 4:  # [B,T,N,D]
        return x.max(dim=2).values
    if x.dim() == 5:  # [B,T,C,H,W]
        return x.flatten(3).max(dim=3).values
    raise ValueError(f'Expected token/map spike tensor, got {tuple(x.shape)}')


class LastTimestepSpikeReadout(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalLinear'
    uses_all_timesteps = False
    produces_spikes_before_readout = True
    accumulates_logits = False

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = pool_spikes(x)  # [B,T,D]
        return self.fc(pooled[:, -1])


class TemporalSpikeVotingReadout(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalLinearTemporalVoting'
    uses_all_timesteps = True
    produces_spikes_before_readout = True
    accumulates_logits = True

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = pool_spikes(x)  # [B,T,D]
        logits_t = self.fc(pooled)
        return logits_t.sum(dim=1)


class SpatialPoolingThresholdHead(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalLinearAfterThreshold'
    uses_all_timesteps = True
    produces_spikes_before_readout = True
    accumulates_logits = False

    def __init__(self, in_dim: int, num_classes: int, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='head_threshold_lif')
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = pool_spikes(x)  # binary [B,T,D]
        spk = self.lif(pooled)
        return self.fc(spk[:, -1])


class ClassNeuronAccumulatorHead(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalClassNeuronAccumulator'
    uses_all_timesteps = True
    produces_spikes_before_readout = False
    accumulates_logits = True

    def __init__(self, in_dim: int, num_classes: int, beta: float = 0.5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = pool_spikes(x)  # [B,T,D]
        currents = self.fc(pooled)  # terminal current, not hidden communication
        mem = torch.zeros_like(currents[:, 0])
        spike_count = torch.zeros_like(mem)
        for t in range(currents.shape[1]):
            mem = self.beta * mem + currents[:, t]
            s = (mem >= 1.0).to(mem.dtype)
            spike_count = spike_count + s
            mem = mem * (1.0 - s)
        return mem + spike_count


def build_head(cfg: dict[str, Any], in_dim: int, num_classes: int, surrogate_alpha: float = 4.0) -> nn.Module:
    name = cfg.get('name', 'last_timestep_spike_readout')
    if name == 'last_timestep_spike_readout':
        return LastTimestepSpikeReadout(in_dim, num_classes)
    if name == 'temporal_spike_voting':
        return TemporalSpikeVotingReadout(in_dim, num_classes)
    if name == 'spatial_pooling_threshold':
        return SpatialPoolingThresholdHead(in_dim, num_classes, surrogate_alpha=surrogate_alpha)
    if name == 'class_neuron_accumulator':
        return ClassNeuronAccumulatorHead(in_dim, num_classes)
    raise ValueError(f'Unknown head: {name}')


def describe_head(head: nn.Module, in_dim: int, num_classes: int) -> dict[str, Any]:
    params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return {
        'name': head.__class__.__name__,
        'op_class': getattr(head, 'op_class', head.__class__.__name__),
        'terminal_readout': bool(getattr(head, 'terminal_readout', False)),
        'uses_all_timesteps': bool(getattr(head, 'uses_all_timesteps', False)),
        'produces_spikes_before_readout': bool(getattr(head, 'produces_spikes_before_readout', False)),
        'accumulates_logits': bool(getattr(head, 'accumulates_logits', False)),
        'input_feature_dim': int(in_dim),
        'num_classes': int(num_classes),
        'terminal_params': int(params),
        'terminal_weight_mem_bits': int(params * 32),
    }
