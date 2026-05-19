from __future__ import annotations

import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .lif import spike_fn


def spatial_average_per_timestep(x: torch.Tensor) -> torch.Tensor:
    """
    Riduce solo la dimensione spaziale/token, mantenendo separati i timestep.

    Input possibili:
        [B, T, N, D]       token format
        [B, T, C, H, W]    feature-map format

    Output:
        [B, T, D] oppure [B, T, C]

    Dopo questa funzione ogni timestep ha un vettore di firing-rate medio
    sulle posizioni spaziali/token.
    """
    if x.dim() == 4:  # [B,T,N,D]
        # Media sui token: conserva quanta parte della scena ha attivato la feature.
        return x.float().mean(dim=2)
    if x.dim() == 5:  # [B,T,C,H,W]
        return x.float().mean(dim=(-1, -2))
    raise ValueError(f'Expected token/map spike tensor, got {tuple(x.shape)}')


class StatelessThreshold(nn.Module):
    """
    Threshold senza memoria temporale.

    Forward:
        output binario 0/1

    Backward:
        surrogate gradient tramite spike_fn

    Serve per sogliare una proiezione terminale senza introdurre uno stato LIF.
    """

    track_metrics = True
    op_class = 'StatelessSurrogateThreshold'
    requires_state = False

    def __init__(self, threshold: float = 1.0, surrogate_alpha: float = 4.0, name: str = 'stateless_threshold') -> None:
        super().__init__()
        self.register_buffer('threshold', torch.tensor(float(threshold)))
        self.surrogate_alpha = float(surrogate_alpha)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward binario, backward surrogate: utile anche in head terminali.
        return spike_fn(x - self.threshold.to(dtype=x.dtype, device=x.device), self.surrogate_alpha)


class SpatioTemporalAvgReadout(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalSpatioTemporalAvgFC'
    uses_all_timesteps = True
    produces_spikes_before_readout = False
    accumulates_logits = False
    accumulates_spikes = False

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.last_pooled_shape: tuple[int, ...] | None = None
        self.last_reduction_ops = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_t = spatial_average_per_timestep(x)  # [B,T,D]
        pooled = pooled_t.mean(dim=1)  # [B,D]
        self.last_pooled_shape = tuple(pooled.shape)
        self.last_reduction_ops = int(x.numel())
        return self.fc(pooled)

    def describe(self) -> dict[str, Any]:
        return {
            'aggregation': 'spatial_mean_then_temporal_mean',
            'readout_signal': 'firing_rate',
        }


class SpikeVisionSpatialPoolingHead(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalSpikeVisionSpatialPooling'
    uses_all_timesteps = True
    produces_spikes_before_readout = True
    accumulates_logits = False
    accumulates_spikes = False
    requires_feature_map = True

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        spatial_size: tuple[int, int],
        pool_regions: int = 1,
        threshold: float = 3.0,
        surrogate_alpha: float = 4.0,
    ) -> None:
        super().__init__()
        h, w = spatial_size
        self.in_channels = int(in_channels)
        self.spatial_size = (int(h), int(w))
        self.pool_regions = int(pool_regions)
        if self.pool_regions < 1:
            raise ValueError('SpikeVisionSpatialPoolingHead requires pool_regions >= 1.')
        # Sieve depthwise: ogni canale ha maschere spaziali proprie, senza mixing
        # tra canali prima del threshold.
        self.spatial_pool = nn.Conv2d(
            self.in_channels,
            self.in_channels * self.pool_regions,
            kernel_size=self.spatial_size,
            groups=self.in_channels,
            bias=False,
        )
        nn.init.constant_(self.spatial_pool.weight, 1.0)
        self.threshold = StatelessThreshold(threshold=threshold, surrogate_alpha=surrogate_alpha, name='spikevision_spatial_threshold')
        self.fc = nn.Linear(self.in_channels * self.pool_regions, num_classes)
        self.last_threshold_shape: tuple[int, ...] | None = None
        self.last_reduction_ops = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                'SpikeVisionSpatialPoolingHead requires feature maps [B,T,C,H,W]. '
                'Set model.feature_extractor.output_format: feature_map.'
            )
        B, T, C, H, W = x.shape
        if C != self.in_channels or (H, W) != self.spatial_size:
            raise ValueError(
                f'SpikeVisionSpatialPoolingHead expected [C,H,W]={self.in_channels,self.spatial_size[0],self.spatial_size[1]}, '
                f'got {(C, H, W)}.'
            )
        pooled = self.spatial_pool(x.reshape(B * T, C, H, W).float())
        pooled = pooled.reshape(B, T, C, self.pool_regions)
        spk = self.threshold(pooled)
        self.last_threshold_shape = tuple(spk.shape)
        # Dopo il threshold torna binario; la media temporale e terminale.
        features = spk.flatten(2).mean(dim=1)
        self.last_reduction_ops = int(spk.numel())
        return self.fc(features)

    def describe(self) -> dict[str, Any]:
        return {
            'aggregation': 'learned_depthwise_spatial_sieve_threshold_then_temporal_mean',
            'spatial_size': list(self.spatial_size),
            'pool_regions': self.pool_regions,
            'threshold': float(self.threshold.threshold.detach().cpu()),
            'readout_signal': 'thresholded_firing_rate',
            'requires_feature_map': True,
        }


class ClassNeuronAccumulatorHead(nn.Module):
    terminal_readout = True
    track_metrics = True
    op_class = 'TerminalClassNeuronAccumulator'
    uses_all_timesteps = True
    produces_spikes_before_readout = True
    accumulates_logits = False
    accumulates_spikes = True

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        beta: float = 0.5,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
        output_mode: str = 'spike_count',
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < float(beta) < 1.0:
            raise ValueError('ClassNeuronAccumulatorHead beta must be in (0,1).')
        if float(threshold) <= 0:
            raise ValueError('ClassNeuronAccumulatorHead threshold must be > 0.')
        if output_mode not in {'spike_count', 'firing_rate'}:
            raise ValueError("ClassNeuronAccumulatorHead output_mode must be 'spike_count' or 'firing_rate'.")
        self.fc = nn.Linear(in_dim, num_classes)
        self.surrogate_alpha = float(surrogate_alpha)
        self.output_mode = output_mode
        self.learn_beta = bool(learn_beta)
        self.learn_threshold = bool(learn_threshold)
        if self.learn_beta:
            self.beta_logit = nn.Parameter(torch.tensor(math.log(float(beta) / (1.0 - float(beta)))))
        else:
            self.register_buffer('beta_value', torch.tensor(float(beta)))
        if self.learn_threshold:
            self.threshold_raw = nn.Parameter(torch.tensor(math.log(math.exp(float(threshold)) - 1.0)))
        else:
            self.register_buffer('threshold_value', torch.tensor(float(threshold)))
        self.last_state_shape: tuple[int, ...] | None = None
        self.last_class_spike_shape: tuple[int, ...] | None = None

    def _beta(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.learn_beta:
            return torch.sigmoid(self.beta_logit).to(dtype=dtype, device=device)
        return self.beta_value.to(dtype=dtype, device=device)

    def _threshold(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.learn_threshold:
            return (F.softplus(self.threshold_raw) + 1e-6).to(dtype=dtype, device=device)
        return self.threshold_value.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = spatial_average_per_timestep(x)  # [B,T,D]
        currents = self.fc(pooled)
        beta = self._beta(currents.dtype, currents.device)
        threshold = self._threshold(currents.dtype, currents.device)
        mem = torch.zeros_like(currents[:, 0])
        spike_count = torch.zeros_like(mem)
        class_spikes = []
        for ti in range(currents.shape[1]):
            # Neuroni di classe terminali: accumulano correnti real-valued e
            # producono spike con surrogate gradient, senza mescolare mem e count.
            mem = beta * mem + currents[:, ti]
            spk = spike_fn(mem - threshold, self.surrogate_alpha)
            spike_count = spike_count + spk
            mem = mem * (1.0 - spk)
            class_spikes.append(spk)
        self.last_state_shape = tuple(mem.shape)
        self.last_class_spike_shape = tuple(torch.stack(class_spikes, dim=1).shape)
        if self.output_mode == 'firing_rate':
            return spike_count / max(1, int(currents.shape[1]))
        return spike_count

    def describe(self) -> dict[str, Any]:
        beta = float(torch.sigmoid(self.beta_logit).detach().cpu()) if self.learn_beta else float(self.beta_value.detach().cpu())
        threshold = float((F.softplus(self.threshold_raw) + 1e-6).detach().cpu()) if self.learn_threshold else float(self.threshold_value.detach().cpu())
        return {
            'aggregation': 'spatial_mean_then_class_neuron_temporal_accumulation',
            'readout_signal': self.output_mode,
            'beta': beta,
            'threshold': threshold,
            'learn_beta': self.learn_beta,
            'learn_threshold': self.learn_threshold,
            'uses_surrogate_class_threshold': True,
        }


def build_head(
    cfg: dict[str, Any],
    in_dim: int,
    num_classes: int,
    surrogate_alpha: float = 4.0,
    feature_shape: tuple[int, ...] | None = None,
) -> nn.Module:
    name = cfg.get('name', 'spatio_temporal_avg_readout')
    if name == 'spatio_temporal_avg_readout':
        return SpatioTemporalAvgReadout(in_dim, num_classes)
    if name == 'spikevision_spatial_pooling':
        if feature_shape is None or len(feature_shape) != 5:
            raise ValueError('spikevision_spatial_pooling requires feature maps [B,T,C,H,W].')
        _, _, c, h, w = feature_shape
        return SpikeVisionSpatialPoolingHead(
            int(c),
            num_classes,
            spatial_size=(int(h), int(w)),
            pool_regions=int(cfg.get('pool_regions', 1)),
            threshold=float(cfg.get('threshold', 3.0)),
            surrogate_alpha=surrogate_alpha,
        )
    if name == 'class_neuron_accumulator':
        return ClassNeuronAccumulatorHead(
            in_dim,
            num_classes,
            beta=float(cfg.get('beta', 0.5)),
            threshold=float(cfg.get('threshold', 1.0)),
            surrogate_alpha=surrogate_alpha,
            output_mode=cfg.get('output_mode', 'spike_count'),
            learn_beta=bool(cfg.get('learn_beta', False)),
            learn_threshold=bool(cfg.get('learn_threshold', False)),
        )
    raise ValueError(f'Unknown head: {name}')


def describe_head(head: nn.Module, in_dim: int, num_classes: int) -> dict[str, Any]:
    params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    meta = {
        'name': head.__class__.__name__,
        'op_class': getattr(head, 'op_class', head.__class__.__name__),
        'terminal_readout': bool(getattr(head, 'terminal_readout', False)),
        'track_metrics': bool(getattr(head, 'track_metrics', False)),
        'uses_all_timesteps': bool(getattr(head, 'uses_all_timesteps', False)),
        'produces_spikes_before_readout': bool(getattr(head, 'produces_spikes_before_readout', False)),
        'accumulates_logits': bool(getattr(head, 'accumulates_logits', False)),
        'accumulates_spikes': bool(getattr(head, 'accumulates_spikes', False)),
        'input_feature_dim': int(in_dim),
        'num_classes': int(num_classes),
        'terminal_params': int(params),
        'terminal_weight_mem_bits': int(params * 32),
    }
    if hasattr(head, 'describe'):
        meta.update(head.describe())
    return meta
