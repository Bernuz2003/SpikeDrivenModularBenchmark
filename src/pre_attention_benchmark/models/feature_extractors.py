from __future__ import annotations

from typing import Any
import torch
from torch import nn
from .blocks import (
    ConvBNLIFMaxPoolStage,
    ConvBNMaxPoolLIFStage,
    LIFConvBNMaxPoolLIFStage,
    DepthwiseSeparableStage,
    MSResidualBlock,
)
from .lif import MultiStepLIF


class FeatureExtractorBase(nn.Module):
    output_format: str = 'tokens'

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W] -> [B,T,N,D]: tokenizziamo solo dopo aver prodotto spike binari.
        B, T, C, H, W = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)


class StackedStages(FeatureExtractorBase):
    def __init__(
        self,
        stages: list[nn.Module],
        output_format: str = 'tokens',
        name: str = 'stacked_stages',
        channels: list[int] | None = None,
        residual: str | None = 'none',
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.output_format = output_format
        self.extractor_name = name
        self.channels = channels or []
        self.residual = residual or 'none'
        self.last_feature_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        self.last_feature_shape = tuple(x.shape)
        if self.output_format == 'tokens':
            return self._to_tokens(x)
        return x

    def describe(self) -> dict[str, Any]:
        feature_shape = list(self.last_feature_shape) if self.last_feature_shape else None
        token_count = None
        embedding_dim = None
        if self.last_feature_shape is not None:
            _, _, c, h, w = self.last_feature_shape
            # N e D vengono misurati da una forward reale, cosi il report segue
            # downsampling e canali effettivi invece di fidarsi della config.
            token_count = int(h * w)
            embedding_dim = int(c)
        primitive_stages = sum(1 for s in self.stages if s.__class__.__name__ != 'MSResidualBlock')
        return {
            'name': self.extractor_name,
            'num_stages': primitive_stages,
            'channels': self.channels,
            'downsampling_factor': 2 ** primitive_stages,
            'output_format': self.output_format,
            'feature_shape': feature_shape,
            'token_count_N': token_count,
            'embedding_dim_D': embedding_dim,
            'uses_ms_residual': self.residual == 'ms',
            'operators': sorted({_stage_operator_name(s) for s in self.stages}),
        }


def _channels(cfg: dict[str, Any]) -> list[int]:
    return [int(c) for c in cfg.get('channels', [16, 32])]


def build_feature_extractor(cfg: dict[str, Any], in_channels: int, surrogate_alpha: float = 4.0) -> nn.Module:
    name = cfg.get('name', 'conv_bn_lif_maxpool')
    channels = _channels(cfg)   # Canali di uscita dei vari stage, es. [16, 32]
    output_format = cfg.get('output_format', 'tokens')
    residual = cfg.get('residual', 'none')

    if residual not in ('none', None, 'ms'):
        raise ValueError(f'Only residual none/ms is allowed in pre-attention benchmark, got {residual!r}')

    stages: list[nn.Module] = []
    prev = in_channels  # Canali in ingresso allo stage corrente

    if name == 'conv_bn_lif_maxpool':  # FE0
        for ch in channels:
            stages.append(ConvBNLIFMaxPoolStage(prev, ch, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch   # Lo stage successivo riceve in input i canali appena prodotti
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'conv_bn_maxpool_lif':  # FE1
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'lif_conv_bn_maxpool_lif':  # FE2
        for ch in channels:
            stages.append(LIFConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'depthwise_separable':  # FE3
        for ch in channels:
            stages.append(DepthwiseSeparableStage(prev, ch, pool=True, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'hierarchical_tokenizer':  # FE4
        # Gerarchico nel senso N giu / C su; riusa FE1 per non introdurre
        # una primitiva nuova mentre stiamo confrontando i componenti.
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'resnet_local_ms':  # FE5
        if residual != 'ms':
            raise ValueError('resnet_local_ms requires residual: ms')
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            num_blocks = int(cfg.get('blocks_per_stage', 1))
            for _ in range(num_blocks):
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'dual_path_high_frequency':  # FE6 optional
        return DualPathHighFrequencyExtractor(in_channels, channels, output_format=output_format, surrogate_alpha=surrogate_alpha)

    raise ValueError(f'Unknown feature extractor: {name}')


class DualPathHighFrequencyExtractor(FeatureExtractorBase):
    """Optional FE6: two local branches fused at membrane level, then LIF.

    Output remains binary. This is included as an experimental placeholder for a
    high-frequency branch; it is intentionally small and transparent.
    """

    def __init__(self, in_channels: int, channels: list[int], output_format: str = 'tokens', surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        c = channels[0] if channels else 16
        self.main = ConvBNMaxPoolLIFStage(in_channels, c, surrogate_alpha=surrogate_alpha)
        self.hf = DepthwiseSeparableStage(in_channels, c, pool=True, surrogate_alpha=surrogate_alpha)
        self.fuse_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='fuse_lif')
        tail = []
        prev = c
        for ch in channels[1:]:
            tail.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        self.tail = nn.ModuleList(tail)
        self.output_format = output_format
        self.channels = channels
        self.last_feature_shape = None
        self.extractor_name = 'dual_path_high_frequency'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fusione a livello membrana: la somma resta interna, fuori passa il LIF.
        x = self.fuse_lif(self.main(x) + self.hf(x))
        for stage in self.tail:
            x = stage(x)
        self.last_feature_shape = tuple(x.shape)
        return self._to_tokens(x) if self.output_format == 'tokens' else x

    def describe(self) -> dict[str, Any]:
        token_count = None
        embedding_dim = None
        if self.last_feature_shape is not None:
            _, _, c, h, w = self.last_feature_shape
            token_count = int(h * w)
            embedding_dim = int(c)
        return {
            'name': self.extractor_name,
            'num_stages': max(1, len(self.channels)),
            'channels': self.channels,
            'downsampling_factor': 2 ** max(1, len(self.channels)),
            'output_format': self.output_format,
            'feature_shape': list(self.last_feature_shape) if self.last_feature_shape else None,
            'token_count_N': token_count,
            'embedding_dim_D': embedding_dim,
            'uses_ms_residual': False,
            'operators': ['Conv-BN-MaxPool-LIF', 'SpikingDepthwiseSeparableConv', 'membrane_fusion_lif'],
        }


def _stage_operator_name(stage: nn.Module) -> str:
    names = {
        'ConvBNLIFMaxPoolStage': 'Conv-BN-LIF-MaxPool',
        'ConvBNMaxPoolLIFStage': 'Conv-BN-MaxPool-LIF',
        'LIFConvBNMaxPoolLIFStage': 'LIF-Conv-BN-MaxPool-LIF',
        'DepthwiseSeparableStage': 'SpikingDepthwiseSeparableConv',
        'MSResidualBlock': 'MSResidual',
    }
    return names.get(stage.__class__.__name__, stage.__class__.__name__)
