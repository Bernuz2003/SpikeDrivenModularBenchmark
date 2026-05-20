from __future__ import annotations

from typing import Any
import torch
from torch import nn
from .blocks import (
    ConvBNLIFMaxPoolStage,
    ConvBNMaxPoolLIFStage,
    LIFConvBNMaxPoolLIFStage,
    DepthwiseSeparableStage,
    StridedConvStemStage,
    PolaritySeparableStemStage,
    MSResidualBlock,
)


class FeatureExtractorBase(nn.Module):
    output_format: str = 'tokens'

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W] -> [B,T,N,D]
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
    return [int(c) for c in cfg.get('channels', [32, 64, 128])]


FEATURE_EXTRACTOR_NAMES = {
    'sps_like_ms',
    'evidence_pooling_ms',
    'spike_input_ms',
    'depthwise_separable_ms',
    'strided_conv_stem_ms',
    'polarity_separable_stem_ms',
}


def _append_stage(stages: list[nn.Module], stage: nn.Module, channels: int, surrogate_alpha: float) -> None:
    stages.append(stage)
    stages.append(MSResidualBlock(channels, surrogate_alpha=surrogate_alpha))


def build_feature_extractor(cfg: dict[str, Any], in_channels: int, surrogate_alpha: float = 4.0) -> nn.Module:
    name = cfg.get('name', 'sps_like_ms')
    channels = _channels(cfg)   # Canali di uscita dei vari stage, es. [32, 64, 128]
    output_format = cfg.get('output_format', 'tokens')
    residual = cfg.get('residual', 'ms')

    if name not in FEATURE_EXTRACTOR_NAMES:
        raise ValueError(f'Unknown feature extractor: {name}')
    if residual != 'ms':
        raise ValueError(f'{name} requires residual: ms in the main pre-attention sweep.')

    stages: list[nn.Module] = []
    prev = in_channels  # Canali in ingresso allo stage corrente

    if name == 'sps_like_ms':  # FE-A
        for ch in channels:
            _append_stage(stages, ConvBNLIFMaxPoolStage(prev, ch, surrogate_alpha=surrogate_alpha), ch, surrogate_alpha)
            prev = ch   # Lo stage successivo riceve in input i canali appena prodotti
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'evidence_pooling_ms':  # FE-B
        for ch in channels:
            _append_stage(stages, ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha), ch, surrogate_alpha)
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'spike_input_ms':  # FE-C
        for ch in channels:
            _append_stage(stages, LIFConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha), ch, surrogate_alpha)
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'depthwise_separable_ms':  # FE-D
        for ch in channels:
            _append_stage(stages, DepthwiseSeparableStage(prev, ch, pool=True, surrogate_alpha=surrogate_alpha), ch, surrogate_alpha)
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'strided_conv_stem_ms':  # FE-E
        for ch in channels:
            _append_stage(stages, StridedConvStemStage(prev, ch, surrogate_alpha=surrogate_alpha), ch, surrogate_alpha)
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'polarity_separable_stem_ms':  # FE-G
        if in_channels != 2:
            raise ValueError('polarity_separable_stem_ms requires DVS polarity input with exactly 2 channels.')
        first = channels[0]
        _append_stage(stages, PolaritySeparableStemStage(first, surrogate_alpha=surrogate_alpha), first, surrogate_alpha)
        prev = first
        for ch in channels[1:]:
            _append_stage(stages, ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha), ch, surrogate_alpha)
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    raise ValueError(f'Unknown feature extractor: {name}')


def _stage_operator_name(stage: nn.Module) -> str:
    names = {
        'ConvBNLIFMaxPoolStage': 'Conv-BN-LIF-MaxPool',
        'ConvBNMaxPoolLIFStage': 'Conv-BN-MaxPool-LIF',
        'LIFConvBNMaxPoolLIFStage': 'LIF-Conv-BN-MaxPool-LIF',
        'DepthwiseSeparableStage': 'SpikingDepthwiseSeparableConv',
        'StridedConvStemStage': 'StridedConvStem',
        'PolaritySeparableStemStage': 'PolaritySeparableStem',
        'MSResidualBlock': 'MSResidual',
    }
    return names.get(stage.__class__.__name__, stage.__class__.__name__)
