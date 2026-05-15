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
        # [B,T,C,H,W] -> [B,T,N,D]
        B, T, C, H, W = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)


class StackedStages(FeatureExtractorBase):
    def __init__(self, stages: list[nn.Module], output_format: str = 'tokens') -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.output_format = output_format
        self.last_feature_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        self.last_feature_shape = tuple(x.shape)
        if self.output_format == 'tokens':
            return self._to_tokens(x)
        return x


def _channels(cfg: dict[str, Any]) -> list[int]:
    return [int(c) for c in cfg.get('channels', [16, 32])]


def build_feature_extractor(cfg: dict[str, Any], in_channels: int, surrogate_alpha: float = 4.0) -> nn.Module:
    name = cfg.get('name', 'conv_bn_lif_maxpool')
    channels = _channels(cfg)
    output_format = cfg.get('output_format', 'tokens')
    residual = cfg.get('residual', 'none')

    if residual not in ('none', None, 'ms'):
        raise ValueError(f'Only residual none/ms is allowed in Milestone 1, got {residual!r}')

    stages: list[nn.Module] = []
    prev = in_channels

    if name == 'conv_bn_lif_maxpool':  # FE0
        for ch in channels:
            stages.append(ConvBNLIFMaxPoolStage(prev, ch, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format)

    if name == 'conv_bn_maxpool_lif':  # FE1
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format)

    if name == 'lif_conv_bn_maxpool_lif':  # FE2
        for ch in channels:
            stages.append(LIFConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format)

    if name == 'depthwise_separable':  # FE3
        for ch in channels:
            stages.append(DepthwiseSeparableStage(prev, ch, pool=True, surrogate_alpha=surrogate_alpha))
            if residual == 'ms':
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format)

    if name == 'hierarchical_tokenizer':  # FE4
        # Explicitly hierarchical N↓, C↑; same primitive as FE1 for stable binarized downsampling.
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format)

    if name == 'resnet_local_ms':  # FE5
        if residual != 'ms':
            raise ValueError('resnet_local_ms requires residual: ms')
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha))
            num_blocks = int(cfg.get('blocks_per_stage', 1))
            for _ in range(num_blocks):
                stages.append(MSResidualBlock(ch, surrogate_alpha=surrogate_alpha))
            prev = ch
        return StackedStages(stages, output_format=output_format)

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
        self.last_feature_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Membrane-level fusion; only thresholded spike is communicated.
        x = self.fuse_lif(self.main(x) + self.hf(x))
        for stage in self.tail:
            x = stage(x)
        self.last_feature_shape = tuple(x.shape)
        return self._to_tokens(x) if self.output_format == 'tokens' else x
