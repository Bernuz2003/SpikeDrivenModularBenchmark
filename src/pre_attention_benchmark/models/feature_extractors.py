from __future__ import annotations

from typing import Any
import torch
from torch import nn
from .blocks import (
    ConvBNLIFMaxPoolStage,
    ConvBNMaxPoolLIFStage,
    LIFConvBNLIFMaxPoolStage,
    StridedConvStemStage,
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
        return {
            'name': self.extractor_name,
            'num_stages': len(self.stages),
            'channels': self.channels,
            'downsampling_factor': 2 ** len(self.stages),
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
    'strided_conv_stem_ms',
}


def build_feature_extractor(cfg: dict[str, Any], in_channels: int, surrogate_alpha: float = 4.0) -> nn.Module:
    name = cfg.get('name', 'sps_like_ms')
    channels = _channels(cfg)   # Canali di uscita dei vari stage, es. [32, 64, 128]
    output_format = cfg.get('output_format', 'tokens')
    residual = cfg.get('residual', 'ms')

    if name not in FEATURE_EXTRACTOR_NAMES:
        raise ValueError(f'Unknown feature extractor: {name}')
    if residual not in ('none', None, 'ms'):
        raise ValueError(f'Only residual none/ms is allowed in pre-attention benchmark, got {residual!r}')
    use_ms_residual = residual == 'ms'

    stages: list[nn.Module] = []
    prev = in_channels  # Canali in ingresso allo stage corrente

    if name == 'sps_like_ms':
        for ch in channels:
            stages.append(ConvBNLIFMaxPoolStage(prev, ch, surrogate_alpha=surrogate_alpha, use_ms_residual=use_ms_residual))
            prev = ch   # Lo stage successivo riceve in input i canali appena prodotti
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'evidence_pooling_ms':
        for ch in channels:
            stages.append(ConvBNMaxPoolLIFStage(prev, ch, surrogate_alpha=surrogate_alpha, use_ms_residual=use_ms_residual))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'spike_input_ms':
        for ch in channels:
            stages.append(LIFConvBNLIFMaxPoolStage(prev, ch, surrogate_alpha=surrogate_alpha, use_ms_residual=use_ms_residual))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    if name == 'strided_conv_stem_ms':
        for ch in channels:
            stages.append(StridedConvStemStage(prev, ch, surrogate_alpha=surrogate_alpha, use_ms_residual=use_ms_residual))
            prev = ch
        return StackedStages(stages, output_format=output_format, name=name, channels=channels, residual=residual)

    raise ValueError(f'Unknown feature extractor: {name}')


def _stage_operator_name(stage: nn.Module) -> str:
    names = {
        'ConvBNLIFMaxPoolStage': 'Conv-BN-LIF-MaxPool',
        'ConvBNMaxPoolLIFStage': 'Conv-BN-MaxPool-LIF',
        'LIFConvBNLIFMaxPoolStage': 'LIF-Conv-BN-LIF-MaxPool',
        'StridedConvStemStage': 'StridedConvStem',
    }
    return names.get(stage.__class__.__name__, stage.__class__.__name__)
