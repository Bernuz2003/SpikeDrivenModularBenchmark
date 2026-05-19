from __future__ import annotations

import torch
from torch import nn
from pre_attention_benchmark.training.utils import tensor_is_binary
from .feature_extractors import build_feature_extractor
from .attention import build_attention
from .heads import build_head, describe_head


class SpikeDrivenBenchmarkModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, attention: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.attention = attention
        self.head = head
        self.last_tokens_shape: tuple[int, ...] | None = None
        self.metadata: dict = {}

    def forward(self, x: torch.Tensor, validate_hidden_binary: bool = False) -> torch.Tensor:
        z = self.feature_extractor(x)
        if validate_hidden_binary and not tensor_is_binary(z):
            raise RuntimeError('Feature extractor output is not binary; hidden spike communication violated.')
        z = self.attention(z)
        # Anche se ora e Identity, lasciamo il controllo qui per non dimenticarlo
        # quando si sperimentera una attention reale.
        if validate_hidden_binary and not tensor_is_binary(z):
            raise RuntimeError('Attention output is not binary; hidden spike communication violated.')
        self.last_tokens_shape = tuple(z.shape)
        return self.head(z)


def infer_head_dim(feature_extractor: nn.Module, T: int, in_channels: int, height: int, width: int, device: torch.device) -> tuple[int, tuple[int, ...]]:
    was_training = feature_extractor.training
    feature_extractor.eval()
    with torch.no_grad():
        # Inferenza robusta della dimensione del readout: evita calcoli manuali
        # fragili quando cambiano pooling, stage o output_format.
        dummy = torch.zeros(1, T, in_channels, height, width, device=device)
        out = feature_extractor(dummy)
        if out.dim() == 4:  # [B,T,N,D]
            dim = int(out.shape[-1])
        elif out.dim() == 5:  # [B,T,C,H,W]
            dim = int(out.shape[2])
        else:
            raise ValueError(f'Unsupported feature output shape: {tuple(out.shape)}')
    feature_extractor.train(was_training)
    return dim, tuple(out.shape)


def build_model(cfg: dict, encoder, device: torch.device) -> SpikeDrivenBenchmarkModel:
    in_channels = int(encoder.channels)
    T = int(encoder.T)
    h = int(encoder.height)
    w = int(encoder.width)
    surrogate_alpha = float(cfg.get('training', {}).get('surrogate_alpha', 4.0))
    fe = build_feature_extractor(cfg['model']['feature_extractor'], in_channels, surrogate_alpha=surrogate_alpha).to(device)
    head_dim, out_shape = infer_head_dim(fe, T, in_channels, h, w, device)
    att = build_attention(cfg['model']['attention']).to(device)
    num_classes = int(cfg['dataset'].get('num_classes', 10))
    head = build_head(cfg['model']['head'], head_dim, num_classes, surrogate_alpha=surrogate_alpha, feature_shape=out_shape).to(device)
    model = SpikeDrivenBenchmarkModel(fe, att, head).to(device)
    model.metadata = {
        'encoder': encoder.describe() if hasattr(encoder, 'describe') else {},
        'feature_extractor': fe.describe() if hasattr(fe, 'describe') else {'output_shape': list(out_shape)},
        'attention': {'name': cfg['model']['attention']['name'], 'params': 0, 'sops_proxy': 0, 'buffer_nxn': False},
        'head': describe_head(head, head_dim, num_classes),
    }
    return model
