from __future__ import annotations

from torch import nn


class IdentityAttention(nn.Module):
    track_metrics = True
    op_class = 'Identity'

    def forward(self, x):
        return x


def build_attention(cfg: dict) -> nn.Module:
    name = cfg.get('name', 'identity')
    if name != 'identity':
        # Questo benchmark misura il frontend: qualsiasi attention reale falserebbe
        # il confronto tra encoder, feature extractor e head.
        raise ValueError(f"pre-attention benchmark requires attention='identity', got {name!r}")
    return IdentityAttention()
