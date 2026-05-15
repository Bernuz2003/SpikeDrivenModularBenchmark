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
        raise ValueError(f"Milestone 1 requires attention='identity', got {name!r}")
    return IdentityAttention()
