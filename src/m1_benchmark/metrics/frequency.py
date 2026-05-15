from __future__ import annotations

import numpy as np
import torch


def high_frequency_ratio(x: torch.Tensor, eps: float = 1e-12) -> float | None:
    """Estimate high-frequency energy ratio for spike feature maps [B,T,C,H,W]."""
    if x.dim() != 5:
        return None
    with torch.no_grad():
        # Mean over B,T,C -> [H,W]
        m = x.detach().float().mean(dim=(0, 1, 2)).cpu().numpy()
        if min(m.shape) < 4:
            return None
        spec = np.fft.fftshift(np.fft.fft2(m))
        power = np.abs(spec) ** 2
        H, W = power.shape
        yy, xx = np.ogrid[:H, :W]
        cy, cx = (H - 1) / 2, (W - 1) / 2
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = rr.max() + eps
        mask = rr >= 0.5 * rmax
        return float(power[mask].sum() / (power.sum() + eps))
