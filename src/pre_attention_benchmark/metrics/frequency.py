from __future__ import annotations

import numpy as np
import torch


def high_frequency_ratio(x: torch.Tensor, eps: float = 1e-12) -> float | None:
    """Estimate high-frequency energy ratio for feature maps or square token grids."""
    if x.dim() == 5:
        m = x.detach().float().mean(dim=(0, 1, 2)).cpu().numpy()
    elif x.dim() == 4:
        # Token [B,T,N,D]: se N e quadrato ricostruiamo una mappa spaziale.
        n = int(x.shape[2])
        side = int(np.sqrt(n))
        if side * side != n:
            return None
        m = x.detach().float().mean(dim=(0, 1, 3)).reshape(side, side).cpu().numpy()
    else:
        return None
    with torch.no_grad():
        if min(m.shape) < 4:
            return None
        # Stima volutamente semplice: energia fuori dal centro dello spettro 2D.
        spec = np.fft.fftshift(np.fft.fft2(m))
        power = np.abs(spec) ** 2
        H, W = power.shape
        yy, xx = np.ogrid[:H, :W]
        cy, cx = (H - 1) / 2, (W - 1) / 2
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = rr.max() + eps
        mask = rr >= 0.5 * rmax
        return float(power[mask].sum() / (power.sum() + eps))
