from __future__ import annotations

from typing import Any
import numpy as np
import torch


def apply_event_augmentations(events: dict[str, Any], cfg: dict[str, Any] | None, rng: np.random.Generator) -> dict[str, Any]:
    if not cfg:
        return events
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in events.items()}
    n = len(out['t'])
    keep = np.ones(n, dtype=bool)

    drop = float(cfg.get('event_drop', 0.0) or 0.0)
    if drop > 0:
        keep &= rng.random(n) >= drop

    polarity_drop = float(cfg.get('polarity_drop', 0.0) or 0.0)
    if polarity_drop > 0:
        pkeep = rng.random(n) >= polarity_drop
        out['p'] = np.where(pkeep, out['p'], 0)

    jitter = float(cfg.get('temporal_jitter', 0.0) or 0.0)
    if jitter > 0 and n > 0:
        span = max(1.0, float(np.max(out['t']) - np.min(out['t'])))
        out['t'] = out['t'] + rng.normal(0, jitter * span, size=n)

    for key in ('t', 'x', 'y', 'p'):
        out[key] = np.asarray(out[key])[keep]
    return out


def apply_spike_augmentations(spikes: torch.Tensor, cfg: dict[str, Any] | None, rng: np.random.Generator | None = None) -> torch.Tensor:
    if not cfg:
        return spikes
    if cfg.get('timestep_shuffle', False):
        if rng is None:
            perm = torch.randperm(spikes.shape[0])
        else:
            perm = torch.as_tensor(rng.permutation(spikes.shape[0]), dtype=torch.long)
        spikes = spikes[perm]
    early = cfg.get('early_truncation', None)
    if early is not None:
        early = int(early)
        if 0 < early < spikes.shape[0]:
            out = torch.zeros_like(spikes)
            out[:early] = spikes[:early]
            spikes = out
    return spikes
