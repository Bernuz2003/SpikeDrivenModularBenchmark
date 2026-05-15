from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch


class EventEncoder(ABC):
    name: str = 'base'

    def __init__(self, T: int, height: int, width: int, polarity_channels: bool = True, binarize: bool = True, **kwargs: Any) -> None:
        self.T = int(T)
        self.height = int(height)
        self.width = int(width)
        self.polarity_channels = bool(polarity_channels)
        self.binarize = bool(binarize)
        if not self.binarize:
            raise ValueError('Milestone 1 encoders must produce binary spikes; set binarize=true.')

    @property
    def channels(self) -> int:
        return 2 if self.polarity_channels else 1

    @abstractmethod
    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        """Return a binary numpy tensor [T,C,H,W]."""

    def __call__(self, events: dict[str, Any]) -> torch.Tensor:
        arr = self.encode_np(events)
        arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(arr)

    def describe(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'T': self.T,
            'height': self.height,
            'width': self.width,
            'channels': self.channels,
            'binarize': self.binarize,
        }

    def _scale_xy(self, events: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(events['x'], dtype=np.int64)
        y = np.asarray(events['y'], dtype=np.int64)
        p = np.asarray(events['p'], dtype=np.int64)
        t = np.asarray(events['t'])
        src_h = int(events.get('height', self.height))
        src_w = int(events.get('width', self.width))
        if src_w != self.width and src_w > 0:
            x = np.floor(x.astype(np.float64) * self.width / src_w).astype(np.int64)
        if src_h != self.height and src_h > 0:
            y = np.floor(y.astype(np.float64) * self.height / src_h).astype(np.int64)
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        if self.polarity_channels:
            p = (p > 0).astype(np.int64)
        else:
            p = np.zeros_like(p, dtype=np.int64)
        return t, x, y, p

    def _time_bins(self, t: np.ndarray, T: int | None = None) -> np.ndarray:
        T = int(T or self.T)
        if t.size == 0:
            return np.zeros(0, dtype=np.int64)
        t = t.astype(np.float64)
        t_min = float(t.min())
        t_max = float(t.max())
        if t_max <= t_min:
            return np.zeros_like(t, dtype=np.int64)
        bins = np.floor((t - t_min) / (t_max - t_min + 1e-12) * T).astype(np.int64)
        return np.clip(bins, 0, T - 1)
