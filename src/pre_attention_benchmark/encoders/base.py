from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch


class EventEncoder(ABC):
    name: str = 'base'

    controls_event_count: bool = False
    preprocessing_cost: str = 'O(num_events)'

    def __init__(
        self,
        T: int,
        height: int,
        width: int,
        polarity_channels: bool = True,
        binarize: bool = True,
        pixel_threshold: int = 0,
    ) -> None:
        self.T = int(T)
        self.height = int(height)
        self.width = int(width)
        self.polarity_channels = bool(polarity_channels)
        self.binarize = bool(binarize)
        self.pixel_threshold = int(pixel_threshold)
        if not self.binarize:
            raise ValueError('pre-attention benchmark encoders must produce binary spikes; set binarize=true.')
        if self.pixel_threshold < 0:
            raise ValueError('encoder.pixel_threshold must be >= 0.')

    @property
    def channels(self) -> int:
        return 2 if self.polarity_channels else 1

    @abstractmethod
    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        """Return a binary numpy tensor [T,C,H,W]."""

    def __call__(self, events: dict[str, Any]) -> torch.Tensor:
        arr = self.encode_np(events)
        # encode_np dovrebbe gia restituire uint8 binario.
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
            'pixel_threshold': self.pixel_threshold,
            'output_shape': [self.T, self.channels, self.height, self.width],
            'controls_event_count': bool(self.controls_event_count),
            'preprocessing_cost': self.preprocessing_cost,
        }

    def _scale_xy(self, events: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # ogni indice i rappresenta un evento
        # e_i = (t[i], x[i], y[i], p[i])
        x = np.asarray(events['x'], dtype=np.int64)
        y = np.asarray(events['y'], dtype=np.int64)
        p = np.asarray(events['p'], dtype=np.int64)
        t = np.asarray(events['t'])
        src_h = int(events.get('height', self.height))
        src_w = int(events.get('width', self.width))

        # Ridimensionamento nearest-neighbor sugli indirizzi eventi; e grezzo ma
        # riproducibile, e non introduce interpolazioni dense.
        if src_w != self.width and src_w > 0:
            x = np.floor(x.astype(np.float64) * self.width / src_w).astype(np.int64)
        if src_h != self.height and src_h > 0:
            y = np.floor(y.astype(np.float64) * self.height / src_h).astype(np.int64)

        # ogni evento cada dentro la griglia target [H, W]
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Se polarity_channels=True => p = 0/1
        #   output encoder: [T, 2, H, W]

        # Se polarity_channels=False => tutte le polarità collassano nel canale 0
        #   output encoder: [T, 1, H, W]
        if self.polarity_channels:
            p = (p > 0).astype(np.int64)
        else:
            p = np.zeros_like(p, dtype=np.int64)
        return t, x, y, p

    def _time_bins(self, t: np.ndarray, T: int | None = None) -> np.ndarray:
        '''
        Restituisce, per ogni evento, il timestep discreto in cui deve cadere.
        Input:
          t.shape = [num_events]
        Output:
          bins.shape = [num_events], con valori in {0, ..., T-1}
        '''
        T = int(T or self.T)
        if t.size == 0:
            return np.zeros(0, dtype=np.int64)
        t = t.astype(np.float64)

        # Ogni sample viene stirato/compresso tra il primo e l'ultimo evento:
        # preserviamo l'ordine temporale relativo, non la durata fisica assoluta.
        t_min = float(t.min())
        t_max = float(t.max())
        if t_max <= t_min:
            return np.zeros_like(t, dtype=np.int64)

        # Mappa [t_min, t_max] -> preserva ordine temporale relativo, non durata assoluta
        bins = np.floor((t - t_min) / (t_max - t_min + 1e-12) * T).astype(np.int64)
        return np.clip(bins, 0, T - 1)
