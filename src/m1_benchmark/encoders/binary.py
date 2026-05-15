from __future__ import annotations

from typing import Any
import numpy as np
from .base import EventEncoder


class FixedTimeBinaryEncoder(EventEncoder):
    name = 'fixed_time_binary'

    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        out = np.zeros((self.T, self.channels, self.height, self.width), dtype=np.uint8)
        t, x, y, p = self._scale_xy(events)
        bins = self._time_bins(t)
        if bins.size:
            out[bins, p, y, x] = 1
        return out


class FixedEventCountBinaryEncoder(EventEncoder):
    name = 'fixed_event_count_binary'

    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        out = np.zeros((self.T, self.channels, self.height, self.width), dtype=np.uint8)
        t, x, y, p = self._scale_xy(events)
        n = t.size
        if n == 0:
            return out
        order = np.argsort(t)
        x, y, p = x[order], y[order], p[order]
        chunks = np.array_split(np.arange(n), self.T)
        for ti, idx in enumerate(chunks):
            if idx.size:
                out[ti, p[idx], y[idx], x[idx]] = 1
        return out


class BinaryVoxelGridEncoder(FixedTimeBinaryEncoder):
    name = 'binary_voxel_grid'


class TemporalSubsampleBinaryEncoder(EventEncoder):
    name = 'temporal_subsample_binary'

    def __init__(self, *args: Any, T_source: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.T_source = int(T_source or self.T * 2)

    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        tmp = np.zeros((self.T_source, self.channels, self.height, self.width), dtype=np.uint8)
        t, x, y, p = self._scale_xy(events)
        bins = self._time_bins(t, T=self.T_source)
        if bins.size:
            tmp[bins, p, y, x] = 1
        # Max-aggregate source bins into target bins, preserving binary output.
        groups = np.array_split(np.arange(self.T_source), self.T)
        out = np.zeros((self.T, self.channels, self.height, self.width), dtype=np.uint8)
        for ti, idx in enumerate(groups):
            if idx.size:
                out[ti] = tmp[idx].max(axis=0)
        return out
