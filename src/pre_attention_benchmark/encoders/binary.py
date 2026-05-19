from __future__ import annotations

from typing import Any
import numpy as np
from .base import EventEncoder


def _threshold_counts(counts: np.ndarray, pixel_threshold: int) -> np.ndarray:
    # Istogramma -> soglia -> spike. Con threshold 0 basta almeno un evento.
    return (counts > int(pixel_threshold)).astype(np.uint8)


class FixedTimeBinaryEncoder(EventEncoder):
    '''
    Divide la durata temporale del sample in T intervalli temporali
    e mette ogni evento nel bin temporale corrispondente.
    '''
    name = 'fixed_time_binary'
    controls_event_count = False

    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        counts = np.zeros((self.T, self.channels, self.height, self.width), dtype=np.uint32)
        t, x, y, p = self._scale_xy(events)
        bins = self._time_bins(t)
        if bins.size:
            # np.add.at gestisce collisioni multiple nello stesso voxel.
            np.add.at(counts, (bins, p, y, x), 1)
        return _threshold_counts(counts, self.pixel_threshold)


class FixedEventCountBinaryEncoder(EventEncoder):
    '''
    Divide la sequenza ordinata di eventi in T gruppi con circa lo stesso numero di eventi.
    '''
    name = 'fixed_event_count_binary'
    controls_event_count = True

    def encode_np(self, events: dict[str, Any]) -> np.ndarray:
        counts = np.zeros((self.T, self.channels, self.height, self.width), dtype=np.uint32)
        t, x, y, p = self._scale_xy(events)
        n = t.size
        if n == 0:
            return _threshold_counts(counts, self.pixel_threshold)
        order = np.argsort(t)
        x, y, p = x[order], y[order], p[order]
        # Equalizza il numero di eventi per timestep => perdiamo durata assoluta,
        # controlla meglio la densita in input
        chunks = np.array_split(np.arange(n), self.T)
        for ti, idx in enumerate(chunks):
            if idx.size:
                np.add.at(counts[ti], (p[idx], y[idx], x[idx]), 1)
        return _threshold_counts(counts, self.pixel_threshold)
