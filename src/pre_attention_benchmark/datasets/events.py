from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EventSample:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: int
    height: int
    width: int

    def as_dict(self) -> dict:
        return {
            't': self.t,
            'x': self.x,
            'y': self.y,
            'p': self.p,
            'label': self.label,
            'height': self.height,
            'width': self.width,
        }


def normalize_event_dict(raw: dict | tuple, height: int | None = None, width: int | None = None) -> dict:
    """Convert common event representations to a standard event dict."""
    if isinstance(raw, tuple) and len(raw) == 2:
        events, label = raw
    else:
        events, label = raw, raw.get('label', 0) if isinstance(raw, dict) else 0

    if isinstance(events, dict):
        # Accettiamo alias comuni per tempo e polarita: tonic e dataset custom non
        # sempre usano gli stessi nomi di campo.
        t = np.asarray(events.get('t', events.get('time', events.get('timestamp', []))))
        x = np.asarray(events.get('x', []))
        y = np.asarray(events.get('y', []))
        p = np.asarray(events.get('p', events.get('polarity', [])))
        h = int(events.get('height', height if height is not None else (int(y.max()) + 1 if y.size else 1)))
        w = int(events.get('width', width if width is not None else (int(x.max()) + 1 if x.size else 1)))
        label = int(events.get('label', label))
        return {'t': t, 'x': x, 'y': y, 'p': p, 'label': label, 'height': h, 'width': w}

    # Array strutturato usato da tonic.
    arr = np.asarray(events)
    if arr.dtype.names:
        names = arr.dtype.names
        t_name = 't' if 't' in names else 'time' if 'time' in names else 'timestamp'
        p_name = 'p' if 'p' in names else 'polarity'
        t = np.asarray(arr[t_name])
        x = np.asarray(arr['x'])
        y = np.asarray(arr['y'])
        p = np.asarray(arr[p_name])
        h = int(height if height is not None else (int(y.max()) + 1 if y.size else 1))
        w = int(width if width is not None else (int(x.max()) + 1 if x.size else 1))
        return {'t': t, 'x': x, 'y': y, 'p': p, 'label': int(label), 'height': h, 'width': w}

    raise TypeError(f'Unsupported event representation: {type(raw)}')
