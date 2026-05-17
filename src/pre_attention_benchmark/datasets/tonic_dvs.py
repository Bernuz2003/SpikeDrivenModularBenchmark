from __future__ import annotations

from typing import Any
from torch.utils.data import Dataset
from .events import normalize_event_dict


class OptionalTonicDataset(Dataset):
    """Thin optional adapter for tonic datasets.

    Install tonic and set dataset.name to cifar10_dvs or dvs128_gesture.
    """

    def __init__(self, name: str, root: str, train: bool = True, height: int = 128, width: int = 128) -> None:
        try:
            import tonic  # type: ignore
        except Exception as e:
            raise ImportError(
                "Dataset '%s' requires the optional package 'tonic'. Install it with `pip install tonic`." % name
            ) from e

        name_l = name.lower()
        # Tonic gestisce download/cache dentro save_to; noi ci limitiamo a
        # normalizzare l'interfaccia degli eventi per il resto della pipeline.
        if name_l in {'cifar10_dvs', 'cifar10-dvs', 'cifar10dvs'}:
            self.ds = tonic.datasets.CIFAR10DVS(save_to=root)
        elif name_l in {'dvs128_gesture', 'dvs_gesture', 'dvs128gesture'}:
            self.ds = tonic.datasets.DVSGesture(save_to=root, train=train)
        else:
            raise ValueError(f'Unsupported tonic dataset: {name}')
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return normalize_event_dict(self.ds[idx], height=self.height, width=self.width)
