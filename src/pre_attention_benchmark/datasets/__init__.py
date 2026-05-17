from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .tonic_dvs import OptionalTonicDataset
from pre_attention_benchmark.encoders.augmentations import apply_event_augmentations, apply_spike_augmentations
from pre_attention_benchmark.encoders.binary import (
    FixedTimeBinaryEncoder,
    FixedEventCountBinaryEncoder,
    BinaryVoxelGridEncoder,
    TemporalSubsampleBinaryEncoder,
)


class EncodedEventDataset(Dataset):
    def __init__(
        self,
        event_dataset: Dataset,
        encoder,
        augmentations: dict[str, Any] | None = None,
        seed: int = 42,
        split_name: str = 'unknown',
        split_indices: list[int] | None = None,
    ) -> None:
        self.event_dataset = event_dataset
        self.encoder = encoder
        self.augmentations = augmentations or {}
        self.seed = int(seed)
        self.split_name = split_name
        self.split_indices = split_indices

    def __len__(self) -> int:
        return len(self.event_dataset)

    def __getitem__(self, idx: int):
        # La seed dipende dall'indice: cosi le perturbazioni restano riproducibili
        # anche se il DataLoader cambia ordine o numero di worker.
        rng = np.random.default_rng(self.seed + idx)
        events = self.event_dataset[idx]
        events = apply_event_augmentations(events, self.augmentations, rng)
        spikes = self.encoder(events)
        spikes = apply_spike_augmentations(spikes, self.augmentations, rng)
        label = int(events['label'])
        return spikes, torch.tensor(label, dtype=torch.long)


def build_encoder(cfg: dict[str, Any], dataset_cfg: dict[str, Any]):
    name = cfg.get('name', 'fixed_time_binary')
    # Le dimensioni arrivano dal dataset, non dall'encoder: evita config incoerenti
    # quando si passa da CIFAR10-DVS a DVS128 Gesture.
    h = int(dataset_cfg.get('height', cfg.get('height', 128)))
    w = int(dataset_cfg.get('width', cfg.get('width', 128)))
    common = dict(
        T=int(cfg.get('T', 8)),
        height=h,
        width=w,
        polarity_channels=cfg.get('polarity_channels', True),
        binarize=cfg.get('binarize', True),
        duration_us=cfg.get('duration_us', dataset_cfg.get('duration_us')),
        time_reference=cfg.get('time_reference', dataset_cfg.get('time_reference', 'sample_start')),
    )
    if name == 'fixed_time_binary':
        return FixedTimeBinaryEncoder(**common)
    if name == 'fixed_event_count_binary':
        return FixedEventCountBinaryEncoder(**common)
    if name == 'binary_voxel_grid':
        return BinaryVoxelGridEncoder(**common)
    if name == 'temporal_subsample_binary':
        return TemporalSubsampleBinaryEncoder(**common, T_source=cfg.get('T_source'))
    raise ValueError(f'Unknown encoder: {name}')


def _split_indices(n: int, n_train: int | None, n_val: int | None, seed: int) -> tuple[list[int], list[int]]:
    # CIFAR10-DVS in tonic non espone uno split train/test ufficiale: usiamo uno
    # split deterministico e lo salviamo negli artifact per rendere la run ripetibile.
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    if n_val is None:
        n_val = max(1, int(0.1 * n))
    n_val = max(1, min(int(n_val), n - 1))
    if n_train is None:
        n_train = n - n_val
    n_train = max(1, min(int(n_train), n - n_val))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    return train_idx, val_idx


def _subset(ds: Dataset, indices: list[int]) -> Dataset:
    return Subset(ds, indices)


def build_datasets(
    cfg: dict[str, Any],
    train_augmentations_override: dict[str, Any] | None = None,
    val_augmentations_override: dict[str, Any] | None = None,
):
    dataset_cfg = cfg['dataset']
    enc = build_encoder(cfg['encoder'], dataset_cfg)
    name = dataset_cfg.get('name', 'cifar10_dvs').lower()
    seed = int(cfg['experiment'].get('seed', 42))
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None
    if name in {'cifar10_dvs', 'cifar10-dvs', 'cifar10dvs'}:
        root = dataset_cfg.get('root')
        if root is None:
            raise ValueError('dataset.root is required for CIFAR10-DVS')
        full = OptionalTonicDataset(name, root=root, train=True, height=dataset_cfg.get('height', 128), width=dataset_cfg.get('width', 128))
        n_train = dataset_cfg.get('num_train')
        n_val = dataset_cfg.get('num_val')
        train_indices, val_indices = _split_indices(
            len(full),
            int(n_train) if n_train is not None else None,
            int(n_val) if n_val is not None else None,
            seed,
        )
        train_events, val_events = _subset(full, train_indices), _subset(full, val_indices)
    elif name in {'dvs128_gesture', 'dvs_gesture', 'dvs128gesture'}:
        root = dataset_cfg.get('root')
        if root is None:
            raise ValueError('dataset.root is required for DVS128 Gesture')
        full_train = OptionalTonicDataset(name, root=root, train=True, height=dataset_cfg.get('height', 128), width=dataset_cfg.get('width', 128))
        full_val = OptionalTonicDataset(name, root=root, train=False, height=dataset_cfg.get('height', 128), width=dataset_cfg.get('width', 128))
        n_train = dataset_cfg.get('num_train')
        n_val = dataset_cfg.get('num_val')
        train_indices = list(range(len(full_train)))
        val_indices = list(range(len(full_val)))
        if n_train is not None:
            train_indices = train_indices[:max(1, min(int(n_train), len(train_indices)))]
        if n_val is not None:
            val_indices = val_indices[:max(1, min(int(n_val), len(val_indices)))]
        train_events, val_events = _subset(full_train, train_indices), _subset(full_val, val_indices)
    else:
        raise ValueError(f'Unknown dataset: {name}')

    train_aug = dataset_cfg.get('train_augmentations', {})
    # Le perturbazioni di robustezza devono colpire la validation, non sporcare il
    # training pulito, salvo quando train_augmentations e impostato esplicitamente.
    val_aug = dataset_cfg.get('val_augmentations', dataset_cfg.get('augmentations', {}))
    if train_augmentations_override is not None:
        train_aug = train_augmentations_override
    if val_augmentations_override is not None:
        val_aug = val_augmentations_override

    train_ds = EncodedEventDataset(train_events, enc, augmentations=train_aug, seed=seed, split_name='train', split_indices=train_indices)
    val_ds = EncodedEventDataset(val_events, enc, augmentations=val_aug, seed=seed + 1, split_name='val', split_indices=val_indices)
    return train_ds, val_ds, enc


def dataset_metadata(train_ds: EncodedEventDataset, val_ds: EncodedEventDataset, encoder) -> dict[str, Any]:
    def split(ds: EncodedEventDataset) -> dict[str, Any]:
        return {
            'name': ds.split_name,
            'num_samples': len(ds),
            'indices': ds.split_indices,
            'augmentations': ds.augmentations,
        }

    return {
        'encoder': encoder.describe() if hasattr(encoder, 'describe') else {},
        'splits': {
            'train': split(train_ds),
            'val': split(val_ds),
        },
    }
