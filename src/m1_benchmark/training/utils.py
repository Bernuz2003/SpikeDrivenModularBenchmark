from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any
import json
import numpy as np
import torch


def configure_torch_runtime(num_threads: int | None = 1) -> None:
    # In small CPU smoke tests PyTorch's default high thread count can be much slower
    # than single-threaded execution because of scheduling overhead. Full experiments
    # can override this from YAML with training.torch_num_threads.
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(max(1, int(num_threads)))
        except RuntimeError:
            pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device_cfg: str = 'auto') -> torch.device:
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_cfg)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tensor_is_binary(x: torch.Tensor, tol: float = 1e-6) -> bool:
    if x.numel() == 0:
        return True
    return bool(torch.all((x >= -tol) & (x <= 1 + tol) & ((x - x.round()).abs() <= tol)).item())


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
