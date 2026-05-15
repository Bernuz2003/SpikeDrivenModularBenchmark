from __future__ import annotations

import torch
from .attention import IdentityAttention
from m1_benchmark.training.utils import tensor_is_binary


def validate_model_static(model: torch.nn.Module) -> None:
    att = getattr(model, 'attention', None)
    if not isinstance(att, IdentityAttention):
        raise RuntimeError('Milestone 1 model must use IdentityAttention only.')
    for name, module in model.named_modules():
        if getattr(module, 'residual_type', None) not in (None, 'ms'):
            raise RuntimeError(f'Forbidden residual at {name}: {getattr(module, "residual_type")}')


def validate_hidden_outputs(model, sample: torch.Tensor, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        sample = sample.to(device)
        if sample.dim() == 4:
            sample = sample.unsqueeze(0)
        if not tensor_is_binary(sample):
            raise RuntimeError('Input spike tensor is not binary.')
        _ = model(sample, validate_hidden_binary=True)
