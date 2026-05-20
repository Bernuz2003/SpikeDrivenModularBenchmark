from __future__ import annotations

import torch
from .attention import IdentityAttention
from pre_attention_benchmark.training.utils import tensor_is_binary


HIDDEN_BOUNDARY_CLASS_NAMES = {
    'ConvBNLIFMaxPoolStage',
    'ConvBNMaxPoolLIFStage',
    'LIFConvBNMaxPoolLIFStage',
    'DepthwiseSeparableStage',
    'MSResidualBlock',
    'StackedStages',
    'DualPathHighFrequencyExtractor',
}


def validate_model_static(model: torch.nn.Module) -> None:
    att = getattr(model, 'attention', None)
    if not isinstance(att, IdentityAttention):
        raise RuntimeError('pre-attention benchmark model must use IdentityAttention only.')
    for name, module in model.named_modules():
        if getattr(module, 'residual_type', None) not in (None, 'ms'):
            raise RuntimeError(f'Forbidden residual at {name}: {getattr(module, "residual_type")}')


def validate_hidden_outputs(model, sample: torch.Tensor, device: torch.device) -> None:
    model.eval()
    violations: list[str] = []
    handles = []

    def hook(name: str):
        def fn(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) and output else output
            if isinstance(out, torch.Tensor) and not tensor_is_binary(out.detach()):
                # Teniamo tutte le violazioni e falliamo alla fine: e molto piu
                # comodo per capire quale blocco ha rotto la pipeline binaria.
                violations.append(f'{name} emitted non-binary hidden output with shape {tuple(out.shape)}')
        return fn

    for name, module in iter_hidden_spike_boundaries(model):
        if name:
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        try:
            sample = sample.to(device)
            if sample.dim() == 4:
                sample = sample.unsqueeze(0)
            if not tensor_is_binary(sample):
                raise RuntimeError('Input spike tensor is not binary.')
            _ = model(sample, validate_hidden_binary=True)
        finally:
            for h in handles:
                h.remove()
    if violations:
        joined = '; '.join(violations[:5])
        more = '' if len(violations) <= 5 else f'; ... {len(violations) - 5} more'
        raise RuntimeError(f'Hidden spike communication violated: {joined}{more}')


def iter_hidden_spike_boundaries(model: torch.nn.Module):
    for name, module in model.named_modules():
        if name in {'feature_extractor', 'attention'}:
            yield name, module
            continue
        if module.__class__.__name__ in HIDDEN_BOUNDARY_CLASS_NAMES:
            yield name, module
