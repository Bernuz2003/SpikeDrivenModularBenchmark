from __future__ import annotations

import torch
from torch import nn


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Surrogata sigmoidale: forward duro e binario, backward abbastanza liscio
        # da poter addestrare con PyTorch standard.
        sig = torch.sigmoid(alpha * input_)
        grad = alpha * sig * (1.0 - sig)
        return grad_output * grad, None


def spike_fn(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    return SpikeFunction.apply(x, float(alpha))


class MultiStepLIF(nn.Module):
    """Simple multi-step LIF neuron.

    Input shape: [B,T,...]. Output is exactly binary in forward pass and uses
    a sigmoid surrogate gradient in backward pass.
    """

    def __init__(self, beta: float = 0.5, threshold: float = 1.0, reset: str = 'hard', surrogate_alpha: float = 4.0, name: str | None = None) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.reset = reset
        self.surrogate_alpha = float(surrogate_alpha)
        self.name = name or 'lif'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f'MultiStepLIF expects [B,T,...], got shape {tuple(x.shape)}')
        B, T = x.shape[:2]
        v = torch.zeros_like(x[:, 0])
        spikes = []
        for t in range(T):
            # Stato locale al layer, non comunicato fuori: all'esterno escono solo spike.
            v = self.beta * v + x[:, t]
            s = spike_fn(v - self.threshold, self.surrogate_alpha)
            if self.reset == 'hard':
                v = v * (1.0 - s)
            elif self.reset == 'soft':
                v = v - s * self.threshold
            else:
                raise ValueError(f'Unknown reset mode {self.reset!r}')
            spikes.append(s)
        return torch.stack(spikes, dim=1)
