from __future__ import annotations

import torch
from torch import nn
from .lif import MultiStepLIF


def time_to_batch(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if x.dim() != 5:
        raise ValueError(f'Expected [B,T,C,H,W], got {tuple(x.shape)}')
    B, T, C, H, W = x.shape
    return x.reshape(B * T, C, H, W), (B, T)


def batch_to_time(x: torch.Tensor, bt: tuple[int, int]) -> torch.Tensor:
    B, T = bt
    C, H, W = x.shape[1:]
    return x.reshape(B, T, C, H, W)


class TimeDistributed2d(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applichiamo moduli 2D riusando il batch come asse B*T; piu semplice e
        # meno fragile che riscrivere ogni layer in versione temporale.
        xb, bt = time_to_batch(x)
        y = self.module(xb)
        return batch_to_time(y, bt)


class ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None, groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, bt = time_to_batch(x)
        y = self.bn(self.conv(xb))
        return batch_to_time(y, bt)


class TDMaxPool(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, bt = time_to_batch(x)
        y = self.pool(xb)
        return batch_to_time(y, bt)


class MSProjectionShortcut(nn.Module):
    """Proiezione della shortcut MS quando canali o risoluzione cambiano."""

    residual_type = 'ms'

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.proj = ConvBN(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvBNLIFMaxPoolStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0, use_ms_residual: bool = False) -> None:
        super().__init__()
        self.convbn = ConvBN(in_ch, out_ch)
        self.lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='lif')
        self.pool = TDMaxPool(2, 2)
        self.shortcut = MSProjectionShortcut(in_ch, out_ch, stride=2) if use_ms_residual else None
        self.out_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='ms_out_lif') if use_ms_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(self.lif(self.convbn(x)))
        if self.shortcut is None:
            return y
        # Somma nel dominio di membrana dello stesso stage, poi ribinarizzazione.
        return self.out_lif(y + self.shortcut(x))


class ConvBNMaxPoolLIFStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0, use_ms_residual: bool = False) -> None:
        super().__init__()
        self.convbn = ConvBN(in_ch, out_ch)
        self.pool = TDMaxPool(2, 2)
        self.lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='lif')
        self.shortcut = MSProjectionShortcut(in_ch, out_ch, stride=2) if use_ms_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(self.convbn(x))
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        return self.lif(y)


class LIFConvBNLIFMaxPoolStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0, use_ms_residual: bool = False) -> None:
        super().__init__()
        self.pre_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='pre_lif')
        self.convbn = ConvBN(in_ch, out_ch)
        self.post_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='post_lif')
        self.pool = TDMaxPool(2, 2)
        self.shortcut = MSProjectionShortcut(in_ch, out_ch, stride=2) if use_ms_residual else None
        self.out_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='ms_out_lif') if use_ms_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(self.post_lif(self.convbn(self.pre_lif(x))))
        if self.shortcut is None:
            return y
        return self.out_lif(y + self.shortcut(x))


class StridedConvStemStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0, use_ms_residual: bool = False) -> None:
        super().__init__()
        self.down = ConvBN(in_ch, out_ch, stride=2)
        self.lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='lif')
        self.shortcut = MSProjectionShortcut(in_ch, out_ch, stride=2) if use_ms_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling appreso: niente MaxPool, la riduzione spaziale la decide
        # una sola convoluzione stride 2. La shortcut MS viene sommata prima del LIF.
        y = self.down(x)
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        return self.lif(y)
