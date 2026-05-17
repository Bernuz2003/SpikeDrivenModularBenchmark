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
    track_metrics = True
    op_class = 'MaxPoolCompare'

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.pool._pre_attention_skip_metrics = True
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, bt = time_to_batch(x)
        y = self.pool(xb)
        return batch_to_time(y, bt)


class ConvBNLIFMaxPoolStage(nn.Module):
    emits_hidden_spikes = True
    op_class = 'Conv-BN-LIF-MaxPool'

    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.convbn = ConvBN(in_ch, out_ch)
        self.lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='lif')
        self.pool = TDMaxPool(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.lif(self.convbn(x)))


class ConvBNMaxPoolLIFStage(nn.Module):
    emits_hidden_spikes = True
    op_class = 'Conv-BN-MaxPool-LIF'

    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.convbn = ConvBN(in_ch, out_ch)
        self.pool = TDMaxPool(2, 2)
        self.lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='lif')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.pool(self.convbn(x)))


class LIFConvBNMaxPoolLIFStage(nn.Module):
    emits_hidden_spikes = True
    op_class = 'LIF-Conv-BN-MaxPool-LIF'

    def __init__(self, in_ch: int, out_ch: int, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.pre_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='pre_lif')
        self.convbn = ConvBN(in_ch, out_ch)
        self.pool = TDMaxPool(2, 2)
        self.post_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='post_lif')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_lif(self.pool(self.convbn(self.pre_lif(x))))


class DepthwiseSeparableStage(nn.Module):
    emits_hidden_spikes = True
    op_class = 'SpikingDepthwiseSeparableConv'

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.pre_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='pre_lif')
        self.dw = ConvBN(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.dw_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='dw_lif')
        self.pw = ConvBN(in_ch, out_ch, kernel_size=1, padding=0)
        self.pw_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='pw_lif')
        self.pool = TDMaxPool(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_lif(x)
        x = self.dw_lif(self.dw(x))
        x = self.pw_lif(self.pw(x))
        x = self.pool(x)
        return x


class MSResidualBlock(nn.Module):
    """Membrane-level fusion followed by LIF, with binary output only.

    This is a practical implementation of the MS-residual principle:
    non-binary sums stay inside the block as membrane current and are immediately
    thresholded before communicating to the next block.
    """

    residual_type = 'ms'
    emits_hidden_spikes = True
    op_class = 'MSResidual'

    def __init__(self, channels: int, surrogate_alpha: float = 4.0) -> None:
        super().__init__()
        self.main = ConvBN(channels, channels)
        self.out_lif = MultiStepLIF(surrogate_alpha=surrogate_alpha, name='ms_out_lif')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Qui la somma non e comunicazione tra blocchi: e corrente di membrana.
        # Il LIF successivo riporta l'uscita a spike binari.
        membrane_current = self.main(x) + x
        return self.out_lif(membrane_current)
