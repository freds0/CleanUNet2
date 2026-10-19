"""
O8 / P8: late time-frequency fusion (see IMPROVEMENT_PLAN.md).

The CleanUNet output y is analysed with the model STFT, a bounded mask computed from
[compressed |Y|, spectral condition (e.g. CleanSpecNet output), compressed noisy |X|] is applied
to Y, and the result is resynthesized:

    y' = iSTFT(M * STFT(y)),   M = 1 + tanh(delta),   delta = 0 at initialization (identity).

The magnitude-only mask keeps the phase of y. The STFT window adds lookahead, so this module
is meant for offline use.
"""

import torch
import torch.nn as nn

from .fusion import CausalConv1d, ChannelNorm, init_last


class SpectralPostFilter(nn.Module):
    def __init__(self, n_freq: int, cond_dim: int, channels: int = 256, kernel: int = 3, zero_init: str = "last"):
        super().__init__()
        in_dim = 2 * n_freq + cond_dim
        self.net = nn.Sequential(
            ChannelNorm(in_dim),
            CausalConv1d(in_dim, channels, kernel), nn.GELU(),
            nn.Conv1d(channels, channels, 1), nn.GELU(),
            nn.Conv1d(channels, n_freq, 1),
        )
        init_last(self.net[-1], zero_init)

    def forward(self, y: torch.Tensor, cond: torch.Tensor, noisy_spec: torch.Tensor, spectrogram) -> torch.Tensor:
        """y: (B, 1, L) waveform; cond: (B, D, T); noisy_spec: compressed noisy magnitude (B, F, T)."""
        L = y.shape[-1]
        with torch.autocast(device_type=y.device.type, enabled=False):
            Y = spectrogram.complex_stft(y.float())
            y_spec = spectrogram.compress(Y.abs() * spectrogram.scale)
            T = Y.shape[-1]
            feats = torch.cat([y_spec, cond.float()[..., :T], noisy_spec.float()[..., :T]], dim=1)
        delta = self.net(feats)
        with torch.autocast(device_type=y.device.type, enabled=False):
            mask = 1.0 + torch.tanh(delta.float())
            return spectrogram.istft(Y * mask, L).to(y.dtype)
