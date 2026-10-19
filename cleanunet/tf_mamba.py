"""
N2 of IMPROVEMENT_PLAN.md: dual-path time-frequency CleanSpecNet with Mamba (SEMamba / MP-SENet style).

    features (B, G, F, T)                       G = 1 (magnitude) or 3 (+ cos / sin of the noisy phase)
      -> Conv2d 1x1 + causal 2-D conv block (full_res_channels at full resolution, to keep it cheap)
      -> frequency downsampling (stride 2 per stage)             F: 513 -> 257 -> 129
      -> n_blocks x [Mamba over time (causal, one sequence per frequency bin)
                     + Mamba over frequency (bidirectional, one sequence per frame)]
      -> magnitude decoder (frequency upsampling with encoder skips) -> mask / residual / direct
      -> optional phase decoder (pseudo real / imaginary parts -> atan2)

Time causality: every convolution is left-padded in time, the normalization is per frame and the time
Mamba is unidirectional, so frame t only depends on frames <= t. The frequency Mamba only looks at the
current frame and may therefore be bidirectional even in a streaming model.

`hidden` (used by the CleanUNet fusion) is a per-frame vector: the last block is averaged into
`hidden_subbands` frequency sub-bands and projected to `hidden_dim`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cleanspecnet import OUTPUT_MODES, apply_output_mode, cumulative_norm, init_output_layer, phase_from
from .sequence import build_mamba_layers, resolve_backend, run_mamba_layers


class FrameNorm(nn.Module):
    """Normalization over (channels, frequency) of every frame of a (B, C, F, T) tensor (causal),
    with a per-channel affine transform."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CausalConv2d(nn.Module):
    """Conv2d over (frequency, time): symmetric padding in frequency, left padding in time (if causal)."""

    def __init__(self, cin: int, cout: int, kernel=(3, 3), stride=(1, 1), causal: bool = True):
        super().__init__()
        kf, kt = kernel
        self.causal = causal
        self.pad_t = (kt - 1, 0) if causal else ((kt - 1) // 2, kt // 2)
        self.conv = nn.Conv2d(cin, cout, kernel, stride=stride, padding=(kf // 2, 0))

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_t))


class ConvBlock2d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel=(3, 3), stride=(1, 1), causal: bool = True):
        super().__init__()
        self.conv = CausalConv2d(cin, cout, kernel, stride, causal)
        self.norm = FrameNorm(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpBlock2d(nn.Module):
    """Frequency upsampling by 2 (F -> 2F - 1 for F = 2^k + 1); time resolution unchanged."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.norm = FrameNorm(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x, n_freq: int):
        x = self.up(x)
        if x.shape[2] < n_freq:
            x = F.pad(x, (0, 0, 0, n_freq - x.shape[2]), mode="replicate")
        return self.act(self.norm(x[:, :, :n_freq]))


class TFMambaBlock(nn.Module):
    """Mamba along time (per frequency bin) then along frequency (per frame); each stack is a pre-norm
    residual Mamba block with a final LayerNorm (same layout as the other Mamba stacks of the repo)."""

    def __init__(self, channels: int, causal: bool = True, d_state: int = 16, d_conv: int = 4, expand: float = 2,
                 backend: str = "auto"):
        super().__init__()
        kw = dict(d_state=d_state, d_conv=d_conv, expand=expand, backend=backend)
        self.time_layers, self.time_norm = build_mamba_layers(channels, 1, bidirectional=not causal, **kw)
        self.freq_layers, self.freq_norm = build_mamba_layers(channels, 1, bidirectional=True, **kw)

    def forward(self, x: Tensor) -> Tensor:
        B, C, Fq, T = x.shape
        t = x.permute(0, 2, 3, 1).reshape(B * Fq, T, C).contiguous()
        t = run_mamba_layers(self.time_layers, self.time_norm, t)
        x = t.reshape(B, Fq, T, C).permute(0, 3, 1, 2)
        f = x.permute(0, 3, 2, 1).reshape(B * T, Fq, C).contiguous()
        f = run_mamba_layers(self.freq_layers, self.freq_norm, f)
        return f.reshape(B, T, Fq, C).permute(0, 3, 2, 1).contiguous()


class TFMambaSpecNet(nn.Module):
    """Drop-in replacement for CleanSpecNet (same `run` / `forward` interface)."""

    def __init__(
        self,
        input_channels: int = 513,
        feature_groups: int = 1,
        channels: int = 64,
        full_res_channels: int = 16,
        n_blocks: int = 4,
        freq_downsample: int = 2,
        time_kernel: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2,
        causal: bool = True,
        output_mode: str = "mask",
        predict_phase: bool = False,
        input_norm: str = "none",
        hidden_dim: int = 512,
        hidden_subbands: int = 16,
        backend: str = "auto",
    ):
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be direct | residual | mask, got '{output_mode}'")
        if input_norm not in ("none", "cumulative"):
            raise ValueError(f"input_norm must be none | cumulative, got '{input_norm}'")
        self.n_freq = input_channels
        self.feature_groups = feature_groups
        self.output_mode = output_mode
        self.predicts_phase = predict_phase
        self.input_norm = input_norm
        self.causal = causal
        self.hidden_dim = hidden_dim
        self.hidden_subbands = hidden_subbands
        self.backbone_type = "tf_mamba"
        backend = resolve_backend(backend)
        if freq_downsample < 1:
            raise ValueError("TFMambaSpecNet needs freq_downsample >= 1")
        C, C0 = channels, full_res_channels
        # channels per frequency level: C0 at full resolution, C after the first downsampling
        level_ch = [C0] + [C] * freq_downsample

        self.stem = nn.Sequential(ConvBlock2d(feature_groups, C0, (1, 1), causal=causal),
                                  ConvBlock2d(C0, C0, (3, time_kernel), causal=causal))
        self.down = nn.ModuleList([ConvBlock2d(level_ch[k], level_ch[k + 1], (3, time_kernel), stride=(2, 1),
                                               causal=causal) for k in range(freq_downsample)])
        self.blocks = nn.ModuleList([TFMambaBlock(C, causal, d_state, d_conv, expand, backend)
                                     for _ in range(n_blocks)])
        # decoders go from the deepest level back to full resolution
        ups = [(level_ch[k + 1], level_ch[k]) for k in reversed(range(freq_downsample))]
        self.mag_up = nn.ModuleList([UpBlock2d(a, b) for a, b in ups])
        self.mag_out = nn.Conv2d(C0, 1, kernel_size=1)
        init_output_layer(self.mag_out, output_mode)
        if predict_phase:
            self.phase_up = nn.ModuleList([UpBlock2d(a, b) for a, b in ups])
            self.phase_out = nn.Conv2d(C0, 2, kernel_size=1)
        self.hidden_proj = nn.Linear(C * hidden_subbands, hidden_dim)

    def forward(self, x: Tensor, return_hidden: bool = False):
        out = self.run(x)
        return (out["spec"], out["hidden"]) if return_hidden else out["spec"]

    def _decode(self, z, skips, ups):
        for up, skip in zip(ups, reversed(skips)):
            z = up(z, skip.shape[2]) + skip
        return z

    def run(self, x: Tensor) -> dict:
        """x: (B, G*F, T) features, magnitude first."""
        B, _, T = x.shape
        spec = x[:, :self.n_freq]
        feats = x.view(B, self.feature_groups, self.n_freq, T)
        if self.input_norm == "cumulative":
            feats = torch.cat([cumulative_norm(feats[:, :1]), feats[:, 1:]], dim=1)
        z = self.stem(feats)
        skips = []
        for down in self.down:
            skips.append(z)
            z = down(z)
        for block in self.blocks:
            z = block(z)

        # per-frame summary for the fusion: average into sub-bands, then project
        pooled = F.adaptive_avg_pool2d(z, (self.hidden_subbands, T))                 # (B, C, S, T)
        hidden = self.hidden_proj(pooled.permute(0, 3, 1, 2).reshape(B, T, -1))

        mag = self.mag_out(self._decode(z, skips, self.mag_up)).squeeze(1)
        out = {"spec": apply_output_mode(mag, spec, self.output_mode), "hidden": hidden, "phase": None}
        if self.predicts_phase:
            out["phase"] = phase_from(self.phase_out(self._decode(z, skips, self.phase_up)).float())
        return out
