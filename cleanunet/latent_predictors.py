"""
Stage-2 latent predictors.

In Stage 2 the model no longer runs the SSL extractor. The `latent_predictor`
receives the plain CleanUNet bottleneck `latent` (B, C, T) and must reproduce the
Stage-1 `fused_latent` (same shape) that distillation targets. Every predictor here
is a drop-in nn.Module with the contract:

    forward(latent: (B, C, T)) -> predicted_latent: (B, C, T)

so the model's forward stays `predicted_latent = self.latent_predictor(latent)`
regardless of the chosen architecture. Select one via `model.latent_predictor.type`
in the config. `baseline` is the original 2-layer 1x1 conv (back-compat default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselinePredictor(nn.Module):
    """Original Stage-2 predictor: per-frame 1x1 conv MLP (no temporal context)."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class _DilatedResidualBlock(nn.Module):
    """Two dilated convs with a residual connection (TCN building block)."""

    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class TCNPredictor(nn.Module):
    """Temporal receptive field: stack of dilated residual blocks.

    Mirrors the multi-scale dilations of the Stage-1 generator (1,2,4,8) so the
    predictor can infer the contextual modulation that produced the target.
    """

    def __init__(self, channels, dilations=(1, 2, 4, 8), kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_DilatedResidualBlock(channels, d, kernel_size) for d in dilations]
        )
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class SkipFiLMHead(nn.Module):
    """Stage-2 skip-connection FiLM predictor (one per modulated early encoder layer).

    Mirrors the Stage-1 hierarchical ACOUSTIC modulation x = (1 + gamma) * x + beta,
    where (gamma, beta) are per-channel [B, C] and broadcast over time. Pools the
    (unmodulated) skip over time, then regresses 2*C params. The output head is
    zero-initialised so the predictor starts as the identity (gamma = beta = 0),
    leaving the warm-started Stage-1 decoder undisturbed at epoch 0.
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(hidden, 2 * channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, skip):
        pooled = skip.mean(dim=-1)                 # [B, C, T] -> [B, C]
        gamma, beta = self.fc2(self.act(self.fc1(pooled))).chunk(2, dim=1)
        return gamma, beta                         # [B, C], [B, C]


def _match_len(x, target_len):
    """Crop or zero-pad x along time so its last dim equals target_len.

    Transposed convs with stride 2 can over/undershoot by one frame on odd-length
    inputs; this realigns an upsampled tensor to its skip connection.
    """
    if x.size(-1) > target_len:
        return x[..., :target_len]
    if x.size(-1) < target_len:
        return F.pad(x, (0, target_len - x.size(-1)))
    return x


class _ConvBlock(nn.Module):
    """Conv(k=3) + PReLU at constant channels (U-Net stage block)."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class UNetTemporalPredictor(nn.Module):
    """Temporal U-Net: down / process / up with additive skips.

    Reaches a long receptive field cheaply by processing at reduced temporal
    resolution (T -> T/2 -> T/4 for depth=2), then reconstructs with transposed
    convs and additive skip connections. A small conv at T/4 already spans a long
    stretch of the original signal, so it captures the multi-scale structure of the
    Stage-1 fused latent with fewer params than an equally wide dilated stack.
    """

    def __init__(self, channels, depth=2, kernel_size=3):
        super().__init__()
        self.enc = nn.ModuleList([_ConvBlock(channels, kernel_size) for _ in range(depth)])
        self.down = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)
             for _ in range(depth)]
        )
        self.bottleneck = _ConvBlock(channels, kernel_size)
        self.up = nn.ModuleList(
            [nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
             for _ in range(depth)]
        )
        self.dec = nn.ModuleList([_ConvBlock(channels, kernel_size) for _ in range(depth)])
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        skips = []
        h = x
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = _match_len(up(h), skip.size(-1))
            h = dec(h + skip)
        return self.proj(h)


_PREDICTORS = {
    'baseline': BaselinePredictor,
    'tcn': TCNPredictor,
    'unet': UNetTemporalPredictor,
}


def build_latent_predictor(predictor_type, channels, **kwargs):
    """Build a Stage-2 latent predictor by name.

    Args:
        predictor_type: one of 'baseline', 'tcn', 'unet'.
        channels: latent dim (bottleneck channels).
        **kwargs: forwarded to the selected predictor's constructor.
    """
    key = (predictor_type or 'baseline').lower()
    if key not in _PREDICTORS:
        raise ValueError(
            f"Unknown latent_predictor type '{predictor_type}'. "
            f"Use one of: {sorted(_PREDICTORS)}."
        )
    return _PREDICTORS[key](channels, **kwargs)
