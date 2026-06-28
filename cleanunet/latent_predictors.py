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


_PREDICTORS = {
    'baseline': BaselinePredictor,
    'tcn': TCNPredictor,
}


def build_latent_predictor(predictor_type, channels, **kwargs):
    """Build a Stage-2 latent predictor by name.

    Args:
        predictor_type: one of 'baseline', 'tcn'.
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
