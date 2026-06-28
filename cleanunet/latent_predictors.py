"""
Stage-2 latent predictors.

In Stage 2 the model no longer runs the SSL extractor. The `latent_predictor`
receives the plain CleanUNet bottleneck `latent` (B, C, T) and must reproduce the
Stage-1 `fused_latent` (same shape) that distillation targets. Every predictor here
is a drop-in nn.Module with the contract:

    forward(latent: (B, C, T)) -> predicted_latent: (B, C, T)

so the model's forward stays `predicted_latent = self.latent_predictor(latent)`
regardless of the chosen architecture. Select one via `model.latent_predictor.type`
in the config. `baseline` is the original 2-layer 1x1 conv (default, back-compat).
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
    """#1 Temporal receptive field: stack of dilated residual blocks.

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


class ResidualMLPPredictor(nn.Module):
    """#2 Residual + identity init: predicts only the delta over the input latent.

    The last layer is zero-initialised so the module starts as the identity, which
    keeps the decoded audio valid from epoch 0 (Stage-1 weights are warm-started).
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.act = nn.PReLU()
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class NormPredictor(nn.Module):
    """#3 Normalization: per-frame MLP with GroupNorm for training stability (AMP)."""

    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(num_groups, channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class ConformerPredictor(nn.Module):
    """#4 Depth + global context: a Conformer-style block (self-attention + conv).

    Self-attention supplies the utterance-level (semantic) context that the Stage-1
    semantic FiLM carried; the depthwise conv supplies local (acoustic) context.
    """

    def __init__(self, channels, num_heads=8, ff_mult=2, kernel_size=7,
                 num_groups=8, dropout=0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln_conv = nn.LayerNorm(channels)
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels),
            nn.GroupNorm(num_groups, channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.ln_ff = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.PReLU(),
            nn.Linear(channels * ff_mult, channels),
        )

    def forward(self, x):
        h = x.transpose(1, 2)  # (B, T, C)
        a, _ = self.attn(self.ln_attn(h), self.ln_attn(h), self.ln_attn(h),
                         need_weights=False)
        h = h + a
        c = self.conv(self.ln_conv(h).transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        h = h + c
        h = h + self.ff(self.ln_ff(h))
        return h.transpose(1, 2)  # (B, C, T)


class FiLMPredictor(nn.Module):
    """#5 Predict FiLM (gamma, beta) instead of the raw latent.

    Bakes in the multiplicative structure of the target: predicted = (1+gamma)*x + beta.
    gamma/beta heads are zero-initialised so the module starts as the identity.
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.shared = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.PReLU(),
        )
        self.to_gamma = nn.Conv1d(hidden, channels, kernel_size=1)
        self.to_beta = nn.Conv1d(hidden, channels, kernel_size=1)
        for head in (self.to_gamma, self.to_beta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        h = self.shared(x)
        return (1.0 + self.to_gamma(h)) * x + self.to_beta(h)


_PREDICTORS = {
    'baseline': BaselinePredictor,
    'tcn': TCNPredictor,
    'residual': ResidualMLPPredictor,
    'norm': NormPredictor,
    'conformer': ConformerPredictor,
    'film': FiLMPredictor,
}


def build_latent_predictor(predictor_type, channels, **kwargs):
    """Build a Stage-2 latent predictor by name.

    Args:
        predictor_type: one of 'baseline', 'tcn', 'residual', 'norm', 'conformer', 'film'.
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
