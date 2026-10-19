"""CleanSpecNet: the spectrogram denoiser of CleanUNet2.

It maps a compressed noisy magnitude (B, F, T) to an enhanced one with a front end (convolutional blocks or a
per-frame linear projection), a sequence backbone over time frames (Transformer, Mamba, Mamba2 or LSTM, causal or
bidirectional) and a 1-D output layer. The output can be a direct estimate, a residual correction or a bounded mask.
``safe_atan2`` gives the optional phase head a bounded gradient.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


OUTPUT_MODES = ("direct", "residual", "mask")


def init_output_layer(layer: nn.Module, mode: str):
    """residual: zero correction (identity); mask: sigmoid(3) ~ 0.95 (near identity); direct: default init."""
    if mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be direct | residual | mask, got '{mode}'")
    if mode == "residual":
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    elif mode == "mask":
        nn.init.zeros_(layer.weight)
        nn.init.constant_(layer.bias, 3.0)


def apply_output_mode(out: Tensor, spec: Tensor, mode: str) -> Tensor:
    """Turn the raw network output into the denoised (compressed) spectrogram."""
    if mode == "residual":
        return spec + out
    if mode == "mask":
        return spec * torch.sigmoid(out)
    return out


def cumulative_norm(mag: Tensor, eps: float = 1e-4) -> Tensor:
    """Causal input normalization (FullSubNet style): divide by the running mean over time of the
    frequency-averaged magnitude. mag: (B, F, T) or (B, C, F, T)."""
    frame_mean = mag.mean(dim=-2, keepdim=True)                                 # (..., 1, T)
    steps = torch.arange(1, mag.shape[-1] + 1, device=mag.device, dtype=mag.dtype)
    return mag / (frame_mean.cumsum(dim=-1) / steps + eps)


class _SafeAtan2(torch.autograd.Function):
    """atan2 whose gradient uses 1 / (x^2 + y^2 + eps) instead of 1 / (x^2 + y^2). The angle is undefined
    at the origin and the exact gradient returns NaN there: it made the phase-output CleanSpecNet (N4 of
    IMPROVEMENT_PLAN.md) diverge to NaN in Atan2Backward."""

    @staticmethod
    def forward(ctx, y: Tensor, x: Tensor, eps: float) -> Tensor:
        """atan2(y, x); saves the inputs for the bounded backward pass."""
        ctx.save_for_backward(y, x)
        ctx.eps = eps
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad: Tensor):
        """Gradient of atan2 with eps added to x^2 + y^2, so it stays bounded near the origin."""
        y, x = ctx.saved_tensors
        r2 = x * x + y * y + ctx.eps
        return grad * x / r2, -grad * y / r2, None


def safe_atan2(y: Tensor, x: Tensor, eps: float = 1e-4) -> Tensor:
    """torch.atan2(y, x) with a bounded gradient (at most 1 / (2 sqrt(eps)) times the upstream gradient)."""
    return _SafeAtan2.apply(y, x, eps)


def phase_from(reim: Tensor) -> Tensor:
    """(B, 2F, T) or (B, 2, F, T) pseudo real/imaginary parts -> phase (B, F, T)."""
    re, im = reim.float().chunk(2, dim=1)
    if re.dim() == 4:
        re, im = re.squeeze(1), im.squeeze(1)
    return safe_atan2(im, re)


class AttentionBlock(nn.Module):
    """
    Single Transformer-style block with:
    - Multi-Head Self-Attention
    - Position-wise Feedforward Network
    - Residual connections + LayerNorm

    Input shape: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Multi-head self-attention with batch_first=True -> (B, T, D)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Position-wise feedforward: D -> 2D -> D
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 2, d_model)

        # Layer normalizations and residual dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function for feedforward network
        self.activation = F.relu

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the AttentionBlock.

        Args:
            x: Input tensor of shape (B, T, D).
            attn_mask: Optional attention mask of shape (T, T),
                       where True indicates positions that should be masked.

        Returns:
            Tensor of shape (B, T, D) after attention + feedforward.
        """
        # --- Multi-head Self-Attention sub-layer ---
        residual = x
        # self_attn returns (attn_output, attn_weights), we only need the output
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout1(attn_output)
        x = self.norm1(x)

        # --- Position-wise Feedforward sub-layer ---
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = residual + self.dropout2(x)
        x = self.norm2(x)

        return x


class CleanSpecNet(nn.Module):
    """
    CleanSpecNet

    A model that operates on spectrograms, combining:
    - 1D convolutions along the time axis for local context
    - GLU-activated convolutional blocks
    - A sequence backbone for long-range temporal dependencies:
        "transformer" (original self-attention blocks), "mamba", "mamba2" or "lstm"
    - Final projection back to the original spectrogram shape

    When ``causal=True`` all convolutions are left-padded and the backbone only looks at
    past frames (causal attention mask / unidirectional Mamba or LSTM). With ``causal=False``
    the original symmetric padding is used and the backbone can be bidirectional.

    Options added for the CleanSpecNet track of IMPROVEMENT_PLAN.md (defaults = original model):
      output_mode     direct | residual | mask (N0)
      frontend        conv (original 64-channel blocks) | linear (per-frame Linear + depthwise conv, N1b)
      feature_groups  input feature maps per bin: 1 = magnitude, 3 = magnitude + cos/sin of the phase (N5)
      input_norm      none | cumulative (causal running-mean normalization of the magnitude, N7b)
      predict_phase   extra phase head (atan2 of two outputs, N4)

    Expected input shape: (batch_size, feature_groups * freq_bins, time_steps); the magnitude comes first
    Output shape:         (batch_size, freq_bins, time_steps)
    """

    def __init__(
        self,
        input_channels: int = 513,
        num_conv_layers: int = 5,
        kernel_size: int = 4,
        stride: int = 1,
        conv_hidden_dim: int = 64,
        hidden_dim: int = 512,
        num_attention_layers: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
        backbone: Optional[dict] = None,
        output_mode: str = "direct",
        frontend: str = "conv",
        feature_groups: int = 1,
        input_norm: str = "none",
        predict_phase: bool = False,
    ) -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be direct | residual | mask, got '{output_mode}'")
        if frontend not in ("conv", "linear"):
            raise ValueError(f"frontend must be conv | linear, got '{frontend}'")
        if input_norm not in ("none", "cumulative"):
            raise ValueError(f"input_norm must be none | cumulative, got '{input_norm}'")
        # direct: regress the clean spectrogram; residual: input + correction (identity at init);
        # mask: input * sigmoid(m), a bounded mask in the compressed domain (~identity at init)
        self.output_mode = output_mode
        self.frontend = frontend
        self.n_freq = input_channels
        self.feature_groups = feature_groups
        self.input_norm = input_norm
        self.predicts_phase = predict_phase
        in_channels = input_channels * feature_groups
        if stride != 1:
            raise ValueError("CleanSpecNet only supports stride=1 (frame rate must be preserved)")

        backbone = dict(backbone or {})
        self.backbone_type = backbone.pop("type", "transformer")
        self.causal = causal
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        if frontend == "linear":
            # N1b: no channel bottleneck; every frame is projected to the backbone width, then a
            # depthwise temporal convolution adds local context
            self.frame_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.frame_temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim,
                                            padding=0 if causal else kernel_size // 2)
            num_conv_layers = 0
        else:
            # Initial 1D convolution to lightly process the input spectrogram
            self.input_layer = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Convolutional blocks to extract local temporal features
        conv_input_channels = in_channels
        self.conv_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            block = nn.Sequential(
                nn.Conv1d(conv_input_channels, conv_hidden_dim, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(conv_hidden_dim, conv_hidden_dim * 2, kernel_size=kernel_size, stride=stride,
                          padding=1),
                # GLU splits the channel dimension in half and applies a gated activation.
                nn.GLU(dim=1),
            )
            self.conv_layers.append(block)
            conv_input_channels = conv_hidden_dim

        if causal and frontend == "conv":
            # Causal mode: convolutions are explicitly left-padded in forward()
            self.input_layer.padding = (0,)
            for block in self.conv_layers:
                block[0].padding = (0,)
                block[2].padding = (0,)

        # Projection from convolutional feature dim -> sequence model hidden dim
        if frontend == "conv":
            self.tsfm_projection = nn.Linear(conv_hidden_dim, hidden_dim)

        # Sequence backbone (keeps the original attribute name for the transformer variant)
        n_layers = backbone.pop("n_layers", num_attention_layers)
        if self.backbone_type == "transformer":
            self.attention_layers = nn.ModuleList(
                [AttentionBlock(d_model=hidden_dim, nhead=backbone.pop("n_head", num_heads), dropout=dropout)
                 for _ in range(n_layers)]
            )
        elif self.backbone_type in ("mamba", "mamba2"):
            from cleanunet.sequence import MambaStack
            bidirectional = backbone.pop("bidirectional", False)
            if bidirectional and causal:
                raise ValueError("A bidirectional backbone cannot be used with causal=True")
            self.backbone = MambaStack(hidden_dim, n_layers, seq_type=self.backbone_type,
                                       bidirectional=bidirectional, **backbone)
            backbone = {}
        elif self.backbone_type == "lstm":
            from cleanunet.sequence import LSTMStack
            bidirectional = backbone.pop("bidirectional", False)
            if bidirectional and causal:
                raise ValueError("A bidirectional backbone cannot be used with causal=True")
            self.backbone = LSTMStack(hidden_dim, n_layers, bidirectional=bidirectional)
        else:
            raise ValueError(f"Unknown CleanSpecNet backbone type '{self.backbone_type}'")
        if backbone:
            raise ValueError(f"Unknown CleanSpecNet backbone options: {sorted(backbone)}")

        # Final projection back to the original frequency dimension
        self.output_layer = nn.Conv1d(hidden_dim, input_channels, kernel_size=3, stride=1,
                                      padding=0 if causal else 1)
        init_output_layer(self.output_layer, output_mode)
        if predict_phase:
            # N4: pseudo real / imaginary parts -> atan2
            self.phase_layer = nn.Conv1d(hidden_dim, 2 * input_channels, kernel_size=3, stride=1,
                                         padding=0 if causal else 1)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """(seq_len, seq_len) boolean mask where True marks future positions (masked)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _conv(self, conv: nn.Conv1d, x: Tensor) -> Tensor:
        if self.causal:
            x = F.pad(x, (conv.kernel_size[0] - 1, 0))
        return conv(x)

    def forward(self, x: Tensor, return_hidden: bool = False):
        """
        Args:
            x: input features (B, feature_groups * F, T), magnitude first.
            return_hidden: also return the backbone features (B, T, hidden_dim).
        Returns:
            enhanced spectrogram (B, F, T) [, hidden features (B, T, hidden_dim)]
        """
        out = self.run(x)
        return (out["spec"], out["hidden"]) if return_hidden else out["spec"]

    def run(self, x: Tensor) -> dict:
        """Full pass: {"spec": (B, F, T), "hidden": (B, T, hidden_dim), "phase": (B, F, T) or None}."""
        h = self.encode(x)
        if self.backbone_type == "transformer":
            mask = self._build_causal_mask(h.size(1), h.device) if self.causal else None
            for attn_block in self.attention_layers:
                h = attn_block(h, attn_mask=mask)
        else:
            h = self.backbone(h)
        return {"spec": self.head(h, x), "hidden": h, "phase": self.phase_head(h)}

    def encode(self, x: Tensor) -> Tensor:
        """Front-end: features (B, G*F, T) -> (B, T, hidden) (input of the backbone)."""
        T = x.shape[-1]
        if self.input_norm == "cumulative":
            mag = x[:, :self.n_freq]
            x = torch.cat([cumulative_norm(mag), x[:, self.n_freq:]], dim=1)
        if self.frontend == "linear":
            h = self._conv(self.frame_temporal, self.frame_proj(x))
            return F.gelu(h[..., :T]).transpose(1, 2)
        x = self._conv(self.input_layer, x)
        for block in self.conv_layers:
            x = block[1](self._conv(block[0], x))
            x = block[3](self._conv(block[2], x))
        return self.tsfm_projection(x[..., :T].transpose(1, 2))

    def head(self, hidden: Tensor, x: Tensor) -> Tensor:
        """Output projection: backbone features (B, T, hidden) and input features (B, G*F, T) -> (B, F, T)."""
        out = self._conv(self.output_layer, hidden.transpose(1, 2))
        return apply_output_mode(out, x[:, :self.n_freq], self.output_mode)

    def phase_head(self, hidden: Tensor):
        """Predicted phase (B, F, T), or None without a phase head."""
        if not self.predicts_phase:
            return None
        return phase_from(self._conv(self.phase_layer, hidden.transpose(1, 2)).float())


# Example usage for quick debugging / sanity check
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for backbone in ({"type": "transformer"}, {"type": "mamba", "n_layers": 5, "d_state": 64, "expand": 2}):
        model = CleanSpecNet(backbone=backbone).to(device).eval()
        x = torch.randn(2, 513, 128, device=device)
        with torch.no_grad():
            y = model(x)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[INFO] backbone={backbone['type']}: {n_params:.2f}M params, output {tuple(y.shape)}")
