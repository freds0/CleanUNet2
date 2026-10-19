# cleanunet.py
# Refactored CleanUNet + Transformer utilities with English comments and clearer structure.
# Original code adapted from NVIDIA/other sources; MIT / original licenses apply where appropriate.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cleanunet.util import weight_scaling_init

# ---------------------------
# Attention & Transformer utilities
# ---------------------------

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional masking and dropout.

    Args:
        temperature (float): scaling factor (usually sqrt(d_k)).
        attn_dropout (float): dropout probability applied to attention weights.
    """
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # q: (B, n_head, L_q, d_k)
        # k: (B, n_head, L_k, d_k)
        # v: (B, n_head, L_v, d_v)
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))  # (..., L_q, L_k)

        if mask is not None:
            # mask shape expected to broadcast to attn shape; masked positions set to very low value
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # (..., L_q, d_v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention wrapper that projects q, k, v to heads, applies
    ScaledDotProductAttention and recombines heads.

    Args:
        n_head (int): number of attention heads
        d_model (int): input/output model dimension
        d_k (int): per-head key dim
        d_v (int): per-head value dim
        dropout (float): dropout probability
    """
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # linear projections (no bias to match original implementation)
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            q, k, v: tensors of shape (batch, seq_len, d_model)
            mask: optional mask of shape (batch, seq_len, seq_len) or broadcastable
        Returns:
            output: (batch, seq_len, d_model)
            attn: attention weights (batch, n_head, seq_len, seq_len)
        """
        batch_size, len_q = q.size(0), q.size(1)

        residual = q

        # Project and split into heads: shape -> (batch, len, n_head, d_k)
        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch_size, k.size(1), self.n_head, self.d_k)
        v = self.w_vs(v).view(batch_size, v.size(1), self.n_head, self.d_v)

        # transpose to (batch, n_head, len, d_k/d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            # expand mask for head dimension: expected shape (batch, 1, len, len)
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # combine heads -> (batch, len, n_head * d_v)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q = q + residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with residual connection."""
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.activation = nn.ReLU(inplace=False)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.w_2(self.activation(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x


def get_subsequent_mask(seq: torch.Tensor):
    """Create a subsequent mask to prevent attention to future positions.

    Args:
        seq: input sequence tensor, shape (batch, seq_len) or (batch, seq_len, ...)
    Returns:
        mask of shape (batch, seq_len, seq_len) with True for allowed positions.
    """
    _, len_s = seq.size()[:2]
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learned)."""
    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position: int, d_hid: int):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, d_model)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    """Single encoder layer composed of multi-head attention + feed-forward."""
    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.0):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input: torch.Tensor, slf_attn_mask: torch.Tensor = None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    """Transformer encoder composed of stacked EncoderLayer blocks."""
    def __init__(
        self,
        d_word_vec: int = 512,
        n_layers: int = 2,
        n_head: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        d_model: int = 512,
        d_inner: int = 2048,
        dropout: float = 0.1,
        n_position: int = 624,
        scale_emb: bool = False
    ):
        super().__init__()
        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq: torch.Tensor, src_mask: torch.Tensor = None, return_attns: bool = False):
        """
        Args:
            src_seq: (batch, seq_len, d_model)
            src_mask: (batch, seq_len, seq_len) or None
        """
        enc_slf_attn_list = []

        enc_output = src_seq
        if self.scale_emb:
            enc_output = enc_output * (self.d_model ** 0.5)
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        return (enc_output, enc_slf_attn_list) if return_attns else enc_output


# ---------------------------
# CleanUNet architecture
# ---------------------------

def padding(x: torch.Tensor, D: int, K: int, S: int) -> torch.Tensor:
    """Pad input x with zeros so that after D strided convs the output length matches target.
    This mirrors the original padding logic used in the codebase.
    """
    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)
    for _ in range(D):
        L = (L - 1) * S + K
    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x


class Activation(nn.Module):
    """Gated activation A * act(B) with optional bypass channels (from CleanUMamba).

    With activation="Sigmoid" and bypass_channels=0 this is exactly nn.GLU(dim=1).
    """

    def __init__(self, activation: str = "Sigmoid", bypass_channels: int = 0):
        super().__init__()
        acts = {"Sigmoid": nn.Sigmoid, "ReLU": nn.ReLU, "SiLU": nn.SiLU, "GELU": nn.GELU}
        if activation not in acts:
            raise ValueError(f"glu_activation={activation} not supported; use one of {list(acts)}")
        self.activation = acts[activation]()
        self.bypass_channels = bypass_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nX = self.bypass_channels
        if nX == 0:
            a, b = x.chunk(2, dim=1)
            return a * self.activation(b)
        nAB = (x.shape[1] - nX) // 2
        bypass, a, b = torch.split(x, [nX, nAB, nAB], dim=1)
        return torch.cat([bypass, a * self.activation(b)], dim=1)


def _gate(glu_activation: str, bypass_channels: int) -> nn.Module:
    if glu_activation == "Sigmoid" and bypass_channels == 0:
        return nn.GLU(dim=1)
    return Activation(glu_activation, bypass_channels)


class CleanUNet(nn.Module):
    """CleanUNet: encoder-decoder convolutional architecture with a sequence model at the bottleneck.

    Bottleneck types (``bottleneck["type"]``):
      - "transformer": original CleanUNet self-attention encoder (``tsfm_encoder``), default.
      - "mamba" / "mamba2": stack of pre-norm Mamba blocks (``tsfm_Mamba_layers`` + ``norm_f``),
        laid out exactly like CleanUMamba so its checkpoints load without remapping.
      - "lstm": unidirectional LSTM (``tsfm_lstm``).
    """

    def __init__(
        self,
        channels_input: int = 1,
        channels_output: int = 1,
        channels_H: int = 64,
        max_H: int = 768,
        encoder_n_layers: int = 8,
        kernel_size: int = 4,
        stride: int = 2,
        tsfm_n_layers: int = 5,
        tsfm_n_head: int = 8,
        tsfm_d_model: int = 512,
        tsfm_d_inner: int = 2048,
        bottleneck: dict = None,
        normalize_input: bool = True,
        encoder_norm: str = None,
        glu_activation: str = "Sigmoid",
        bypass_channels=0,
        encoder_groups=1,
        residual_projection: bool = False,
    ):
        super().__init__()

        bottleneck = dict(bottleneck or {})
        self.bottleneck_type = bottleneck.pop("type", "transformer")
        if self.bottleneck_type not in ("transformer", "mamba", "mamba2", "lstm"):
            raise ValueError(f"Unknown bottleneck type '{self.bottleneck_type}'")
        if encoder_norm not in (None, "none", "batch"):
            raise ValueError(f"encoder_norm must be null/'none'/'batch', got '{encoder_norm}'")
        use_bn = encoder_norm == "batch"

        # save config
        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalize_input = normalize_input
        self.residual_projection = residual_projection

        self.tsfm_n_layers = bottleneck.pop("n_layers", tsfm_n_layers)
        self.tsfm_d_model = bottleneck.pop("d_model", tsfm_d_model)
        self.tsfm_n_head = bottleneck.pop("n_head", tsfm_n_head)
        self.tsfm_d_inner = bottleneck.pop("d_inner", tsfm_d_inner)

        # Build encoder and decoder as ModuleLists for flexible depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        if residual_projection:
            self.residual_projection_layers = nn.ModuleList()
        in_ch = channels_input
        out_ch = channels_output
        H = channels_H
        self.level_channels = []        # encoder output channels per level (= decoder input channels)

        for i in range(encoder_n_layers):
            groups = encoder_groups[i] if isinstance(encoder_groups, (list, tuple)) else encoder_groups
            bp = bypass_channels[i] if isinstance(bypass_channels, (list, tuple)) else bypass_channels
            gate_channels = bp + (H - bp) * 2

            # Downsampling block: Conv1d -> [BN] -> ReLU -> 1x1 Conv -> GLU
            down = [nn.Conv1d(in_ch, H, kernel_size, stride, groups=groups if i > 0 else 1)]
            if use_bn:
                down.append(nn.BatchNorm1d(H))
            down += [nn.ReLU(inplace=False), nn.Conv1d(H, gate_channels, 1), _gate(glu_activation, bp)]
            self.encoder.append(nn.Sequential(*down))

            if residual_projection:
                self.residual_projection_layers.append(nn.Conv1d(H, H, 1))

            # Upsampling block: 1x1 Conv -> GLU -> ConvTranspose1d -> [BN -> ReLU] (no ReLU on the output layer)
            up = [nn.Conv1d(H, gate_channels, 1), _gate(glu_activation, bp),
                  nn.ConvTranspose1d(H, out_ch, kernel_size, stride)]
            if i > 0:
                if use_bn:
                    up.append(nn.BatchNorm1d(out_ch))
                up.append(nn.ReLU(inplace=False))
            # Prepend so decoder has reverse order
            self.decoder.insert(0, nn.Sequential(*up))

            self.level_channels.append(H)
            in_ch = H
            out_ch = H
            H = min(H * 2, max_H)

        # Bottleneck: 1x1 conv -> sequence model -> 1x1 conv
        d_model = self.tsfm_d_model
        self.tsfm_conv1 = nn.Conv1d(out_ch, d_model, kernel_size=1)
        if self.bottleneck_type == "transformer":
            self.tsfm_causal = bottleneck.pop("causal", True)
            self.tsfm_encoder = TransformerEncoder(
                d_word_vec=d_model,
                n_layers=self.tsfm_n_layers,
                n_head=self.tsfm_n_head,
                d_k=d_model // self.tsfm_n_head,
                d_v=d_model // self.tsfm_n_head,
                d_model=d_model,
                d_inner=self.tsfm_d_inner,
                dropout=0.0,
                n_position=0,
                scale_emb=False
            )
        elif self.bottleneck_type in ("mamba", "mamba2"):
            from cleanunet.sequence import build_mamba_layers, resolve_backend
            self.mamba_backend = resolve_backend(bottleneck.pop("backend", "auto"))
            self.tsfm_Mamba_layers, self.norm_f = build_mamba_layers(
                d_model, self.tsfm_n_layers, seq_type=self.bottleneck_type, backend=self.mamba_backend,
                d_state=bottleneck.pop("d_state", d_model // self.tsfm_n_head),
                expand=bottleneck.pop("expand", self.tsfm_d_inner / d_model),
                **bottleneck)
            bottleneck = {}
        else:  # lstm
            self.tsfm_lstm = nn.LSTM(d_model, d_model, num_layers=self.tsfm_n_layers, batch_first=True)
        if bottleneck:
            unknown = set(bottleneck) - {"causal", "backend", "d_state", "d_conv", "expand", "headdim",
                                         "use_fast_path", "residual_in_fp32", "norm_epsilon"}
            if unknown:
                raise ValueError(f"Unknown bottleneck options: {sorted(unknown)}")
        self.tsfm_conv2 = nn.Conv1d(d_model, out_ch, kernel_size=1)

        # Weight scaling initialization for all conv layers (as in CleanUNet / CleanUMamba)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)) and layer.bias is not None:
                weight_scaling_init(layer)

    # ------------------------------------------------------------------
    @property
    def total_stride(self) -> int:
        return self.stride ** self.encoder_n_layers

    def fusion_points(self) -> dict:
        """Named feature points that accept conditioning hooks: {name: (channels, stride in samples)}."""
        points = {}
        for level, ch in enumerate(self.level_channels, start=1):
            s = self.stride ** level
            points[f"enc_s{s}"] = (ch, s)
            points[f"skip_s{s}"] = (ch, s)
            points[f"dec_s{s}"] = (ch, s)
        points["bott_pre"] = (self.tsfm_d_model, self.total_stride)
        points["bott_post"] = (self.tsfm_d_model, self.total_stride)
        return points

    def valid_length(self, length: int) -> int:
        """Smallest length >= `length` that the encoder/decoder maps back without remainder."""
        D, K, S = self.encoder_n_layers, self.kernel_size, self.stride
        for _ in range(D):
            length = 1 if length < K else 1 + np.ceil((length - K) / S)
        for _ in range(D):
            length = (length - 1) * S + K
        return int(length)

    def run_bottleneck(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """Sequence model on (B, T, d_model)."""
        if self.bottleneck_type == "transformer":
            mask = None
            if self.tsfm_causal:
                len_s = x.shape[1]
                mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)).bool()
            return self.tsfm_encoder(x, src_mask=mask)
        if self.bottleneck_type in ("mamba", "mamba2"):
            from cleanunet.sequence import run_mamba_layers
            return run_mamba_layers(self.tsfm_Mamba_layers, self.norm_f, x, inference_params)
        return self.tsfm_lstm(x)[0]

    def forward(self, noisy_audio: torch.Tensor, bottleneck_hook=None, return_skip_connections: bool = False,
                hooks: dict = None, bottleneck_runner=None):
        """Forward pass.

        Args:
            noisy_audio: (B, L) or (B, 1, L).
            bottleneck_hook: optional callable applied to the bottleneck features (B, d_model, T)
                after ``tsfm_conv1`` (used for spectrogram fusion in CleanUNet2). Same as hooks["bott_pre"].
            hooks: optional {point_name: callable(features) -> features} for the points listed by
                ``fusion_points()``; features are (B, C, T).
            bottleneck_runner: optional callable replacing the sequence model, (B, T, d) -> (B, T, d).
        Returns:
            denoised audio (B, 1, L)
        """
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == self.channels_input, f"Expected {self.channels_input} input channel(s), got {C}."

        # normalize by standard deviation per example (not in-place) and pad to a valid length
        if self.normalize_input:
            std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
            noisy_audio = noisy_audio / std
        x = padding(noisy_audio, self.encoder_n_layers, self.kernel_size, self.stride)

        # encoder: collect skip connections
        hooks = dict(hooks or {})
        if bottleneck_hook is not None:
            if "bott_pre" in hooks:
                raise ValueError("Use either bottleneck_hook or hooks['bott_pre'], not both")
            hooks["bott_pre"] = bottleneck_hook

        def hook(name, h):
            fn = hooks.get(name)
            return h if fn is None else fn(h)

        skip_connections = []
        for level, downsampling_block in enumerate(self.encoder, start=1):
            x = downsampling_block(x)
            x = hook(f"enc_s{self.stride ** level}", x)
            skip_connections.append(x)
        if self.residual_projection:
            skip_connections = [proj(s) for proj, s in zip(self.residual_projection_layers, skip_connections)]
        skip_connections = [hook(f"skip_s{self.stride ** level}", s)
                            for level, s in enumerate(skip_connections, start=1)]
        skip_connections = skip_connections[::-1]

        # bottleneck
        x = self.tsfm_conv1(x)                      # (B, d_model, T)
        x = hook("bott_pre", x)
        runner = bottleneck_runner or self.run_bottleneck
        tsfm_out = runner(x.permute(0, 2, 1)).permute(0, 2, 1)
        tsfm_out = hook("bott_post", tsfm_out)
        x = self.tsfm_conv2(tsfm_out)

        # decoder with skip connections
        D = self.encoder_n_layers
        for i, upsampling_block in enumerate(self.decoder):
            x = x + skip_connections[i][:, :, :x.shape[-1]]
            x = hook(f"dec_s{self.stride ** (D - i)}", x)
            x = upsampling_block(x)

        x = x[:, :, :L]
        if self.normalize_input:
            x = x * std
        if return_skip_connections:
            return x, skip_connections + [tsfm_out]
        return x


# ---------------------------
# Quick self-test / example usage when running as script
# ---------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for cfg in ({}, {"type": "mamba", "n_layers": 3}):
        model = CleanUNet(bottleneck=cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        x = torch.randn(2, 1, 32768, device=device) * 0.1
        y = model(x)
        y.pow(2).mean().backward()
        print(f"bottleneck={cfg.get('type', 'transformer')}: {n_params:.2f}M params, output {tuple(y.shape)}")
