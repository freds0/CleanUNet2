# cleanunet_model.py
# Refactored CleanUNet + Transformer utilities with English comments and clearer structure.
# Original code adapted from NVIDIA/other sources; MIT / original licenses apply where appropriate.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cleanunet.util import weight_scaling_init
# If weight_scaling_init is not available, provide a simple fallback:
# def weight_scaling_init(layer): pass

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


class CleanUNet(nn.Module):
    """CleanUNet: encoder-decoder convolutional architecture with a Transformer bottleneck.

    The U-Net progressively downsamples using strided convs and then upsamples with ConvTranspose1d.
    A Transformer encoder sits at the bottleneck to capture long-range dependencies.
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
        tsfm_d_inner: int = 2048
    ):
        super().__init__()

        # save config
        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # Build encoder and decoder as ModuleLists for flexible depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        in_ch = channels_input
        out_ch = channels_output
        H = channels_H

        for i in range(encoder_n_layers):
            # Downsampling block: Conv1d -> ReLU -> 1x1 Conv -> GLU
            down_block = nn.Sequential(
                nn.Conv1d(in_ch, H, kernel_size, stride),
                nn.ReLU(inplace=False),
                nn.Conv1d(H, H * 2, 1),
                nn.GLU(dim=1)
            )
            self.encoder.append(down_block)

            # Build corresponding upsampling block (Conv1d 1x1 -> GLU -> ConvTranspose1d -> maybe ReLU)
            if i == 0:
                up_block = nn.Sequential(
                    nn.Conv1d(H, H * 2, 1),
                    nn.GLU(dim=1),
                    nn.ConvTranspose1d(H, out_ch, kernel_size, stride)
                )
            else:
                up_block = nn.Sequential(
                    nn.Conv1d(H, H * 2, 1),
                    nn.GLU(dim=1),
                    nn.ConvTranspose1d(H, out_ch, kernel_size, stride),
                    nn.ReLU(inplace=False)
                )
            # Prepend so decoder has reverse order
            self.decoder.insert(0, up_block)

            # update channel variables for next layer
            in_ch = H
            out_ch = H
            H = min(H * 2, max_H)

        # Transformer bottleneck projection: conv -> transformer encoder -> conv
        self.tsfm_conv1 = nn.Conv1d(out_ch, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(
            d_word_vec=tsfm_d_model,
            n_layers=tsfm_n_layers,
            n_head=tsfm_n_head,
            d_k=tsfm_d_model // tsfm_n_head,
            d_v=tsfm_d_model // tsfm_n_head,
            d_model=tsfm_d_model,
            d_inner=tsfm_d_inner,
            dropout=0.0,
            n_position=0,
            scale_emb=False
        )
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, out_ch, kernel_size=1)

        # Apply weight scaling initialization for conv layers if provided
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                try:
                    weight_scaling_init(layer)
                except Exception:
                    # If user doesn't have weight_scaling_init or it fails, skip gracefully
                    pass

        # --- Internal state for encode/decode split ---
        self._stored_skips = None
        self._stored_std = None
        self._original_len = None

    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        latent = self.encode(noisy_audio)
        return self.decode(latent)

    def encode(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Encodes the noisy audio into latent representation (inside bottleneck).
        Stores skip connections and stats internally for subsequent decode call.
        
        Returns:
            latent: (B, tsfm_d_model, T)
        """
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        
        # Store original length and std for reconstruction
        self._original_len = noisy_audio.shape[-1]
        self._stored_std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
        
        # Normalize and Pad
        x = noisy_audio / self._stored_std
        x = padding(x, self.encoder_n_layers, self.kernel_size, self.stride)

        # Encoder Pass
        skips = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skips.append(x)
        
        # Store skips (reverse order for decoder)
        self._stored_skips = skips[::-1]

        # Bottleneck: Part 1 (Conv -> Transformer)
        # Prepare mask
        len_s = x.shape[-1]
        attn_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)).bool()

        x = self.tsfm_conv1(x)           # (B, H, T) -> (B, d_model, T)
        x = x.permute(0, 2, 1)           # -> (B, T, d_model)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)           # -> (B, d_model, T)
        
        # Return at this point (latent space)
        return x

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to waveform.
        Uses stored skip connections and stats from the last encode call.
        
        Args:
            latent: (B, tsfm_d_model, T)
        Returns:
            denoised_audio: (B, 1, L)
        """
        if self._stored_skips is None:
            raise RuntimeError("encode() must be called before decode()")

        x = latent
        
        # Bottleneck: Part 2 (Transformer Output -> Conv)
        x = self.tsfm_conv2(x)

        # Decoder Pass
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = self._stored_skips[i]
            # Crop skip to match current length (due to conv artifacts/padding)
            if skip_i.shape[-1] > x.shape[-1]:
                skip_i = skip_i[..., :x.shape[-1]]
            
            x = x + skip_i
            x = upsampling_block(x)

        # Crop to original length and denormalize
        if self._original_len is not None:
            x = x[:, :, :self._original_len]
        
        if self._stored_std is not None:
            x = x * self._stored_std
            
        return x

# ---------------------------
# Quick self-test / example usage when running as script
# ---------------------------
if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check for CleanUNet model")
    parser.add_argument('-c', '--config', type=str, default='configs/DNS-large-full.json', help='JSON config file (network_config key expected)')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config.get("network_config", {})

    # Build model and run forward/backward pass to validate shapes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CleanUNet(**network_config).to(device)

    # Helper: print model size (optional); implement your own print_size if needed
    try:
        from util import print_size
        print_size(model, keyword="tsfm")
    except Exception:
        pass

    # Example input: 4 examples, 1 channel, 4.5s at 16kHz
    input_data = torch.ones([4, 1, int(4.5 * 16000)], device=device)
    # If an optional speaker-vector forward was used, xvector_data would be provided
    # xvector_data = torch.ones([4, 512], device=device)

    output = model(input_data)
    print(f"Output shape: {output.shape}")

    # quick backward pass sanity check
    y = torch.rand_like(output)
    loss = torch.nn.MSELoss()(y, output)
    loss.backward()
    print("Loss (scalar):", loss.item())
