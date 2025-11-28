import math
import os
import random
import json
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

    A model that operates on spectrograms in the frequency domain, combining:
    - 1D convolutions along the time axis for local context
    - GLU-activated convolutional blocks
    - Multiple self-attention layers for long-range temporal dependencies
    - Final projection back to the original spectrogram shape

    Expected input shape: (batch_size, freq_bins, time_steps)
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
    ) -> None:
        super().__init__()

        # Initial 1D convolution to lightly process the input spectrogram
        self.input_layer = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Convolutional blocks to extract local temporal features
        conv_input_channels = input_channels
        self.conv_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            block = nn.Sequential(
                nn.Conv1d(
                    conv_input_channels,
                    conv_hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    conv_hidden_dim,
                    conv_hidden_dim * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                ),
                # GLU splits the channel dimension in half internally
                # and applies a gated activation.
                nn.GLU(dim=1),
            )
            self.conv_layers.append(block)
            conv_input_channels = conv_hidden_dim

        # Projection from convolutional feature dim -> transformer hidden dim
        self.tsfm_projection = nn.Linear(conv_hidden_dim, hidden_dim)

        # Stack of self-attention blocks for modeling long-range temporal structure
        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
                for _ in range(num_attention_layers)
            ]
        )

        # Final projection back to the original frequency dimension
        self.output_layer = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Build a causal mask so each time step can only attend to itself and past positions.
        Shape: (seq_len, seq_len), where True indicates masked positions.

        Upper-triangular (excluding diagonal) is masked: future positions.
        """
        # 1s in the upper triangle above the main diagonal -> future positions
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for CleanSpecNet.

        Args:
            x: Input spectrogram of shape (B, F, T), where
               B = batch size, F = frequency bins, T = time steps.

        Returns:
            Tensor of shape (B, F, T) representing the enhanced/cleaned spectrogram.
        """
        # Initial conv: process along time axis while staying in (B, F, T)
        # Current shape: (batch_size, freq_bins, time_steps)
        x = self.input_layer(x)

        # Convolutional feature extractor:
        # shape stays (B, C, T) where C = conv_hidden_dim
        for conv in self.conv_layers:
            x = conv(x)

        # Prepare for attention: transform to (B, T, C)
        x = x.transpose(1, 2)  # (batch_size, time_steps, channels)

        # Project to transformer hidden dimension
        x = self.tsfm_projection(x)

        # Build causal mask so attention is autoregressive in time
        seq_len = x.size(1)
        causal_mask = self._build_causal_mask(seq_len, x.device)

        # Apply stacked self-attention blocks
        for attn_block in self.attention_layers:
            x = attn_block(x, attn_mask=causal_mask)

        # Back to (B, C, T) for final convolution
        x = x.transpose(1, 2)

        # Final projection to original spectrogram dimension (freq_bins)
        x = self.output_layer(x)  # (batch_size, freq_bins, time_steps)

        return x


# Example usage for quick debugging / sanity check
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration from JSON file
    config_path = "configs/config_cleanspecnet.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. Please check the path."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    network_config = config.get("network_config", {})
    print("[INFO] Loaded network configuration:")
    print(network_config)

    # Instantiate model
    model = CleanSpecNet(**network_config).to(device)
    model.eval()

    # Simulate a dummy input spectrogram
    # Shape: (batch_size, freq_bins, time_steps)
    batch_size = 2
    freq_bins = network_config.get("input_channels", 513)
    time_steps = 1024

    dummy_input = torch.randn(batch_size, freq_bins, time_steps, device=device)

    print("[INFO] Running a forward pass with dummy input...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"[INFO] Input shape : {dummy_input.shape}")
    print(f"[INFO] Output shape: {output.shape}")

