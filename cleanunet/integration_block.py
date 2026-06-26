"""
Integration Blocks for fusing embeddings with latent features.
Supports both pooled embeddings and sequence embeddings (SSL hidden states).
"""

import torch
import torch.nn as nn


class IntegrationBlock(nn.Module):
    """
    Integration block for fusing X-Vector embeddings with latent features (1D).
    Used for waveform-domain processing.
    
    Similar to the paper's approach: concatenate embeddings with latent features,
    then apply 1x1 convolution for fusion.
    """
    
    def __init__(self, latent_channels, xvector_dim=512):
        """
        Initialize the integration block.
        
        Args:
            latent_channels (int): Number of channels in latent features
            xvector_dim (int): Dimension of X-Vector embeddings (default: 512)
        """
        super().__init__()
        
        self.latent_channels = latent_channels
        self.xvector_dim = xvector_dim
        
        # 1x1 convolution for feature fusion
        # Input: latent_channels + xvector_dim
        # Output: latent_channels
        self.fusion_conv = nn.Conv1d(
            in_channels=latent_channels + xvector_dim,
            out_channels=latent_channels,
            kernel_size=1,
            bias=False
        )
        
        # Layer normalization for stabilization
        self.norm = nn.LayerNorm(latent_channels)
        
        # PReLU activation
        self.activation = nn.PReLU()
        
    def forward(self, latent_features, xvector_embeddings):
        """
        Forward pass: fuse latent features with X-Vector embeddings.
        
        Args:
            latent_features (torch.Tensor): Latent features from encoder 
                                            Shape: (batch, latent_channels, time)
            xvector_embeddings (torch.Tensor): X-Vector embeddings 
                                               Shape: (batch, xvector_dim, time)
            
        Returns:
            fused_features (torch.Tensor): Fused features 
                                          Shape: (batch, latent_channels, time)
        """
        # Concatenate along channel dimension
        # Shape: (batch, latent_channels + xvector_dim, time)
        concatenated = torch.cat([latent_features, xvector_embeddings], dim=1)
        
        # Apply fusion convolution
        # Shape: (batch, latent_channels, time)
        fused = self.fusion_conv(concatenated)
        
        # Apply layer normalization
        # Need to transpose for LayerNorm: (batch, time, channels)
        fused = fused.transpose(1, 2)
        fused = self.norm(fused)
        # Transpose back: (batch, channels, time)
        fused = fused.transpose(1, 2)
        
        # Apply activation
        fused = self.activation(fused)
        
        return fused


class SpecIntegrationBlock(nn.Module):
    """
    Integration block for fusing X-Vector embeddings with spectrogram features (2D).
    Used for frequency-domain processing.
    """
    
    def __init__(self, latent_channels, xvector_dim=512):
        """
        Initialize the spectrogram integration block.
        
        Args:
            latent_channels (int): Number of channels in latent features
            xvector_dim (int): Dimension of X-Vector embeddings (default: 512)
        """
        super().__init__()
        
        # Project X-Vector to match latent channels
        self.xvector_projection = nn.Sequential(
            nn.Linear(xvector_dim, latent_channels),
            nn.LayerNorm(latent_channels),
            nn.PReLU()
        )
        
        # 1x1 convolution for fusion
        self.fusion_conv = nn.Conv2d(
            in_channels=latent_channels * 2,  # Concatenated features
            out_channels=latent_channels,
            kernel_size=1,
            bias=False
        )
        
        # Batch normalization for 2D features
        self.norm = nn.BatchNorm2d(latent_channels)
        
        # PReLU activation
        self.activation = nn.PReLU()
        
    def forward(self, latent_features, xvector_embeddings):
        """
        Forward pass: fuse spectrogram features with X-Vector embeddings.
        
        Args:
            latent_features (torch.Tensor): Latent features from encoder 
                                            Shape: (batch, channels, freq, time)
            xvector_embeddings (torch.Tensor): X-Vector embeddings 
                                               Shape: (batch, xvector_dim)
            
        Returns:
            fused_features (torch.Tensor): Fused features 
                                          Shape: (batch, channels, freq, time)
        """
        batch, channels, freq, time = latent_features.shape
        
        # Project X-Vector embeddings
        # Shape: (batch, channels)
        xvec_proj = self.xvector_projection(xvector_embeddings)
        
        # Expand to match spatial dimensions (freq, time)
        # Shape: (batch, channels, freq, time)
        xvec_expanded = xvec_proj.view(batch, channels, 1, 1).expand(-1, -1, freq, time)
        
        # Concatenate along channel dimension
        # Shape: (batch, channels * 2, freq, time)
        concatenated = torch.cat([latent_features, xvec_expanded], dim=1)
        
        # Apply fusion convolution
        fused = self.fusion_conv(concatenated)
        
        # Apply normalization and activation
        fused = self.norm(fused)
        fused = self.activation(fused)

        return fused


class SequenceIntegrationBlock(nn.Module):
    """
    Integration block for fusing sequence SSL embeddings with latent features.
    Uses self-attention to process sequence embeddings before fusion.

    This handles embeddings with temporal dimension: (batch, seq_len, embedding_dim)
    """

    def __init__(self, latent_channels, embedding_dim=1024, num_heads=8, dropout=0.1):
        """
        Initialize the sequence integration block with self-attention.

        Args:
            latent_channels (int): Number of channels in latent features
            embedding_dim (int): Dimension of sequence embeddings (e.g., 1024 for wavlm-large, 1920 for wav2vec2-xls-r-2b)
            num_heads (int): Number of attention heads (default: 8)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Multi-head self-attention for processing sequence embeddings
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization after attention
        self.attn_norm = nn.LayerNorm(embedding_dim)

        # Learnable query for pooling attended features
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Cross-attention for pooling
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization after cross-attention
        self.cross_norm = nn.LayerNorm(embedding_dim)

        # Project to latent space for fusion
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, latent_channels),
            nn.LayerNorm(latent_channels),
            nn.PReLU()
        )

        # 1x1 convolution for feature fusion
        self.fusion_conv = nn.Conv1d(
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            kernel_size=1,
            bias=False
        )

        # Layer normalization for stabilization
        self.norm = nn.LayerNorm(latent_channels)

        # PReLU activation
        self.activation = nn.PReLU()

    def forward(self, latent_features, sequence_embeddings):
        """
        Forward pass: process sequence embeddings with self-attention, then fuse with latent features.

        Args:
            latent_features (torch.Tensor): Latent features from encoder
                                            Shape: (batch, latent_channels, time)
            sequence_embeddings (torch.Tensor): Raw sequence embeddings
                                                Shape: (batch, seq_len, embedding_dim)

        Returns:
            fused_features (torch.Tensor): Fused features
                                          Shape: (batch, latent_channels, time)
        """
        batch_size, latent_channels, time_steps = latent_features.shape

        # Step 1: Self-attention on sequence embeddings
        # Capture long-range contextual dependencies
        attn_output, _ = self.self_attention(
            query=sequence_embeddings,
            key=sequence_embeddings,
            value=sequence_embeddings
        )
        attn_output = self.attn_norm(attn_output + sequence_embeddings)  # Residual connection

        # Step 2: Cross-attention pooling
        # Use learnable query to pool attended embeddings
        query = self.query.expand(batch_size, -1, -1)
        pooled_output, _ = self.cross_attention(
            query=query,
            key=attn_output,
            value=attn_output
        )
        pooled_output = pooled_output.squeeze(1)  # (batch, embedding_dim)
        pooled_output = self.cross_norm(pooled_output)

        # Step 3: Project to latent space
        projected = self.projection(pooled_output)  # (batch, latent_channels)

        # Step 4: Expand to match temporal dimension
        projected = projected.unsqueeze(2).expand(-1, -1, time_steps)  # (batch, latent_channels, time)

        # Step 5: Concatenate and fuse
        concatenated = torch.cat([latent_features, projected], dim=1)  # (batch, 2*latent_channels, time)
        fused = self.fusion_conv(concatenated)  # (batch, latent_channels, time)

        # Step 6: Apply layer normalization
        fused = fused.transpose(1, 2)  # (batch, time, latent_channels)
        fused = self.norm(fused)
        fused = fused.transpose(1, 2)  # (batch, latent_channels, time)

        # Step 7: Apply activation
        fused = self.activation(fused)

        return fused


# ============================================================================
# Hierarchical Multi-Scale fusion (Option 3) — model-agnostic across SSL backbones
# ============================================================================
def _resolve_layer_range(rng, n):
    """
    Clamp an inclusive (lo, hi) layer range into a stack of ``n`` layers.

    The hierarchical split is expressed as indices INTO THE SELECTED SSL stack,
    not absolute backbone layers, so the same block works for any SSL family
    regardless of how many layers it exposes (wavlm 25, wav2vec2-xls-r-2b 49,
    whisper 33, ...) and for both '+' (few layers) and '++' (all layers).
    """
    lo, hi = rng
    lo = max(0, min(int(lo), n - 1))
    hi = max(lo, min(int(hi), n - 1))
    return lo, hi


class MultiScaleDilatedConv(nn.Module):
    """
    Four parallel dilated 1-D convolutions (dilations [1, 2, 4, 8], kernel 3) whose
    outputs are concatenated along channels and projected back to ``out_channels``.
    Temporal length is preserved (padding = dilation).
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 4, 8), kernel_size=3):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=d, dilation=d)
            for d in dilations
        ])
        # 1x1 conv merges the concatenated multi-scale features back to out_channels.
        self.project = nn.Conv1d(out_channels * len(dilations), out_channels, kernel_size=1)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, in_channels, T]
        Returns:
            Tensor: [B, out_channels, T]
        """
        outs = [branch(x) for branch in self.branches]       # each [B, out_channels, T]
        cat = torch.cat(outs, dim=1)                         # [B, out_channels * 4, T]
        return self.activation(self.project(cat))            # [B, out_channels, T]


class HierarchicalMultiScaleBlock(nn.Module):
    """
    Layer-wise hierarchical fusion that works with ANY SSL backbone.

    It receives the stacked SELECTED SSL layers ``[B, N, T, D]`` and splits them
    into an Acoustic group (lower layers, fine acoustic detail) and a Semantic
    group (upper layers, phonetic/content), each processed by a
    ``MultiScaleDilatedConv``. It then produces FiLM parameters for hierarchical
    injection inside ``CleanUNet.encode``:

        - Acoustic group -> GLOBAL FiLM for the early CleanUNet encoder layers (1 & 2).
        - Semantic group -> PER-FRAME FiLM for the Transformer bottleneck.

    The split is RELATIVE to ``N`` (the number of selected layers), so it adapts
    automatically to each SSL family and to the '+'/'++' layer strategy. The
    acoustic/semantic ranges may be overridden via config (indices into the
    selected stack); when ``None`` the lower/upper halves are used.

    This block only GENERATES the FiLM parameters; the actual application happens
    inside ``CleanUNet.encode`` (encoder_film / bottleneck_film arguments).
    """

    def __init__(self, embedding_dim, bottleneck_channels, early_channels=(64, 128),
                 acoustic_layers=None, semantic_layers=None, hidden=256):
        """
        Args:
            embedding_dim (int): SSL hidden size D.
            bottleneck_channels (int): CleanUNet bottleneck channels C_unet.
            early_channels (tuple[int]): channel counts of encoder layers 1 & 2.
            acoustic_layers (tuple[int]|None): inclusive index range into the
                selected SSL stack for the low group. None -> lower half.
            semantic_layers (tuple[int]|None): inclusive index range for the high
                group. None -> upper half.
            hidden (int): hidden width of the multi-scale processors.
        """
        super().__init__()
        self.early_channels = tuple(early_channels)
        self.acoustic_layers = tuple(acoustic_layers) if acoustic_layers is not None else None
        self.semantic_layers = tuple(semantic_layers) if semantic_layers is not None else None

        self.acoustic_proc = MultiScaleDilatedConv(embedding_dim, hidden)   # -> [B, hidden, T]
        self.semantic_proc = MultiScaleDilatedConv(embedding_dim, hidden)   # -> [B, hidden, T]

        # GLOBAL FiLM heads (one per early encoder layer): hidden -> 2 * channels.
        self.early_film = nn.ModuleDict({
            str(ch): nn.Linear(hidden, ch * 2) for ch in self.early_channels
        })

        # PER-FRAME FiLM heads for the bottleneck (semantic group).
        self.bottleneck_gamma = nn.Conv1d(hidden, bottleneck_channels, kernel_size=1)
        self.bottleneck_beta = nn.Conv1d(hidden, bottleneck_channels, kernel_size=1)

    def _group_ranges(self, n):
        """Resolve (acoustic, semantic) inclusive ranges for a stack of n layers."""
        if self.acoustic_layers is not None:
            a_lo, a_hi = _resolve_layer_range(self.acoustic_layers, n)
        else:
            a_lo, a_hi = 0, max(0, (n // 2) - 1)            # lower half
        if self.semantic_layers is not None:
            s_lo, s_hi = _resolve_layer_range(self.semantic_layers, n)
        else:
            s_lo, s_hi = n // 2, n - 1                       # upper half
        return (a_lo, a_hi), (s_lo, s_hi)

    def forward(self, stacked_states):
        """
        Args:
            stacked_states (Tensor): [B, N, T, D] stacked SELECTED SSL layers
                (N = number of selected layers; from extractor.extract_selected_layers).

        Returns:
            early_params (dict[int, tuple[Tensor, Tensor]]): {channels: (gamma [B, C],
                beta [B, C])} GLOBAL FiLM params for early encoder layers.
            bottleneck_params (tuple[Tensor, Tensor]): (gamma, beta), each
                [B, C_unet, T] PER-FRAME FiLM params for the bottleneck.
        """
        n = stacked_states.size(1)
        (a_lo, a_hi), (s_lo, s_hi) = self._group_ranges(n)

        # Group-average over the selected layers -> [B, T, D]
        acoustic = stacked_states[:, a_lo:a_hi + 1].mean(dim=1)
        semantic = stacked_states[:, s_lo:s_hi + 1].mean(dim=1)

        # To conv layout [B, D, T]
        acoustic = acoustic.transpose(1, 2)                  # [B, D, T]
        semantic = semantic.transpose(1, 2)                  # [B, D, T]

        a_feat = self.acoustic_proc(acoustic)                # [B, hidden, T]
        s_feat = self.semantic_proc(semantic)                # [B, hidden, T]

        # Early layers: GLOBAL FiLM (pool over time, then per-channel scale/shift).
        a_pooled = a_feat.mean(dim=-1)                       # [B, hidden]
        early_params = {}
        for ch_str, head in self.early_film.items():
            gamma_beta = head(a_pooled)                      # [B, 2*ch]
            gamma, beta = gamma_beta.chunk(2, dim=1)         # [B, ch], [B, ch]
            early_params[int(ch_str)] = (gamma, beta)

        # Bottleneck: PER-FRAME FiLM at SSL rate (resampled to T_unet in encode()).
        b_gamma = self.bottleneck_gamma(s_feat)              # [B, C_unet, T]
        b_beta = self.bottleneck_beta(s_feat)                # [B, C_unet, T]

        return early_params, (b_gamma, b_beta)