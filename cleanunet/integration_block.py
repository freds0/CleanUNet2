"""
Integration Blocks for fusing embeddings with latent features.
Supports both pooled embeddings (X-Vectors) and sequence embeddings (raw Wav2Vec2).

Three configurable WavLM->CleanUNet fusion strategies are implemented at the bottom of
this file and selected via the ``fusion_type`` config flag:

    - "cross_attention_film"   -> FiLMCrossAttentionBlock      (Option 1)
    - "cvae_bottleneck"        -> VariationalLatentBlock        (Option 2)
    - "hierarchical_multiscale"-> HierarchicalMultiScaleBlock   (Option 3)

Tensor shape convention used in comments: [B, C, T] for 1-D feature maps,
[B, T, D] for transformer-style sequences. C_unet = CleanUNet bottleneck channels,
T_unet = bottleneck time steps, T_wavlm = WavLM frames, D_wavlm = WavLM hidden size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Integration block for fusing sequence embeddings (raw Wav2Vec2) with latent features.
    Uses self-attention to process sequence embeddings before fusion.

    This handles embeddings with temporal dimension: (batch, seq_len, embedding_dim)
    """

    def __init__(self, latent_channels, embedding_dim=1024, num_heads=8, dropout=0.1):
        """
        Initialize the sequence integration block with self-attention.

        Args:
            latent_channels (int): Number of channels in latent features
            embedding_dim (int): Dimension of sequence embeddings (e.g., 1920 for wav2vec2-xls-r-2b)
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
# OPTION 1 — Dynamic Modulation Pipeline (Frame-to-Frame Cross-Attention + FiLM)
# ============================================================================
class FiLMCrossAttentionBlock(nn.Module):
    """
    Frame-aligned fusion: every CleanUNet bottleneck frame (query) attends over the
    full WavLM sequence (keys/values), then the attended context predicts per-frame
    FiLM parameters (gamma, beta) that modulate the latent.

    No temporal pooling is performed, so the temporal structure of the WavLM
    representation is preserved (unlike the pooled SequenceIntegrationBlock).
    """

    def __init__(self, latent_channels, embedding_dim=1024, num_heads=8, dropout=0.1):
        """
        Args:
            latent_channels (int): CleanUNet bottleneck channels C_unet (query/output dim).
            embedding_dim (int): WavLM hidden size D_wavlm (key/value dim).
            num_heads (int): Attention heads (latent_channels must be divisible by it).
            dropout (float): Attention dropout.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim

        # MultiheadAttention with distinct query vs key/value dims (kdim/vdim).
        # Query  = latent frames  -> embed_dim = latent_channels (C_unet)
        # Key/Val = WavLM frames   -> kdim = vdim = embedding_dim (D_wavlm)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_channels,
            num_heads=num_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(latent_channels)

        # Two 1x1 conv heads predict per-frame FiLM scale/shift from the attended context.
        self.gamma_conv = nn.Conv1d(latent_channels, latent_channels, kernel_size=1)
        self.beta_conv = nn.Conv1d(latent_channels, latent_channels, kernel_size=1)

    def forward(self, latent_features, sequence_embeddings, key_padding_mask=None):
        """
        Args:
            latent_features (Tensor): [B, C_unet, T_unet] CleanUNet bottleneck latent.
            sequence_embeddings (Tensor): [B, T_wavlm, D_wavlm] full WavLM sequence.
            key_padding_mask (Tensor|None): [B, T_wavlm] True where padded (ignored in attn).

        Returns:
            fused (Tensor): [B, C_unet, T_unet] FiLM-modulated latent.
        """
        B, C, T = latent_features.shape                      # [B, C_unet, T_unet]

        query = latent_features.transpose(1, 2)              # [B, T_unet, C_unet]
        # Cross-attention: queries (latent frames) attend over WavLM frames.
        attn_out, _ = self.cross_attention(
            query=query,                                     # [B, T_unet, C_unet]
            key=sequence_embeddings,                         # [B, T_wavlm, D_wavlm]
            value=sequence_embeddings,                       # [B, T_wavlm, D_wavlm]
            key_padding_mask=key_padding_mask,
        )                                                    # -> [B, T_unet, C_unet]
        attn_out = self.attn_norm(attn_out + query)          # residual + norm, [B, T_unet, C_unet]

        attn_out = attn_out.transpose(1, 2)                  # [B, C_unet, T_unet]
        gamma = self.gamma_conv(attn_out)                    # [B, C_unet, T_unet]
        beta = self.beta_conv(attn_out)                      # [B, C_unet, T_unet]

        # FiLM modulation: Out = (1 + gamma) * x + beta
        fused = (1.0 + gamma) * latent_features + beta       # [B, C_unet, T_unet]
        return fused


# ============================================================================
# OPTION 2 — Robust Variational Blueprint (CVAE Latent Filter + Stable Predictor)
# ============================================================================
class VariationalLatentBlock(nn.Module):
    """
    Conditional-VAE style fusion. WavLM features parameterize a Gaussian over a latent
    conditioning vector z (per channel). In Stage-1 z is sampled (reparameterization
    trick) to regularize the latent space with a KL term; in Stage-2 the mean ``mu`` is
    used deterministically. z is broadcast over time and fused into the bottleneck.

    Accepts either sequence embeddings [B, T_wavlm, D] (mean-pooled internally) or
    pre-pooled embeddings [B, D].
    """

    def __init__(self, latent_channels, embedding_dim=1024):
        """
        Args:
            latent_channels (int): CleanUNet bottleneck channels C_unet.
            embedding_dim (int): WavLM hidden size D_wavlm.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim

        # Gaussian parameterization heads.
        self.fc_mu = nn.Linear(embedding_dim, latent_channels)      # D_wavlm -> C_unet
        self.fc_logvar = nn.Linear(embedding_dim, latent_channels)  # D_wavlm -> C_unet

        # Fuse the (broadcast) conditioning vector with the latent.
        self.fusion_conv = nn.Conv1d(latent_channels * 2, latent_channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(latent_channels)
        self.activation = nn.PReLU()

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + eps * std,  std = exp(0.5 * logvar),  eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)                        # [B, C_unet]
        eps = torch.randn_like(std)                          # [B, C_unet] (matches device/dtype)
        return mu + eps * std                                # [B, C_unet]

    def forward(self, latent_features, embeddings, sample=True):
        """
        Args:
            latent_features (Tensor): [B, C_unet, T_unet] CleanUNet bottleneck latent.
            embeddings (Tensor): [B, T_wavlm, D_wavlm] sequence OR [B, D_wavlm] pooled.
            sample (bool): True (Stage-1) -> sample z; False (Stage-2) -> z = mu.

        Returns:
            fused (Tensor): [B, C_unet, T_unet] conditioned latent.
            mu (Tensor):    [B, C_unet] Gaussian mean (KL term / Stage-2 target).
            logvar (Tensor):[B, C_unet] Gaussian log-variance (KL term).
        """
        if embeddings.dim() == 3:
            pooled = embeddings.mean(dim=1)                  # [B, D_wavlm] global average pool
        else:
            pooled = embeddings                              # [B, D_wavlm]

        mu = self.fc_mu(pooled)                              # [B, C_unet]
        logvar = self.fc_logvar(pooled)                      # [B, C_unet]

        z = self.reparameterize(mu, logvar) if sample else mu  # [B, C_unet]

        T = latent_features.size(-1)
        z_expanded = z.unsqueeze(-1).expand(-1, -1, T)       # [B, C_unet, T_unet]

        concatenated = torch.cat([latent_features, z_expanded], dim=1)  # [B, 2*C_unet, T_unet]
        fused = self.fusion_conv(concatenated)               # [B, C_unet, T_unet]

        fused = fused.transpose(1, 2)                        # [B, T_unet, C_unet]
        fused = self.norm(fused)
        fused = fused.transpose(1, 2)                        # [B, C_unet, T_unet]
        fused = self.activation(fused)
        return fused, mu, logvar


# ============================================================================
# OPTION 3 — Multi-Scale Structural Network (Dilated Convs + Layer-Wise Hierarchy)
# ============================================================================
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
    Splits the WavLM layer stack into an Acoustic group (low layers, fine acoustic
    detail) and a Semantic group (high layers, phonetic/content), processes each with
    MultiScaleDilatedConv, and produces FiLM parameters for hierarchical injection:

        - Acoustic group -> GLOBAL FiLM for the early CleanUNet encoder layers (1 & 2).
        - Semantic group -> PER-FRAME FiLM for the Transformer bottleneck.

    The actual FiLM application happens inside ``CleanUNet.encode`` (encoder_film /
    bottleneck_film arguments); this block only generates the parameters.
    """

    def __init__(self, embedding_dim, bottleneck_channels, early_channels=(64, 128),
                 acoustic_layers=(1, 8), semantic_layers=(17, 24), hidden=256):
        """
        Args:
            embedding_dim (int): WavLM hidden size D_wavlm.
            bottleneck_channels (int): CleanUNet bottleneck channels C_unet.
            early_channels (tuple[int]): channel counts of encoder layers 1 & 2.
            acoustic_layers (tuple[int]): inclusive WavLM layer index range (low group).
            semantic_layers (tuple[int]): inclusive WavLM layer index range (high group).
            hidden (int): hidden width of the multi-scale processors.
        """
        super().__init__()
        self.early_channels = tuple(early_channels)
        self.acoustic_layers = acoustic_layers
        self.semantic_layers = semantic_layers

        self.acoustic_proc = MultiScaleDilatedConv(embedding_dim, hidden)   # -> [B, hidden, T_wavlm]
        self.semantic_proc = MultiScaleDilatedConv(embedding_dim, hidden)   # -> [B, hidden, T_wavlm]

        # GLOBAL FiLM heads (one per early encoder layer): hidden -> 2 * channels.
        self.early_film = nn.ModuleDict({
            str(ch): nn.Linear(hidden, ch * 2) for ch in self.early_channels
        })

        # PER-FRAME FiLM heads for the bottleneck (semantic group).
        self.bottleneck_gamma = nn.Conv1d(hidden, bottleneck_channels, kernel_size=1)
        self.bottleneck_beta = nn.Conv1d(hidden, bottleneck_channels, kernel_size=1)

    def forward(self, all_layer_states):
        """
        Args:
            all_layer_states (Tensor): [B, L_wavlm, T_wavlm, D_wavlm] stacked hidden
                states (L_wavlm = num_hidden_layers + 1).

        Returns:
            early_params (dict[int, tuple[Tensor, Tensor]]): {channels: (gamma [B, C],
                beta [B, C])} GLOBAL FiLM params for early encoder layers.
            bottleneck_params (tuple[Tensor, Tensor]): (gamma, beta), each
                [B, C_unet, T_wavlm] PER-FRAME FiLM params for the bottleneck.
        """
        a_lo, a_hi = self.acoustic_layers
        s_lo, s_hi = self.semantic_layers

        # Group-average over the selected WavLM layers -> [B, T_wavlm, D_wavlm]
        acoustic = all_layer_states[:, a_lo:a_hi + 1].mean(dim=1)
        semantic = all_layer_states[:, s_lo:s_hi + 1].mean(dim=1)

        # To conv layout [B, D_wavlm, T_wavlm]
        acoustic = acoustic.transpose(1, 2)                  # [B, D_wavlm, T_wavlm]
        semantic = semantic.transpose(1, 2)                  # [B, D_wavlm, T_wavlm]

        a_feat = self.acoustic_proc(acoustic)                # [B, hidden, T_wavlm]
        s_feat = self.semantic_proc(semantic)                # [B, hidden, T_wavlm]

        # Early layers: GLOBAL FiLM (pool over time, then per-channel scale/shift).
        a_pooled = a_feat.mean(dim=-1)                       # [B, hidden]
        early_params = {}
        for ch_str, head in self.early_film.items():
            gamma_beta = head(a_pooled)                      # [B, 2*ch]
            gamma, beta = gamma_beta.chunk(2, dim=1)         # [B, ch], [B, ch]
            early_params[int(ch_str)] = (gamma, beta)

        # Bottleneck: PER-FRAME FiLM at WavLM rate (resampled to T_unet in encode()).
        b_gamma = self.bottleneck_gamma(s_feat)              # [B, C_unet, T_wavlm]
        b_beta = self.bottleneck_beta(s_feat)                # [B, C_unet, T_wavlm]

        return early_params, (b_gamma, b_beta)