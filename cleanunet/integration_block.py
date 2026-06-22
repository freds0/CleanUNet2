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