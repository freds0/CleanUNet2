"""
Integration Blocks for fusing X-Vectors with latent features.
Similar to the Integration Block described in the CNUNet-TB paper.
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