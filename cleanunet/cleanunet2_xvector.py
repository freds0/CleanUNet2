"""
CleanUNet2 with X-Vector Integration
Implements two-stage training similar to CNUNet-TB paper:
- Stage 1: Train with X-Vectors injected into latent space
- Stage 2: Train to replicate latent vectors without X-Vector extractor
"""

import torch
import torch.nn as nn
from .cleanunet import CleanUNet
from .integration_block import IntegrationBlock
from .xvector_extractor import XVectorExtractor


class CleanUNet2XVector(nn.Module):
    """
    CleanUNet2 model with X-Vector integration for speech enhancement.
    
    Architecture:
        1. Encoder extracts latent features from noisy audio
        2. X-Vectors are integrated with latent features (Stage 1 only)
        3. Decoder reconstructs enhanced audio
        
    Two-stage training:
        - Stage 1: Use X-Vector extractor, save latent vectors
        - Stage 2: Train without X-Vector extractor, replicate saved latents
    """
    
    def __init__(
        self,
        channels_input=1,
        channels_output=1,
        channels_H=64,
        max_H=768,
        encoder_n_layers=8,
        kernel_size=4,
        stride=2,
        tsfm_n_layers=5,
        tsfm_n_head=8,
        tsfm_d_model=512,
        tsfm_d_inner=2048,
        use_xvector=True,
        xvector_stage='stage1',  # 'stage1', 'stage2', or 'none'
        xvector_dim=512
    ):
        """
        Initialize CleanUNet2 with X-Vector integration.
        
        Args:
            channels_input (int): Number of input channels
            channels_output (int): Number of output channels
            channels_H (int): Base number of hidden channels
            max_H (int): Maximum number of hidden channels
            encoder_n_layers (int): Number of encoder layers
            kernel_size (int): Kernel size for convolutions
            stride (int): Stride for convolutions
            tsfm_n_layers (int): Number of transformer layers
            tsfm_n_head (int): Number of attention heads
            tsfm_d_model (int): Transformer model dimension
            tsfm_d_inner (int): Transformer inner dimension
            use_xvector (bool): Whether to use X-Vectors
            xvector_stage (str): Training stage ('stage1', 'stage2', or 'none')
            xvector_dim (int): Dimension of X-Vector embeddings
        """
        super().__init__()
        
        self.use_xvector = use_xvector
        self.xvector_stage = xvector_stage
        self.latent_dim = tsfm_d_model
        self.xvector_dim = xvector_dim
        
        print(f"[CleanUNet2XVector] Initializing model...")
        print(f"  - Use X-Vectors: {use_xvector}")
        print(f"  - Training stage: {xvector_stage}")
        print(f"  - Latent dimension: {self.latent_dim}")
        
        # Base CleanUNet model (waveform processing)
        self.cleanunet = CleanUNet(
            channels_input=channels_input,
            channels_output=channels_output,
            channels_H=channels_H,
            max_H=max_H,
            encoder_n_layers=encoder_n_layers,
            kernel_size=kernel_size,
            stride=stride,
            tsfm_n_layers=tsfm_n_layers,
            tsfm_n_head=tsfm_n_head,
            tsfm_d_model=tsfm_d_model,
            tsfm_d_inner=tsfm_d_inner
        )
        
        # X-Vector extractor (only needed in Stage 1)
        if use_xvector and xvector_stage == 'stage1':
            print("[CleanUNet2XVector] Loading X-Vector extractor...")
            self.xvector_extractor = XVectorExtractor()
        else:
            self.xvector_extractor = None
            print("[CleanUNet2XVector] X-Vector extractor not loaded (Stage 2 or disabled)")
        
        # Integration block for fusing X-Vectors with latent features
        if use_xvector:
            print("[CleanUNet2XVector] Creating integration block...")
            self.integration_block = IntegrationBlock(
                latent_channels=self.latent_dim,
                xvector_dim=xvector_dim
            )
        
        # Latent predictor for Stage 2
        # Trains the model to replicate latents without X-Vectors
        if xvector_stage == 'stage2':
            print("[CleanUNet2XVector] Creating latent predictor for Stage 2...")
            self.latent_predictor = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, 1),
                nn.PReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1)
            )
        
        print("[CleanUNet2XVector] Model initialized successfully!")
    
    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path):
        """Helper method to load a checkpoint and extract the state_dict."""
        print(f"[CleanUNet2XVector] Loading checkpoint from: {checkpoint_path}")
        # Load to CPU to avoid device compatibility issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Search for common keys where model weights are stored
        if 'generator' in checkpoint:
            return checkpoint['generator']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            return checkpoint

    def load_cleanunet_weights(self, checkpoint_path):
        """Loads pre-trained weights specifically for the CleanUNet submodule."""
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        
        clean_unet_state_dict = {}

        # Check if the checkpoint comes from a LightningModule (usually prefixed with "model.")
        # or if it is a raw model checkpoint.
        prefix = "model."
        
        for k, v in state_dict.items():
            # If the checkpoint has "model." prefix, strip it
            if k.startswith(prefix):
                clean_unet_state_dict[k[len(prefix):]] = v
            else:
                # If no prefix, assume it matches directly
                clean_unet_state_dict[k] = v
        
        # Verify if we found keys
        if not clean_unet_state_dict:
            raise ValueError(f"No compatible keys found. keys: {list(state_dict.keys())[:5]}...")

        # Load into self.cleanunet (Note: in your code it is .cleanunet, not .clean_unet)
        try:
            self.cleanunet.load_state_dict(clean_unet_state_dict, strict=False)
            print("[CleanUNet2XVector] Weights loaded successfully into self.cleanunet.")
        except RuntimeError as e:
            print(f"[Warning] strict=False loading failed slightly: {e}")
            
    def forward(self, noisy_audio, clean_audio=None, return_latents=False):
        """
        Forward pass through the model.
        
        Args:
            noisy_audio (torch.Tensor): Noisy input audio 
                                        Shape: (batch, 1, samples)
            clean_audio (torch.Tensor): Clean reference audio (only for Stage 1)
                                        Shape: (batch, 1, samples)
            return_latents (bool): Whether to return latent vectors
            
        Returns:
            enhanced_audio (torch.Tensor): Enhanced audio output
                                          Shape: (batch, 1, samples)
            latents (dict): Dictionary of latent vectors (if return_latents=True)
        """
        latents = {}
        
        # ============ STAGE 1: With X-Vectors ============
        if self.xvector_stage == 'stage1' and self.xvector_extractor is not None:
            assert clean_audio is not None, "Clean audio is required for Stage 1 training"
            
            # Extract X-Vectors from clean audio
            with torch.no_grad():
                xvector_emb = self.xvector_extractor.extract_embeddings(
                    clean_audio.squeeze(1)
                )  
                # SpeechBrain returns (batch, 1, 512) typically, we need (batch, 512)
                if xvector_emb.dim() == 3:
                    xvector_emb = xvector_emb.squeeze(1)
            
            # Forward pass with X-Vector integration
            enhanced_audio, fused_latent = self._forward_with_xvector(
                noisy_audio, xvector_emb
            )
            
            # Store latents for Stage 2 training
            latents['fused_latent'] = fused_latent.detach()
            latents['xvector_emb'] = xvector_emb.detach()
        
        # ============ STAGE 2: Without X-Vectors (Replicating Latents) ============
        elif self.xvector_stage == 'stage2':
            # Forward pass predicting latents without X-Vectors
            enhanced_audio, predicted_latent = self._forward_predict_latent(
                noisy_audio
            )
            
            latents['predicted_latent'] = predicted_latent
        
        # ============ No X-Vectors (Baseline) ============
        else:
            enhanced_audio = self.cleanunet(noisy_audio)
        
        if return_latents:
            return enhanced_audio, latents
        else:
            return enhanced_audio
    
    def _forward_with_xvector(self, noisy_audio, xvector_emb):
        """
        Forward pass with X-Vector integration (Stage 1).
        
        Args:
            noisy_audio (torch.Tensor): Noisy audio (batch, 1, samples)
            xvector_emb (torch.Tensor): X-Vector embeddings (batch, 512)
            
        Returns:
            enhanced_audio (torch.Tensor): Enhanced audio (batch, 1, samples)
            fused_latent (torch.Tensor): Fused latent features
        """
        # Encode noisy audio to latent space
        encoded = self.cleanunet.encode(noisy_audio)  
        # Shape: (batch, latent_dim, time)
        
        # Interpolate X-Vectors to match temporal dimension
        time_steps = encoded.shape[-1]
        
        # xvector_emb is (B, 512) -> unsqueeze -> (B, 512, 1) -> expand -> (B, 512, T)
        xvector_interpolated = xvector_emb.unsqueeze(-1).expand(-1, -1, time_steps)
        # Shape: (batch, 512, time)
        
        # Integrate X-Vectors with latent features
        fused_latent = self.integration_block(encoded, xvector_interpolated)
        # Shape: (batch, latent_dim, time)
        
        # Decode to enhanced audio
        enhanced_audio = self.cleanunet.decode(fused_latent)
        
        return enhanced_audio, fused_latent
    
    def _forward_predict_latent(self, noisy_audio):
        """
        Forward pass predicting latents without X-Vectors (Stage 2).
        
        Args:
            noisy_audio (torch.Tensor): Noisy audio (batch, 1, samples)
            
        Returns:
            enhanced_audio (torch.Tensor): Enhanced audio (batch, 1, samples)
            predicted_latent (torch.Tensor): Predicted latent features
        """
        # Encode noisy audio
        encoded = self.cleanunet.encode(noisy_audio)
        
        # Predict latent (trying to replicate Stage 1's fused latent)
        predicted_latent = self.latent_predictor(encoded)
        
        # Decode to enhanced audio
        enhanced_audio = self.cleanunet.decode(predicted_latent)
        
        return enhanced_audio, predicted_latent