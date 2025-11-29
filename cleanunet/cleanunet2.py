"""
CleanUNet2: Hybrid speech denoising model on waveform and spectrogram.
Based on the architecture proposed in "CleanUNet 2: A Hybrid Speech Denoising Model on Waveform and Spectrogram".

Implementation details:
 - The combination module is named `Conditioner` (referred to as "conditioning method" in the paper).
 - Supports 3 conditioning types: Addition, Concatenation, and FiLM.
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports (adjust relative paths if needed)
from .cleanspecnet import CleanSpecNet
from .cleanunet import CleanUNet


class SpecUpsampler(nn.Module):
    """
    Upsamples the spectrogram in time and collapses the frequency axis to produce a time-domain feature.
    
    Input: spec (B, F, T)
    Output: time_feat (B, 1, L)
    """
    def __init__(self, in_channels=1, hidden_channels=32, freq_kernel=3, time_kernel=16, leaky_slope=0.4):
        super().__init__()
        # Two ConvTranspose2d layers for temporal upsampling (factor 16*16 = 256)
        self.up1 = nn.ConvTranspose2d(
            in_channels, hidden_channels, 
            (freq_kernel, time_kernel), 
            stride=(1, time_kernel), 
            padding=(freq_kernel // 2, 0)
        )
        self.act1 = nn.LeakyReLU(leaky_slope)
        
        self.up2 = nn.ConvTranspose2d(
            hidden_channels, in_channels, 
            (freq_kernel, time_kernel), 
            stride=(1, time_kernel), 
            padding=(freq_kernel // 2, 0)
        )
        self.act2 = nn.LeakyReLU(leaky_slope)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # Add channel dimension -> (B, 1, F, T)
        x = spec.unsqueeze(1) if spec.dim() == 3 else spec
        
        x = self.act2(self.up2(self.act1(self.up1(x))))

        # Average across frequency axis to obtain 1D temporal feature
        return x.mean(dim=2, keepdim=True).squeeze(2)


class Conditioner(nn.Module):
    """
    Implements the "Conditioning Method" described in the CleanUNet 2 paper.
    It combines the noisy waveform and the spectrogram feature (upsampled).
    
    Supported methods:
     1. 'addition': Element-wise addition (Paper default).
     2. 'concatenation': Concatenation along channels + projection.
     3. 'film': Feature-wise Linear Modulation.

    Input: Waveform and Condition tensors
    Output: Conditioned waveform (B, out_channels, L)
    """
    def __init__(self, method: str = "addition", input_channels: int = 1, cond_channels: int = 1):
        super().__init__()
        self.method = method.lower()
        
        if self.method == "concatenation":
            # Concatenates channels and projects back to 1 channel via Conv1d
            total_channels = input_channels + cond_channels
            self.concat_proj = nn.Sequential(
                nn.Conv1d(total_channels, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.2),
                nn.Conv1d(16, input_channels, kernel_size=1)
            )
            
        elif self.method == "film":
            # FiLM: Predicts scale (gamma) and shift (beta) from the condition
            # cond (B, C, L) -> gamma (B, C, L), beta (B, C, L)
            self.film_gen = nn.Conv1d(cond_channels, input_channels * 2, kernel_size=3, padding=1)
            
        elif self.method == "addition":
            # Direct addition (no extra parameters needed if dimensions match)
            pass
        else:
            raise ValueError(f"Unknown conditioning method '{method}'. Use: addition, concatenation, film")

    def forward(self, waveform: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, 1, L) - Noisy waveform
            condition: (B, 1, L) - Feature from SpecUpsampler
        Returns:
            (B, 1, L) - Conditioned waveform ready for CleanUNet
        """
        if self.method == "addition":
            return waveform + condition

        elif self.method == "concatenation":
            x = torch.cat([waveform, condition], dim=1)
            return self.concat_proj(x)

        elif self.method == "film":
            params = self.film_gen(condition) # (B, 2, L) assuming input_channels=1
            gamma, beta = torch.chunk(params, 2, dim=1)
            # Apply affine transformation: (1 + gamma) * x + beta
            return (1.0 + gamma) * waveform + beta
            
        return waveform


class CleanUNet2(nn.Module):
    def __init__(self,
                 conditioning_type: str = "addition",
                 cleanunet_params: dict = None,
                 cleanspecnet_params: dict = None):
        super().__init__()
        
        if cleanunet_params is None: cleanunet_params = {}
        if cleanspecnet_params is None: cleanspecnet_params = {}

        # 1. CleanUNet (Waveform Denoiser)
        self.clean_unet = CleanUNet(**cleanunet_params)

        # 2. CleanSpecNet (Spectrogram Denoiser)
        self.clean_spec_net = CleanSpecNet(**cleanspecnet_params)

        # 3. Upsampler
        self.spec_upsampler = SpecUpsampler()

        # 4. Conditioner (Selectable)
        print(f"[INFO] CleanUNet2 initialized with conditioning method: {conditioning_type}")
        self.conditioner = Conditioner(method=conditioning_type, input_channels=1, cond_channels=1)

    def forward(self, noisy_waveform, noisy_spectrogram, debug=False):
        # 1. Denoise Spectrogram
        denoised_spec = self.clean_spec_net(noisy_spectrogram)
        
        # 2. Upsample to time domain
        cond_feature = self.spec_upsampler(denoised_spec)
        
        # 3. Align lengths (Linear interpolation if sizes mismatch)
        if cond_feature.shape[-1] != noisy_waveform.shape[-1]:
            cond_feature = F.interpolate(cond_feature, size=noisy_waveform.shape[-1], mode='linear')

        # 4. Apply Conditioning (Addition, Concat, or FiLM)
        conditioned_input = self.conditioner(noisy_waveform, cond_feature)
        
        # 5. Denoise Waveform
        denoised_waveform = self.clean_unet(conditioned_input)
        
        return denoised_waveform, denoised_spec

    # ... (Load weight methods remain the same, just ensure logs are in English) ...
    def load_cleanunet_weights(self, checkpoint_path: str):
        # Implementation omitted for brevity (same logic as before)
        pass

    def load_cleanspecnet_weights(self, checkpoint_path: str):
        # Implementation omitted for brevity
        pass