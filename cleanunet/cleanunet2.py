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

        # Option 2
        self.freq_projector = nn.Conv1d(513, 1, kernel_size=1)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # Add channel dimension -> (B, 1, F, T)
        x = spec.unsqueeze(1) if spec.dim() == 3 else spec
        
        x = self.act2(self.up2(self.act1(self.up1(x))))

        # Average across frequency axis to obtain 1D temporal feature
        #return x.mean(dim=2, keepdim=True).squeeze(2)

        # The article describes upsampling the predicted spectrogram 256 times using two 2D transposed convolutions. It states: "we combined the noisy waveform and the upsampled spectrogram through a conditioning method". There is no explicit mention of discarding frequency information prior to this combination.

        # Option 1
        # Calculates weights via Softmax along the frequency dimension (dim 2)
        ''' 
        attn_weights = F.softmax(x, dim=2)
        Multiply by the weights and sum (weighted average)
        return (x * attn_weights).sum(dim=2)
        ''' 
        # Option 2
        # Instead of forcing an average, you let the model learn a weight for each frequency. You will need to "flatten" the frequency dimension to the channel dimension and then project it back.
        B, C, F, T = x.shape
        x = x.view(B, C * F, T) # Flatten Frequency and Channels: (B, F, T)
        return self.freq_projector(x) # Learn how to combine frequencies.


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


    # ------------------------------------------------------------------
    # Checkpoint Loading Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path):
        """Helper method to load a checkpoint and extract the state_dict."""
        print(f"Loading checkpoint from: {checkpoint_path}")
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

        # The correct prefix, based on output, is "model."
        prefix = "model." 
        
        for k, v in state_dict.items():
            # Only process keys starting with the expected prefix
            if k.startswith(prefix):
                # Remove the prefix so keys match the submodule
                clean_unet_state_dict[k[len(prefix):]] = v
        
        if not clean_unet_state_dict:
            raise ValueError(f"No compatible keys found in checkpoint. Check the prefix. Available keys: {state_dict.keys()}")

        self.clean_unet.load_state_dict(clean_unet_state_dict)
        print("[SUCCESS] Weights loaded successfully into self.clean_unet.")


    def load_cleanspecnet_weights(self, checkpoint_path):
        """Loads pre-trained weights specifically for the CleanSpecNet submodule."""
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        
        clean_spec_net_state_dict = {}
        # The correct prefix, based on error logs, is also "model."
        prefix = "model."
        
        for k, v in state_dict.items():
            # Only process keys starting with the expected prefix
            if k.startswith(prefix):
                # Remove the prefix so keys match the submodule
                clean_spec_net_state_dict[k[len(prefix):]] = v
        
        if not clean_spec_net_state_dict:
            raise ValueError(f"No compatible keys found in checkpoint for CleanSpecNet. Check the prefix. Available keys: {state_dict.keys()}")

        self.clean_spec_net.load_state_dict(clean_spec_net_state_dict)
        print("[SUCCESS] Weights loaded successfully into self.clean_spec_net.")


    def load_cleanunet2_weights(self, checkpoint_path):
        """Loads pre-trained weights for the full CleanUNet2 model."""
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        self.load_state_dict(state_dict)
        print("[SUCCESS] Weights loaded successfully into the full CleanUNet2 model.")
