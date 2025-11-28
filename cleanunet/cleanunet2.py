"""
CleanUNet2: high-level model that composes a spectrogram denoiser (CleanSpecNet)
and a waveform denoiser (CleanUNet), plus small conditioning utilities.

This refactor keeps the original architecture and behavior but:
 - adds clear English comments and docstrings,
 - improves checkpoint loading robustness,
 - ensures shapes & device handling,
 - provides debug prints that are easy to toggle,
 - documents expected input shapes.
"""

from __future__ import annotations

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local module imports (adjust relative paths if needed)
from .cleanspecnet import CleanSpecNet
from .cleanunet import CleanUNet


class SpecUpsampler(nn.Module):
    """
    Upsample a spectrogram in time and collapse freq to a time-domain feature.

    Input:  spec (B, F, T)  -- magnitude spectrogram (no explicit channel dim)
    Output: time_feat (B, 1, L) -- condensed time-domain feature (single channel)

    Implementation notes:
    - Uses two ConvTranspose2d layers with stride=(1, time_kernel).
      With time_kernel=16 stacked twice, time is upsampled by 16*16=256.
    - Frequency dimension is preserved by padding; finally averaged to collapse freq.
    """
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 32,
                 freq_kernel: int = 3,
                 time_kernel: int = 16,
                 leaky_slope: float = 0.4):
        super().__init__()

        # First transpose conv: increases time length by factor time_kernel
        self.up1 = nn.ConvTranspose2d(
            in_channels, hidden_channels,
            kernel_size=(freq_kernel, time_kernel),
            stride=(1, time_kernel),
            padding=(freq_kernel // 2, 0),
            output_padding=(0, 0)
        )
        self.act1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)

        # Second transpose conv: another factor time_kernel => total factor time_kernel^2
        self.up2 = nn.ConvTranspose2d(
            hidden_channels, in_channels,
            kernel_size=(freq_kernel, time_kernel),
            stride=(1, time_kernel),
            padding=(freq_kernel // 2, 0),
            output_padding=(0, 0)
        )
        self.act2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (B, F, T) magnitude spectrogram
        Returns:
            time_feat: (B, 1, T_high) upsampled time-domain feature
        """
        # add channel dim -> (B, 1, F, T)
        x = spec.unsqueeze(1)
        x = self.up1(x)
        x = self.act1(x)
        x = self.up2(x)
        x = self.act2(x)

        # Collapse/fuse frequency axis: here we use a mean across freq for stability.
        # Alternatives: learned conv to fuse freq -> time, or weighted pooling.
        time_feat = x.mean(dim=2, keepdim=True)  # (B, 1, 1, T_high)
        time_feat = time_feat.squeeze(2)         # (B, 1, T_high)
        return time_feat


class WaveformCombiner(nn.Module):
    """
    Combiner that processes (noisy_waveform, upsampled_feature) -> conditioned waveform.

    Expected input: concatenation along channel axis (B, C_in, L)
    Output: (B, out_channels, L)
    """
    def __init__(self,
                 in_channels: int = 2,
                 hidden_channels: int = 16,
                 out_channels: int = 1,
                 kernel_size: int = 7,
                 norm_type: str = "batch"):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for symmetric padding"
        padding = (kernel_size - 1) // 2

        if norm_type.lower() == "batch":
            norm_layer = nn.BatchNorm1d(hidden_channels)
        else:
            raise NotImplementedError("Only 'batch' norm_type is implemented for WaveformConditioner")

        self.conditioner_block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            norm_layer,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, L)
        Returns:
            conditioned: (B, out_channels, L)
        """
        return self.conditioner_block(x)


class CleanUNet2(nn.Module):
    """
    High-level model combining:
      - CleanSpecNet: denoises spectrograms (frequency domain)
      - SpecUpsampler: converts denoised spectrogram -> time feature
      - WaveformCombiner: combine waveform and the upsampled feature
      - CleanUNet: denoises waveform (time domain)

    Forward returns: (denoised_waveform, denoised_spectrogram)
    """

    def __init__(self,
                 # CleanUNet (waveform) parameters
                 cleanunet_input_channels: int = 1,
                 cleanunet_output_channels: int = 1,
                 cleanunet_channels_H: int = 64,
                 cleanunet_max_H: int = 768,
                 cleanunet_encoder_n_layers: int = 8,
                 cleanunet_kernel_size: int = 4,
                 cleanunet_stride: int = 2,
                 cleanunet_tsfm_n_layers: int = 5,
                 cleanunet_tsfm_n_head: int = 8,
                 cleanunet_tsfm_d_model: int = 512,
                 cleanunet_tsfm_d_inner: int = 2048,
                 # CleanSpecNet (spectrogram) parameters
                 cleanspecnet_input_channels: int = 513,
                 cleanspecnet_num_conv_layers: int = 5,
                 cleanspecnet_kernel_size: int = 4,
                 cleanspecnet_stride: int = 1,
                 cleanspecnet_num_attention_layers: int = 5,
                 cleanspecnet_num_heads: int = 8,
                 cleanspecnet_hidden_dim: int = 512,
                 cleanspecnet_dropout: float = 0.1
                 ):
        super().__init__()

        # Instantiate submodules
        self.clean_unet = CleanUNet(
            channels_input=cleanunet_input_channels,
            channels_output=cleanunet_output_channels,
            channels_H=cleanunet_channels_H,
            max_H=cleanunet_max_H,
            encoder_n_layers=cleanunet_encoder_n_layers,
            kernel_size=cleanunet_kernel_size,
            stride=cleanunet_stride,
            tsfm_n_layers=cleanunet_tsfm_n_layers,
            tsfm_n_head=cleanunet_tsfm_n_head,
            tsfm_d_model=cleanunet_tsfm_d_model,
            tsfm_d_inner=cleanunet_tsfm_d_inner
        )

        self.clean_spec_net = CleanSpecNet(
            input_channels=cleanspecnet_input_channels,
            num_conv_layers=cleanspecnet_num_conv_layers,
            kernel_size=cleanspecnet_kernel_size,
            stride=cleanspecnet_stride,
            hidden_dim=cleanspecnet_hidden_dim,
            num_attention_layers=cleanspecnet_num_attention_layers,
            num_heads=cleanspecnet_num_heads,
            dropout=cleanspecnet_dropout
        )

        # Upsampler converts spectrogram -> coarse time feature (single channel)
        self.spec_upsampler = SpecUpsampler(in_channels=1, hidden_channels=32, freq_kernel=3, time_kernel=16)

        # Combiner merges noisy waveform + upsampled feature into a waveform for CleanUNet
        self.WaveformCombiner = WaveformCombiner(in_channels=2, hidden_channels=16, out_channels=1, kernel_size=7)


    def forward(self, noisy_waveform: torch.Tensor, noisy_spectrogram: torch.Tensor, debug: bool = False):
        """
        Forward pass.

        Args:
            noisy_waveform: (B, 1, L) or (B, L) waveform tensor
            noisy_spectrogram: (B, F, T) magnitude spectrogram tensor (no channel dim)
            debug: if True prints debug info

        Returns:
            (denoised_waveform: (B, 1, L), denoised_spectrogram: (B, F, T))
        """
        if debug:
            print(f"[DEBUG] noisy_waveform shape: {noisy_waveform.shape}, noisy_spectrogram shape: {noisy_spectrogram.shape}")

        # 1) Spectrogram denoising (frequency domain)
        denoised_spectrogram = self.clean_spec_net(noisy_spectrogram)  # (B, F, T)
        if debug:
            print(f"[DEBUG] denoised_spectrogram shape: {denoised_spectrogram.shape}")

        # 2) Upsample spectrogram -> time feature
        up_spec_time = self.spec_upsampler(denoised_spectrogram)  # (B, 1, T_high)
        if debug:
            print(f"[DEBUG] up_spec_time shape: {up_spec_time.shape}")

        # 3) Ensure waveform shape is (B, 1, L)
        if noisy_waveform.dim() == 2:
            noisy_waveform = noisy_waveform.unsqueeze(1)
        B, C, L = noisy_waveform.shape
        assert C == 1, f"Expected 1 channel waveform, got {C}"

        # 4) Align sizes: upsampled feature length may differ from waveform length.
        T_high = up_spec_time.shape[-1]
        if T_high != L:
            # Interpolate the time feature to the exact waveform sample count
            up_spec_time = F.interpolate(up_spec_time, size=L, mode="linear", align_corners=False)
            if debug:
                print(f"[DEBUG] interpolated up_spec_time to length {L}")

        # 5) Combine noisy waveform and upsampled feature.
        #    You can add or concatenate. Here we keep both:
        #    - combined_add = noisy + up_spec_time (element-wise conditioning)
        #    - conditioner_input = concat(noisy, up_spec_time) along channel axis
        combined_add = noisy_waveform + up_spec_time
        conditioner_input = torch.cat([noisy_waveform, up_spec_time], dim=1)  # (B, 2, L)
        if debug:
            print(f"[DEBUG] conditioner_input shape: {conditioner_input.shape}")

        # 6) Combiner -> small conv block -> conditioned waveform
        combined_waveform = self.WaveformCombiner(conditioner_input)  # (B, 1, L)
        if debug:
            print(f"[DEBUG] conditioned_waveform shape: {conditioned_waveform.shape}")

        # 7) Waveform denoising (time domain) using CleanUNet
        denoised_waveform = self.clean_unet(combined_waveform)  # (B, 1, L)
        if debug:
            print(f"[DEBUG] denoised_waveform shape: {denoised_waveform.shape}")

        return denoised_waveform, denoised_spectrogram

    # ----------------------------
    # Checkpoint helpers
    # ----------------------------
    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """
        Load a checkpoint file (map_location cpu) and try to extract a state_dict-like object.

        Supports common keys: 'generator', 'state_dict', or raw dict saved with torch.save(model.state_dict()).
        """
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # If a trainer checkpoint is used, weights may be under 'state_dict' or 'model'
        if isinstance(checkpoint, dict):
            for key_candidate in ("generator", "state_dict", "model", "state_dict_cleanunet2"):
                if key_candidate in checkpoint:
                    print(f"[INFO] Found key '{key_candidate}' in checkpoint - using it as state_dict source.")
                    return checkpoint[key_candidate]
            # Fallback: the checkpoint may already be a state_dict
            return checkpoint
        else:
            raise ValueError("Checkpoint content is not a dict. Cannot extract weights.")

    def load_cleanunet_weights(self, checkpoint_path: str, prefix: str = "model."):
        """
        Load pre-trained weights from a checkpoint into the self.clean_unet submodule.
        If the checkpoint keys have a prefix (e.g. 'model.'), that prefix will be removed.

        Args:
            checkpoint_path: path to the checkpoint file
            prefix: optional prefix to strip from keys (default 'model.')
        """
        state_dict = self._load_and_extract_state_dict(checkpoint_path)

        # Filter keys that belong to CleanUNet (we assume they are prefixed in the checkpoint)
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped = k[len(prefix):]
                # Accept keys that start with 'clean_unet.' or which directly match the submodule names
                if stripped.startswith("clean_unet."):
                    new_key = stripped[len("clean_unet."):]
                    filtered[new_key] = v
                # If checkpoint already contains keys for clean_unet without extra prefix, try to match
                elif k.startswith("clean_unet."):
                    filtered[k[len("clean_unet."):]] = v

        if not filtered:
            # Try direct matching fallback: take any key that matches the clean_unet's keys partially
            sub_keys = set(self.clean_unet.state_dict().keys())
            for k, v in state_dict.items():
                # try to align last parts of keys
                key_tail = k.split(".")[-1]
                # attempt to match by suffix (dangerous but useful fallback)
                for sk in sub_keys:
                    if sk.endswith(key_tail):
                        filtered[sk] = v
                        break

        if not filtered:
            raise RuntimeError("[ERROR] No matching CleanUNet keys found in checkpoint. Check the checkpoint format and prefix.")

        self.clean_unet.load_state_dict(filtered, strict=False)
        print("[INFO] Loaded weights into clean_unet (partial load allowed).")

    def load_cleanspecnet_weights(self, checkpoint_path: str, prefix: str = "model."):
        """
        Load pre-trained weights for the CleanSpecNet submodule.
        Works similarly to load_cleanunet_weights.
        """
        state_dict = self._load_and_extract_state_dict(checkpoint_path)

        filtered = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped = k[len(prefix):]
                if stripped.startswith("clean_spec_net."):
                    new_key = stripped[len("clean_spec_net."):]
                    filtered[new_key] = v
                elif k.startswith("clean_spec_net."):
                    filtered[k[len("clean_spec_net."):]] = v

        if not filtered:
            # fallback by suffix matching
            sub_keys = set(self.clean_spec_net.state_dict().keys())
            for k, v in state_dict.items():
                key_tail = k.split(".")[-1]
                for sk in sub_keys:
                    if sk.endswith(key_tail):
                        filtered[sk] = v
                        break

        if not filtered:
            raise RuntimeError("[ERROR] No matching CleanSpecNet keys found in checkpoint. Check the checkpoint format and prefix.")

        self.clean_spec_net.load_state_dict(filtered, strict=False)
        print("[INFO] Loaded weights into clean_spec_net (partial load allowed).")

    def load_cleanunet2_weights(self, checkpoint_path: str):
        """
        Load a checkpoint into the entire CleanUNet2 model (attempt to load state_dict directly).
        """
        state_dict = self._load_and_extract_state_dict(checkpoint_path)
        # Try to load the dict directly (will raise if mismatch)
        self.load_state_dict(state_dict, strict=False)
        print("[INFO] Loaded checkpoint into CleanUNet2 (partial load allowed).")


# Quick sanity check (example) when running module as script
if __name__ == "__main__":
    # This small test demonstrates basic forward shapes and debug prints.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CleanUNet2().to(device)

    batch = 2
    waveform_len = 16000 * 4  # 4s at 16kHz
    noisy_w = torch.randn(batch, 1, waveform_len, device=device)
    noisy_spec = torch.randn(batch, 513, 200, device=device)  # e.g. 200 frames of spec

    denoised_wave, denoised_spec = model(noisy_w, noisy_spec, debug=True)
    print(f"[SANITY] denoised_wave shape: {denoised_wave.shape}, denoised_spec shape: {denoised_spec.shape}")

