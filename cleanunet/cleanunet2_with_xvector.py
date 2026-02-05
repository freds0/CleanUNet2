"""
CleanUNet2 with X-Vector Integration for Two-Stage Training
Based on CNUNet-TB paper: Two-stage training using self-supervised speech embeddings

Architecture:
    - Stage 1: Train with X-Vectors injected into latent space
    - Stage 2: Train to replicate latent vectors without X-Vector extractor

Implementation inspired by CNUNet-TB but adapted for CleanUNet2 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cleanunet2 import CleanUNet2, SpecUpsampler, Conditioner
from .cleanunet import CleanUNet
from .cleanspecnet import CleanSpecNet
from .integration_block import IntegrationBlock
from .xvector_extractor import XVectorExtractor


class CleanUNet2WithXVector(nn.Module):
    """
    CleanUNet2 model with X-Vector integration for two-stage training.

    Stage 1: Uses X-Vector extractor to inject speaker embeddings into latent space
    Stage 2: Replicates latent vectors without X-Vector extractor

    This approach allows the model to benefit from speaker information during training
    while maintaining fast inference speed (Stage 2 doesn't use X-Vector extractor).
    """

    def __init__(
        self,
        stage='stage1',
        use_xvector=True,
        xvector_dim=512,
        conditioning_type='addition',
        cleanunet_params=None,
        cleanspecnet_params=None,
        xvector_local_path=None
    ):
        """
        Initialize CleanUNet2 with X-Vector integration.

        Args:
            stage (str): Training stage ('stage1' or 'stage2')
            use_xvector (bool): Whether to use X-Vectors
            xvector_dim (int): Dimension of X-Vector embeddings (default: 512)
            conditioning_type (str): Conditioning method (addition, concatenation, film)
            cleanunet_params (dict): Parameters for CleanUNet
            cleanspecnet_params (dict): Parameters for CleanSpecNet
        """
        super().__init__()

        self.stage = stage
        self.use_xvector = use_xvector
        self.xvector_dim = xvector_dim

        if cleanunet_params is None:
            cleanunet_params = {}
        if cleanspecnet_params is None:
            cleanspecnet_params = {}

        print(f"[CleanUNet2WithXVector] Initializing model...")
        print(f"  - Stage: {stage}")
        print(f"  - Use X-Vectors: {use_xvector}")
        print(f"  - X-Vector Dim: {xvector_dim}")
        print(f"  - Conditioning: {conditioning_type}")

        # Calculate latent dimension from CleanUNet params
        # The latent is the encoder output channels, not tsfm_d_model
        # Encoder grows: channels_H -> channels_H*2 -> ... -> min(channels_H*2^n, max_H)
        channels_H = cleanunet_params.get('channels_H', 64)
        max_H = cleanunet_params.get('max_H', 768)
        encoder_n_layers = cleanunet_params.get('encoder_n_layers', 8)

        # Calculate final encoder output channels
        H = channels_H
        for i in range(encoder_n_layers):
            if i > 0:  # First layer uses channels_H directly
                H = min(H * 2, max_H)

        self.latent_dim = H
        print(f"  - Calculated Latent Dim: {self.latent_dim}")

        # ============ Base CleanUNet2 Components ============

        # 1. CleanUNet (Waveform Denoiser)
        self.clean_unet = CleanUNet(**cleanunet_params)

        # 2. CleanSpecNet (Spectrogram Denoiser)
        self.clean_spec_net = CleanSpecNet(**cleanspecnet_params)

        # 3. SpecUpsampler (Upsample spec to waveform domain)
        self.spec_upsampler = SpecUpsampler()

        # 4. Conditioner (Combine waveform + spec)
        self.conditioner = Conditioner(
            method=conditioning_type,
            input_channels=1,
            cond_channels=1
        )

        # ============ X-Vector Components ============

        # X-Vector Extractor (Stage 1 only)
        if stage == 'stage1' and use_xvector:
            print("[CleanUNet2WithXVector] Loading X-Vector extractor...")
            self.xvector_extractor = XVectorExtractor(
                device='cpu',  # Will be moved to correct device by Lightning
                local_path=xvector_local_path
            )
            # Freeze X-Vector extractor
            for param in self.xvector_extractor.parameters():
                param.requires_grad = False
            self.xvector_extractor.eval()
        else:
            self.xvector_extractor = None
            print("[CleanUNet2WithXVector] X-Vector extractor not loaded")

        # Integration Block (for fusing X-Vectors with latent features)
        if use_xvector:
            print("[CleanUNet2WithXVector] Creating integration block...")
            self.integration_block = IntegrationBlock(
                latent_channels=self.latent_dim,
                xvector_dim=xvector_dim
            )
        else:
            self.integration_block = None

        # Latent Predictor (Stage 2 only)
        if stage == 'stage2' and use_xvector:
            print("[CleanUNet2WithXVector] Creating latent predictor for Stage 2...")
            self.latent_predictor = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1),
                nn.PReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1)
            )
        else:
            self.latent_predictor = None

        print("[CleanUNet2WithXVector] Model initialized successfully!")

    def load_vanilla_checkpoint(self, checkpoint_path):
        """
        Load weights from vanilla CleanUNet2 checkpoint.
        This allows warm-starting Stage-1 training with pre-trained vanilla weights.

        Args:
            checkpoint_path (str): Path to vanilla CleanUNet2 checkpoint
        """
        print(f"\n[CleanUNet2WithXVector] Loading vanilla checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state dict (handle Lightning format)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present (from Lightning)
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k[len('model.'):]
            else:
                new_key = k

            # Only load compatible components (skip X-Vector related components)
            if not new_key.startswith(('xvector_extractor', 'integration_block', 'latent_predictor')):
                filtered_state_dict[new_key] = v

        # Load with strict=False to allow missing keys (X-Vector components)
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

        # Report loading status
        print(f"[CleanUNet2WithXVector] Loaded {len(filtered_state_dict)} parameters from vanilla checkpoint")

        if missing_keys:
            # Filter out expected missing keys (X-Vector related)
            expected_missing = [k for k in missing_keys if any(
                k.startswith(prefix) for prefix in ['xvector_extractor', 'integration_block', 'latent_predictor']
            )]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]

            if expected_missing:
                print(f"[CleanUNet2WithXVector] Expected missing keys (new components): {len(expected_missing)}")
                if len(expected_missing) <= 5:
                    for key in expected_missing:
                        print(f"  - {key}")

            if unexpected_missing:
                print(f"[WARNING] Unexpected missing keys: {unexpected_missing[:5]}")

        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys[:5]}")

        print("[CleanUNet2WithXVector] Vanilla checkpoint loaded successfully!\n")

    def forward(self, noisy_waveform, noisy_spectrogram, clean_audio=None, return_latents=False):
        """
        Forward pass through the model.

        Args:
            noisy_waveform (torch.Tensor): Noisy input waveform
                                          Shape: (batch, 1, samples)
            noisy_spectrogram (torch.Tensor): Noisy input spectrogram
                                              Shape: (batch, freq, time)
            clean_audio (torch.Tensor): Clean reference audio (Stage 1 only)
                                        Shape: (batch, 1, samples)
            return_latents (bool): Whether to return latent vectors

        Returns:
            enhanced_waveform (torch.Tensor): Enhanced audio output
            enhanced_spec (torch.Tensor): Enhanced spectrogram output
            latents (dict): Dictionary of latent vectors (if return_latents=True)
        """
        latents = {}

        # ============ Process Spectrogram Branch ============
        denoised_spec = self.clean_spec_net(noisy_spectrogram)
        cond_feature = self.spec_upsampler(denoised_spec)

        # Align lengths if needed
        if cond_feature.shape[-1] != noisy_waveform.shape[-1]:
            cond_feature = F.interpolate(
                cond_feature,
                size=noisy_waveform.shape[-1],
                mode='linear'
            )

        # Apply conditioning
        conditioned_input = self.conditioner(noisy_waveform, cond_feature)

        # ============ Encode to Latent Space ============
        latent, encoder_states = self.clean_unet.encode(conditioned_input)
        # latent shape: (batch, latent_dim, time)

        # ============ STAGE 1: With X-Vectors ============
        if self.stage == 'stage1' and self.xvector_extractor is not None:
            assert clean_audio is not None, "Clean audio is required for Stage 1 training"

            # Extract X-Vectors from clean audio (frozen)
            # NOTE: X-Vector extraction is done in float32 for stability
            with torch.no_grad():
                xvector_emb = self.xvector_extractor.extract_embeddings(
                    clean_audio.squeeze(1)
                )
                # xvector_emb shape: (batch, 512) or (batch, 1, 512)

                # Ensure shape is (batch, 512)
                if xvector_emb.dim() == 3:
                    xvector_emb = xvector_emb.squeeze(1)

            # Expand X-Vectors to match temporal dimension
            time_steps = latent.shape[-1]
            xvector_expanded = xvector_emb.unsqueeze(-1).expand(-1, -1, time_steps)
            # xvector_expanded shape: (batch, 512, time)

            # Match dtype with latent (important for AMP compatibility)
            xvector_expanded = xvector_expanded.to(dtype=latent.dtype)

            # Integrate X-Vectors with latent features
            fused_latent = self.integration_block(latent, xvector_expanded)
            # fused_latent shape: (batch, latent_dim, time)

            # Decode to enhanced audio
            enhanced_waveform = self.clean_unet.decode(fused_latent, encoder_states)

            # Store latents for Stage 2
            latents['fused_latent'] = fused_latent.detach()
            latents['xvector_emb'] = xvector_emb.detach()
            latents['latent'] = latent.detach()

        # ============ STAGE 2: Replicating Latents ============
        elif self.stage == 'stage2' and self.latent_predictor is not None:
            # Predict latent (trying to replicate Stage 1's fused latent)
            predicted_latent = self.latent_predictor(latent)
            # predicted_latent shape: (batch, latent_dim, time)

            # Decode to enhanced audio
            enhanced_waveform = self.clean_unet.decode(predicted_latent, encoder_states)

            latents['predicted_latent'] = predicted_latent
            latents['latent'] = latent.detach()

        # ============ No X-Vectors (Baseline) ============
        else:
            # Standard CleanUNet2 forward (no X-Vectors)
            enhanced_waveform = self.clean_unet.decode(latent, encoder_states)
            latents['latent'] = latent.detach()

        if return_latents:
            return enhanced_waveform, denoised_spec, latents
        else:
            return enhanced_waveform, denoised_spec

    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path):
        """Helper method to load a checkpoint and extract the state_dict."""
        print(f"[CleanUNet2WithXVector] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'generator' in checkpoint:
            return checkpoint['generator']
        else:
            return checkpoint

    def load_stage1_weights(self, checkpoint_path):
        """
        Load Stage 1 weights to initialize Stage 2 model.
        Filters out X-Vector extractor weights (not needed in Stage 2).
        """
        state_dict = self._load_and_extract_state_dict(checkpoint_path)

        # Remove 'model.' prefix if present (from LightningModule)
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # Remove prefix
            if k.startswith('model.'):
                new_key = k[len('model.'):]
            else:
                new_key = k

            # Skip X-Vector extractor weights
            if not new_key.startswith('xvector_extractor'):
                filtered_state_dict[new_key] = v

        # Load with strict=False to allow missing keys (e.g., latent_predictor)
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

        if missing_keys:
            print(f"[INFO] Missing keys (expected for new components): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}")

        print("[CleanUNet2WithXVector] Stage 1 weights loaded successfully")
