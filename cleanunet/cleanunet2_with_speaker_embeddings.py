"""
CleanUNet2 with Speaker Embeddings for Two-Stage Training

Architecture:
    - Stage 1: Train with speaker embeddings injected into latent space
    - Stage 2: Train to replicate latent vectors without embedding extractor

Two-stage training using speaker embeddings (SpeechBrain). Supports both
X-Vector (512-dim) and ECAPA-TDNN (192-dim) models via the 'speaker_model' config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .cleanunet2 import CleanUNet2, SpecUpsampler, Conditioner
from .cleanunet import CleanUNet
from .cleanspecnet import CleanSpecNet
from .integration_block import IntegrationBlock
from .speaker_extractor import SpeakerExtractor, SPEAKER_MODELS


class CleanUNet2WithSpeakerEmbeddings(nn.Module):
    """
    CleanUNet2 model with Speaker Embeddings for two-stage training.

    Stage 1: Uses speaker embedding extractor to inject embeddings into latent space
    Stage 2: Replicates latent vectors without the speaker embedding extractor

    Supports X-Vector (512-dim) and ECAPA-TDNN (192-dim) via speaker_model config.
    """

    def __init__(
        self,
        stage='stage1',
        conditioning_type='addition',
        cleanunet_params=None,
        cleanspecnet_params=None,
        speaker_model='xvector',
        speaker_model_local_path=None,
        embedding_cache_dir=None,
        use_preextracted_embeddings=False,
    ):
        """
        Initialize CleanUNet2 with speaker embeddings integration.

        Args:
            stage (str): Training stage ('stage1' or 'stage2')
            conditioning_type (str): Conditioning method (addition, concatenation, film)
            cleanunet_params (dict): Parameters for CleanUNet
            cleanspecnet_params (dict): Parameters for CleanSpecNet
            speaker_model (str): Speaker model to use ('xvector' or 'ecapa')
            speaker_model_local_path (str): Local path to pre-downloaded model directory
            embedding_cache_dir (str): Directory with pre-extracted embeddings
            use_preextracted_embeddings (bool): Whether to use pre-extracted embeddings
        """
        super().__init__()

        self.stage = stage
        self.use_preextracted_embeddings = use_preextracted_embeddings
        self.speaker_model_name = speaker_model
        self.embedding_type = 'speaker_embeddings'

        if speaker_model not in SPEAKER_MODELS:
            raise ValueError(
                f"Unknown speaker_model: '{speaker_model}'. "
                f"Supported: {list(SPEAKER_MODELS.keys())}"
            )
        self.embedding_dim = SPEAKER_MODELS[speaker_model]['embedding_dim']

        if cleanunet_params is None:
            cleanunet_params = {}
        if cleanspecnet_params is None:
            cleanspecnet_params = {}

        display_name = SPEAKER_MODELS[speaker_model]['display_name']
        print(f"[CleanUNet2WithSpeakerEmbeddings] Initializing model...")
        print(f"  - Stage: {stage}")
        print(f"  - Speaker Model: {display_name} ({speaker_model})")
        print(f"  - Embedding Dim: {self.embedding_dim}")
        print(f"  - Use Pre-extracted: {use_preextracted_embeddings}")
        print(f"  - Conditioning: {conditioning_type}")

        # Calculate latent dimension from CleanUNet params
        # The latent is the encoder output channels, not tsfm_d_model
        # Encoder grows: channels_H -> channels_H*2 -> ... -> min(channels_H*2^n, max_H)
        channels_H = cleanunet_params.get('channels_H', 64)
        max_H = cleanunet_params.get('max_H', 768)
        encoder_n_layers = cleanunet_params.get('encoder_n_layers', 8)

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

        # ============ Speaker Embedding Components ============

        self.embedding_extractor = None
        self.embedding_cache = None

        if stage == 'stage1':
            if use_preextracted_embeddings:
                print("[CleanUNet2WithSpeakerEmbeddings] Using pre-extracted speaker embeddings...")
                if embedding_cache_dir:
                    from .xvector_cache import XVectorCache
                    self.embedding_cache = XVectorCache(
                        cache_dir=embedding_cache_dir,
                        enabled=True
                    )
                else:
                    raise ValueError("embedding_cache_dir must be specified when use_preextracted_embeddings=True")
            else:
                print(f"[CleanUNet2WithSpeakerEmbeddings] Loading {display_name} extractor...")
                self.embedding_extractor = SpeakerExtractor(
                    model_name=speaker_model,
                    device='cpu',
                    local_path=speaker_model_local_path
                )
                for param in self.embedding_extractor.parameters():
                    param.requires_grad = False
                self.embedding_extractor.eval()

        # Integration Block (for fusing pooled speaker embeddings with latent features)
        print(f"[CleanUNet2WithSpeakerEmbeddings] Creating IntegrationBlock (embedding_dim={self.embedding_dim})...")
        self.integration_block = IntegrationBlock(
            latent_channels=self.latent_dim,
            xvector_dim=self.embedding_dim
        )

        # Latent Predictor (Stage 2 only)
        if stage == 'stage2':
            print("[CleanUNet2WithSpeakerEmbeddings] Creating latent predictor for Stage 2...")
            self.latent_predictor = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1),
                nn.PReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1)
            )
        else:
            self.latent_predictor = None

        print("[CleanUNet2WithSpeakerEmbeddings] Model initialized successfully!")

    def load_vanilla_checkpoint(self, checkpoint_path):
        """
        Load weights from vanilla CleanUNet2 checkpoint.
        This allows warm-starting Stage-1 training with pre-trained vanilla weights.

        Args:
            checkpoint_path (str): Path to vanilla CleanUNet2 checkpoint
        """
        print(f"\n[CleanUNet2WithSpeakerEmbeddings] Loading vanilla checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint

        filtered_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len('model.'):] if k.startswith('model.') else k
            # Only load compatible components (skip embedding/integration/predictor components)
            if not new_key.startswith(('xvector_extractor', 'embedding_extractor', 'integration_block', 'latent_predictor')):
                filtered_state_dict[new_key] = v

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"[CleanUNet2WithSpeakerEmbeddings] Loaded {len(filtered_state_dict)} parameters from vanilla checkpoint")

        if missing_keys:
            expected_missing = [k for k in missing_keys if any(
                k.startswith(prefix) for prefix in ['xvector_extractor', 'embedding_extractor', 'integration_block', 'latent_predictor']
            )]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]
            if expected_missing:
                print(f"[CleanUNet2WithSpeakerEmbeddings] Expected missing keys (new components): {len(expected_missing)}")
            if unexpected_missing:
                print(f"[WARNING] Unexpected missing keys: {unexpected_missing[:5]}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys[:5]}")

        print("[CleanUNet2WithSpeakerEmbeddings] Vanilla checkpoint loaded successfully!\n")

    def forward(self, noisy_waveform, noisy_spectrogram, clean_audio=None, clean_audio_paths=None, return_latents=False):
        """
        Forward pass through the model.

        Args:
            noisy_waveform (torch.Tensor): Noisy input waveform. Shape: (batch, 1, samples)
            noisy_spectrogram (torch.Tensor): Noisy input spectrogram. Shape: (batch, freq, time)
            clean_audio (torch.Tensor): Clean reference audio (Stage 1 only). Shape: (batch, 1, samples)
            clean_audio_paths (list): List of file paths for clean audio (for caching)
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

        # ============ STAGE 1: With Speaker Embeddings ============
        if self.stage == 'stage1' and self.embedding_type is not None:
            assert clean_audio is not None, "Clean audio is required for Stage 1 training"

            # Extract speaker embeddings from clean audio
            # NOTE: X-Vector extraction is done in float32 for stability
            with torch.no_grad():
                batch_size = clean_audio.shape[0]

                if self.use_preextracted_embeddings:
                    # Load pre-extracted X-Vector embeddings from cache
                    assert clean_audio_paths is not None, "Audio paths required for pre-extracted embeddings"
                    embedding_list = []

                    for i in range(batch_size):
                        audio_path = clean_audio_paths[i]
                        cached_data = self.embedding_cache.get(audio_path, device=clean_audio.device)

                        if cached_data is None:
                            raise RuntimeError(
                                f"Pre-extracted embedding not found for: {audio_path}\n"
                                f"Please extract x-vectors first!"
                            )

                        if isinstance(cached_data, dict) and 'embedding' in cached_data:
                            cached_embedding = cached_data['embedding']
                        else:
                            cached_embedding = cached_data

                        embedding_list.append(cached_embedding)

                    embedding = torch.stack(embedding_list, dim=0)
                    # embedding shape: (batch, 512)

                else:
                    # Extract X-Vectors on-the-fly (pooled, utterance-level)
                    embedding = self.embedding_extractor.extract_embeddings(
                        clean_audio.squeeze(1),
                        sample_rate=16000
                    )
                    # embedding shape: (batch, 512) or (batch, 1, 512)
                    if embedding.dim() == 3:
                        embedding = embedding.squeeze(1)

            # Expand pooled embeddings to match temporal dimension
            # embedding: (batch, 512) -> (batch, 512, time)
            time_steps = latent.shape[-1]
            embedding_expanded = embedding.unsqueeze(-1).expand(-1, -1, time_steps)

            # Match dtype with latent (important for AMP compatibility)
            embedding_expanded = embedding_expanded.to(dtype=latent.dtype)

            # Integrate speaker embeddings with latent features
            fused_latent = self.integration_block(latent, embedding_expanded)
            # fused_latent shape: (batch, latent_dim, time)

            # Decode to enhanced audio
            enhanced_waveform = self.clean_unet.decode(fused_latent, encoder_states)

            # Store latents for Stage 2
            latents['fused_latent'] = fused_latent.detach()
            latents['embedding'] = embedding.detach()
            latents['latent'] = latent.detach()

        # ============ STAGE 2: Replicating Latents ============
        elif self.stage == 'stage2' and self.latent_predictor is not None:
            predicted_latent = self.latent_predictor(latent)
            enhanced_waveform = self.clean_unet.decode(predicted_latent, encoder_states)

            latents['predicted_latent'] = predicted_latent
            latents['latent'] = latent.detach()

        # ============ Baseline (no embeddings) ============
        else:
            enhanced_waveform = self.clean_unet.decode(latent, encoder_states)
            latents['latent'] = latent.detach()

        if return_latents:
            return enhanced_waveform, denoised_spec, latents
        else:
            return enhanced_waveform, denoised_spec

    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path):
        """Helper method to load a checkpoint and extract the state_dict."""
        print(f"[CleanUNet2WithSpeakerEmbeddings] Loading checkpoint from: {checkpoint_path}")
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
        Filters out embedding extractor/cache weights (not needed in Stage 2).
        """
        state_dict = self._load_and_extract_state_dict(checkpoint_path)

        filtered_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len('model.'):] if k.startswith('model.') else k
            if not new_key.startswith(('xvector_extractor', 'embedding_extractor', 'embedding_cache')):
                filtered_state_dict[new_key] = v

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"[INFO] Missing keys (expected for new components): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}")
        print("[CleanUNet2WithSpeakerEmbeddings] Stage 1 weights loaded successfully")
