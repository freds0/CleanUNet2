"""
CleanUNet2 with Self-Supervised Speech Embeddings for Two-Stage Training
Based on CNUNet-TB paper: Two-stage training using self-supervised speech embeddings

Supports two types of embeddings:
    1. X-Vectors (SpeechBrain) - 512 dimensions
    2. Wav2Vec2 (facebook/wav2vec2-xls-r-300m) - 1024 dimensions

Architecture:
    - Stage 1: Train with embeddings injected into latent space
    - Stage 2: Train to replicate latent vectors without embedding extractor

Implementation inspired by CNUNet-TB but adapted for CleanUNet2 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .cleanunet2 import CleanUNet2, SpecUpsampler, Conditioner
from .cleanunet import CleanUNet
from .cleanspecnet import CleanSpecNet
from .integration_block import IntegrationBlock
from .xvector_extractor import XVectorExtractor
from .wav2vec2_extractor import Wav2Vec2Extractor


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
        xvector_local_path=None,
        xvector_cache_dir=None,
        xvector_cache_enabled=False,
        # Wav2Vec2 parameters
        use_wav2vec2=False,
        wav2vec2_model='facebook/wav2vec2-xls-r-300m',
        wav2vec2_cache_dir=None,
        use_preextracted_embeddings=False,
        wav2vec2_pooling_method='self_attention',
        wav2vec2_attention_heads=8
    ):
        """
        Initialize CleanUNet2 with self-supervised speech embedding integration.

        Args:
            stage (str): Training stage ('stage1' or 'stage2')
            use_xvector (bool): Whether to use X-Vectors
            xvector_dim (int): Dimension of X-Vector embeddings (default: 512)
            conditioning_type (str): Conditioning method (addition, concatenation, film)
            cleanunet_params (dict): Parameters for CleanUNet
            cleanspecnet_params (dict): Parameters for CleanSpecNet
            xvector_local_path (str): Local path to x-vector model weights
            xvector_cache_dir (str): Directory to store cached x-vectors
            xvector_cache_enabled (bool): Whether to use x-vector caching
            use_wav2vec2 (bool): Whether to use Wav2Vec2 embeddings instead of x-vectors
            wav2vec2_model (str): Wav2Vec2 model name (e.g., facebook/wav2vec2-xls-r-300m)
            wav2vec2_cache_dir (str): Directory with pre-extracted wav2vec2 embeddings
            use_preextracted_embeddings (bool): Whether to use pre-extracted embeddings
            wav2vec2_pooling_method (str): Pooling method - 'mean' or 'self_attention' (default: 'self_attention')
            wav2vec2_attention_heads (int): Number of attention heads for self-attention pooling (default: 8)
        """
        super().__init__()

        self.stage = stage
        self.use_xvector = use_xvector
        self.use_wav2vec2 = use_wav2vec2
        self.use_preextracted_embeddings = use_preextracted_embeddings
        self.xvector_cache_enabled = xvector_cache_enabled

        # Determine embedding type and dimension
        if use_wav2vec2 and use_xvector:
            raise ValueError("Cannot use both X-Vectors and Wav2Vec2 simultaneously. Choose one.")

        if use_wav2vec2:
            # Wav2Vec2 embeddings (1024 dim for wav2vec2-xls-r-300m)
            self.embedding_type = 'wav2vec2'
            # We'll get actual dim from extractor or use default
            self.embedding_dim = 1024  # Default for wav2vec2-xls-r-300m
        elif use_xvector:
            # X-Vector embeddings (512 dim)
            self.embedding_type = 'xvector'
            self.embedding_dim = xvector_dim
        else:
            # No embeddings
            self.embedding_type = None
            self.embedding_dim = 0

        if cleanunet_params is None:
            cleanunet_params = {}
        if cleanspecnet_params is None:
            cleanspecnet_params = {}

        print(f"[CleanUNet2WithXVector] Initializing model...")
        print(f"  - Stage: {stage}")
        print(f"  - Embedding Type: {self.embedding_type}")
        if self.embedding_type:
            print(f"  - Embedding Dim: {self.embedding_dim}")
            if use_wav2vec2:
                print(f"  - Wav2Vec2 Model: {wav2vec2_model}")
                print(f"  - Use Pre-extracted: {use_preextracted_embeddings}")
                if not use_preextracted_embeddings:
                    print(f"  - Pooling Method: {wav2vec2_pooling_method}")
                    if wav2vec2_pooling_method == 'self_attention':
                        print(f"  - Attention Heads: {wav2vec2_attention_heads}")
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

        # ============ Embedding Components ============

        # Initialize embedding extractor/cache based on type
        self.embedding_extractor = None
        self.embedding_cache = None

        if stage == 'stage1' and self.embedding_type:
            if use_wav2vec2:
                # Wav2Vec2 Embeddings
                if use_preextracted_embeddings:
                    # Use pre-extracted embeddings from disk
                    print("[CleanUNet2WithXVector] Using pre-extracted Wav2Vec2 embeddings...")
                    if wav2vec2_cache_dir:
                        from .wav2vec2_cache import Wav2Vec2Cache
                        self.embedding_cache = Wav2Vec2Cache(
                            cache_dir=wav2vec2_cache_dir,
                            enabled=True
                        )
                        # Update embedding_dim from cache metadata
                        metadata_file = Path(wav2vec2_cache_dir) / 'metadata.yaml'
                        if metadata_file.exists():
                            import yaml
                            with open(metadata_file, 'r') as f:
                                metadata = yaml.safe_load(f)
                            self.embedding_dim = metadata.get('embedding_dim', 1024)
                    else:
                        raise ValueError("wav2vec2_cache_dir must be specified when use_preextracted_embeddings=True")
                else:
                    # Extract wav2vec2 on-the-fly (slower)
                    print("[CleanUNet2WithXVector] Loading Wav2Vec2 extractor...")
                    self.embedding_extractor = Wav2Vec2Extractor(
                        model_name=wav2vec2_model,
                        device='cpu',  # Will be moved to correct device by Lightning
                        layer=-1,
                        pooling_method=wav2vec2_pooling_method,
                        num_attention_heads=wav2vec2_attention_heads
                    )
                    self.embedding_dim = self.embedding_extractor.get_embedding_dim()

                    # Freeze Wav2Vec2 model (but NOT attention pooling if trainable)
                    for param in self.embedding_extractor.model.parameters():
                        param.requires_grad = False
                    self.embedding_extractor.model.eval()

                    # Keep attention pooling trainable if using self-attention
                    if wav2vec2_pooling_method == 'self_attention':
                        print("[CleanUNet2WithXVector] Self-Attention Pooling will be trained!")
                        for param in self.embedding_extractor.attention_pooling.parameters():
                            param.requires_grad = True
                        self.embedding_extractor.attention_pooling.train()
                    else:
                        self.embedding_extractor.eval()

            elif use_xvector:
                # X-Vector Embeddings
                print("[CleanUNet2WithXVector] Loading X-Vector extractor...")
                self.embedding_extractor = XVectorExtractor(
                    device='cpu',  # Will be moved to correct device by Lightning
                    local_path=xvector_local_path
                )
                # Freeze X-Vector extractor
                for param in self.embedding_extractor.parameters():
                    param.requires_grad = False
                self.embedding_extractor.eval()

                # X-Vector Cache (on-the-fly extraction + caching)
                if xvector_cache_enabled and xvector_cache_dir:
                    from .xvector_cache import XVectorCache
                    self.embedding_cache = XVectorCache(
                        cache_dir=xvector_cache_dir,
                        enabled=True
                    )
        else:
            print("[CleanUNet2WithXVector] No embedding extractor loaded")

        # Integration Block (for fusing embeddings with latent features)
        if self.embedding_type:
            print(f"[CleanUNet2WithXVector] Creating integration block (embedding_dim={self.embedding_dim})...")
            self.integration_block = IntegrationBlock(
                latent_channels=self.latent_dim,
                xvector_dim=self.embedding_dim  # This now works for any embedding dim
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

    def forward(self, noisy_waveform, noisy_spectrogram, clean_audio=None, clean_audio_paths=None, return_latents=False):
        """
        Forward pass through the model.

        Args:
            noisy_waveform (torch.Tensor): Noisy input waveform
                                          Shape: (batch, 1, samples)
            noisy_spectrogram (torch.Tensor): Noisy input spectrogram
                                              Shape: (batch, freq, time)
            clean_audio (torch.Tensor): Clean reference audio (Stage 1 only)
                                        Shape: (batch, 1, samples)
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

        # ============ STAGE 1: With Embeddings ============
        if self.stage == 'stage1' and self.embedding_type is not None:
            assert clean_audio is not None, "Clean audio is required for Stage 1 training"

            # Extract embeddings from clean audio
            with torch.no_grad():
                batch_size = clean_audio.shape[0]

                if self.use_wav2vec2 and self.use_preextracted_embeddings:
                    # Load pre-extracted Wav2Vec2 embeddings from cache
                    assert clean_audio_paths is not None, "Audio paths required for pre-extracted embeddings"
                    embedding_list = []

                    for i in range(batch_size):
                        audio_path = clean_audio_paths[i]
                        cached_embedding = self.embedding_cache.get(audio_path, device=clean_audio.device)

                        if cached_embedding is None:
                            raise RuntimeError(
                                f"Pre-extracted embedding not found for: {audio_path}\n"
                                f"Please run extract_wav2vec2_embeddings.py first!"
                            )

                        embedding_list.append(cached_embedding)

                    # Stack all embeddings
                    embedding = torch.stack(embedding_list, dim=0)
                    # embedding shape: (batch, embedding_dim)

                elif self.use_wav2vec2 and not self.use_preextracted_embeddings:
                    # Extract Wav2Vec2 on-the-fly (slower)
                    embedding = self.embedding_extractor.extract_embeddings(
                        clean_audio.squeeze(1),
                        sample_rate=16000,  # Wav2Vec2 expects 16kHz
                        return_mean=True
                    )
                    # embedding shape: (batch, embedding_dim)

                elif self.use_xvector:
                    # X-Vector extraction (original logic)
                    embedding_list = []

                    # Try to use cache if enabled and paths are provided
                    if self.embedding_cache is not None and clean_audio_paths is not None:
                        for i in range(batch_size):
                            audio_path = clean_audio_paths[i]
                            cached_emb = self.embedding_cache.get(audio_path, device=clean_audio.device)

                            if cached_emb is not None:
                                embedding_list.append(cached_emb)
                            else:
                                # Extract and cache
                                emb_single = self.embedding_extractor.extract_embeddings(
                                    clean_audio[i:i+1].squeeze(1)
                                )
                                if emb_single.dim() == 3:
                                    emb_single = emb_single.squeeze(1)
                                embedding_list.append(emb_single.squeeze(0))
                                # Save to cache
                                self.embedding_cache.set(audio_path, emb_single.squeeze(0))

                        embedding = torch.stack(embedding_list, dim=0)
                    else:
                        # No cache, extract normally
                        embedding = self.embedding_extractor.extract_embeddings(
                            clean_audio.squeeze(1)
                        )
                        if embedding.dim() == 3:
                            embedding = embedding.squeeze(1)

                    # embedding shape: (batch, embedding_dim)

            # Expand embeddings to match temporal dimension
            time_steps = latent.shape[-1]
            embedding_expanded = embedding.unsqueeze(-1).expand(-1, -1, time_steps)
            # embedding_expanded shape: (batch, embedding_dim, time)

            # Match dtype with latent (important for AMP compatibility)
            embedding_expanded = embedding_expanded.to(dtype=latent.dtype)

            # Integrate embeddings with latent features
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
