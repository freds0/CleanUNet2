"""
CleanUNet2 with Self-Supervised Speech Embeddings (Wav2Vec2) for Two-Stage Training

Architecture:
    - Stage 1: Train with SSL embeddings (Wav2Vec2) injected into latent space
    - Stage 2: Train to replicate latent vectors without embedding extractor

Two-stage training using self-supervised speech embeddings (Wav2Vec2-XLS-R-2B).
The model benefits from speaker information during training while maintaining fast
inference speed (Stage 2 doesn't use embedding extractor).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .cleanunet2 import CleanUNet2, SpecUpsampler, Conditioner
from .cleanunet import CleanUNet
from .cleanspecnet import CleanSpecNet
from .integration_block import SequenceIntegrationBlock
from .wav2vec2_extractor import Wav2Vec2Extractor


class CleanUNet2WithSSLEmbeddings(nn.Module):
    """
    CleanUNet2 model with SSL Embeddings (Wav2Vec2) for two-stage training.

    Stage 1: Uses Wav2Vec2 extractor to inject SSL embeddings into latent space
    Stage 2: Replicates latent vectors without SSL embedding extractor

    This approach allows the model to benefit from speaker information during training
    while maintaining fast inference speed (Stage 2 doesn't use embedding extractor).
    """

    def __init__(
        self,
        stage='stage1',
        conditioning_type='addition',
        cleanunet_params=None,
        cleanspecnet_params=None,
        # Wav2Vec2 SSL Embeddings parameters
        wav2vec2_model='facebook/wav2vec2-xls-r-2b',
        wav2vec2_layer=24,
        wav2vec2_cache_dir=None,
        use_preextracted_embeddings=False,
        wav2vec2_pooling_method='self_attention',
        wav2vec2_attention_heads=8,
        wav2vec2_use_weighted_layers=True
    ):
        """
        Initialize CleanUNet2 with SSL embeddings (Wav2Vec2) integration.

        Args:
            stage (str): Training stage ('stage1' or 'stage2')
            conditioning_type (str): Conditioning method (addition, concatenation, film)
            cleanunet_params (dict): Parameters for CleanUNet
            cleanspecnet_params (dict): Parameters for CleanSpecNet
            wav2vec2_model (str): Wav2Vec2 model name (default: facebook/wav2vec2-xls-r-2b)
            wav2vec2_layer (int): Which layer to extract from Wav2Vec2 (default: 24)
            wav2vec2_cache_dir (str): Directory with pre-extracted wav2vec2 embeddings
            use_preextracted_embeddings (bool): Whether to use pre-extracted embeddings
            wav2vec2_pooling_method (str): Pooling method - 'mean' or 'self_attention' (default: 'self_attention')
            wav2vec2_attention_heads (int): Number of attention heads for self-attention pooling (default: 8)
        """
        super().__init__()

        self.stage = stage
        self.use_preextracted_embeddings = use_preextracted_embeddings
        self.use_weighted_layers = wav2vec2_use_weighted_layers
        self.embedding_type = 'ssl_embeddings'  # Always using SSL embeddings (Wav2Vec2)
        self.embedding_dim = 1920  # Default for wav2vec2-xls-r-2b

        if cleanunet_params is None:
            cleanunet_params = {}
        if cleanspecnet_params is None:
            cleanspecnet_params = {}

        print(f"[CleanUNet2WithSSLEmbeddings] Initializing model...")
        print(f"  - Stage: {stage}")
        print(f"  - Embedding Type: SSL Embeddings (Wav2Vec2)")
        print(f"  - Wav2Vec2 Model: {wav2vec2_model}")
        print(f"  - Wav2Vec2 Layer: {wav2vec2_layer}")
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

        # ============ SSL Embedding Components (Wav2Vec2) ============

        self.embedding_extractor = None
        self.embedding_cache = None

        if stage == 'stage1':
            if use_preextracted_embeddings:
                print("[CleanUNet2WithSSLEmbeddings] Using pre-extracted SSL embeddings...")
                if wav2vec2_cache_dir:
                    from .wav2vec2_cache import Wav2Vec2Cache
                    self.embedding_cache = Wav2Vec2Cache(
                        cache_dir=wav2vec2_cache_dir,
                        enabled=True
                    )
                    # Update embedding_dim from cache metadata or first cached file
                    metadata_file = Path(wav2vec2_cache_dir) / 'metadata.yaml'
                    num_layers = None
                    if metadata_file.exists():
                        import yaml
                        with open(metadata_file, 'r') as f:
                            metadata = yaml.safe_load(f)
                        self.embedding_dim = metadata.get('embedding_dim', 1920)
                        num_layers = metadata.get('num_layers', None)
                    else:
                        # Load first cached file to get embedding dimension
                        cache_files = list(Path(wav2vec2_cache_dir).glob("*.pt"))
                        if cache_files:
                            first_file = torch.load(cache_files[0], map_location='cpu')
                            if isinstance(first_file, dict) and 'embedding' in first_file:
                                first_file = first_file['embedding']
                            if torch.is_tensor(first_file):
                                self.embedding_dim = first_file.shape[-1]
                                if first_file.dim() == 3:  # (num_layers, time, dim) -> all-layer cache
                                    num_layers = first_file.shape[0]
                                print(f"[CleanUNet2WithSSLEmbeddings] Detected embedding_dim={self.embedding_dim} from cache")

                    # Learnable softmax over the cached all-layer stacks (mirrors the
                    # on-the-fly weighted-sum, but reuses the disk cache for speed).
                    if wav2vec2_use_weighted_layers:
                        if num_layers is None:
                            raise ValueError(
                                "wav2vec2_use_weighted_layers=True with pre-extracted embeddings requires an "
                                "all-layer cache of shape (num_layers, time, dim). Re-run "
                                "extract_wav2vec2_embeddings.py to regenerate the cache."
                            )
                        self.cached_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
                        print(f"[CleanUNet2WithSSLEmbeddings] Cached weighted-sum over {num_layers} layers "
                              f"enabled (learnable softmax)")
                else:
                    raise ValueError("wav2vec2_cache_dir must be specified when use_preextracted_embeddings=True")
            else:
                # Extract Wav2Vec2 on-the-fly (slower)
                print("[CleanUNet2WithSSLEmbeddings] Loading Wav2Vec2 extractor...")
                self.embedding_extractor = Wav2Vec2Extractor(
                    model_name=wav2vec2_model,
                    device='cpu',
                    layer=wav2vec2_layer,
                    pooling_method=wav2vec2_pooling_method,
                    num_attention_heads=wav2vec2_attention_heads,
                    use_weighted_layers=wav2vec2_use_weighted_layers
                )
                self.embedding_dim = self.embedding_extractor.get_embedding_dim()

                # Freeze Wav2Vec2 model (keep attention pooling trainable)
                for param in self.embedding_extractor.model.parameters():
                    param.requires_grad = False
                self.embedding_extractor.model.eval()

                # The all-layer softmax weights are NOT part of the frozen backbone -> trainable.
                if wav2vec2_use_weighted_layers:
                    print("[CleanUNet2WithSSLEmbeddings] Weighted-sum over ALL Wav2Vec2 layers "
                          "enabled — layer weights will be trained!")
                    self.embedding_extractor.layer_weights.requires_grad = True

                if wav2vec2_pooling_method == 'self_attention':
                    print("[CleanUNet2WithSSLEmbeddings] Self-Attention Pooling will be trained!")
                    for param in self.embedding_extractor.attention_pooling.parameters():
                        param.requires_grad = True
                    self.embedding_extractor.attention_pooling.train()
                else:
                    self.embedding_extractor.eval()

        # Integration Block (for fusing embeddings with latent features)
        # Always use SequenceIntegrationBlock — embeddings are never pooled
        print(f"[CleanUNet2WithSSLEmbeddings] Creating SequenceIntegrationBlock (embedding_dim={self.embedding_dim})...")
        self.integration_block = SequenceIntegrationBlock(
            latent_channels=self.latent_dim,
            embedding_dim=self.embedding_dim,
            num_heads=wav2vec2_attention_heads,
            dropout=0.1
        )

        # Latent Predictor (Stage 2 only)
        if stage == 'stage2':
            print("[CleanUNet2WithSSLEmbeddings] Creating latent predictor for Stage 2...")
            self.latent_predictor = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1),
                nn.PReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1)
            )
        else:
            self.latent_predictor = None

        print("[CleanUNet2WithSSLEmbeddings] Model initialized successfully!")

    def load_vanilla_checkpoint(self, checkpoint_path):
        """
        Load weights from vanilla CleanUNet2 checkpoint.
        This allows warm-starting Stage-1 training with pre-trained vanilla weights.

        Args:
            checkpoint_path (str): Path to vanilla CleanUNet2 checkpoint
        """
        print(f"\n[CleanUNet2WithSSLEmbeddings] Loading vanilla checkpoint: {checkpoint_path}")

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
        print(f"[CleanUNet2WithSSLEmbeddings] Loaded {len(filtered_state_dict)} parameters from vanilla checkpoint")

        if missing_keys:
            # Filter out expected missing keys (X-Vector related)
            expected_missing = [k for k in missing_keys if any(
                k.startswith(prefix) for prefix in ['xvector_extractor', 'integration_block', 'latent_predictor']
            )]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]

            if expected_missing:
                print(f"[CleanUNet2WithSSLEmbeddings] Expected missing keys (new components): {len(expected_missing)}")
                if len(expected_missing) <= 5:
                    for key in expected_missing:
                        print(f"  - {key}")

            if unexpected_missing:
                print(f"[WARNING] Unexpected missing keys: {unexpected_missing[:5]}")

        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys[:5]}")

        print("[CleanUNet2WithSSLEmbeddings] Vanilla checkpoint loaded successfully!\n")

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
            batch_size = clean_audio.shape[0]

            if self.use_preextracted_embeddings:
                # Load pre-extracted Wav2Vec2 embeddings from cache.
                # Each cached item is either:
                #   (num_layers, time, dim)  -> all-layer cache, or
                #   (time, dim)              -> legacy single-layer cache.
                assert clean_audio_paths is not None, "Audio paths required for pre-extracted embeddings"
                with torch.no_grad():
                    raw_list = []
                    for i in range(batch_size):
                        audio_path = clean_audio_paths[i]
                        cached_data = self.embedding_cache.get(audio_path, device=clean_audio.device)

                        if cached_data is None:
                            raise RuntimeError(
                                f"Pre-extracted embedding not found for: {audio_path}\n"
                                f"Please run extract_wav2vec2_embeddings.py first!"
                            )

                        if isinstance(cached_data, dict) and 'embedding' in cached_data:
                            cached_data = cached_data['embedding']

                        raw_list.append(cached_data)

                # Combine layers per item. For all-layer caches the softmax is computed
                # OUTSIDE no_grad, so cached_layer_weights are learned during training.
                embedding_list = []
                for emb in raw_list:
                    if emb.dim() == 3:  # (num_layers, time, dim)
                        if self.use_weighted_layers:
                            weights = F.softmax(self.cached_layer_weights, dim=0)
                            emb = torch.einsum('l,ltd->td', weights, emb)
                        else:
                            emb = emb.mean(dim=0)
                    embedding_list.append(emb)  # (time, dim)

                # Pad variable-length sequences to same length before stacking
                max_len = max(emb.shape[0] for emb in embedding_list)
                padded_embeddings = []
                for emb in embedding_list:
                    if emb.shape[0] < max_len:
                        padding = (0, 0, 0, max_len - emb.shape[0])
                        emb = F.pad(emb, padding, mode='constant', value=0)
                    padded_embeddings.append(emb)
                embedding = torch.stack(padded_embeddings, dim=0)
                # embedding shape: (batch, seq_len, embedding_dim)

            else:
                # Extract Wav2Vec2 on-the-fly. The frozen backbone runs under no_grad inside
                # the extractor; the softmax over ALL layers stays differentiable.
                embedding = self.embedding_extractor.extract_embeddings(
                    clean_audio.squeeze(1),
                    sample_rate=16000,
                    return_mean=False
                )
                    # embedding shape: (batch, seq_len, embedding_dim)

            # Match dtype with latent (important for AMP compatibility)
            embedding = embedding.to(dtype=latent.dtype)

            # Integrate sequence embeddings with latent features
            # embedding shape: (batch, seq_len, embedding_dim)
            fused_latent = self.integration_block(latent, embedding)

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
        print(f"[CleanUNet2WithSSLEmbeddings] Loading checkpoint from: {checkpoint_path}")
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
        Filters out embedding extractor weights (not needed in Stage 2).
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

            # Skip embedding extractor/cache weights (not needed in Stage 2)
            # This includes: xvector_extractor, embedding_extractor, embedding_cache
            if not new_key.startswith(('xvector_extractor', 'embedding_extractor', 'embedding_cache')):
                filtered_state_dict[new_key] = v

        # Load with strict=False to allow missing keys (e.g., latent_predictor)
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

        if missing_keys:
            print(f"[INFO] Missing keys (expected for new components): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}")

        print("[CleanUNet2WithSSLEmbeddings] Stage 1 weights loaded successfully")
