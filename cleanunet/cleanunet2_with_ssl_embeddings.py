"""
CleanUNet2 with Self-Supervised Speech Embeddings (multi-backbone) for Two-Stage Training

Architecture:
    - Stage 1: Train with SSL embeddings injected into latent space
    - Stage 2: Train to replicate latent vectors without embedding extractor

Two-stage training using self-supervised speech embeddings (wav2vec2 / hubert / wavlm / w2v-bert / whisper).
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
from .integration_block import SequenceIntegrationBlock, HierarchicalMultiScaleBlock
from .latent_predictors import build_latent_predictor, SkipFiLMHead
from .ssl_extractor_factory import build_ssl_extractor


class CleanUNet2WithSSLEmbeddings(nn.Module):
    """
    CleanUNet2 model with SSL Embeddings (multi-backbone) for two-stage training.

    Stage 1: Uses the SSL extractor to inject embeddings into latent space
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
        # SSL embedding parameters (backbone family selected by ssl_type)
        ssl_type='wav2vec2',
        ssl_model='facebook/wav2vec2-xls-r-2b',
        ssl_layer=24,
        ssl_cache_dir=None,
        use_preextracted_embeddings=False,
        ssl_pooling_method='mean',
        ssl_attention_heads=8,
        ssl_use_weighted_layers=True,
        # Layer-fusion strategy: list of indices ('+') or the string 'all' ('++').
        ssl_selected_layers=None,
        ssl_num_selected_layers=3,
        # Backbone embedding dim. Authoritative for Stage 2 (no extractor/cache to infer
        # it from); in Stage 1 it is overwritten by the extractor/cache.
        ssl_embedding_dim=None,
        # SSL conditioning input source: 'clean' (Stage-1 oracle, default) or 'noisy'
        # (this fork's Stage-2 -- extract SSL from the noisy input, available at inference).
        ssl_audio_source='clean',
        # Fusion strategy: how the SSL features modulate the CleanUNet.
        #   'hierarchical_multiscale' -> HierarchicalMultiScaleBlock (default)
        #   'legacy_pooling'          -> SequenceIntegrationBlock     (pooled fusion)
        fusion_type='hierarchical_multiscale',
        acoustic_layers=None,
        semantic_layers=None,
        # Stage-2 latent predictor architecture (see cleanunet/latent_predictors.py):
        #   'baseline' | 'tcn' | 'unet'
        latent_predictor_type='baseline',
        latent_predictor_params=None,
        # Stage-2 skip-connection distillation: also reproduce the Stage-1 early-layer
        # (acoustic) FiLM modulation on the encoder skips, not just the bottleneck.
        distill_skips=False,
    ):
        """
        Initialize CleanUNet2 with SSL embeddings integration.

        Args:
            stage (str): Training stage ('stage1' or 'stage2')
            conditioning_type (str): Conditioning method (addition, concatenation, film)
            cleanunet_params (dict): Parameters for CleanUNet
            cleanspecnet_params (dict): Parameters for CleanSpecNet
            ssl_type (str): SSL family (wav2vec2|hubert|wavlm|w2v-bert|whisper)
            ssl_model (str): HuggingFace model id for that family
            ssl_cache_dir (str): Directory with pre-extracted SSL embeddings
            use_preextracted_embeddings (bool): Whether to use pre-extracted embeddings
            ssl_pooling_method (str): Pooling method - 'mean' or 'self_attention' (default: 'self_attention')
            ssl_attention_heads (int): Number of attention heads for self-attention pooling (default: 8)
        """
        super().__init__()

        self.stage = stage
        if ssl_audio_source not in ('clean', 'noisy'):
            raise ValueError(
                f"ssl_audio_source must be 'clean' or 'noisy', got '{ssl_audio_source}'."
            )
        self.ssl_audio_source = ssl_audio_source
        # SSL-conditioned forward (extractor + hierarchical FiLM fusion) runs for
        # Stage-1 (clean SSL) and for this fork's Stage-2 when SSL comes from the noisy
        # input. Plain Stage-2 distillation keeps ssl_audio_source='clean'.
        self._ssl_conditioned = (stage == 'stage1') or (
            stage == 'stage2' and ssl_audio_source == 'noisy'
        )
        self.use_preextracted_embeddings = use_preextracted_embeddings
        self.use_weighted_layers = ssl_use_weighted_layers
        self.embedding_type = 'ssl_embeddings'  # Always using SSL embeddings
        # Stage 2 uses this config-provided dim (no extractor/cache to infer from); it
        # must match what Stage 1 trained so the integration_block shapes line up with
        # the checkpoint. Stage 1 overwrites it from the extractor/cache below.
        self.embedding_dim = ssl_embedding_dim if ssl_embedding_dim is not None else 1024

        # Fusion strategy selector (see fusion block construction below).
        self.fusion_type = fusion_type
        self.acoustic_layers = acoustic_layers
        self.semantic_layers = semantic_layers

        # Stage-2 latent predictor selector (built below, Stage 2 only).
        self.latent_predictor_type = latent_predictor_type
        self.latent_predictor_params = latent_predictor_params
        self.distill_skips = distill_skips

        if cleanunet_params is None:
            cleanunet_params = {}
        if cleanspecnet_params is None:
            cleanspecnet_params = {}

        strategy = 'all_layers (++)' if ssl_selected_layers == 'all' else \
                   (f'selected {ssl_selected_layers} (+)' if ssl_selected_layers else
                    f'{ssl_num_selected_layers} evenly-spaced (+)')
        print(f"[CleanUNet2WithSSLEmbeddings] Initializing model...")
        print(f"  - Stage: {stage}")
        print(f"  - SSL Type: {ssl_type}")
        print(f"  - SSL Model: {ssl_model}")
        print(f"  - Layer fusion: {strategy}")
        print(f"  - Use Pre-extracted: {use_preextracted_embeddings}")
        if not use_preextracted_embeddings:
            print(f"  - Pooling Method: {ssl_pooling_method}")
            if ssl_pooling_method == 'self_attention':
                print(f"  - Attention Heads: {ssl_attention_heads}")
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

        # ============ SSL Embedding Components ============

        self.embedding_extractor = None
        self.embedding_cache = None

        if self._ssl_conditioned:
            if use_preextracted_embeddings:
                print("[CleanUNet2WithSSLEmbeddings] Using pre-extracted SSL embeddings...")
                if ssl_cache_dir:
                    from .ssl_cache import SSLEmbeddingCache
                    self.embedding_cache = SSLEmbeddingCache(
                        cache_dir=ssl_cache_dir,
                        enabled=True
                    )
                    # Update embedding_dim from cache metadata or first cached file
                    metadata_file = Path(ssl_cache_dir) / 'metadata.yaml'
                    num_layers = None
                    if metadata_file.exists():
                        import yaml
                        with open(metadata_file, 'r') as f:
                            metadata = yaml.safe_load(f)
                        self.embedding_dim = metadata.get('embedding_dim', 1920)
                        num_layers = metadata.get('num_layers', None)
                    else:
                        # Load first cached file to get embedding dimension
                        cache_files = list(Path(ssl_cache_dir).glob("*.pt"))
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
                    if ssl_use_weighted_layers:
                        if num_layers is None:
                            raise ValueError(
                                "ssl_use_weighted_layers=True with pre-extracted embeddings requires an "
                                "all-layer cache of shape (num_layers, time, dim). Re-run "
                                "extract_ssl_embeddings.py to regenerate the cache."
                            )
                        self.cached_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
                        print(f"[CleanUNet2WithSSLEmbeddings] Cached weighted-sum over {num_layers} layers "
                              f"enabled (learnable softmax)")
                else:
                    raise ValueError("ssl_cache_dir must be specified when use_preextracted_embeddings=True")
            else:
                # Extract SSL embeddings on-the-fly (slower). Required when the layer
                # weights are learnable (both '+' and '++'), since the softmax must
                # receive gradients each step.
                print(f"[CleanUNet2WithSSLEmbeddings] Loading {ssl_type} extractor...")
                self.embedding_extractor = build_ssl_extractor(
                    ssl_type=ssl_type,
                    model_name=ssl_model,
                    device='cpu',
                    layer=ssl_layer,
                    pooling_method=ssl_pooling_method,
                    num_attention_heads=ssl_attention_heads,
                    selected_layers=ssl_selected_layers,
                    num_selected_layers=ssl_num_selected_layers,
                    use_weighted_layers=ssl_use_weighted_layers,
                )
                self.embedding_dim = self.embedding_extractor.get_embedding_dim()

                # Freeze SSL backbone (keep attention pooling trainable)
                for param in self.embedding_extractor.model.parameters():
                    param.requires_grad = False
                self.embedding_extractor.model.eval()

                # The all-layer softmax weights are NOT part of the frozen backbone -> trainable.
                if ssl_use_weighted_layers:
                    print("[CleanUNet2WithSSLEmbeddings] Weighted-sum over the selected SSL layers "
                          "enabled — layer weights will be trained!")
                    self.embedding_extractor.layer_weights.requires_grad = True

                if ssl_pooling_method == 'self_attention':
                    print("[CleanUNet2WithSSLEmbeddings] Self-Attention Pooling will be trained!")
                    for param in self.embedding_extractor.attention_pooling.parameters():
                        param.requires_grad = True
                    self.embedding_extractor.attention_pooling.train()
                else:
                    self.embedding_extractor.eval()

        # Channel counts of the first two encoder layers (targets of the hierarchical
        # ACOUSTIC injection). Layer 1 -> channels_H, layer 2 -> min(channels_H*2, max_H).
        self.early_channels = (channels_H, min(channels_H * 2, max_H))

        # ----- Build the selected fusion block -----
        # Only one of integration_block / fusion_block is active; the other is None.
        print(f"[CleanUNet2WithSSLEmbeddings] Fusion type: {self.fusion_type} "
              f"(latent_dim={self.latent_dim}, embedding_dim={self.embedding_dim})")
        if self.fusion_type == 'hierarchical_multiscale':
            # Option 3: multi-scale dilated convs + hierarchical FiLM injection.
            # Works with any SSL backbone; the acoustic/semantic split is relative
            # to the number of selected layers (resolved inside the block).
            self.fusion_block = HierarchicalMultiScaleBlock(
                embedding_dim=self.embedding_dim,
                bottleneck_channels=self.latent_dim,
                early_channels=self.early_channels,
                acoustic_layers=self.acoustic_layers,
                semantic_layers=self.semantic_layers,
            )
            self.integration_block = None
        elif self.fusion_type == 'legacy_pooling':
            # Original behaviour: pool SSL sequence -> broadcast -> concat + conv.
            self.fusion_block = None
            self.integration_block = SequenceIntegrationBlock(
                latent_channels=self.latent_dim,
                embedding_dim=self.embedding_dim,
                num_heads=ssl_attention_heads,
                dropout=0.1
            )
        else:
            raise ValueError(
                f"Unknown fusion_type '{self.fusion_type}'. Use one of: "
                "'hierarchical_multiscale', 'legacy_pooling'."
            )

        # Latent Predictor (Stage 2 distillation only -- not built when Stage 2 is
        # SSL-conditioned from noisy audio, which trains the full denoiser instead).
        if stage == 'stage2' and not self._ssl_conditioned:
            print(f"[CleanUNet2WithSSLEmbeddings] Creating latent predictor for Stage 2 "
                  f"(type='{self.latent_predictor_type}')...")
            self.latent_predictor = build_latent_predictor(
                self.latent_predictor_type,
                self.latent_dim,
                **(self.latent_predictor_params or {})
            )
        else:
            self.latent_predictor = None

        # Stage-2 skip-FiLM predictors: one head per modulated early encoder layer
        # (0, 1), mirroring the Stage-1 hierarchical ACOUSTIC FiLM. Each head regresses
        # a GLOBAL (gamma, beta) from the unmodulated skip; zero-init -> starts as the
        # identity so the warm-started decoder is undisturbed at epoch 0.
        if stage == 'stage2' and self.distill_skips and not self._ssl_conditioned:
            self.skip_film_predictors = nn.ModuleDict({
                '0': SkipFiLMHead(self.early_channels[0]),
                '1': SkipFiLMHead(self.early_channels[1]),
            })
            print(f"[CleanUNet2WithSSLEmbeddings] Skip-FiLM distillation ON "
                  f"(early_channels={self.early_channels}).")
        else:
            self.skip_film_predictors = None

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

            # Only load compatible components (skip fusion / X-Vector related components)
            if not new_key.startswith(('xvector_extractor', 'integration_block', 'fusion_block', 'latent_predictor')):
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

        # SSL conditioning reads the clean reference (Stage-1 oracle) or the noisy input
        # (this fork's Stage-2), selected by ssl_audio_source.
        ssl_input = noisy_waveform if self.ssl_audio_source == 'noisy' else clean_audio

        # ============ STAGE 1: Hierarchical fusion (FiLM inside encode) ============
        # The hierarchical option injects FiLM INSIDE the encoder (early acoustic +
        # bottleneck semantic), so it cannot reuse a plain encode(). It needs the
        # full per-layer SSL stack, hence on-the-fly extraction only.
        if (self._ssl_conditioned and self.embedding_type is not None
                and self.fusion_type == 'hierarchical_multiscale'):
            assert ssl_input is not None, (
                f"SSL conditioning requires the {self.ssl_audio_source} audio, but it was None."
            )
            if self.use_preextracted_embeddings:
                raise ValueError(
                    "fusion_type='hierarchical_multiscale' requires on-the-fly extraction "
                    "(set ssl.use_preextracted=false / data.use_preextracted_embeddings=false)."
                )

            # Stacked SELECTED SSL layers: (batch, N, time, dim). The frozen backbone
            # runs inside the extractor; the hierarchical convs downstream are trained.
            stacked = self.embedding_extractor.extract_selected_layers(
                ssl_input.squeeze(1), sample_rate=16000
            )
            stacked = stacked.to(dtype=conditioned_input.dtype)

            early_params, bottleneck_params = self.fusion_block(stacked)
            # Map ACOUSTIC global-FiLM params to encoder layers 1 & 2 (indices 0, 1).
            encoder_film = {
                0: early_params[self.early_channels[0]],
                1: early_params[self.early_channels[1]],
            }
            fused_latent, encoder_states = self.clean_unet.encode(
                conditioned_input, encoder_film=encoder_film, bottleneck_film=bottleneck_params
            )  # fused_latent: (batch, latent_dim, time), already FiLM-fused
            enhanced_waveform = self.clean_unet.decode(fused_latent, encoder_states)

            latents['fused_latent'] = fused_latent.detach()
            latents['embedding'] = stacked.mean(dim=1).detach()  # (batch, time, dim) placeholder
            latents['latent'] = fused_latent.detach()
            # Skip-FiLM distillation targets: the GLOBAL (gamma, beta) applied to the
            # early encoder skips (keyed by encoder layer index 0, 1).
            latents['encoder_film'] = {
                k: (g.detach(), b.detach()) for k, (g, b) in encoder_film.items()
            }

            if return_latents:
                return enhanced_waveform, denoised_spec, latents
            return enhanced_waveform, denoised_spec

        # ============ Encode to Latent Space ============
        latent, encoder_states = self.clean_unet.encode(conditioned_input)
        # latent shape: (batch, latent_dim, time)

        # ============ STAGE 1: With Embeddings ============
        if self._ssl_conditioned and self.embedding_type is not None:
            assert ssl_input is not None, (
                f"SSL conditioning requires the {self.ssl_audio_source} audio, but it was None."
            )

            # Extract embeddings from the SSL source audio (clean or noisy)
            batch_size = ssl_input.shape[0]

            if self.use_preextracted_embeddings:
                # Load pre-extracted SSL embeddings from cache.
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
                                f"Please run extract_ssl_embeddings.py first!"
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
                # Extract SSL embeddings on-the-fly. The frozen backbone runs under no_grad inside
                # the extractor; the softmax over ALL layers stays differentiable.
                embedding = self.embedding_extractor.extract_embeddings(
                    ssl_input.squeeze(1),
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

            # Skip-FiLM distillation: reproduce the Stage-1 early-layer FiLM on the
            # encoder skips. The modulation is GLOBAL (time-broadcast), so we can apply
            # it to the stored skips in place before decoding. encode() reverses the skip
            # list, so encoder layer i sits at position (N-1) - i.
            if self.skip_film_predictors is not None:
                skips = encoder_states['skip_connections']
                n_skips = len(skips)
                predicted_film = {}
                for idx_str, head in self.skip_film_predictors.items():
                    enc_idx = int(idx_str)
                    pos = (n_skips - 1) - enc_idx
                    skip = skips[pos]
                    gamma, beta = head(skip)                        # [B, C], [B, C]
                    skips[pos] = (1.0 + gamma.unsqueeze(-1)) * skip + beta.unsqueeze(-1)
                    predicted_film[enc_idx] = (gamma, beta)
                latents['predicted_encoder_film'] = predicted_film

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
