"""
CleanUNet2 with Self-Supervised Speech Embeddings (WavLM) for Two-Stage Training

Architecture:
    - Stage 1: Train with SSL embeddings (WavLM) injected into latent space
    - Stage 2: Train to replicate latent vectors without embedding extractor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .cleanunet2 import CleanUNet2, SpecUpsampler, Conditioner
from .cleanunet import CleanUNet
from .cleanspecnet import CleanSpecNet
from .integration_block import (
    SequenceIntegrationBlock,
    FiLMCrossAttentionBlock,
    VariationalLatentBlock,
    HierarchicalMultiScaleBlock,
)
from .wavlm_extractor import WavLMExtractor


class CleanUNet2WithSSLEmbeddings(nn.Module):

    def __init__(
        self,
        stage='stage1',
        conditioning_type='addition',
        cleanunet_params=None,
        cleanspecnet_params=None,
        wavlm_model='microsoft/wavlm-large',
        wavlm_layer=12,
        wavlm_cache_dir=None,
        use_preextracted_embeddings=False,
        wavlm_pooling_method='self_attention',
        wavlm_attention_heads=8,
        wavlm_use_weighted_layers=True,
        fusion_type='cross_attention_film',
        acoustic_layers=(1, 8),
        semantic_layers=(17, 24),
    ):
        super().__init__()

        self.stage = stage
        self.use_preextracted_embeddings = use_preextracted_embeddings
        self.use_weighted_layers = wavlm_use_weighted_layers
        self.embedding_type = 'ssl_embeddings'
        self.embedding_dim = 1024

        # Fusion strategy selector. One of:
        #   "cross_attention_film"    -> FiLMCrossAttentionBlock      (Option 1)
        #   "cvae_bottleneck"         -> VariationalLatentBlock        (Option 2)
        #   "hierarchical_multiscale" -> HierarchicalMultiScaleBlock   (Option 3)
        #   "legacy_pooling"          -> SequenceIntegrationBlock      (original)
        self.fusion_type = fusion_type
        self.acoustic_layers = tuple(acoustic_layers)
        self.semantic_layers = tuple(semantic_layers)

        if cleanunet_params is None:
            cleanunet_params = {}
        if cleanspecnet_params is None:
            cleanspecnet_params = {}

        print(f"[CleanUNet2WithSSLEmbeddings] Initializing model...")
        print(f"  - Stage: {stage}")
        print(f"  - Embedding Type: SSL Embeddings (WavLM)")
        print(f"  - WavLM Model: {wavlm_model}")
        print(f"  - WavLM Layer: {wavlm_layer}")
        print(f"  - Use Pre-extracted: {use_preextracted_embeddings}")
        if not use_preextracted_embeddings:
            print(f"  - Pooling Method: {wavlm_pooling_method}")
            if wavlm_pooling_method == 'self_attention':
                print(f"  - Attention Heads: {wavlm_attention_heads}")
            print(f"  - Weighted Layers (softmax over all layers): {wavlm_use_weighted_layers}")
        print(f"  - Conditioning: {conditioning_type}")

        channels_H = cleanunet_params.get('channels_H', 64)
        max_H = cleanunet_params.get('max_H', 768)
        encoder_n_layers = cleanunet_params.get('encoder_n_layers', 8)

        H = channels_H
        for i in range(encoder_n_layers):
            if i > 0:
                H = min(H * 2, max_H)

        self.latent_dim = H
        print(f"  - Calculated Latent Dim: {self.latent_dim}")

        self.clean_unet = CleanUNet(**cleanunet_params)
        self.clean_spec_net = CleanSpecNet(**cleanspecnet_params)
        self.spec_upsampler = SpecUpsampler()
        self.conditioner = Conditioner(
            method=conditioning_type,
            input_channels=1,
            cond_channels=1
        )

        self.embedding_extractor = None
        self.embedding_cache = None

        if stage == 'stage1':
            if use_preextracted_embeddings:
                print("[CleanUNet2WithSSLEmbeddings] Using pre-extracted SSL embeddings...")
                if wavlm_cache_dir:
                    from .wavlm_cache import WavLMCache
                    self.embedding_cache = WavLMCache(
                        cache_dir=wavlm_cache_dir,
                        enabled=True
                    )
                    metadata_file = Path(wavlm_cache_dir) / 'metadata.yaml'
                    num_layers = None
                    if metadata_file.exists():
                        import yaml
                        with open(metadata_file, 'r') as f:
                            metadata = yaml.safe_load(f)
                        self.embedding_dim = metadata.get('embedding_dim', 1024)
                        num_layers = metadata.get('num_layers', None)
                    else:
                        cache_files = list(Path(wavlm_cache_dir).glob("*.pt"))
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
                    if wavlm_use_weighted_layers:
                        if num_layers is None:
                            raise ValueError(
                                "wavlm_use_weighted_layers=True with pre-extracted embeddings requires an "
                                "all-layer cache of shape (num_layers, time, dim). Re-run "
                                "extract_wavlm_embeddings.py to regenerate the cache."
                            )
                        self.cached_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
                        print(f"[CleanUNet2WithSSLEmbeddings] Cached weighted-sum over {num_layers} layers "
                              f"enabled (learnable softmax)")
                else:
                    raise ValueError("wavlm_cache_dir must be specified when use_preextracted_embeddings=True")
            else:
                print("[CleanUNet2WithSSLEmbeddings] Loading WavLM extractor...")
                self.embedding_extractor = WavLMExtractor(
                    model_name=wavlm_model,
                    device='cpu',
                    layer=wavlm_layer,
                    pooling_method=wavlm_pooling_method,
                    num_attention_heads=wavlm_attention_heads,
                    use_weighted_layers=wavlm_use_weighted_layers
                )
                self.embedding_dim = self.embedding_extractor.get_embedding_dim()

                for param in self.embedding_extractor.model.parameters():
                    param.requires_grad = False
                self.embedding_extractor.model.eval()

                # Learnable per-layer softmax weights are NOT part of the frozen
                # backbone, so they remain trainable.
                if wavlm_use_weighted_layers:
                    print("[CleanUNet2WithSSLEmbeddings] Weighted-sum over ALL WavLM layers enabled "
                          "— layer weights will be trained!")
                    self.embedding_extractor.layer_weights.requires_grad = True

                if wavlm_pooling_method == 'self_attention':
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
        print(f"[CleanUNet2WithSSLEmbeddings] Fusion type: {self.fusion_type} "
              f"(latent_dim={self.latent_dim}, embedding_dim={self.embedding_dim})")

        if self.fusion_type == 'cross_attention_film':
            # Option 1: frame-to-frame cross-attention -> per-frame FiLM.
            self.fusion_block = FiLMCrossAttentionBlock(
                latent_channels=self.latent_dim,
                embedding_dim=self.embedding_dim,
                num_heads=wavlm_attention_heads,
                dropout=0.1,
            )
        elif self.fusion_type == 'cvae_bottleneck':
            # Option 2: CVAE latent filter (mu/logvar + reparameterization).
            self.fusion_block = VariationalLatentBlock(
                latent_channels=self.latent_dim,
                embedding_dim=self.embedding_dim,
            )
        elif self.fusion_type == 'hierarchical_multiscale':
            # Option 3: multi-scale dilated convs + hierarchical FiLM injection.
            self.fusion_block = HierarchicalMultiScaleBlock(
                embedding_dim=self.embedding_dim,
                bottleneck_channels=self.latent_dim,
                early_channels=self.early_channels,
                acoustic_layers=self.acoustic_layers,
                semantic_layers=self.semantic_layers,
            )
        elif self.fusion_type == 'legacy_pooling':
            # Original behaviour: pool WavLM sequence -> broadcast -> concat + conv.
            self.fusion_block = SequenceIntegrationBlock(
                latent_channels=self.latent_dim,
                embedding_dim=self.embedding_dim,
                num_heads=wavlm_attention_heads,
                dropout=0.1,
            )
        else:
            raise ValueError(
                f"Unknown fusion_type '{self.fusion_type}'. Use one of: "
                "'cross_attention_film', 'cvae_bottleneck', 'hierarchical_multiscale', "
                "'legacy_pooling'."
            )

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
        print(f"\n[CleanUNet2WithSSLEmbeddings] Loading vanilla checkpoint: {checkpoint_path}")

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
            if not new_key.startswith(('xvector_extractor', 'integration_block', 'fusion_block', 'latent_predictor')):
                filtered_state_dict[new_key] = v

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"[CleanUNet2WithSSLEmbeddings] Loaded {len(filtered_state_dict)} parameters from vanilla checkpoint")

        if missing_keys:
            expected_missing = [k for k in missing_keys if any(
                k.startswith(prefix) for prefix in ['xvector_extractor', 'integration_block', 'latent_predictor']
            )]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]
            if expected_missing:
                print(f"[CleanUNet2WithSSLEmbeddings] Expected missing keys (new components): {len(expected_missing)}")
            if unexpected_missing:
                print(f"[WARNING] Unexpected missing keys: {unexpected_missing[:5]}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys[:5]}")

        print("[CleanUNet2WithSSLEmbeddings] Vanilla checkpoint loaded successfully!\n")

    def _get_sequence_embedding(self, clean_audio, clean_audio_paths):
        """
        Return the WavLM sequence embedding [B, T_wavlm, D_wavlm] used by the
        cross-attention / CVAE / legacy fusion paths (NOT the hierarchical path,
        which needs the full per-layer stack).
        """
        batch_size = clean_audio.shape[0]

        if self.use_preextracted_embeddings:
            assert clean_audio_paths is not None, "Audio paths required for pre-extracted embeddings"

            # Load cached tensors (data only -> no_grad). Each cached item is either:
            #   (num_layers, time, dim)  -> all-layer cache, or
            #   (time, dim)              -> legacy single-layer cache.
            with torch.no_grad():
                raw_list = []
                for i in range(batch_size):
                    audio_path = clean_audio_paths[i]
                    cached_data = self.embedding_cache.get(audio_path, device=clean_audio.device)

                    if cached_data is None:
                        raise RuntimeError(
                            f"Pre-extracted embedding not found for: {audio_path}\n"
                            f"Please run extract_wavlm_embeddings.py first!"
                        )

                    if isinstance(cached_data, dict) and 'embedding' in cached_data:
                        cached_data = cached_data['embedding']

                    raw_list.append(cached_data)

            # Combine layers per item. For all-layer caches this softmax is computed
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

            # Pad to same length and stack -> (batch, time, dim)
            max_len = max(emb.shape[0] for emb in embedding_list)
            padded_embeddings = []
            for emb in embedding_list:
                if emb.shape[0] < max_len:
                    pad = (0, 0, 0, max_len - emb.shape[0])
                    emb = F.pad(emb, pad, mode='constant', value=0)
                padded_embeddings.append(emb)
            embedding = torch.stack(padded_embeddings, dim=0)  # [B, T_wavlm, D_wavlm]
        else:
            # On-the-fly extraction. The frozen WavLM backbone is wrapped in no_grad
            # inside the extractor; the learnable per-layer softmax weights stay
            # differentiable, so gradients flow to them here.
            embedding = self.embedding_extractor.extract_embeddings(
                clean_audio.squeeze(1), sample_rate=16000, return_mean=False
            )  # [B, T_wavlm, D_wavlm]

        return embedding

    def forward(self, noisy_waveform, noisy_spectrogram, clean_audio=None, clean_audio_paths=None, return_latents=False):
        latents = {}

        # ----- Shared front-end: spectrogram branch + conditioning -----
        denoised_spec = self.clean_spec_net(noisy_spectrogram)             # [B, F, T_spec]
        cond_feature = self.spec_upsampler(denoised_spec)                  # [B, 1, ~L]

        if cond_feature.shape[-1] != noisy_waveform.shape[-1]:
            cond_feature = F.interpolate(cond_feature, size=noisy_waveform.shape[-1], mode='linear')

        conditioned_input = self.conditioner(noisy_waveform, cond_feature)  # [B, 1, L]

        # ============================ STAGE 1 ============================
        if self.stage == 'stage1' and self.embedding_type is not None:
            assert clean_audio is not None, "Clean audio is required for Stage 1 training"

            if self.fusion_type == 'hierarchical_multiscale':
                # Option 3: inject FiLM INSIDE the encoder (early acoustic + bottleneck
                # semantic). Requires the full per-layer WavLM stack.
                all_states = self.embedding_extractor.extract_all_layers(
                    clean_audio.squeeze(1), sample_rate=16000
                )  # [B, L_wavlm, T_wavlm, D_wavlm]
                all_states = all_states.to(dtype=conditioned_input.dtype)

                early_params, bottleneck_params = self.fusion_block(all_states)
                # Map ACOUSTIC global-FiLM params to encoder layers 1 & 2 (indices 0, 1).
                encoder_film = {
                    0: early_params[self.early_channels[0]],
                    1: early_params[self.early_channels[1]],
                }
                latent, encoder_states = self.clean_unet.encode(
                    conditioned_input, encoder_film=encoder_film, bottleneck_film=bottleneck_params
                )  # latent: [B, C_unet, T_unet] (already fused via FiLM)
                fused_latent = latent
                enhanced_waveform = self.clean_unet.decode(fused_latent, encoder_states)

                latents['fused_latent'] = fused_latent.detach()
                latents['embedding'] = all_states.mean(dim=1).detach()  # [B, T_wavlm, D] placeholder
                latents['latent'] = fused_latent.detach()

            else:
                # Options 1, 2, legacy: encode first, then fuse at the bottleneck.
                latent, encoder_states = self.clean_unet.encode(conditioned_input)  # [B, C_unet, T_unet]
                embedding = self._get_sequence_embedding(clean_audio, clean_audio_paths)  # [B, T_wavlm, D]
                embedding = embedding.to(dtype=latent.dtype)

                if self.fusion_type == 'cross_attention_film':
                    fused_latent = self.fusion_block(latent, embedding)            # [B, C_unet, T_unet]
                elif self.fusion_type == 'cvae_bottleneck':
                    # Stage-1 samples z; mu/logvar are surfaced for the KL term.
                    fused_latent, mu, logvar = self.fusion_block(latent, embedding, sample=True)
                    latents['kl_mu'] = mu          # [B, C_unet] (NOT detached -> KL grad)
                    latents['kl_logvar'] = logvar  # [B, C_unet]
                else:  # 'legacy_pooling'
                    fused_latent = self.fusion_block(latent, embedding)

                enhanced_waveform = self.clean_unet.decode(fused_latent, encoder_states)

                latents['fused_latent'] = fused_latent.detach()
                latents['embedding'] = embedding.detach()
                latents['latent'] = latent.detach()

        # ============================ STAGE 2 ============================
        elif self.stage == 'stage2' and self.latent_predictor is not None:
            latent, encoder_states = self.clean_unet.encode(conditioned_input)  # [B, C_unet, T_unet]
            predicted_latent = self.latent_predictor(latent)                    # [B, C_unet, T_unet]
            enhanced_waveform = self.clean_unet.decode(predicted_latent, encoder_states)
            latents['predicted_latent'] = predicted_latent
            latents['latent'] = latent.detach()

        # ====================== VANILLA / FALLBACK ======================
        else:
            latent, encoder_states = self.clean_unet.encode(conditioned_input)
            enhanced_waveform = self.clean_unet.decode(latent, encoder_states)
            latents['latent'] = latent.detach()

        if return_latents:
            return enhanced_waveform, denoised_spec, latents
        else:
            return enhanced_waveform, denoised_spec

    @staticmethod
    def _load_and_extract_state_dict(checkpoint_path):
        print(f"[CleanUNet2WithSSLEmbeddings] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'generator' in checkpoint:
            return checkpoint['generator']
        return checkpoint

    def load_stage1_weights(self, checkpoint_path):
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
        print("[CleanUNet2WithSSLEmbeddings] Stage 1 weights loaded successfully")
