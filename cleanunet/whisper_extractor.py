"""
Whisper Embedding Extractor Module
Uses openai/whisper-large-v3 (encoder only) for speech embeddings.

Whisper is an encoder-decoder ASR model; here we use ONLY the frozen audio
encoder as a self-supervised feature extractor. Audio is converted to a log-mel
spectrogram (padded/truncated to 30 s) by the Whisper feature extractor, then the
encoder produces a FIXED 1500-frame hidden-state sequence per layer. We combine a
few selected layers (initial / middle / final) with a learnable softmax-weighted sum.

Note: because Whisper pads every clip to 30 s, the encoder output length is always
1500 frames regardless of the input duration (the integration block pools it anyway).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperFeatureExtractor
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


def select_layer_indices(num_selected, total_states):
    """
    Pick `num_selected` layer indices evenly spaced over [0, total_states - 1]
    (endpoints included). For whisper-large-v3 there are 33 hidden states
    (conv/embedding output + 32 encoder layers), so:
        num_selected=3 -> [0, 16, 32]  (initial, middle, final)
        num_selected=5 -> [0, 8, 16, 24, 32]

    Args:
        num_selected (int): how many layers to select (N).
        total_states (int): total number of hidden states available.

    Returns:
        list[int]: sorted, de-duplicated layer indices.
    """
    if num_selected >= total_states:
        return list(range(total_states))
    if num_selected <= 1:
        return [total_states - 1]  # just the final layer
    idx = torch.linspace(0, total_states - 1, steps=num_selected).round().long().tolist()
    # de-duplicate while preserving order
    seen = []
    for i in idx:
        if i not in seen:
            seen.append(i)
    return seen


class SelfAttentionPooling(nn.Module):
    """
    Self-Attention Pooling layer for temporal aggregation.
    Uses multi-head self-attention followed by weighted averaging.
    """

    def __init__(self, embedding_dim, num_heads=8, dropout=0.1):
        """
        Initialize Self-Attention Pooling.

        Args:
            embedding_dim (int): Dimension of input embeddings
            num_heads (int): Number of attention heads (default: 8)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq_len, embed_dim)
        )

        # Learnable query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hidden_states):
        """
        Apply self-attention pooling to aggregate temporal features.

        Args:
            hidden_states (torch.Tensor): Input features
                                         Shape: (batch, time_steps, embedding_dim)

        Returns:
            pooled (torch.Tensor): Pooled features
                                  Shape: (batch, embedding_dim)
        """
        batch_size = hidden_states.shape[0]

        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, embedding_dim)

        # Apply multi-head attention
        # query attends to hidden_states
        attn_output, attn_weights = self.self_attention(
            query=query,
            key=hidden_states,
            value=hidden_states
        )
        # attn_output shape: (batch, 1, embedding_dim)

        # Remove the query dimension and apply layer norm
        pooled = attn_output.squeeze(1)  # (batch, embedding_dim)
        pooled = self.layer_norm(pooled)

        return pooled


class WhisperExtractor(nn.Module):
    """
    Whisper embedding extractor using the (frozen) openai/whisper-large-v3 encoder.
    Extracts speech representations for speech enhancement.
    """

    def __init__(self, model_name="openai/whisper-large-v3", device='cpu', layer=12,
                 pooling_method='self_attention', num_attention_heads=8,
                 num_selected_layers=3, selected_layers=None, use_weighted_layers=True):
        """
        Initialize the Whisper extractor.

        Args:
            model_name (str): HuggingFace model name (default: openai/whisper-large-v3)
            device (str): Device to run the model on
            layer (int): Single layer to use when use_weighted_layers=False (default: 12, -1 = last)
            pooling_method (str): Pooling method - 'mean' or 'self_attention' (default: 'self_attention')
            num_attention_heads (int): Number of attention heads for self-attention pooling (default: 8)
            num_selected_layers (int): Number N of layers to use when use_weighted_layers=True and
                                       selected_layers is None. Default 3 -> initial, middle, final.
            selected_layers (list[int]): Explicit layer indices to use (overrides num_selected_layers).
            use_weighted_layers (bool): If True, combine the N selected layers with a learnable
                                        softmax-weighted sum (default: True).
        """
        super().__init__()

        self.device = device
        self.layer = layer
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.use_weighted_layers = use_weighted_layers
        self._num_selected_layers = num_selected_layers
        self._selected_layers_override = selected_layers

        print(f"[WhisperExtractor] Loading model: {model_name}")
        print(f"[WhisperExtractor] Device: {device}")
        print(f"[WhisperExtractor] Layer: {layer}")
        print(f"[WhisperExtractor] Pooling method: {pooling_method}")

        try:
            # Load the pre-trained Whisper model and keep ONLY the audio encoder.
            # The decoder is not needed for feature extraction, so we drop it to
            # save memory. Use safetensors for secure loading (CVE-2025-32434).
            print("[WhisperExtractor] Using safetensors format for secure loading...")
            full_model = WhisperModel.from_pretrained(
                model_name,
                cache_dir="pretrained_models/whisper",
                use_safetensors=True
            )
            self.model = full_model.encoder
            del full_model  # free the (unused) decoder

            # Feature extractor turns raw audio into the log-mel spectrogram the
            # encoder expects (it also pads/truncates to 30 s and picks the right
            # number of mel bins, e.g. 128 for large-v3).
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_name,
                cache_dir="pretrained_models/whisper"
            )

            print(f"[WhisperExtractor] Model loaded successfully!")
            print(f"[WhisperExtractor] Model cached at: pretrained_models/whisper")

        except Exception as e:
            print("\n" + "=" * 80)
            print("[ERROR] Failed to load Whisper model from HuggingFace!")
            print("=" * 80)
            print(f"Error: {type(e).__name__}: {str(e)[:200]}\n")
            print("SOLUTION: Download the model on a machine with internet:")
            print("-" * 80)
            print("  pip install safetensors")
            print("  from transformers import WhisperModel, WhisperFeatureExtractor")
            print(f"  WhisperModel.from_pretrained('{model_name}',")
            print(f"      cache_dir='pretrained_models/whisper', use_safetensors=True)")
            print(f"  WhisperFeatureExtractor.from_pretrained('{model_name}',")
            print(f"      cache_dir='pretrained_models/whisper')")
            print("")
            print("Then copy 'pretrained_models/whisper' to this machine.")
            print("=" * 80 + "\n")
            raise RuntimeError("Whisper model loading failed. See instructions above.") from e

        # Move model to device
        self.model = self.model.to(device)

        # Freeze all parameters (we only use it for inference)
        for param in self.model.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.model.eval()

        # Get embedding dimension (Whisper encoder hidden size = d_model)
        self.embedding_dim = self.model.config.d_model
        print(f"[WhisperExtractor] Embedding dimension: {self.embedding_dim}")

        # ===== Multi-layer selection (N layers: initial / middle / final by default) =====
        self.total_states = self.model.config.encoder_layers + 1  # conv/embedding output + each layer
        if self._selected_layers_override == 'all':
            # '++' mode: learnable softmax over ALL hidden states (every layer).
            self.selected_layers = list(range(self.total_states))
        elif self._selected_layers_override is not None:
            self.selected_layers = list(self._selected_layers_override)
        else:
            self.selected_layers = select_layer_indices(self._num_selected_layers, self.total_states)
        print(f"[WhisperExtractor] Selected layers (of {self.total_states}): {self.selected_layers}")

        # Learnable softmax weights over the N selected layers (NOT part of the frozen backbone).
        if self.use_weighted_layers:
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.selected_layers)) / len(self.selected_layers)
            )
            print(f"[WhisperExtractor] Weighted-sum over {len(self.selected_layers)} selected layers "
                  f"enabled (learnable softmax)")

        # Initialize pooling layer
        if self.pooling_method == 'self_attention':
            print(f"[WhisperExtractor] Initializing Self-Attention Pooling (heads={num_attention_heads})...")
            self.attention_pooling = SelfAttentionPooling(
                embedding_dim=self.embedding_dim,
                num_heads=num_attention_heads,
                dropout=0.1
            )
            self.attention_pooling = self.attention_pooling.to(device)
            print(f"[WhisperExtractor] Self-Attention Pooling initialized!")
        elif self.pooling_method == 'mean':
            self.attention_pooling = None
            print(f"[WhisperExtractor] Using mean pooling (no learnable parameters)")
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}. Use 'mean' or 'self_attention'")

        print(f"[WhisperExtractor] Model ready!")

    def _waveform_to_features(self, waveform, sample_rate):
        """
        Convert a batch of raw waveforms into Whisper log-mel input features.

        Args:
            waveform (torch.Tensor): (batch, samples) at `sample_rate`.
            sample_rate (int): input sample rate (Whisper expects 16kHz).

        Returns:
            torch.Tensor: input_features of shape (batch, num_mel_bins, 3000) on CPU.
        """
        # Resample to 16kHz if needed (Whisper is trained at 16kHz)
        if sample_rate != 16000:
            print(f"[WhisperExtractor] Warning: Input sample rate is {sample_rate}Hz, "
                  f"but Whisper expects 16kHz. Resampling...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(waveform.device)
            waveform = resampler(waveform)

        # The HF feature extractor runs on CPU numpy. This is fine: the backbone is
        # frozen and runs under no_grad, so no gradient needs to flow through the mel.
        wav_list = [w.detach().cpu().float().numpy() for w in waveform]
        features = self.feature_extractor(
            wav_list, sampling_rate=16000, return_tensors="pt"
        ).input_features  # (batch, num_mel_bins, 3000)
        return features

    def extract_embeddings(self, waveform, sample_rate=16000, return_mean=True):
        """
        Extract Whisper embeddings from audio waveform.

        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples) or (batch, 1, samples)
            sample_rate (int): Sample rate of the audio (Whisper expects 16kHz)
            return_mean (bool): If True, return pooled embeddings (batch, dim)
                              If False, return full sequence (batch, time_steps, dim)

        Returns:
            embeddings (torch.Tensor): Whisper embeddings
                - If return_mean=True: shape (batch, embedding_dim) [e.g., (batch, 1280)]
                - If return_mean=False: shape (batch, time_steps, embedding_dim) [time_steps=1500]
        """
        # Save original device
        original_device = waveform.device

        # Ensure correct format (batch, samples)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Derive the actual device/dtype from the model parameters. self.device is
        # stale after Lightning moves the module to GPU and/or casts it to fp16, so
        # relying on it would mismatch the input against the (cuda/half) weights.
        param = next(self.model.parameters())
        model_device, model_dtype = param.device, param.dtype

        # Build log-mel input features and match the model device/dtype
        input_features = self._waveform_to_features(waveform, sample_rate)
        input_features = input_features.to(device=model_device, dtype=model_dtype)

        # Run the frozen Whisper encoder under no_grad.
        with torch.no_grad():
            outputs = self.model(
                input_features,
                output_hidden_states=True,
                return_dict=True
            )

        # Combine layers. The weighted-sum is computed OUTSIDE no_grad so the learnable
        # layer weights (over the N selected layers) receive gradients during training.
        if self.use_weighted_layers:
            weights = F.softmax(self.layer_weights, dim=0)
            selected = [outputs.hidden_states[i] for i in self.selected_layers]
            hidden_states = sum(w * h for w, h in zip(weights, selected))
        elif self.layer == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[self.layer]

        # hidden_states shape: (batch, time_steps, embedding_dim)

        if return_mean:
            # Apply pooling based on method
            if self.pooling_method == 'self_attention':
                embeddings = self.attention_pooling(hidden_states)
            elif self.pooling_method == 'mean':
                embeddings = hidden_states.mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        else:
            # Return full sequence
            embeddings = hidden_states

        # Move back to original device
        embeddings = embeddings.to(original_device)

        return embeddings

    @torch.no_grad()
    def extract_selected_layers(self, waveform, sample_rate=16000):
        """
        Extract ONLY the N selected layers (initial/middle/final by default), stacked.

        Used to pre-extract embeddings to disk: a learnable softmax over these N layers
        can then be applied cheaply at training time (without re-running Whisper).

        Returns:
            embeddings (torch.Tensor): shape (batch, N, time_steps, embedding_dim),
                                       where N = len(self.selected_layers).
        """
        original_device = waveform.device

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Derive the actual device/dtype from the model parameters (self.device is
        # stale once Lightning moves/casts the module).
        param = next(self.model.parameters())
        model_device, model_dtype = param.device, param.dtype

        input_features = self._waveform_to_features(waveform, sample_rate)
        input_features = input_features.to(device=model_device, dtype=model_dtype)

        outputs = self.model(input_features, output_hidden_states=True, return_dict=True)
        selected = [outputs.hidden_states[i] for i in self.selected_layers]
        # list of N tensors (batch, time, dim) -> (batch, N, time, dim)
        stacked = torch.stack(selected, dim=1)

        return stacked.to(original_device)

    @torch.no_grad()
    def extract_and_interpolate(self, waveform, target_length, sample_rate=16000):
        """
        Extract Whisper embeddings and interpolate to match target temporal length.
        This is useful for integrating embeddings with encoder features.

        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples)
            target_length (int): Target temporal length for interpolation
            sample_rate (int): Sample rate of the audio

        Returns:
            embeddings (torch.Tensor): Interpolated embeddings of shape (batch, embedding_dim, target_length)
        """
        # Extract mean pooled embeddings (batch, embedding_dim)
        embeddings = self.extract_embeddings(waveform, sample_rate, return_mean=True)

        # Expand to (batch, embedding_dim, 1) for interpolation
        embeddings = embeddings.unsqueeze(-1)

        # Interpolate to (batch, embedding_dim, target_length)
        embeddings = torch.nn.functional.interpolate(
            embeddings,
            size=target_length,
            mode='linear',
            align_corners=False
        )

        return embeddings

    def get_embedding_dim(self):
        """Return the embedding dimension."""
        return self.embedding_dim

    def get_num_selected_layers(self):
        """Return N = the number of selected layers."""
        return len(self.selected_layers)

    def get_selected_layers(self):
        """Return the list of selected layer indices."""
        return list(self.selected_layers)
