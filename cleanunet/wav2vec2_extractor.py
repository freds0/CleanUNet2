"""
Wav2Vec2 Embedding Extractor Module
Uses facebook/wav2vec2-xls-r-300m for self-supervised speech embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


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


class Wav2Vec2Extractor(nn.Module):
    """
    Wav2Vec2 embedding extractor using facebook/wav2vec2-xls-r-300m.
    Extracts self-supervised speech representations for speech enhancement.
    """

    def __init__(self, model_name="facebook/wav2vec2-xls-r-300m", device='cpu', layer=24,
                 pooling_method='self_attention', num_attention_heads=8,
                 use_weighted_layers=False):
        """
        Initialize the Wav2Vec2 extractor.

        Args:
            model_name (str): HuggingFace model name (default: facebook/wav2vec2-xls-r-300m)
            device (str): Device to run the model on
            layer (int): Single layer to use when use_weighted_layers=False (default: 24, -1 = last)
            pooling_method (str): Pooling method - 'mean' or 'self_attention' (default: 'self_attention')
            num_attention_heads (int): Number of attention heads for self-attention pooling (default: 8)
            use_weighted_layers (bool): If True, combine ALL layers with a learnable softmax-weighted
                                        sum (SUPERB-style) instead of a single layer. (default: False)
        """
        super().__init__()

        self.device = device
        self.layer = layer
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.use_weighted_layers = use_weighted_layers

        print(f"[Wav2Vec2Extractor] Loading model: {model_name}")
        print(f"[Wav2Vec2Extractor] Device: {device}")
        print(f"[Wav2Vec2Extractor] Layer: {layer}")
        print(f"[Wav2Vec2Extractor] Pooling method: {pooling_method}")

        try:
            # Load pre-trained model
            # Note: We don't need Wav2Vec2Processor because:
            # 1. Wav2Vec2 models don't use tokenizers (they process audio directly)
            # 2. We normalize audio manually in extract_embeddings()

            # Use safetensors format for security (required by newer transformers)
            # This avoids the torch.load vulnerability issue (CVE-2025-32434)
            print("[Wav2Vec2Extractor] Using safetensors format for secure loading...")
            self.model = Wav2Vec2Model.from_pretrained(
                model_name,
                cache_dir="pretrained_models/wav2vec2",
                use_safetensors=True  # Force safetensors format (secure)
            )

            print(f"[Wav2Vec2Extractor] Model loaded successfully!")
            print(f"[Wav2Vec2Extractor] Model cached at: pretrained_models/wav2vec2")

        except Exception as e:
            print("\n" + "=" * 80)
            print("[ERROR] Failed to load Wav2Vec2 model from HuggingFace!")
            print("=" * 80)
            print(f"Error: {type(e).__name__}: {str(e)[:200]}\n")

            error_msg = str(e).lower()
            if "safetensors" in error_msg or "torch.load" in error_msg:
                print("SOLUTION 1: Install safetensors (recommended)")
                print("-" * 80)
                print("This error occurs due to security requirements in newer transformers.")
                print("Install safetensors to fix:")
                print("")
                print("  pip install safetensors")
                print("")
                print("Then run the extraction again.")
                print("")
                print("SOLUTION 2: Download with safetensors format")
                print("-" * 80)
                print("On a machine with internet:")
                print("")
                print("  pip install safetensors")
                print("  from transformers import Wav2Vec2Model")
                print(f"  model = Wav2Vec2Model.from_pretrained('{model_name}', ")
                print(f"      cache_dir='pretrained_models/wav2vec2', use_safetensors=True)")
                print("")
                print("Then copy 'pretrained_models/wav2vec2' to this machine.")
            else:
                print("SOLUTION: Download the model manually")
                print("-" * 80)
                print("Run this Python code on a machine with internet:")
                print("")
                print("  pip install safetensors")
                print("  from transformers import Wav2Vec2Model")
                print(f"  model = Wav2Vec2Model.from_pretrained('{model_name}', ")
                print(f"      cache_dir='pretrained_models/wav2vec2', use_safetensors=True)")
                print("")
                print("Then copy 'pretrained_models/wav2vec2' to this machine.")
            print("=" * 80 + "\n")
            raise RuntimeError("Wav2Vec2 model loading failed. See instructions above.") from e

        # Move model to device
        self.model = self.model.to(device)

        # Freeze all parameters (we only use it for inference)
        for param in self.model.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"[Wav2Vec2Extractor] Embedding dimension: {self.embedding_dim}")

        # ===== Learnable weighted sum over ALL transformer layers (SUPERB-style) =====
        # A softmax over these per-layer weights determines which layers matter most.
        # These weights stay trainable (they are NOT part of the frozen backbone).
        if self.use_weighted_layers:
            self.num_layers = self.model.config.num_hidden_layers + 1  # embedding output + each layer
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
            print(f"[Wav2Vec2Extractor] Weighted-sum over ALL {self.num_layers} layers enabled "
                  f"(learnable softmax; `layer` argument ignored)")
        else:
            self.num_layers = None

        # Initialize pooling layer
        if self.pooling_method == 'self_attention':
            print(f"[Wav2Vec2Extractor] Initializing Self-Attention Pooling (heads={num_attention_heads})...")
            self.attention_pooling = SelfAttentionPooling(
                embedding_dim=self.embedding_dim,
                num_heads=num_attention_heads,
                dropout=0.1
            )
            self.attention_pooling = self.attention_pooling.to(device)
            print(f"[Wav2Vec2Extractor] Self-Attention Pooling initialized!")
        elif self.pooling_method == 'mean':
            self.attention_pooling = None
            print(f"[Wav2Vec2Extractor] Using mean pooling (no learnable parameters)")
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}. Use 'mean' or 'self_attention'")

        print(f"[Wav2Vec2Extractor] Model ready!")

    def extract_embeddings(self, waveform, sample_rate=16000, return_mean=True):
        """
        Extract Wav2Vec2 embeddings from audio waveform.

        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples) or (batch, 1, samples)
            sample_rate (int): Sample rate of the audio (wav2vec2 expects 16kHz)
            return_mean (bool): If True, return mean pooled embeddings (batch, dim)
                              If False, return full sequence (batch, time_steps, dim)

        Returns:
            embeddings (torch.Tensor): Wav2Vec2 embeddings
                - If return_mean=True: shape (batch, embedding_dim) [e.g., (batch, 1024)]
                - If return_mean=False: shape (batch, time_steps, embedding_dim)
        """
        # Save original device
        original_device = waveform.device

        # Ensure correct format (batch, samples)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Move to model device
        waveform = waveform.to(self.device)

        # Normalize audio to [-1, 1] range (wav2vec2 expects this)
        max_val = waveform.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)  # Avoid division by zero
        waveform = waveform / max_val

        # Resample if needed (wav2vec2 expects 16kHz)
        if sample_rate != 16000:
            print(f"[Wav2Vec2Extractor] Warning: Input sample rate is {sample_rate}Hz, "
                  f"but wav2vec2 expects 16kHz. Resampling...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            ).to(self.device)
            waveform = resampler(waveform)

        # Run the frozen wav2vec2 backbone under no_grad.
        with torch.no_grad():
            outputs = self.model(
                waveform,
                output_hidden_states=True,
                return_dict=True
            )

        # Combine layers. The weighted-sum is computed OUTSIDE no_grad so the learnable
        # layer weights (over ALL layers) receive gradients during training.
        if self.use_weighted_layers:
            weights = F.softmax(self.layer_weights, dim=0)
            hidden_states = sum(w * h for w, h in zip(weights, outputs.hidden_states))
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
    def extract_all_layers(self, waveform, sample_rate=16000):
        """
        Extract the hidden states of ALL layers, stacked along a new layer axis.

        Used to pre-extract embeddings to disk: a learnable softmax over ALL layers
        can then be applied cheaply at training time (without re-running wav2vec2).

        Returns:
            embeddings (torch.Tensor): shape (batch, num_layers, time_steps, embedding_dim),
                                       where num_layers = config.num_hidden_layers + 1.
        """
        original_device = waveform.device

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        waveform = waveform.to(self.device)

        max_val = waveform.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        waveform = waveform / max_val

        if sample_rate != 16000:
            print(f"[Wav2Vec2Extractor] Warning: Input sample rate is {sample_rate}Hz, "
                  f"but wav2vec2 expects 16kHz. Resampling...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(self.device)
            waveform = resampler(waveform)

        outputs = self.model(waveform, output_hidden_states=True, return_dict=True)
        # tuple of (num_layers + 1) tensors (batch, time, dim) -> (batch, num_layers + 1, time, dim)
        stacked = torch.stack(outputs.hidden_states, dim=1)

        return stacked.to(original_device)

    @torch.no_grad()
    def extract_and_interpolate(self, waveform, target_length, sample_rate=16000):
        """
        Extract Wav2Vec2 embeddings and interpolate to match target temporal length.
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

    def get_num_layers(self):
        """Return the number of hidden-state layers (embedding output + transformer layers)."""
        return self.model.config.num_hidden_layers + 1
