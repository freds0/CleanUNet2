"""
HuBERT Embedding Extractor Module
Uses facebook/hubert-large-ll60k for self-supervised speech embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


def select_layer_indices(num_selected, total_states):
    """
    Pick `num_selected` layer indices evenly spaced over [0, total_states - 1]
    (endpoints included). For a 24-layer model there are 25 hidden states
    (embedding output + 24 transformer layers), so:
        num_selected=3 -> [0, 12, 24]  (initial, middle, final)
        num_selected=5 -> [0, 6, 12, 18, 24]
    For a 48-layer model (49 states): num_selected=3 -> [0, 24, 48].

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


class HubertExtractor(nn.Module):
    """
    HuBERT embedding extractor using facebook/hubert-large-ll60k.
    Extracts self-supervised speech representations for speech enhancement.
    """

    def __init__(self, model_name="facebook/hubert-large-ll60k", device='cpu', layer=24,
                 pooling_method='self_attention', num_attention_heads=8,
                 num_selected_layers=3, selected_layers=None, use_weighted_layers=True):
        """
        Initialize the HuBERT extractor.

        Args:
            model_name (str): HuggingFace model name (default: facebook/hubert-large-ll60k)
            device (str): Device to run the model on
            layer (int): Single layer to use when use_weighted_layers=False (default: 24, -1 = last)
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

        print(f"[HubertExtractor] Loading model: {model_name}")
        print(f"[HubertExtractor] Device: {device}")
        print(f"[HubertExtractor] Layer: {layer}")
        print(f"[HubertExtractor] Pooling method: {pooling_method}")

        try:
            # Load pre-trained model
            # Note: We don't need Wav2Vec2Processor because:
            # 1. HuBERT models don't use tokenizers (they process audio directly)
            # 2. We normalize audio manually in extract_embeddings()

            # Use safetensors format for security (required by newer transformers)
            # This avoids the torch.load vulnerability issue (CVE-2025-32434)
            print("[HubertExtractor] Using safetensors format for secure loading...")
            self.model = HubertModel.from_pretrained(
                model_name,
                cache_dir="pretrained_models/hubert",
                use_safetensors=True  # Force safetensors format (secure)
            )

            print(f"[HubertExtractor] Model loaded successfully!")
            print(f"[HubertExtractor] Model cached at: pretrained_models/hubert")

        except Exception as e:
            print("\n" + "=" * 80)
            print("[ERROR] Failed to load HuBERT model from HuggingFace!")
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
                print("  from transformers import HubertModel")
                print(f"  model = HubertModel.from_pretrained('{model_name}', ")
                print(f"      cache_dir='pretrained_models/hubert', use_safetensors=True)")
                print("")
                print("Then copy 'pretrained_models/hubert' to this machine.")
            else:
                print("SOLUTION: Download the model manually")
                print("-" * 80)
                print("Run this Python code on a machine with internet:")
                print("")
                print("  pip install safetensors")
                print("  from transformers import HubertModel")
                print(f"  model = HubertModel.from_pretrained('{model_name}', ")
                print(f"      cache_dir='pretrained_models/hubert', use_safetensors=True)")
                print("")
                print("Then copy 'pretrained_models/hubert' to this machine.")
            print("=" * 80 + "\n")
            raise RuntimeError("HuBERT model loading failed. See instructions above.") from e

        # Move model to device
        self.model = self.model.to(device)

        # Freeze all parameters (we only use it for inference)
        for param in self.model.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"[HubertExtractor] Embedding dimension: {self.embedding_dim}")

        # ===== Multi-layer selection (N layers: initial / middle / final by default) =====
        self.total_states = self.model.config.num_hidden_layers + 1  # embedding output + each layer
        if self._selected_layers_override == 'all':
            # '++' mode: learnable softmax over ALL hidden states (every layer).
            self.selected_layers = list(range(self.total_states))
        elif self._selected_layers_override is not None:
            self.selected_layers = list(self._selected_layers_override)
        else:
            self.selected_layers = select_layer_indices(self._num_selected_layers, self.total_states)
        print(f"[HubertExtractor] Selected layers (of {self.total_states}): {self.selected_layers}")

        # Learnable softmax weights over the N selected layers (NOT part of the frozen backbone).
        if self.use_weighted_layers:
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.selected_layers)) / len(self.selected_layers)
            )
            print(f"[HubertExtractor] Weighted-sum over {len(self.selected_layers)} selected layers "
                  f"enabled (learnable softmax)")

        # Initialize pooling layer
        if self.pooling_method == 'self_attention':
            print(f"[HubertExtractor] Initializing Self-Attention Pooling (heads={num_attention_heads})...")
            self.attention_pooling = SelfAttentionPooling(
                embedding_dim=self.embedding_dim,
                num_heads=num_attention_heads,
                dropout=0.1
            )
            self.attention_pooling = self.attention_pooling.to(device)
            print(f"[HubertExtractor] Self-Attention Pooling initialized!")
        elif self.pooling_method == 'mean':
            self.attention_pooling = None
            print(f"[HubertExtractor] Using mean pooling (no learnable parameters)")
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}. Use 'mean' or 'self_attention'")

        print(f"[HubertExtractor] Model ready!")

    def extract_embeddings(self, waveform, sample_rate=16000, return_mean=True):
        """
        Extract HuBERT embeddings from audio waveform.

        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples) or (batch, 1, samples)
            sample_rate (int): Sample rate of the audio (HuBERT expects 16kHz)
            return_mean (bool): If True, return mean pooled embeddings (batch, dim)
                              If False, return full sequence (batch, time_steps, dim)

        Returns:
            embeddings (torch.Tensor): HuBERT embeddings
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

        # Normalize audio to [-1, 1] range (HuBERT expects this)
        max_val = waveform.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)  # Avoid division by zero
        waveform = waveform / max_val

        # Resample if needed (HuBERT expects 16kHz)
        if sample_rate != 16000:
            print(f"[HubertExtractor] Warning: Input sample rate is {sample_rate}Hz, "
                  f"but HuBERT expects 16kHz. Resampling...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            ).to(self.device)
            waveform = resampler(waveform)

        # Run the frozen HuBERT backbone under no_grad.
        with torch.no_grad():
            outputs = self.model(
                waveform,
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
        can then be applied cheaply at training time (without re-running wav2vec2).

        Returns:
            embeddings (torch.Tensor): shape (batch, N, time_steps, embedding_dim),
                                       where N = len(self.selected_layers).
        """
        original_device = waveform.device

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        waveform = waveform.to(self.device)

        max_val = waveform.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        waveform = waveform / max_val

        if sample_rate != 16000:
            print(f"[HubertExtractor] Warning: Input sample rate is {sample_rate}Hz, "
                  f"but HuBERT expects 16kHz. Resampling...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(self.device)
            waveform = resampler(waveform)

        outputs = self.model(waveform, output_hidden_states=True, return_dict=True)
        selected = [outputs.hidden_states[i] for i in self.selected_layers]
        # list of N tensors (batch, time, dim) -> (batch, N, time, dim)
        stacked = torch.stack(selected, dim=1)

        return stacked.to(original_device)

    @torch.no_grad()
    def extract_and_interpolate(self, waveform, target_length, sample_rate=16000):
        """
        Extract HuBERT embeddings and interpolate to match target temporal length.
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
