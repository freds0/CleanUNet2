"""
X-Vector Extractor Module
Uses SpeechBrain's pre-trained X-Vector model for speaker embeddings
"""

import torch
import torch.nn as nn
import os
import huggingface_hub

# ------------------------------------------------------------------------------
# FIX 1: Ruamel.yaml >= 0.18 Compatibility Patch
# Error: AttributeError: 'Loader' object has no attribute 'max_depth'
# ------------------------------------------------------------------------------
try:
    import ruamel.yaml
    # Versões antigas do hyperpyyaml definem um Loader sem 'max_depth'.
    # Versões novas do ruamel.yaml exigem esse atributo.
    # Injetamos como atributo de classe para resolver o erro de herança.
    if hasattr(ruamel.yaml, 'Loader') and not hasattr(ruamel.yaml.Loader, 'max_depth'):
        ruamel.yaml.Loader.max_depth = None
    if hasattr(ruamel.yaml, 'SafeLoader') and not hasattr(ruamel.yaml.SafeLoader, 'max_depth'):
        ruamel.yaml.SafeLoader.max_depth = None
except ImportError:
    pass
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# FIX 2: Compatibility Patch for huggingface_hub >= 0.27.0 and SpeechBrain
# 1. Map 'use_auth_token' to 'token' (fixing TypeError)
# 2. Handle missing 'custom.py' for standard models (fixing 404 Not Found)
# ------------------------------------------------------------------------------

_original_hf_download = huggingface_hub.hf_hub_download

def _patched_hf_download(*args, **kwargs):
    # Fix 1: Handle use_auth_token deprecation
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    
    # Fix 2: Handle missing custom.py gracefully
    # Some versions of SpeechBrain try to fetch 'custom.py' even if it doesn't exist.
    # We try the download, and if it fails with 404 for custom.py, we return a dummy local file.
    try:
        return _original_hf_download(*args, **kwargs)
    except Exception as e:
        # Check if the error is about custom.py and it's a 404 (Not Found)
        # We check both filename arg and possible kwargs
        filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
        error_str = str(e).lower()
        
        if filename == 'custom.py' and ('404' in error_str or 'not found' in error_str):
            print("[XVectorExtractor] Warning: custom.py not found on HF Hub. Using dummy local file to satisfy SpeechBrain.")
            
            # Create a dummy custom.py if it doesn't exist locally
            dummy_path = os.path.abspath('custom_dummy.py')
            if not os.path.exists(dummy_path):
                with open(dummy_path, 'w') as f:
                    f.write("# Dummy custom interface file for SpeechBrain compatibility\n")
            
            return dummy_path
        
        # If it's another error, re-raise it
        raise e

huggingface_hub.hf_hub_download = _patched_hf_download
# ---------------------------------------------------------

# Update import to avoid deprecation warning
try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


class XVectorExtractor(nn.Module):
    """
    X-Vector extractor using SpeechBrain's pre-trained model.
    Extracts speaker embeddings that can be used for speech enhancement.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize the X-Vector extractor.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device
        
        # Load pre-trained model from HuggingFace
        print(f"[XVectorExtractor] Loading pre-trained model from HuggingFace...")
        
        # Determine run_opts device string
        run_opts_device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("[XVectorExtractor] Warning: CUDA requested but not available. Using CPU.")
            run_opts_device = 'cpu'

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": run_opts_device}
        )
        
        # Freeze all parameters (we only use it for inference)
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode
        self.classifier.eval()
        print(f"[XVectorExtractor] Model loaded successfully!")
    
    def extract_embeddings(self, waveform, sample_rate=16000):
        """
        Extract X-Vector embeddings from audio waveform.
        
        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples) or (batch, 1, samples)
            sample_rate (int): Sample rate of the audio (default: 16000)
            
        Returns:
            embeddings (torch.Tensor): X-Vector embeddings of shape (batch, 512)
        """
        with torch.no_grad():
            # Ensure correct format (batch, samples)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            
            # Normalize audio if necessary
            max_val = waveform.abs().max()
            if max_val > 1.0:
                waveform = waveform / max_val
            
            # Extract embeddings using SpeechBrain's encoder
            embeddings = self.classifier.encode_batch(waveform)
            
        return embeddings
    
    @torch.no_grad()
    def extract_and_interpolate(self, waveform, target_length, sample_rate=16000):
        """
        Extract X-Vectors and interpolate to match target temporal length.
        This is useful for integrating embeddings with encoder features.
        
        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples)
            target_length (int): Target temporal length for interpolation
            sample_rate (int): Sample rate of the audio
            
        Returns:
            embeddings (torch.Tensor): Interpolated embeddings of shape (batch, 512, target_length)
        """
        # Extract embeddings (batch, 512)
        embeddings = self.extract_embeddings(waveform, sample_rate)
        
        # Expand to (batch, 512, 1) for interpolation
        embeddings = embeddings.unsqueeze(-1)
        
        # Interpolate to (batch, 512, target_length)
        embeddings = torch.nn.functional.interpolate(
            embeddings,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        return embeddings
