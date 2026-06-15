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
    
    def __init__(self, device='cpu', local_path=None):
        """
        Initialize the X-Vector extractor.

        Args:
            device (str): Device to run the model on ('cpu' recommended for stability)
            local_path (str): Optional local path to pre-downloaded model directory
        """
        super().__init__()

        # FORCE CPU for X-Vector extractor to avoid device/dtype conflicts
        # The model is frozen anyway, so CPU performance is acceptable
        self.device = 'cpu'
        run_opts_device = 'cpu'

        print("[XVectorExtractor] X-Vector model will run on CPU (avoids device/dtype conflicts)")

        # Try loading from local path first if provided
        if local_path and os.path.exists(local_path):
            print(f"[XVectorExtractor] Loading model from local path: {local_path}")
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source=local_path,
                    savedir=local_path,
                    run_opts={"device": run_opts_device}
                )
                print("[XVectorExtractor] Model loaded successfully from local path!")
            except Exception as e:
                print(f"[XVectorExtractor] Failed to load from local path: {e}")
                print("[XVectorExtractor] Falling back to HuggingFace download...")
                local_path = None  # Fall back to HuggingFace

        # Load pre-trained model from HuggingFace if no local path
        if not local_path:
            print(f"[XVectorExtractor] Loading pre-trained model from HuggingFace...")
            print(f"[XVectorExtractor] This requires internet connection for first-time download.")

            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-xvect-voxceleb",
                    savedir="pretrained_models/spkrec-xvect-voxceleb",
                    run_opts={"device": run_opts_device}
                )
                print(f"[XVectorExtractor] Model loaded successfully!")
                print(f"[XVectorExtractor] Model cached at: pretrained_models/spkrec-xvect-voxceleb")
            except Exception as e:
                print("\n" + "=" * 80)
                print("[ERROR] Failed to download X-Vector model from HuggingFace!")
                print("=" * 80)
                print(f"Error: {type(e).__name__}: {str(e)[:200]}\n")
                print("SOLUTION 1: Download the model manually on a machine with internet")
                print("-" * 80)
                print("Run this Python code:")
                print("")
                print("  from speechbrain.pretrained import EncoderClassifier")
                print("  classifier = EncoderClassifier.from_hparams(")
                print("      source='speechbrain/spkrec-xvect-voxceleb',")
                print("      savedir='pretrained_models/spkrec-xvect-voxceleb'")
                print("  )")
                print("")
                print("Then copy 'pretrained_models/spkrec-xvect-voxceleb' to this machine.")
                print("")
                print("SOLUTION 2: Use pre-downloaded model")
                print("-" * 80)
                print("If you already have the model downloaded, add to your config:")
                print("")
                print("  model:")
                print("    xvector_local_path: '/path/to/pretrained_models/spkrec-xvect-voxceleb'")
                print("")
                print("SOLUTION 3: Skip Stage-1 (if you have a Stage-1 checkpoint)")
                print("-" * 80)
                print("If you already have a Stage-1 checkpoint, go directly to Stage-2:")
                print("")
                print("  python train.py --config configs/train_xvector_vanilla_stage2.yaml --stage stage2")
                print("")
                print("=" * 80 + "\n")
                raise RuntimeError("X-Vector model download failed. See instructions above.") from e

        # Freeze all parameters first (we only use it for inference)
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.classifier.eval()

        # CRITICAL: Aggressively move model to CPU and convert to float32
        # This must be done AFTER loading to override SpeechBrain's device placement
        print("[XVectorExtractor] Force moving model to CPU...")

        # Move entire model to CPU
        self.classifier = self.classifier.to('cpu')

        # Force float32 on entire model
        self.classifier = self.classifier.float()

        # Recursively ensure ALL submodules are on CPU and float32
        def move_module_to_cpu_float32(module):
            """Recursively move module to CPU and float32"""
            module.to('cpu')
            module.float()

            # Move all parameters
            for param in module._parameters.values():
                if param is not None:
                    param.data = param.data.to('cpu').float()
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to('cpu').float()

            # Move all buffers
            for buffer in module._buffers.values():
                if buffer is not None:
                    buffer.data = buffer.data.to('cpu').float()

            # Recurse to children
            for child in module.children():
                move_module_to_cpu_float32(child)

        move_module_to_cpu_float32(self.classifier)

        # Verify model is on CPU
        sample_param = next(self.classifier.parameters())
        print(f"[XVectorExtractor] Model device: {sample_param.device}, dtype: {sample_param.dtype}")

        if sample_param.device.type != 'cpu':
            raise RuntimeError(f"Failed to move X-Vector model to CPU! Device is: {sample_param.device}")

        print(f"[XVectorExtractor] Model ready on CPU with float32!")
    
    def extract_embeddings(self, waveform, sample_rate=16000):
        """
        Extract X-Vector embeddings from audio waveform.

        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, samples) or (batch, 1, samples)
            sample_rate (int): Sample rate of the audio (default: 16000)

        Returns:
            embeddings (torch.Tensor): X-Vector embeddings of shape (batch, 512)
        """
        # CRITICAL: Disable AMP/autocast for X-Vector extraction
        # This prevents any automatic dtype conversion that could cause mismatches
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            # Save original device for restoration
            original_device = waveform.device

            # Ensure correct format (batch, samples)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)

            # CRITICAL: Convert to float32 and move to CPU
            # X-Vector model always runs on CPU to avoid device/dtype conflicts
            waveform_cpu = waveform.float().cpu()

            # Normalize audio if necessary
            max_val = waveform_cpu.abs().max()
            if max_val > 1.0:
                waveform_cpu = waveform_cpu / max_val

            # Double-check model is on CPU (safety check)
            model_device = next(self.classifier.parameters()).device
            if model_device.type != 'cpu':
                print(f"[XVectorExtractor] WARNING: Model drifted to {model_device}, moving back to CPU!")
                self.classifier = self.classifier.to('cpu')

            # Extract embeddings using SpeechBrain's encoder (on CPU)
            embeddings = self.classifier.encode_batch(waveform_cpu)

            # Move embeddings back to original device (GPU if training on GPU)
            embeddings = embeddings.to(original_device)

            # Ensure output is float32 (don't convert to float16)
            embeddings = embeddings.float()

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
