"""
Speaker Embedding Extractor Module
Supports multiple models:
  - X-Vector (512-dim) via SpeechBrain
  - ECAPA-TDNN (192-dim) via SpeechBrain
  - Clova ResNet (512-dim) via local checkpoint (Coqui-TTS architecture)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import os
import huggingface_hub

# ------------------------------------------------------------------------------
# FIX 1: Ruamel.yaml >= 0.18 Compatibility Patch
# ------------------------------------------------------------------------------
try:
    import ruamel.yaml
    if hasattr(ruamel.yaml, 'Loader') and not hasattr(ruamel.yaml.Loader, 'max_depth'):
        ruamel.yaml.Loader.max_depth = None
    if hasattr(ruamel.yaml, 'SafeLoader') and not hasattr(ruamel.yaml.SafeLoader, 'max_depth'):
        ruamel.yaml.SafeLoader.max_depth = None
except ImportError:
    pass

# ------------------------------------------------------------------------------
# FIX 2: Compatibility Patch for huggingface_hub >= 0.27.0 and SpeechBrain
# ------------------------------------------------------------------------------
_original_hf_download = huggingface_hub.hf_hub_download

def _patched_hf_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    try:
        return _original_hf_download(*args, **kwargs)
    except Exception as e:
        filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
        error_str = str(e).lower()
        if filename == 'custom.py' and ('404' in error_str or 'not found' in error_str):
            print("[SpeakerExtractor] Warning: custom.py not found on HF Hub. Using dummy file.")
            dummy_path = os.path.abspath('custom_dummy.py')
            if not os.path.exists(dummy_path):
                with open(dummy_path, 'w') as f:
                    f.write("# Dummy custom interface file for SpeechBrain compatibility\n")
            return dummy_path
        raise e

huggingface_hub.hf_hub_download = _patched_hf_download

try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


# Registry of supported speaker embedding models
SPEAKER_MODELS = {
    'xvector': {
        'source': 'speechbrain/spkrec-xvect-voxceleb',
        'savedir': 'pretrained_models/spkrec-xvect-voxceleb',
        'embedding_dim': 512,
        'display_name': 'X-Vector',
        'backend': 'speechbrain',
    },
    'ecapa': {
        'source': 'speechbrain/spkrec-ecapa-voxceleb',
        'savedir': 'pretrained_models/spkrec-ecapa-voxceleb',
        'embedding_dim': 192,
        'display_name': 'ECAPA-TDNN',
        'backend': 'speechbrain',
    },
    'clova': {
        'embedding_dim': 512,
        'display_name': 'Clova ResNet',
        'backend': 'clova',
    },
    'redimnet': {
        'embedding_dim': 192,
        'display_name': 'ReDimNet',
        'backend': 'redimnet',
        'hub_repo': 'IDRnD/ReDimNet',
        'default_model_name': 'b2',
        'default_train_type': 'ft_lm',
        'default_dataset': 'vox2',
    },
    'titanet': {
        'embedding_dim': 192,
        'display_name': 'TitaNet-Large',
        'backend': 'nemo',
        'nemo_model_name': 'titanet_large',
    },
    'speakernet': {
        'embedding_dim': 256,
        'display_name': 'SpeakerNet',
        'backend': 'nemo',
        'nemo_model_name': 'speakerverification_speakernet',
    },
}


# ==============================================================================
# Clova ResNet Speaker Encoder (self-contained, no TTS dependency)
# Adapted from: https://github.com/clovaai/voxceleb_trainer
# ==============================================================================

class _PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2
        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class _SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = _SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ClovaResNetEncoder(nn.Module):
    """
    ResNet speaker encoder (H/ASP) from Clova AI / Coqui-TTS.
    Produces 512-dim speaker embeddings from raw waveforms at 16kHz.
    """

    def __init__(self, input_dim=64, proj_dim=512, layers=None, num_filters=None, audio_config=None):
        super().__init__()
        if layers is None:
            layers = [3, 4, 6, 3]
        if num_filters is None:
            num_filters = [32, 64, 128, 256]

        self.input_dim = input_dim
        self.log_input = True
        self.proj_dim = proj_dim

        # Mel spectrogram front-end
        self.torch_spec = nn.Sequential(
            _PreEmphasis(audio_config.get("preemphasis", 0.97)),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config.get("sample_rate", 16000),
                n_fft=audio_config.get("fft_size", 512),
                win_length=audio_config.get("win_length", 400),
                hop_length=audio_config.get("hop_length", 160),
                window_fn=torch.hamming_window,
                n_mels=audio_config.get("num_mels", 64),
            ),
        )

        self.instancenorm = nn.InstanceNorm1d(input_dim)
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self._make_layer(_SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self._make_layer(_SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(_SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(_SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        outmap_size = int(input_dim / 8)
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        # ASP encoder: mean + std
        out_dim = num_filters[3] * outmap_size * 2
        self.fc = nn.Linear(out_dim, proj_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, l2_norm=True):
        """
        Args:
            x: (batch, samples) raw waveform
        Returns:
            embeddings: (batch, proj_dim)
        """
        x = self.torch_spec(x)
        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        w = self.attention(x)

        # ASP pooling
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    @torch.no_grad()
    def compute_embedding(self, x, num_frames=250, num_eval=10):
        """Compute embedding with multiple crops and averaging."""
        num_frames_samples = num_frames * 160  # hop_length=160
        max_len = x.shape[1]

        if max_len < num_frames_samples:
            num_frames_samples = max_len

        offsets = np.linspace(0, max_len - num_frames_samples, num=num_eval)
        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            frames_batch.append(x[:, offset:offset + num_frames_samples])

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.forward(frames_batch, l2_norm=True)
        return torch.mean(embeddings, dim=0, keepdim=True)


# ==============================================================================
# Main SpeakerExtractor class
# ==============================================================================

class SpeakerExtractor(nn.Module):
    """
    Generic speaker embedding extractor.
    Supports X-Vector, ECAPA-TDNN (via SpeechBrain), Clova ResNet (local), and ReDimNet (torch.hub).
    """

    def __init__(self, model_name='xvector', device='cpu', local_path=None):
        """
        Args:
            model_name (str): Model to use ('xvector', 'ecapa', 'clova', or 'redimnet')
            device (str): Device to run on ('cpu' recommended)
            local_path (str): Path to model directory.
                For clova: directory containing model_se.pth and config_se.json
                For redimnet: optional "model_name:train_type:dataset" string (e.g. "b2:ft_lm:vox2")
                For speechbrain models: optional local path to pre-downloaded model
        """
        super().__init__()

        if model_name not in SPEAKER_MODELS:
            raise ValueError(
                f"Unknown speaker model: '{model_name}'. "
                f"Supported: {list(SPEAKER_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_info = SPEAKER_MODELS[model_name]
        self.embedding_dim = self.model_info['embedding_dim']
        self.device = 'cpu'
        self._backend = self.model_info['backend']

        display = self.model_info['display_name']
        print(f"[SpeakerExtractor] Model: {display} (dim={self.embedding_dim})")
        print(f"[SpeakerExtractor] Backend: {self._backend}")
        print(f"[SpeakerExtractor] Running on CPU (avoids device/dtype conflicts)")

        if self._backend == 'speechbrain':
            self._init_speechbrain(local_path)
        elif self._backend == 'clova':
            self._init_clova(local_path)
        elif self._backend == 'redimnet':
            self._init_redimnet(local_path)
        elif self._backend == 'nemo':
            self._init_nemo(local_path)

    def _init_speechbrain(self, local_path):
        """Initialize SpeechBrain-based model (xvector or ecapa)."""
        run_opts_device = 'cpu'
        display = self.model_info['display_name']

        print(f"[SpeakerExtractor] HuggingFace source: {self.model_info['source']}")

        if local_path and os.path.exists(local_path):
            print(f"[SpeakerExtractor] Loading from local path: {local_path}")
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source=local_path,
                    savedir=local_path,
                    run_opts={"device": run_opts_device}
                )
                print(f"[SpeakerExtractor] Loaded from local path!")
                local_path = "loaded"
            except Exception as e:
                print(f"[SpeakerExtractor] Failed: {e}. Falling back to HuggingFace...")
                local_path = None

        if not local_path:
            print(f"[SpeakerExtractor] Downloading from HuggingFace...")
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_info['source'],
                    savedir=self.model_info['savedir'],
                    run_opts={"device": run_opts_device}
                )
                print(f"[SpeakerExtractor] Model loaded! Cached at: {self.model_info['savedir']}")
            except Exception as e:
                print(f"\n{'=' * 80}")
                print(f"[ERROR] Failed to download {display} model!")
                print(f"{'=' * 80}")
                print(f"Error: {type(e).__name__}: {str(e)[:200]}")
                print(f"{'=' * 80}\n")
                raise RuntimeError(f"{display} model download failed.") from e

        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        self.classifier = self.classifier.to('cpu').float()

        def move_to_cpu_float32(module):
            module.to('cpu')
            module.float()
            for param in module._parameters.values():
                if param is not None:
                    param.data = param.data.to('cpu').float()
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to('cpu').float()
            for buffer in module._buffers.values():
                if buffer is not None:
                    buffer.data = buffer.data.to('cpu').float()
            for child in module.children():
                move_to_cpu_float32(child)

        move_to_cpu_float32(self.classifier)

        sample_param = next(self.classifier.parameters())
        print(f"[SpeakerExtractor] Device: {sample_param.device}, dtype: {sample_param.dtype}")
        print(f"[SpeakerExtractor] Ready!")

    def _init_clova(self, local_path):
        """Initialize Clova ResNet speaker encoder from local checkpoint."""
        if not local_path or not os.path.exists(local_path):
            raise ValueError(
                f"Clova model requires 'speaker_model_local_path' pointing to the directory "
                f"containing model_se.pth and config_se.json. Got: {local_path}"
            )

        model_path = os.path.join(local_path, 'model_se.pth')
        config_path = os.path.join(local_path, 'config_se.json')

        if not os.path.exists(model_path):
            # Try alternate name
            alt = os.path.join(local_path, 'model_se.pth.tar')
            if os.path.exists(alt):
                model_path = alt
            else:
                raise FileNotFoundError(f"Clova model not found: {model_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Clova config not found: {config_path}")

        print(f"[SpeakerExtractor] Loading Clova config: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        audio_config = config.get('audio', {})
        model_params = config.get('model_params', {})

        print(f"[SpeakerExtractor] Loading Clova checkpoint: {model_path}")
        self.encoder = ClovaResNetEncoder(
            input_dim=model_params.get('input_dim', 64),
            proj_dim=model_params.get('proj_dim', 512),
            audio_config=audio_config,
        )

        state = torch.load(model_path, map_location='cpu')
        if 'model' in state:
            self.encoder.load_state_dict(state['model'])
        else:
            self.encoder.load_state_dict(state)

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.encoder = self.encoder.to('cpu').float()

        sample_param = next(self.encoder.parameters())
        print(f"[SpeakerExtractor] Device: {sample_param.device}, dtype: {sample_param.dtype}")
        print(f"[SpeakerExtractor] Ready!")

    def _init_nemo(self, local_path):
        """Initialize NeMo speaker model (TitaNet or SpeakerNet)."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "nemo_toolkit[asr] is required for TitaNet/SpeakerNet models. "
                "Install with: pip install 'nemo_toolkit[asr]'"
            )

        nemo_model_name = self.model_info['nemo_model_name']
        display = self.model_info['display_name']

        # Allow local .nemo file via local_path
        if local_path and os.path.isfile(local_path) and local_path.endswith('.nemo'):
            print(f"[SpeakerExtractor] Loading NeMo model from local file: {local_path}")
            self.nemo_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(local_path)
        else:
            print(f"[SpeakerExtractor] Loading NeMo pretrained: {nemo_model_name}")
            try:
                self.nemo_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(nemo_model_name)
            except Exception as e:
                print(f"\n{'=' * 80}")
                print(f"[ERROR] Failed to load {display} model!")
                print(f"{'=' * 80}")
                print(f"Error: {type(e).__name__}: {str(e)[:200]}")
                print(f"\nRequires: pip install 'nemo_toolkit[asr]'")
                print(f"{'=' * 80}\n")
                raise RuntimeError(f"{display} model load failed.") from e

        for param in self.nemo_model.parameters():
            param.requires_grad = False
        self.nemo_model.eval()
        self.nemo_model = self.nemo_model.to('cpu')

        sample_param = next(self.nemo_model.parameters())
        print(f"[SpeakerExtractor] Device: {sample_param.device}, dtype: {sample_param.dtype}")
        print(f"[SpeakerExtractor] Ready!")

    def _init_redimnet(self, local_path):
        """Initialize ReDimNet model via torch.hub."""
        info = self.model_info
        model_name = info.get('default_model_name', 'b2')
        train_type = info.get('default_train_type', 'ft_lm')
        dataset = info.get('default_dataset', 'vox2')

        # Allow overriding via local_path as "model_name:train_type:dataset"
        if local_path and ':' in str(local_path):
            parts = str(local_path).split(':')
            if len(parts) >= 1:
                model_name = parts[0]
            if len(parts) >= 2:
                train_type = parts[1]
            if len(parts) >= 3:
                dataset = parts[2]

        print(f"[SpeakerExtractor] ReDimNet variant: {model_name}, train_type={train_type}, dataset={dataset}")
        print(f"[SpeakerExtractor] Loading via torch.hub (IDRnD/ReDimNet)...")

        try:
            self.redimnet_model = torch.hub.load(
                'IDRnD/ReDimNet',
                'ReDimNet',
                model_name=model_name,
                train_type=train_type,
                dataset=dataset,
            )
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"[ERROR] Failed to load ReDimNet model!")
            print(f"{'=' * 80}")
            print(f"Error: {type(e).__name__}: {str(e)[:200]}")
            print(f"\nSOLUTION: Ensure internet access for first download.")
            print(f"Available variants: b0, b1, b2, b3, b4, b5, b6, S, M")
            print(f"Train types: ptn, ft_lm, ft_mix")
            print(f"Datasets: vox2, vb2, vb2+vox2+cnc")
            print(f"{'=' * 80}\n")
            raise RuntimeError("ReDimNet model load failed.") from e

        for param in self.redimnet_model.parameters():
            param.requires_grad = False
        self.redimnet_model.eval()
        self.redimnet_model = self.redimnet_model.to('cpu').float()

        sample_param = next(self.redimnet_model.parameters())
        print(f"[SpeakerExtractor] Device: {sample_param.device}, dtype: {sample_param.dtype}")
        print(f"[SpeakerExtractor] Ready!")

    def extract_embeddings(self, waveform, sample_rate=16000):
        """
        Extract speaker embeddings from audio waveform.

        Args:
            waveform (torch.Tensor): Shape (batch, samples) or (batch, 1, samples)
            sample_rate (int): Sample rate (default: 16000)

        Returns:
            embeddings (torch.Tensor): Shape (batch, embedding_dim)
        """
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            original_device = waveform.device

            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)

            waveform_cpu = waveform.float().cpu()

            max_val = waveform_cpu.abs().max()
            if max_val > 1.0:
                waveform_cpu = waveform_cpu / max_val

            if self._backend == 'speechbrain':
                model_device = next(self.classifier.parameters()).device
                if model_device.type != 'cpu':
                    self.classifier = self.classifier.to('cpu')
                embeddings = self.classifier.encode_batch(waveform_cpu)
            elif self._backend == 'clova':
                # Force encoder back to CPU (Lightning may have moved it to CUDA)
                self.encoder = self.encoder.to('cpu').float()
                # Process each item with multi-crop averaging
                batch_embeddings = []
                for i in range(waveform_cpu.shape[0]):
                    emb = self.encoder.compute_embedding(waveform_cpu[i:i+1])
                    batch_embeddings.append(emb)
                embeddings = torch.cat(batch_embeddings, dim=0)
            elif self._backend == 'redimnet':
                # Force model back to CPU (Lightning may have moved it to CUDA)
                self.redimnet_model = self.redimnet_model.to('cpu').float()
                embeddings = self.redimnet_model(waveform_cpu)
            elif self._backend == 'nemo':
                # Force model back to CPU (Lightning may have moved it to CUDA)
                self.nemo_model = self.nemo_model.to('cpu').float()
                lengths = torch.tensor([waveform_cpu.shape[1]] * waveform_cpu.shape[0])
                _, embeddings = self.nemo_model(
                    input_signal=waveform_cpu,
                    input_signal_length=lengths
                )

            embeddings = embeddings.to(original_device).float()

        return embeddings
