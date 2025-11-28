import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from typing import List, Tuple, Optional
from librosa.util import normalize
from torchaudio import load as torchaudio_load
import torchaudio.transforms as T
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torch.multiprocessing as mp

# Use "spawn" multiprocessing start method for dataloaders (safer for CUDA in some setups)
mp.set_start_method("spawn", force=True)

MAX_WAV_VALUE = 32768.0  # legacy constant (if needed)

# Caches for mel basis and hann window per device/fmax to avoid recomputing
_mel_basis_cache = {}
_hann_window_cache = {}


def load_wav(full_path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file using torchaudio and resample if needed.

    Returns:
        waveform: Tensor shape (channels, samples) with float values in [-1, 1] (because normalize=True)
        sampling_rate: int (Hz)
    """
    waveform, sampling_rate = torchaudio_load(full_path, normalize=True)  # waveform: (channels, samples)
    if sampling_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sampling_rate = target_sr
    return waveform, sampling_rate


# ---------------------------
# Dynamic range helpers
# ---------------------------
def dynamic_range_compression(x: np.ndarray, C: float = 1.0, clip_val: float = 1e-5) -> np.ndarray:
    """NumPy version of log compression: log(max(x, clip_val) * C)."""
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x: np.ndarray, C: float = 1.0) -> np.ndarray:
    """NumPy inverse of dynamic_range_compression."""
    return np.exp(x) / C


def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    """Torch version of dynamic range compression for tensors."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    """Torch version of dynamic range decompression (inverse of compression)."""
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Apply dynamic-range compression to spectrogram magnitudes (torch)."""
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Inverse operation of spectral_normalize_torch."""
    return dynamic_range_decompression_torch(magnitudes)


# ---------------------------
# Mel-spectrogram utility
# ---------------------------
def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False
) -> torch.Tensor:
    """
    Compute a mel-scaled magnitude spectrogram with caching for mel basis and hann-window.
    - y: (B, T) or (T,)  (expects float tensor in [-1, 1])
    - returns: mel-spectrogram (B, n_mels, time) or (n_mels, time) if input is 1D
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)  # (1, T)

    if torch.min(y) < -1.0:
        print("Warning: input waveform has min < -1.0")
    if torch.max(y) > 1.0:
        print("Warning: input waveform has max > 1.0")

    device_key = f"{fmax}_{y.device}"
    # Create mel basis for this fmax/device if not present
    if device_key not in _mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[device_key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)

    mel_basis = _mel_basis_cache[device_key]
    hann_window = _hann_window_cache[str(y.device)]

    # Pad and compute STFT
    # The original code pads by (n_fft - hop) // 2 on both sides using reflect
    pad_amount = int((n_fft - hop_size) / 2)
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)

    # STFT: return_complex=False -> shape (B, freq*2, frames, 2) in some versions; here we request return_complex=False and compute magnitude similar to original
    spec = torch.stft(
        y_padded,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=False
    )
    # spec shape when return_complex=False: (..., 2) last dim = real/imag
    # compute magnitude: sqrt(real^2 + imag^2)
    if spec.dim() == 4 and spec.size(-1) == 2:
        mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)  # (B, F, T)
    else:
        # If torch returns complex tensor (newer versions), handle separately
        mag = spec.abs()

    # Apply mel filterbank: mel_basis @ (F, T) -> (n_mels, T)
    # mag currently (B, F, T)
    mel_spec = torch.matmul(mel_basis, mag)
    mel_spec = spectral_normalize_torch(mel_spec)
    return mel_spec.squeeze(0) if mel_spec.size(0) == 1 else mel_spec


# ---------------------------
# File list helper
# ---------------------------
def get_dataset_filelist(filelist_path: str) -> List[Tuple[str, str]]:
    """
    Read a filelist where each line is 'clean_path|noisy_path' and return list of tuples.
    """
    with open(filelist_path, "r", encoding="utf-8") as ifile:
        lines = [l.strip() for l in ifile.readlines() if l.strip()]
    pairs = []
    for l in lines:
        parts = l.split("|")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
        else:
            raise ValueError(f"Invalid line in filelist (expected 'clean|noisy'): {l}")
    return pairs


# ---------------------------
# Collate function
# ---------------------------
def custom_collate_fn(batch):
    """
    Collate function for DataLoader.
    Expects batch items like (audio, spec, clean_audio, clean_spec) where audio/spec tensors could already be
    padded to fixed length. If variable-length sequences are expected, replace this with padding logic.
    """
    audios, specs, clean_audios, clean_specs = zip(*batch)
    # Try stacking directly (fast path). If shapes mismatch, fall back to padding.
    try:
        audios_stacked = torch.stack(audios)           # [B, T]
        specs_stacked = torch.stack(specs)             # [B, F, T]
        clean_audios_stacked = torch.stack(clean_audios)
        clean_specs_stacked = torch.stack(clean_specs)
    except RuntimeError:
        # Fall back to padding on time dimension (assumes dims: [T] or [1, T])
        def pad_list(tensors: List[torch.Tensor], dim=-1):
            # Ensure 2D or 3D tensors; pad along last dim
            shapes = [t.shape for t in tensors]
            max_len = max(s[-1] for s in shapes)
            padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[-1])) for t in tensors]
            return torch.stack(padded)
        audios_stacked = pad_list([a.squeeze() for a in audios])
        specs_stacked = pad_list([s for s in specs])
        clean_audios_stacked = pad_list([c.squeeze() for c in clean_audios])
        clean_specs_stacked = pad_list([cs for cs in clean_specs])

    return audios_stacked, specs_stacked, clean_audios_stacked, clean_specs_stacked


# ---------------------------
# Dataset
# ---------------------------
class MelDataset(torch.utils.data.Dataset):
    """
    Dataset that yields pairs (noisy_audio, noisy_spec, clean_audio, clean_spec).
    - data_dir: base directory where file paths in data_files are relative to.
    - data_files: path to text file with lines "clean_path|noisy_path".
    - segment_size: number of audio samples to crop (if split=True).
    - spectrogram_fn: torchaudio transform used to compute spectrograms (magnitude).
    """
    def __init__(
        self,
        data_dir: str,
        data_files: str,
        segment_size: int = 8192,
        n_fft: int = 1025,
        num_mels: int = 80,
        hop_size: int = 256,
        win_size: int = 1024,
        sampling_rate: int = 16000,
        fmin: int = 0,
        fmax: int = 8000,
        split: bool = True,
        shuffle: bool = True,
        n_cache_reuse: int = 1,
        device: Optional[torch.device] = None,
        fmax_loss: Optional[int] = None,
        noise_addition: bool = False,
        augmentations = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.audio_files = get_dataset_filelist(data_files)  # list[(clean_rel, noisy_rel)]

        # Deterministic shuffling seed for reproducibility
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.noise_addition = noise_addition

        # Placeholder for optional audio augmentation pipeline (user can pass augmentations)
        # self.audio_augmenter = AudioAugmenter(augmentations) if augmentations else None

        # Simple caching (reuse last loaded wav for a few iterations)
        self.cached_wav = None
        self.cached_wav_input = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

        # A spectrogram transform (power=1.0 gives magnitude)
        self.spectrogram_fn = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_size,
                                           win_length=self.win_size, power=1.0, normalized=True, center=False)

    def __getitem__(self, index: int):
        """
        Return a tuple:
            (noisy_audio_tensor, noisy_spec_tensor, clean_audio_tensor, clean_spec_tensor)
        Shapes:
            audio tensors -> (1, samples)
            spec tensors -> (n_freq_bins, time_frames)  (squeezed)
        """
        clean_rel, noisy_rel = self.audio_files[index]
        clean_path = os.path.join(self.data_dir, clean_rel)
        noisy_path = os.path.join(self.data_dir, noisy_rel)

        try:
            clean_audio, clean_sr = load_wav(clean_path, self.sampling_rate)
            noisy_audio, noisy_sr = load_wav(noisy_path, self.sampling_rate)
            assert clean_sr == noisy_sr, "Sampling rates do not match between clean and noisy files"

            # Normalize each audio to unit peak to avoid amplitude mismatch
            clean_audio = clean_audio / (clean_audio.abs().max() + 1e-9)
            noisy_audio = noisy_audio / (noisy_audio.abs().max() + 1e-9)

            # Cache loaded data for reuse if requested (faster IO for repeated epochs)
            self.cached_wav = clean_audio.clone()
            self.cached_wav_input = noisy_audio.clone()
            self._cache_ref_count = self.n_cache_reuse

        except Exception as e:
            # Provide a helpful error message for debugging
            filename = os.path.basename(clean_path)
            print(f"Error processing file {filename}: {e}")
            raise

        # If cache is active, reuse previously loaded audio (cheap)
        if self._cache_ref_count > 0 and self.cached_wav is not None:
            clean_audio = self.cached_wav
            noisy_audio = self.cached_wav_input
            self._cache_ref_count -= 1

        # Crop or pad to fixed segment length if requested
        if self.split:
            # Ensure we have shape (channels, samples)
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                # keep even start index (original code enforced even start)
                if audio_start % 2 != 0:
                    audio_start = audio_start - 1 if audio_start > 0 else 0
                audio_end = audio_start + self.segment_size
                clean_audio = clean_audio[:, audio_start:audio_end]
                noisy_audio = noisy_audio[:, audio_start:audio_end]
            else:
                pad_len_clean = self.segment_size - clean_audio.size(1)
                pad_len_noisy = self.segment_size - noisy_audio.size(1)
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_len_clean), "constant")
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_len_noisy), "constant")

        # Compute spectrograms (magnitude)
        # spectrogram_fn returns shape (channels, freq, time) when input shape is (channels, samples)
        noisy_spec = self.spectrogram_fn(noisy_audio).squeeze(0)  # -> (freq, time)
        clean_spec = self.spectrogram_fn(clean_audio).squeeze(0)

        # Ensure returned audio has channel dim first (1, samples)
        noisy_audio = noisy_audio.squeeze().unsqueeze(0)
        clean_audio = clean_audio.squeeze().unsqueeze(0)

        return noisy_audio, noisy_spec, clean_audio, clean_spec

    def __len__(self) -> int:
        return len(self.audio_files)

