import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from typing import List, Tuple, Optional
from torchaudio import load as torchaudio_load
import torchaudio.transforms as T
from librosa.filters import mel as librosa_mel_fn
import torchaudio

# Use "spawn" multiprocessing start method for dataloaders (safer for CUDA in some setups)
# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)

MAX_WAV_VALUE = 32768.0 

_mel_basis_cache = {}
_hann_window_cache = {}

def load_wav(full_path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, convert to mono, and resample if needed.
    """
    waveform, sampling_rate = torchaudio_load(full_path, normalize=True)
    
    # -------------------------------------------------------
    # CORREÇÃO: Forçar Mono
    # Se tiver mais de 1 canal (ex: stereo), faz a média
    # -------------------------------------------------------
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sampling_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sampling_rate = target_sr
        
    return waveform, sampling_rate

# ---------------------------
# Dynamic range helpers
# ---------------------------
def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(magnitudes)

# ---------------------------
# Spectrogram utility (Linear or Mel)
# ---------------------------
def get_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
    use_mel: bool = True # NOVO ARGUMENTO: Controla o tipo de espectrograma
) -> torch.Tensor:
    """
    Calcula o espectrograma (Linear ou Mel).
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Cache da janela Hann (reaproveita se o device for o mesmo)
    if str(y.device) not in _hann_window_cache:
        _hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)
    hann_window = _hann_window_cache[str(y.device)]

    # Padding com modo reflect para evitar bordas abruptas
    pad_amount = int((n_fft - hop_size) / 2)
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)

    # Calcula STFT
    spec = torch.stft(
        y_padded,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True
    )
    
    # Magnitude: sqrt(real^2 + imag^2)
    # Shape: [Batch, n_fft // 2 + 1, Time]
    mag = spec.abs()

    if use_mel:
        # Lógica Mel-Spectrograma
        device_key = f"{fmax}_{y.device}_{num_mels}"
        if device_key not in _mel_basis_cache:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            _mel_basis_cache[device_key] = torch.from_numpy(mel).float().to(y.device)
        
        mel_basis = _mel_basis_cache[device_key]
        # Aplica o filtro Mel
        spec_final = torch.matmul(mel_basis, mag)
    else:
        # Lógica Espectrograma Linear
        spec_final = mag

    # Aplica compressão dinâmica (log)
    spec_final = spectral_normalize_torch(spec_final)
    
    return spec_final.squeeze(0) if spec_final.size(0) == 1 else spec_final

# ---------------------------
# File list helper
# ---------------------------
def get_dataset_filelist(filelist_path: str) -> List[Tuple[str, str]]:
    with open(filelist_path, "r", encoding="utf-8") as ifile:
        lines = [l.strip() for l in ifile.readlines() if l.strip()]
    pairs = []
    for l in lines:
        parts = l.split("|")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs

# ---------------------------
# Collate function
# ---------------------------
def custom_collate_fn(batch):
    audios, specs, clean_audios, clean_specs = zip(*batch)
    
    def pad_list(tensors: List[torch.Tensor]):
        # Encontra o tamanho máximo no último eixo (tempo)
        max_len = max(t.shape[-1] for t in tensors)
        # Pad
        padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[-1])) for t in tensors]
        return torch.stack(padded)

    # Tenta empilhar se os tamanhos forem iguais
    try:
        audios_stacked = torch.stack(audios)
        specs_stacked = torch.stack(specs)
        clean_audios_stacked = torch.stack(clean_audios)
        clean_specs_stacked = torch.stack(clean_specs)
    except RuntimeError:
        # Se tamanhos diferentes, faz padding
        audios_stacked = pad_list(audios)
        specs_stacked = pad_list(specs)
        clean_audios_stacked = pad_list(clean_audios)
        clean_specs_stacked = pad_list(clean_specs)

    return audios_stacked, specs_stacked, clean_audios_stacked, clean_specs_stacked

# ---------------------------
# Dataset
# ---------------------------
class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        data_files: str,
        segment_size: int = 8192,
        n_fft: int = 1024,
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
        augmentations = None,
        use_mel_spec: bool = True, # NOVO PARÂMETRO (Default True para manter compatibilidade)
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.audio_files = get_dataset_filelist(data_files)
        
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
        
        self.cached_wav = None
        self.cached_wav_input = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.use_mel_spec = use_mel_spec # Salva a configuração

    def __getitem__(self, index: int):
        clean_rel, noisy_rel = self.audio_files[index]
        clean_path = os.path.join(self.data_dir, clean_rel)
        noisy_path = os.path.join(self.data_dir, noisy_rel)

        try:
            clean_audio, clean_sr = load_wav(clean_path, self.sampling_rate)
            noisy_audio, noisy_sr = load_wav(noisy_path, self.sampling_rate)

            # Normalização de pico
            clean_audio = clean_audio / (clean_audio.abs().max() + 1e-9)
            noisy_audio = noisy_audio / (noisy_audio.abs().max() + 1e-9)

        except Exception as e:
            print(f"Error processing file {clean_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.audio_files)-1))

        # Corte de segmentos
        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                if audio_start % 2 != 0:
                    audio_start = audio_start - 1 if audio_start > 0 else 0
                audio_end = audio_start + self.segment_size
                clean_audio = clean_audio[:, audio_start:audio_end]
                noisy_audio = noisy_audio[:, audio_start:audio_end]
            else:
                pad_len = self.segment_size - clean_audio.size(1)
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_len))
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_len))

        # Geração do espectrograma (Linear ou Mel, dependendo de self.use_mel_spec)
        noisy_spec = get_spectrogram(
            noisy_audio, self.n_fft, self.num_mels, self.sampling_rate, 
            self.hop_size, self.win_size, self.fmin, self.fmax, 
            use_mel=self.use_mel_spec
        )
        clean_spec = get_spectrogram(
            clean_audio, self.n_fft, self.num_mels, self.sampling_rate, 
            self.hop_size, self.win_size, self.fmin, self.fmax, 
            use_mel=self.use_mel_spec
        )

        return noisy_audio, noisy_spec.squeeze(0), clean_audio, clean_spec.squeeze(0)

    def __len__(self) -> int:
        return len(self.audio_files)