"""
Clean/noisy pair dataset for speech denoising.

- Audio is read with soundfile (torchaudio.load is deprecated since torchaudio 2.8).
- Clean and noisy signals are scaled by the SAME factor (peak of the noisy signal), so the
  gain relation between input and target is preserved. The model itself normalizes by std.
- Training: random crops of `segment_size` samples (zero-padded if the file is shorter).
  With `concat_utterances` = n > 1, each example is the requested file followed by n-1 random files
  (each with its own gain normalization), joined with a short linear crossfade applied identically to
  clean and noisy, so long crops contain real speech instead of padding (IMPROVEMENT_PLAN.md, N7a).
- Validation / test: whole utterances (use batch_size=1).
- Spectrograms are no longer computed here: the model owns the STFT (see cleanunet/spectral.py).
- `ext_noise` (training only, noise_bank.py): with probability p the noise of a crop is replaced by an excerpt of
  another noise corpus at an SNR drawn uniformly from `snr_db` (P.56 speech level); the crop is then rescaled so
  that its mixture peaks at 1, like the other examples. `random_sign: true` gives every babble talker and noise
  excerpt a random polarity (noise_bank.py).
"""

import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torchaudio.functional as AF

from noise_bank import NoiseBank, mix_at_snr


def get_dataset_filelist(filelist_path: str) -> List[Tuple[str, str]]:
    """Read a filelist where each line is 'clean_path|noisy_path'."""
    pairs = []
    with open(filelist_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                raise ValueError(f"Invalid line in filelist (expected 'clean|noisy'): {line}")
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def load_audio(path: str, target_sr: int, retries: int = 3) -> torch.Tensor:
    """Load a mono float32 waveform (L,), resampling if needed.

    A failed read is retried (a transient libsndfile error once stopped a training run), then reported
    as a RuntimeError naming the file: soundfile's own exception loses its message when a DataLoader
    worker re-raises it in the main process."""
    for attempt in range(retries):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"cannot read '{path}': {type(exc).__name__}: {exc}") from None
            time.sleep(0.5)
    wav = torch.from_numpy(data.mean(axis=1))
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav


class PairDataset(torch.utils.data.Dataset):
    """Yields (noisy (1, L), clean (1, L), file_id)."""

    def __init__(self, data_dir: str, filelist: str, segment_size: Optional[int] = None,
                 sample_rate: int = 16000, max_files: Optional[int] = None, concat_utterances: int = 1,
                 crossfade: int = 160, ext_noise: Optional[dict] = None):
        super().__init__()
        if concat_utterances > 1 and segment_size is None:
            raise ValueError("concat_utterances > 1 requires a segment_size (training crops)")
        self.concat_utterances = concat_utterances
        self.crossfade = crossfade
        self.data_dir = data_dir
        self.files = get_dataset_filelist(filelist)
        if max_files is not None:
            self.files = self.files[:max_files]
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        ext_noise = dict(ext_noise or {})
        self.ext_p = float(ext_noise.get("p", 0.0))
        self.noise_bank = None
        if self.ext_p > 0:
            if segment_size is None:
                raise ValueError("ext_noise applies to training crops (it needs a segment_size)")
            self.noise_bank = NoiseBank(ext_noise["dir"], weights=ext_noise.get("weights"),
                                        babble_talkers=ext_noise.get("babble_talkers", (3, 7)),
                                        sample_rate=sample_rate, random_sign=ext_noise.get("random_sign", False))
            self.ext_snr = tuple(float(v) for v in ext_noise.get("snr_db", (-5.0, 20.0)))

    def __len__(self) -> int:
        """Number of (clean, noisy) pairs in the file list."""
        return len(self.files)

    def _load_pair(self, index: int):
        clean_rel, noisy_rel = self.files[index]
        clean = load_audio(os.path.join(self.data_dir, clean_rel), self.sample_rate)
        noisy = load_audio(os.path.join(self.data_dir, noisy_rel), self.sample_rate)
        L = min(len(clean), len(noisy))
        clean, noisy = clean[:L], noisy[:L]
        # Same gain for both signals (keeps the clean/noisy relation intact)
        gain = 1.0 / (noisy.abs().max() + 1e-8)
        return clean * gain, noisy * gain

    def _join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate with a linear crossfade of `self.crossfade` samples."""
        n = min(self.crossfade, len(a), len(b))
        if n == 0:
            return torch.cat([a, b])
        ramp = torch.linspace(0.0, 1.0, n + 2)[1:-1]
        return torch.cat([a[:-n], a[-n:] * (1 - ramp) + b[:n] * ramp, b[n:]])

    def __getitem__(self, index: int):
        """(noisy, clean, id), signals of shape (1, L).

        Training: optionally concatenates utterances, then takes a random crop of ``segment_size`` samples (zero-padded
        when shorter). With probability ``ext_p`` the noisy crop is replaced by the clean crop plus an external noise at a
        random SNR, and both are peak-normalized. Validation (``segment_size=None``): the whole utterance.
        """
        clean_rel, _ = self.files[index]
        clean, noisy = self._load_pair(index)
        for _ in range(self.concat_utterances - 1):
            c2, n2 = self._load_pair(random.randrange(len(self.files)))
            clean, noisy = self._join(clean, c2), self._join(noisy, n2)
        L = len(clean)

        if self.segment_size is not None:
            if L >= self.segment_size:
                start = random.randint(0, L - self.segment_size)
                clean = clean[start:start + self.segment_size]
                noisy = noisy[start:start + self.segment_size]
            else:
                clean = torch.nn.functional.pad(clean, (0, self.segment_size - L))
                noisy = torch.nn.functional.pad(noisy, (0, self.segment_size - L))

        if self.noise_bank is not None and random.random() < self.ext_p:
            noise, _ = self.noise_bank.sample(len(clean))
            mixed = mix_at_snr(clean.numpy(), noise, random.uniform(*self.ext_snr), self.sample_rate)
            gain = 1.0 / (float(np.abs(mixed).max()) + 1e-8)
            clean, noisy = clean * gain, torch.from_numpy(mixed.astype(np.float32)) * gain

        file_id = os.path.splitext(os.path.basename(clean_rel))[0]
        return noisy.unsqueeze(0), clean.unsqueeze(0), file_id


def seed_worker(worker_id: int):
    """Give every dataloader worker its own (reproducible) python/numpy seed."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
