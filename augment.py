"""
Batch-level data augmentation for speech denoising (applied on GPU), in the style of the
Demucs denoiser (Defossez et al., 2020): the noise is recovered as noise = noisy - clean,
transformed, and the noisy mixture is rebuilt as clean + noise.

  - Remix:    shuffle the noises within the batch (new clean/noise combinations).
  - BandMask: remove a random mel-scale band (band-reject) from BOTH clean and noise
              (the CleanUNet_MambaSSM version was a no-op: 0 dB gain and a mis-passed sample rate).
  - Shift:    random time shift of the clean+noise pair (same shift, keeps alignment).
  - Gain:     random global gain in dB (the model is scale-normalized, this improves robustness).
  - SNR:      optional random rescaling of the noise to change the SNR.
"""

import math
import random

import torch
import torch.nn.functional as F


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _lowpass(x: torch.Tensor, cutoff: float, sample_rate: int, taps: int = 101) -> torch.Tensor:
    """Windowed-sinc low-pass FIR filter along the last axis (zero phase via 'same' padding)."""
    fc = cutoff / sample_rate
    n = torch.arange(taps, device=x.device, dtype=x.dtype) - (taps - 1) / 2
    h = 2 * fc * torch.sinc(2 * fc * n) * torch.hann_window(taps, periodic=False, device=x.device, dtype=x.dtype)
    h = h / h.sum()
    shape = x.shape
    y = F.conv1d(x.reshape(-1, 1, shape[-1]), h.view(1, 1, -1), padding=taps // 2)
    return y.reshape(shape)


class Augmenter(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000, remix_p: float = 0.0, bandmask_p: float = 0.0,
                 bandmask_maxwidth: float = 0.2, bandmask_bands: int = 120, shift_p: float = 0.0,
                 shift_max: int = 8000, gain_p: float = 0.0, gain_db: float = 6.0,
                 snr_p: float = 0.0, snr_db: float = 5.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.remix_p, self.bandmask_p, self.shift_p = remix_p, bandmask_p, shift_p
        self.bandmask_maxwidth, self.bandmask_bands = bandmask_maxwidth, bandmask_bands
        self.shift_max = shift_max
        self.gain_p, self.gain_db = gain_p, gain_db
        self.snr_p, self.snr_db = snr_p, snr_db
        mels = torch.linspace(_hz_to_mel(40.0), _hz_to_mel(sample_rate / 2), bandmask_bands)
        self.band_edges_hz = [_mel_to_hz(float(m)) for m in mels]

    @property
    def enabled(self) -> bool:
        return any(p > 0 for p in (self.remix_p, self.bandmask_p, self.shift_p, self.gain_p, self.snr_p))

    def _bandmask(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, L) clean+noise stacked; removes one random band per example."""
        out = x.clone()
        for b in range(x.shape[0]):
            if random.random() >= self.bandmask_p:
                continue
            width = max(1, min(self.bandmask_bands - 1, int(self.bandmask_maxwidth * self.bandmask_bands)))
            low = random.randrange(self.bandmask_bands - width)
            high = random.randint(low + 1, low + width)
            f_low, f_high = self.band_edges_hz[low], min(self.band_edges_hz[high], self.sample_rate / 2 - 1)
            band = _lowpass(x[b], f_high, self.sample_rate) - _lowpass(x[b], f_low, self.sample_rate)
            out[b] = x[b] - band
        return out

    @torch.no_grad()
    def forward(self, noisy: torch.Tensor, clean: torch.Tensor):
        """noisy, clean: (B, 1, L) -> augmented (noisy, clean)."""
        if not self.enabled:
            return noisy, clean
        noise = noisy - clean
        B = noisy.shape[0]

        if self.remix_p > 0 and B > 1 and random.random() < self.remix_p:
            noise = noise[torch.randperm(B, device=noise.device)]

        if self.snr_p > 0:
            mask = (torch.rand(B, 1, 1, device=noise.device) < self.snr_p).float()
            db = (torch.rand(B, 1, 1, device=noise.device) * 2 - 1) * self.snr_db
            noise = noise * (1 + mask * (10 ** (db / 20) - 1))

        if self.bandmask_p > 0:
            stacked = self._bandmask(torch.cat([clean, noise], dim=1))
            clean, noise = stacked[:, :1], stacked[:, 1:]

        if self.shift_p > 0 and random.random() < self.shift_p:
            shift = random.randint(1, self.shift_max)
            L = clean.shape[-1]
            pair = torch.cat([clean, noise], dim=1)
            pair = F.pad(pair, (shift, 0))[..., :L]
            clean, noise = pair[:, :1], pair[:, 1:]

        noisy = clean + noise
        if self.gain_p > 0:
            mask = (torch.rand(B, 1, 1, device=noisy.device) < self.gain_p).float()
            db = (torch.rand(B, 1, 1, device=noisy.device) * 2 - 1) * self.gain_db
            g = 1 + mask * (10 ** (db / 20) - 1)
            noisy, clean = noisy * g, clean * g
        return noisy, clean
