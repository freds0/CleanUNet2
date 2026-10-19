"""
Metric discriminator (MetricGAN+ / CMGAN / MP-SENet style, IMPROVEMENT_PLAN.md, W2): a small CNN that predicts the
normalized PESQ-WB of an estimate from the compressed magnitudes of the clean target and of the estimate. It is
trained to output 1 for (clean, clean) and the measured score for (clean, estimate); the generator gets
weight x (D(clean, estimate) - 1)^2, a differentiable push towards a higher PESQ.

With highpass_hz > 0 the target is the band-limited PESQ (both signals high-passed, evaluation.metrics.highpass) and
the discriminator sees the high-passed signals, so the metric loss leaves the band below highpass_hz to the other
losses: PESQ-WB still weighs 50-100 Hz, and half of the gain of the plain metric loss sat there
(IMPROVEMENT_PLAN.md, 14.7.2).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def pesq_norm(ref: np.ndarray, est: np.ndarray, sample_rate: int = 16000, highpass_hz: float = 0.0) -> float:
    """(PESQ-WB - 1) / 3.5, or NaN when PESQ cannot score the pair (e.g. a crop without speech). With highpass_hz,
    both signals are high-passed first."""
    from pesq import pesq
    try:
        if highpass_hz:
            from evaluation.metrics import highpass
            ref, est = highpass(ref, highpass_hz, sample_rate), highpass(est, highpass_hz, sample_rate)
        return (float(pesq(sample_rate, ref, est, "wb")) - 1.0) / 3.5
    except Exception:      # NoUtterancesError, BufferTooShortError, ...
        return float("nan")


def _pesq_norm_star(args):
    return pesq_norm(*args)


def batch_pesq(ref: torch.Tensor, est: torch.Tensor, pool=None, sample_rate: int = 16000,
               highpass_hz: float = 0.0) -> torch.Tensor:
    """Normalized PESQ-WB of every (ref, est) row of two (B, L) / (B, 1, L) tensors; NaN where it fails.
    PESQ holds the GIL, so `pool` is a process pool (multiprocessing.Pool) or None (serial)."""
    r = ref.detach().float().reshape(ref.shape[0], -1).cpu().numpy().astype(np.float64)
    e = est.detach().float().reshape(est.shape[0], -1).cpu().numpy().astype(np.float64)
    jobs = [(r[i], e[i], sample_rate, highpass_hz) for i in range(r.shape[0])]
    scores = pool.map(_pesq_norm_star, jobs) if pool is not None else [_pesq_norm_star(j) for j in jobs]
    return torch.tensor(scores, dtype=torch.float32)


class LearnableSigmoid(nn.Module):
    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def fft_highpass(x: torch.Tensor, cutoff_hz: float, sample_rate: int = 16000) -> torch.Tensor:
    """Differentiable zero-phase high-pass of (B, L) signals: the gain of a 4th-order Butterworth applied forward and
    backward (|H|^2, as in evaluation.metrics.highpass), on the FFT of the whole signal."""
    L = x.shape[-1]
    f = torch.fft.rfftfreq(L, 1.0 / sample_rate, device=x.device)
    gain = 1.0 / (1.0 + (cutoff_hz / f.clamp(min=1e-3)) ** 8)
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * gain, n=L, dim=-1)


class MetricDiscriminator(nn.Module):
    def __init__(self, channels: int = 16, n_fft: int = 400, hop_length: int = 100, power: float = 0.3,
                 highpass_hz: float = 0.0, sample_rate: int = 16000):
        super().__init__()
        self.n_fft, self.hop_length, self.power = n_fft, hop_length, power
        self.highpass_hz, self.sample_rate = float(highpass_hz), sample_rate
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        ch = channels

        def block(cin, cout):
            return [spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1, bias=False)), nn.InstanceNorm2d(cout, affine=True),
                    nn.PReLU(cout)]

        self.net = nn.Sequential(*block(2, ch), *block(ch, 2 * ch), *block(2 * ch, 4 * ch), *block(4 * ch, 8 * ch),
                                 nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                                 spectral_norm(nn.Linear(8 * ch, 4 * ch)), nn.Dropout(0.3), nn.PReLU(4 * ch),
                                 spectral_norm(nn.Linear(4 * ch, 1)), LearnableSigmoid(2.0))

    def magnitude(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) / (B, 1, L) -> compressed magnitude (B, F, T) in float32."""
        x = wav.float().reshape(wav.shape[0], -1)
        if self.highpass_hz:
            x = fft_highpass(x, self.highpass_hz, self.sample_rate)
        X = torch.stft(x, self.n_fft, self.hop_length, self.n_fft, self.window.to(x.device), return_complex=True)
        return (X.real ** 2 + X.imag ** 2 + 1e-9).sqrt() ** self.power

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """Predicted normalized PESQ (B,) of `est` against `ref` (normalized waveforms)."""
        with torch.autocast(device_type=ref.device.type, enabled=False):
            x = torch.stack([self.magnitude(ref), self.magnitude(est)], dim=1)
            return self.net(x).view(-1)


def generator_loss(disc: MetricDiscriminator, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(disc(ref, est), torch.ones(ref.shape[0], device=ref.device))


def discriminator_loss(disc: MetricDiscriminator, ref: torch.Tensor, est: torch.Tensor,
                       scores: torch.Tensor) -> torch.Tensor:
    """(D(ref, ref) - 1)^2 + (D(ref, est) - PESQ)^2, the second term over the rows with a valid score."""
    ones = torch.ones(ref.shape[0], device=ref.device)
    loss = F.mse_loss(disc(ref, ref), ones)
    ok = torch.isfinite(scores)
    if ok.any():
        idx = ok.nonzero().view(-1).to(ref.device)
        loss = loss + F.mse_loss(disc(ref[idx], est[idx].detach()), scores[ok].to(ref.device))
    return loss
