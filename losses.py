import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_loss_fn(clean_audio, denoised_audio, clean_spec, denoised_spec):
    loss_audio = F.mse_loss(denoised_audio, clean_audio)
    loss_spec = F.l1_loss(denoised_spec, clean_spec)
    return loss_audio + loss_spec


class CleanUnetLoss():
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        self.ell_p = ell_p
        self.ell_p_lambda = ell_p_lambda
        self.stft_lambda = stft_lambda
        self.mrstftloss = mrstftloss

    def __call__(self, clean_audio, denoised_audio):
        B, C, L = clean_audio.shape
        output_dic = {}
        loss = 0.0

        # Reconstruction loss (L1 or L2)
        if self.ell_p == 2:
            ae_loss = F.mse_loss(denoised_audio, clean_audio)
        elif self.ell_p == 1:
            ae_loss = F.l1_loss(denoised_audio, clean_audio)
        else:
            raise NotImplementedError(f"ell_p={self.ell_p} is not supported. Use 1 (L1) or 2 (L2).")

        loss += ae_loss * self.ell_p_lambda
        output_dic["reconstruct"] = ae_loss.detach() * self.ell_p_lambda

        # STFT-based losses
        if self.stft_lambda > 0:
            sc_loss, mag_loss = self.mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
            loss += (sc_loss + mag_loss) * self.stft_lambda
            output_dic["stft_sc"] = sc_loss.detach() * self.stft_lambda
            output_dic["stft_mag"] = mag_loss.detach() * self.stft_lambda

        return loss, output_dic


class CleanUNet2Loss:
    """Waveform loss of CleanUNet / CleanUMamba: ell_p reconstruction + multi-resolution STFT.

    (The previous version added a second L1 term on top of CleanUnetLoss, doubling the L1 weight.)
    Inputs are cast to float32 so the STFTs never run in reduced precision.
    """
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        self.cleanunet_loss = CleanUnetLoss(ell_p, ell_p_lambda, stft_lambda, mrstftloss)

    def __call__(self, clean_audio, denoised_audio, return_components=False):
        loss, components = self.cleanunet_loss(clean_audio.float(), denoised_audio.float())
        return (loss, components) if return_components else loss


def stft(x, fft_size, shift_size, win_length, window):
    window = window.to(x.device)
    x_stft = torch.stft(
        x, n_fft=fft_size, hop_length=shift_size, win_length=win_length,
        window=window, return_complex=True, center=True
    )
    return x_stft


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(torch.clamp(y_mag, min=1e-7)), torch.log(torch.clamp(x_mag, min=1e-7)))


class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", band="full"):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window).abs()
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window).abs()

        if self.band == "high":
            freq_mask_ind = x_mag.shape[1] // 2
            x_mag = x_mag[:, freq_mask_ind:, :]
            y_mag = y_mag[:, freq_mask_ind:, :]

        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        sc_lambda=0.1,
        mag_lambda=0.1,
        band="full"
    ):
        super().__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, hs, wl, window, band)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x, y):
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))
            y = y.view(-1, y.size(2))

        sc_loss = 0.0
        mag_loss = 0.0
        for stft_loss in self.stft_losses:
            sc_l, mag_l = stft_loss(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss = sc_loss * self.sc_lambda / len(self.stft_losses)
        mag_loss = mag_loss * self.mag_lambda / len(self.stft_losses)

        return sc_loss, mag_loss


class CompressedMagnitudeLoss(nn.Module):
    """MSE between power-law compressed STFT magnitudes |X|^power of an estimate and its target (the magnitude loss
    of MP-SENet / CMGAN; IMPROVEMENT_PLAN.md, W2). The compression gives low-energy bins (residual noise in pauses)
    more weight than the linear spectral convergence and less than the log magnitude of the MR-STFT loss.
    Inputs (B, L) / (B, 1, L) should be normalized by the input level; computed in float32."""

    def __init__(self, n_fft: int = 400, hop_length: int = 100, power: float = 0.3):
        super().__init__()
        self.n_fft, self.hop_length, self.power = n_fft, hop_length, power
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def magnitude(self, x):
        x = x.float().reshape(x.shape[0], -1)
        X = torch.stft(x, self.n_fft, self.hop_length, self.n_fft, self.window.to(x.device), return_complex=True)
        return (X.real ** 2 + X.imag ** 2 + 1e-9).sqrt() ** self.power

    def forward(self, est, ref):
        with torch.autocast(device_type=est.device.type, enabled=False):
            return F.mse_loss(self.magnitude(est), self.magnitude(ref))


class AntiWrappingPhaseLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, window="hann_window", eps=1e-8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.eps = eps

    def forward(self, est_wave, clean_wave):
        """
        Args:
            est_wave: (B, T) or (B, 1, T) - estimated audio
            clean_wave: (B, T) or (B, 1, T) - reference audio
        """
        # Ensure (B, T) shape and float32 precision
        if est_wave.dim() == 3: est_wave = est_wave.squeeze(1)
        if clean_wave.dim() == 3: clean_wave = clean_wave.squeeze(1)
        est_wave, clean_wave = est_wave.float(), clean_wave.float()

        # 1. STFT
        est_stft = torch.stft(est_wave, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)
        clean_stft = torch.stft(clean_wave, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)

        # 2. Magnitude and phase
        # clean_mag weights the loss (focus on regions with energy)
        clean_mag = clean_stft.abs()
        
        est_angle = est_stft.angle()
        clean_angle = clean_stft.angle()

        # 3. Anti-Wrapping Loss (1 - cos(delta))
        # Equal phases: cos(0)=1 -> loss=0
        # Opposite phases: cos(pi)=-1 -> loss=2
        delta_phase = est_angle - clean_angle
        phase_loss_raw = 1 - torch.cos(delta_phase)

        # 4. Magnitude weighting
        # Avoids fitting the phase of silence / low-energy bins
        weighted_loss = phase_loss_raw * clean_mag

        # Normalize by the total magnitude to keep the scale
        loss = torch.sum(weighted_loss) / (torch.sum(clean_mag) + self.eps)

        return loss

# Alternative to AntiWrappingPhaseLoss
class ComplexL1Loss(nn.Module):
    def forward(self, est_wave, clean_wave):
        est_stft = torch.stft(est_wave, n_fft=1024, return_complex=True)
        clean_stft = torch.stft(clean_wave, n_fft=1024, return_complex=True)
        
        # Difference in the complex plane (affects magnitude and phase)
        return (est_stft - clean_stft).abs().mean()


# ---------------------------------------------------------------------------------------------
# Phase-domain losses for spectral branches with a phase output (IMPROVEMENT_PLAN.md, N4)
# ---------------------------------------------------------------------------------------------
def anti_wrapping(x: torch.Tensor) -> torch.Tensor:
    """|x - 2*pi*round(x / (2*pi))|: distance between phases, insensitive to 2*pi wrapping."""
    return torch.abs(x - 2 * math.pi * torch.round(x / (2 * math.pi)))


def phase_losses(phase_est: torch.Tensor, phase_ref: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """MP-SENet style anti-wrapping losses on (B, F, T) phase spectra: instantaneous phase,
    group delay (difference along frequency) and instantaneous angular frequency (difference along time).
    `weight` (B, F, T), e.g. the clean compressed magnitude, focuses the loss on bins that carry speech."""
    def wmean(e, w):
        return e.mean() if w is None else (e * w).sum() / (w.sum() + 1e-8)

    w_f = None if weight is None else torch.minimum(weight[:, 1:], weight[:, :-1])
    w_t = None if weight is None else torch.minimum(weight[..., 1:], weight[..., :-1])
    ip = wmean(anti_wrapping(phase_est - phase_ref), weight)
    gd = wmean(anti_wrapping(torch.diff(phase_est, dim=1) - torch.diff(phase_ref, dim=1)), w_f)
    iaf = wmean(anti_wrapping(torch.diff(phase_est, dim=2) - torch.diff(phase_ref, dim=2)), w_t)
    return ip + gd + iaf
