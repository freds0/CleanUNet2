import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


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
        output_dic["reconstruct"] = ae_loss.item() * self.ell_p_lambda

        # STFT-based losses
        if self.stft_lambda > 0:
            sc_loss, mag_loss = self.mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
            loss += (sc_loss + mag_loss) * self.stft_lambda
            output_dic["stft_sc"] = sc_loss.item() * self.stft_lambda
            output_dic["stft_mag"] = mag_loss.item() * self.stft_lambda

        return loss, output_dic


class CleanUNet2Loss:
    def __init__(self, ell_p, ell_p_lambda, stft_lambda, mrstftloss, **kwargs):
        self.cleanunet_loss = CleanUnetLoss(ell_p, ell_p_lambda, stft_lambda, mrstftloss)

    def __call__(self, clean_audio, denoised_audio):
        loss_cleanunet, _ = self.cleanunet_loss(clean_audio, denoised_audio)

        # ⚠️ Remover esta linha se já estiver usando L1 na CleanUnetLoss
        loss_l1 = F.l1_loss(clean_audio, denoised_audio, reduction='mean')

        return loss_cleanunet + loss_l1  # ou apenas `return loss_cleanunet`


def stft(x, fft_size, shift_size, win_length, window):
    window = window.to(x.device)
    x_stft = torch.stft(
        x, n_fft=fft_size, hop_length=shift_size, win_length=win_length,
        window=window, return_complex=True, center=True
    )
    return x_stft


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        eps = 1e-9
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + eps)


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
            est_wave: (B, T) ou (B, 1, T) - Áudio Estimado
            clean_wave: (B, T) ou (B, 1, T) - Áudio Real
        """
        # Garante dimensão correta
        if est_wave.dim() == 3: est_wave = est_wave.squeeze(1)
        if clean_wave.dim() == 3: clean_wave = clean_wave.squeeze(1)

        # 1. Calcula STFT
        est_stft = torch.stft(est_wave, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)
        clean_stft = torch.stft(clean_wave, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)

        # 2. Extrai Magnitude e Fase (Ângulo)
        # Nota: Usamos clean_mag para ponderar a loss (Focus on regions with energy)
        clean_mag = clean_stft.abs()
        
        est_angle = est_stft.angle()
        clean_angle = clean_stft.angle()

        # 3. Anti-Wrapping Loss (1 - cos(delta))
        # Se as fases forem iguais, cos(0)=1 -> Loss=0
        # Se forem opostas, cos(pi)=-1 -> Loss=2
        delta_phase = est_angle - clean_angle
        phase_loss_raw = 1 - torch.cos(delta_phase)

        # 4. Ponderação pela Magnitude (Importante!)
        # Evita que o modelo tente alinhar fase de ruído/silêncio
        weighted_loss = phase_loss_raw * clean_mag

        # Normaliza pela soma das magnitudes para manter a escala
        loss = torch.sum(weighted_loss) / (torch.sum(clean_mag) + self.eps)

        return loss

# Alternative to AntiWrappingPhaseLoss
class ComplexL1Loss(nn.Module):
    def forward(self, est_wave, clean_wave):
        est_stft = torch.stft(est_wave, n_fft=1024, return_complex=True)
        clean_stft = torch.stft(clean_wave, n_fft=1024, return_complex=True)
        
        # Diferença direta no plano complexo (afeta Mag e Fase)
        return (est_stft - clean_stft).abs().mean()



def feature_loss(fmap_r, fmap_g):
    """
    Calculates the L1 distance between feature maps of real and generated audio.
    Source: HiFi-GAN code.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Calculates the Least Squares (MSE) Loss for the Discriminator.
    Real -> 1, Generated -> 0.
    Source: HiFi-GAN code.
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    """
    Calculates the Least Squares (MSE) Loss for the Generator.
    Generated -> 1 (trying to fool D).
    Source: HiFi-GAN code.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l.item()) # .item() for logging purposes usually, but here it stays inside the tensor graph if used for backward, but here it appends to list. In original code it appends tensor. Careful with detach.
        loss += l

    return loss, gen_losses