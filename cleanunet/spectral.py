"""
Single spectrogram front-end used for training, inference and the spectral loss.

Frames are aligned with the waveform: with T = ceil(L / hop) frames, frame j is
associated with samples [j*hop, (j+1)*hop). Upsampling the frame sequence by `hop`
therefore gives T*hop >= L samples, which are cropped to L (no time stretching).

Alignment modes:
  - "center": the analysis window of frame j is centered on its hop region
              (window = [j*hop + hop/2 - n_fft/2, j*hop + hop/2 + n_fft/2)).
  - "causal": the window ends at the end of its hop region
              (window = [(j+1)*hop - n_fft, (j+1)*hop)), so no future samples are used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectrogram(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 alignment: str = "center", compression: str = "log1p", compression_power: float = 0.3):
        super().__init__()
        if alignment not in ("center", "causal"):
            raise ValueError(f"alignment must be 'center' or 'causal', got '{alignment}'")
        if compression not in ("log1p", "pow", "none"):
            raise ValueError(f"compression must be 'log1p', 'pow' or 'none', got '{compression}'")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.alignment = alignment
        self.compression = compression
        self.compression_power = compression_power
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)
        # Magnitudes are divided by the window energy (same as torchaudio's normalized=True)
        self.scale = 1.0 / float(window.pow(2).sum().sqrt())

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1

    def num_frames(self, length: int) -> int:
        return math.ceil(length / self.hop_length)

    def _pad(self, x: torch.Tensor):
        L = x.shape[-1]
        T = self.num_frames(L)
        if self.alignment == "center":
            pad_left = self.n_fft // 2 - self.hop_length // 2
        else:
            pad_left = self.n_fft - self.hop_length
        pad_right = (T - 1) * self.hop_length + self.n_fft - L - pad_left
        return F.pad(x, (pad_left, pad_right))

    def magnitude(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, L) or (B, 1, L) -> linear magnitude (B, F, T), computed in float32."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        x = self._pad(wav.float())
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window.to(x.device), center=False, return_complex=True)
        return spec.abs() * self.scale

    def complex_stft(self, wav: torch.Tensor) -> torch.Tensor:
        """Complex STFT (B, F, T) with the same framing as `magnitude` (not scaled), in float32."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        x = self._pad(wav.float())
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window.to(x.device), center=False, return_complex=True)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Inverse of `complex_stft` (weighted overlap-add), returns (B, 1, length) in float32.

        torch.istft(center=False) rejects Hann windows (zero envelope at the padded edges), so the
        overlap-add is done here and only the (well covered) signal region is returned.
        """
        if self.win_length != self.n_fft:
            raise NotImplementedError("istft requires win_length == n_fft")
        T = spec.shape[-1]
        win = self.window.to(spec.device)
        frames = torch.fft.irfft(spec, n=self.n_fft, dim=1) * win.view(1, -1, 1)       # (B, n_fft, T)
        total = (T - 1) * self.hop_length + self.n_fft
        fold = dict(output_size=(1, total), kernel_size=(1, self.n_fft), stride=(1, self.hop_length))
        y = F.fold(frames, **fold).view(spec.shape[0], -1)
        env = F.fold(win.pow(2).view(1, -1, 1).expand(1, -1, T).contiguous(), **fold).view(1, -1)
        # With causal alignment the last few samples are only seen by the tail of one window; a relative
        # floor fades them out smoothly instead of amplifying numerical noise.
        y = y / env.clamp(min=1e-3 * float(env.max()))
        pad_left = self.n_fft // 2 - self.hop_length // 2 if self.alignment == "center" else self.n_fft - self.hop_length
        return y[:, pad_left:pad_left + length].unsqueeze(1)

    def compress(self, mag: torch.Tensor) -> torch.Tensor:
        if self.compression == "log1p":
            return torch.log1p(mag)
        if self.compression == "pow":
            return mag.clamp(min=1e-8).pow(self.compression_power)
        return mag

    def decompress(self, spec: torch.Tensor) -> torch.Tensor:
        """Inverse of `compress`, returning the (window-normalized) linear magnitude."""
        if self.compression == "log1p":
            return torch.expm1(spec).clamp(min=0)
        if self.compression == "pow":
            return spec.clamp(min=0).pow(1.0 / self.compression_power)
        return spec.clamp(min=0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Compressed magnitude spectrogram (B, F, T) with T = ceil(L / hop)."""
        return self.compress(self.magnitude(wav))
