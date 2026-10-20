"""
Training-free blend of a waveform estimate with the magnitude of a CleanSpecNet (IMPROVEMENT_PLAN.md, V1 / W):
|Y|^(1-w) |S|^w in the CleanSpecNet's STFT, with the phase of the waveform estimate. The same computation as the
D2(c) blend of scripts/diagnose_spec_branch.py, per example, so it also works on training batches (distillation
teachers, lightning_modules/cleanunet_module.py). `blend_with_spec` is the differentiable version used to train the
spectral branch for its role in the blend (loss.mode blend, IMPROVEMENT_PLAN.md, 14.11).
"""

import torch


def blend_with_spec(y: torch.Tensor, x: torch.Tensor, spec_model, w: float, spec: torch.Tensor = None) -> torch.Tensor:
    """Differentiable blend. y: waveform estimate, x: noisy input, both (B, 1, L); spec_model: a CleanUNet2 with a
    spectral branch; spec: its compressed estimate (B, F, T) for x when already computed. Returns (B, 1, L) float32."""
    S = spec_model.spectrogram
    y = y.float().view(y.shape[0], 1, -1)
    x = x.float().view(x.shape[0], 1, -1)
    L = x.shape[-1]
    scale = x.std(dim=-1, keepdim=True) + 1e-3                        # the input normalization of CleanUNet2
    T = S.num_frames(L)
    if spec is None:
        spec = spec_model.spec_branch(x / scale)["spec"]
    with torch.autocast(device_type=x.device.type, enabled=False):
        mag_s = S.decompress(spec[..., :T].float())
        Y = S.complex_stft(y / scale)
        mag_y = Y.abs() * S.scale
        mag = (mag_y.clamp(min=1e-8).log() * (1 - w) + mag_s.clamp(min=1e-8).log() * w).exp()
        return S.istft(mag / S.scale * (Y / Y.abs().clamp(min=1e-8)), L) * scale


@torch.no_grad()
def blend(y: torch.Tensor, x: torch.Tensor, spec_model, w: float) -> torch.Tensor:
    """y: waveform estimate, x: noisy input, both (B, 1, L); spec_model: a CleanUNet2 with a spectral branch.
    Returns (B, 1, L) in float32."""
    return blend_with_spec(y, x, spec_model, w)
