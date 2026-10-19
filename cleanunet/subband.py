"""
N3 of IMPROVEMENT_PLAN.md: full-band + sub-band CleanSpecNet (FullSubNet style).

The full-band model is the 1-D CleanSpecNet. A small sub-band model, shared by every frequency bin,
then refines each bin f from
    - the noisy compressed magnitudes of bins f-K .. f+K (local spectral pattern, stationarity cues),
    - the full-band estimate at bin f,
with a causal Mamba over time. Its output is a zero-initialized correction of the full-band estimate,
so at initialization the model equals its full-band part.

Weight sharing across frequency gives the frequency equivariance the 1-D model lacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cleanspecnet import CleanSpecNet, cumulative_norm
from .sequence import build_mamba_layers, resolve_backend, run_mamba_layers


class SubbandSpecNet(nn.Module):
    """Drop-in replacement for CleanSpecNet (same `run` / `forward` interface)."""

    def __init__(self, input_channels: int = 513, feature_groups: int = 1, causal: bool = True,
                 output_mode: str = "mask", predict_phase: bool = False, input_norm: str = "none",
                 fullband: dict = None, neighbors: int = 15, d_model: int = 32, n_layers: int = 2,
                 d_state: int = 16, expand: float = 2, backend: str = "auto"):
        super().__init__()
        if not causal:
            raise ValueError("SubbandSpecNet is implemented for the causal setting only")
        self.fullband = CleanSpecNet(input_channels=input_channels, feature_groups=feature_groups, causal=True,
                                     output_mode=output_mode, predict_phase=predict_phase, input_norm=input_norm,
                                     **(fullband or {}))
        self.n_freq = input_channels
        self.feature_groups = feature_groups
        self.neighbors = neighbors
        self.input_norm = input_norm
        self.causal = True
        self.output_mode = output_mode
        self.predicts_phase = predict_phase
        self.hidden_dim = self.fullband.hidden_dim
        self.backbone_type = f"subband+{self.fullband.backbone_type}"
        backend = resolve_backend(backend)

        self.sub_in = nn.Linear(2 * neighbors + 2, d_model)
        self.sub_layers, self.sub_norm = build_mamba_layers(d_model, n_layers, d_state=d_state, expand=expand,
                                                            backend=backend)
        self.sub_out = nn.Linear(d_model, 1)
        nn.init.zeros_(self.sub_out.weight)     # identity w.r.t. the full-band estimate at initialization
        nn.init.zeros_(self.sub_out.bias)

    def forward(self, x: Tensor, return_hidden: bool = False):
        """Enhanced spectrogram (B, F, T), plus the hidden features with ``return_hidden``."""
        out = self.run(x)
        return (out["spec"], out["hidden"]) if return_hidden else out["spec"]

    def run(self, x: Tensor) -> dict:
        """x: (B, G*F, T) features, magnitude first."""
        full = self.fullband.run(x)
        B, _, T = x.shape
        Fq, K = self.n_freq, self.neighbors
        mag = x[:, :Fq]
        if self.input_norm == "cumulative":
            mag = cumulative_norm(mag)
        # (B, F, 2K+1, T): bins f-K .. f+K of every bin (reflection at the edges)
        local = F.pad(mag.transpose(1, 2), (K, K), mode="reflect").transpose(1, 2)        # (B, F+2K, T)
        local = local.unfold(1, 2 * K + 1, 1)                                               # (B, F, T, 2K+1)
        sub = torch.cat([local, full["spec"].unsqueeze(-1)], dim=-1)                        # (B, F, T, 2K+2)
        h = self.sub_in(sub.reshape(B * Fq, T, 2 * K + 2))
        h = run_mamba_layers(self.sub_layers, self.sub_norm, h)
        delta = self.sub_out(h).reshape(B, Fq, T)
        return {"spec": full["spec"] + delta.to(full["spec"].dtype), "hidden": full["hidden"],
                "phase": full["phase"]}
